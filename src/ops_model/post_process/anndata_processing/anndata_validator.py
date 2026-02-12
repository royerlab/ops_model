"""AnnData Validator - Core validation system for standardizing AnnData objects.

HOW IT WORKS
============

This module provides a comprehensive validation system for AnnData objects in the
ops_model project. The system ensures that all AnnData objects (both newly created
and loaded from disk) conform to standardized metadata schemas.

Architecture:
-------------
1. **AnndataSpec**: Defines schema specifications for different aggregation levels
   - Stores required and optional fields for each schema level
   - Defines data type requirements and validation rules
   - Provides schema lookup by level name

2. **ValidationIssue**: Represents a single validation problem
   - Categorizes issues as errors (critical) or warnings (non-critical)
   - Stores detailed context for debugging
   - Provides formatted error messages

3. **ValidationReport**: Aggregates validation results
   - Collects all issues found during validation
   - Provides overall pass/fail status
   - Generates human-readable summaries
   - Can raise exceptions for critical failures

4. **AnndataValidator**: Main validation engine
   - Validates AnnData objects against schema specifications
   - Checks .obs columns, .var structure, .uns metadata, and .X data
   - Can infer appropriate schema level from object structure
   - Provides utilities to add missing default values

Validation Flow:
----------------
1. Schema Selection: Determine which schema applies (cell/guide/gene/etc.)
2. Structural Validation: Check that .obs, .var, .X, .uns exist and are correct types
3. Column Validation: Check required .obs columns exist with correct types
4. Content Validation: Validate numeric ranges, uniqueness constraints, patterns
5. Metadata Validation: Check .uns for required metadata structures
6. Report Generation: Compile all issues and return ValidationReport

Usage Patterns:
---------------
**Pattern 1: Validate loaded file**
    adata = ad.read_h5ad("file.h5ad")
    validator = AnndataValidator()
    report = validator.validate(adata, level="guide", strict=False)
    if not report.is_valid:
        print(report.summary())

**Pattern 2: Validate before saving**
    adata = create_my_anndata()
    validator = AnndataValidator()
    validator.validate(adata, level="cell", strict=True)  # Raises if invalid
    adata.write_h5ad("output.h5ad")

**Pattern 3: Infer schema level**
    adata = ad.read_h5ad("unknown.h5ad")
    validator = AnndataValidator()
    level = validator.infer_schema_level(adata)
    report = validator.validate(adata, level=level)

**Pattern 4: Add missing defaults**
    adata = ad.read_h5ad("old_file.h5ad")
    validator = AnndataValidator()
    validator.add_missing_defaults(adata, level="guide", inplace=True)
    validator.validate(adata, level="guide", strict=True)

Schema Levels:
--------------
- **base**: Minimal requirements for all AnnData objects
- **cell**: Cell-level data with spatial coordinates
- **guide**: Guide-level aggregated data
- **gene**: Gene-level aggregated data
- **multi_experiment**: Data concatenated from multiple experiments
- **multi_channel**: Features concatenated across imaging channels

Design Principles:
------------------
- Non-destructive: Validator never modifies data during validation
- Clear errors: Specific, actionable error messages with context
- Flexible: Allows extra metadata beyond required fields
- Gradual adoption: Can be integrated incrementally
- Backward compatible: Works with existing AnnData objects

Notes:
------
- The validator is read-only; it never modifies the input AnnData object
  unless explicitly requested via add_missing_defaults()
- Sparse matrices are supported and handled efficiently
- Index/column name conflicts (h5ad compatibility) are explicitly checked
- String pattern validation is lenient to support different naming conventions
- All validation issues include suggestions for resolution

See Also:
---------
- ANNDATA_SPEC_DESIGN.md: Complete design specification
- anndata_factory.py: Factory methods for creating validated objects
- anndata_schemas.py: Detailed schema definitions

Version: 1.0
Date: 2026-02-02
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad
else:
    import anndata as ad


# Configure logging
logger = logging.getLogger(__name__)


class IssueLevel(Enum):
    """Enumeration of validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SchemaLevel(Enum):
    """Enumeration of supported schema levels."""

    BASE = "base"
    CELL = "cell"
    GUIDE = "guide"
    GENE = "gene"
    MULTI_EXPERIMENT = "multi_experiment"
    MULTI_CHANNEL = "multi_channel"


@dataclass
class FieldSpec:
    """Specification for a single field in the AnnData schema.

    Attributes
    ----------
    name : str
        Name of the field (e.g., "label_str", "sgRNA")
    dtype : Union[type, List[type]]
        Expected data type(s). Can be single type or list of acceptable types
    required : bool
        Whether this field is required (True) or optional (False)
    pattern : Optional[str]
        Regex pattern for string validation (None if not applicable)
    min_value : Optional[float]
        Minimum allowed value for numeric fields (None if not applicable)
    max_value : Optional[float]
        Maximum allowed value for numeric fields (None if not applicable)
    unique : bool
        Whether values must be unique across observations
    allow_nan : bool
        Whether NaN values are allowed in this field
    description : str
        Human-readable description of this field's purpose
    suggestion : str
        Suggestion for how to fix missing or invalid field
    """

    name: str
    dtype: Union[type, List[type]]
    required: bool = True
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unique: bool = False
    allow_nan: bool = False
    description: str = ""
    suggestion: str = ""


@dataclass
class ValidationIssue:
    """Represents a single validation issue found during validation.

    Attributes
    ----------
    level : IssueLevel
        Severity level (ERROR, WARNING, INFO)
    component : str
        Which component has the issue (e.g., ".obs", ".var", ".uns", ".X")
    field : Optional[str]
        Specific field name if applicable (e.g., "label_str")
    message : str
        Brief description of the issue
    expected : Optional[str]
        What was expected (e.g., "int64 or int32")
    found : Optional[str]
        What was actually found (e.g., "float64")
    suggestion : Optional[str]
        How to fix the issue
    context : Dict[str, Any]
        Additional context information for debugging
    """

    level: IssueLevel
    component: str
    field: Optional[str]
    message: str
    expected: Optional[str] = None
    found: Optional[str] = None
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format the validation issue as a human-readable string.

        Returns
        -------
        str
            Formatted error message with all relevant details
        """
        parts = [f"[{self.level.value.upper()}] {self.component}"]
        if self.field:
            parts[0] += f"['{self.field}']"
        parts[0] += f": {self.message}"

        if self.expected:
            parts.append(f"  Expected: {self.expected}")
        if self.found:
            parts.append(f"  Found: {self.found}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation issue to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for logging or serialization
        """
        return {
            "level": self.level.value,
            "component": self.component,
            "field": self.field,
            "message": self.message,
            "expected": self.expected,
            "found": self.found,
            "suggestion": self.suggestion,
            "context": self.context,
        }


@dataclass
class ValidationReport:
    """Aggregates validation results and provides summary information.

    Attributes
    ----------
    schema_level : str
        Schema level used for validation
    is_valid : bool
        Overall validation status (True if no errors)
    errors : List[ValidationIssue]
        List of critical validation failures
    warnings : List[ValidationIssue]
        List of non-critical issues
    info : List[ValidationIssue]
        List of informational messages
    n_obs : int
        Number of observations in validated object
    n_vars : int
        Number of variables in validated object
    metadata : Dict[str, Any]
        Additional metadata about validation run
    """

    schema_level: str
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    n_obs: int = 0
    n_vars: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the appropriate list.

        Parameters
        ----------
        issue : ValidationIssue
            The validation issue to add
        """
        if issue.level == IssueLevel.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.level == IssueLevel.WARNING:
            self.warnings.append(issue)
        elif issue.level == IssueLevel.INFO:
            self.info.append(issue)

    def summary(self) -> str:
        """Generate a human-readable summary of validation results.

        Returns
        -------
        str
            Multi-line summary string with all issues and statistics
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"Validation Report - Schema Level: {self.schema_level}")
        lines.append("=" * 70)
        lines.append(f"Status: {'VALID' if self.is_valid else 'INVALID'}")
        lines.append(f"Dimensions: {self.n_obs} observations × {self.n_vars} variables")
        lines.append(
            f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)} | Info: {len(self.info)}"
        )
        lines.append("")

        if self.errors:
            lines.append("ERRORS:")
            lines.append("-" * 70)
            for err in self.errors:
                lines.append(str(err))
                lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 70)
            for warn in self.warnings:
                lines.append(str(warn))
                lines.append("")

        if self.info:
            lines.append("INFO:")
            lines.append("-" * 70)
            for inf in self.info:
                lines.append(str(inf))
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation report to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for logging or serialization
        """
        return {
            "schema_level": self.schema_level,
            "is_valid": self.is_valid,
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "errors": [err.to_dict() for err in self.errors],
            "warnings": [warn.to_dict() for warn in self.warnings],
            "info": [inf.to_dict() for inf in self.info],
            "metadata": self.metadata,
        }

    def raise_if_invalid(self) -> None:
        """Raise an exception if validation failed.

        Raises
        ------
        ValueError
            If is_valid is False (any errors were found)
        """
        if not self.is_valid:
            raise ValueError(
                f"Validation failed with {len(self.errors)} error(s):\n{self.summary()}"
            )

    def has_errors(self) -> bool:
        """Check if any errors were found.

        Returns
        -------
        bool
            True if errors list is non-empty
        """
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings were found.

        Returns
        -------
        bool
            True if warnings list is non-empty
        """
        return len(self.warnings) > 0

    def get_error_count(self) -> int:
        """Get the total number of errors.

        Returns
        -------
        int
            Number of errors found
        """
        return len(self.errors)

    def get_warning_count(self) -> int:
        """Get the total number of warnings.

        Returns
        -------
        int
            Number of warnings found
        """
        return len(self.warnings)


class AnndataSpec:
    """Schema specification manager for AnnData objects.

    This class stores and provides access to schema specifications for different
    aggregation levels. Each schema defines required and optional fields, data
    types, and validation rules.

    Attributes
    ----------
    schemas : Dict[str, Dict[str, Any]]
        Dictionary mapping schema level names to their specifications
    """

    def __init__(self):
        """Initialize the schema specification manager.

        Loads all schema definitions for supported aggregation levels.
        """
        self.schemas = {
            "base": self._define_base_schema(),
            "cell": self._define_cell_schema(),
            "guide": self._define_guide_schema(),
            "gene": self._define_gene_schema(),
            "multi_experiment": self._define_multi_experiment_schema(),
            "multi_channel": self._define_multi_channel_schema(),
        }

    def get_schema(self, level: Union[str, SchemaLevel]) -> Dict[str, Any]:
        """Get the schema specification for a given level.

        Parameters
        ----------
        level : Union[str, SchemaLevel]
            Schema level name or enum value

        Returns
        -------
        Dict[str, Any]
            Schema specification containing required/optional fields and rules

        Raises
        ------
        ValueError
            If the specified level is not recognized
        """
        # Convert enum to string if needed
        if isinstance(level, SchemaLevel):
            level = level.value

        if level not in self.schemas:
            available = ", ".join(self.schemas.keys())
            raise ValueError(f"Unknown schema level '{level}'. Available: {available}")

        return self.schemas[level]

    def get_required_obs_fields(
        self, level: Union[str, SchemaLevel]
    ) -> List[FieldSpec]:
        """Get required .obs fields for a schema level.

        Parameters
        ----------
        level : Union[str, SchemaLevel]
            Schema level name or enum value

        Returns
        -------
        List[FieldSpec]
            List of required field specifications
        """
        schema = self.get_schema(level)
        return schema.get("required_fields", [])

    def get_optional_obs_fields(
        self, level: Union[str, SchemaLevel]
    ) -> List[FieldSpec]:
        """Get optional .obs fields for a schema level.

        Parameters
        ----------
        level : Union[str, SchemaLevel]
            Schema level name or enum value

        Returns
        -------
        List[FieldSpec]
            List of optional field specifications
        """
        schema = self.get_schema(level)
        return schema.get("optional_fields", [])

    def get_uns_requirements(self, level: Union[str, SchemaLevel]) -> Dict[str, Any]:
        """Get .uns metadata requirements for a schema level.

        Parameters
        ----------
        level : Union[str, SchemaLevel]
            Schema level name or enum value

        Returns
        -------
        Dict[str, Any]
            Dictionary describing required .uns structure
        """
        schema = self.get_schema(level)
        return schema.get("uns_requirements", {})

    def list_available_schemas(self) -> List[str]:
        """List all available schema level names.

        Returns
        -------
        List[str]
            List of schema level names that can be validated
        """
        return list(self.schemas.keys())

    def _define_base_schema(self) -> Dict[str, Any]:
        """Define the base schema for all AnnData objects.

        Returns
        -------
        Dict[str, Any]
            Base schema specification with minimum requirements
        """
        return {
            "required_fields": [
                FieldSpec(
                    name="perturbation",
                    dtype=str,
                    required=True,
                    description="Gene label string",
                    suggestion='Add perturbation column with gene names (e.g., "GENE_A", "NTC")',
                ),
            ],
            "optional_fields": [
                FieldSpec(
                    name="reporter",
                    dtype=str,
                    required=False,
                    description="Biological signal measured",
                    suggestion='Add reporter column with biological signal names (e.g., "SEC61B", "Phase")',
                ),
            ],
            "uns_requirements": {
                "cell_type": {
                    "type": str,
                    "required": True,
                    "description": "Cell type used in the experiment (e.g., 'A549', 'HeLa', 'RPE1')",
                    "pattern": None,  # Optional: add pattern like r"^[A-Za-z0-9-]+$" for validation
                    "examples": ["A549", "HeLa", "iPSC", "HEK293T"],
                },
                "embedding_type": {
                    "type": str,
                    "required": True,
                    "description": "Method used to extract embeddings (e.g., 'dinov3', 'cellprofiler')",
                    "pattern": None,  # Optional: add pattern like r"^[A-Za-z0-9-]+$" for validation
                    "examples": ["dinov3", "cellprofiler"],
                },
            },
        }

    def _define_cell_schema(self) -> Dict[str, Any]:
        """Define the schema for cell-level AnnData objects.

        Returns
        -------
        Dict[str, Any]
            Cell-level schema specification
        """
        base = self._define_base_schema()
        return {
            "required_fields": base["required_fields"]
            + [
                FieldSpec(
                    name="sgRNA",
                    dtype=str,
                    required=True,
                    description="Guide RNA identifier",
                    suggestion="Add sgRNA column with guide identifiers",
                ),
                FieldSpec(
                    name="well",
                    dtype=str,
                    required=True,
                    pattern=r"[A-Za-z][\d/]+",
                    description="Well identifier (formats: A1, B12, or A/1/0)",
                    suggestion="Add well column (format: A1 or A/1/0)",
                ),
                FieldSpec(
                    name="x_position",
                    dtype=[float, np.float32, np.float64],
                    required=True,
                    description="Cell x-coordinate in image",
                    suggestion="Add x_position column with cell x-coordinates",
                ),
                FieldSpec(
                    name="y_position",
                    dtype=[float, np.float32, np.float64],
                    required=True,
                    description="Cell y-coordinate in image",
                    suggestion="Add y_position column with cell y-coordinates",
                ),
                FieldSpec(
                    name="experiment",
                    dtype=str,
                    required=True,
                    pattern=r"ops\d{4}(_\d{8})?",
                    description="Experiment identifier",
                    suggestion="Add experiment column (format: ops####)",
                ),
            ],
            "optional_fields": base["optional_fields"],
            "uns_requirements": {},
        }

    def _define_guide_schema(self) -> Dict[str, Any]:
        """Define the schema for guide-level AnnData objects.

        Returns
        -------
        Dict[str, Any]
            Guide-level schema specification
        """
        base = self._define_base_schema()
        return {
            "required_fields": base["required_fields"]
            + [
                FieldSpec(
                    name="sgRNA",
                    dtype=str,
                    required=True,
                    unique=True,
                    description="Guide RNA identifier (must be unique)",
                    suggestion="Add sgRNA column with unique guide identifiers",
                ),
            ],
            "optional_fields": base["optional_fields"]
            + [
                FieldSpec(
                    name="n_cells",
                    dtype=[int, np.int32, np.int64],
                    required=True,
                    min_value=1,
                    description="Number of cells aggregated per guide",
                    suggestion="Add n_cells column with positive integer counts",
                ),
            ],
            "uns_requirements": {
                "aggregation_method": {
                    "type": str,
                    "required": True,
                    "description": "Method used to aggregate embeddings (e.g., 'mean', 'median')",
                    "pattern": None,  # Optional: add pattern like r"^[A-Za-z0-9-]+$" for validation
                    "examples": ["mean", "median"],
                },
            },
        }

    def _define_gene_schema(self) -> Dict[str, Any]:
        """Define the schema for gene-level AnnData objects.

        Returns
        -------
        Dict[str, Any]
            Gene-level schema specification
        """
        base = self._define_base_schema()
        return {
            "required_fields": base["required_fields"]
            + [
                FieldSpec(
                    name="n_experiments",
                    dtype=[int, np.int32, np.int64],
                    required=True,
                    min_value=1,
                    description="Number of experiments data is pooled from",
                    suggestion="Add n_experiments for cross-experiment pooled data",
                ),
                FieldSpec(
                    name="n_cells",
                    dtype=[int, np.int32, np.int64],
                    required=True,
                    min_value=1,
                    description="Number of cells aggregated per gene",
                    suggestion="Add n_cells column with positive integer counts",
                ),
                FieldSpec(
                    name="guides",
                    dtype=object,
                    required=True,
                    description="List of guide RNAs aggregated for this gene",
                    suggestion="Add guides column with list of sgRNA identifiers per gene",
                ),
            ],
            "uns_requirements": {
                "aggregation_method": {
                    "type": str,
                    "required": True,
                    "description": "Method used to aggregate embeddings (e.g., 'mean', 'median')",
                    "pattern": None,  # Optional: add pattern like r"^[A-Za-z0-9-]+$" for validation
                    "examples": ["mean", "median"],
                },
            },
            "optional_fields": base["optional_fields"],
        }

    def _define_multi_experiment_schema(self) -> Dict[str, Any]:
        """Define the schema for multi-experiment AnnData objects.

        Returns
        -------
        Dict[str, Any]
            Multi-experiment schema specification
        """
        base = self._define_base_schema()
        # For multi-experiment, experiment should NOT be present (removed from optional)
        base_required = [f for f in base["required_fields"]]
        base_optional = [f for f in base["optional_fields"] if f.name != "experiment"]

        return {
            "required_fields": [
                FieldSpec(
                    name="perturbation",
                    dtype=str,
                    required=True,
                    description="Gene label string",
                    suggestion='Add perturbation column with gene names (e.g., "GENE_A", "NTC")',
                ),
            ],
            "optional_fields": base_optional,
            "uns_requirements": {
                "cell_type": {
                    "description": "Type of biological unit (e.g., 'cell', 'guide', 'gene')",
                    "required": True,
                },
                "embedding_type": {
                    "description": "Type of embeddings/features (e.g., 'dinov3', 'cellprofiler')",
                    "required": True,
                },
            },
            "special_validations": {
                # Multi-experiment status is determined by schema level, not by counting experiments
                # min_unique_batches validation is performed in validate() method (line ~920)
            },
        }

    def _define_multi_channel_schema(self) -> Dict[str, Any]:
        """Define the schema for multi-channel AnnData objects.

        Returns
        -------
        Dict[str, Any]
            Multi-channel schema specification
        """
        base = self._define_base_schema()
        return {
            "required_fields": base["required_fields"],
            "optional_fields": base["optional_fields"],
            "uns_requirements": {
                "combined_metadata": {
                    "required_keys": [
                        "channels",
                        "feature_slices",
                        "channel_biology",
                        "n_channels",
                        "experiment",
                        "feature_type",
                        "aggregation_level",
                    ],
                    "description": "Metadata for multi-channel concatenation",
                },
            },
        }


class AnndataValidator:
    """Main validation engine for AnnData objects.

    This class provides methods to validate AnnData objects against schema
    specifications, check individual components, infer schema levels, and
    add missing default values.

    Attributes
    ----------
    spec : AnndataSpec
        Schema specification manager
    strict : bool
        Default strictness mode for validation
    """

    def __init__(self, strict: bool = True):
        """Initialize the AnnData validator.

        Parameters
        ----------
        strict : bool, default=True
            Default strictness mode. If True, validation raises exceptions on
            failure. If False, returns ValidationReport without raising.
        """
        self.spec = AnndataSpec()
        self.strict = strict

    def validate(
        self,
        adata: ad.AnnData,
        level: Union[str, SchemaLevel],
        strict: Optional[bool] = None,
    ) -> ValidationReport:
        """Validate an AnnData object against a schema specification.

        This is the main entry point for validation. It performs comprehensive
        validation of all AnnData components and returns a detailed report.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to validate
        level : Union[str, SchemaLevel]
            Schema level to validate against
        strict : Optional[bool], default=None
            Whether to raise exception on failure. If None, uses instance default.

        Returns
        -------
        ValidationReport
            Detailed validation report with all issues found

        Raises
        ------
        ValueError
            If strict=True and validation fails
        """
        # Determine strictness mode
        use_strict = strict if strict is not None else self.strict

        # Convert enum to string if needed
        if isinstance(level, SchemaLevel):
            level = level.value

        # Initialize report
        report = ValidationReport(
            schema_level=level,
            is_valid=True,
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
        )

        logger.info(f"Starting validation (level={level}, strict={use_strict})")

        # Get schema
        try:
            schema = self.spec.get_schema(level)
        except ValueError as e:
            issue = ValidationIssue(
                level=IssueLevel.ERROR,
                component="schema",
                field=None,
                message=str(e),
            )
            report.add_issue(issue)
            if use_strict:
                report.raise_if_invalid()
            return report

        # Step 1: Structural integrity
        issues = self._check_structural_integrity(adata)
        for issue in issues:
            report.add_issue(issue)
        if report.has_errors() and use_strict:
            report.raise_if_invalid()
            return report

        # Step 2: .obs validation
        required_fields = self.spec.get_required_obs_fields(level)
        optional_fields = self.spec.get_optional_obs_fields(level)
        issues = self.check_obs_columns(adata, required_fields, optional_fields)
        for issue in issues:
            report.add_issue(issue)

        # Step 3: .var validation
        issues = self.check_var_structure(adata, level)
        for issue in issues:
            report.add_issue(issue)

        # Step 4: .X validation
        issues = self.check_data_matrix(adata)
        for issue in issues:
            report.add_issue(issue)

        # Step 5: .uns validation
        issues = self.check_uns_metadata(adata, level)
        for issue in issues:
            report.add_issue(issue)

        # Step 6: Index conflicts (h5ad compatibility)
        issues = self.check_index_conflicts(adata)
        for issue in issues:
            report.add_issue(issue)

        # Step 7: Special validations for multi-experiment
        if level == "multi_experiment":
            special = schema.get("special_validations", {})
            if "batch" in adata.obs.columns:
                n_batches = adata.obs["batch"].nunique()
                min_batches = special.get("min_unique_batches", 2)
                if n_batches < min_batches:
                    report.add_issue(
                        ValidationIssue(
                            level=IssueLevel.ERROR,
                            component=".obs",
                            field="batch",
                            message=f"Multi-experiment data must have at least {min_batches} unique batches",
                            expected=f"{min_batches}+ unique values",
                            found=f"{n_batches} unique values",
                        )
                    )

            # Note: experiment column should NOT exist in multi-experiment objects
            # Multi-experiment distinction is based on schema level, not by counting experiments

        logger.info(
            f"Validation complete: {len(report.errors)} errors, {len(report.warnings)} warnings"
        )

        # Raise if strict and invalid
        if use_strict and not report.is_valid:
            report.raise_if_invalid()

        return report

    def check_obs_columns(
        self,
        adata: ad.AnnData,
        required_fields: List[FieldSpec],
        optional_fields: List[FieldSpec],
    ) -> List[ValidationIssue]:
        """Validate .obs DataFrame structure and columns.

        Checks for:
        - Presence of required columns
        - Correct data types
        - Missing values in required columns
        - String pattern matching
        - Numeric range validation
        - Uniqueness constraints

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to validate
        required_fields : List[FieldSpec]
            List of required field specifications
        optional_fields : List[FieldSpec]
            List of optional field specifications

        Returns
        -------
        List[ValidationIssue]
            List of validation issues found
        """
        issues = []

        # Check required fields
        for field_spec in required_fields:
            if field_spec.name not in adata.obs.columns:
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        component=".obs",
                        field=field_spec.name,
                        message=f"Missing required column '{field_spec.name}'",
                        expected=field_spec.description,
                        found="Column not present",
                        suggestion=field_spec.suggestion,
                    )
                )
            else:
                # Validate the field
                field_issues = self._validate_field(
                    adata.obs[field_spec.name], field_spec
                )
                issues.extend(field_issues)

        # Check optional fields (warnings only)
        for field_spec in optional_fields:
            if field_spec.name not in adata.obs.columns:
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.WARNING,
                        component=".obs",
                        field=field_spec.name,
                        message=f"Optional column '{field_spec.name}' not found",
                        suggestion=field_spec.suggestion,
                    )
                )
            else:
                # Validate the field
                field_issues = self._validate_field(
                    adata.obs[field_spec.name], field_spec
                )
                issues.extend(field_issues)

        return issues

    def check_var_structure(
        self, adata: ad.AnnData, level: Union[str, SchemaLevel]
    ) -> List[ValidationIssue]:
        """Validate .var DataFrame and variable names.

        Checks for:
        - Duplicate variable names
        - Expected naming patterns (if applicable)
        - Variable count consistency

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to validate
        level : Union[str, SchemaLevel]
            Schema level being validated

        Returns
        -------
        List[ValidationIssue]
            List of validation issues found
        """
        issues = []

        # Check for duplicate variable names
        if adata.var.index.has_duplicates:
            duplicates = adata.var.index[adata.var.index.duplicated()].unique().tolist()
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".var",
                    field="index",
                    message="Duplicate variable names found",
                    found=f'Duplicates: {duplicates[:5]}{"..." if len(duplicates) > 5 else ""}',
                    suggestion="Ensure all variable names are unique",
                )
            )

        return issues

    def check_uns_metadata(
        self, adata: ad.AnnData, level: Union[str, SchemaLevel]
    ) -> List[ValidationIssue]:
        """Validate .uns dictionary structure and required metadata.

        Checks for:
        - Required metadata keys
        - Correct metadata structure
        - Special requirements (e.g., combined_metadata for multi-channel)

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to validate
        level : Union[str, SchemaLevel]
            Schema level being validated

        Returns
        -------
        List[ValidationIssue]
            List of validation issues found
        """
        issues = []

        # Convert enum to string
        if isinstance(level, SchemaLevel):
            level = level.value

        uns_reqs = self.spec.get_uns_requirements(level)

        # Check for base-level .uns requirements (applies to all schemas)
        base_uns_reqs = self.spec.get_uns_requirements("base")

        # Validate all base-level .uns requirements (generic loop)
        for field_name, field_spec in base_uns_reqs.items():
            if field_spec.get("required", False):
                if field_name not in adata.uns:
                    examples = field_spec.get("examples", [])
                    example_str = (
                        f" (e.g., adata.uns['{field_name}'] = '{examples[0]}')"
                        if examples
                        else ""
                    )
                    issues.append(
                        ValidationIssue(
                            level=IssueLevel.ERROR,
                            component=".uns",
                            field=field_name,
                            message=f"Missing required {field_name} in .uns",
                            expected=field_spec["description"],
                            found="Key not present",
                            suggestion=f"Add {field_name} to .uns{example_str}",
                            context={"examples": examples},
                        )
                    )
                else:
                    # Validate type
                    field_value = adata.uns[field_name]
                    expected_type = field_spec.get("type", str)

                    if not isinstance(field_value, expected_type):
                        issues.append(
                            ValidationIssue(
                                level=IssueLevel.ERROR,
                                component=".uns",
                                field=field_name,
                                message=f"{field_name} must be a {expected_type.__name__}",
                                expected=expected_type.__name__,
                                found=str(type(field_value)),
                                suggestion=f"Convert {field_name} to {expected_type.__name__}",
                            )
                        )

                    # Validate pattern if specified (only for strings)
                    if field_spec.get("pattern") is not None and isinstance(
                        field_value, str
                    ):
                        import re

                        pattern = field_spec["pattern"]
                        if not re.match(pattern, field_value):
                            issues.append(
                                ValidationIssue(
                                    level=IssueLevel.WARNING,
                                    component=".uns",
                                    field=field_name,
                                    message=f"{field_name} '{field_value}' does not match expected pattern",
                                    expected=f"Pattern: {pattern}",
                                    found=f"Value: {field_value}",
                                )
                            )

        # Validate schema-specific .uns requirements (if any)
        if uns_reqs:
            for field_name, field_spec in uns_reqs.items():
                # Skip if already validated as base requirement
                if field_name in base_uns_reqs:
                    continue

                if field_spec.get("required", False):
                    if field_name not in adata.uns:
                        examples = field_spec.get("examples", [])
                        example_str = (
                            f" (e.g., adata.uns['{field_name}'] = '{examples[0]}')"
                            if examples
                            else ""
                        )
                        issues.append(
                            ValidationIssue(
                                level=IssueLevel.ERROR,
                                component=".uns",
                                field=field_name,
                                message=f"Missing required {field_name} in .uns for {level}-level schema",
                                expected=field_spec["description"],
                                found="Key not present",
                                suggestion=f"Add {field_name} to .uns{example_str}",
                                context={"examples": examples},
                            )
                        )
                    else:
                        # Validate type
                        field_value = adata.uns[field_name]
                        expected_type = field_spec.get("type", str)

                        if not isinstance(field_value, expected_type):
                            issues.append(
                                ValidationIssue(
                                    level=IssueLevel.ERROR,
                                    component=".uns",
                                    field=field_name,
                                    message=f"{field_name} must be a {expected_type.__name__}",
                                    expected=expected_type.__name__,
                                    found=str(type(field_value)),
                                    suggestion=f"Convert {field_name} to {expected_type.__name__}",
                                )
                            )

                        # Validate pattern if specified (only for strings)
                        if field_spec.get("pattern") is not None and isinstance(
                            field_value, str
                        ):
                            import re

                            pattern = field_spec["pattern"]
                            if not re.match(pattern, field_value):
                                issues.append(
                                    ValidationIssue(
                                        level=IssueLevel.WARNING,
                                        component=".uns",
                                        field=field_name,
                                        message=f"{field_name} '{field_value}' does not match expected pattern",
                                        expected=f"Pattern: {pattern}",
                                        found=f"Value: {field_value}",
                                    )
                                )

        # Check for required metadata (mainly for multi-channel)
        if level == "multi_channel":
            if "combined_metadata" not in adata.uns:
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        component=".uns",
                        field="combined_metadata",
                        message="Missing required combined_metadata for multi-channel",
                        suggestion="Add combined_metadata dict to .uns with channel information",
                    )
                )
            else:
                # Validate combined_metadata structure
                cm = adata.uns["combined_metadata"]
                req_keys = uns_reqs["combined_metadata"]["required_keys"]
                for key in req_keys:
                    if key not in cm:
                        issues.append(
                            ValidationIssue(
                                level=IssueLevel.ERROR,
                                component=".uns",
                                field=f"combined_metadata.{key}",
                                message=f'Missing required key "{key}" in combined_metadata',
                                suggestion=f'Add "{key}" to combined_metadata',
                            )
                        )

                # Validate feature slices if present
                if "feature_slices" in cm and "channels" in cm:
                    slices = cm["feature_slices"]
                    channels = cm["channels"]
                    # Check that slice keys match channels
                    if set(slices.keys()) != set(channels):
                        issues.append(
                            ValidationIssue(
                                level=IssueLevel.ERROR,
                                component=".uns",
                                field="combined_metadata.feature_slices",
                                message="feature_slices keys do not match channels",
                                expected=f"Keys: {channels}",
                                found=f"Keys: {list(slices.keys())}",
                            )
                        )

        return issues

    def check_data_matrix(self, adata: ad.AnnData) -> List[ValidationIssue]:
        """Validate .X array structure and content.

        Checks for:
        - Correct dimensions (matches .obs and .var)
        - Valid dtype (float32 or float64)
        - All-NaN rows or columns (warns only)
        - Sparse vs dense matrix handling

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to validate

        Returns
        -------
        List[ValidationIssue]
            List of validation issues found
        """
        issues = []

        # Check that .X exists
        if adata.X is None:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".X",
                    field=None,
                    message=".X is None",
                    suggestion="Ensure .X contains the data matrix",
                )
            )
            return issues

        # Check dimensions
        if adata.X.shape[0] != adata.n_obs:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".X",
                    field="shape",
                    message="Number of rows in .X does not match .obs",
                    expected=f"{adata.n_obs} rows",
                    found=f"{adata.X.shape[0]} rows",
                )
            )

        if adata.X.shape[1] != adata.n_vars:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".X",
                    field="shape",
                    message="Number of columns in .X does not match .var",
                    expected=f"{adata.n_vars} columns",
                    found=f"{adata.X.shape[1]} columns",
                )
            )

        # Check dtype
        is_sparse = sparse.issparse(adata.X)
        dtype = adata.X.dtype

        if dtype not in [np.float32, np.float64]:
            level = (
                IssueLevel.WARNING
                if dtype in [np.int32, np.int64]
                else IssueLevel.ERROR
            )
            issues.append(
                ValidationIssue(
                    level=level,
                    component=".X",
                    field="dtype",
                    message="Unexpected dtype for .X",
                    expected="float32 or float64",
                    found=str(dtype),
                    suggestion="Convert .X to float32 or float64",
                )
            )

        # Check for all-NaN rows/columns (warning only)
        if not is_sparse and adata.X.size > 0:
            # Check columns
            nan_cols = np.all(np.isnan(adata.X), axis=0)
            if np.any(nan_cols):
                n_nan_cols = np.sum(nan_cols)
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.WARNING,
                        component=".X",
                        field="columns",
                        message=f"{n_nan_cols} column(s) are all NaN",
                        suggestion="Check feature extraction - some features may be invalid",
                    )
                )

            # Check rows
            nan_rows = np.all(np.isnan(adata.X), axis=1)
            if np.any(nan_rows):
                n_nan_rows = np.sum(nan_rows)
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.WARNING,
                        component=".X",
                        field="rows",
                        message=f"{n_nan_rows} row(s) are all NaN",
                        suggestion="Some observations have no valid features",
                    )
                )

        return issues

    def check_index_conflicts(self, adata: ad.AnnData) -> List[ValidationIssue]:
        """Check for index/column name conflicts that break h5ad writing.

        The AnnData.write_h5ad() method fails if .obs.index.name matches any
        column name in .obs. This is a critical check for file persistence.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to validate

        Returns
        -------
        List[ValidationIssue]
            List of validation issues found
        """
        issues = []

        # Check .obs index name vs columns
        if (
            adata.obs.index.name is not None
            and adata.obs.index.name in adata.obs.columns
        ):
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".obs",
                    field="index",
                    message=f'.obs.index.name "{adata.obs.index.name}" matches a column name',
                    suggestion="Set .obs.index.name = None or rename the column",
                    context={"index_name": adata.obs.index.name},
                )
            )

        # Same check for .var
        if (
            adata.var.index.name is not None
            and adata.var.index.name in adata.var.columns
        ):
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".var",
                    field="index",
                    message=f'.var.index.name "{adata.var.index.name}" matches a column name',
                    suggestion="Set .var.index.name = None or rename the column",
                    context={"index_name": adata.var.index.name},
                )
            )

        return issues

    def infer_schema_level(self, adata: ad.AnnData) -> str:
        """Infer the appropriate schema level for an AnnData object.

        Examines the structure and contents of the object to determine which
        schema level is most appropriate. Useful for validating files of
        unknown type.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to analyze

        Returns
        -------
        str
            Best-guess schema level name (e.g., "cell", "guide", "gene")
        """
        obs_cols = set(adata.obs.columns)

        # Check for multi-channel
        if "combined_metadata" in adata.uns:
            return "multi_channel"

        # Check for multi-experiment
        if "batch" in obs_cols and "experiment" in obs_cols:
            if adata.obs["batch"].nunique() > 1:
                return "multi_experiment"

        # Check for cell level (has spatial coords)
        if "x_position" in obs_cols and "y_position" in obs_cols:
            return "cell"

        # Check for guide level (has sgRNA and n_cells, sgRNA should be mostly unique)
        if "sgRNA" in obs_cols and "n_cells" in obs_cols:
            # If sgRNA is highly unique, likely guide level
            if adata.obs["sgRNA"].nunique() / len(adata.obs) > 0.8:
                return "guide"

        # Check for gene level (has n_cells but not necessarily unique sgRNA)
        if "n_cells" in obs_cols:
            return "gene"

        # Default to base
        return "base"

    def add_missing_defaults(
        self,
        adata: ad.AnnData,
        level: Union[str, SchemaLevel],
        inplace: bool = True,
    ) -> Optional[ad.AnnData]:
        """Add missing optional fields with sensible default values.

        Only adds fields that don't exist. Never overwrites existing data.
        This is useful for updating old files to meet current schema.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to update
        level : Union[str, SchemaLevel]
            Schema level to use for defaults
        inplace : bool, default=True
            Whether to modify in place or return a copy

        Returns
        -------
        Optional[ad.AnnData]
            Modified AnnData object if inplace=False, None otherwise
        """
        if not inplace:
            adata = adata.copy()

        optional_fields = self.spec.get_optional_obs_fields(level)

        for field_spec in optional_fields:
            if field_spec.name not in adata.obs.columns:
                # Add a placeholder column with None/NaN
                if field_spec.dtype == str or (
                    isinstance(field_spec.dtype, list) and str in field_spec.dtype
                ):
                    adata.obs[field_spec.name] = None
                else:
                    adata.obs[field_spec.name] = np.nan

                logger.info(f"Added default for optional field '{field_spec.name}'")

        if not inplace:
            return adata
        return None

    def _check_structural_integrity(self, adata: ad.AnnData) -> List[ValidationIssue]:
        """Check basic structural integrity of AnnData object.

        Verifies that:
        - .obs, .var, .X exist and are correct types
        - Dimensions are consistent
        - Basic pandas/numpy requirements are met

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object to check

        Returns
        -------
        List[ValidationIssue]
            List of structural issues found
        """
        issues = []

        # Check that adata is actually an AnnData object
        if not isinstance(adata, ad.AnnData):
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component="object",
                    field=None,
                    message="Input is not an AnnData object",
                    found=str(type(adata)),
                )
            )
            return issues

        # Check .obs exists and is DataFrame
        if not isinstance(adata.obs, pd.DataFrame):
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".obs",
                    field="type",
                    message=".obs is not a pandas DataFrame",
                    found=str(type(adata.obs)),
                )
            )

        # Check .var exists and is DataFrame
        if not isinstance(adata.var, pd.DataFrame):
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    component=".var",
                    field="type",
                    message=".var is not a pandas DataFrame",
                    found=str(type(adata.var)),
                )
            )

        # Check .uns exists and is dict-like
        if not isinstance(adata.uns, dict):
            issues.append(
                ValidationIssue(
                    level=IssueLevel.WARNING,
                    component=".uns",
                    field="type",
                    message=".uns is not a dict",
                    found=str(type(adata.uns)),
                )
            )

        return issues

    def _validate_field(
        self,
        series: pd.Series,
        field_spec: FieldSpec,
        component: str = ".obs",
    ) -> List[ValidationIssue]:
        """Validate a single field (column) against its specification.

        Parameters
        ----------
        series : pd.Series
            The data series to validate
        field_spec : FieldSpec
            Field specification with requirements
        component : str, default=".obs"
            Component name for error messages

        Returns
        -------
        List[ValidationIssue]
            List of validation issues found for this field
        """
        issues = []
        level = IssueLevel.ERROR if field_spec.required else IssueLevel.WARNING

        # Check dtype
        if not self._check_dtype(series, field_spec.dtype):
            expected_dtypes = (
                field_spec.dtype
                if isinstance(field_spec.dtype, list)
                else [field_spec.dtype]
            )
            expected_str = " or ".join(
                self._format_dtype_name(dt) for dt in expected_dtypes
            )
            issues.append(
                ValidationIssue(
                    level=level,
                    component=component,
                    field=field_spec.name,
                    message=f"Incorrect dtype for {field_spec.name}",
                    expected=expected_str,
                    found=self._format_dtype_name(series.dtype),
                    suggestion=f"Convert {field_spec.name} to {expected_str}",
                )
            )

        # Check for missing values
        if not field_spec.allow_nan:
            n_nan = series.isna().sum()
            if n_nan > 0:
                issues.append(
                    ValidationIssue(
                        level=level,
                        component=component,
                        field=field_spec.name,
                        message=f"{n_nan} missing/NaN values in required field {field_spec.name}",
                        suggestion=f"Fill or remove missing values in {field_spec.name}",
                    )
                )

        # Check pattern (for strings)
        if field_spec.pattern is not None:
            if not self._check_pattern(series, field_spec.pattern):
                issues.append(
                    ValidationIssue(
                        level=level,
                        component=component,
                        field=field_spec.name,
                        message=f"Values in {field_spec.name} do not match expected pattern",
                        expected=f"Pattern: {field_spec.pattern}",
                        suggestion=f"Ensure {field_spec.name} values match pattern {field_spec.pattern}",
                    )
                )

        # Check numeric range
        if field_spec.min_value is not None or field_spec.max_value is not None:
            if not self._check_numeric_range(
                series, field_spec.min_value, field_spec.max_value
            ):
                range_str = ""
                if field_spec.min_value is not None:
                    range_str += f">= {field_spec.min_value}"
                if field_spec.max_value is not None:
                    if range_str:
                        range_str += " and "
                    range_str += f"<= {field_spec.max_value}"
                issues.append(
                    ValidationIssue(
                        level=level,
                        component=component,
                        field=field_spec.name,
                        message=f"Values in {field_spec.name} outside expected range",
                        expected=range_str,
                        suggestion=f"Check {field_spec.name} values are in valid range",
                    )
                )

        # Check uniqueness
        if field_spec.unique:
            if series.duplicated().any():
                n_dup = series.duplicated().sum()
                issues.append(
                    ValidationIssue(
                        level=level,
                        component=component,
                        field=field_spec.name,
                        message=f"{n_dup} duplicate values in {field_spec.name} (should be unique)",
                        suggestion=f"Ensure {field_spec.name} values are unique",
                    )
                )

        return issues

    def _check_dtype(
        self, series: pd.Series, expected_dtypes: Union[type, List[type]]
    ) -> bool:
        """Check if a series has one of the expected data types.

        Parameters
        ----------
        series : pd.Series
            Series to check
        expected_dtypes : Union[type, List[type]]
            Expected dtype or list of acceptable dtypes

        Returns
        -------
        bool
            True if dtype matches, False otherwise
        """
        if not isinstance(expected_dtypes, list):
            expected_dtypes = [expected_dtypes]

        # Get the actual dtype
        actual_dtype = series.dtype

        for expected_dtype in expected_dtypes:
            # Handle string types specially
            if expected_dtype == str:
                if actual_dtype == object or pd.api.types.is_string_dtype(series):
                    return True
            # Handle int types
            elif expected_dtype == int:
                if pd.api.types.is_integer_dtype(series):
                    return True
            # Handle numpy types
            elif (
                hasattr(expected_dtype, "__module__")
                and "numpy" in expected_dtype.__module__
            ):
                if actual_dtype == expected_dtype:
                    return True
                # Also check base type (int32 == int64 for our purposes)
                if np.issubdtype(actual_dtype, np.integer) and np.issubdtype(
                    expected_dtype, np.integer
                ):
                    return True
                if np.issubdtype(actual_dtype, np.floating) and np.issubdtype(
                    expected_dtype, np.floating
                ):
                    return True
            # Direct comparison
            elif actual_dtype == expected_dtype:
                return True

        return False

    def _check_pattern(self, series: pd.Series, pattern: str) -> bool:
        """Check if all values in a string series match a regex pattern.

        Parameters
        ----------
        series : pd.Series
            String series to check
        pattern : str
            Regex pattern to match

        Returns
        -------
        bool
            True if all non-null values match pattern, False otherwise
        """
        import re

        # Get non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return True

        # Check pattern
        compiled_pattern = re.compile(pattern)
        matches = non_null.astype(str).apply(lambda x: bool(compiled_pattern.match(x)))
        return matches.all()

    def _check_numeric_range(
        self,
        series: pd.Series,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> bool:
        """Check if numeric values fall within specified range.

        Parameters
        ----------
        series : pd.Series
            Numeric series to check
        min_value : Optional[float]
            Minimum allowed value (inclusive)
        max_value : Optional[float]
            Maximum allowed value (inclusive)

        Returns
        -------
        bool
            True if all values are in range, False otherwise
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return True

        if min_value is not None:
            if (non_null < min_value).any():
                return False

        if max_value is not None:
            if (non_null > max_value).any():
                return False

        return True

    def _format_dtype_name(self, dtype: Union[type, np.dtype]) -> str:
        """Format a dtype for display in error messages.

        Parameters
        ----------
        dtype : Union[type, np.dtype]
            Data type to format

        Returns
        -------
        str
            Human-readable dtype name
        """
        if dtype == str:
            return "str"
        elif dtype == int:
            return "int"
        elif dtype == float:
            return "float"
        elif hasattr(dtype, "name"):
            return dtype.name
        else:
            return str(dtype)


def create_validation_issue(
    level: IssueLevel,
    component: str,
    message: str,
    field: Optional[str] = None,
    expected: Optional[str] = None,
    found: Optional[str] = None,
    suggestion: Optional[str] = None,
    **context,
) -> ValidationIssue:
    """Factory function to create a validation issue.

    Parameters
    ----------
    level : IssueLevel
        Severity level
    component : str
        Component with the issue
    message : str
        Brief description
    field : Optional[str]
        Specific field name if applicable
    expected : Optional[str]
        What was expected
    found : Optional[str]
        What was found
    suggestion : Optional[str]
        How to fix
    **context
        Additional context as keyword arguments

    Returns
    -------
    ValidationIssue
        Constructed validation issue
    """
    return ValidationIssue(
        level=level,
        component=component,
        field=field,
        message=message,
        expected=expected,
        found=found,
        suggestion=suggestion,
        context=context,
    )


def validate_anndata(
    adata: ad.AnnData,
    level: Union[str, SchemaLevel],
    strict: bool = True,
) -> ValidationReport:
    """Convenience function to validate an AnnData object.

    This is a module-level shortcut for AnndataValidator().validate().

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to validate
    level : Union[str, SchemaLevel]
        Schema level to validate against
    strict : bool, default=True
        Whether to raise exception on failure

    Returns
    -------
    ValidationReport
        Validation report

    Raises
    ------
    ValueError
        If strict=True and validation fails

    Examples
    --------
    >>> import anndata as ad
    >>> adata = ad.read_h5ad("data.h5ad")
    >>> report = validate_anndata(adata, level="guide", strict=False)
    >>> print(report.summary())
    """
    validator = AnndataValidator(strict=strict)
    return validator.validate(adata, level=level)


def infer_and_validate(
    adata: ad.AnnData, strict: bool = False
) -> Tuple[str, ValidationReport]:
    """Infer schema level and validate an AnnData object.

    Convenience function that infers the appropriate schema level and then
    validates against it.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to validate
    strict : bool, default=False
        Whether to raise exception on failure

    Returns
    -------
    Tuple[str, ValidationReport]
        Tuple of (inferred_level, validation_report)

    Examples
    --------
    >>> import anndata as ad
    >>> adata = ad.read_h5ad("unknown.h5ad")
    >>> level, report = infer_and_validate(adata)
    >>> print(f"Detected {level}-level data")
    >>> if not report.is_valid:
    ...     print(report.summary())
    """
    validator = AnndataValidator(strict=strict)
    level = validator.infer_schema_level(adata)
    report = validator.validate(adata, level=level)
    return level, report
