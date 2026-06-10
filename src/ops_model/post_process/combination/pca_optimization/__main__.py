"""Enable ``python -m ops_model.post_process.combination.pca_optimization``.

The package's ``__init__.py`` defines ``main()`` and historically lived
in a flat ``pca_optimization.py`` file with ``if __name__ == "__main__":
main()`` at the bottom — that no longer fires once the module became a
package, so this thin shim re-creates the entry point.
"""

from ops_model.post_process.combination.pca_optimization import main


if __name__ == "__main__":
    main()
