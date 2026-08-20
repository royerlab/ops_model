"""Generate a standalone scale-bar SVG for attention-atlas cell tiles.

The SVG's coordinate system matches one cell tile produced by
``attention_atlas.py`` (default ``--crop-size 200`` at 20x OPS imaging,
0.325 µm/px → 65 µm per tile). Drop the SVG onto a figure in Illustrator
and scale it to cover exactly one cell tile; the bar's physical length is
then automatically calibrated.

Bar and label are pure vector primitives, so font / color / position stay
fully editable downstream.

Usage:
    python make_scale_bar.py                          # 10 µm, white, default tile
    python make_scale_bar.py --length-um 20           # 20 µm bar
    python make_scale_bar.py --color black            # for light backgrounds
    python make_scale_bar.py --output ~/Desktop/scale_5um.svg --length-um 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/analysis/attention_accuracy"
)


def build_svg(length_um: float, pixel_size_um: float, tile_size_px: int,
              color: str, font_size: float, margin_px: float,
              bar_height_px: float, with_tile_preview: bool) -> str:
    """Return an SVG document as a string.

    Bar sits in the bottom-right corner of the tile coordinate system,
    with the label centered above it. When ``with_tile_preview`` is True,
    a faint outline of the full tile is drawn so the bar's location is
    visible against a blank Illustrator artboard; delete that rect once
    the SVG is placed on the figure.
    """
    bar_px = length_um / pixel_size_um
    bar_x = tile_size_px - margin_px - bar_px
    bar_y = tile_size_px - margin_px - bar_height_px
    label_x = bar_x + bar_px / 2
    label_y = bar_y - 2  # baseline 2 px above the top of the bar

    # Faint tile-bounds rectangle. Stroke-only so it doesn't print over
    # the cell image when placed; mainly there to make the SVG visible
    # standalone (otherwise white-on-white = invisible).
    preview_rect = (
        f'  <rect id="tile-preview" x="0" y="0" '
        f'width="{tile_size_px}" height="{tile_size_px}" '
        f'fill="none" stroke="#888" stroke-width="0.5" '
        f'stroke-dasharray="2,2"/>\n'
        if with_tile_preview else ""
    )

    # Paint-order trick puts a thin contrasting outline behind the
    # white fill, so the bar/label read on both dark and light tiles.
    contrast = "black" if color.lower() in ("white", "#fff", "#ffffff") else "white"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  Scale bar for attention atlas cell tiles.
  Tile coord system: {tile_size_px} px = {tile_size_px * pixel_size_um:g} um wide
  ({pixel_size_um} um/px). Bar = {length_um} um = {bar_px:.4f} px.

  Illustrator workflow:
    1. File > Place this SVG onto your figure (or drag it in).
    2. Open Window > Transform. Lock the aspect ratio (chain icon).
    3. Set W = exact width of one cell tile in your figure (in mm/pt).
       Height auto-fits and the bar's physical length is now calibrated.
    4. Delete the dashed 'tile-preview' rect once positioned.
-->
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {tile_size_px} {tile_size_px}"
     width="{tile_size_px}" height="{tile_size_px}">
{preview_rect}  <g id="scale-bar">
    <rect x="{bar_x:.4f}" y="{bar_y:.4f}"
          width="{bar_px:.4f}" height="{bar_height_px}"
          fill="{color}" stroke="{contrast}" stroke-width="0.3"/>
    <text x="{label_x:.4f}" y="{label_y:.4f}"
          font-family="Arial, Helvetica, sans-serif"
          font-size="{font_size}" fill="{color}"
          stroke="{contrast}" stroke-width="0.25"
          paint-order="stroke fill"
          text-anchor="middle">{length_um:g} µm</text>
  </g>
</svg>
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--length-um", type=float, default=10.0,
                    help="Scale bar length in micrometers (default 10).")
    p.add_argument("--pixel-size-um", type=float, default=0.325,
                    help="Pixel size in um (default 0.325 for 20x OPS).")
    p.add_argument("--tile-size-px", type=int, default=200,
                    help="Cell tile size in pixels (default 200, "
                         "matching attention_atlas.py --crop-size).")
    p.add_argument("--color", default="white",
                    help="Bar/label color (default 'white' for dark tiles; "
                         "use 'black' for light backgrounds).")
    p.add_argument("--font-size", type=float, default=9.0,
                    help="Label font size in tile-coord units (default 9).")
    p.add_argument("--margin-px", type=float, default=10.0,
                    help="Margin from bottom/right tile edge (default 10).")
    p.add_argument("--bar-height-px", type=float, default=2.5,
                    help="Bar thickness in tile-coord units (default 2.5).")
    p.add_argument("--no-tile-preview", action="store_true",
                    help="Omit the dashed tile-bounds rectangle. "
                         "(By default the rect is included so the bar is "
                         "visible when the SVG is opened standalone; "
                         "delete it after placing in Illustrator.)")
    p.add_argument("--output", type=Path, default=None,
                    help="Output .svg path. Defaults to "
                         f"{DEFAULT_OUTPUT_DIR}/scale_bar_<N>um.svg.")
    args = p.parse_args()

    out = args.output or (
        DEFAULT_OUTPUT_DIR / f"scale_bar_{args.length_um:g}um.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    svg = build_svg(
        length_um=args.length_um,
        pixel_size_um=args.pixel_size_um,
        tile_size_px=args.tile_size_px,
        color=args.color,
        font_size=args.font_size,
        margin_px=args.margin_px,
        bar_height_px=args.bar_height_px,
        with_tile_preview=not args.no_tile_preview,
    )
    out.write_text(svg)
    bar_px = args.length_um / args.pixel_size_um
    print(f"Wrote: {out}")
    print(f"  Tile: {args.tile_size_px} px = "
          f"{args.tile_size_px * args.pixel_size_um:g} um")
    print(f"  Bar:  {args.length_um:g} um = {bar_px:.2f} px "
          f"({bar_px / args.tile_size_px * 100:.1f}% of tile width)")


if __name__ == "__main__":
    main()
