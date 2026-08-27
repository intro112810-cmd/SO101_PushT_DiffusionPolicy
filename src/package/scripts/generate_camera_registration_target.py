#!/usr/bin/env python3
"""Generate the exact vector A4 ChArUco registration target."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import cast, Protocol

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim_to_real.camera_registration_target import (
    BOARD_COLUMNS,
    BOARD_ROWS,
    MARKER_SIZE_MM,
    SQUARE_SIZE_MM,
    TARGET_SCHEMA,
)


class _Aruco(Protocol):
    DICT_5X5_100: int

    def getPredefinedDictionary(self, identifier: int) -> object: ...

    def generateImageMarker(
        self, dictionary: object, marker_id: int, side_pixels: int, border_bits: int = 1
    ) -> NDArray[np.uint8]: ...


class _Cv2(Protocol):
    aruco: _Aruco


def _marker_rectangles() -> list[str]:
    cv2 = cast("_Cv2", import_module("cv2"))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    module = MARKER_SIZE_MM / 7.0
    margin = (SQUARE_SIZE_MM - MARKER_SIZE_MM) / 2.0
    result: list[str] = []
    marker_id = 0
    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLUMNS):
            x = column * SQUARE_SIZE_MM
            y = row * SQUARE_SIZE_MM
            if (row + column) % 2 == 0:
                result.append(f'<rect x="{x:g}" y="{y:g}" width="25" height="25"/>')
                continue
            image = cv2.aruco.generateImageMarker(dictionary, marker_id, 7, 1)
            for module_row, module_column in np.argwhere(image == 0):
                module_x = x + margin + float(module_column) * module
                module_y = y + margin + float(module_row) * module
                result.append(
                    f'<rect id="marker-{marker_id}-r{module_row}c{module_column}" '
                    f'x="{module_x:.9f}" y="{module_y:.9f}" '
                    f'width="{module:.9f}" height="{module:.9f}"/>'
                )
            marker_id += 1
    if marker_id != 24:
        raise RuntimeError("unexpected ChArUco marker inventory")
    return result


def render_svg() -> str:
    board = "\n      ".join(_marker_rectangles())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  {TARGET_SCHEMA}; OpenCV DICT_5X5_100.
  Print A4 at ACTUAL SIZE / 100%; disable fit-to-page and scaling.
  ChArUco: 8 x 6 squares, square 25.0 mm, marker 18.0 mm, 7 x 5 ID-bound corners.
  Finished board 200.0 x 150.0 mm; verification ruler 100.0 mm.
-->
<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">
  <rect width="210" height="297" fill="white"/>
  <g font-family="DejaVu Sans,Arial,sans-serif" fill="black" font-size="4">
    <text x="5" y="18">OUTER TOP-LEFT</text>
    <text x="72" y="14">KEEP THIS EDGE ALONG +X  ----&gt;</text>
  </g>
  <g id="charuco-board" transform="translate(5 20)" fill="black" shape-rendering="crispEdges">
      {board}
    <rect x="0" y="0" width="200" height="150" fill="none" stroke="black" stroke-width="0.2"/>
  </g>
  <g id="scale-ruler" transform="translate(10 190)" fill="none" stroke="black" stroke-width="0.25">
    <path d="M0 0H100 M0 0v8 M10 0v4 M20 0v4 M30 0v4 M40 0v4 M50 0v8 M60 0v4 M70 0v4 M80 0v4 M90 0v4 M100 0v8"/>
    <path d="M0 10H100V20H0Z M10 10V20 M20 10V20 M30 10V20 M40 10V20 M50 10V20 M60 10V20 M70 10V20 M80 10V20 M90 10V20"/>
  </g>
  <g font-family="DejaVu Sans,Arial,sans-serif" fill="black">
    <text x="10" y="187" font-size="4">PRINT-SCALE CHECK: endpoint-to-endpoint = exactly 100.0 mm</text>
    <text x="10" y="218" font-size="5" font-weight="bold">SO-101 CAMERA REGISTRATION CHARUCO v1</text>
    <text x="10" y="226" font-size="4">DICT_5X5_100; 8 x 6 squares; 7 x 5 corners; square = exactly 25.0 mm; marker = 18.0 mm</text>
    <text x="10" y="233" font-size="4">Board = exactly 200.0 x 150.0 mm. Mount flat on rigid matte backing.</text>
    <text x="10" y="240" font-size="4">Print A4 at 100% / Actual Size. Disable Fit, Shrink, Scale, and borderless expansion.</text>
    <text x="10" y="247" font-size="4">Reject if ruler is outside 99.5-100.5 mm or a square outside 24.875-25.125 mm.</text>
    <text x="10" y="254" font-size="4">Do not crop. No folds, gloss, bubbles, or warped backing.</text>
    <text x="10" y="265" font-size="3.5">Target contract: {TARGET_SCHEMA}</text>
  </g>
</svg>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output: Path = args.output
    if output.is_symlink() or not output.parent.is_dir():
        raise ValueError("target output parent must be an existing non-symlink directory")
    output.write_text(render_svg(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
