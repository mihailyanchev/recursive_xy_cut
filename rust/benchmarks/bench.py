"""
Benchmark the Rust xy_cut against the pure Python and original numpy implementations.

Run from the ``rust`` directory with::

    uv run python benchmarks/bench.py
"""

import random
import timeit
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import original_xy_cut as original
import recursive_xy_cut_rs as rs

import recursive_xy_cut as py

WIDTH, HEIGHT = 2480, 3508  # A4 at 300 dpi

Coords = list[tuple[int, int, int, int]]


def random_coords(n: int, seed: int) -> Coords:
    """Return ``n`` random boxes up to a quarter page wide and an eighth of a page tall."""
    rng = random.Random(seed)
    coords = []
    for _ in range(n):
        x1, y1 = rng.randrange(WIDTH - 1), rng.randrange(HEIGHT - 1)
        x2 = rng.randint(x1 + 1, min(WIDTH, x1 + WIDTH // 4))
        y2 = rng.randint(y1 + 1, min(HEIGHT, y1 + HEIGHT // 8))
        coords.append((x1, y1, x2, y2))
    return coords


def grid_coords(columns: int, rows: int) -> Coords:
    """Return a shuffled grid of boxes, like a multi-column page of paragraphs."""
    column_width, row_height = WIDTH // columns, HEIGHT // rows
    coords = [
        (
            c * column_width + 20,
            r * row_height + 5,
            (c + 1) * column_width - 20,
            (r + 1) * row_height - 5,
        )
        for c in range(columns)
        for r in range(rows)
    ]
    random.Random(0).shuffle(coords)
    return coords


class Implementation:
    """How to build boxes for, and run, one xy_cut implementation."""

    def __init__(
        self,
        name: str,
        build: Callable[[Coords], list[Any]],
        run: Callable[[list[Any]], list[Any]],
    ) -> None:
        self.name = name
        self.build = build
        self.run = run


def builder(point: Any, rectangle: Any) -> Callable[[Coords], list[Any]]:
    """Return a function that builds a page of boxes from the given classes."""

    def build(coords: Coords) -> list[Any]:
        return [
            rectangle(top_left=point(x=x1, y=y1), bottom_right=point(x=x2, y=y2))
            for x1, y1, x2, y2 in coords
        ]

    return build


IMPLEMENTATIONS = [
    Implementation(
        "original (numpy)",
        builder(original.Point, original.Rectangle),
        lambda boxes: original.xy_cut(boxes, image_dims=(HEIGHT, WIDTH)),
    ),
    Implementation("pure Python", builder(py.Point, py.Rectangle), py.xy_cut),
    Implementation("Rust", builder(rs.Point, rs.Rectangle), rs.xy_cut),
    Implementation("Rust, Python boxes", builder(py.Point, py.Rectangle), rs.xy_cut),
]

CASES = [
    ("30 random boxes", random_coords(30, 1)),
    ("200 random boxes", random_coords(200, 2)),
    ("1000 random boxes", random_coords(1000, 3)),
    ("10000 random boxes", random_coords(10_000, 4)),
    ("3x40 grid", grid_coords(3, 40)),
]


def best_time(function: Callable[[], object]) -> float:
    """Return the best per-call time in seconds over 5 repeats of about 0.2 s each."""
    number, _ = timeit.Timer(function).autorange()
    return min(timeit.repeat(function, number=number, repeat=5)) / number


def build_and_run(implementation: Implementation, coords: Coords) -> None:
    """Build a page of boxes and order it, as a caller would."""
    implementation.run(implementation.build(coords))


def positions(result: Sequence[Any], page: Sequence[Any]) -> list[int]:
    """Map each returned box to its position in the input, by object identity."""
    position = {id(box): i for i, box in enumerate(page)}
    return [position[id(box)] for box in result]


def print_table(title: str, times: dict[str, dict[str, float]]) -> None:
    """Print times in milliseconds, with each implementation's speed-up over the original."""
    names = [implementation.name for implementation in IMPLEMENTATIONS]
    print(f"\n{title}\n{'':20s}" + "".join(f"{name:>24s}" for name in names))
    for case, row in times.items():
        baseline = row[names[0]]
        cells = [f"{row[name] * 1e3:9.3f} ms ({baseline / row[name]:5.1f}x)" for name in names]
        print(f"{case:20s}" + "".join(f"{cell:>24s}" for cell in cells))


def main() -> None:
    """Check all implementations agree, then time them."""
    cut_times: dict[str, dict[str, float]] = {}
    total_times: dict[str, dict[str, float]] = {}
    for case, coords in CASES:
        cut_times[case], total_times[case] = {}, {}
        expected = None
        for implementation in IMPLEMENTATIONS:
            page = implementation.build(coords)
            order = positions(implementation.run(page), page)
            assert expected is None or order == expected, (case, implementation.name)
            expected = order

            cut_times[case][implementation.name] = best_time(partial(implementation.run, page))
            total_times[case][implementation.name] = best_time(
                partial(build_and_run, implementation, coords)
            )

    print_table("xy_cut only", cut_times)
    print_table("building the boxes + xy_cut", total_times)


if __name__ == "__main__":
    main()
