"""Recursive xy cut method for ordering bounding boxes in a document layout."""

from collections.abc import Iterator
from dataclasses import dataclass
from operator import itemgetter
from typing import Self

# A box's corner coordinates (x1, y1, x2, y2), read once up front, followed by
# the box's position in the input list.
_Item = tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Point:
    """A point with two dimensional, non-negative pixel coordinates."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(f"coordinates must be non-negative, got ({self.x}, {self.y})")


@dataclass(frozen=True, slots=True)
class Rectangle:
    """A rectangular bounding box. Subclass to attach further metadata."""

    bottom_right: Point
    top_left: Point
    text: str | None = None

    def __post_init__(self) -> None:
        if self.bottom_right.x < self.top_left.x or self.bottom_right.y < self.top_left.y:
            raise ValueError("bottom_right must not be above or to the left of top_left")

    def __repr__(self) -> str:
        return (
            f"Rectangle(top_left=({self.top_left.x}, {self.top_left.y}), "
            f"bottom_right=({self.bottom_right.x}, {self.bottom_right.y}))"
        )

    def intersects(self, other: Self) -> bool:
        """Check if a Rectangle intersects another one."""
        return not (
            self.top_left.x > other.bottom_right.x
            or self.bottom_right.x < other.top_left.x
            or self.top_left.y > other.bottom_right.y
            or self.bottom_right.y < other.top_left.y
        )


def xy_cut(
    bboxes: list[Rectangle],
    depth: int = 0,
    max_depth: int = 10,
    horizontal: bool = True,
) -> list[Rectangle]:
    """
    Perform the recursive xy cut on a list of bounding boxes.

    Parameters
    ----------
    bboxes: list[Rectangle]
        A list of predetermined bounding boxes.
    depth: int
        Initial depth of xy cut tree. Must be non-negative.
    max_depth: int
        Maximum depth of the xy cut tree - determines when to stop the recursion.
        Must be non-negative.
    horizontal: bool
        Flag determining whether the cut is going to be horizontal or vertical.

    Returns
    -------
    list[Rectangle]
        Reordered bounding boxes on the basis of the recursive cuts made.

    Raises
    ------
    ValueError
        If ``depth`` or ``max_depth`` is negative.

    Note: Currently the algorithm finds the largest gap in a given direction and makes
    a cut there and then alternates between horizontal and vertical cuts. However, the
    algorithm would be more accurate if it makes a cut at all the gaps in the same direction
    per iteration.

    Boxes with zero extent along the cut axis never create or split a gap, but are still
    assigned to one side of each cut.

    Sources:
    The paper on the xy cut used to implement the algorithm:
    http://haralick.org/conferences/71280952.pdf
    """
    if depth < 0 or max_depth < 0:
        raise ValueError(f"depth and max_depth must be non-negative, got {depth}, {max_depth}")

    # If the maximum depth is reached return the boxes.
    if depth >= max_depth or not bboxes:
        return bboxes

    items = [
        (box.top_left.x, box.top_left.y, box.bottom_right.x, box.bottom_right.y, position)
        for position, box in enumerate(bboxes)
    ]
    # Sort the boxes first vertically, then horizontally. Splitting preserves relative
    # order, so every subset stays sorted and this only needs doing once.
    items.sort(key=itemgetter(1, 0))
    return [bboxes[item[4]] for item in _cut(items, depth, max_depth, horizontal)]


def _cut(
    items: list[_Item],
    depth: int,
    max_depth: int,
    horizontal: bool,
    other_direction_failed: bool = False,
) -> Iterator[_Item]:
    """Recursively split ``items`` and yield them in reading order."""
    if depth >= max_depth or len(items) < 2:
        yield from items
        return

    # Indices of the start and end coordinates along the axis being cut.
    start, end = (0, 2) if horizontal else (1, 3)
    cut_point = _widest_gap([(item[start], item[end]) for item in items])

    # If there are no gaps, retry with the same boxes in the other direction.
    # If that direction had no gaps either, further recursion can't change anything.
    if cut_point is None:
        if other_direction_failed:
            yield from items
        else:
            yield from _cut(
                items, depth + 1, max_depth, not horizontal, other_direction_failed=True
            )
        return

    # Split the boxes around the cut point. No box straddles a gap,
    # so every box ends up in exactly one part.
    yield from _cut(
        [item for item in items if item[end] <= cut_point], depth + 1, max_depth, not horizontal
    )
    yield from _cut(
        [item for item in items if item[end] > cut_point], depth + 1, max_depth, not horizontal
    )


def _widest_gap(spans: list[tuple[int, int]]) -> int | None:
    """
    Find the start of the widest gap between ``(start, end)`` spans on one axis.

    Sweeping the spans in order of their start while tracking the furthest end reached
    so far, any span starting beyond that point opens a gap. Ties go to the first
    widest gap.

    Returns
    -------
    int | None
        The coordinate where the widest gap starts, or ``None`` if there is no gap.
    """
    # Spans with zero extent cover nothing, so they neither create nor split a gap.
    covering = sorted(span for span in spans if span[1] > span[0])
    if not covering:
        return None

    reach = covering[0][1]
    widest = 0
    cut_point: int | None = None
    for start, end in covering[1:]:
        if start - reach > widest:
            widest = start - reach
            cut_point = reach
        if end > reach:
            reach = end
    return cut_point
