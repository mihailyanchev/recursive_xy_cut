from collections.abc import Sequence
from typing import Protocol

class _HasCoordinates(Protocol):
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...

class BoxLike(Protocol):
    """Any box with integer corner coordinates, such as ``recursive_xy_cut.Rectangle``."""

    @property
    def top_left(self) -> _HasCoordinates: ...
    @property
    def bottom_right(self) -> _HasCoordinates: ...

class Point:
    """A point with two dimensional, non-negative pixel coordinates."""

    def __new__(cls, x: int, y: int) -> Point: ...
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Rectangle:
    """A rectangular bounding box. Subclass to attach further metadata."""

    def __new__(
        cls, bottom_right: Point, top_left: Point, text: str | None = None
    ) -> Rectangle: ...
    @property
    def bottom_right(self) -> Point: ...
    @property
    def top_left(self) -> Point: ...
    @property
    def text(self) -> str | None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def intersects(self, other: Rectangle) -> bool:
        """Check if a Rectangle intersects another one."""

def xy_cut[B: BoxLike](
    bboxes: Sequence[B],
    depth: int = 0,
    max_depth: int = 10,
    horizontal: bool = True,
) -> list[B]:
    """Perform the recursive xy cut on a list of bounding boxes."""
