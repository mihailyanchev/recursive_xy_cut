import pytest

from recursive_xy_cut import Point, Rectangle, xy_cut


def make_box(x1: int, y1: int, x2: int, y2: int, index: int) -> Rectangle:
    return Rectangle(top_left=Point(x=x1, y=y1), bottom_right=Point(x=x2, y=y2), index=index)


def order(boxes: list[Rectangle]) -> list[int | None]:
    return [box.index for box in boxes]


def test_empty_input_returns_empty_list() -> None:
    assert xy_cut([], image_dims=(100, 100)) == []


def test_single_box_is_returned() -> None:
    box = make_box(10, 10, 50, 50, index=0)
    assert xy_cut([box], image_dims=(100, 100)) == [box]


def test_two_column_layout_reads_column_by_column() -> None:
    # Left column: boxes 0, 1; right column: boxes 2, 3. Shuffled on input.
    boxes = [
        make_box(110, 60, 190, 90, index=3),
        make_box(10, 10, 90, 40, index=0),
        make_box(110, 10, 190, 40, index=2),
        make_box(10, 60, 90, 90, index=1),
    ]
    assert order(xy_cut(boxes, image_dims=(100, 200))) == [0, 1, 2, 3]


def test_header_above_two_columns() -> None:
    boxes = [
        make_box(110, 60, 190, 90, index=2),
        make_box(10, 60, 90, 90, index=1),
        make_box(10, 10, 190, 40, index=0),  # full-width header
    ]
    assert order(xy_cut(boxes, image_dims=(100, 200))) == [0, 1, 2]


def test_no_boxes_are_lost_or_duplicated() -> None:
    boxes = [
        make_box(x, y, x + 15, y + 15, index=i)
        for i, (x, y) in enumerate(
            [(0, 0), (20, 0), (40, 0), (0, 20), (20, 20), (40, 20), (5, 45), (35, 45)]
        )
    ]
    result = xy_cut(boxes, image_dims=(70, 70))
    assert sorted(order(result)) == list(range(len(boxes)))  # type: ignore[type-var]


@pytest.mark.parametrize("max_depth", [0, 1, 3])
def test_max_depth_is_respected(max_depth: int) -> None:
    boxes = [make_box(10, 10, 90, 40, index=1), make_box(110, 10, 190, 40, index=0)]
    result = xy_cut(boxes, image_dims=(100, 200), max_depth=max_depth)
    assert len(result) == len(boxes)
    if max_depth == 0:
        assert result == boxes


def test_many_overlapping_boxes_do_not_overflow_profile() -> None:
    # More than 127 stacked boxes previously overflowed an int8 projection profile.
    boxes = [make_box(10, 10, 50, 50, index=i) for i in range(200)]
    assert len(xy_cut(boxes, image_dims=(100, 100))) == 200


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        ((20, 20, 60, 60), True),
        ((50, 50, 60, 60), True),  # touching corners count as intersecting
        ((51, 51, 60, 60), False),
        ((0, 60, 10, 70), False),
    ],
)
def test_rectangle_intersects(other: tuple[int, int, int, int], expected: bool) -> None:
    box = make_box(10, 10, 50, 50, index=0)
    other_box = make_box(*other, index=1)
    assert box.intersects(other_box) is expected
    assert other_box.intersects(box) is expected
