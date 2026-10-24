import random
from dataclasses import FrozenInstanceError

import pytest

from recursive_xy_cut import Point, Rectangle, xy_cut
from recursive_xy_cut.xy_cut import _widest_gap


def make_box(x1: int, y1: int, x2: int, y2: int) -> Rectangle:
    return Rectangle(top_left=Point(x=x1, y=y1), bottom_right=Point(x=x2, y=y2))


def ids(boxes: list[Rectangle]) -> list[int]:
    """Identify boxes by object identity, since boxes with equal coordinates compare equal."""
    return [id(box) for box in boxes]


def profile_xy_cut(
    bboxes: list[Rectangle], depth: int = 0, max_depth: int = 10, horizontal: bool = True
) -> list[Rectangle]:
    """The original pixel projection profile implementation, kept as a test oracle."""
    if depth == max_depth or not bboxes:
        return bboxes
    bboxes = sorted(bboxes, key=lambda bbox: (bbox.top_left.y, bbox.top_left.x))

    def span(bbox: Rectangle) -> tuple[int, int]:
        if horizontal:
            return bbox.top_left.x, bbox.bottom_right.x
        return bbox.top_left.y, bbox.bottom_right.y

    low = min(span(bbox)[0] for bbox in bboxes)
    high = max(span(bbox)[1] for bbox in bboxes)
    profile = [0] * high
    for bbox in bboxes:
        start, end = span(bbox)
        for i in range(start, end):
            profile[i] += 1

    # Find the first longest run of zeroes between the margins.
    gap_start, gap_length, run_start = None, 0, None
    for i, count in enumerate([*profile[low:high], 1]):
        if count == 0 and run_start is None:
            run_start = i
        elif count != 0 and run_start is not None:
            if i - run_start > gap_length:
                gap_start, gap_length = run_start, i - run_start
            run_start = None

    def recurse(subset: list[Rectangle]) -> list[Rectangle]:
        return profile_xy_cut(subset, depth + 1, max_depth, not horizontal)

    if gap_start is None or gap_start < 1:
        return recurse(bboxes)
    cut_point = gap_start + low
    return recurse([bbox for bbox in bboxes if span(bbox)[1] <= cut_point]) + recurse(
        [bbox for bbox in bboxes if span(bbox)[1] > cut_point]
    )


def random_page(rng: random.Random, width: int, height: int) -> list[Rectangle]:
    boxes = []
    for _ in range(rng.randint(1, 40)):
        x1, y1 = rng.randrange(width - 1), rng.randrange(height - 1)
        x2 = rng.randint(x1 + 1, min(width, x1 + width // 4))
        y2 = rng.randint(y1 + 1, min(height, y1 + height // 8))
        boxes.append(make_box(x1, y1, x2, y2))
    return boxes


def test_empty_input_returns_empty_list() -> None:
    assert xy_cut([]) == []


def test_single_box_is_returned() -> None:
    box = make_box(10, 10, 50, 50)
    assert xy_cut([box]) == [box]


def test_two_column_layout_reads_column_by_column() -> None:
    left_top, left_bottom = make_box(10, 10, 90, 40), make_box(10, 60, 90, 90)
    right_top, right_bottom = make_box(110, 10, 190, 40), make_box(110, 60, 190, 90)
    result = xy_cut([right_bottom, left_top, right_top, left_bottom])
    assert result == [left_top, left_bottom, right_top, right_bottom]


def test_header_above_two_columns() -> None:
    header = make_box(10, 10, 190, 40)
    left, right = make_box(10, 60, 90, 90), make_box(110, 60, 190, 90)
    assert xy_cut([right, left, header]) == [header, left, right]


def test_no_boxes_are_lost_or_duplicated() -> None:
    boxes = [
        make_box(x, y, x + 15, y + 15)
        for x, y in [(0, 0), (20, 0), (40, 0), (0, 20), (20, 20), (40, 20), (5, 45), (35, 45)]
    ]
    assert sorted(ids(xy_cut(boxes))) == sorted(ids(boxes))


@pytest.mark.parametrize("max_depth", [0, 1, 3])
def test_max_depth_is_respected(max_depth: int) -> None:
    boxes = [make_box(110, 10, 190, 40), make_box(10, 10, 90, 40)]
    result = xy_cut(boxes, max_depth=max_depth)
    assert len(result) == len(boxes)
    if max_depth == 0:
        assert result == boxes


def test_many_overlapping_boxes_are_kept() -> None:
    boxes = [make_box(10, 10, 50, 50) for _ in range(200)]
    assert sorted(ids(xy_cut(boxes))) == sorted(ids(boxes))


def test_uncuttable_boxes_stop_early_regardless_of_max_depth() -> None:
    # Overlapping boxes have no gap in either direction. Recursing down to max_depth
    # would exceed the interpreter's recursion limit.
    first, second = make_box(10, 10, 50, 50), make_box(20, 20, 60, 60)
    assert xy_cut([second, first], max_depth=10_000) == [first, second]


@pytest.mark.parametrize("seed", range(5))
def test_matches_original_profile_implementation(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(100):
        page = random_page(rng, width=rng.choice([60, 300]), height=rng.choice([80, 400]))
        for max_depth in (1, 3, 10):
            expected = ids(profile_xy_cut(page, max_depth=max_depth))
            assert ids(xy_cut(page, max_depth=max_depth)) == expected


def test_zero_width_box_does_not_split_a_gap() -> None:
    # Gaps at 50..70 (width 20) and 100..115 (width 15). A zero-width box at x=60 sits
    # inside the wider gap; counting it would halve that gap and move the cut to 100.
    assert _widest_gap([(10, 50), (60, 60), (70, 100), (115, 190)]) == 50
    boxes = [
        make_box(10, 10, 50, 40),
        make_box(60, 10, 60, 40),
        make_box(70, 10, 100, 40),
        make_box(115, 10, 190, 40),
    ]
    assert sorted(ids(xy_cut(boxes))) == sorted(ids(boxes))


@pytest.mark.parametrize(("x", "y"), [(-1, 0), (0, -1)])
def test_point_rejects_negative_coordinates(x: int, y: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Point(x=x, y=y)


def test_point_is_immutable() -> None:
    point = Point(x=1, y=2)
    with pytest.raises(FrozenInstanceError):
        point.x = 3  # type: ignore[misc]


@pytest.mark.parametrize("corners", [(50, 10, 10, 50), (10, 50, 50, 10)])
def test_rectangle_rejects_inverted_corners(corners: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="bottom_right"):
        make_box(*corners)


@pytest.mark.parametrize(
    ("field", "value"),
    [("top_left", Point(x=20, y=20)), ("bottom_right", Point(x=20, y=20)), ("text", "Title")],
)
def test_rectangle_is_immutable(field: str, value: object) -> None:
    box = make_box(10, 10, 50, 50)
    with pytest.raises(FrozenInstanceError):
        setattr(box, field, value)
    assert box == make_box(10, 10, 50, 50)


def test_rectangle_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError):
        Rectangle(top_left=Point(x=0, y=0), bottom_right=Point(x=1, y=1), index=0)  # type: ignore[call-arg]


@pytest.mark.parametrize("kwargs", [{"depth": -1}, {"max_depth": -1}])
def test_xy_cut_rejects_negative_depths(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        xy_cut([make_box(10, 10, 50, 50)], **kwargs)  # type: ignore[arg-type]


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
    box = make_box(10, 10, 50, 50)
    other_box = make_box(*other)
    assert box.intersects(other_box) is expected
    assert other_box.intersects(box) is expected
