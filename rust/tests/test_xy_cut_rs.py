import random

import pytest
import recursive_xy_cut_rs as rs

import recursive_xy_cut as py


def make_box(x1: int, y1: int, x2: int, y2: int) -> rs.Rectangle:
    return rs.Rectangle(top_left=rs.Point(x=x1, y=y1), bottom_right=rs.Point(x=x2, y=y2))


def make_py_box(x1: int, y1: int, x2: int, y2: int) -> py.Rectangle:
    return py.Rectangle(top_left=py.Point(x=x1, y=y1), bottom_right=py.Point(x=x2, y=y2))


def positions[B](result: list[B], page: list[B]) -> list[int]:
    """Map each returned box to its position in the input, by object identity."""
    position = {id(box): i for i, box in enumerate(page)}
    return [position[id(box)] for box in result]


def random_coords(rng: random.Random, width: int, height: int) -> list[tuple[int, int, int, int]]:
    # Boxes may have zero width or height, to cover gap handling for degenerate boxes.
    coords = []
    for _ in range(rng.randint(1, 60)):
        x1, y1 = rng.randrange(width), rng.randrange(height)
        x2 = rng.randint(x1, min(width, x1 + width // 4))
        y2 = rng.randint(y1, min(height, y1 + height // 8))
        coords.append((x1, y1, x2, y2))
    return coords


@pytest.mark.parametrize("seed", range(5))
def test_matches_python_implementation(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        coords = random_coords(rng, rng.choice([60, 300, 2480]), rng.choice([80, 400, 3508]))
        page, py_page = [make_box(*c) for c in coords], [make_py_box(*c) for c in coords]
        for max_depth in (1, 3, 10):
            for horizontal in (True, False):
                expected = positions(
                    py.xy_cut(py_page, max_depth=max_depth, horizontal=horizontal), py_page
                )
                result = rs.xy_cut(page, max_depth=max_depth, horizontal=horizontal)
                assert positions(result, page) == expected
                # Boxes from the Python package are read through their attributes.
                py_result = rs.xy_cut(py_page, max_depth=max_depth, horizontal=horizontal)
                assert positions(py_result, py_page) == expected


def test_empty_input_returns_empty_list() -> None:
    assert rs.xy_cut([]) == []


def test_two_column_layout_reads_column_by_column() -> None:
    left_top, left_bottom = make_box(10, 10, 90, 40), make_box(10, 60, 90, 90)
    right_top, right_bottom = make_box(110, 10, 190, 40), make_box(110, 60, 190, 90)
    result = rs.xy_cut([right_bottom, left_top, right_top, left_bottom])
    assert result == [left_top, left_bottom, right_top, right_bottom]


def test_max_depth_zero_returns_boxes_unchanged() -> None:
    boxes = [make_box(110, 10, 190, 40), make_box(10, 10, 90, 40)]
    assert rs.xy_cut(boxes, max_depth=0) == boxes


def test_uncuttable_boxes_stop_early_regardless_of_max_depth() -> None:
    first, second = make_box(10, 10, 50, 50), make_box(20, 20, 60, 60)
    assert rs.xy_cut([second, first], max_depth=10_000) == [first, second]


@pytest.mark.parametrize("kwargs", [{"depth": -1}, {"max_depth": -1}])
def test_xy_cut_rejects_negative_depths(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rs.xy_cut([make_box(10, 10, 50, 50)], **kwargs)  # type: ignore[arg-type]


def test_xy_cut_rejects_objects_without_coordinates() -> None:
    with pytest.raises(AttributeError):
        rs.xy_cut([object(), object()])  # type: ignore[type-var]


@pytest.mark.parametrize(("x", "y"), [(-1, 0), (0, -1)])
def test_point_rejects_negative_coordinates(x: int, y: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rs.Point(x=x, y=y)


@pytest.mark.parametrize("corners", [(50, 10, 10, 50), (10, 50, 50, 10)])
def test_rectangle_rejects_inverted_corners(corners: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="bottom_right"):
        make_box(*corners)


@pytest.mark.parametrize(
    ("obj", "field", "value"),
    [
        (rs.Point(x=1, y=2), "x", 3),
        (make_box(10, 10, 50, 50), "top_left", rs.Point(x=20, y=20)),
        (make_box(10, 10, 50, 50), "text", "Title"),
    ],
)
def test_models_are_immutable(obj: object, field: str, value: object) -> None:
    with pytest.raises(AttributeError):
        setattr(obj, field, value)


def test_models_compare_and_hash_by_value() -> None:
    assert rs.Point(x=1, y=2) == rs.Point(x=1, y=2)
    assert make_box(10, 10, 50, 50) == make_box(10, 10, 50, 50)
    assert make_box(10, 10, 50, 50) != make_box(10, 10, 50, 51)
    assert len({make_box(10, 10, 50, 50), make_box(10, 10, 50, 50)}) == 1


def test_repr_matches_python_implementation() -> None:
    assert repr(rs.Point(x=1, y=2)) == repr(py.Point(x=1, y=2))
    assert repr(make_box(10, 20, 30, 40)) == repr(make_py_box(10, 20, 30, 40))


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
