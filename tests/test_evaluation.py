from typing import Any

import pytest

from evaluation.chunkr import Element, Page, page_scores, parse_coco, without_page_furniture
from evaluation.metrics import exact_match, kendall_tau, position_accuracy, spearman_rho
from recursive_xy_cut import Point, Rectangle


def make_box(x1: int, y1: int, x2: int, y2: int) -> Rectangle:
    return Rectangle(top_left=Point(x=x1, y=y1), bottom_right=Point(x=x2, y=y2))


@pytest.mark.parametrize(
    ("predicted", "exact", "position", "tau", "rho"),
    [
        ([0, 1, 2, 3], 1.0, 1.0, 1.0, 1.0),
        ([3, 2, 1, 0], 0.0, 0.0, -1.0, -1.0),
        # One swap: 2 of 3 pairs concordant, squared distances sum to 2.
        ([1, 0, 2], 0.0, 1 / 3, 1 / 3, 0.5),
        ([0], 1.0, 1.0, 1.0, 1.0),
    ],
)
def test_metrics(
    predicted: list[int], exact: float, position: float, tau: float, rho: float
) -> None:
    assert exact_match(predicted) == exact
    assert position_accuracy(predicted) == pytest.approx(position)
    assert kendall_tau(predicted) == pytest.approx(tau)
    assert spearman_rho(predicted) == pytest.approx(rho)


def test_parse_coco_orders_elements_by_annotation_id() -> None:
    coco: dict[str, Any] = {
        "categories": [{"id": 0, "name": "Title"}, {"id": 1, "name": "Text Block"}],
        "images": [{"id": 7, "file_name": "page.png", "doc_category": "legal"}],
        "annotations": [
            {"id": 12, "image_id": 7, "category_id": 1, "bbox": [10, 50, 80, 30]},
            {"id": 11, "image_id": 7, "category_id": 0, "bbox": [10, 10, 80, 20]},
        ],
    }
    assert parse_coco(coco) == [
        Page(
            file_name="page.png",
            doc_category="legal",
            elements=(
                Element(box=make_box(10, 10, 90, 30), category="Title"),
                Element(box=make_box(10, 50, 90, 80), category="Text Block"),
            ),
        )
    ]


def test_without_page_furniture_keeps_content_in_order() -> None:
    header, footer = (
        Element(make_box(0, 0, 9, 9), "Header"),
        Element(make_box(0, 90, 9, 99), "Footer"),
    )
    title, text = (
        Element(make_box(0, 10, 9, 19), "Title"),
        Element(make_box(0, 20, 9, 29), "Text Block"),
    )
    page = Page("page.png", "legal", (header, title, text, footer))
    assert without_page_furniture(page).elements == (title, text)


def test_page_scores_compares_predicted_with_ground_truth_order() -> None:
    first, second = (
        Element(make_box(0, 0, 9, 9), "Title"),
        Element(make_box(0, 20, 9, 29), "Text Block"),
    )

    def reverse(boxes: list[Rectangle]) -> list[Rectangle]:
        return boxes[::-1]

    scores = page_scores(Page("page.png", "legal", (first, second)), reverse)
    assert scores["Exact match"] == 0.0
    assert scores["Kendall's τ"] == -1.0
