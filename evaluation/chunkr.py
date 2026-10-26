"""
Evaluate the recursive xy cut on the Chunkr reading order benchmark.

Dataset: https://huggingface.co/datasets/ChunkrAI/chunkr-reading-order-bench-oss

Run from the repository root with::

    uv run python -m evaluation.chunkr

Only the COCO annotation file (about 2 MB) is downloaded, at a pinned revision, into
``data/chunkr``. The page images aren't needed, since the methods only use the boxes.

The dataset has no explicit reading order field. Within a page, the annotation ids
increase in reading order, so the ground truth order is the order of the ids.
"""

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.metrics import METRICS
from recursive_xy_cut import Point, Rectangle, xy_cut

DATASET = "ChunkrAI/chunkr-reading-order-bench-oss"
REVISION = "d6b5ddf06a6479a42bb0b33c243801171e042fc7"
ANNOTATIONS_URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/_annotations.coco.json"
)
# Fingerprint of the annotation file at REVISION, so every run scores against
# exactly the same ground truth.
ANNOTATIONS_SHA256 = "93974a16cb43a44656f293b933abd1a713d2bff2bfa71cd7b74987edb26bdbfa"

# Running headers, footers and page numbers, which can be left out of the evaluation.
PAGE_FURNITURE = frozenset({"Header", "Footer", "Page Number"})

# Document categories with fewer pages than this are too small to compare reliably.
MIN_CATEGORY_PAGES = 20

Method = Callable[[list[Rectangle]], list[Rectangle]]


def top_to_bottom(boxes: list[Rectangle]) -> list[Rectangle]:
    """Order boxes top to bottom, then left to right, by their top left corner."""
    return sorted(boxes, key=lambda box: (box.top_left.y, box.top_left.x))


METHODS: dict[str, Method] = {
    "TBLR": top_to_bottom,
    "xy_cut": xy_cut,
    "xy_cut (horizontal=False)": partial(xy_cut, horizontal=False),
}


@dataclass(frozen=True, slots=True)
class Element:
    """A layout element on a page."""

    box: Rectangle
    category: str


@dataclass(frozen=True, slots=True)
class Page:
    """A page and its layout elements, in ground truth reading order."""

    file_name: str
    doc_category: str
    elements: tuple[Element, ...]


def load_annotations(data_dir: Path) -> dict[str, Any]:
    """Return the COCO annotations, downloading and caching them on first use."""
    path = data_dir / REVISION / "_annotations.coco.json"
    if path.exists():
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(ANNOTATIONS_URL) as response:
            content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != ANNOTATIONS_SHA256:
        raise ValueError(f"unexpected SHA-256 {digest} for {ANNOTATIONS_URL}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    annotations: dict[str, Any] = json.loads(content)
    return annotations


def parse_coco(coco: dict[str, Any]) -> list[Page]:
    """Convert COCO annotations to pages with their elements in reading order."""
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    pages = []
    for image in coco["images"]:
        annotations = sorted(annotations_by_image[image["id"]], key=lambda a: a["id"])
        elements = tuple(
            Element(box=_rectangle(a["bbox"]), category=categories[a["category_id"]])
            for a in annotations
        )
        pages.append(Page(image["file_name"], image["doc_category"], elements))
    return pages


def _rectangle(bbox: list[int]) -> Rectangle:
    """Convert a COCO ``[x, y, width, height]`` box to a Rectangle."""
    x, y, width, height = bbox
    return Rectangle(top_left=Point(x=x, y=y), bottom_right=Point(x=x + width, y=y + height))


def without_page_furniture(page: Page) -> Page:
    """Return the page without its running headers, footers and page numbers."""
    elements = tuple(e for e in page.elements if e.category not in PAGE_FURNITURE)
    return replace(page, elements=elements)


def page_scores(page: Page, method: Method) -> dict[str, float]:
    """Order a page's boxes with ``method`` and score the order with every metric."""
    boxes = [element.box for element in page.elements]
    position = {id(box): i for i, box in enumerate(boxes)}
    predicted = [position[id(box)] for box in method(boxes)]
    return {name: metric(predicted) for name, metric in METRICS.items()}


def score_pages(pages: list[Page]) -> dict[str, list[dict[str, float]]]:
    """Return every method's per-page scores, in the order of ``pages``."""
    return {name: [page_scores(page, method) for page in pages] for name, method in METHODS.items()}


def print_summary(title: str, pages: list[Page]) -> None:
    """Print a Markdown table of each method's mean scores over ``pages``."""
    scores = score_pages(pages)
    print(f"\n### {title} ({len(pages)} pages)\n")
    print("| Method | " + " | ".join(METRICS) + " |")
    print("| --- |" + " --- |" * len(METRICS))
    for name, page_results in scores.items():
        means = [mean(result[metric] for result in page_results) for metric in METRICS]
        print(f"| {name} | " + " | ".join(f"{value:.1%}" for value in means) + " |")


def print_by_category(pages: list[Page], metric: str) -> None:
    """Print a Markdown table of each method's mean ``metric`` per document category."""
    scores = score_pages(pages)
    indices_by_category = defaultdict(list)
    for i, page in enumerate(pages):
        indices_by_category[page.doc_category].append(i)

    print(f"\n### {metric} by document category\n")
    print("| Category | Pages | " + " | ".join(METHODS) + " |")
    print("| --- | --- |" + " --- |" * len(METHODS))
    for category, indices in sorted(indices_by_category.items(), key=lambda c: -len(c[1])):
        flag = "*" if len(indices) < MIN_CATEGORY_PAGES else ""
        means = [mean(scores[name][i][metric] for i in indices) for name in METHODS]
        cells = " | ".join(f"{value:.1%}" for value in means)
        print(f"| {category}{flag} | {len(indices)} | {cells} |")
    print(f"\n\\* Fewer than {MIN_CATEGORY_PAGES} pages, too few to compare reliably.")


def main() -> None:
    """Download the annotations if needed and print the evaluation tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/chunkr"),
        help="where to cache the annotations (default: %(default)s)",
    )
    args = parser.parse_args()

    pages = parse_coco(load_annotations(args.data_dir))
    content_pages = [without_page_furniture(page) for page in pages]

    print_summary("All pages", pages)
    print_summary("Pages with at least two elements", [p for p in pages if len(p.elements) > 1])
    print_summary(
        "Without headers, footers and page numbers", [p for p in content_pages if p.elements]
    )
    print_by_category(pages, "Exact match")
    print_by_category(pages, "Kendall's τ")


if __name__ == "__main__":
    main()
