"""Implementation of the recursive xy cut method for ordering bounding boxes in a document layout."""
from pydantic import BaseModel
from enum import Enum
import numpy as np


class Point(BaseModel):
    """A point with two dimensional coordinates."""
    x: int
    y: int


class Rectangle:
"""A rectangular bounding box."""

    def __init__(self, bottom_right: Point, top_left: Point, boxtype: BoxType = None, text: str = None, index: int = None):
        self.bottom_right = bottom_right
        self.top_left = top_left
        self.boxtype = boxtype
        self.text = text
        self.index = index

    def intersects(self, other: 'Rectangle'):
        """Check if a Rectangle intersects another one."""
        return not (
            self.top_left.x > other.bottom_right.x
            or self.bottom_right.x < other.top_left.x
            or self.top_left.y > other.bottom_right.y
            or self.bottom_right.y < other.top_left.y
        )


def xy_cut(
    bboxes: list[Rectangle],
    image_dims: tuple,
    depth: int = 0,
    max_depth: int = 10,
    horizontal: bool = True,
):
    """
    Perform the recursive xy cut on a list of bounding boxes.

    Parameters
    ----------
    bboxes: List[Rectangle]
        A list of predetermined bounding boxes.
    image_dims: tuple
        A tuple countaining the image dimensions where the zeroth element is the height
        and the first element is the width.
    depth: int
        Initial depth of xy cut tree.
    max_depth: int
        Maximum depth of the xy cut tree - determines when to stop the recursion.
    horizontal: bool
        Flag determining whether the cut is going to be horizontal or vertical.

    Returns
    -------
    List[Rectangle]
        Reordered bounding boxes on the basis of the recursive cuts made.

    Note: Currently the algorithm finds the largest gap in a given direction and makes
    a cut there and then alternates between horizontal and vertical cuts. However, the 
    algorithm would be more accurate if it makes a cut at all the gaps in the same direction
    per iteration.
    
    Sources:
    The paper on the xy cut used to implement the algorithm:
    http://haralick.org/conferences/71280952.pdf
    The method for finding the gaps in the projection profiles using numpy:
    https://stackoverflow.com/a/38161867
    """
    # If the maximum depth is reached return the boxes.
    if depth == max_depth or not bboxes:
        return bboxes

    # Sort the bounding boxes first vertically, then horizontally.
    bboxes = sorted(bboxes, key=lambda bbox: (bbox.top_left.y, bbox.top_left.x))

    # Get page margins so that no cut points are assigned within
    top_margin = min([box.top_left.y for box in bboxes])
    bottom_margin = max([box.bottom_right.y for box in bboxes])
    left_margin = min([box.top_left.x for box in bboxes])
    right_margin = max([box.bottom_right.x for box in bboxes])

    # Create a blank projections profile (1D vector of length the dim of the image)
    vector_length = image_dims[1 if horizontal else 0]
    profile = np.zeros(vector_length, dtype=np.int8)

    # Assign the bounding box projections to the profile: add 1 if a bounding box is present in this pixel
    for box in bboxes:
        if horizontal:
            profile[box.top_left.x : box.bottom_right.x] += 1
        else:
            profile[box.top_left.y : box.bottom_right.y] += 1

    # Get start, stop index pairs for islands/sequences of zeroes
    if horizontal:
        idx_pairs = np.where(
            np.diff(
                np.hstack(([False], profile[left_margin:right_margin] == 0, [False]))
            )
        )[0].reshape(-1, 2)
    else:
        idx_pairs = np.where(
            np.diff(
                np.hstack(([False], profile[top_margin:bottom_margin] == 0, [False]))
            )
        )[0].reshape(-1, 2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    if len(idx_pairs) > 0:
        start_longest_seq = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 0]
    else:
        return xy_cut(
            bboxes=bboxes,
            image_dims=image_dims,
            depth=depth + 1,
            horizontal=not horizontal,
        )

    # Get the max distance:
    if start_longest_seq < 1:
        # if there are no gaps just restart with the same bounding boxes
        # but alternate the horizontal argument
        return xy_cut(
            bboxes=bboxes,
            image_dims=image_dims,
            depth=depth + 1,
            horizontal=not horizontal,
        )

    # The cut point is made at the start of the logest gap,
    # but we also add the top margin in the vertical case
    # and the left margin to the horizontal one, because
    # we searched for the gap in the region between the margins.
    if horizontal:
        cut_point = start_longest_seq + left_margin
    else:
        cut_point = start_longest_seq + top_margin

    if horizontal:
        # Split list into two parts around the horizontal cut point
        bboxes1 = [bbox for bbox in bboxes if (bbox.bottom_right.x) <= cut_point]
        bboxes2 = [bbox for bbox in bboxes if (bbox.top_left.x) >= cut_point]
    else:
        # Split list into two parts around the vertical cut point
        bboxes1 = [bbox for bbox in bboxes if (bbox.bottom_right.y) <= cut_point]
        bboxes2 = [bbox for bbox in bboxes if (bbox.top_left.y) >= cut_point]

    # Rerun the the split lists through the xy_cut
    return xy_cut(
        bboxes=bboxes1,
        image_dims=image_dims,
        depth=depth + 1,
        horizontal=not horizontal,
    ) + xy_cut(
        bboxes=bboxes2,
        image_dims=image_dims,
        depth=depth + 1,
        horizontal=not horizontal,
    )
