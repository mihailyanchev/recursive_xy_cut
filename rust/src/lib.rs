//! Rust implementation of the recursive XY cut algorithm, exposed to Python with PyO3.
//!
//! Mirrors the pure Python `recursive_xy_cut` package: same classes, same validation and
//! the same reading order for the same input.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

/// A point with two dimensional, non-negative pixel coordinates.
#[pyclass(module = "recursive_xy_cut_rs", frozen, eq, hash, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Point {
    #[pyo3(get)]
    x: i64,
    #[pyo3(get)]
    y: i64,
}

#[pymethods]
impl Point {
    #[new]
    fn new(x: i64, y: i64) -> PyResult<Self> {
        if x < 0 || y < 0 {
            return Err(PyValueError::new_err(format!(
                "coordinates must be non-negative, got ({x}, {y})"
            )));
        }
        Ok(Self { x, y })
    }

    fn __repr__(&self) -> String {
        format!("Point(x={}, y={})", self.x, self.y)
    }
}

/// A rectangular bounding box. Subclass to attach further metadata.
#[pyclass(module = "recursive_xy_cut_rs", frozen, eq, hash, subclass)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct Rectangle {
    #[pyo3(get)]
    bottom_right: Point,
    #[pyo3(get)]
    top_left: Point,
    #[pyo3(get)]
    text: Option<String>,
}

#[pymethods]
impl Rectangle {
    #[new]
    #[pyo3(signature = (bottom_right, top_left, text = None))]
    fn new(bottom_right: Point, top_left: Point, text: Option<String>) -> PyResult<Self> {
        if bottom_right.x < top_left.x || bottom_right.y < top_left.y {
            return Err(PyValueError::new_err(
                "bottom_right must not be above or to the left of top_left",
            ));
        }
        Ok(Self {
            bottom_right,
            top_left,
            text,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Rectangle(top_left=({}, {}), bottom_right=({}, {}))",
            self.top_left.x, self.top_left.y, self.bottom_right.x, self.bottom_right.y
        )
    }

    /// Check if a Rectangle intersects another one.
    fn intersects(&self, other: PyRef<'_, Rectangle>) -> bool {
        !(self.top_left.x > other.bottom_right.x
            || self.bottom_right.x < other.top_left.x
            || self.top_left.y > other.bottom_right.y
            || self.bottom_right.y < other.top_left.y)
    }
}

/// A box's corner coordinates `[x1, y1, x2, y2]`, read once up front, and its position
/// in the input list.
#[derive(Clone, Copy)]
struct Item {
    coords: [i64; 4],
    position: usize,
}

/// A group of boxes still to be split, and how to split it.
struct Group {
    items: Vec<Item>,
    depth: i64,
    horizontal: bool,
    other_direction_failed: bool,
}

/// Perform the recursive xy cut on a list of bounding boxes.
///
/// Accepts this package's `Rectangle`s, whose coordinates are read without going through
/// Python, or any objects with `top_left` and `bottom_right` attributes that have integer
/// `x` and `y` attributes, such as `recursive_xy_cut.Rectangle`. Returns the same box
/// objects in reading order.
#[pyfunction]
#[pyo3(signature = (bboxes, depth = 0, max_depth = 10, horizontal = true))]
fn xy_cut<'py>(
    bboxes: Vec<Bound<'py, PyAny>>,
    depth: i64,
    max_depth: i64,
    horizontal: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if depth < 0 || max_depth < 0 {
        return Err(PyValueError::new_err(format!(
            "depth and max_depth must be non-negative, got {depth}, {max_depth}"
        )));
    }

    // If the maximum depth is reached return the boxes.
    if depth >= max_depth || bboxes.is_empty() {
        return Ok(bboxes);
    }

    let mut items = bboxes
        .iter()
        .enumerate()
        .map(|(position, bbox)| {
            Ok(Item {
                coords: coordinates(bbox)?,
                position,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    // Sort the boxes first vertically, then horizontally. The sort is stable and splitting
    // preserves relative order, so every group stays sorted and this only needs doing once.
    items.sort_by_key(|item| (item.coords[1], item.coords[0]));

    Ok(reading_order(items, depth, max_depth, horizontal)
        .into_iter()
        .map(|position| bboxes[position].clone())
        .collect())
}

/// Read a box's `[x1, y1, x2, y2]` corner coordinates.
fn coordinates(bbox: &Bound<'_, PyAny>) -> PyResult<[i64; 4]> {
    if let Ok(rectangle) = bbox.cast::<Rectangle>() {
        let rectangle = rectangle.get();
        return Ok([
            rectangle.top_left.x,
            rectangle.top_left.y,
            rectangle.bottom_right.x,
            rectangle.bottom_right.y,
        ]);
    }
    let py = bbox.py();
    let top_left = bbox.getattr(intern!(py, "top_left"))?;
    let bottom_right = bbox.getattr(intern!(py, "bottom_right"))?;
    Ok([
        top_left.getattr(intern!(py, "x"))?.extract()?,
        top_left.getattr(intern!(py, "y"))?.extract()?,
        bottom_right.getattr(intern!(py, "x"))?.extract()?,
        bottom_right.getattr(intern!(py, "y"))?.extract()?,
    ])
}

/// Return the input positions of `items` in reading order.
///
/// Uses an explicit stack rather than recursion, so deep cut trees can't overflow the
/// native stack.
fn reading_order(items: Vec<Item>, depth: i64, max_depth: i64, horizontal: bool) -> Vec<usize> {
    let mut order = Vec::with_capacity(items.len());
    let mut stack = vec![Group {
        items,
        depth,
        horizontal,
        other_direction_failed: false,
    }];

    while let Some(group) = stack.pop() {
        if group.depth >= max_depth || group.items.len() < 2 {
            order.extend(group.items.iter().map(|item| item.position));
            continue;
        }

        // Indices of the start and end coordinates along the axis being cut.
        let (start, end) = if group.horizontal { (0, 2) } else { (1, 3) };

        match widest_gap(&group.items, start, end) {
            // Neither direction can be cut, so further splitting can't change anything.
            None if group.other_direction_failed => {
                order.extend(group.items.iter().map(|item| item.position));
            }
            // No gap in this direction: retry the same boxes in the other direction.
            None => stack.push(Group {
                depth: group.depth + 1,
                horizontal: !group.horizontal,
                other_direction_failed: true,
                ..group
            }),
            // No box straddles a gap, so every box ends up in exactly one part. The second
            // part is pushed first so that the first part is read first.
            Some(cut_point) => {
                let (first, second): (Vec<Item>, Vec<Item>) = group
                    .items
                    .into_iter()
                    .partition(|item| item.coords[end] <= cut_point);
                for items in [second, first] {
                    stack.push(Group {
                        items,
                        depth: group.depth + 1,
                        horizontal: !group.horizontal,
                        other_direction_failed: false,
                    });
                }
            }
        }
    }
    order
}

/// Find the start of the widest gap between the boxes' `(start, end)` spans on one axis.
///
/// Sweeping the spans in order of their start while tracking the furthest end reached so
/// far, any span starting beyond that point opens a gap. Ties go to the first widest gap.
/// Spans with zero extent cover nothing, so they neither create nor split a gap.
fn widest_gap(items: &[Item], start: usize, end: usize) -> Option<i64> {
    let mut spans: Vec<(i64, i64)> = items
        .iter()
        .map(|item| (item.coords[start], item.coords[end]))
        .filter(|(span_start, span_end)| span_end > span_start)
        .collect();
    spans.sort_unstable();

    let ((_, first_end), rest) = spans.split_first()?;
    let mut reach = *first_end;
    let mut widest = 0;
    let mut cut_point = None;
    for &(span_start, span_end) in rest {
        let gap = span_start.saturating_sub(reach);
        if gap > widest {
            widest = gap;
            cut_point = Some(reach);
        }
        reach = reach.max(span_end);
    }
    cut_point
}

#[pymodule]
fn recursive_xy_cut_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Rectangle>()?;
    m.add_function(wrap_pyfunction!(xy_cut, m)?)?;
    Ok(())
}
