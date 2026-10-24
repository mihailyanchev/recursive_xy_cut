# recursive_xy_cut_rs

A Rust implementation of the [recursive XY cut](../README.md) algorithm, packaged as a Python
extension with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs).

It has the same API as the pure Python `recursive_xy_cut` package, gives the same reading order for
the same input, and is about 10x faster.

## Usage

```python
from recursive_xy_cut_rs import Point, Rectangle, xy_cut

boxes = [
    Rectangle(top_left=Point(x=110, y=10), bottom_right=Point(x=190, y=40), text="right"),
    Rectangle(top_left=Point(x=10, y=10), bottom_right=Point(x=90, y=40), text="left top"),
    Rectangle(top_left=Point(x=10, y=60), bottom_right=Point(x=90, y=90), text="left bottom"),
]

ordered = xy_cut(boxes)
print([box.text for box in ordered])  # ['left top', 'left bottom', 'right']
```

`Point` and `Rectangle` are immutable, hashable and compare by value, and reject negative
coordinates and inverted corners with `ValueError`, like their pure Python counterparts.

`xy_cut` also accepts any objects with `top_left` and `bottom_right` attributes that have integer
`x` and `y` attributes, such as `recursive_xy_cut.Rectangle`, and returns those same objects in
reading order. Reading coordinates through Python attributes is slower, so this package's own
`Rectangle` is the fastest input.

## Development

Building requires a [Rust toolchain](https://rustup.rs). From this directory:

| Task | Command |
| --- | --- |
| Build the extension and install dependencies | `uv sync` |
| Tests (including agreement with the pure Python package) | `uv run pytest` |
| Type check the stubs, tests and benchmarks | `uv run mypy` |
| Format and lint the Rust code | `cargo fmt` / `cargo clippy --release -- -D warnings` |
| Benchmark | `uv run python benchmarks/bench.py` |

`uv sync` and `uv run` rebuild the extension in release mode whenever the Rust sources change.
Building with plain `cargo build` fails to link on macOS, because Python extensions resolve their
Python symbols at load time; use uv (or `maturin develop`) instead.

## Benchmark

[`benchmarks/bench.py`](benchmarks/bench.py) compares this package with the pure Python package and
with the original numpy/pydantic implementation, on an A4 page at 300 dpi. Times are the best of 5
runs on an Apple Silicon Mac, with the speed-up over the original in brackets.

**`xy_cut` only**

| Layout | Original (numpy) | Pure Python | Rust | Rust, Python boxes |
| --- | --- | --- | --- | --- |
| 30 random boxes | 0.949 ms | 0.074 ms (13x) | 0.006 ms (150x) | 0.009 ms (105x) |
| 200 random boxes | 2.179 ms | 0.109 ms (20x) | 0.010 ms (212x) | 0.028 ms (78x) |
| 1000 random boxes | 11.019 ms | 0.693 ms (16x) | 0.056 ms (196x) | 0.145 ms (76x) |
| 10000 random boxes | 137.351 ms | 9.251 ms (15x) | 0.775 ms (177x) | 1.679 ms (82x) |
| 3x40 grid | 1.832 ms | 0.227 ms (8x) | 0.019 ms (97x) | 0.030 ms (61x) |

**Building the boxes + `xy_cut`**

| Layout | Original (numpy) | Pure Python | Rust | Rust, Python boxes |
| --- | --- | --- | --- | --- |
| 30 random boxes | 0.991 ms | 0.119 ms (8x) | 0.023 ms (43x) | 0.054 ms (18x) |
| 200 random boxes | 2.460 ms | 0.404 ms (6x) | 0.121 ms (20x) | 0.326 ms (8x) |
| 1000 random boxes | 12.545 ms | 2.171 ms (6x) | 0.624 ms (20x) | 1.616 ms (8x) |
| 10000 random boxes | 151.354 ms | 24.043 ms (6x) | 6.215 ms (24x) | 16.511 ms (9x) |
| 3x40 grid | 2.005 ms | 0.406 ms (5x) | 0.085 ms (24x) | 0.209 ms (10x) |

Once the algorithm runs in Rust, creating the box objects in Python takes most of the time.
