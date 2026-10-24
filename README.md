# recursive_xy_cut

A Python implementation of the **recursive XY cut** algorithm for determining the reading order of
bounding boxes in a document layout.

The algorithm projects the bounding boxes on a page onto the horizontal and vertical axes, finds the
widest gap (whitespace) between the projected boxes, splits the page at that gap and recurses on each
part, alternating between vertical and horizontal cuts. Reading the leaves of the resulting tree
left-to-right gives a reading order for the boxes, for example column by column in a multi-column
layout.

The implementation is based on [this paper on the XY cut](http://haralick.org/conferences/71280952.pdf).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for package and environment management.

```bash
git clone git@github.com:mihailyanchev/recursive_xy_cut.git
cd recursive_xy_cut
uv sync
```

## Usage

```python
from recursive_xy_cut import Point, Rectangle, xy_cut

boxes = [
    Rectangle(top_left=Point(x=110, y=10), bottom_right=Point(x=190, y=40), text="right"),
    Rectangle(top_left=Point(x=10, y=10), bottom_right=Point(x=90, y=40), text="left top"),
    Rectangle(top_left=Point(x=10, y=60), bottom_right=Point(x=90, y=90), text="left bottom"),
]

ordered = xy_cut(boxes)
print([box.text for box in ordered])  # ['left top', 'left bottom', 'right']
```

## Rust implementation

[`rust/`](rust/) contains `recursive_xy_cut_rs`, a Rust implementation with the same API, packaged
as a Python extension. It gives the same reading order and is about 10x faster than this package.
See [its README](rust/README.md) for usage and benchmarks.

## Development

| Tool | Purpose | Command |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Dependency and environment management | `uv sync` |
| [ruff](https://docs.astral.sh/ruff/) | Linting and formatting | `uv run ruff check .` / `uv run ruff format .` |
| [mypy](https://mypy.readthedocs.io/) | Static type checking (strict) | `uv run mypy` |
| [bandit](https://bandit.readthedocs.io/) | Security checks | `uv run bandit -c pyproject.toml -r src` |
| [pytest](https://docs.pytest.org/) | Tests and coverage | `uv run pytest --cov` |
| [pre-commit](https://pre-commit.com/) | Runs all of the above on each commit | `uv run pre-commit install` |

Tool configuration lives in [`pyproject.toml`](pyproject.toml) and
[`.pre-commit-config.yaml`](.pre-commit-config.yaml). The same checks run in CI on every pull request.

To set up the git hooks once after cloning:

```bash
uv run pre-commit install
uv run pre-commit run --all-files  # optional: run against the whole repo
```

## Project structure

```
├── src/recursive_xy_cut/   # the package
│   ├── __init__.py
│   ├── py.typed            # marks the package as typed (PEP 561)
│   └── xy_cut.py           # algorithm implementation
├── tests/                  # pytest test suite
├── rust/                   # Rust implementation (recursive_xy_cut_rs)
├── pyproject.toml          # project metadata and tool configuration
└── uv.lock                 # locked dependency versions
```

## Roadmap

- [ ] Evaluation of reading-order accuracy against a benchmark dataset
- [ ] Cut at all gaps in the same direction per iteration (rather than only the widest one)

## License

[Apache 2.0](LICENSE)
