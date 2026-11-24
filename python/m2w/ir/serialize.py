from __future__ import annotations

import json
from pathlib import Path

from .graph import Graph


def dumps(graph: Graph, indent: int | None = None) -> str:
    return json.dumps(graph.to_dict(), indent=indent, separators=(",", ":") if indent is None else None)


def dump(graph: Graph, path) -> None:
    Path(path).write_text(dumps(graph), encoding="utf-8", newline="\n")   # LF everywhere: the package index records the sha256 of the bytes


def loads(text: str) -> Graph:
    return Graph.from_dict(json.loads(text))


def load(path) -> Graph:
    return loads(Path(path).read_text(encoding="utf-8"))
