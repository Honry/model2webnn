"""Executes the COMPILED image tower graphs of a Qwen3.5 package (HEAD / BLOCK x depth / MERGER of one grid preset) with the numpy interpreter and the packed weights, following the runtime protocol:
the executable specification of what the browser runtime does for an image (docs/models/qwen35-vision.md). `Qwen35PackageReader.vision_hf_state_dict()` gives the dequantized weights under the
transformers names, to feed `Qwen3_5VisionModel` as the independent reference (qwen35_hf.Qwen35VisionHF).

patches [N, patch_vector] (protocol.patchify of a preprocessed image, float16 at the graph boundary) -> HEAD -> hidden_states [1, N, hidden] -> BLOCK i with the weights of layer i, i = 0 .. depth - 1
-> MERGER -> image_embeds [N / 4, out_hidden] float16: the rows that replace the <|image_pad|> embeddings of the prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..frontends.hf_llm.builders import qwen35_vision as QV
from ..ir import serialize
from ..ir.interpreter import run
from .qwen35_runner import Qwen35PackageReader


class Qwen35VisionRunner:
    def __init__(self, package, spec: ModelSpec, compute_type: str = ""):
        """compute_type: build the graphs here with this type instead of reading the package's (the float16 interior is what the NPU does with float32 graphs: it shows the overflows)"""
        package = Path(package)
        self.compute_type = compute_type
        self.spec, self.plan = spec, json.loads((package / "plan.json").read_text(encoding="utf-8"))["vision"]
        self.reader = Qwen35PackageReader(spec, package / "weights")
        self.package = package
        self.presets = {tuple(p["grid"]): p for p in self.plan["presets"]}
        self.graphs: dict = {}
        self.trace: list = []

    @property
    def grids(self) -> list:
        return list(self.presets)

    def _graphs(self, grid) -> dict:
        if grid not in self.graphs:
            if self.compute_type:
                w = self.plan["weights"]
                self.graphs[grid] = {"head": QV.build_head_graph(self.spec, grid, w, self.compute_type), "block": QV.build_block_graph(self.spec, grid, w, self.compute_type),
                                     "merger": QV.build_merger_graph(self.spec, grid, w, self.compute_type)}
            else:
                self.graphs[grid] = {k: serialize.load(self.package / path) for k, path in self.presets[grid]["graphs"].items()}
        return self.graphs[grid]

    def _weights(self, shard: dict, names) -> dict:
        return {n: self.reader.raw(shard, n) for n in names}

    def encode(self, patches: np.ndarray, grid) -> np.ndarray:
        """patches [N, patch_vector] -> image_embeds [N / merge^2, out_hidden] float16. `self.trace` = the hidden state after HEAD and after every block (float16 [N, hidden])."""
        grid = tuple(grid)
        if grid not in self.presets:
            raise ValueError(f"the package has no preset {grid[0]}x{grid[1]} (it has {', '.join(f'{g[0]}x{g[1]}' for g in self.presets)})")
        g, names = self._graphs(grid), self.plan["tensors"]
        tables = self._weights(self.reader.vision_shard("vision_grid", grid=grid), names["tables"])
        head = self.reader.vision_shard("vision_head")
        x = run(g["head"], {"patches": np.asarray(patches, np.float16), **self._weights(head, names["head"])}, externals=tables)["hidden_states"]
        self.trace = [x[0]]
        for i in range(self.plan["depth"]):
            x = run(g["block"], {"hidden_states": x, **self._weights(self.reader.vision_shard("vision_block", layer=i), names["block"])}, externals=tables)["next_hidden_states"]
            self.trace.append(x[0])
        merger = self.reader.vision_shard("vision_merger")
        return run(g["merger"], {"hidden_states": x, **self._weights(merger, names["merger"])}, externals=tables)["image_embeds"][0]
