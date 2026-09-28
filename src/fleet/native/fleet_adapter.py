from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import tempfile
import zipfile

import torch
from safetensors.torch import load_file, save_file



FLEET_ADAPTER_FORMAT = "fleet-adapter"
FLEET_ADAPTER_VERSION = "0.9.3"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _top_right_proto(A: torch.Tensor, B: torch.Tensor, *, iters: int = 15) -> torch.Tensor:
    """Dominant right singular direction of B@A without materializing B@A."""
    A = A.float()
    B = B.float()
    c = B.T @ B
    v = torch.ones(A.shape[1], dtype=torch.float32)
    v = v / v.norm().clamp_min(1e-8)
    for _ in range(int(iters)):
        w = A.T @ (c @ (A @ v))
        n = w.norm()
        if float(n.item()) <= 1e-8:
            return torch.zeros_like(v)
        v = w / n
    return v


@dataclass
class FleetAdapter:
    """
    Portable, already-compiled Fleet expert.

    A package is a single `.fleet` ZIP containing:
      manifest.json
      weights.safetensors

    The weights contain canonical per-module A/B matrices, weight prototypes,
    and a per-layer semantic prototype + activation threshold.
    """

    manifest: dict
    tensors: dict[str, torch.Tensor]

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def rank(self) -> int:
        return int(self.manifest["rank"])

    @property
    def base_model(self) -> str:
        return str(self.manifest["base_model"])

    @property
    def targets(self) -> list[str]:
        return list(self.manifest.get("targets", []))

    @property
    def metadata(self) -> dict:
        return dict(self.manifest.get("metadata", {}))

    @property
    def semantic_prototypes(self) -> torch.Tensor:
        return self.tensors["semantic.prototypes"]

    @property
    def semantic_thresholds(self) -> torch.Tensor:
        return self.tensors["semantic.thresholds"]

    def module_tensors(self, semantic_id: str) -> dict[str, torch.Tensor] | None:
        prefix = f"module.{semantic_id}."
        a = self.tensors.get(prefix + "A")
        b = self.tensors.get(prefix + "B")
        if a is None or b is None:
            return None
        proto = self.tensors.get(prefix + "weight_prototype")
        scale = self.tensors.get(prefix + "scaling")
        return {
            "A": a,
            "B": b,
            "weight_prototype": (
                proto if proto is not None else _top_right_proto(a, b)
            ),
            "scaling": (
                scale.reshape(()) if scale is not None else torch.tensor(1.0)
            ),
        }

    def validate(self) -> None:
        if self.manifest.get("format") != FLEET_ADAPTER_FORMAT:
            raise ValueError("Not a FleetAdapter package.")
        if not self.name:
            raise ValueError("FleetAdapter name is empty.")
        if self.rank <= 0:
            raise ValueError("FleetAdapter rank must be positive.")
        if "semantic.prototypes" not in self.tensors:
            raise ValueError("FleetAdapter is missing semantic.prototypes.")
        if "semantic.thresholds" not in self.tensors:
            raise ValueError("FleetAdapter is missing semantic.thresholds.")
        p = self.semantic_prototypes
        t = self.semantic_thresholds
        if p.ndim != 2:
            raise ValueError("semantic.prototypes must be [layers, hidden].")
        if t.ndim != 1 or t.shape[0] != p.shape[0]:
            raise ValueError("semantic.thresholds must be [layers].")
        for semantic_id in self.targets:
            pair = self.module_tensors(semantic_id)
            if pair is None:
                raise ValueError(f"Missing A/B tensors for active target {semantic_id}.")
            if pair["A"].ndim != 2 or pair["B"].ndim != 2:
                raise ValueError(f"Invalid A/B rank for {semantic_id}.")
            if pair["A"].shape[0] != self.rank or pair["B"].shape[1] != self.rank:
                raise ValueError(f"Canonical rank mismatch for {semantic_id}.")

    def save(self, output: str | Path) -> Path:
        self.validate()
        output = Path(output)
        if output.suffix != ".fleet":
            output = output.with_suffix(".fleet")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fleet_adapter_save_") as td:
            root = Path(td)
            _write_json(root / "manifest.json", self.manifest)
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in self.tensors.items()},
                str(root / "weights.safetensors"),
                metadata={
                    "format": FLEET_ADAPTER_FORMAT,
                    "version": FLEET_ADAPTER_VERSION,
                    "name": self.name,
                },
            )
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(root / "manifest.json", "manifest.json")
                zf.write(root / "weights.safetensors", "weights.safetensors")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FleetAdapter":
        raw = str(path)
        if raw.startswith("hf://"):
            # hf://org/repo/path/to/expert.fleet
            from huggingface_hub import hf_hub_download
            spec = raw[len("hf://"):].strip("/")
            parts = spec.split("/")
            if len(parts) < 3:
                raise ValueError(
                    "HF FleetAdapter URI must be hf://org/repo/path/to/file.fleet"
                )
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        else:
            path = Path(path)
        if path.is_dir():
            manifest = _read_json(path / "manifest.json")
            tensors = load_file(str(path / "weights.safetensors"), device="cpu")
            obj = cls(manifest=manifest, tensors=tensors)
            obj.validate()
            return obj
        if not path.exists():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory(prefix="fleet_adapter_load_") as td:
            root = Path(td)
            with zipfile.ZipFile(path, "r") as zf:
                names = set(zf.namelist())
                required = {"manifest.json", "weights.safetensors"}
                if not required.issubset(names):
                    raise ValueError(f"Invalid FleetAdapter package; missing {required - names}")
                zf.extract("manifest.json", root)
                zf.extract("weights.safetensors", root)
            manifest = _read_json(root / "manifest.json")
            tensors = load_file(str(root / "weights.safetensors"), device="cpu")
        obj = cls(manifest=manifest, tensors=tensors)
        obj.validate()
        return obj
