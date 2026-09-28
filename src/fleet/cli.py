from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import build_bank
from .validation import validate_bank


def build_cmd(args):
    result = build_bank(
        args.config,
        args.output,
    )

    print(
        json.dumps(
            {
                "output": str(
                    Path(
                        args.output
                    ).resolve()
                ),
                "experts": [
                    x["name"]
                    for x in result[
                        "experts"
                    ]
                ],
                "dropped_for_retention": (
                    result[
                        "dropped_for_retention"
                    ]
                ),
            },
            indent=2,
        )
    )


def semantic_build_cmd(args):
    from .semantic_build import (
        build_semantic_router,
    )

    meta = (
        build_semantic_router(
            bank_dir=args.bank,
            anchors_path=args.anchors,
            load_in_4bit=(
                not args.no_4bit
            ),
        )
    )

    print(
        json.dumps(
            meta,
            indent=2,
        )
    )


def validate_cmd(args):
    result = validate_bank(
        args.bank
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if not result["ok"]:
        raise SystemExit(1)



def benchmark_cmd(args):
    from .benchmark import run_benchmark

    result = run_benchmark(
        args.bank,
        output_path=args.output,
        max_failures=args.max_failures,
    )

    print(json.dumps(result, indent=2))

    if not result["passed"]:
        raise SystemExit(1)


def release_cmd(args):
    from .release import release_to_hub

    result = release_to_hub(
        bank_dir=args.bank,
        repo_id=args.repo,
        release_dir=args.release_dir,
        private=args.private,
        run_release_benchmark=(not args.skip_benchmark),
        max_benchmark_failures=args.max_failures,
        num_workers=args.num_workers,
    )

    print(json.dumps(result, indent=2))


def export_model_cmd(args):
    from .export_model import export_single_model

    result = export_single_model(
        bank_dir=args.bank,
        output_dir=args.output,
    )

    print(json.dumps(result, indent=2))


def native_build_cmd(args):
    from .native_build import build_native_checkpoint

    result = build_native_checkpoint(
        bank_dir=args.bank,
        output_dir=args.output,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
    )

    print(json.dumps(result, indent=2))


def native_validate_cmd(args):
    from .native_validate import validate_native_checkpoint

    result = validate_native_checkpoint(
        args.model,
    )

    print(json.dumps(result, indent=2))

    if not result["ok"]:
        raise SystemExit(1)


def native_release_cmd(args):
    from .native_release import release_native

    result = release_native(
        bank_dir=args.bank,
        output_dir=args.output,
        repo_id=args.repo,
        private=args.private,
        max_shard_size=args.max_shard_size,
        num_workers=args.num_workers,
    )

    print(json.dumps(result, indent=2))


def adapter_pack_cmd(args):
    from .fleet_adapter import package_bank_expert

    expert = str(args.expert or "").strip()
    if not expert:
        bank_path = Path(args.bank) / "fleet_bank.json"
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        available = [x["name"] for x in bank.get("experts", [])]
        raise SystemExit(
            "ERROR: --expert resolved to an empty string. "
            "Available experts: " + ", ".join(available)
        )

    path = package_bank_expert(
        args.bank,
        expert,
        args.output,
    )
    print(json.dumps({"output": str(path), "expert": expert}, indent=2))


def adapter_list_cmd(args):
    bank_path = Path(args.bank) / "fleet_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(bank.get("experts", [])):
        rows.append({
            "index": i,
            "name": item.get("name"),
            "repo": item.get("repo"),
            "license": item.get("license"),
            "retention": item.get("retention"),
        })
    print(json.dumps(rows, indent=2))


def adapter_inspect_cmd(args):
    from .fleet_adapter import FleetAdapter

    adapter = FleetAdapter.load(args.adapter)
    print(json.dumps({
        "name": adapter.name,
        "base_model": adapter.base_model,
        "rank": adapter.rank,
        "targets": adapter.targets,
        "semantic_shape": list(adapter.semantic_prototypes.shape),
        "metadata": adapter.metadata,
    }, indent=2))

def serve_cmd(args):
    import uvicorn

    if args.router == "semantic":
        from .server import (
            create_semantic_app,
        )

        app = create_semantic_app(
            args.bank
        )
    else:
        from .server import (
            create_arrow_app,
        )

        app = create_arrow_app(
            args.bank
        )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="fleet",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build = sub.add_parser(
        "build",
        help=(
            "Compile a heterogeneous "
            "Fleet expert bank."
        ),
    )

    build.add_argument(
        "--config",
        required=True,
    )
    build.add_argument(
        "--output",
        required=True,
    )
    build.set_defaults(
        func=build_cmd
    )

    semantic = sub.add_parser(
        "semantic-build",
        help=(
            "Build frozen-Qwen semantic "
            "routing prototypes."
        ),
    )
    semantic.add_argument(
        "--bank",
        required=True,
    )
    semantic.add_argument(
        "--anchors",
        default=(
            "configs/"
            "semantic_anchors.json"
        ),
    )
    semantic.add_argument(
        "--no-4bit",
        action="store_true",
    )
    semantic.set_defaults(
        func=semantic_build_cmd
    )

    validate = sub.add_parser(
        "validate",
        help=(
            "Validate a compiled bank."
        ),
    )
    validate.add_argument(
        "bank",
    )
    validate.set_defaults(
        func=validate_cmd
    )

    benchmark = sub.add_parser(
        "benchmark",
        help="Run the release routing benchmark.",
    )
    benchmark.add_argument(
        "--bank",
        required=True,
    )
    benchmark.add_argument(
        "--output",
        default="artifacts/fleet-bank/benchmark_results.json",
    )
    benchmark.add_argument(
        "--max-failures",
        type=int,
        default=0,
    )
    benchmark.set_defaults(
        func=benchmark_cmd
    )

    release = sub.add_parser(
        "release",
        help=(
            "Validate, benchmark, build a clean Hub artifact, "
            "and automatically upload it."
        ),
    )
    release.add_argument(
        "--bank",
        required=True,
    )
    release.add_argument(
        "--repo",
        required=True,
    )
    release.add_argument(
        "--release-dir",
        default=None,
    )
    release.add_argument(
        "--private",
        action="store_true",
    )
    release.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Use an existing benchmark_results.json instead of rerunning probes.",
    )
    release.add_argument(
        "--max-failures",
        type=int,
        default=0,
    )
    release.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    release.set_defaults(
        func=release_cmd
    )

    export_model = sub.add_parser(
        "export-model",
        help=(
            "Consolidate the compiled expert bank and semantic router "
            "into one Fleet model artifact."
        ),
    )
    export_model.add_argument(
        "--bank",
        required=True,
    )
    export_model.add_argument(
        "--output",
        required=True,
    )
    export_model.set_defaults(
        func=export_model_cmd
    )

    adapter_list = sub.add_parser(
        "adapter-list",
        help="List experts available in a compiled Fleet bank.",
    )
    adapter_list.add_argument("--bank", required=True)
    adapter_list.set_defaults(func=adapter_list_cmd)

    adapter_pack = sub.add_parser(
        "adapter-pack",
        help="Package one compiled bank expert as a portable .fleet FleetAdapter.",
    )
    adapter_pack.add_argument("--bank", required=True)
    adapter_pack.add_argument("--expert", required=True)
    adapter_pack.add_argument("--output", required=True)
    adapter_pack.set_defaults(func=adapter_pack_cmd)

    adapter_inspect = sub.add_parser(
        "adapter-inspect",
        help="Inspect a portable FleetAdapter package.",
    )
    adapter_inspect.add_argument("adapter")
    adapter_inspect.set_defaults(func=adapter_inspect_cmd)

    native_build = sub.add_parser(
        "native-build",
        help=(
            "Build a standalone FleetForCausalLM checkpoint "
            "containing Qwen backbone + Fleet experts + router."
        ),
    )
    native_build.add_argument(
        "--bank",
        required=True,
    )
    native_build.add_argument(
        "--output",
        required=True,
    )
    native_build.add_argument(
        "--dtype",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
        default="float16",
    )
    native_build.add_argument(
        "--max-shard-size",
        default="4GB",
    )
    native_build.set_defaults(
        func=native_build_cmd
    )

    native_validate = sub.add_parser(
        "native-validate",
        help=(
            "Round-trip load a standalone Fleet checkpoint "
            "through AutoModelForCausalLM."
        ),
    )
    native_validate.add_argument(
        "--model",
        required=True,
    )
    native_validate.set_defaults(
        func=native_validate_cmd
    )

    native_release = sub.add_parser(
        "native-release",
        help=(
            "Build, round-trip validate, and upload "
            "the standalone FleetForCausalLM checkpoint."
        ),
    )
    native_release.add_argument(
        "--bank",
        required=True,
    )
    native_release.add_argument(
        "--output",
        required=True,
    )
    native_release.add_argument(
        "--repo",
        required=True,
    )
    native_release.add_argument(
        "--private",
        action="store_true",
    )
    native_release.add_argument(
        "--max-shard-size",
        default="4GB",
    )
    native_release.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    native_release.set_defaults(
        func=native_release_cmd
    )

    serve = sub.add_parser(
        "serve",
        help=(
            "Serve Fleet over HTTP."
        ),
    )
    serve.add_argument(
        "--bank",
        required=True,
    )
    serve.add_argument(
        "--router",
        choices=[
            "semantic",
            "arrow",
        ],
        default="semantic",
    )
    serve.add_argument(
        "--host",
        default="0.0.0.0",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    serve.set_defaults(
        func=serve_cmd
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
