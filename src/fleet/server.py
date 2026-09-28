from __future__ import annotations

import argparse
import os

from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import (
    BaseModel,
    Field,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=32768,
    )
    max_new_tokens: int = Field(
        default=180,
        ge=1,
        le=2048,
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )


class GenerateResponse(BaseModel):
    text: str
    routing: dict


def _make_app(
    bank_dir: str,
    *,
    semantic: bool,
) -> FastAPI:
    router_name = (
        "fleet-semantic-v0.6"
        if semantic
        else "arrow-baseline"
    )

    app = FastAPI(
        title="Fleet API",
        version="0.6.0",
        description=(
            "Heterogeneous-LoRA compiler "
            "and semantic routing API."
        ),
    )

    runtime_box = {}

    @app.on_event(
        "startup"
    )
    def startup():
        if semantic:
            from .semantic_runtime import (
                FleetSemanticRuntime,
            )

            runtime = (
                FleetSemanticRuntime(
                    bank_dir,
                    load_in_4bit=(
                        os.getenv(
                            "FLEET_LOAD_IN_4BIT",
                            "1",
                        )
                        != "0"
                    ),
                )
            )
        else:
            from .runtime import (
                FleetRuntime,
            )

            runtime = (
                FleetRuntime(
                    bank_dir,
                    load_in_4bit=(
                        os.getenv(
                            "FLEET_LOAD_IN_4BIT",
                            "1",
                        )
                        != "0"
                    ),
                )
            )

        runtime_box[
            "runtime"
        ] = runtime

    @app.get(
        "/health"
    )
    def health():
        runtime = (
            runtime_box.get(
                "runtime"
            )
        )

        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Fleet runtime "
                    "not loaded."
                ),
            )

        return {
            "ok": True,
            "router": router_name,
            "base_model": (
                runtime.base_model_name
            ),
            "experts": len(
                runtime.expert_order
            ),
            "router_info": (
                runtime.router_info
            ),
        }

    @app.get(
        "/v1/experts"
    )
    def experts():
        runtime = (
            runtime_box[
                "runtime"
            ]
        )

        return {
            "experts": (
                runtime.experts()
            ),
        }

    @app.post(
        "/v1/generate",
        response_model=(
            GenerateResponse
        ),
    )
    def generate(
        req: GenerateRequest,
    ):
        runtime = (
            runtime_box[
                "runtime"
            ]
        )

        result = (
            runtime.generate(
                req.prompt,
                max_new_tokens=(
                    req.max_new_tokens
                ),
                temperature=(
                    req.temperature
                ),
            )
        )

        return GenerateResponse(
            **result
        )

    return app


def create_semantic_app(
    bank_dir: str | None = None,
) -> FastAPI:
    return _make_app(
        bank_dir
        or os.getenv(
            "FLEET_BANK_DIR",
            "/models/fleet-bank",
        ),
        semantic=True,
    )


def create_arrow_app(
    bank_dir: str | None = None,
) -> FastAPI:
    return _make_app(
        bank_dir
        or os.getenv(
            "FLEET_BANK_DIR",
            "/models/fleet-bank",
        ),
        semantic=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bank",
        default=os.getenv(
            "FLEET_BANK_DIR",
            "/models/fleet-bank",
        ),
    )
    parser.add_argument(
        "--router",
        choices=[
            "semantic",
            "arrow",
        ],
        default="semantic",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    args = parser.parse_args()

    import uvicorn

    app = (
        create_semantic_app(
            args.bank
        )
        if args.router
        == "semantic"
        else create_arrow_app(
            args.bank
        )
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
