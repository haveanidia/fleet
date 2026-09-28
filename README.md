# Fleet

**Fleet** is a research architecture for building **semantic-routed, dynamically extensible language models** from a dense backbone plus independently trained LoRA specialists.

The first standalone checkpoint is **Fleet Semantic 4B**, built on `Qwen/Qwen3-4B-Instruct-2507`.

**Model:** https://huggingface.co/noumenon-labs/Fleet-Semantic-4B

---

## What Fleet is

Fleet turns a normal causal language model into a native expert system:

```text
prompt
  │
  ▼
Qwen3-4B backbone
  │
  ▼
semantic router
  │
  ├── k = 0 → base / NULL path
  ├── k = 1 → one specialist
  └── k = 2 → two specialists
  │
  ▼
next-token generation
```

Routing happens **inside `FleetForCausalLM`**. Fleet is not an external prompt router that chooses between separate models.

Fleet's current research direction can be described as a **Semantic Inference Language Model (SiLM)**: a language model that performs semantic inference over an internal specialist bank while generating.

---

## Current status

`v0.9.5` has passed a native standalone round-trip validation:

- `AutoModelForCausalLM.from_pretrained(...)`
- loads as `FleetForCausalLM`
- full standalone backbone + Fleet state
- generation
- six installed experts
- `.fleet` expert export
- expert uninstall
- dynamic semantic-bank shrink
- expert reinstall
- semantic-bank restoration
- Accelerate CPU offload support
- dynamic APIs under `device_map="auto"`

Validated expert bank:

```text
math_reasoning
medical_reasoning
legal_ops
intent
function_calling
structured
```

The architecture is functional. Task-quality benchmarking versus the base model and source specialists is still ongoing.

---

## Why Fleet exists

LoRA ecosystems contain many independently trained adapters that differ in:

- rank
- target modules
- scaling
- training data
- specialization
- provenance

Most multi-adapter systems assume a fixed bank or require adapters to share a compatible structure.

Fleet explores whether heterogeneous LoRAs can instead be **compiled post-hoc into a shared expert ABI**, then installed into one native causal language model while preserving expert identity and source-layer eligibility.

---

## Core pieces

### `FleetForCausalLM`

A native Transformers architecture based on Qwen3.

### `FleetLinear`

Selected Qwen projections are wrapped once and gain sparse local expert banks:

```text
FleetLinear
├── weight
├── expert_A
├── expert_B
├── expert_scaling
├── expert_ids
└── weight_prototypes
```

Only experts that actually target a projection occupy local slots.

### `FleetRouterState`

The semantic router owns:

```text
semantic_prototypes
semantic_thresholds
null_prototypes
centers
```

Experts use per-layer, per-expert semantic thresholds.

### `FleetAdapter`

A portable `.fleet` package contains one compiled expert:

```text
manifest.json
weights.safetensors
```

It carries:

- canonical LoRA tensors
- source-module eligibility
- weight-space prototypes
- semantic prototypes
- semantic thresholds
- provenance metadata
- semantic-space compatibility fingerprint

---

## Dynamic expert API

```python
model.experts()

model.install("robotics.fleet")

model.uninstall("robotics")

model.export_adapter(
    "structured",
    "structured.fleet",
)
```

Hub-hosted packages can also use:

```python
model.install(
    "hf://org/repo/robotics.fleet"
)
```

A user does **not** choose an expert during ordinary generation:

```python
model.generate(...)
```

Fleet decides internally whether the request should use zero, one, or two specialists.

---

## Quick start: use the released model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "noumenon-labs/Fleet-Semantic-4B"

tokenizer = AutoTokenizer.from_pretrained(repo)

model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": (
            "A store sells 8 pens for $12. "
            "At the same rate, how much do 14 pens cost?"
        ),
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

device = model.get_input_embeddings().weight.device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

new_tokens = output[0, inputs["input_ids"].shape[-1]:]

print(
    tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )
)
```

---

## Install Fleet from source

```bash
git clone https://github.com/noumenon-labs/fleet
cd fleet

pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Build a compiled expert bank

```bash
fleet build \
  --config configs/experts.json \
  --output artifacts/fleet-bank
```

Build semantic prototypes and thresholds:

```bash
fleet semantic-build \
  --bank artifacts/fleet-bank \
  --anchors configs/semantic_anchors.json
```

Validate:

```bash
fleet validate artifacts/fleet-bank
```

---

## Build a standalone Fleet model

```bash
fleet native-build \
  --bank artifacts/fleet-bank \
  --output artifacts/Fleet-Semantic-4B \
  --dtype float16 \
  --max-shard-size 4GB
```

Then run the mandatory native round-trip:

```bash
fleet native-validate \
  --model artifacts/Fleet-Semantic-4B
```

The validator checks:

```text
load FleetForCausalLM
→ list experts
→ export one expert as .fleet
→ uninstall it
→ verify banks shrink
→ reinstall it
→ verify expert order restores
→ generate a probe
```

---

## Heterogeneous LoRA compiler

Fleet compiles adapter updates into a canonical target rank.

For a LoRA update

```text
ΔW = sBA
```

lower/equal-rank source adapters can be preserved through scaling and zero-padding.

For a source rank above the target rank, Fleet reduces the problem before truncation:

```text
B = Qb Rb
Aᵀ = Qa Ra

core = s Rb Raᵀ
```

Then:

```text
core = U S Vᵀ
```

and the top singular components are reconstructed into the target-rank Fleet expert.

This avoids performing a full dense SVD over the original weight update.

See:

- `docs/ARCHITECTURE.md`
- `docs/ADAPTER_FORMAT.md`
- `docs/SEMANTIC_ROUTER.md`
- `docs/RESEARCH_NOTES.md`

---

## Repository layout

```text
fleet/
├── src/fleet/              # compiler, router, runtime, CLI
│   └── native/             # FleetForCausalLM remote-code runtime
├── tests/                  # regression + architecture tests
├── configs/                # expert bank + semantic-anchor configs
├── scripts/                # build, validate, benchmark, Hub helpers
├── docs/                   # architecture and research documentation
├── examples/               # minimal usage examples
├── notebooks/              # development/research notebooks
└── .github/workflows/      # CI
```

---

## Examples

### List experts

```python
print(model.experts())
```

### Export one expert

```python
model.export_adapter(
    "structured",
    "structured.fleet",
)
```

### Remove it

```python
model.uninstall("structured")
```

### Put it back

```python
model.install("structured.fleet")
```

---

## Research questions

Fleet is currently investigating:

1. Can independently trained heterogeneous LoRAs be compiled into a shared expert ABI without losing their specialization?
2. Can a semantic NULL route prevent unnecessary specialist activation?
3. Can expert packages remain installable and removable without rebuilding the backbone?
4. Does dynamic semantic routing outperform the dense base model and individual source adapters on their respective domains?
5. How should expert routing quality be measured beyond raw selection share?

---

## Model weights

Large model weights are intentionally **not stored in this GitHub repository**.

Use the Hugging Face checkpoint instead:

**https://huggingface.co/noumenon-labs/Fleet-Semantic-4B**

---

## Version

Current research release:

```text
Fleet v0.9.5
```

---

## License

See [`LICENSE`](LICENSE) for this repository and [`THIRD_PARTY.md`](THIRD_PARTY.md) for upstream components and adapter provenance.

The released model combines the Qwen backbone with third-party specialist transformations. Review upstream licenses before redistribution or commercial deployment.

---

## Disclaimer

Fleet is a research prototype. Passing architecture and round-trip validation does not by itself establish superior task performance, medical/legal reliability, or production suitability.
