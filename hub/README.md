---
library_name: peft
base_model: Qwen/Qwen3-4B-Instruct-2507
license: other
tags:
- peft
- lora
- qwen3
- mixture-of-experts
- research
---

# Fleet Compiler v0.6.1

Fleet Compiler is an experimental no-training system for compiling independently
trained LoRA adapters into a shared expert ABI and routing them token-by-token.

This is **not** a newly pretrained foundation model.

See `fleet_bank.json` for the exact expert sources, licenses, retention values,
and routing configuration used by a compiled bank.

Third-party adapter licenses continue to apply.
