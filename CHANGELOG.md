# Changelog

## 0.9.4

- Fixed `FleetForCausalLM.from_pretrained(..., device_map="auto")` meta-device initialization.
- `_sync_config_from_runtime()` now treats serialized FleetConfig as authoritative while
  `expert_ids` buffers are meta tensors.
- Prevents `NotImplementedError: Cannot copy out of meta tensor; no data!`.
- No checkpoint weight rebuild is required for existing v0.9.x artifacts; patching
  `modeling_fleet.py` is sufficient.


## 0.9.3

- Split runtime `FleetAdapter` from build-time adapter packaging.
- Standalone `fleet_adapter.py` no longer imports `compiler.py`.
- `from_compiled_bank()` and `package_bank_expert()` remain source-side only.
- Added recursive AST-based remote-code import-closure validation.
- Prevents missing dependency failures from appearing after long checkpoint writes.
- Existing v0.9.x checkpoints can be hot-fixed without rebuilding weight shards.


## 0.9.2

- Fixed standalone `FleetForCausalLM` reload under current Transformers.
- Removed parent-relative `from ..fleet_adapter` from remote model code.
- Native runtime now ships a self-contained `fleet_adapter.py` beside `modeling_fleet.py`.
- Native builder preflights remote-code dependencies before checkpoint serialization.
- `native-validate` catches missing runtime source files before invoking AutoModel.
- Existing v0.9/v0.9.1 checkpoints can be hot-fixed without rebuilding weight shards.


## 0.9.0

- Added portable single-file `FleetAdapter` (`.fleet`) packages.
- Added `FleetForCausalLM.install()`.
- Added `FleetForCausalLM.uninstall()`.
- Added `FleetForCausalLM.experts()`.
- Added `FleetForCausalLM.export_adapter()`.
- Added `hf://org/repo/file.fleet` package loading.
- Reworked `FleetLinear` into sparse dynamic local expert banks with global `expert_ids`.
- All supported Qwen projections are wrapped up-front, including projections with empty current banks.
- Added dynamic semantic prototype append/remove.
- Replaced one bank-wide semantic threshold with per-expert per-layer thresholds.
- Changed semantic common-center construction to fixed NULL/base anchors only so semantic coordinates no longer depend on installed specialists.
- Added semantic-space SHA-256 compatibility checks for hot-installed adapters.
- Dynamic expert mutations now update serialized Fleet config so `save_pretrained()` remains reloadable.
- Added `fleet adapter-pack` and `fleet adapter-inspect`.
- Native validation now proves `.fleet` export → uninstall → reinstall → generation before release.

## 0.8.0

- Added native `FleetConfig`, `FleetForCausalLM`, `FleetLinear`, and standalone checkpoint building.
