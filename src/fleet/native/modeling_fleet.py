from __future__ import annotations

import threading
from pathlib import Path
from collections.abc import Mapping, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM

from .configuration_fleet import FleetConfig

from .fleet_adapter import FleetAdapter


SUPPORTED_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class _FleetWeightsOverlay(Mapping):
    """Read-through overlay for Accelerate's possibly-lazy offload weights_map."""
    def __init__(self, base: Mapping):
        self.base = base
        self.overrides: dict[str, torch.Tensor] = {}
    def __getitem__(self, key):
        return self.overrides[key] if key in self.overrides else self.base[key]
    def __iter__(self) -> Iterator:
        seen=set()
        for key in self.overrides:
            seen.add(key); yield key
        for key in self.base:
            if key not in seen: yield key
    def __len__(self) -> int:
        try: base_len=len(self.base)
        except Exception: base_len=sum(1 for _ in self.base)
        return base_len + sum(1 for k in self.overrides if k not in self.base)


def _iter_hf_hooks(hook):
    if hook is None:
        return
    yield hook
    nested=getattr(hook,'hooks',None)
    if nested is not None:
        for child in nested:
            yield from _iter_hf_hooks(child)


def _offload_hook(module: nn.Module):
    root=getattr(module,'_hf_hook',None)
    for hook in _iter_hf_hooks(root):
        if getattr(hook,'weights_map',None) is not None:
            return hook
    return None


def _runtime_tensor_cpu(module: nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(tensor,'is_meta',False)):
        return tensor.detach().cpu()
    hook=_offload_hook(module)
    if hook is None:
        raise RuntimeError(f"{type(module).__name__}.{name} is meta but has no Accelerate weights_map.")
    try:
        value=hook.weights_map[name]
    except Exception as exc:
        raise RuntimeError(f"Accelerate weights_map is missing {name!r} for {getattr(module,'semantic_id',type(module).__name__)}.") from exc
    if value is None:
        raise RuntimeError(f"Offloaded tensor {name!r} resolved to None.")
    return value.detach().cpu()


def _ensure_weights_overlay(hook):
    if isinstance(hook.weights_map,_FleetWeightsOverlay):
        return hook.weights_map
    overlay=_FleetWeightsOverlay(hook.weights_map)
    hook.weights_map=overlay
    return overlay


def _replace_runtime_tensor(module: nn.Module, name: str, value: torch.Tensor, *, parameter: bool, requires_grad: bool=False) -> None:
    current=getattr(module,name)
    value_cpu=value.detach().cpu().contiguous()
    if bool(getattr(current,'is_meta',False)):
        hook=_offload_hook(module)
        if hook is None:
            raise RuntimeError(f"Cannot mutate offloaded tensor {name!r}: no Accelerate weights_map.")
        overlay=_ensure_weights_overlay(hook)
        overlay.overrides[name]=value_cpu
        placeholder=torch.empty(value_cpu.shape,dtype=value_cpu.dtype,device='meta')
        if parameter:
            setattr(module,name,nn.Parameter(placeholder,requires_grad=requires_grad))
        else:
            setattr(module,name,placeholder)
        return
    target=value_cpu.to(device=current.device,dtype=current.dtype)
    if parameter:
        setattr(module,name,nn.Parameter(target,requires_grad=requires_grad))
    else:
        setattr(module,name,target)


def _unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=1e-6)


def _pooled_request_direction(x: torch.Tensor, window: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[0] != 1:
        raise RuntimeError("Fleet v0.9 currently supports semantic routing at batch size 1.")
    n = min(int(window), int(x.shape[1]))
    return _unit(_unit(x[0, -n:], dim=-1).mean(dim=0), dim=-1)


def _dynamic_coefficients(
    ranking_scores: torch.Tensor,
    semantic_scores: torch.Tensor,
    semantic_thresholds: torch.Tensor,
    *,
    single_margin: float,
    temperature: float,
) -> torch.Tensor:
    """
    Dynamic k=0/1/2 with per-expert semantic thresholds.

    An expert first has to clear its own semantic threshold. Ranking only decides
    between experts that have already passed that semantic gate.
    """
    tokens, experts = ranking_scores.shape
    coeff = torch.zeros_like(ranking_scores)
    if experts == 0:
        return coeff

    thresholds = semantic_thresholds.reshape(1, experts).to(
        semantic_scores.device, dtype=semantic_scores.dtype
    )
    active = semantic_scores >= thresholds
    active_count = active.sum(dim=1)
    if not torch.any(active_count > 0):
        return coeff

    masked = ranking_scores.masked_fill(~active, float("-inf"))
    rows = torch.arange(tokens, device=ranking_scores.device)

    if experts == 1:
        on = active_count == 1
        coeff[on, 0] = 1.0
        return coeff

    top_values, top_indices = torch.topk(masked, k=2, dim=1)
    only_one = active_count == 1
    at_least_two = active_count >= 2
    margin = top_values[:, 0] - top_values[:, 1]
    k1 = only_one | (at_least_two & (margin >= float(single_margin)))
    k2 = at_least_two & ~k1

    if k1.any():
        coeff[rows[k1], top_indices[k1, 0]] = 1.0
    if k2.any():
        pair = torch.softmax(top_values[k2] / float(temperature), dim=1)
        coeff[rows[k2].unsqueeze(1), top_indices[k2]] = pair
    return coeff


class FleetRouterState(nn.Module):
    """Dynamic global semantic expert bank, registered once on the model."""

    def __init__(self, config: FleetConfig):
        super().__init__()
        self.num_layers = int(config.num_hidden_layers)
        self.hidden_size = int(config.hidden_size)
        n = len(config.fleet_expert_names)
        policy = config.fleet_router_policy or {}

        self.anchor_token_window = int(policy.get("anchor_token_window", 32))
        self.semantic_weight = float(policy.get("semantic_weight", 0.90))
        self.weight_weight = float(policy.get("weight_weight", 0.10))
        self.prompt_prior_weight = float(policy.get("prompt_prior_weight", 0.95))
        self.single_margin = float(policy.get("single_margin", 0.18))
        self.router_temperature = float(policy.get("router_temperature", 0.70))

        self.register_buffer(
            "centers",
            torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float16),
        )
        self.register_buffer(
            "semantic_prototypes",
            torch.zeros(self.num_layers, n, self.hidden_size, dtype=torch.float16),
        )
        self.register_buffer(
            "null_prototypes",
            torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float16),
        )
        self.register_buffer(
            "semantic_thresholds",
            torch.zeros(self.num_layers, n, dtype=torch.float32),
        )
        self._request_prior: dict[int, torch.Tensor] = {}

    @property
    def num_experts(self) -> int:
        return int(self.semantic_prototypes.shape[1])

    def reset_request(self) -> None:
        self._request_prior.clear()

    def append_expert(
        self,
        prototypes: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> int:
        existing_proto=_runtime_tensor_cpu(self,'semantic_prototypes',self.semantic_prototypes)
        existing_thresholds=_runtime_tensor_cpu(self,'semantic_thresholds',self.semantic_thresholds)
        prototypes=prototypes.detach().cpu().to(dtype=existing_proto.dtype)
        thresholds=thresholds.detach().cpu().to(dtype=existing_thresholds.dtype)
        expected=(self.num_layers,self.hidden_size)
        if tuple(prototypes.shape)!=expected: raise ValueError(f"Semantic prototype shape {tuple(prototypes.shape)} != {expected}.")
        if tuple(thresholds.shape)!=(self.num_layers,): raise ValueError(f"Semantic threshold shape {tuple(thresholds.shape)} != {(self.num_layers,)}.")
        index=int(existing_proto.shape[1])
        _replace_runtime_tensor(self,'semantic_prototypes',torch.cat([existing_proto,prototypes[:,None,:]],dim=1),parameter=False)
        _replace_runtime_tensor(self,'semantic_thresholds',torch.cat([existing_thresholds,thresholds[:,None]],dim=1),parameter=False)
        self.reset_request()
        return index

    def remove_expert(self,index:int)->None:
        proto=_runtime_tensor_cpu(self,'semantic_prototypes',self.semantic_prototypes)
        thresholds=_runtime_tensor_cpu(self,'semantic_thresholds',self.semantic_thresholds)
        index=int(index); num_experts=int(proto.shape[1])
        if index<0 or index>=num_experts: raise IndexError(index)
        keep=torch.arange(num_experts)!=index
        _replace_runtime_tensor(self,'semantic_prototypes',proto[:,keep,:].contiguous(),parameter=False)
        _replace_runtime_tensor(self,'semantic_thresholds',thresholds[:,keep].contiguous(),parameter=False)
        self.reset_request()

    def semantic_scores(
        self,
        *,
        layer_id: int,
        q_input: torch.Tensor | None,
        sequence_length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        thresholds = self.semantic_thresholds[layer_id]
        if self.num_experts == 0:
            return thresholds, thresholds

        if q_input is None:
            return self._request_prior.get(layer_id), thresholds

        center = self.centers[layer_id].to(q_input.device, dtype=torch.float32)
        expert_proto = self.semantic_prototypes[layer_id].to(
            q_input.device, dtype=torch.float32
        )
        null_proto = self.null_prototypes[layer_id].to(
            q_input.device, dtype=torch.float32
        )

        if sequence_length > 1 or layer_id not in self._request_prior:
            query = _pooled_request_direction(q_input, self.anchor_token_window)
        else:
            query = _unit(q_input[0, -1], dim=-1)

        query = _unit(query.float() - center, dim=-1)
        expert_proto = _unit(expert_proto, dim=-1)
        null_proto = _unit(null_proto, dim=-1)
        semantic = query @ expert_proto.T - (query @ null_proto)

        if sequence_length > 1 or layer_id not in self._request_prior:
            self._request_prior[layer_id] = semantic.detach()
            return semantic, thresholds

        prior = self._request_prior[layer_id].to(semantic.device)
        a = self.prompt_prior_weight
        return a * prior + (1.0 - a) * semantic, thresholds


class FleetLinear(nn.Module):
    """
    Qwen-compatible projection with a sparse dynamic local expert bank.

    `expert_ids` maps local bank slots to global Fleet semantic expert indices.
    A module therefore stores tensors only for experts that actually target it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        rank: int,
        layer_id: int,
        semantic_id: str,
        router: FleetRouterState,
        expert_ids: list[int] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.layer_id = int(layer_id)
        self.semantic_id = str(semantic_id)

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        ids = torch.tensor(expert_ids or [], dtype=torch.long, device=device)
        n = int(ids.numel())
        self.expert_A = nn.Parameter(
            torch.zeros(n, self.rank, self.in_features, dtype=torch.float16, device=device),
            requires_grad=False,
        )
        self.expert_B = nn.Parameter(
            torch.zeros(n, self.out_features, self.rank, dtype=torch.float16, device=device),
            requires_grad=False,
        )
        self.register_buffer("expert_scaling", torch.ones(n, dtype=torch.float32, device=device))
        self.register_buffer("weight_prototypes", torch.zeros(
            n, self.in_features, dtype=torch.float16, device=device
        ))
        self.register_buffer("expert_ids", ids)
        object.__setattr__(self, "_fleet_router", router)

    @property
    def expert_count(self) -> int:
        return int(self.expert_ids.numel())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        *,
        rank: int,
        layer_id: int,
        semantic_id: str,
        router: FleetRouterState,
        expert_ids: list[int] | None = None,
    ) -> "FleetLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=getattr(linear, "bias", None) is not None,
            rank=rank,
            layer_id=layer_id,
            semantic_id=semantic_id,
            router=router,
            expert_ids=expert_ids,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.weight = linear.weight
        if getattr(linear, "bias", None) is not None:
            module.bias = linear.bias
        return module

    def append_expert(
        self,
        *,
        global_index:int,
        A:torch.Tensor,
        B:torch.Tensor,
        scaling:float|torch.Tensor=1.0,
        weight_prototype:torch.Tensor|None=None,
    )->None:
        if tuple(A.shape)!=(self.rank,self.in_features): raise ValueError(f"{self.semantic_id}: incompatible A shape {tuple(A.shape)}")
        if tuple(B.shape)!=(self.out_features,self.rank): raise ValueError(f"{self.semantic_id}: incompatible B shape {tuple(B.shape)}")
        ids=_runtime_tensor_cpu(self,'expert_ids',self.expert_ids).long()
        if int(global_index) in set(ids.tolist()): raise ValueError(f"{self.semantic_id}: global expert {global_index} already installed.")
        current_A=_runtime_tensor_cpu(self,'expert_A',self.expert_A)
        current_B=_runtime_tensor_cpu(self,'expert_B',self.expert_B)
        current_scale=_runtime_tensor_cpu(self,'expert_scaling',self.expert_scaling)
        current_proto=_runtime_tensor_cpu(self,'weight_prototypes',self.weight_prototypes)
        A=A.detach().cpu().to(dtype=current_A.dtype); B=B.detach().cpu().to(dtype=current_B.dtype)
        if weight_prototype is None: weight_prototype=torch.zeros(self.in_features)
        proto=weight_prototype.detach().cpu().to(dtype=current_proto.dtype)
        scale=torch.as_tensor(scaling,dtype=current_scale.dtype).reshape(1)
        _replace_runtime_tensor(self,'expert_A',torch.cat([current_A,A.unsqueeze(0)],0),parameter=True,requires_grad=False)
        _replace_runtime_tensor(self,'expert_B',torch.cat([current_B,B.unsqueeze(0)],0),parameter=True,requires_grad=False)
        _replace_runtime_tensor(self,'expert_scaling',torch.cat([current_scale,scale],0),parameter=False)
        _replace_runtime_tensor(self,'weight_prototypes',torch.cat([current_proto,proto.unsqueeze(0)],0),parameter=False)
        _replace_runtime_tensor(self,'expert_ids',torch.cat([ids,torch.tensor([int(global_index)],dtype=torch.long)],0),parameter=False)

    def remove_global_expert(self,global_index:int)->bool:
        ids=_runtime_tensor_cpu(self,'expert_ids',self.expert_ids).long()
        if ids.numel()==0: return False
        matches=torch.nonzero(ids==int(global_index),as_tuple=False).flatten(); removed=bool(matches.numel())
        if removed:
            remove_local=int(matches[0].item()); keep=torch.arange(ids.numel())!=remove_local
            A=_runtime_tensor_cpu(self,'expert_A',self.expert_A)
            B=_runtime_tensor_cpu(self,'expert_B',self.expert_B)
            scaling=_runtime_tensor_cpu(self,'expert_scaling',self.expert_scaling)
            proto=_runtime_tensor_cpu(self,'weight_prototypes',self.weight_prototypes)
            _replace_runtime_tensor(self,'expert_A',A[keep].contiguous(),parameter=True,requires_grad=False)
            _replace_runtime_tensor(self,'expert_B',B[keep].contiguous(),parameter=True,requires_grad=False)
            _replace_runtime_tensor(self,'expert_scaling',scaling[keep].contiguous(),parameter=False)
            _replace_runtime_tensor(self,'weight_prototypes',proto[keep].contiguous(),parameter=False)
            ids=ids[keep].contiguous()
        ids=torch.where(ids>int(global_index),ids-1,ids)
        _replace_runtime_tensor(self,'expert_ids',ids,parameter=False)
        return removed

    def local_slot(self,global_index:int)->int|None:
        ids=_runtime_tensor_cpu(self,'expert_ids',self.expert_ids).long()
        match=torch.nonzero(ids==int(global_index),as_tuple=False).flatten()
        return None if match.numel()==0 else int(match[0].item())

    def _weight_scores(self, x_flat: torch.Tensor) -> torch.Tensor:
        if self.expert_count == 0:
            return x_flat.new_zeros((x_flat.shape[0], 0), dtype=torch.float32)
        proto = self.weight_prototypes.to(x_flat.device, dtype=torch.float32)
        score = torch.abs(_unit(x_flat, dim=-1) @ _unit(proto, dim=-1).T)
        if score.shape[1] <= 1:
            return torch.zeros_like(score)
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.05)
        return (score - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.expert_count == 0:
            return base

        router: FleetRouterState = object.__getattribute__(self, "_fleet_router")
        target_name = self.semantic_id.rsplit(".", 1)[-1]
        q_input = x if target_name == "q_proj" else None
        semantic_global, threshold_global = router.semantic_scores(
            layer_id=self.layer_id,
            q_input=q_input,
            sequence_length=int(x.shape[-2]),
        )
        if semantic_global is None or semantic_global.numel() == 0:
            return base

        ids = self.expert_ids.to(semantic_global.device)
        semantic = semantic_global.index_select(0, ids)
        thresholds = threshold_global.to(semantic_global.device).index_select(0, ids)
        flat = x.reshape(-1, x.shape[-1])
        semantic_score = semantic.view(1, -1).expand(flat.shape[0], -1)
        weight_score = self._weight_scores(flat)
        ranking = (
            router.semantic_weight * semantic_score
            + router.weight_weight * weight_score
        )
        coeff = _dynamic_coefficients(
            ranking,
            semantic_score,
            thresholds,
            single_margin=router.single_margin,
            temperature=router.router_temperature,
        )
        if not torch.any(coeff != 0):
            return base

        A = self.expert_A.to(x.device, dtype=x.dtype)
        B = self.expert_B.to(x.device, dtype=x.dtype)
        scaling = self.expert_scaling.to(x.device, dtype=x.dtype)
        low = torch.einsum("tf,erf->ter", flat, A)
        expert_out = torch.einsum("ter,eor->teo", low, B)
        expert_out = expert_out * scaling.view(1, -1, 1)
        delta = torch.einsum("te,teo->to", coeff.to(expert_out.dtype), expert_out)
        return base + delta.view(*x.shape[:-1], self.out_features).to(base.dtype)


class FleetForCausalLM(Qwen3ForCausalLM):
    config_class = FleetConfig

    def __init__(self, config: FleetConfig):
        super().__init__(config)
        object.__setattr__(self, "_fleet_mutation_lock", threading.RLock())
        self._install_native_fleet()

    def _iter_fleet_linears(self):
        for layer_id, layer in enumerate(self.model.layers):
            blocks = {"self_attn": layer.self_attn, "mlp": layer.mlp}
            for block_name, block in blocks.items():
                for target_name in SUPPORTED_TARGETS:
                    module = getattr(block, target_name, None)
                    if isinstance(module, FleetLinear):
                        semantic_id = f"layers.{layer_id}.{block_name}.{target_name}"
                        yield semantic_id, module

    def _fleet_linear_map(self) -> dict[str, FleetLinear]:
        return dict(self._iter_fleet_linears())

    def _install_native_fleet(self) -> None:
        if hasattr(self, "fleet_router"):
            return
        self.fleet_router = FleetRouterState(self.config)
        rank = int(self.config.fleet_rank)
        eligibility = self.config.fleet_eligibility or {}

        # v0.9 wraps every supported Qwen projection, even when its current
        # local expert bank is empty. This is what makes future hot-plugging possible.
        for layer_id, layer in enumerate(self.model.layers):
            blocks = {"self_attn": layer.self_attn, "mlp": layer.mlp}
            for block_name, block in blocks.items():
                for target_name in SUPPORTED_TARGETS:
                    original = getattr(block, target_name, None)
                    if original is None or isinstance(original, FleetLinear):
                        continue
                    semantic_id = f"layers.{layer_id}.{block_name}.{target_name}"
                    mask = list(eligibility.get(semantic_id, []))
                    local_ids = [i for i, active in enumerate(mask) if active]
                    setattr(
                        block,
                        target_name,
                        FleetLinear.from_linear(
                            original,
                            rank=rank,
                            layer_id=layer_id,
                            semantic_id=semantic_id,
                            router=self.fleet_router,
                            expert_ids=local_ids,
                        ),
                    )
        self._sync_config_from_runtime()

    def _sync_config_from_runtime(self) -> None:
        """
        Synchronize serializable config metadata from live FleetLinear banks.

        Transformers/Accelerate may construct the model under torch.device("meta")
        before checkpoint shards are materialized. During that phase buffer values
        such as expert_ids do not exist yet and must not be copied to CPU.

        The serialized FleetConfig is already authoritative during initialization,
        so this method becomes a no-op until every expert_ids buffer is materialized.
        """
        names = list(self.config.fleet_expert_names)
        n = len(names)
        modules = list(self._iter_fleet_linears())

        if any(
            bool(getattr(module.expert_ids, "is_meta", False))
            for _, module in modules
        ):
            return

        eligibility: dict[str, list[bool]] = {}
        targets = []

        for semantic_id, module in modules:
            mask = [False] * n
            ids = _runtime_tensor_cpu(module, "expert_ids", module.expert_ids)
            for gid in ids.tolist():
                if 0 <= int(gid) < n:
                    mask[int(gid)] = True
            eligibility[semantic_id] = mask
            if any(mask):
                targets.append(semantic_id)

        self.config.fleet_eligibility = eligibility
        self.config.fleet_targets = sorted(targets)
        self.config.fleet_version = "0.9.5"
        self.config.fleet_semantic_mode = "per_expert_thresholds"

    def _coerce_adapter(self, adapter) -> FleetAdapter:
        if isinstance(adapter, FleetAdapter):
            result = adapter
        else:
            result = FleetAdapter.load(Path(adapter))
        result.validate()
        return result

    def _validate_adapter_compatibility(self, adapter: FleetAdapter) -> None:
        if adapter.name in self.config.fleet_expert_names:
            raise ValueError(f"Expert {adapter.name!r} is already installed.")
        if adapter.rank != int(self.config.fleet_rank):
            raise ValueError(
                f"FleetAdapter rank {adapter.rank} != model rank {self.config.fleet_rank}."
            )
        expected_base = getattr(self.config, "fleet_base_model", None)
        if expected_base and adapter.base_model != expected_base:
            raise ValueError(
                f"FleetAdapter base {adapter.base_model!r} != model base {expected_base!r}."
            )
        model_space = getattr(self.config, "fleet_semantic_space_sha256", None)
        adapter_space = adapter.metadata.get("semantic_space_sha256")
        if model_space and adapter_space != model_space:
            raise ValueError(
                "FleetAdapter semantic space is missing or incompatible with this model. "
                "Rebuild the adapter with the same v0.9 null-anchor semantic ABI."
            )
        if tuple(adapter.semantic_prototypes.shape) != (
            int(self.config.num_hidden_layers), int(self.config.hidden_size)
        ):
            raise ValueError("FleetAdapter semantic prototype dimensions are incompatible.")
        if tuple(adapter.semantic_thresholds.shape) != (int(self.config.num_hidden_layers),):
            raise ValueError("FleetAdapter semantic threshold dimensions are incompatible.")

        modules = self._fleet_linear_map()
        unknown = sorted(set(adapter.targets) - set(modules))
        if unknown:
            raise ValueError(f"FleetAdapter targets unsupported modules: {unknown[:8]}")
        for semantic_id in adapter.targets:
            module = modules[semantic_id]
            pair = adapter.module_tensors(semantic_id)
            assert pair is not None
            if tuple(pair["A"].shape) != (module.rank, module.in_features):
                raise ValueError(f"{semantic_id}: incompatible A shape {tuple(pair['A'].shape)}")
            if tuple(pair["B"].shape) != (module.out_features, module.rank):
                raise ValueError(f"{semantic_id}: incompatible B shape {tuple(pair['B'].shape)}")

    def install(self, adapter) -> str:
        """Install a FleetAdapter into the live model without rebuilding the checkpoint."""
        adapter = self._coerce_adapter(adapter)
        lock = object.__getattribute__(self, "_fleet_mutation_lock")
        with lock:
            self._validate_adapter_compatibility(adapter)
            global_index = len(self.config.fleet_expert_names)
            modules = self._fleet_linear_map()

            # All shape/target validation happened above; the remaining mutation
            # operations are deterministic and append-only.
            self.fleet_router.append_expert(
                adapter.semantic_prototypes,
                adapter.semantic_thresholds,
            )
            for semantic_id in adapter.targets:
                pair = adapter.module_tensors(semantic_id)
                assert pair is not None
                modules[semantic_id].append_expert(
                    global_index=global_index,
                    A=pair["A"],
                    B=pair["B"],
                    scaling=pair["scaling"],
                    weight_prototype=pair["weight_prototype"],
                )

            self.config.fleet_expert_names = [
                *self.config.fleet_expert_names,
                adapter.name,
            ]
            metadata = list(getattr(self.config, "fleet_expert_metadata", []) or [])
            metadata.append({
                "name": adapter.name,
                "description": adapter.manifest.get("description"),
                "targets": list(adapter.targets),
                **adapter.metadata,
            })
            self.config.fleet_expert_metadata = metadata
            self._sync_config_from_runtime()
            self.fleet_router.reset_request()
            return adapter.name

    def uninstall(self, name: str) -> dict:
        """Remove an expert and compact both local LoRA banks and semantic bank."""
        lock = object.__getattribute__(self, "_fleet_mutation_lock")
        with lock:
            names = list(self.config.fleet_expert_names)
            if name not in names:
                raise KeyError(f"Expert {name!r} is not installed.")
            index = names.index(name)
            metadata = list(getattr(self.config, "fleet_expert_metadata", []) or [])
            removed_metadata = metadata[index] if index < len(metadata) else {"name": name}

            for _, module in self._iter_fleet_linears():
                module.remove_global_expert(index)
            self.fleet_router.remove_expert(index)

            names.pop(index)
            if index < len(metadata):
                metadata.pop(index)
            self.config.fleet_expert_names = names
            self.config.fleet_expert_metadata = metadata
            self._sync_config_from_runtime()
            self.fleet_router.reset_request()
            return removed_metadata

    def experts(self) -> list[dict]:
        """Return the installed expert registry in global semantic-bank order."""
        names = list(self.config.fleet_expert_names)
        metadata = list(getattr(self.config, "fleet_expert_metadata", []) or [])
        counts = [0] * len(names)
        for _, module in self._iter_fleet_linears():
            ids = _runtime_tensor_cpu(module, "expert_ids", module.expert_ids)
            for gid in ids.tolist():
                if 0 <= int(gid) < len(counts):
                    counts[int(gid)] += 1
        result = []
        for i, name in enumerate(names):
            meta = dict(metadata[i]) if i < len(metadata) else {"name": name}
            meta.update({
                "index": i,
                "name": name,
                "active_modules": counts[i],
                "semantic_layers": int(self.fleet_router.num_layers),
            })
            result.append(meta)
        return result

    def export_adapter(self, name: str, output: str | Path) -> Path:
        """Export one currently-installed expert back into a portable .fleet package."""
        names = list(self.config.fleet_expert_names)
        if name not in names:
            raise KeyError(name)
        index = names.index(name)
        metadata = list(getattr(self.config, "fleet_expert_metadata", []) or [])
        meta = dict(metadata[index]) if index < len(metadata) else {}
        tensors: dict[str, torch.Tensor] = {}
        targets = []
        for semantic_id, module in self._iter_fleet_linears():
            slot = module.local_slot(index)
            if slot is None:
                continue
            prefix = f"module.{semantic_id}."
            tensors[prefix + "A"] = _runtime_tensor_cpu(module, "expert_A", module.expert_A)[slot].contiguous()
            tensors[prefix + "B"] = _runtime_tensor_cpu(module, "expert_B", module.expert_B)[slot].contiguous()
            tensors[prefix + "scaling"] = _runtime_tensor_cpu(module, "expert_scaling", module.expert_scaling)[slot].reshape(1).contiguous()
            tensors[prefix + "weight_prototype"] = _runtime_tensor_cpu(module, "weight_prototypes", module.weight_prototypes)[slot].contiguous()
            targets.append(semantic_id)
        tensors["semantic.prototypes"] = _runtime_tensor_cpu(self.fleet_router, "semantic_prototypes", self.fleet_router.semantic_prototypes)[:, index, :].contiguous()
        tensors["semantic.thresholds"] = _runtime_tensor_cpu(self.fleet_router, "semantic_thresholds", self.fleet_router.semantic_thresholds)[:, index].contiguous()
        manifest = {
            "format": "fleet-adapter",
            "version": "0.9.5",
            "name": name,
            "description": meta.get("description", name),
            "base_model": getattr(self.config, "fleet_base_model", ""),
            "rank": int(self.config.fleet_rank),
            "targets": targets,
            "num_layers": int(self.config.num_hidden_layers),
            "hidden_size": int(self.config.hidden_size),
            "metadata": {
                **{k: v for k, v in meta.items() if k not in {"name", "targets"}},
                "semantic_space_sha256": getattr(
                    self.config, "fleet_semantic_space_sha256", None
                ),
            },
        }
        return FleetAdapter(manifest=manifest, tensors=tensors).save(output)

    def reset_fleet_router(self) -> None:
        self.fleet_router.reset_request()

    def forward(self, *args, **kwargs):
        lock = object.__getattribute__(self, "_fleet_mutation_lock")
        with lock:
            return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        lock = object.__getattribute__(self, "_fleet_mutation_lock")
        with lock:
            self.reset_fleet_router()
            return super().generate(*args, **kwargs)
