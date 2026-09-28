# Fleet Semantic Router v0.9

Fleet Semantic is a frozen-base routing system. It does not train a router network.

## Dynamic semantic ABI

v0.9 uses a semantic coordinate system that is independent of the installed specialist bank:

- common center: fixed NULL/base anchors only,
- NULL prototype: fixed NULL/base anchors,
- specialist prototype: that expert's anchors,
- activation threshold: calibrated independently per expert and layer.

This matters because a portable FleetAdapter must remain semantically meaningful when installed into a different expert combination.

## State

```text
centers               [layers, hidden]
null_prototypes       [layers, hidden]
semantic_prototypes   [layers, experts, hidden]
semantic_thresholds   [layers, experts]
```

## Semantic-space fingerprint

`semantic_space_sha256` fingerprints the fixed NULL anchor set and semantic window policy. FleetAdapter packages include the fingerprint and `FleetForCausalLM.install()` checks it.

## Gate

For expert `e`:

```text
margin_e = similarity(query, prototype_e)
           - similarity(query, null_prototype)
```

Expert `e` is eligible only if:

```text
margin_e >= threshold_e
```

If no expert passes, routing is `k=0` and the base model handles the token.

If one passes, routing is `k=1`.

If multiple pass, semantic + weight-space ranking chooses one or two experts depending on the top-score margin.
