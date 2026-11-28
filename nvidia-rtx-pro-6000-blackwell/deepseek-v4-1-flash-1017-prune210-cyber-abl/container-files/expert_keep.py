"""Expert pruning for DeepSeek V4.1 at load time (onprem.ai).

DSV41_EXPERT_KEEP_SET names a keep-set JSON: {"0": [expert ids], ..., "39": [...],
"mtp.0": [...], ...} (the format of tacos4me/DeepSeek-V4.1-Flash-Prune-2xRTX6K's keep sets).
For every listed layer the checkpoint stream that SGLang's load_weights receives is
rewritten:

- routed expert tensors of experts not in the set are dropped; kept experts are renumbered
  0..k-1 in ascending id order;
- the router (gate.weight, gate.bias, gate.bias_vl) keeps the same rows in the same order.

The model's config must say n_routed_experts = k (and dspark_n_routed_experts for the
drafter stages); make_view.py writes that config when DSV41_EXPERT_KEEP_SET is set.

This is exact, not an approximation of masking: V4.1 scores each expert on its own
(scoring_func sqrtsoftplus, no softmax across experts), selects the top k by score plus
bias (noaux_tc, no groups, no hash-routed layers in the 1017 config) and normalizes the
weights over the selected experts only. Removing a router row is the same as giving that
expert a bias of -inf.
"""
import json
import os
import re

_EXPERT = re.compile(r"^(?P<pre>.*?)(?P<kind>layers|mtp)\.(?P<idx>\d+)\.ffn\.experts\.(?P<e>\d+)\.(?P<rest>.+)$")
_GATE = re.compile(r"^(?P<pre>.*?)(?P<kind>layers|mtp)\.(?P<idx>\d+)\.ffn\.gate\.(?P<what>weight|bias|bias_vl)$")


def load_keep_set(path):
    raw = json.load(open(path))
    keep = {}
    for key, ids in raw.items():
        ids = sorted(int(i) for i in ids)
        if len(set(ids)) != len(ids):
            raise ValueError("keep set %s: duplicate expert ids in %r" % (path, key))
        keep[key] = ids
    return keep


def counts(keep):
    """(experts kept per target layer, experts kept per drafter stage); each must be uniform."""
    target = {len(v) for k, v in keep.items() if not k.startswith("mtp.")}
    draft = {len(v) for k, v in keep.items() if k.startswith("mtp.")}
    if len(target) > 1 or len(draft) > 1:
        raise ValueError("keep set sizes differ between layers: %s / %s" % (target, draft))
    return (target.pop() if target else None, draft.pop() if draft else None)


def _key(kind, idx):
    return idx if kind == "layers" else "mtp." + idx


def filter_weights(weights, keep, index_select):
    """Rewrite a (name, tensor) stream. index_select(tensor, ids) returns tensor rows ids."""
    remap = {key: {e: i for i, e in enumerate(ids)} for key, ids in keep.items()}
    for name, tensor in weights:
        m = _EXPERT.match(name)
        if m:
            table = remap.get(_key(m["kind"], m["idx"]))
            if table is None:
                yield name, tensor
                continue
            new = table.get(int(m["e"]))
            if new is None:
                continue
            yield "%s%s.%s.ffn.experts.%d.%s" % (m["pre"], m["kind"], m["idx"], new, m["rest"]), tensor
            continue
        m = _GATE.match(name)
        if m:
            ids = keep.get(_key(m["kind"], m["idx"]))
            if ids is not None:
                if tensor.shape[0] <= ids[-1]:
                    raise ValueError("%s has %d rows, keep set needs id %d" % (name, tensor.shape[0], ids[-1]))
                tensor = index_select(tensor, ids)
        yield name, tensor


def _torch_index_select(tensor, ids):
    import torch
    return tensor.index_select(0, torch.tensor(ids, dtype=torch.long, device=tensor.device)).contiguous()


def wrap_load_weights(cls):
    """Wrap cls.load_weights so its weight stream is pruned by DSV41_EXPERT_KEEP_SET."""
    path = os.environ.get("DSV41_EXPERT_KEEP_SET")
    if not path or getattr(cls.load_weights, "_dsv41_expert_keep", False):
        return
    keep = load_keep_set(path)
    original = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        return original(self, filter_weights(weights, keep, _torch_index_select), *args, **kwargs)

    load_weights._dsv41_expert_keep = True
    cls.load_weights = load_weights
    print("dsv41 expert keep set: %s on %s (%d layers)" % (path, cls.__name__, len(keep)), flush=True)
