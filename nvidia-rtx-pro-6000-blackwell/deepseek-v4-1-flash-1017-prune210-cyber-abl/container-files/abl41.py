"""Refusal-direction work for DeepSeek V4.1 Flash on SGLang (onprem.ai).

Three independent parts, each switched on by its own environment variable:

ABL_RUNTIME=1  hooks on every attention output projection (`wo_b`) of the target model and
               of the DSpark drafter, plus control routes on SGLang's HTTP server:
               - capture: with a suite armed, each prefill forward (>= 8 tokens) records the
                 last token's `wo_b` output per target layer. SGLang builds `wo_b` with
                 reduce_results=True when attention is tensor-parallel over all ranks, so the
                 hook sees the all-reduced output and every rank records the same vectors;
                 the per-rank digests prove it.
               - ablation: y <- y - lambda * r_l (r_l^T y) on every `wo_b` output (the
                 drafter's stages use the deepest target layer's direction). The direction
                 and lambda live in GPU buffers that the hook reads, so the ablation is
                 recorded into SGLang's CUDA graphs and follows buffer updates. lambda 0 is
                 exactly the identity (bf16 -> fp32 -> bf16 round trip).
               Routes: POST /abl/control (JSON merged into the control file), POST
               /abl/directions (an .npz with r [layers, hidden] and r_draft [hidden]), GET
               /abl/status, GET /abl/file?name=... (files the ranks wrote).
ABL_BAKE_DIRECTIONS=<npz> with ABL_BAKE_LAMBDA=<float>
               the load-time weight edit: W <- W - lambda * r (r^T W) on each `wo_b`
               (axis 0 is the output axis), re-quantized to FP8 e4m3 with the checkpoint's
               own E8M0 block scales (32 x 32 blocks); a block's scale is raised only where
               the edit overflows e4m3, and each tensor's raised-block count and effective
               lambda are printed. The drafter stages (mtp.N) get r_draft.
"""
import json
import os
import re
import sys
import threading
import time
import traceback

WORK = os.environ.get("ABL_WORK", "/tmp/abl")
CONTROL = os.path.join(WORK, "control.json")
DIRECTIONS = os.path.join(WORK, "directions.npz")
MIN_PREFILL_TOKENS = 8
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_WO_B = re.compile(r"^(?P<pre>.*?)(?P<kind>layers|mtp)\.(?P<idx>\d+)\.attn\.wo_b\.(?P<what>weight|scale)$")


def _log(msg):
    print("[abl41] " + msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------
# runtime hooks (capture and ablation)
# ---------------------------------------------------------------------------------------
class _Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.rank = None
        self.target = []   # (layer index, module, r buffer)
        self.draft = []    # (module, r buffer)
        self.lam = {}      # device -> 0-dim float32 tensor
        self.suite = ""
        self.records = {}  # layer -> list of float16 numpy vectors
        self.dirs_version = None
        self.lam_value = 0.0
        self.thread = None
        self.device = None
        self.errors = []

    def rank_dir(self):
        d = os.path.join(WORK, "rank%s" % self.rank)
        os.makedirs(d, exist_ok=True)
        return d

    def lam_on(self, device):
        import torch
        t = self.lam.get(device)
        if t is None:
            t = torch.zeros((), dtype=torch.float32, device=device)
            self.lam[device] = t
        return t

    def attach(self, model, drafter):
        import torch
        import torch.distributed as dist
        if self.rank is None:
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        n = 0
        for name, mod in model.named_modules():
            if not name.endswith("wo_b"):
                continue
            p = next(mod.parameters(), None)
            device = p.device if p is not None else torch.device("cuda")
            hidden = getattr(mod, "output_size", None) or getattr(mod, "output_size_per_partition", None)
            r = torch.zeros((int(hidden),), dtype=torch.float32, device=device)
            if self.device is None and device.type == "cuda":
                self.device = device
            lam = self.lam_on(device)
            if drafter:
                self.draft.append((mod, r))
                layer = None
            else:
                m = _LAYER.search(name)
                if not m:
                    self.errors.append("no layer index in %s" % name)
                    continue
                layer = int(m.group(1))
                self.target.append((layer, mod, r))
            mod.register_forward_hook(self._make_hook(layer, r, lam))
            n += 1
        _log("rank %s: hooked %d wo_b modules of the %s" % (self.rank, n, "drafter" if drafter else "target"))
        if self.thread is None:
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()

    def _make_hook(self, layer, r, lam):
        import torch

        def hook(_mod, _inputs, output):
            y = output[0] if isinstance(output, tuple) else output
            if layer is not None and self.suite and y.dim() == 2 and y.shape[0] >= MIN_PREFILL_TOKENS \
                    and not torch.cuda.is_current_stream_capturing():
                vec = y[-1].detach().to(torch.float16).cpu().numpy()
                with self.lock:
                    self.records.setdefault(layer, []).append(vec)
            yf = y.float()
            out = (yf - lam * (yf @ r).unsqueeze(-1) * r).to(y.dtype)
            if isinstance(output, tuple):
                return (out,) + tuple(output[1:])
            return out
        return hook

    def _flush(self, suite):
        import hashlib
        import numpy as np
        with self.lock:
            recs, self.records = self.records, {}
        if not recs:
            return
        layers = sorted(recs)
        counts = {l: len(recs[l]) for l in layers}
        if len(set(counts.values())) != 1:
            self.errors.append("suite %s: uneven counts %s" % (suite, counts))
        stacked = {"l%02d" % l: np.stack(recs[l]) for l in layers}
        h = hashlib.sha256()
        for k in sorted(stacked):
            h.update(k.encode())
            h.update(stacked[k].tobytes())
        path = os.path.join(self.rank_dir(), suite + ".npz")
        np.savez(path + ".partial.npz", **stacked)
        os.replace(path + ".partial.npz", path)
        meta = {"suite": suite, "n": counts[layers[0]], "layers": len(layers), "digest": h.hexdigest()}
        with open(os.path.join(self.rank_dir(), suite + ".json"), "w") as f:
            json.dump(meta, f)
        _log("rank %s: saved %s (%d forwards, digest %s)" % (self.rank, suite, meta["n"], meta["digest"][:16]))

    def _apply_directions(self, version):
        import numpy as np
        import torch
        d = np.load(DIRECTIONS)
        r = torch.from_numpy(d["r"].astype("float32"))
        rd = torch.from_numpy(d["r_draft"].astype("float32"))
        for layer, _m, buf in self.target:
            buf.copy_(r[layer].to(buf.device))
        for _m, buf in self.draft:
            buf.copy_(rd.to(buf.device))
        torch.cuda.synchronize(self.device)
        self.dirs_version = version
        _log("rank %s: directions version %s loaded" % (self.rank, version))

    def _apply_lam(self, value):
        import torch
        for t in self.lam.values():
            t.fill_(float(value))
        torch.cuda.synchronize(self.device)
        self.lam_value = float(value)
        _log("rank %s: lambda %s" % (self.rank, value))

    def _control_loop(self):
        import torch
        if self.device is not None:
            torch.cuda.set_device(self.device)  # never touch another rank's GPU
        last_mtime = None
        while True:
            time.sleep(0.2)
            try:
                st = os.stat(CONTROL)
            except FileNotFoundError:
                continue
            if st.st_mtime_ns == last_mtime:
                continue
            last_mtime = st.st_mtime_ns
            try:
                ctl = json.load(open(CONTROL))
                suite = ctl.get("capture") or ""
                if suite != self.suite:
                    old, self.suite = self.suite, ""
                    if old:
                        self._flush(old)
                    with self.lock:
                        self.records = {}
                    self.suite = suite
                if ctl.get("dirs_version") not in (None, self.dirs_version):
                    self._apply_directions(ctl["dirs_version"])
                if float(ctl.get("lambda", 0.0)) != self.lam_value:
                    self._apply_lam(ctl.get("lambda", 0.0))
                with open(os.path.join(self.rank_dir(), "heartbeat.json"), "w") as f:
                    json.dump({"rank": self.rank, "suite": self.suite, "lambda": self.lam_value,
                               "dirs_version": self.dirs_version, "target_hooks": len(self.target),
                               "draft_hooks": len(self.draft), "errors": self.errors[-20:],
                               "time": time.time()}, f)
            except Exception as e:  # report, never kill serving
                self.errors.append("control: %s: %s" % (type(e).__name__, e))
                _log(traceback.format_exc())


RUNTIME = _Runtime()


def add_routes(module):
    """Add the /abl routes to SGLang's FastAPI app (HTTP server process)."""
    from fastapi import Request
    from fastapi.responses import FileResponse, JSONResponse

    app = module.app
    os.makedirs(WORK, exist_ok=True)

    async def control(request: Request):
        body = await request.json()
        ctl = {}
        if os.path.exists(CONTROL):
            ctl = json.load(open(CONTROL))
        ctl.update(body)
        tmp = CONTROL + ".tmp"
        with open(tmp, "w") as f:
            json.dump(ctl, f)
        os.replace(tmp, CONTROL)
        return JSONResponse(ctl)

    async def directions(request: Request):
        data = await request.body()
        tmp = DIRECTIONS + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, DIRECTIONS)
        return JSONResponse({"bytes": len(data)})

    async def status():
        out = {"control": json.load(open(CONTROL)) if os.path.exists(CONTROL) else {}, "ranks": {}, "files": []}
        for root, _d, files in os.walk(WORK):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, WORK)
                if fn == "heartbeat.json":
                    try:
                        out["ranks"][rel.split("/")[0]] = json.load(open(p))
                    except Exception:
                        pass
                else:
                    out["files"].append([rel, os.path.getsize(p)])
        return JSONResponse(out)

    async def file(name: str):
        p = os.path.realpath(os.path.join(WORK, name))
        if not p.startswith(os.path.realpath(WORK) + "/") or not os.path.isfile(p):
            return JSONResponse({"error": "no such file"}, status_code=404)
        return FileResponse(p)

    app.add_api_route("/abl/control", control, methods=["POST"])
    app.add_api_route("/abl/directions", directions, methods=["POST"])
    app.add_api_route("/abl/status", status, methods=["GET"])
    app.add_api_route("/abl/file", file, methods=["GET"])
    _log("routes /abl/* added")


# ---------------------------------------------------------------------------------------
# load-time weight edit
# ---------------------------------------------------------------------------------------
E4M3_MAX = 448.0
BLOCK = 32


def edit_wo_b(weight, scale, r, lam):
    """Return (new fp8 weight, new e8m0 scale, raised blocks, effective lambda)."""
    import torch
    w8 = weight.cpu()
    s8 = scale.cpu()
    if s8.dtype == torch.uint8:                   # raw E8M0 codes
        s = torch.pow(2.0, s8.double() - 127.0)
    else:
        s = s8.float().double()                   # 2**e per 32 x 32 block
    S = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    S_orig = S
    W = w8.double() * S
    rr = r.double()
    rW = rr @ W                                   # [in]
    W2 = W - lam * torch.outer(rr, rW)
    q = W2 / S
    o, i = q.shape
    blockmax = q.abs().reshape(o // BLOCK, BLOCK, i // BLOCK, BLOCK).amax(dim=(1, 3))
    over = blockmax > E4M3_MAX
    raised = int(over.sum())
    if raised:
        k = torch.ceil(torch.log2(blockmax[over] / E4M3_MAX))
        s_new = s.clone()
        s_new[over] = s[over] * torch.pow(2.0, k)
        s = s_new
        S = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        q = W2 / S
    w_new = q.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    # an element whose value did not move keeps its original byte (this also keeps -0.0)
    same = w_new.double() == w8.double() * (S_orig / S)
    w_new = torch.where(same, w8.view(torch.uint8), w_new.view(torch.uint8)).view(torch.float8_e4m3fn)
    if s8.dtype == torch.uint8:
        s_out = (torch.log2(s) + 127.0).round().to(torch.uint8)
        back = torch.pow(2.0, s_out.double() - 127.0)
    else:
        s_out = s.float().to(s8.dtype)
        back = s_out.float().double()
    if not torch.equal(back, s):
        raise ValueError("block scale not representable in %s" % s8.dtype)
    Wq = w_new.double() * S
    eff = 1.0 - float((rr @ Wq) @ rW) / float(rW @ rW)
    return w_new.contiguous(), s_out.contiguous(), raised, eff


def bake_weights(weights):
    """Rewrite the (name, tensor) stream: edit each wo_b weight with its scale."""
    import numpy as np
    import torch
    import base64
    import io
    raw = open(os.environ["ABL_BAKE_DIRECTIONS"], "rb").read()
    if raw[:2] != b"PK":                          # base64 text of the .npz (catalog files are text)
        raw = base64.b64decode(raw)
    d = np.load(io.BytesIO(raw))
    r = torch.from_numpy(d["r"].astype("float64"))
    rd = torch.from_numpy(d["r_draft"].astype("float64"))
    lam = float(os.environ["ABL_BAKE_LAMBDA"])
    pending = {}
    for name, tensor in weights:
        m = _WO_B.match(name)
        if not m:
            yield name, tensor
            continue
        key = (m["pre"], m["kind"], m["idx"])
        pending.setdefault(key, {})[m["what"]] = (name, tensor)
        if len(pending[key]) < 2:
            continue
        parts = pending.pop(key)
        direction = rd if m["kind"] == "mtp" else r[int(m["idx"])]
        (wn, w), (sn, s) = parts["weight"], parts["scale"]
        t0 = time.time()
        w2, s2, raised, eff = edit_wo_b(w, s, direction, lam)
        import hashlib
        _log("bake %s.%s: lambda %.3f -> effective %.4f, %d of %d blocks raised, %.1fs, weight sha256 %s, scale sha256 %s"
             % (m["kind"], m["idx"], lam, eff, raised, s.numel(), time.time() - t0,
                hashlib.sha256(w2.view(torch.uint8).numpy().tobytes()).hexdigest(),
                hashlib.sha256(s2.view(torch.uint8).numpy().tobytes()).hexdigest()))
        yield wn, w2.to(w.device)
        yield sn, s2.to(s.device)
    if pending:
        raise ValueError("wo_b without its weight or scale: %s" % sorted(pending))


# ---------------------------------------------------------------------------------------
# overlay weights: the published edited tensors replace the checkpoint's by name
# ---------------------------------------------------------------------------------------
def _read_overlay(path):
    import struct
    import torch
    dt = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16, "F32": torch.float32}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = bytearray(f.read())
    out = {}
    for name, m in header.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        out[name] = torch.frombuffer(data, dtype=torch.uint8, offset=a, count=b - a).view(dt[m["dtype"]]).reshape(m["shape"])
    return out


def overlay_weights(weights):
    """Rewrite the (name, tensor) stream: each tensor the overlay holds comes from the overlay."""
    overlay = _read_overlay(os.environ["ABL_OVERLAY"])
    seen = set()
    for name, tensor in weights:
        m = _WO_B.match(name)
        key = "%s.%s.attn.wo_b.%s" % (m["kind"], m["idx"], m["what"]) if m else None
        if key in overlay:
            new = overlay[key]
            if tuple(new.shape) != tuple(tensor.shape) or new.dtype != tensor.dtype:
                raise ValueError("overlay %s: %s %s, checkpoint %s %s" % (key, new.dtype, tuple(new.shape), tensor.dtype, tuple(tensor.shape)))
            seen.add(key)
            yield name, new.to(tensor.device)
            continue
        yield name, tensor
    _log("overlay: replaced %d tensors from %s" % (len(seen), os.environ["ABL_OVERLAY"]))
    if os.environ.get("ABL_OVERLAY_EXPECT"):
        want = int(os.environ["ABL_OVERLAY_EXPECT"])
        if len(seen) != want:
            raise ValueError("overlay: replaced %d tensors, expected %d" % (len(seen), want))
