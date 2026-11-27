"""Build the checkpoint view, then write a config.json with the pruned expert counts.

Runs make_view.py of the image first (links every weight file except the Engram tables),
then replaces the config.json link with a copy whose text_config says
n_routed_experts = experts kept per layer and dspark_n_routed_experts = experts kept per
drafter stage, both taken from the keep set DSV41_EXPERT_KEEP_SET. dsv41-start runs
make_view.py again later; it keeps existing entries, so this config stays.
Usage: make_view_prune.py SRC DST
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_keep import counts, load_keep_set  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
subprocess.run([sys.executable, "/opt/dsv41/bin/make_view.py", src, dst], check=True)
target, draft = counts(load_keep_set(os.environ["DSV41_EXPERT_KEEP_SET"]))
config = json.load(open(os.path.join(src, "config.json")))
text = config["text_config"]
print("make_view_prune: n_routed_experts %s -> %s, dspark_n_routed_experts %s -> %s" % (
    text["n_routed_experts"], target, text.get("dspark_n_routed_experts"), draft), flush=True)
text["n_routed_experts"] = target
if draft is not None:
    text["dspark_n_routed_experts"] = draft
link = os.path.join(dst, "config.json")
if os.path.lexists(link):
    os.unlink(link)
with open(link, "w") as f:
    json.dump(config, f, indent=2)
