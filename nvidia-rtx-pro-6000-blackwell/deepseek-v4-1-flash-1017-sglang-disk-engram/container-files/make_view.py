"""Build a loader view of the checkpoint without the two Engram lookup tables.

SGLang's safetensors loader maps each whole shard file; shards 47 and 48 are 101.5 GB each
(one Engram embed table + four small Engram tensors). A private mapping larger than RAM +
swap fails with ENOMEM on small hosts. The Engram adapter reads the tables straight from the
original shards, so the loader only needs the small tensors. This script:
  - symlinks every file of SRC into DST, except shards that hold an Engram embed table and
    the index;
  - copies each such shard's other (small) tensors into DST/engram-small-<n>.safetensors,
    reading only their byte ranges with pread;
  - writes DST/model.safetensors.index.json without the embed tables.
Usage: python3 make_view.py SRC DST
"""
import json, os, struct, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
weight_map = index["weight_map"]
is_table = lambda name: ".engram.embed." in name
table_files = sorted({f for n, f in weight_map.items() if is_table(n)})

for name in os.listdir(src):
    if name in table_files or name == "model.safetensors.index.json":
        continue
    link = os.path.join(dst, name)
    if not os.path.lexists(link):
        os.symlink(os.path.join(src, name), link)

new_map = {n: f for n, f in weight_map.items() if f not in table_files}
for i, shard in enumerate(table_files, 1):
    path = os.path.join(src, shard)
    fd = os.open(path, os.O_RDONLY)
    (length,) = struct.unpack("<Q", os.pread(fd, 8, 0))
    header = json.loads(os.pread(fd, length, 8))
    base = 8 + length
    keep = {n: m for n, m in header.items() if n != "__metadata__" and not is_table(n)}
    out_header, blobs, offset = {}, [], 0
    for n in sorted(keep):
        m = keep[n]
        start, end = m["data_offsets"]
        data = os.pread(fd, end - start, base + start)
        assert len(data) == end - start, f"short read of {n}"
        out_header[n] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    os.close(fd)
    raw = json.dumps(out_header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    out_name = f"engram-small-{i}.safetensors"
    with open(os.path.join(dst, out_name), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)
    for n in keep:
        new_map[n] = out_name
    print(f"make_view: {shard}: kept {len(keep)} small tensors ({offset / 2**20:.1f} MiB) in {out_name}", flush=True)

index["weight_map"] = new_map
json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
print(f"make_view: {len(new_map)} tensors in the view, {len(weight_map) - len(new_map)} Engram table tensors left to the adapter", flush=True)
