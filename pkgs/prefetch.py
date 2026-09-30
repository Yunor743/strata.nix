#!/usr/bin/env python3
"""Strata prefetch for NixOS: download, verify and prepare the model files, then emit a run config.

    strata-prefetch --family swift --model IQ2_XS --context 32768 --data-dir /var/lib/strata \
        --emit-config /var/lib/strata/config.json --engine /run/current-system/.../bin/strata

Steps (each skipped when already done):
  1. download shard 1 + the PLE shard from Hugging Face (resumable, size-verified)
  2. build the pack with upstream tools (iq_pack.py: tensors + tokenizer)
  3. fetch and pack the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  4. write prepared.json and, with --emit-config, the engine run config for serve/server.py

No model weights are installed by nix: they live in --data-dir.  A fully prepared
data dir needs no network (the .done marks short-circuit every download).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SHARE = Path(os.environ.get("STRATA_SHARE", "")) if os.environ.get("STRATA_SHARE") else None
if SHARE is None:
    sys.exit("strata-prefetch: STRATA_SHARE is not set (run it through the strata package's wrapper)")

TOOLS = SHARE / "tools"
sys.path.insert(0, str(TOOLS))

HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

FAMILIES = {
    "qwen": {
        "base": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/{q}/",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "tag": "",
        "name": "qwen3.8-flash-next",
        "profile": "expert-profile.bin",
    },
    "swift": {
        "base": "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "tag": "swift-",
        "name": "swift-1.5",
        "profile": "expert-profile.bin",
    },
    "coder": {
        "base": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/{q}/",
        "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
        "tag": "coder-",
        "name": "qwen3.8-flash-next-coder",
        "profile": "expert-profile-coder.bin",
    },
}

QUANTS = {
    "Q2_0": {"arena_gb": 34.0, "ram_gb": 48},
    "IQ2_XS": {"arena_gb": 35.5, "ram_gb": 48},
    "IQ3_XXS": {"arena_gb": 42.9, "ram_gb": 60},
    "IQ3_S": {"arena_gb": 50.3, "ram_gb": 62, "families": ("qwen",)},
    "IQ1_M": {"arena_gb": 23.4, "ram_gb": 32, "families": ("coder",)},
}


def say(msg=""):
    print(msg, flush=True)


def fail(msg, hint=None):
    say(f"[strata-prefetch] ERROR: {msg}")
    if hint:
        say(f"  {hint}")
    sys.exit(1)


def ok(msg):
    say(f"[strata-prefetch] ok: {msg}")


def done(path: Path) -> bool:
    return path.with_name(path.name + ".done").exists()


def mark(path: Path):
    path.with_name(path.name + ".done").write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


def token_headers() -> dict:
    tok = ""
    for env in HF_TOKEN_ENV:
        if os.environ.get(env):
            tok = os.environ[env]
    return {"Authorization": f"Bearer {tok}", "User-Agent": "strata-nix"} if tok else {"User-Agent": "strata-nix"}


def download(url: str, dst: Path, what: str, token=None) -> None:
    """Resumable download with a size check, like upstream setup.py (a .done mark short-circuits)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what} already present")
        return
    headers = token_headers()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=headers)
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check the network and run again (everything else is kept)")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:
        mark(dst)
        ok(f"{what} already present")
        return
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 10:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            say(f"  download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}", "check the network and run again (the download resumes)")
    part.replace(dst)
    mark(dst)
    ok(f"{what} downloaded")


def check_shard(shard: Path) -> None:
    """The shard is present and whole: its header must not name tensors running past the file's end."""
    import gguf_reader as G
    if not shard.exists():
        fail(f"missing {shard}")
    try:
        g = G.GGUFFile(shard)
    except (ValueError, struct.error) as e:
        fail(f"{shard.name} is not a whole GGUF shard ({e})", "delete it and run again")
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    if shard.stat().st_size < need:
        fail(f"{shard.name} is short: {shard.stat().st_size:,} of {need:,} bytes",
             "delete it and run again (or put a whole shard into --gguf-dir)")


def run_tool(script: str, *args, env=None) -> None:
    cmd = [sys.executable, str(TOOLS / script), *(str(a) for a in args)]
    say("  > " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        fail(f"{script} failed (exit {r.returncode})")


def mem_total_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="swift", choices=sorted(FAMILIES))
    ap.add_argument("--model", default="IQ2_XS", choices=sorted(QUANTS))
    ap.add_argument("--context", type=int, default=32768, choices=[8192, 32768, 65536, 131072, 262144])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gguf-dir", help="a directory holding the family's shard files already downloaded")
    ap.add_argument("--low-ram", choices=["on", "off"], help="default: on when the experts would not fit the RAM")
    ap.add_argument("--emit-config")
    ap.add_argument("--engine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--gpu")
    ap.add_argument("--api-key-file")
    ap.add_argument("--hf-token-file")
    ap.add_argument("--sampling-json")
    ap.add_argument("--mcp-json")
    ap.add_argument("--extra-arg", action="append", default=[])
    a = ap.parse_args()

    fam, quant = FAMILIES[a.family], QUANTS[a.model]
    allowed = fam.get("families") or tuple(FAMILIES)
    if a.family not in allowed:
        fail(f"{a.model} is only published for: {', '.join(allowed)} (not {a.family})")

    data = Path(a.data_dir)
    models = data / "models"
    pack = data / "packs" / (fam["tag"].lower() + a.model.lower())
    mtp = data / "mtp"
    rt = mtp / "rt"

    if a.gguf_dir:
        gdir = Path(a.gguf_dir)
        shards = [gdir / fam["file"].format(q=a.model, i=i) for i in (1, 2)]
        ok(f"using GGUF files from {gdir}")
    else:
        base = fam["base"].format(q=a.model)
        shards = [models / fam["file"].format(q=a.model, i=i) for i in (1, 2)]
        token = None
        if a.hf_token_file:
            token = Path(a.hf_token_file).read_text(encoding="utf-8").strip()
        for i, (s, label) in enumerate(zip(shards, (f"{a.family}/{a.model} shard 1", f"{a.family}/{a.model} shard 2"))):
            download(base + s.name, s, label, token=token)

    shard1 = shards[0]
    for s in shards:
        check_shard(s)

    # the PLE table's shard: found by scanning the GGUF headers (shard 2 for qwen/coder, 1 for swift)
    import gguf_reader as G
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in G.GGUFFile(s).tensors)),
               None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")

    if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        say("[strata-prefetch] building the pack (iq_pack.py, a few minutes) ...")
        run_tool("iq_pack.py", "--gguf", shard1, "--out", pack)
        ok(f"pack built: {pack}")
    else:
        ok(f"pack already built: {pack}")

    low_ram = (a.low_ram or "auto")
    if low_ram == "auto":
        low_ram = "on" if mem_total_gb() < quant["ram_gb"] else "off"
    if low_ram == "on":
        if not (pack / "experts.bin").exists():
            say("[strata-prefetch] building the experts.bin file (low-RAM mode, a few minutes) ...")
            run_tool("iq_pack.py", "--gguf", shard1, "--out", pack, "--experts-bin")
        ok("low-RAM mode: the engine maps experts from the pack's experts.bin")

    if not (rt / "experts.bin").exists():
        say("[strata-prefetch] fetching and packing the MTP draft layer (~5 GB, once) ...")
        run_tool("mtp_fetch.py", "fetch", "--out", mtp)
        run_tool("mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf")
        run_tool("mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", rt)
        if not (rt / "draft_vocab.bin").exists():
            shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
        ok(f"MTP ready: {rt}")
    else:
        if not (rt / "draft_vocab.bin").exists():
            shutil.copyfile(SHARE / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
        ok(f"MTP already prepared: {rt}")

    args = [
        "--pack", str(pack),
        "--native", str(shard1),
        "--ple-gguf", str(ple),
        "--expert-profile", str(SHARE / "data" / fam["profile"]),
        "--expert-cache", "auto",
        "--prefill", "auto",
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--mtp", str(rt),
        "--max-context", str(a.context),
    ]
    if a.context >= 65536 and low_ram != "on":
        args += ["--kv-resident", "32768"]
    if low_ram == "on":
        args += ["--mmap-experts"]
    args += a.extra_arg

    prepared = {"args": args, "tokenizer": str(pack / "tokenizer"), "model_name": fam["name"],
                "context": a.context, "low_ram": low_ram == "on"}
    (data / f"prepared-{fam['tag'].lower()}{a.model.lower()}.json").write_text(json.dumps(prepared, indent=1),
                                                                              encoding="utf-8")
    ok(f"prepared: pack={pack} mtp={rt} context={a.context}")

    if a.emit_config:
        if not a.engine:
            fail("--emit-config needs --engine")
        cfg = {"exe": a.engine, "args": args, "tokenizer": prepared["tokenizer"], "model_name": fam["name"],
               "log": str(data / f"engine-{fam['tag'].lower()}{a.model.lower()}.log")}
        if a.host:
            cfg["host"] = a.host
        if a.gpu:
            cfg["gpu"] = a.gpu
        if a.api_key_file:
            cfg["api_key"] = Path(a.api_key_file).read_text(encoding="utf-8").strip()
        if a.sampling_json:
            cfg["sampling"] = json.loads(Path(a.sampling_json).read_text(encoding="utf-8"))
        if a.mcp_json:
            cfg["mcp_servers"] = json.loads(Path(a.mcp_json).read_text(encoding="utf-8"))
        out = Path(a.emit_config)
        out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if a.api_key_file:
            os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
        ok(f"config written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
