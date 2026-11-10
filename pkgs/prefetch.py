#!/usr/bin/env python3
"""Strata prefetch for NixOS: download, verify and prepare the model files, then emit a run config.

    strata-prefetch --family swift --model IQ2_XS --context 32768 --data-dir /var/lib/strata \
        --emit-config /var/lib/strata/config.json --engine /run/current-system/.../bin/strata

Steps (each skipped when already done):
  1. download shard 1 + the PLE shard from Hugging Face (resumable, size-verified)
  2. build the pack with upstream tools (iq_pack.py: tensors + tokenizer)
  3. fetch and pack the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  4. with images on: fetch the picture reader's mmproj (sha256-verified) and write the
     config's vision block (exe, mmproj, model, gpu, ...) for serve/server.py
  5. write prepared.json and, with --emit-config, the engine run config for serve/server.py

No model weights are installed by nix: they live in --data-dir.  A fully prepared
data dir needs no network (the .done marks short-circuit every download).
"""
from __future__ import annotations

import argparse
import hashlib
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
    "abliterated": {
        "base": "https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/resolve/main/{q}/",
        "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-{q}-0000{i}-of-00002.gguf",
        "tag": "abliterated-",
        "name": "swift-1.5-abliterated",
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

# The picture reader's mmproj file per family, at the revision upstream setup.py pins (its HF_REVISIONS and
# ISTA_FILES / SWIFT_FILES tables; the sha256 values are the Hub's LFS pointers).  The coder family shares the
# original's picture reader (upstream: "its shard 2 and vision encoder are the original's files").  The
# abliterated family takes SC117's own picture reader (same name, published beside the text shards, 64 bytes
# shorter than the original's - so a pin of its own, not an alias).
VISION_FILES = {
    "qwen": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
        "file": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "sha256": "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
    },
    "swift": {
        "repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "revision": "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d",
        "file": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
        "sha256": "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60",
    },
    "abliterated": {
        "repo": "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
        "revision": "3db919a964b1f0ae0d1000ba7be90b3e38b1f7a3",
        "file": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "sha256": "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355",
    },
}
VISION_FILES["coder"] = VISION_FILES["qwen"]

# The GPU image encoder takes about 1.4 GB of VRAM; upstream puts it on cards of 24 GB or more
# (strata_control/vision_reader.py's GPU_MIN_VRAM_GB = 23.5) and reads pictures on the CPU below.
VISION_GPU_MIN_VRAM_MIB = 24064
VISION_MAX_TOKENS = {"gpu": 1024, "cpu": 300}   # setup.py's VISION table (the most image tokens a picture becomes)
VISION_THREADS_CAP = 16


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


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_ok(path: Path, want: str) -> bool:
    return sha256_of(path) == want


def download(url: str, dst: Path, what: str, token=None, want=None) -> None:
    """Resumable download with a size check, like upstream setup.py (a .done mark short-circuits).
    `want` is a sha256 the file is checked against when given (the picture reader's mmproj; the
    shards stay size-checked, a 40 GB re-hash at every start would cost minutes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        if want and not sha_ok(dst, want):
            fail(f"{what}: on disk but its sha256 is not the published one", "delete it and run again")
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
        if want and not sha_ok(dst, want):
            fail(f"{what}: already the right size but its sha256 is not the published one",
                 "delete it and run again")
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
    if want and not sha_ok(dst, want):
        dst.unlink()
        fail(f"{dst.name}: the sha256 does not match the published one", "delete it and run again")
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


def nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    for cand in (found, "/run/current-system/sw/bin/nvidia-smi", "/usr/bin/nvidia-smi"):
        if cand and os.path.exists(cand):
            return cand
    return None


def vram_mib(gpu: str | None) -> int | None:
    """The card's VRAM as nvidia-smi reports it, in MiB: the --gpu card when one is named, else the card with
    the most (the engine's own pick).  None when nvidia-smi cannot be asked."""
    smi = nvidia_smi()
    if smi is None:
        return None
    cmd = [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    if gpu:
        # nvidia-smi -i and the engine's CUDA_DEVICE_ORDER=PCI_BUS_ID number the cards alike
        cmd += ["-i", gpu.split(",")[0].strip()]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    values = [int(v) for v in r.stdout.splitlines() if v.strip().isdigit()]
    return max(values) if values else None


def resolve_vision(asked: str, gpu: str | None) -> str:
    """'auto' -> the GPU encoder on a card of 24 GB of VRAM or more, the CPU encoder below (upstream's rule);
    what was asked for otherwise, kept as it is."""
    if asked != "auto":
        mib = vram_mib(gpu)
        if asked == "gpu" and mib is not None and 0 < mib < VISION_GPU_MIN_VRAM_MIB:
            say(f"[strata-prefetch] images: the GPU encoder keeps ~1.4 GB of VRAM that a {mib / 1024:.0f} GB card "
                "would use for experts; kept as asked (--vision cpu reads pictures on the CPU instead)")
        return asked
    mib = vram_mib(gpu)
    if mib is None:
        say("[strata-prefetch] images: no nvidia-smi to ask the card's memory; the CPU encoder it is")
        return "cpu"
    mode = "gpu" if mib >= VISION_GPU_MIN_VRAM_MIB else "cpu"
    say(f"[strata-prefetch] images: the {mode} encoder (the card has {mib / 1024:.1f} GB of VRAM)")
    return mode


def vision_threads() -> int:
    """The CPU encoder's threads: one per physical core (half the logical ones), at most 16 (upstream's rule)."""
    return max(1, min((os.cpu_count() or 8) // 2, VISION_THREADS_CAP))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="swift", choices=sorted(FAMILIES))
    ap.add_argument("--model", default="IQ2_XS", choices=sorted(QUANTS))
    ap.add_argument("--context", type=int, default=32768,
                    choices=[8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288, 786432, 1048576])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gguf-dir", help="a directory holding the family's shard files already downloaded")
    ap.add_argument("--low-ram", choices=["on", "off"], help="default: on when the experts would not fit the RAM")
    ap.add_argument("--vision", choices=["off", "auto", "cpu", "gpu"], default="auto",
                    help="images: auto picks the GPU encoder from 24 GB of VRAM, the CPU one below")
    ap.add_argument("--vision-exe", help="the packaged strata-vision binary (needed when --vision is not off)")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="several requests at once (2-8); written into the config's 'parallel' key, "
                         "which the server turns into the engine's batch slots (it lowers N itself when VRAM is short)")
    ap.add_argument("--emit-config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be downloaded and built, then stop")
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
    allowed = quant.get("families") or tuple(FAMILIES)
    if a.family not in allowed:
        fail(f"{a.model} is only published for: {', '.join(allowed)} (not {a.family})")

    low_ram = (a.low_ram or "auto")
    if low_ram == "auto":
        low_ram = "on" if mem_total_gb() < quant["ram_gb"] else "off"

    # images: which encoder runs, and the family's picture reader file (None: none is published for it)
    mode = resolve_vision(a.vision, a.gpu)
    v = VISION_FILES.get(a.family) if mode != "off" else None
    if mode != "off" and v is None:
        fail(f"no picture reader file is published for the {a.family} family",
             'set vision = "off" (text only), or use a family that has one (qwen, swift, coder)')
    if v is not None and not a.vision_exe:
        fail("--vision needs --vision-exe (the packaged strata-vision binary)")

    data = Path(a.data_dir)
    models = data / "models"
    pack = data / "packs" / (fam["tag"].lower() + a.model.lower())
    mtp = data / "mtp"
    rt = mtp / "rt"

    token = Path(a.hf_token_file).read_text(encoding="utf-8").strip() if a.hf_token_file else None
    if a.gguf_dir:
        gdir = Path(a.gguf_dir)
        shards = [gdir / fam["file"].format(q=a.model, i=i) for i in (1, 2)]
        ok(f"using GGUF files from {gdir}")
    else:
        base = fam["base"].format(q=a.model)
        shards = [models / fam["file"].format(q=a.model, i=i) for i in (1, 2)]

    plan = []
    if not a.gguf_dir:
        for s, label in zip(shards, (f"{a.family}/{a.model} shard 1", f"{a.family}/{a.model} shard 2")):
            plan.append((base + s.name, s, label, None))
    mmproj = None
    if v is not None:
        mmproj = models / v["file"]
        plan.append(("https://huggingface.co/" + v["repo"] + "/resolve/" + v["revision"] + "/" + v["file"],
                     mmproj, f"picture reader {v['file']}", v["sha256"]))

    if a.dry_run:
        say(f"[strata-prefetch] plan for {a.family}/{a.model} (context {a.context}, vision {mode}, "
            f"parallel {a.parallel or 1}, low-ram {low_ram}):")
        for url, dst, what, want in plan:
            say(f"  {what}\n    {url}\n    into {dst}" + (f"\n    sha256 {want}" if want else ""))
        if not plan:
            say("  (nothing to download)")
        say(f"  then: the pack in {pack} with tools/iq_pack.py"
            + (", then the pack's experts.bin file (low-RAM mode)" if low_ram == "on" else "")
            + f", then the MTP draft layer in {rt}")
        return 0

    for url, dst, what, want in plan:
        download(url, dst, what, token=token, want=want)

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
    if a.context > 262144 and not any(x == "--rope-scaling" for x in a.extra_arg):
        # over the trained 262144 the engine wants a scaling that covers it: yarn,
        # factor = final / trained, like upstream setup does.
        # An explicit --rope-scaling among the extra args takes over.
        args += ["--rope-scaling", "yarn", "--rope-scale", f"{a.context / 262144:g}"]
    if mode != "off":
        # the engine flag that turns image input on, and, for a GPU encoder, the VRAM the encoder
        # needs kept free (upstream setup writes the same 700 MiB; an explicit reserve of the
        # user's own in the extra args takes over)
        args.append("--vision")
        if mode == "gpu" and not any(x == "--vram-reserve-mib" for x in a.extra_arg):
            args += ["--vram-reserve-mib", "700"]
    args += a.extra_arg

    vision = None
    if v is not None:
        # the run config's vision block (serve/server.py's Vision: exe --mmproj --model [--gpu]
        # [--threads] [--max-tokens]); "model" is the text model's first shard, opened vocab-only
        vision = {"exe": a.vision_exe, "mmproj": str(mmproj), "model": str(shard1),
                  "gpu": mode == "gpu", "max_tokens": VISION_MAX_TOKENS[mode]}
        if mode == "cpu":
            vision["threads"] = vision_threads()

    prepared = {"args": args, "tokenizer": str(pack / "tokenizer"), "model_name": fam["name"],
                "context": a.context, "low_ram": low_ram == "on", "vision": mode}
    (data / f"prepared-{fam['tag'].lower()}{a.model.lower()}.json").write_text(json.dumps(prepared, indent=1),
                                                                              encoding="utf-8")
    ok(f"prepared: pack={pack} mtp={rt} context={a.context} vision={mode}")

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
        if a.parallel is not None and a.parallel >= 2:
            # the server turns this into the engine's batch slots (--batch N), lowering N itself
            # when the card is short of VRAM (serve/server.py's parallel_args / fit_parallel)
            cfg["parallel"] = a.parallel
        if vision:
            cfg["vision"] = vision
        out = Path(a.emit_config)
        out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if a.api_key_file:
            os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
        ok(f"config written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
