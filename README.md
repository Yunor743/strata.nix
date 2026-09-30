# strata.nix

Run [Strata](https://github.com/Niko1221/Strata) — the engine that serves
**Qwen3.8-Flash-Next** (a 125B-parameter MoE, ~6B active) on a single NVIDIA GPU —
as a native NixOS service, with an OpenAI- and Anthropic-compatible API on localhost.

```
┌─────────────┐   OpenAI/Anthropic    ┌───────────────┐   stdin/stdout   ┌────────────────┐
│ your apps   │ ────  HTTP  API  ───▶ │  strata-serve │ ───────────────▶ │ strata engine  │
│ / web chat  │   http://127.0.0.1    │  (python)     │                  │ (CUDA, built   │
└─────────────┘                       └───────────────┘                  │  from source)  │
                                                                          └────────────────┘
        weights: downloaded once (resumable) into /var/lib/strata, not in the nix store
```

## Requirements

- NixOS on **x86_64-linux** with an NVIDIA RTX 20/30/40/50 GPU (12 GB VRAM or more)
  and the official driver (580+). About 64 GB of RAM for IQ2_XS and below.
- ~70 GB of free disk space for the model files (downloaded at first service start).

## Quick start

```nix
# flake.nix of your system
{
  inputs.strata-nix.url = "github:Yunor743/strata.nix";
  # ...
  outputs = { self, nixpkgs, strata-nix, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        strata-nix.nixosModules.default
        {
          services.strata.enable = true;
          # defaults: Swift 1.5, IQ2_XS, 32K context, API on 127.0.0.1:8080
        }
      ];
    };
  };
}
```

The service:

1. (first start) downloads the model shards from Hugging Face, builds the pack and
   the MTP draft layer — resumable, so an interrupted start continues where it left off;
2. writes `/var/lib/strata/config.json` (engine arguments, API key if any);
3. starts `strata-serve` on `http://127.0.0.1:8080`.

While the model loads (35-55 GB into RAM) the machine can be slow for 1-3 minutes:
that is normal. API: `http://127.0.0.1:8080/v1/chat/completions` (OpenAI) and
`/v1/messages` (Anthropic). Chat in the browser: `http://127.0.0.1:8080/`.

## Options (`services.strata.*`)

| Option | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | Enable the service. |
| `package` | `strataPkgs.strata` | The strata-serve bundle (carries the engine). |
| `family` | `swift` | `swift` (Swift 1.5: same accuracy, answers ~1.8x sooner), `qwen` (original) or `coder` (half the experts, code-focused, ~32 GB RAM). |
| `model` | `IQ2_XS` | `Q2_0` / `IQ2_XS` / `IQ3_XXS` / `IQ3_S` (qwen only) / `IQ1_M` (coder only). |
| `context` | `32768` | 8192–262144 tokens. |
| `host` / `port` | `127.0.0.1` / `8080` | Listen address. |
| `apiKeyFile` | `null` | Path to a file with the API key (systemd credential). Required when `host` is not loopback. |
| `hfTokenFile` | `null` | Hugging Face token file, only for gated repos. |
| `openFirewall` | `false` | Open the TCP port in the firewall. |
| `gpus` | `[ ]` | CUDA device indices; several = layer split. Empty: the card with the most VRAM. |
| `lowRam` | `null` | Map experts from disk instead of RAM copies (auto when RAM is short). |
| `sampling` | `{ }` | Sampling defaults (temperature, top_p, top_k, …). |
| `mcpServers` | `{ }` | MCP servers for the web chat (Claude Desktop format; they run with the service's rights). |
| `extraEngineArgs` | `[ ]` | Extra engine arguments, e.g. `[ "--kv" "q4_0" ]`. |
| `mockEngine` | `false` | Scripted mock instead of the model (tests; no GPU, no download). |

Example with the original model, the best quant and a LAN-reachable API:

```nix
services.strata = {
  enable = true;
  family = "qwen";
  model = "IQ3_XXS";
  context = 131072;
  host = "0.0.0.0";
  port = 8080;
  apiKeyFile = "/run/secrets/strata-api-key";
  sampling = { temperature = 1.0; top_p = 0.95; top_k = 20; };
};
```

## Packages

- `strata` — the server bundle: `strata-serve`, `strata-chat` and `strata-prefetch`
  (the download/prepare tool, usable standalone; `strata-prefetch --help`).
- `strata-engine` — the CUDA inference binary, built from source
  (`strata-engine.override { cudaArch = "120"; }` for an RTX 50 card; default `89` = RTX 40).
- The model weights are **not** packaged: they are downloaded at runtime into the
  service's state directory (the nix store would not survive a 70 GB model well).

## How the engine is built

`strata-engine` compiles the pinned Strata source with nixpkgs' CUDA toolchain
(`backendStdenv`, CUDA 12.9) against the llama.cpp commit pinned in Strata's
`third_party/ggml/VERSION.txt`. If Strata bumps that pin, evaluation fails with the
new commit and an instruction to update the `llama-pinned` input.

The flake input `strata` tracks upstream `main`. The lock file decides what you
actually run — `nix flake lock --update-input strata` pulls new releases
(and new engine features); audit what lands in it before doing so.

## Testing

- `nix flake check` — a serve smoke test (mock engine: HTTP endpoints without a GPU)
  and a NixOS VM test of the systemd service with the mock engine.
- `nix build .#strata-engine` — the real compile (20-40 minutes, once, cached after).

## Licenses

- This flake: MIT.
- Strata (engine, server, tools): MIT, upstream <https://github.com/Niko1221/Strata>.
- The model weights are not part of this flake; each model's license applies
  (Qwen Community 1.0; Swift Open License 1.0; see the models' Hugging Face pages).
