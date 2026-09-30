{ strataPkgs }:
{ pkgs, config, lib, ... }:
let
  cfg = config.services.strata;
in
{
  options.services.strata = {
    enable = lib.mkEnableOption "Strata, a local OpenAI/Anthropic-compatible API for Qwen3.8-Flash-Next (125B MoE)";

    package = lib.mkPackageOption strataPkgs "strata" { };

    family = lib.mkOption {
      type = lib.types.enum [ "qwen" "swift" "coder" ];
      default = "swift";
      description = ''
        Which model: `swift` (Swift 1.5, answers ~1.8x sooner for the same accuracy),
        `qwen` (the original Qwen3.8-Flash-Next) or `coder` (half the experts kept, code-focused).
      '';
    };

    model = lib.mkOption {
      type = lib.types.enum [ "Q2_0" "IQ2_XS" "IQ3_XXS" "IQ3_S" "IQ1_M" ];
      default = "IQ2_XS";
      description = ''
        Quantization size: Q2_0 (fastest), IQ2_XS (recommended), IQ3_XXS (better, slower),
        IQ3_S (best, qwen only, ~62 GB RAM), IQ1_M (coder only, ~32 GB RAM).
      '';
    };

    context = lib.mkOption {
      type = lib.types.enum [ 8192 32768 65536 131072 262144 ];
      default = 32768;
      description = "Context window in tokens. From 64K on, the KV cache streams from RAM.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Listen address. Use 0.0.0.0 to serve the LAN (requires apiKeyFile).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };

    apiKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding the API key (read through a systemd credential, never in the unit).";
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding a Hugging Face token (only for gated model repos).";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };

    gpus = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = ''
        CUDA device numbers (as nvidia-smi numbers them, PCI order).
        Empty: the engine picks the card with the most VRAM.
        Several: layer split across the cards.
      '';
    };

    lowRam = lib.mkOption {
      type = lib.types.nullOr lib.types.bool;
      default = null;
      description = ''
        Map the experts from the pack's experts.bin instead of copying them into RAM
        (needs a big GPU to stay fast). Null: automatic from the model's RAM needs.
      '';
    };

    sampling = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      example = { temperature = 1.0; top_p = 0.95; top_k = 20; };
      description = "Sampling defaults for requests that leave the fields out.";
    };

    mcpServers = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = ''
        MCP servers for the web app's chat, in Claude Desktop's format.
        They run with this service's rights; only add ones you trust.
      '';
    };

    extraEngineArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra arguments appended to the engine command line (e.g. --kv q4_0).";
    };

    mockEngine = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Serve a scripted mock engine instead of the model (tests; no GPU or download).";
    };

    prefetchOnStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Download/prepare missing model files before starting (resumable, skipped when done).";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.model != "IQ3_S" || cfg.family == "qwen";
        message = "services.strata: IQ3_S is only published for family = qwen.";
      }
      {
        assertion = cfg.model != "IQ1_M" || cfg.family == "coder";
        message = "services.strata: IQ1_M is only published for family = coder.";
      }
      {
        assertion = cfg.model != "IQ2_XS" || cfg.family != "coder";
        message = "services.strata: the coder family ships IQ1_M only; pick model = IQ1_M.";
      }
      {
        assertion = cfg.host == "127.0.0.1" || cfg.host == "localhost" || cfg.apiKeyFile != null;
        message = "services.strata: exposing the API beyond loopback requires apiKeyFile.";
      }
      {
        assertion = !cfg.mockEngine || !cfg.prefetchOnStart;
        message = "services.strata: mockEngine requires prefetchOnStart = false.";
      }
      {
        assertion = cfg.mockEngine || (cfg.package ? passthru.engine);
        message = "services.strata: package must be the strata-serve bundle (it carries the engine under passthru.engine).";
      }
    ];

    systemd.services.strata = {
      description = "Strata: Qwen3.8-Flash-Next (125B MoE) local inference API";
      documentation = [ "https://github.com/Niko1221/Strata" ];
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ];

      environment = lib.optionalAttrs (cfg.gpus != [ ]) {
        CUDA_DEVICE_ORDER = "PCI_BUS_ID";
        CUDA_VISIBLE_DEVICES = lib.concatStringsSep "," cfg.gpus;
      };

      preStart = lib.mkIf (cfg.prefetchOnStart && !cfg.mockEngine) (toString (
        pkgs.writeShellScript "strata-prestart" (''
          set -euo pipefail
          state="$STATE_DIRECTORY"
          exec ${cfg.package}/bin/strata-prefetch \
            --family ${lib.escapeShellArg cfg.family} \
            --model ${lib.escapeShellArg cfg.model} \
            --context ${toString cfg.context} \
            --data-dir "$state" \
        '' + lib.optionalString (cfg.lowRam != null) ''
            --low-ram ${if cfg.lowRam then "on" else "off"} \
        '' + lib.optionalString (cfg.hfTokenFile != null) ''
            --hf-token-file "/run/credentials/strata.service/hf-token" \
        '' + ''
            --emit-config "$state/config.json" \
            --engine ${lib.escapeShellArg (toString cfg.package.passthru.engine)}/bin/strata \
            --host ${lib.escapeShellArg cfg.host} \
        '' + lib.optionalString (cfg.apiKeyFile != null) ''
            --api-key-file "/run/credentials/strata.service/api-key" \
        '' + lib.optionalString (cfg.gpus != [ ]) ''
            --gpu ${lib.escapeShellArg (lib.concatStringsSep "," cfg.gpus)} \
        '' + lib.optionalString (cfg.sampling != { }) ''
            --sampling-json ${pkgs.writeText "strata-sampling.json" (builtins.toJSON cfg.sampling)} \
        '' + lib.optionalString (cfg.mcpServers != { }) ''
            --mcp-json ${pkgs.writeText "strata-mcp.json" (builtins.toJSON cfg.mcpServers)} \
        '' + lib.concatMapStrings (x: ''--extra-arg ${lib.escapeShellArg x} '') cfg.extraEngineArgs
        )
      ));

      serviceConfig = {
        Type = "simple";
        DynamicUser = true;
        StateDirectory = "strata";
        StateDirectoryMode = "0750";
        # First start: preStart downloads ~70 GB and builds the pack - the default
        # 90 s TimeoutStartSec kills it mid-download (the service is resumable, but
        # converge it in one pass). Applies to the start, incl. ExecStartPre.
        TimeoutStartSec = "0";
        WorkingDirectory = "%S/strata";
        UMask = "0077";
        Restart = "on-failure";
        RestartSec = 10;
        LoadCredential =
          (lib.optional (cfg.apiKeyFile != null) "api-key:${cfg.apiKeyFile}")
          ++ (lib.optional (cfg.hfTokenFile != null) "hf-token:${cfg.hfTokenFile}");
        ExecStart =
          if cfg.mockEngine
          then
            "${cfg.package}/bin/strata-serve --engine mock --port ${toString cfg.port} " +
            "--host ${lib.escapeShellArg cfg.host}"
          else
            "${cfg.package}/bin/strata-serve --engine strata --config %S/strata/config.json " +
            "--port ${toString cfg.port}";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ];
        LockPersonality = true;
        MemoryDenyWriteExecute = false;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}
