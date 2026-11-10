{
  lib,
  pkgs,
  runCommand,
  strataModule,
}:

# extraEngineArgs reaches the engine through strata-prefetch's --extra-arg, one flag per element.
# An element that starts with a dash (every engine flag does) must travel in the --extra-arg=VALUE
# form: argparse classifies a dash-leading value given after a space as an option of its own
# ("expected one argument").  Regression for PR #3.
let
  fullPkg = {
    outPath = "/nix/store/00000000000000000000000000000000-strata";
    passthru.engine = "/nix/store/11111111111111111111111111111111-strata-engine";
    passthru.vision = "/nix/store/22222222222222222222222222222222-strata-vision";
  };

  # The module is a NixOS module; this stub carries the pieces of NixOS it names that a bare
  # evalModules does not (networking.firewall.allowedTCPPorts, the top-level assertions list,
  # the systemd unit).  The unit is read back as written (types.raw).
  shim = {
    options.networking.firewall.allowedTCPPorts = lib.mkOption {
      type = lib.types.listOf lib.types.port;
      default = [ ];
    };
    options.assertions = lib.mkOption {
      type = lib.types.listOf lib.types.unspecified;
      default = [ ];
    };
    options.systemd.services = lib.mkOption {
      type = lib.types.lazyAttrsOf lib.types.raw;
    };
  };

  eval = cfg: (lib.evalModules {
    modules = [
      { _module.args.pkgs = pkgs; }
      shim
      strataModule
      { services.strata = cfg // { package = fullPkg; }; }
    ];
  }).config;

  good = eval {
    enable = true;
    vision = "off";
    extraEngineArgs = [ "--kv" "int8" ];
  };

  # types.raw keeps mkIf wrappers ({ _type = "if"; ... }) inside the value; unwrap the one level this
  # check reads (the service's preStart is mkIf'd on prefetchOnStart).  What comes out is the store
  # path string of the writeShellScript derivation; interpolating it below builds it and greps it.
  unIf = v: if (v ? _type && v._type == "if") then (if v.condition then v.content else "") else v;
  psGood = unIf good.systemd.services.strata.preStart;
in
runCommand "strata-module-extra-args" { } ''
  grep -q -- "--extra-arg=--kv" ${psGood}
  grep -q -- "--extra-arg=int8" ${psGood}
  touch $out
''
