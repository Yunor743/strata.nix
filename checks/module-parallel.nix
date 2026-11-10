{
  lib,
  pkgs,
  runCommand,
  strataModule,
}:

let
  # Stand-ins for the store paths: the check evaluates the module, it does not build anything (the
  # prestart scripts below are writeShellScript one-liners, built by the runCommand that greps them).
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

  two = eval { enable = true; vision = "off"; parallel = 2; };
  eight = eval { enable = true; vision = "off"; parallel = 8; };
  none = eval { enable = true; vision = "off"; };
  # 9 is outside the option's type (ints.between 1 8): forcing the config must fail
  nine = builtins.tryEval (eval { enable = true; vision = "off"; parallel = 9; });

  # types.raw keeps mkIf wrappers ({ _type = "if"; ... }) inside the value; unwrap the one level this
  # check reads (the service's preStart is mkIf'd on prefetchOnStart).  What comes out is the store
  # path string of the writeShellScript derivation; interpolating it below builds it and greps it.
  unIf = v: if (v ? _type && v._type == "if") then (if v.condition then v.content else "") else v;
  psTwo = unIf two.systemd.services.strata.preStart;
  psEight = unIf eight.systemd.services.strata.preStart;
  psNone = unIf none.systemd.services.strata.preStart;
in
if !nine.success then throw "strata-module-parallel: parallel = 9 must not evaluate"
else runCommand "strata-module-parallel" { } ''
  grep -q -- "--parallel 2" ${psTwo}
  grep -q -- "--parallel 8" ${psEight}
  if grep -q -- "--parallel" ${psNone}; then echo "the default config must not pass --parallel"; exit 1; fi
  touch $out
''
