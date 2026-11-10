{
  lib,
  pkgs,
  runCommand,
  strataModule,
}:

let
  # Stand-ins for the store paths: the check evaluates the module, it does not build anything (the
  # prestart scripts below are writeShellScript one-liners, built by the runCommand that greps them).
  enginePath = "/nix/store/11111111111111111111111111111111-strata-engine";
  visionPath = "/nix/store/22222222222222222222222222222222-strata-vision";
  fullPkg = {
    outPath = "/nix/store/00000000000000000000000000000000-strata";
    passthru.engine = enginePath;
    passthru.vision = visionPath;
  };
  barePkg = {
    outPath = "/nix/store/33333333333333333333333333333333-strata";
    passthru.engine = enginePath;
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

  eval = cfg: pkg: (lib.evalModules {
    modules = [
      { _module.args.pkgs = pkgs; }
      shim
      strataModule
      { services.strata = cfg // { package = pkg; }; }
    ];
  }).config;

  good = eval { enable = true; vision = "gpu"; } fullPkg;
  cpu = eval { enable = true; vision = "cpu"; } fullPkg;
  off = eval { enable = true; vision = "off"; } fullPkg;
  auto = eval { enable = true; } fullPkg;

  # a bare evalModules does not throw on failed assertions: it collects them in config.assertions
  firedAssertions = cfg: map (a: a.message) (lib.filter (a: !a.assertion) cfg.assertions);
  mockFired = firedAssertions
    (eval { enable = true; mockEngine = true; prefetchOnStart = false; vision = "gpu"; } fullPkg);
  bareFired = firedAssertions (eval { enable = true; vision = "gpu"; } barePkg);

  # types.raw keeps mkIf wrappers ({ _type = "if"; ... }) inside the value; unwrap the one level this
  # check reads (the service's preStart is mkIf'd on prefetchOnStart).  What comes out is the store
  # path string of the writeShellScript derivation; interpolating it below builds it and greps it.
  unIf = v: if (v ? _type && v._type == "if") then (if v.condition then v.content else "") else v;
  psGood = unIf good.systemd.services.strata.preStart;
  psCpu = unIf cpu.systemd.services.strata.preStart;
  psOff = unIf off.systemd.services.strata.preStart;

  evalOk = builtins.all lib.id [
    (auto.services.strata.vision == "auto")
    (mockFired == [ "services.strata: mockEngine serves text only; set vision = \"off\" with it." ])
    (bareFired == [ "services.strata: with vision on, package must be the strata-serve bundle (it carries the vision encoder under passthru.vision)." ])
  ];
in
if !evalOk then throw "strata-module-vision: the vision option did not wire up as expected"
else runCommand "strata-module-vision" { } ''
  grep -q -- "--vision gpu" ${psGood}
  grep -q -- "${visionPath}/bin/strata-vision" ${psGood}
  grep -q -- "--vision cpu" ${psCpu}
  grep -q -- "${visionPath}/bin/strata-vision" ${psCpu}
  grep -q -- "--vision off" ${psOff}
  if grep -q -- "vision-exe" ${psOff}; then echo "off prestart must not name the encoder"; exit 1; fi
  touch $out
''
