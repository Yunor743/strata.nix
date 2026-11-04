{
  testers,
  strataModule,
}:

testers.nixosTest {
  name = "strata-module-vm";

  nodes.machine = { pkgs, ... }: {
    imports = [ strataModule ];
    services.strata = {
      enable = true;
      mockEngine = true;
      prefetchOnStart = false;
      port = 8080;
    };
    environment.systemPackages = [ pkgs.curl ];
    virtualisation.memorySize = 1024;
  };

  # Non-default dataDir (with a space, to pin the unit/tmpfiles quoting):
  # exercises the static strata user, ReadWritePaths, RequiresMountsFor and
  # the tmpfiles rule instead of DynamicUser/StateDirectory.
  nodes.static = { pkgs, ... }: {
    imports = [ strataModule ];
    services.strata = {
      enable = true;
      mockEngine = true;
      prefetchOnStart = false;
      port = 8081;
      dataDir = "/var/lib/other strata";
    };
    environment.systemPackages = [ pkgs.curl ];
    virtualisation.memorySize = 1024;
  };

  testScript = ''
    machine.start()
    machine.wait_for_open_port(8080)

    machine.succeed("curl -sf http://127.0.0.1:8080/health | grep -q '\"status\": \"ok\"'")

    machine.succeed(
      "curl -sf http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' "
      "-d '{\"model\":\"mock\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":200}' "
      "| grep -q 'Hello from the mock engine.'"
    )

    machine.succeed("curl -sf http://127.0.0.1:8080/v1/models | grep -q '\"object\": \"model\"'")

    static.start()
    static.wait_for_unit("strata.service")
    static.wait_for_open_port(8081)
    static.succeed("stat -c '%U %a' '/var/lib/other strata' | grep -q '^strata 750$'")
    static.succeed("curl -sf http://127.0.0.1:8081/health | grep -q '\"status\": \"ok\"'")
  '';
}
