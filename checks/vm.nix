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
  '';
}
