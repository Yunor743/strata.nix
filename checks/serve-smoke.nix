{
  stdenv,
  strataPkg,
  curl,
}:

stdenv.mkDerivation {
  name = "strata-serve-smoke";

  nativeBuildInputs = [ strataPkg curl ];

  buildCommand = ''
    export HOME=$NIX_BUILD_TOP
    ${strataPkg}/bin/strata-serve --engine mock --port 8093 --script "smoke test answer" > server.log 2>&1 &
    pid=$!
    trap 'kill $pid 2>/dev/null || true; cat server.log' EXIT

    ready=""
    for i in $(seq 1 30); do
      if curl -sf http://127.0.0.1:8093/health > /dev/null 2>&1; then ready=yes; break; fi
      sleep 1
    done
    if [[ -z "$ready" ]]; then
      echo "server did not come up:"; cat server.log; exit 1
    fi

    curl -sf http://127.0.0.1:8093/health | grep -q '"status": "ok"' || {
      echo "health check failed"; exit 1; }

    curl -sf http://127.0.0.1:8093/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":200}' \
      | grep -q 'smoke test answer' || {
      echo "chat completion failed"; exit 1; }

    curl -sf http://127.0.0.1:8093/v1/models | grep -q '"object": "model"' || {
      echo "models endpoint failed"; exit 1; }

    curl -sf http://127.0.0.1:8093/metrics | grep -q '"live"' || {
      echo "metrics endpoint failed"; exit 1; }

    kill $pid
    trap - EXIT
    touch $out
  '';
}
