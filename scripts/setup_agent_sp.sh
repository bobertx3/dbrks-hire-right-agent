#!/usr/bin/env bash
#
# One-time (idempotent) setup of the Hire Right agent's runtime identity.
#
#   ./scripts/setup_agent_sp.sh [PROFILE] [--rotate]
#
# Defaults: PROFILE=DEFAULT. Requires workspace admin.
#
# Creates the `hire-right-agent-sp` service principal, mints an OAuth secret for
# it, and stores the client ID + secret in the `hire-right-agent` secret scope.
# Notebook 06 then:
#   - grants the SP exactly what the agent uses (UC schema, LLM + ML endpoints,
#     warehouse, Genie, trace tables) — re-applied on every deploy, and
#   - injects DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET into the agent
#     endpoint, so the agent (and its MLflow trace export to Unity Catalog)
#     runs as this SP rather than as a person.
#
# Re-running is safe: an existing SP / scope is reused, and the secret is only
# minted if the scope doesn't already hold one. Pass --rotate to mint a fresh
# secret (then redeploy or update the endpoint so it picks up the new value).
#
# Override defaults with env vars: SP_NAME, SECRET_SCOPE, SECRET_LIFETIME
# (seconds; default 2 years, the maximum).
#
set -euo pipefail

PROFILE="DEFAULT"
ROTATE=false
for arg in "$@"; do
  case "$arg" in
    --rotate) ROTATE=true ;;
    *)        PROFILE="$arg" ;;
  esac
done

SP_NAME="${SP_NAME:-hire-right-agent-sp}"
SECRET_SCOPE="${SECRET_SCOPE:-hire-right-agent}"
SECRET_LIFETIME="${SECRET_LIFETIME:-63072000}"

db() { databricks "$@" --profile "$PROFILE"; }
json_field() { python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['$1'] if isinstance(d, list) and d else (d.get('$1','') if isinstance(d, dict) else ''), end='')"; }

# ── 1. Service principal ──────────────────────────────────────────────────────
echo "▶ service principal  ($SP_NAME)"
SP_JSON=$(db service-principals list --filter "displayName eq \"$SP_NAME\"" -o json)
SP_ID=$(printf '%s' "$SP_JSON" | json_field id)
if [ -z "$SP_ID" ]; then
  SP_JSON=$(db service-principals create --display-name "$SP_NAME" --active -o json)
  SP_ID=$(printf '%s' "$SP_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'], end='')")
  echo "  ✓ created (id=$SP_ID)"
else
  echo "  ✓ exists  (id=$SP_ID)"
fi
APP_ID=$(db service-principals get "$SP_ID" -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['applicationId'], end='')")
echo "  application id: $APP_ID"

# ── 2. Secret scope ───────────────────────────────────────────────────────────
echo "▶ secret scope  ($SECRET_SCOPE)"
if db secrets list-scopes -o json | python3 -c "import sys,json; sys.exit(0 if any(s['name']=='$SECRET_SCOPE' for s in json.load(sys.stdin)) else 1)"; then
  echo "  ✓ exists"
else
  db secrets create-scope "$SECRET_SCOPE"
  echo "  ✓ created"
fi

# ── 3. OAuth client ID + secret → secret scope ────────────────────────────────
echo "▶ OAuth credentials  ($SECRET_SCOPE/sp-client-id, $SECRET_SCOPE/sp-client-secret)"
HAS_SECRET=$(db secrets list-secrets "$SECRET_SCOPE" -o json | python3 -c "import sys,json; print(any(s['key']=='sp-client-secret' for s in json.load(sys.stdin)), end='')")
printf '%s' "$APP_ID" | db secrets put-secret "$SECRET_SCOPE" sp-client-id
if [ "$HAS_SECRET" = "True" ] && [ "$ROTATE" = false ]; then
  echo "  ✓ secret already stored (pass --rotate to mint a new one)"
else
  # The secret value is piped straight into the scope and never printed.
  db service-principal-secrets-proxy create "$SP_ID" --lifetime "${SECRET_LIFETIME}s" -o json \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['secret'], end='')" \
    | db secrets put-secret "$SECRET_SCOPE" sp-client-secret
  echo "  ✓ new secret stored (lifetime ${SECRET_LIFETIME}s)"
fi

echo
echo "✅ Agent identity ready: $SP_NAME ($APP_ID)"
echo "   Next: run the pipeline (./scripts/deploy.sh) or notebook 06 — it grants the SP"
echo "   its permissions and deploys the agent endpoint with these credentials."
