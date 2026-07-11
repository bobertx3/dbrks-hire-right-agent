#!/usr/bin/env bash
#
# One-command deploy: provision the bundle, then kick off the workflow that
# wires everything up (data → Lakebase → ML → agent → Genie → app + all grants).
#
#   ./scripts/deploy.sh [TARGET] [PROFILE]
#
# Defaults: TARGET=default, PROFILE=DEFAULT
#
# Step 1  `bundle deploy` provisions the Job, the dashboard, and the Databricks
#         App — including the app's service-principal resource bindings (agent
#         endpoint, warehouse, Genie, Lakebase), declaratively.
# Step 2  `bundle run hrd_setup_job` runs the end-to-end pipeline, which does all
#         the dynamic wiring: builds Bronze→Gold, sets up Lakebase, creates the
#         Genie space, trains + serves the ML model, deploys the agent, deploys
#         the app source, and (grant_app_permissions / notebook 09) grants the
#         current app SP UC + Genie + warehouse + endpoint access.
#
set -euo pipefail

TARGET="${1:-default}"
PROFILE="${2:-DEFAULT}"

echo "▶ bundle deploy  (target=$TARGET, profile=$PROFILE)"
databricks bundle deploy -t "$TARGET" --profile "$PROFILE"

echo
echo "▶ bundle run hrd_setup_job  (full setup: train_model=true, deploy_agent=true)"
databricks bundle run hrd_setup_job -t "$TARGET" --profile "$PROFILE" \
  --params train_model=true,deploy_agent=true

echo
echo "✓ Deploy + workflow complete. App:"
databricks apps get "$(databricks bundle summary -t "$TARGET" --profile "$PROFILE" -o json 2>/dev/null \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["resources"]["apps"]["hire_right_app"]["name"])' 2>/dev/null \
  || echo hire-right-agent-app)" --profile "$PROFILE" 2>/dev/null \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  "+ (d.get("url") or "(url pending)"))' 2>/dev/null || true
