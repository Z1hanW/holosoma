#!/usr/bin/env bash

SIM2REAL_REGISTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_REGISTRY_FILE="${SIM2REAL_REGISTRY_DIR}/registry.tsv"

sim2real_registry_rows() {
  if [[ -f "$SIM2REAL_REGISTRY_FILE" ]]; then
    cat "$SIM2REAL_REGISTRY_FILE"
    return
  fi

  cat <<'EOF'
policy	run_id	inference_config	model_ref
w5qostjn	w5qostjn	g1-29dof-wbt-object-distill	https://wandb.ai/zihanw22/boxer/runs/w5qostjn
contact-aware-near0p3	xxehngzo	g1-29dof-wbt-object-contact-aware-depth-distill	https://wandb.ai/zihanw22/boxer/runs/xxehngzo
g1_box_perception_pure_sd_ppo_first_contact14	shoo7sr1	g1-29dof-wbt-object-distill	https://wandb.ai/zihanw22/boxer/runs/shoo7sr1
w5qostjn_linvel_contact_aware	tvtwx4to	g1-29dof-wbt-object-linvel-contact-aware-depth-distill	https://wandb.ai/zihanw22/boxer/runs/tvtwx4to
w5qostjn_linvel_action_history	haap1tjl	g1-29dof-wbt-object-linvel-action-history-depth-distill	https://wandb.ai/zihanw22/boxer/runs/haap1tjl
w5qostjn_action_history	5aotqbdq	g1-29dof-wbt-object-action-history-depth-distill	https://wandb.ai/zihanw22/boxer/runs/5aotqbdq
w5qostjn_linvel	c1gaknfu	g1-29dof-wbt-object-linvel-depth-distill	https://wandb.ai/zihanw22/boxer/runs/c1gaknfu
EOF
}

sim2real_known_policies() {
  sim2real_registry_rows | awk 'NR > 1 {print $1}'
}

sim2real_print_registry() {
  sim2real_registry_rows | column -t -s $'\t' 2>/dev/null || sim2real_registry_rows
}

sim2real_load_policy() {
  local policy_name="$1"
  local row
  row="$(sim2real_registry_rows | awk -F '\t' -v policy="$policy_name" 'NR > 1 && $1 == policy {print; exit}')"
  if [[ -z "$row" ]]; then
    return 1
  fi

  IFS=$'\t' read -r SIM2REAL_POLICY_NAME SIM2REAL_RUN_ID SIM2REAL_INFERENCE_CONFIG SIM2REAL_MODEL_REF <<<"$row"
  export SIM2REAL_POLICY_NAME
  export SIM2REAL_RUN_ID
  export SIM2REAL_INFERENCE_CONFIG
  export SIM2REAL_MODEL_REF
}
