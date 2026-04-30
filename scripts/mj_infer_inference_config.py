#!/usr/bin/env python3
import json
import sys

import onnx


def main() -> None:
    model = onnx.load(sys.argv[1])
    input_dims = {
        value.name: [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
        for value in model.graph.input
    }
    obs_shape = input_dims.get("obs") or input_dims.get("actor_obs") or []
    obs_dim = obs_shape[1] if len(obs_shape) >= 2 and isinstance(obs_shape[1], int) else None
    metadata = {}
    for prop in model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except Exception:
            metadata[prop.key] = prop.value
    actor_input_dim = (
        metadata.get("experiment_config", {})
        .get("algo", {})
        .get("config", {})
        .get("module_dict", {})
        .get("actor", {})
        .get("input_dim")
    )
    actor_input_dim = actor_input_dim if isinstance(actor_input_dim, list) else []
    if "perception_obs" in input_dims and obs_dim == 96 and actor_input_dim == [
        "actor_obs_root_contact_aware",
        "actor_obs_proprio",
        "actor_obs_actions",
    ]:
        print("g1-29dof-wbt-object-contact-aware-depth-distill")
    elif "perception_obs" in input_dims and obs_dim == 308:
        print("g1-29dof-wbt-object-distill")
    elif obs_dim == 105:
        print("g1-29dof-wbt-object-mocap-distill")
    else:
        raise SystemExit(f"Unable to infer inference config from {sys.argv[1]}: obs_dim={obs_dim}, inputs={input_dims}")


if __name__ == "__main__":
    main()
