from holosoma.managers.observation.terms import wbt
from holosoma.managers.utils import resolve_callable


def test_resolve_callable_maps_legacy_object_goal_observation_alias() -> None:
    legacy_name = "obj_" + "spa" + "rse_" + "goal_xy_pick_root_heading"
    func = resolve_callable(
        f"holosoma.managers.observation.terms.wbt:{legacy_name}",
        context="observation term",
    )

    assert func is wbt.obj_goal_xy_pick_root_heading
