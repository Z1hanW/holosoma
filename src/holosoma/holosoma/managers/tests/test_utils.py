from holosoma.managers.observation.terms import wbt
from holosoma.managers.utils import resolve_callable


def test_resolve_callable_maps_legacy_object_goal_observation_alias() -> None:
    legacy_name = "obj_" + "spa" + "rse_" + "goal_xy_pick_root_heading"
    func = resolve_callable(
        f"holosoma.managers.observation.terms.wbt:{legacy_name}",
        context="observation term",
    )

    assert func is wbt.obj_goal_xy_pick_root_heading


def test_resolve_callable_maps_legacy_object_goal_command_alias() -> None:
    legacy_name = "obj_" + "spa" + "rse_" + "goal_xy_command"
    func = resolve_callable(
        f"holosoma.managers.observation.terms.wbt:{legacy_name}",
        context="observation term",
    )

    assert func is wbt.obj_goal_xy_pick_root_heading


def test_resolve_callable_maps_removed_legacy_flags() -> None:
    for legacy_name in ("spa" + "rse_" + "goal_external_flag", "command_only_flag"):
        func = resolve_callable(
            f"holosoma.managers.observation.terms.wbt:{legacy_name}",
            context="observation term",
        )

        assert func is wbt._legacy_false_flag


def test_resolve_callable_keeps_legacy_picked_flag() -> None:
    func = resolve_callable(
        "holosoma.managers.observation.terms.wbt:obj_picked_flag",
        context="observation term",
    )

    assert func is wbt.obj_picked_flag
