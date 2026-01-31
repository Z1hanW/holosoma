from __future__ import annotations
from typing import Tuple

import numpy as np

Pair = Tuple[str, str]
# mujoco_utils.py


from typing import Tuple, Optional
import numpy as np

def _mesh_full_local_vertices(model, mesh_id: int) -> np.ndarray:
    """Return all vertices (local frame) of a MuJoCo mesh asset."""
    v0 = int(model.mesh_vertadr[mesh_id])
    nv = int(model.mesh_vertnum[mesh_id])
    return model.mesh_vert[v0 : v0 + nv].astype(np.float64, copy=True)

def _mesh_convex_hull_local_vf(
    model,
    geom_id: int,
    *,
    fallback_to_scipy: bool = True,
    shrink_vertices: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (V_hull, F_hull) in the mesh LOCAL frame for a MuJoCo mesh geom's CONVEX HULL.

    - If model.mesh_graphadr[mesh_id] != -1: decode hull from mjModel.mesh_graph (exactly what MuJoCo stored).
    - Else: optionally fall back to scipy.spatial.ConvexHull(V_full).
    """
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None, None

    V_full = _mesh_full_local_vertices(model, mesh_id)
    graph_adr = int(model.mesh_graphadr[mesh_id])

    # ---- 1) Preferred: decode stored convex hull from mesh_graph ----
    if graph_adr != -1:
        g = np.asarray(model.mesh_graph, dtype=np.int32)
        idx = graph_adr

        numvert = int(g[idx]); idx += 1
        numface = int(g[idx]); idx += 1

        # record layout (see MuJoCo docs):
        # vert_edgeadr[numvert]
        idx += numvert
        # vert_globalid[numvert]
        idx += numvert
        # edge_localid[numvert + 3*numface]
        idx += (numvert + 3 * numface)
        # face_globalid[3*numface]
        face_globalid = g[idx : idx + 3 * numface].copy()
        F_global = face_globalid.reshape(numface, 3)

        # F_global indexes into the FULL mesh vertex array (0..nv-1)
        if not shrink_vertices:
            return V_full, F_global.astype(np.int32, copy=False)

        # shrink to only hull-used vertices (nicer for visualization)
        used = np.unique(F_global.reshape(-1))
        V_hull = V_full[used]
        # map global -> local by searchsorted (used is sorted)
        F_hull = np.searchsorted(used, F_global).astype(np.int32)
        return V_hull, F_hull

    # ---- 2) Fallback: compute convex hull ourselves ----
    if not fallback_to_scipy:
        return None, None

    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return None, None

    if V_full.shape[0] < 4:
        return None, None

    hull = ConvexHull(V_full)
    # hull.simplices are triangles indexing V_full
    F_global = hull.simplices.astype(np.int32)

    if not shrink_vertices:
        return V_full, F_global

    used = np.unique(F_global.reshape(-1))
    V_hull = V_full[used]
    F_hull = np.searchsorted(used, F_global).astype(np.int32)
    return V_hull, F_hull


def _world_hull_from_geom(model, data, geom_id: int):
    """Return (V_world, F) for the mesh geom's convex hull (if available)."""
    V_local, F = _mesh_convex_hull_local_vf(model, geom_id)
    if V_local is None or F is None:
        return None, None
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]
    V_world = V_local @ R.T + t
    return V_world, F


def _mesh_local_vf(model, geom_id):
    """Return local vertices and faces for a MuJoCo mesh geom."""
    mesh_id = int(model.geom_dataid[geom_id])  # Note: sometime geom does not have mesh, mesh_id will be -1

    v0, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    f0, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])

    V = model.mesh_vert[v0 : v0 + nv].astype(np.float64, copy=True)

    F = model.mesh_face[f0 : f0 + nf].astype(np.int32, copy=True)

    return V, F


def _to_world(v_local, data, geom_id):
    """Transform local vertices to world using geom pose."""
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]

    return v_local @ R.T + t


def _world_mesh_from_geom(model, data, geom_id, geom_name):
    V_local, F = _mesh_local_vf(model, geom_id)

    V_world = _to_world(V_local, data, geom_id)

    return V_world, F
