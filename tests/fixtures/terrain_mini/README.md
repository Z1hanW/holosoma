# terrain_mini

Small in-repo fixture for terrain/motion mismatch debugging.

Includes 3 paired clips:

- `stair_16`
- `stair_47`
- `stair_88`

Layout:

- `___crisp_clean_motion/*.npz`
- `___crisp_clean_geometry/*.obj`
- `___crisp_clean_geometry/*.support.npz`

These files are copied from the local CRISP clean dataset and kept intentionally minimal so `test.sh` can run the same terrain-generalist training path with:

- `NUM_ENVS=2`
- `HEADLESS=False`
- pairing enabled (`PAIR_TERRAIN_WITH_MOTION=True`)
