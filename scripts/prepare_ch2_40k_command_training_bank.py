"""Publish exact final ch2 rollouts, add commands, and validate 32-rank shards."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
AUDIT = Path('/data/holosoma_eval_audits/ch2ckwzw_model40000_batch137_native_20260908')
BASE = Path('/data/holosoma_inputs/ch2ckwzw_model40000_rollout137_precomputed_turn_forward_v1/by-source')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(script, *args):
    return subprocess.check_output([sys.executable, str(SCRIPTS / script), *map(str, args)], text=True).strip()


def main():
    if (AUDIT / 'bank_identity.json').exists():
        raise RuntimeError('Bank identity already exists; verify and reuse explicitly, do not regenerate')
    publication = json.loads(run('publish_ch2_40k_rollout137.py').splitlines()[-1])
    raw = Path(publication['target'])
    raw_manifest = json.loads((raw / 'manifest.json').read_text())
    source = raw_manifest['source_identity']
    assert source['checkpoint']['wandb_path'] == 'zihanw22/carry-any/ch2ckwzw/model_40000.pt'
    BASE.mkdir(parents=True, exist_ok=True)
    draft = BASE / 'prepared_generation'
    run('build_decoupled_root_command_bank.py', '--source', raw, '--output', draft,
        '--expected-clip-count', 137, '--expected-category-counts-json',
        '{"ball":34,"barrel":34,"bin":34,"box":35}',
        '--expected-source-payload-digest', publication['source_digest'],
        '--expected-source-manifest-sha256', publication['manifest_sha256'])
    m = json.loads((draft / 'manifest.json').read_text())
    bank = BASE / m['derived_payload_digest']
    if bank.exists():
        raise RuntimeError(f'Publication already exists: {bank}')
    draft.rename(bank)
    fd = os.open(BASE, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    fields = ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
              'body_lin_vel_w', 'body_ang_vel_w', 'object_pos_w',
              'object_quat_w', 'object_lin_vel_w', 'object_ang_vel_w']
    original = Path('/data/holosoma_worktrees/ch2ckwzw_batch_eval_dc1a1d8a/data/prism137_eval')
    differs = 0
    for row in m['clips']:
        name = row['clip_id'] + '.npz'
        assert sha(bank / name) == row['derived_npz_sha256']
        with np.load(raw / name, allow_pickle=False) as a, np.load(bank / name, allow_pickle=False) as b:
            for key in fields:
                assert np.array_equal(a[key], b[key]), (name, key)
            with np.load(original / name, allow_pickle=False) as old:
                differs += int(not np.array_equal(b['joint_pos'], old['joint_pos']))
    assert len(m['clips']) == differs == 137
    args = ['--motion-dir', bank, '--object-map', bank / '_clip_object_urdf_map.json',
            '--world-size', 32, '--environments-per-rank', 2048]
    digest = run('prepare_as_rank_shards.py', *args, '--source-digest-only')
    shards = bank / '_rank_shards' / 'by-source' / digest / 'ws32'
    # Add only the generated shard directory; do not mutate sealed NPZ/map/URDF payloads.
    bank.chmod(0o755)
    try:
        run('prepare_as_rank_shards.py', *args, '--output-root', shards, '--expected-source-digest', digest)
    finally:
        bank.chmod(0o555)
    shard_manifest = json.loads((shards / 'manifest.json').read_text())
    assert shard_manifest['exact_clip_partition']
    assert set(shard_manifest['clip_cover_counts'].values()) == {1}
    dataset = {
        'semantics': 'checkpoint_actor_simulator_rollout_motion_bank_with_precomputed_policy_commands',
        'effective_command_bank': str(bank), 'manifest_sha256': sha(bank / 'manifest.json'),
        'object_map_sha256': sha(bank / '_clip_object_urdf_map.json'),
        'single_slot_source_digest': m['source_view_digest'],
        'single_slot_view_digest': m['derived_payload_digest'],
        'raw_rollout_bank': str(raw), 'raw_rollout_digest': publication['source_digest'],
        'raw_rollout_manifest_sha256': publication['manifest_sha256'],
        'clip_count': 137, 'fps': 50, 'frames_per_clip': 359,
        'category_counts': m['category_counts'], 'all_137_retained_without_success_filtering': True,
        'parent_checkpoint': {
            'wandb_run': 'zihanw22/carry-any/ch2ckwzw', 'checkpoint': 'model_40000',
            'role': 'rollout_producer_only_not_online_training_teacher',
            'pt_sha256': source['checkpoint']['pt_sha256'],
            'onnx_sha256': source['checkpoint']['onnx_sha256'],
        },
        'rank_shards': {'world_size': 32, 'environments_per_rank': 2048,
            'source_digest': digest, 'manifest_sha256': sha(shards / 'manifest.json'),
            'exact_global_coverage_once': True, 'duplicates': False},
        'command_timeline': {'mode': 'precomputed_turn_then_forward',
            'phase_counts_zero_forward_yaw': m['total_phase_counts'],
            'lateral_command_always_zero': True, 'forward_and_yaw_never_overlap': True,
            'runtime_pickup_latch_required': True},
        'contact_sidecar_training_dependency': False,
    }
    identity = {'bank': str(bank), 'bank_digest': m['derived_payload_digest'],
        'source_digest': m['source_view_digest'], 'manifest_sha256': sha(bank / 'manifest.json'),
        'phase_counts': m['total_phase_counts'], 'shard_digest': digest,
        'shard_manifest_sha256': sha(shards / 'manifest.json'), 'dataset': dataset,
        'trajectory_fields_exactly_preserved': fields, 'old_reference_different_clips': differs,
        'raw_collection_core_success_count': source['core_success_count'],
        'raw_collection_strict_success_count': source['strict_success_count']}
    p = AUDIT / 'bank_identity.json'
    with p.open('x') as stream:
        json.dump(identity, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    p.chmod(0o444)
    print(json.dumps(identity, sort_keys=True))


if __name__ == '__main__':
    main()
