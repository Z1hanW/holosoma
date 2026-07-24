from __future__ import annotations

import hashlib
import json
import struct
import sys
import threading
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
import zmq


REPO_ROOT = Path(__file__).resolve().parents[2]
for package_root in (REPO_ROOT / "src" / "holosoma", REPO_ROOT / "src" / "holosoma_inference"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from holosoma.utils.perception_obs import (  # noqa: E402
    PERCEPTION_OBS_SHM_CHECKSUM_OFFSET,
    PERCEPTION_OBS_SHM_HEADER_SIZE,
    PERCEPTION_OBS_SHM_PUBLISH_MONOTONIC_NS_OFFSET,
    PERCEPTION_OBS_SHM_SEQUENCE_OFFSET,
    PerceptionObsPub,
    PerceptionObsShmPub,
)
from holosoma.relay_perception_obs import Publisher as RelayPublisher  # noqa: E402
from holosoma.relay_perception_obs import ShmSource as RelayShmSource  # noqa: E402
from holosoma.relay_perception_obs import ZmqSource as RelayZmqSource  # noqa: E402
from holosoma.relay_perception_obs import _frame_identity as relay_frame_identity  # noqa: E402
from holosoma_inference.utils.perception_obs import (  # noqa: E402
    PerceptionObsShmSub,
    PerceptionObsSub,
)


def _name() -> str:
    return f"holosoma_perception_test_{uuid.uuid4().hex}"


def _checksum(payload: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _cleanup_name(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    finally:
        shm.close()


def test_zmq_publisher_fails_closed_instead_of_silently_dropping() -> None:
    pub = PerceptionObsPub(port=5558)
    with pytest.raises(RuntimeError, match="not enabled"):
        pub.publish({"perception_obs": [1.0]})

    class _WouldBlockSocket:
        @staticmethod
        def send_string(payload: str, flags: int) -> None:
            del payload, flags
            raise zmq.Again()

    pub.enabled = True
    pub.socket = _WouldBlockSocket()
    with pytest.raises(RuntimeError, match="would drop a frame"):
        pub.publish({"perception_obs": [1.0]})


def test_protocol_round_trip_exact_size_and_timestamps() -> None:
    name = _name()
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    sub.start()
    try:
        pub.publish([1.25, -2.5, 3.75], sim_time_ms=123.5)
        obs = sub.get_obs(3)
        assert obs is not None
        np.testing.assert_array_equal(obs, np.array([[1.25, -2.5, 3.75]], dtype=np.float32))
        assert sub.last_sequence == 2
        assert sub.last_publish_monotonic_ns is not None
        assert sub.last_sim_time_ms == pytest.approx(123.5)

        raw = shared_memory.SharedMemory(name=name, create=False)
        try:
            assert len(raw.buf) == PERCEPTION_OBS_SHM_HEADER_SIZE + 3 * np.dtype("<f4").itemsize
        finally:
            raw.close()
    finally:
        sub.close()
        pub.close(unlink=True)


def test_protocol_authenticates_effective_perception_contract() -> None:
    name = _name()
    expected_contract = "11" * 32
    wrong_contract = "22" * 32
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish([1.0, 2.0], contract_sha256=expected_contract)
        np.testing.assert_array_equal(
            sub.get_obs(2, expected_contract),
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
        assert sub.get_obs(2, wrong_contract) is None
        with pytest.raises(RuntimeError, match="contract changed"):
            pub.publish([3.0, 4.0], contract_sha256=wrong_contract)
    finally:
        sub.close()
        pub.close(unlink=True)


def test_protocol_authenticates_each_frame_episode_generation() -> None:
    name = _name()
    contract_sha256 = "ab" * 32
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish(
            [1.0, 2.0],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=7,
        )
        np.testing.assert_array_equal(
            sub.get_obs(
                2,
                contract_sha256,
                expected_episode_generation=7,
            ),
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
        assert sub.last_episode_generation == 7
        assert (
            sub.get_obs(
                2,
                contract_sha256,
                expected_episode_generation=8,
            )
            is None
        )

        pub.publish(
            [3.0, 4.0],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=8,
        )
        np.testing.assert_array_equal(
            sub.get_obs(
                2,
                contract_sha256,
                expected_episode_generation=8,
            ),
            np.array([[3.0, 4.0]], dtype=np.float32),
        )
        assert sub.last_episode_generation == 8
    finally:
        sub.close()
        pub.close(unlink=True)


def test_protocol_rejects_changing_episode_authentication_presence() -> None:
    name = _name()
    pub = PerceptionObsShmPub(name)
    pub.start()
    try:
        pub.publish([1.0], episode_generation=0)
        with pytest.raises(RuntimeError, match="episode-generation authentication changed"):
            pub.publish([2.0])
    finally:
        pub.close(unlink=True)


def test_publisher_rejects_nonfinite_without_overwriting_last_valid_frame() -> None:
    name = _name()
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish([4.0, 5.0])
        before = sub.get_obs(2)
        assert before is not None
        sequence = sub.last_sequence

        with pytest.raises(ValueError, match="NaN or Inf"):
            pub.publish([np.nan, 7.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            pub.publish([8.0, np.inf])

        after = sub.get_obs(2)
        np.testing.assert_array_equal(after, before)
        assert sub.last_sequence == sequence
    finally:
        sub.close()
        pub.close(unlink=True)


def test_second_writer_cannot_replace_live_publisher_even_in_same_process() -> None:
    name = _name()
    first = PerceptionObsShmPub(name)
    second = PerceptionObsShmPub(name)
    first.start()
    second.start()
    try:
        first.publish([1.0, 2.0])
        with pytest.raises(RuntimeError, match="live publisher"):
            second.publish([3.0, 4.0])
        sub = PerceptionObsShmSub(name)
        try:
            np.testing.assert_array_equal(sub.get_obs(2), np.array([[1.0, 2.0]], dtype=np.float32))
        finally:
            sub.close()
    finally:
        second.close(unlink=True)
        first.close(unlink=True)


def test_subscriber_rejects_legacy_bare_float_layout_and_dimension_mismatch() -> None:
    legacy_name = _name()
    legacy = shared_memory.SharedMemory(name=legacy_name, create=True, size=4 * np.dtype(np.float32).itemsize)
    try:
        np.ndarray((4,), dtype=np.float32, buffer=legacy.buf)[:] = 1.0
        sub = PerceptionObsShmSub(legacy_name)
        try:
            assert sub.get_obs(4) is None
        finally:
            sub.close()
    finally:
        legacy.unlink()
        legacy.close()

    name = _name()
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish([1.0, 2.0, 3.0, 4.0])
        assert sub.get_obs(3) is None
        np.testing.assert_array_equal(sub.get_obs(4), np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    finally:
        sub.close()
        pub.close(unlink=True)


def test_subscriber_rejects_odd_sequence_bad_checksum_and_nonfinite_payload() -> None:
    name = _name()
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish([1.0, 1.0, 1.0, 1.0])
        assert sub.get_obs(4) is not None
        raw = shared_memory.SharedMemory(name=name, create=False)
        try:
            struct.pack_into("<Q", raw.buf, PERCEPTION_OBS_SHM_SEQUENCE_OFFSET, 3)
            np.ndarray(
                (4,),
                dtype=np.dtype("<f4"),
                buffer=raw.buf,
                offset=PERCEPTION_OBS_SHM_HEADER_SIZE,
            )[:] = [9.0, 8.0, 7.0, 6.0]
            assert sub.get_obs(4) is None

            struct.pack_into("<Q", raw.buf, PERCEPTION_OBS_SHM_SEQUENCE_OFFSET, 2)
            assert sub.get_obs(4) is None

            payload = np.array([np.nan, 2.0, 3.0, 4.0], dtype="<f4").tobytes()
            raw.buf[PERCEPTION_OBS_SHM_HEADER_SIZE :] = payload
            struct.pack_into("<Q", raw.buf, PERCEPTION_OBS_SHM_CHECKSUM_OFFSET, _checksum(payload))
            assert sub.get_obs(4) is None
        finally:
            raw.close()

        pub.publish([5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(sub.get_obs(4), np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32))
    finally:
        sub.close()
        pub.close(unlink=True)


def test_concurrent_reader_never_accepts_a_torn_frame() -> None:
    name = _name()
    dim = 1024
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name, max_age_s=2.0)
    pub.start()
    pub.publish(np.zeros(dim, dtype=np.float32), episode_generation=0)
    stop = threading.Event()
    writer_error: list[BaseException] = []

    def _writer() -> None:
        try:
            value = 1.0
            while not stop.is_set():
                pub.publish(
                    np.full(dim, value, dtype=np.float32),
                    episode_generation=int(value),
                )
                value = 2.0 if value == 1.0 else 1.0
        except BaseException as exc:  # pragma: no cover - assertion reports worker exception
            writer_error.append(exc)

    thread = threading.Thread(target=_writer, daemon=True)
    thread.start()
    accepted = 0
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and accepted < 100:
            obs = sub.get_obs(dim)
            if obs is None:
                continue
            flat = obs.reshape(-1)
            assert np.all(flat == flat[0])
            assert float(flat[0]) in {0.0, 1.0, 2.0}
            assert sub.last_episode_generation == int(flat[0])
            accepted += 1
        assert accepted >= 10
        assert not writer_error
    finally:
        stop.set()
        thread.join(timeout=2.0)
        sub.close()
        pub.close(unlink=True)


def test_publisher_invalidates_old_episode_frame_until_replacement() -> None:
    name = _name()
    contract_sha256 = "78" * 32
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish(
            [1.0, 2.0],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=300,
        )
        assert sub.get_obs(
            2,
            contract_sha256,
            expected_episode_generation=300,
        ) is not None

        pub.invalidate()
        assert sub.get_obs(2, contract_sha256) is None
        assert sub.get_obs(
            2,
            contract_sha256,
            expected_episode_generation=301,
        ) is None

        pub.publish(
            [3.0, 4.0],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=301,
        )
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(
                2,
                0.0,
                contract_sha256,
                expected_episode_generation=301,
            ),
            np.array([[3.0, 4.0]], dtype=np.float32),
        )
    finally:
        sub.close()
        pub.close(unlink=True)


def test_subscriber_drops_unlinked_mapping_and_attaches_new_generation() -> None:
    name = _name()
    pub1 = PerceptionObsShmPub(name)
    pub1.start()
    sub = PerceptionObsShmSub(name)
    pub2: PerceptionObsShmPub | None = None
    try:
        pub1.publish([1.0, 2.0])
        np.testing.assert_array_equal(sub.get_obs(2), np.array([[1.0, 2.0]], dtype=np.float32))
        first_generation = sub.generation

        pub1.close(unlink=True)
        assert sub.get_obs(2) is None
        assert sub.shm is None

        pub2 = PerceptionObsShmPub(name)
        pub2.start()
        pub2.publish([7.0, 8.0])
        np.testing.assert_array_equal(sub.get_obs(2), np.array([[7.0, 8.0]], dtype=np.float32))
        assert sub.generation != first_generation
    finally:
        sub.close()
        pub1.close(unlink=True)
        if pub2 is not None:
            pub2.close(unlink=True)
        _cleanup_name(name)


def test_subscriber_rejects_stale_or_future_publisher_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _name()
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name, max_age_s=0.05)
    pub.start()
    try:
        pub.publish([3.0, 4.0])
        publish_ns = pub._last_publish_monotonic_ns
        monkeypatch.setattr(time, "monotonic_ns", lambda: publish_ns + 100_000_000)
        assert sub.get_obs(2) is None

        raw = shared_memory.SharedMemory(name=name, create=False)
        try:
            struct.pack_into("<Q", raw.buf, PERCEPTION_OBS_SHM_PUBLISH_MONOTONIC_NS_OFFSET, publish_ns + 200_000_000)
        finally:
            raw.close()
        monkeypatch.setattr(time, "monotonic_ns", lambda: publish_ns + 150_000_000)
        assert sub.get_obs(2) is None
    finally:
        sub.close()
        pub.close(unlink=True)


def test_subscriber_selects_cached_frame_at_or_before_robot_state_time() -> None:
    name = _name()
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish([1.0, 10.0], sim_time_ms=100.0)
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(2, 100.0),
            np.array([[1.0, 10.0]], dtype=np.float32),
        )

        pub.publish([2.0, 20.0], sim_time_ms=120.0)
        # The newest frame is from the future relative to robot state t=110;
        # the reader must retain and select t=100 rather than leaking t=120.
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(2, 110.0),
            np.array([[1.0, 10.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(2, 120.0),
            np.array([[2.0, 20.0]], dtype=np.float32),
        )
        assert sub.get_obs_at_or_before(2, 99.0) is None
    finally:
        sub.close()
        pub.close(unlink=True)


def test_subscriber_never_crosses_episode_when_sim_time_restarts_at_zero() -> None:
    name = _name()
    contract_sha256 = "cd" * 32
    pub = PerceptionObsShmPub(name)
    sub = PerceptionObsShmSub(name)
    pub.start()
    try:
        pub.publish(
            [1.0, 10.0],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=41,
        )
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(
                2,
                0.0,
                contract_sha256,
                expected_episode_generation=41,
            ),
            np.array([[1.0, 10.0]], dtype=np.float32),
        )

        pub.publish(
            [2.0, 20.0],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=42,
        )
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(
                2,
                0.0,
                contract_sha256,
                expected_episode_generation=42,
            ),
            np.array([[2.0, 20.0]], dtype=np.float32),
        )
        # Both frames have the same reset timestamp. Episode identity, not
        # time, is what keeps the old depth frame out of the new control tick.
        np.testing.assert_array_equal(
            sub.get_obs_at_or_before(
                2,
                0.0,
                contract_sha256,
                expected_episode_generation=41,
            ),
            np.array([[1.0, 10.0]], dtype=np.float32),
        )
        assert (
            sub.get_obs_at_or_before(
                2,
                0.0,
                contract_sha256,
                expected_episode_generation=43,
            )
            is None
        )
    finally:
        sub.close()
        pub.close(unlink=True)


def test_zmq_subscriber_selects_exact_episode_before_time_filtering() -> None:
    sub = PerceptionObsSub(port=5558)
    old_payload = {
        "sim_time_ms": 0.0,
        "episode_generation": 11,
        "perception_obs": [1.0],
    }
    new_payload = {
        "sim_time_ms": 0.0,
        "episode_generation": 12,
        "perception_obs": [2.0],
    }
    sub.payload_buffer.extend([old_payload, new_payload])
    sub.last_payload = new_payload

    assert sub.get_payload_at_or_before(
        0.0,
        expected_episode_generation=11,
    ) is old_payload
    assert sub.get_payload_at_or_before(
        0.0,
        expected_episode_generation=12,
    ) is new_payload
    assert (
        sub.get_payload_at_or_before(
            0.0,
            expected_episode_generation=13,
        )
        is None
    )


def test_relay_source_and_destination_use_the_versioned_protocol() -> None:
    source_name = _name()
    dest_name = _name()
    contract_sha256 = "33" * 32
    source_pub = PerceptionObsShmPub(source_name)
    source = RelayShmSource(source_name, expected_dim=3, contract_sha256=contract_sha256)
    relay = RelayPublisher(
        dest_port=None,
        dest_shm_name=dest_name,
        expected_dim=3,
        initial_value=0.5,
        contract_sha256=contract_sha256,
    )
    dest_sub = PerceptionObsShmSub(dest_name)
    source_pub.start()
    try:
        assert dest_sub.get_obs(3, contract_sha256) is None
        source_pub.publish(
            [0.1, 0.2, 0.3],
            sim_time_ms=77.0,
            contract_sha256=contract_sha256,
            episode_generation=55,
        )
        source_obs = source.next_obs()
        assert source_obs is not None
        assert source.next_obs() is None
        assert source.last_sim_time_ms == pytest.approx(77.0)
        assert source.last_episode_generation == 55
        relay.publish(
            source_obs,
            frame_idx=0,
            sim_time_ms=source.last_sim_time_ms,
            episode_generation=source.last_episode_generation,
        )
        np.testing.assert_allclose(
            dest_sub.get_obs(
                3,
                contract_sha256,
                expected_episode_generation=55,
            ),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        )
        assert dest_sub.last_sim_time_ms == pytest.approx(77.0)
        assert dest_sub.last_episode_generation == 55

        with pytest.raises(ValueError, match="NaN or Inf"):
            relay.publish(
                np.array([1.0, np.nan, 2.0], dtype=np.float32),
                frame_idx=1,
                episode_generation=55,
            )
        np.testing.assert_allclose(
            dest_sub.get_obs(
                3,
                contract_sha256,
                expected_episode_generation=55,
            ),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        )
    finally:
        source.close()
        dest_sub.close()
        relay.close()
        source_pub.close(unlink=True)
        _cleanup_name(source_name)
        _cleanup_name(dest_name)


def test_relay_freshness_identity_changes_across_episode_with_same_pixels() -> None:
    obs = np.array([0.5, 0.5], dtype=np.float32)

    assert relay_frame_identity(
        obs,
        sim_time_ms=0.0,
        episode_generation=100,
    ) != relay_frame_identity(
        obs,
        sim_time_ms=0.0,
        episode_generation=101,
    )


def test_relay_shm_source_accepts_same_sequence_from_new_segment_session() -> None:
    name = _name()
    contract_sha256 = "56" * 32
    source = RelayShmSource(name, expected_dim=2, contract_sha256=contract_sha256)
    first = PerceptionObsShmPub(name)
    second: PerceptionObsShmPub | None = None
    first.start()
    try:
        first.publish(
            [0.1, 0.2],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=200,
        )
        assert source.next_obs() is not None
        first_segment_generation = source.sub.generation
        assert source.sub.last_sequence == 2
        assert source.next_obs() is None

        first.close(unlink=True)
        second = PerceptionObsShmPub(name)
        second.start()
        second.publish(
            [0.1, 0.2],
            sim_time_ms=0.0,
            contract_sha256=contract_sha256,
            episode_generation=201,
        )
        # A new segment starts its seqlock at 2 as well; its independent
        # segment generation must make this a fresh source frame.
        assert source.next_obs() is not None
        assert source.sub.last_sequence == 2
        assert source.sub.generation != first_segment_generation
        assert source.last_episode_generation == 201
    finally:
        source.close()
        first.close(unlink=True)
        if second is not None:
            second.close(unlink=True)
        _cleanup_name(name)


def test_relay_zmq_source_does_not_republish_after_source_disconnects() -> None:
    contract_sha256 = "34" * 32
    payload = {
        "sim_time_ms": 10.0,
        "episode_generation": 88,
        "perception_contract_sha256": contract_sha256,
        "perception_obs": [0.1, 0.2],
    }

    class Socket:
        def __init__(self) -> None:
            self.messages = [json.dumps(payload)]

        def recv_string(self, flags: int) -> str:
            del flags
            if not self.messages:
                raise zmq.Again()
            return self.messages.pop(0)

    source = object.__new__(RelayZmqSource)
    source.expected_dim = 2
    source.key = "perception_obs"
    source.contract_sha256 = contract_sha256
    source.socket = Socket()
    source.last_obs = None
    source.last_sim_time_ms = None
    source.last_episode_generation = None

    np.testing.assert_allclose(
        source.next_obs(),
        np.array([0.1, 0.2], dtype=np.float32),
    )
    assert source.next_obs() is None
