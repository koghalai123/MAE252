#!/usr/bin/env python3
"""
Pause-on-demand vs Buffered-sensor comparison test.

Flies an L-shaped path with sharp direction changes while simultaneously:
  A) Collecting LiDAR + GT pose via pause-on-demand (simPause → step → read → resume)
  B) Polling state at ~200 Hz into ring buffers and interpolating to LiDAR timestamps

Writes both approaches' positions, orientations, and yaw angles to a text file
for analysis.  If pause-on-demand is working correctly, the two should agree
closely (within a few degrees of yaw and a few cm of position).
"""

import asyncio
import bisect
import os
import sys
import time
import threading
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))
import cosysairsim as airsim
from scipy.spatial.transform import Rotation, Slerp

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sync_comparison.txt")

# ── Flight parameters ─────────────────────────────────────────────────
SPEED        = 3.0
POLL_HZ      = 20.0
ALTITUDE     = -8.0
SCAN_RATE_HZ = 2.0       # collect ~2 scans/sec (matches exploration default)

# L-shaped waypoints: north → east → south (sharp 90° direction changes)
WAYPOINTS = [
    np.array([15.0,  0.0, ALTITUDE]),    # north
    np.array([15.0, 20.0, ALTITUDE]),    # east
    np.array([ 0.0, 20.0, ALTITUDE]),    # south
]


# ── Helpers ───────────────────────────────────────────────────────────

def quat_to_yaw(ori_wxyz):
    """Extract yaw (degrees) from (w,x,y,z) quaternion."""
    r = Rotation.from_quat([ori_wxyz[1], ori_wxyz[2], ori_wxyz[3], ori_wxyz[0]])
    return r.as_euler('ZYX', degrees=True)[0]

def interp_vec(buf_ts, buf_vals, ts):
    """Linear-interpolate a vector from a timestamped buffer."""
    if len(buf_ts) < 2:
        return None, None
    idx = bisect.bisect_right(buf_ts, ts)
    ib = max(idx - 1, 0)
    ia = min(idx, len(buf_ts) - 1)
    ta, tb = buf_ts[ib], buf_ts[ia]
    near_ms = min(abs(ts - ta), abs(ts - tb)) / 1e6
    far_ms  = max(abs(ts - ta), abs(ts - tb)) / 1e6
    if tb != ta:
        t = np.clip((ts - ta) / (tb - ta), 0.0, 1.0)
        val = (1 - t) * np.asarray(buf_vals[ib]) + t * np.asarray(buf_vals[ia])
    else:
        val = np.asarray(buf_vals[ib]).copy()
    return val, (near_ms, far_ms)

def interp_quat(buf_ts, buf_ori, ts):
    """SLERP-interpolate a (w,x,y,z) quaternion from a timestamped buffer."""
    if len(buf_ts) < 2:
        return None
    idx = bisect.bisect_right(buf_ts, ts)
    ib = max(idx - 1, 0)
    ia = min(idx, len(buf_ts) - 1)
    ta, tb = buf_ts[ib], buf_ts[ia]
    if tb != ta:
        t_param = np.clip((ts - ta) / (tb - ta), 0.0, 1.0)
        o_a, o_b = buf_ori[ib], buf_ori[ia]
        rots = Rotation.from_quat([
            [o_a[1], o_a[2], o_a[3], o_a[0]],
            [o_b[1], o_b[2], o_b[3], o_b[0]],
        ])
        slerp = Slerp([0.0, 1.0], rots)
        q_scipy = slerp([t_param])[0].as_quat()  # x,y,z,w
        return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    else:
        return np.asarray(buf_ori[ib]).copy()


# ── Shared state ──────────────────────────────────────────────────────

done = threading.Event()

# Ring buffers for the high-rate sensor thread
sensor_lock = threading.Lock()
state_buf_ts  = []
state_buf_pos = []
state_buf_ori = []
BUF_MAX = 5000

# Results collected by the pause-on-demand thread
# Each entry: (wall_time, lidar_ts, raw_pos, raw_ori, raw_yaw, n_points, steps,
#              bp_pos, bp_ori, bp_yaw, delta_ms)
pause_results = []

# Interpolated results from buffer (computed inline)
# Each entry: (lidar_ts, interp_pos, interp_ori, interp_yaw, bracket_near_ms, bracket_far_ms)
buffer_results = []


# ── Thread functions ──────────────────────────────────────────────────

def sensor_thread_fn():
    """High-rate state polling at ~200 Hz (like exploration's _sensor_loop)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("  [sensor] connected, polling at ~200 Hz")

    while not done.is_set():
        try:
            state = client.getMultirotorState()
            ts = float(state.timestamp)
            sp = state.kinematics_estimated.position
            so = state.kinematics_estimated.orientation
            pos = np.array([sp.x_val, sp.y_val, sp.z_val])
            ori = np.array([so.w_val, so.x_val, so.y_val, so.z_val])

            with sensor_lock:
                state_buf_ts.append(ts)
                state_buf_pos.append(pos)
                state_buf_ori.append(ori)
                if len(state_buf_ts) > BUF_MAX:
                    trim = len(state_buf_ts) - BUF_MAX
                    del state_buf_ts[:trim]
                    del state_buf_pos[:trim]
                    del state_buf_ori[:trim]
        except Exception as e:
            print(f"  [sensor] error: {e}")
        time.sleep(0.003)  # ~200 Hz


def collection_thread_fn():
    """Pause-on-demand LiDAR collection with velocity back-projection."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("  [collect] connected, pause-on-demand mode (with velocity back-projection)")

    min_interval = 1.0 / SCAN_RATE_HZ
    last_scan_time = 0.0
    last_lidar_ts = 0

    while not done.is_set():
        now = time.time()
        if (now - last_scan_time) < min_interval:
            time.sleep(0.01)
            continue

        # ── Pause, read GT before, check LiDAR, step if needed, resume ──
        client.simPause(True)
        time.sleep(0.005)

        # Read GT + timestamp BEFORE any stepping
        gt_before = client.simGetGroundTruthKinematics()
        state_before = client.getMultirotorState()
        ts_before = float(state_before.timestamp)

        # Check if LiDAR is already fresh at this paused instant
        lidar_data = client.getLidarData()
        cur_ts = float(lidar_data.time_stamp)

        if cur_ts != last_lidar_ts and len(lidar_data.point_cloud) >= 9:
            # LiDAR already fresh — raw GT at paused instant
            gt_raw = gt_before
            last_lidar_ts = cur_ts
            steps = 0
            client.simPause(False)

            # ── Velocity back-projection ──────────────────────────
            # The GT state.timestamp may be ahead of the LiDAR scan
            # timestamp.  Back-project the GT pose to the LiDAR time.
            raw_pos = np.array([gt_raw.position.x_val, gt_raw.position.y_val,
                                gt_raw.position.z_val])
            raw_ori = np.array([gt_raw.orientation.w_val, gt_raw.orientation.x_val,
                                gt_raw.orientation.y_val, gt_raw.orientation.z_val])

            delta_s = (ts_before - cur_ts) / 1e9  # positive = GT ahead
            if abs(delta_s) > 0.001:
                vel = gt_before.linear_velocity
                v = np.array([vel.x_val, vel.y_val, vel.z_val])
                bp_pos = raw_pos - v * delta_s

                avel = gt_before.angular_velocity
                w = np.array([avel.x_val, avel.y_val, avel.z_val])
                dtheta = w * delta_s
                R_delta = Rotation.from_rotvec(-dtheta)
                R_gt = Rotation.from_quat([raw_ori[1], raw_ori[2], raw_ori[3], raw_ori[0]])
                R_corr = R_delta * R_gt
                q = R_corr.as_quat()  # scipy x,y,z,w
                bp_ori = np.array([q[3], q[0], q[1], q[2]])  # AirSim w,x,y,z
            else:
                bp_pos = raw_pos.copy()
                bp_ori = raw_ori.copy()
                delta_s = 0.0

        else:
            # Step until fresh LiDAR, then SLERP-interpolate GT
            MAX_STEPS = 60
            found = False
            steps = 0
            for _step in range(MAX_STEPS):
                client.simContinueForFrames(1)
                time.sleep(0.01)
                steps = _step + 1

                lidar_data = client.getLidarData()
                cur_ts = float(lidar_data.time_stamp)

                if cur_ts != last_lidar_ts and len(lidar_data.point_cloud) >= 9:
                    gt_after = client.simGetGroundTruthKinematics()
                    state_after = client.getMultirotorState()
                    ts_after = float(state_after.timestamp)
                    last_lidar_ts = cur_ts
                    found = True
                    break

            client.simPause(False)
            if not found:
                continue

            # Raw GT = post-step GT (old approach)
            raw_pos = np.array([gt_after.position.x_val, gt_after.position.y_val,
                                gt_after.position.z_val])
            raw_ori = np.array([gt_after.orientation.w_val, gt_after.orientation.x_val,
                                gt_after.orientation.y_val, gt_after.orientation.z_val])

            # SLERP-interpolated GT (corrected)
            if ts_after != ts_before:
                alpha = np.clip(
                    (cur_ts - ts_before) / (ts_after - ts_before), 0.0, 1.0)
            else:
                alpha = 0.0

            p0 = np.array([gt_before.position.x_val, gt_before.position.y_val,
                           gt_before.position.z_val])
            p1 = raw_pos
            bp_pos = (1 - alpha) * p0 + alpha * p1

            o0 = np.array([gt_before.orientation.w_val, gt_before.orientation.x_val,
                           gt_before.orientation.y_val, gt_before.orientation.z_val])
            o1 = raw_ori
            rots = Rotation.from_quat([
                [o0[1], o0[2], o0[3], o0[0]],
                [o1[1], o1[2], o1[3], o1[0]],
            ])
            slerp_fn = Slerp([0.0, 1.0], rots)
            q_interp = slerp_fn([alpha])[0].as_quat()  # x,y,z,w
            bp_ori = np.array([q_interp[3], q_interp[0],
                               q_interp[1], q_interp[2]])

            delta_s = (ts_before - cur_ts) / 1e9

        raw_yaw = quat_to_yaw(raw_ori)
        bp_yaw = quat_to_yaw(bp_ori)
        n_pts = len(lidar_data.point_cloud) // 3

        pause_results.append((
            time.time(), cur_ts, raw_pos.copy(), raw_ori.copy(), raw_yaw,
            n_pts, steps, bp_pos.copy(), bp_ori.copy(), bp_yaw,
            delta_s * 1000  # delta_ms
        ))
        last_scan_time = time.time()

        # Also snapshot the buffer state for interpolation
        with sensor_lock:
            snap_ts  = list(state_buf_ts)
            snap_pos = list(state_buf_pos)
            snap_ori = list(state_buf_ori)

        buf_pos_interp, bracket = interp_vec(snap_ts, snap_pos, cur_ts)
        buf_ori_interp = interp_quat(snap_ts, snap_ori, cur_ts)

        if buf_pos_interp is not None and buf_ori_interp is not None:
            buf_yaw = quat_to_yaw(buf_ori_interp)
            buffer_results.append((
                cur_ts, buf_pos_interp.copy(), buf_ori_interp.copy(),
                buf_yaw, bracket[0], bracket[1]
            ))
        else:
            buffer_results.append((cur_ts, None, None, None, None, None))

        time.sleep(0.01)


def flight_thread_fn():
    """Fly the L-shaped path using moveByVelocityAsync (like PathFollower)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("  [flight] connected, starting L-path")

    poll_interval = 1.0 / POLL_HZ
    cmd_duration  = poll_interval * 3
    wp_idx = 0
    t0 = time.time()

    while not done.is_set() and wp_idx < len(WAYPOINTS):
        state = client.getMultirotorState()
        p = state.kinematics_estimated.position
        pos = np.array([p.x_val, p.y_val, p.z_val])

        target = WAYPOINTS[wp_idx]
        to_tgt = target - pos
        to_tgt[2] = 0  # keep horizontal
        dist = np.linalg.norm(to_tgt[:2])

        if dist < 1.5 and wp_idx < len(WAYPOINTS) - 1:
            wp_idx += 1
            target = WAYPOINTS[wp_idx]
            to_tgt = target - pos
            to_tgt[2] = 0
            dist = np.linalg.norm(to_tgt[:2])

        if wp_idx == len(WAYPOINTS) - 1 and dist < 1.0:
            break

        direction = to_tgt / max(dist, 1e-6)
        speed = min(SPEED, max(dist * 0.8, 0.5))
        vx, vy = float(direction[0] * speed), float(direction[1] * speed)
        yaw_deg = float(np.degrees(np.arctan2(vy, vx)))

        client.moveByVelocityAsync(
            vx, vy, 0.0,
            duration=cmd_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, yaw_deg),
        )

        if time.time() - t0 > 60:
            print("  [flight] timeout")
            break

        time.sleep(poll_interval)

    print(f"  [flight] path complete (wp_idx={wp_idx})")
    done.set()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("Pause-on-demand vs Buffered-sensor comparison")
    print("="*72)

    # Connect and takeoff
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.reset()
    time.sleep(1)
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, ALTITUDE, SPEED).join()
    time.sleep(1)
    print("At altitude, starting threads.\n")

    # Start all three threads
    t_sensor  = threading.Thread(target=sensor_thread_fn,     name="sensor",  daemon=True)
    t_collect = threading.Thread(target=collection_thread_fn, name="collect", daemon=True)
    t_flight  = threading.Thread(target=flight_thread_fn,     name="flight",  daemon=True)

    t_sensor.start()
    time.sleep(0.5)   # let sensor buffer fill a bit
    t_collect.start()
    t_flight.start()

    # Wait for flight to finish
    t_flight.join(timeout=65)
    done.set()
    time.sleep(0.5)

    print(f"\nCollected {len(pause_results)} scans via pause-on-demand")
    print(f"Interpolated {sum(1 for r in buffer_results if r[1] is not None)} scans from buffer\n")

    # ── Write results to text file ────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        f.write("Pause-on-demand vs Buffered-sensor comparison\n")
        f.write("Three methods compared against buffer-interpolated pose (ground truth reference):\n")
        f.write("  A) Raw GT (at paused instant) — no correction for LiDAR temporal offset\n")
        f.write("  B) Corrected GT (velocity back-projection / SLERP) — NEW fix\n")
        f.write("  C) Buffer interpolation       — 200Hz state polling + SLERP to LiDAR timestamp\n")
        f.write(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scans: {len(pause_results)}\n")
        f.write(f"Scan rate: {SCAN_RATE_HZ} Hz  |  Speed: {SPEED} m/s\n")
        f.write(f"Waypoints: {[w.tolist() for w in WAYPOINTS]}\n\n")

        # Header
        f.write(f"{'#':>3} {'st':>2}  "
                f"{'rawGT_yaw':>10} {'corrGT_yaw':>11} {'buf_yaw':>10}  "
                f"{'raw-buf':>8} {'corr-buf':>9}  "
                f"{'rawGT_x':>8} {'rawGT_y':>8} {'corGT_x':>8} {'corGT_y':>8} {'buf_x':>8} {'buf_y':>8}  "
                f"{'pos_raw':>8} {'pos_cor':>8}  "
                f"{'Δt_ms':>6} {'near':>6} {'far':>6}\n")
        f.write("-" * 180 + "\n")

        raw_yaw_diffs = []
        interp_yaw_diffs = []
        raw_pos_errs = []
        interp_pos_errs = []

        for i, pr in enumerate(pause_results):
            (wall_t, lidar_ts, raw_pos, raw_ori, raw_yaw,
             n_pts, steps, bp_pos, bp_ori, bp_yaw, delta_ms) = pr

            if i < len(buffer_results):
                _, b_pos, b_ori, b_yaw, near_ms, far_ms = buffer_results[i]
            else:
                b_pos = b_ori = b_yaw = near_ms = far_ms = None

            if b_pos is not None and b_yaw is not None:
                # Raw GT vs buffer
                raw_dyaw = (raw_yaw - b_yaw + 180) % 360 - 180
                raw_dpos = np.linalg.norm(raw_pos - b_pos)
                raw_yaw_diffs.append(abs(raw_dyaw))
                raw_pos_errs.append(raw_dpos)

                # Corrected GT (back-projected / SLERP) vs buffer
                bp_dyaw = (bp_yaw - b_yaw + 180) % 360 - 180
                bp_dpos = np.linalg.norm(bp_pos - b_pos)
                interp_yaw_diffs.append(abs(bp_dyaw))
                interp_pos_errs.append(bp_dpos)

                f.write(f"{i:>3} {steps:>2}  "
                        f"{raw_yaw:>10.3f} {bp_yaw:>11.3f} {b_yaw:>10.3f}  "
                        f"{raw_dyaw:>+8.3f} {bp_dyaw:>+9.3f}  "
                        f"{raw_pos[0]:>8.3f} {raw_pos[1]:>8.3f} "
                        f"{bp_pos[0]:>8.3f} {bp_pos[1]:>8.3f} "
                        f"{b_pos[0]:>8.3f} {b_pos[1]:>8.3f}  "
                        f"{raw_dpos*100:>7.1f}cm {bp_dpos*100:>7.1f}cm  "
                        f"{delta_ms:>6.1f} {near_ms:>6.1f} {far_ms:>6.1f}\n")
            else:
                f.write(f"{i:>3} {steps:>2}  "
                        f"{raw_yaw:>10.3f} {bp_yaw:>11.3f} {'N/A':>10}  "
                        f"{'N/A':>8} {'N/A':>9}  "
                        f"{raw_pos[0]:>8.3f} {raw_pos[1]:>8.3f} "
                        f"{bp_pos[0]:>8.3f} {bp_pos[1]:>8.3f} "
                        f"{'N/A':>8} {'N/A':>8}  "
                        f"{'N/A':>8} {'N/A':>8}  "
                        f"{delta_ms:>6.1f} {'N/A':>6} {'N/A':>6}\n")

        f.write("\n")
        f.write("="*72 + "\n")
        f.write("SUMMARY\n")
        f.write("="*72 + "\n\n")

        f.write("  A) Raw GT (at paused instant) vs Buffer:\n")
        if raw_yaw_diffs:
            f.write(f"    Yaw:  mean={np.mean(raw_yaw_diffs):.4f}°  "
                    f"max={np.max(raw_yaw_diffs):.4f}°  "
                    f"std={np.std(raw_yaw_diffs):.4f}°\n")
            f.write(f"    Pos:  mean={np.mean(raw_pos_errs)*100:.2f}cm  "
                    f"max={np.max(raw_pos_errs)*100:.2f}cm\n")

        f.write("\n  B) Corrected GT (velocity back-proj / SLERP) vs Buffer:\n")
        if interp_yaw_diffs:
            f.write(f"    Yaw:  mean={np.mean(interp_yaw_diffs):.4f}°  "
                    f"max={np.max(interp_yaw_diffs):.4f}°  "
                    f"std={np.std(interp_yaw_diffs):.4f}°\n")
            f.write(f"    Pos:  mean={np.mean(interp_pos_errs)*100:.2f}cm  "
                    f"max={np.max(interp_pos_errs)*100:.2f}cm\n")

        # Improvement ratio
        if raw_yaw_diffs and interp_yaw_diffs:
            yaw_improve = np.mean(raw_yaw_diffs) / max(np.mean(interp_yaw_diffs), 1e-6)
            pos_improve = np.mean(raw_pos_errs) / max(np.mean(interp_pos_errs), 1e-6)
            f.write(f"\n  Improvement: {yaw_improve:.1f}× yaw, {pos_improve:.1f}× position\n")

            f.write(f"\n  Verdict: ")
            max_iy = np.max(interp_yaw_diffs)
            max_ip = np.max(interp_pos_errs)
            if max_iy < 1.0 and max_ip < 0.05:
                f.write("EXCELLENT — interpolated GT closely matches buffer.\n")
            elif max_iy < 3.0:
                f.write("GOOD — small residual differences.\n")
            else:
                f.write("NEEDS WORK — still significant disagreement.\n")

    print(f"Results written to: {OUTPUT_FILE}")

    # Print summary to console
    if raw_yaw_diffs and interp_yaw_diffs:
        print(f"\n  Raw GT vs Buffer:       yaw mean={np.mean(raw_yaw_diffs):.3f}° max={np.max(raw_yaw_diffs):.3f}°  pos max={np.max(raw_pos_errs)*100:.1f}cm")
        print(f"  Corrected GT vs Buffer: yaw mean={np.mean(interp_yaw_diffs):.3f}° max={np.max(interp_yaw_diffs):.3f}°  pos max={np.max(interp_pos_errs)*100:.1f}cm")
        yaw_improve = np.mean(raw_yaw_diffs) / max(np.mean(interp_yaw_diffs), 1e-6)
        print(f"  Improvement: {yaw_improve:.1f}× yaw accuracy")

    # Land
    print("\nLanding...")
    client.simPause(False)
    time.sleep(0.5)
    client.landAsync().join()
    print("Done.")


if __name__ == "__main__":
    main()
