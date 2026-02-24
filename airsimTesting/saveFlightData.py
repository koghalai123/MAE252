"""Record AirSim LiDAR + pose while flying waypoints.

Saves one .npz per LiDAR frame into SAVE_DIR.
Uses Viewer3D from sensorFeed.py for live display.

State, IMU, and GPS are sampled as fast as possible in the inner loop
and buffered in memory.  LiDAR is grabbed only at MAX_SCAN_HZ.
Each LiDAR frame sits in a queue and is saved continuously once
SAVE_DELAY_SCANS more LiDAR intervals of high-rate data have arrived
after its timestamp, ensuring every frame has sensor data on both sides
for tight interpolation.
"""

import cosysairsim as airsim
import numpy as np
import time
import os
import bisect
from scipy.spatial.transform import Rotation, Slerp
from sensorFeed import Viewer3D

# ── Config ──────────────────────────────────────────────────────────
SAVE_DIR = "/home/koghalai/MAE252/airsimTesting/flight_recordings/"
MAX_SCAN_HZ = 4                      # LiDAR capture rate
MIN_SCAN_INTERVAL = 1.0 / MAX_SCAN_HZ
SAVE_DELAY_SCANS = 2                 # wait this many extra LiDAR intervals
                                     # before saving, so high-rate buffers
                                     # extend well past the LiDAR timestamp


# ── Pose interpolation helpers ──────────────────────────────────────
def _extract_pose(state):
    """Return (timestamp_ns, position[3], orientation[4 wxyz]) from state."""
    ts = float(state.timestamp)
    p  = state.kinematics_estimated.position
    o  = state.kinematics_estimated.orientation
    pos  = np.array([p.x_val, p.y_val, p.z_val])
    quat = np.array([o.w_val, o.x_val, o.y_val, o.z_val])  # w,x,y,z
    return ts, pos, quat


def _extract_imu(imu):
    """Return (timestamp_ns, angular_vel[3], linear_acc[3])."""
    ts = float(imu.time_stamp)
    ang = np.array([imu.angular_velocity.x_val,
                    imu.angular_velocity.y_val,
                    imu.angular_velocity.z_val])
    acc = np.array([imu.linear_acceleration.x_val,
                    imu.linear_acceleration.y_val,
                    imu.linear_acceleration.z_val])
    return ts, ang, acc


def _extract_gps(gps):
    """Return (timestamp_ns, np.array([lat, lon, alt]))."""
    ts = float(gps.time_stamp)
    gp = gps.gnss.geo_point
    return ts, np.array([gp.latitude, gp.longitude, gp.altitude])


def interpolate_pose(ts_before, pos_before, ori_before,
                     ts_after,  pos_after,  ori_after,
                     target_ts):
    """Interpolate between a before/after pose pair at *target_ts*.

    Returns (position[3], orientation[4 wxyz]).
    """
    dt = ts_after - ts_before
    if dt <= 0:
        return pos_before.copy(), ori_before.copy()

    t = np.clip((target_ts - ts_before) / dt, 0.0, 1.0)

    # Linear position
    pos = (1 - t) * pos_before + t * pos_after

    # SLERP orientation  (scipy wants x,y,z,w)
    rots = Rotation.from_quat([
        [ori_before[1], ori_before[2], ori_before[3], ori_before[0]],
        [ori_after[1],  ori_after[2],  ori_after[3],  ori_after[0]],
    ])
    slerp = Slerp([0.0, 1.0], rots)
    q_scipy = slerp([t])[0].as_quat()            # x,y,z,w
    quat = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # w,x,y,z

    return pos, quat


def interpolate_vec3(ts_a, vec_a, ts_b, vec_b, target_ts):
    """Linearly interpolate between two 3-vectors at target_ts."""
    dt = ts_b - ts_a
    if dt <= 0:
        return vec_a.copy()
    t = np.clip((target_ts - ts_a) / dt, 0.0, 1.0)
    return (1 - t) * vec_a + t * vec_b


def _find_bracket(timestamps, target):
    """Find indices (i_before, i_after) that bracket *target* in a sorted list.

    Returns the two closest indices.  If target is outside the range,
    the nearest edge is duplicated.
    """
    idx = bisect.bisect_right(timestamps, target)
    i_before = max(idx - 1, 0)
    i_after  = min(idx, len(timestamps) - 1)
    return i_before, i_after


# ── Connect ─────────────────────────────────────────────────────────
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(0.5)

# ── Initial data + visualizer ───────────────────────────────────────
lidarData = client.getLidarData()
points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))

tvis = Viewer3D()
tvis.start(initial_points=points)

# ── Output directory ────────────────────────────────────────────────
out_dir = os.path.join(SAVE_DIR, f"flight_{int(time.time())}")
os.makedirs(out_dir, exist_ok=True)
print(f"Saving frames to {out_dir}/  (LiDAR @ {MAX_SCAN_HZ} Hz, save delay={SAVE_DELAY_SCANS} scans)")

# ── Takeoff ─────────────────────────────────────────────────────────
print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# ── Waypoints ───────────────────────────────────────────────────────
waypoints = [
    (0, 0, -15),
    (25, 0, -15),
    (25, -25, -15),
    (0, -25, -15),
    (-10, 0, -5),
]

# ── In-memory buffers ──────────────────────────────────────────────
# High-rate sensor buffers (state, IMU, GPS)
state_buf_ts  = []   # float timestamps (ns)
state_buf_pos = []   # np.array[3]
state_buf_ori = []   # np.array[4] wxyz

imu_buf_ts  = []
imu_buf_ang = []     # angular velocity
imu_buf_acc = []     # linear acceleration

gps_buf_ts  = []
gps_buf_geo = []     # [lat, lon, alt]

# Pending LiDAR frames (FIFO queue)
lidar_queue = []     # list of dicts: {ts, points, lpos, lori}

frame_counter = 0    # global saved-frame number
last_lidar_time = 0.0  # wall-clock gate for LiDAR captures
SAVE_DELAY_NS = SAVE_DELAY_SCANS * MIN_SCAN_INTERVAL * 1e9  # delay in ns


def _save_one_frame(lf, frame_num):
    """Interpolate sensors to a single LiDAR timestamp and write .npz."""
    lidar_ts = lf["ts"]

    # ── State interpolation ────────────────────────────────────
    ia, ib = _find_bracket(state_buf_ts, lidar_ts)
    interp_pos, interp_ori = interpolate_pose(
        state_buf_ts[ia], state_buf_pos[ia], state_buf_ori[ia],
        state_buf_ts[ib], state_buf_pos[ib], state_buf_ori[ib],
        lidar_ts,
    )

    # ── IMU interpolation ──────────────────────────────────────
    ia_i, ib_i = _find_bracket(imu_buf_ts, lidar_ts)
    interp_ang = interpolate_vec3(imu_buf_ts[ia_i], imu_buf_ang[ia_i],
                                  imu_buf_ts[ib_i], imu_buf_ang[ib_i],
                                  lidar_ts)
    interp_acc = interpolate_vec3(imu_buf_ts[ia_i], imu_buf_acc[ia_i],
                                  imu_buf_ts[ib_i], imu_buf_acc[ib_i],
                                  lidar_ts)

    # ── GPS interpolation ──────────────────────────────────────
    ia_g, ib_g = _find_bracket(gps_buf_ts, lidar_ts)
    interp_gps = interpolate_vec3(gps_buf_ts[ia_g], gps_buf_geo[ia_g],
                                  gps_buf_ts[ib_g], gps_buf_geo[ib_g],
                                  lidar_ts)

    # ── Compute near/far gaps for each sensor ─────────────────
    state_near_ms = min(abs(lidar_ts - state_buf_ts[ia]),
                        abs(lidar_ts - state_buf_ts[ib])) / 1e6
    state_far_ms  = max(abs(lidar_ts - state_buf_ts[ia]),
                        abs(lidar_ts - state_buf_ts[ib])) / 1e6

    imu_near_ms = min(abs(lidar_ts - imu_buf_ts[ia_i]),
                      abs(lidar_ts - imu_buf_ts[ib_i])) / 1e6
    imu_far_ms  = max(abs(lidar_ts - imu_buf_ts[ia_i]),
                      abs(lidar_ts - imu_buf_ts[ib_i])) / 1e6

    gps_near_ms = min(abs(lidar_ts - gps_buf_ts[ia_g]),
                      abs(lidar_ts - gps_buf_ts[ib_g])) / 1e6
    gps_far_ms  = max(abs(lidar_ts - gps_buf_ts[ia_g]),
                      abs(lidar_ts - gps_buf_ts[ib_g])) / 1e6

    # ── Print timing gaps (nearest / furthest bracket sample) ─
    print(f"  Frame {frame_num:05d}  "
          f"| state: {state_near_ms:5.1f}/{state_far_ms:5.1f} ms  "
          f"| IMU: {imu_near_ms:5.1f}/{imu_far_ms:5.1f} ms  "
          f"| GPS: {gps_near_ms:5.1f}/{gps_far_ms:5.1f} ms")

    # ── Save .npz ─────────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, f"frame_{frame_num:05d}.npz"),
        points=lf["points"],
        # Timestamps
        timestamp=np.array(lidar_ts),
        state_before_ts=np.array(state_buf_ts[ia]),
        state_after_ts=np.array(state_buf_ts[ib]),
        # Interpolated vehicle world pose (time-matched to LiDAR)
        position=interp_pos,
        orientation=interp_ori,
        # LiDAR sensor mount offset
        lidar_position=lf["lpos"],
        lidar_orientation=lf["lori"],
        # Interpolated IMU
        imu_angular_vel=interp_ang,
        imu_linear_acc=interp_acc,
        # Interpolated GPS
        gps=interp_gps,
    )


def _drain_ready_frames():
    """Save queued LiDAR frames whose timestamps now have enough data on both sides.

    A frame is 'ready' when the latest high-rate timestamp exceeds
    the LiDAR timestamp by at least SAVE_DELAY_NS.
    Also trims old buffer entries that are no longer needed.
    """
    global frame_counter, lidar_queue
    global state_buf_ts, state_buf_pos, state_buf_ori
    global imu_buf_ts, imu_buf_ang, imu_buf_acc
    global gps_buf_ts, gps_buf_geo

    if not lidar_queue or not state_buf_ts:
        return

    latest_ts = state_buf_ts[-1]

    while lidar_queue and (latest_ts - lidar_queue[0]["ts"]) >= SAVE_DELAY_NS:
        lf = lidar_queue.pop(0)
        _save_one_frame(lf, frame_counter)
        frame_counter += 1

        # Trim buffers: keep everything from slightly before the
        # next pending frame (or the current latest if queue empty)
        if lidar_queue:
            trim_ts = lidar_queue[0]["ts"]
        else:
            trim_ts = latest_ts
        cutoff = trim_ts - SAVE_DELAY_NS
        cut_idx = bisect.bisect_left(state_buf_ts, cutoff)
        if cut_idx > 0:
            state_buf_ts  = state_buf_ts[cut_idx:]
            state_buf_pos = state_buf_pos[cut_idx:]
            state_buf_ori = state_buf_ori[cut_idx:]
        cut_idx = bisect.bisect_left(imu_buf_ts, cutoff)
        if cut_idx > 0:
            imu_buf_ts  = imu_buf_ts[cut_idx:]
            imu_buf_ang = imu_buf_ang[cut_idx:]
            imu_buf_acc = imu_buf_acc[cut_idx:]
        cut_idx = bisect.bisect_left(gps_buf_ts, cutoff)
        if cut_idx > 0:
            gps_buf_ts  = gps_buf_ts[cut_idx:]
            gps_buf_geo = gps_buf_geo[cut_idx:]


def _sample_high_rate():
    """Sample state, IMU, GPS once and append to buffers."""
    state = client.getMultirotorState()
    imuData = client.getImuData()
    gpsData = client.getGpsData()

    ts_s, pos_s, ori_s = _extract_pose(state)
    state_buf_ts.append(ts_s)
    state_buf_pos.append(pos_s)
    state_buf_ori.append(ori_s)

    ts_i, ang_i, acc_i = _extract_imu(imuData)
    imu_buf_ts.append(ts_i)
    imu_buf_ang.append(ang_i)
    imu_buf_acc.append(acc_i)

    ts_g, geo_g = _extract_gps(gpsData)
    gps_buf_ts.append(ts_g)
    gps_buf_geo.append(geo_g)


# ── Main loop ───────────────────────────────────────────────────────
for i, waypoint in enumerate(waypoints):
    print(f"Moving to waypoint {i+1}: {waypoint}...")
    future = client.moveToPositionAsync(
        waypoint[0], waypoint[1], waypoint[2], velocity=2
    )
    while not future._set_flag:
        # ── Always: high-rate state / IMU / GPS ────────────────────
        _sample_high_rate()

        # ── Rate-limited: LiDAR capture ────────────────────────────
        now = time.time()
        if now - last_lidar_time >= MIN_SCAN_INTERVAL:
            last_lidar_time = now

            lidarData = client.getLidarData()
            pts = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))
            tvis.update(pts)

            lpos = lidarData.pose.position
            lori = lidarData.pose.orientation
            lidar_queue.append({
                "ts":     float(lidarData.time_stamp),
                "points": pts,
                "lpos":   np.array([lpos.x_val, lpos.y_val, lpos.z_val]),
                "lori":   np.array([lori.w_val, lori.x_val, lori.y_val, lori.z_val]),
            })

        # ── Save any LiDAR frames that now have enough data ───────
        _drain_ready_frames()

# ── Flush remaining queued frames ───────────────────────────────────
if lidar_queue:
    print(f"  [final] collecting extra samples for {len(lidar_queue)} remaining frame(s)...")
    # Keep sampling high-rate data until we have enough to cover the last frame
    deadline = time.time() + SAVE_DELAY_SCANS * MIN_SCAN_INTERVAL + 0.2
    while time.time() < deadline:
        _sample_high_rate()
    # Force-save anything still queued
    for lf in lidar_queue:
        _save_one_frame(lf, frame_counter)
        frame_counter += 1
    lidar_queue = []

print(f"\nDone - saved {frame_counter} frames to {out_dir}/")
tvis.stop()
