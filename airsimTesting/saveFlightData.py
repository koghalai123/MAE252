"""Record AirSim LiDAR + pose while flying waypoints.

Saves one .npz per frame into SAVE_DIR.
Uses Viewer3D from sensorFeed.py for live display.

Pose is sampled immediately BEFORE and AFTER each getLidarData() call,
then linearly interpolated (SLERP for orientation) to the LiDAR
timestamp.  This brackets the true scan time tightly, eliminating the
40-230 ms timing mismatch seen when using a single getMultirotorState().
"""

import cosysairsim as airsim
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation, Slerp
from sensorFeed import Viewer3D

# --- Config ---
SAVE_DIR = "/home/koghalai/MAE252/airsimTesting/flight_recordings/"
MAX_SCAN_HZ = 4
MIN_SCAN_INTERVAL = 1.0 / MAX_SCAN_HZ


# ---------- pose interpolation helpers ----------
def _extract_pose(state):
    """Return (timestamp_ns, position[3], orientation[4 wxyz]) from state."""
    ts = float(state.timestamp)
    p  = state.kinematics_estimated.position
    o  = state.kinematics_estimated.orientation
    pos  = np.array([p.x_val, p.y_val, p.z_val])
    quat = np.array([o.w_val, o.x_val, o.y_val, o.z_val])  # w,x,y,z
    return ts, pos, quat


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


# --- Connect ---
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(0.5)

# --- Initial data + visualizer ---
lidarData = client.getLidarData()
points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))

tvis = Viewer3D()
tvis.start(initial_points=points)

# --- Output directory ---
out_dir = os.path.join(SAVE_DIR, f"flight_{int(time.time())}")
os.makedirs(out_dir, exist_ok=True)
print(f"Saving frames to {out_dir}/  (max {MAX_SCAN_HZ} Hz)")

# --- Takeoff ---
print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# --- Fly waypoints and record ---
waypoints = [
    (0, 0, -15),
    (25, 0, -15),
    (25, -25, -15),
    (0, -25, -15),
    (-10, 0, -5),
]

frame = 0
last_save_time = 0.0

timing_log_path = os.path.join(out_dir, "timing_log.txt")
timing_log = open(timing_log_path, "w")
timing_log.write(f"{'frame':>5}  {'before_ts':>22}  {'lidar_ts':>22}  {'after_ts':>22}  "
                 f"{'bracket_ms':>12}  {'t_interp':>10}\n")
timing_log.write("-" * 120 + "\n")
for i, waypoint in enumerate(waypoints):
    print(f"Moving to waypoint {i+1}: {waypoint}...")
    future = client.moveToPositionAsync(
        waypoint[0], waypoint[1], waypoint[2], velocity=2
    )
    while not future._set_flag:
        # Sample state BEFORE LiDAR
        state_before = client.getMultirotorState()

        # Get sensor data
        lidarData = client.getLidarData()
        imuData = client.getImuData()
        gpsData = client.getGpsData()

        # Sample state AFTER LiDAR
        state_after = client.getMultirotorState()

        points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))
        tvis.update(points)

        now = time.time()
        if now - last_save_time >= MIN_SCAN_INTERVAL:
            last_save_time = now

            # Extract before/after poses
            ts_before, pos_before, ori_before = _extract_pose(state_before)
            ts_after,  pos_after,  ori_after  = _extract_pose(state_after)

            # LiDAR timestamp
            lidar_ts = float(lidarData.time_stamp)

            # Interpolate pose to LiDAR timestamp
            interp_pos, interp_ori = interpolate_pose(
                ts_before, pos_before, ori_before,
                ts_after,  pos_after,  ori_after,
                lidar_ts,
            )

            # LiDAR sensor mount pose (local offset on vehicle)
            lpos = lidarData.pose.position
            lori = lidarData.pose.orientation

            # GPS
            gp = gpsData.gnss.geo_point

            # Log timing
            bracket_ms = (ts_after - ts_before) / 1e6
            dt = ts_after - ts_before
            t_interp = (lidar_ts - ts_before) / dt if dt > 0 else 0.0
            timing_log.write(f"{frame:05d}  {ts_before:>22.0f}  {lidar_ts:>22.0f}  {ts_after:>22.0f}  "
                             f"{bracket_ms:>+12.3f}  {t_interp:>10.4f}\n")
            timing_log.flush()

            np.savez(
                os.path.join(out_dir, f"frame_{frame:05d}.npz"),
                points=points,
                # Timestamps
                timestamp=np.array(lidar_ts),
                state_before_ts=np.array(ts_before),
                state_after_ts=np.array(ts_after),
                # Interpolated vehicle world pose (time-matched to LiDAR)
                position=interp_pos,
                orientation=interp_ori,
                # LiDAR sensor mount offset
                lidar_position=np.array([lpos.x_val, lpos.y_val, lpos.z_val]),
                lidar_orientation=np.array([lori.w_val, lori.x_val, lori.y_val, lori.z_val]),
                # IMU
                imu_angular_vel=np.array([imuData.angular_velocity.x_val,
                                          imuData.angular_velocity.y_val,
                                          imuData.angular_velocity.z_val]),
                imu_linear_acc=np.array([imuData.linear_acceleration.x_val,
                                         imuData.linear_acceleration.y_val,
                                         imuData.linear_acceleration.z_val]),
                # GPS
                gps=np.array([gp.latitude, gp.longitude, gp.altitude]),
            )
            frame += 1

timing_log.close()
print(f"\nDone - saved {frame} frames to {out_dir}/")
print(f"Timing log: {timing_log_path}")
tvis.stop()
