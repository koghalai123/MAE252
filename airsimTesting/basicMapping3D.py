"""3D Mapping — stitch LiDAR scans into a global point cloud.

Modes
-----
LIVE   : fly waypoints in Unreal / AirSim, accumulate scans in real time.
REPLAY : load a flight_recordings/ directory of .npz frames saved by
         saveFlightData.py and rebuild the map offline.

Toggle the mode with USE_RECORDING / RECORDING_DIR below.
Uses Viewer3D (multiprocessing Open3D) from sensorFeed.py.

NOTE: With ExternalLocal=false in AirSim settings, LiDAR points are
returned in the sensor's local frame.  We transform them to world
coordinates using: (1) the sensor mount offset, then (2) the vehicle
world pose from getMultirotorState().
"""

import cosysairsim as airsim
import numpy as np
import time
import os
import glob
from scipy.spatial.transform import Rotation
from sensorFeed import Viewer3D

USE_RECORDING = True
RECORDING_DIR = "/home/koghalai/MAE252/airsimTesting/flight_recordings/"
#   If USE_RECORDING is True, set RECORDING_DIR to a specific flight
#   folder, e.g. ".../flight_recordings/flight_1234567890"
#   or to the parent directory to auto-pick the latest flight.


MAX_SCAN_HZ    = 10         # live-mode scan rate limit

WAYPOINTS = [
    (0, 0, -10),
    (10, 0, -10),
    (10, 10, -10),
    (0, 10, -10),
    (0, 0, -10),
    (5, 5, -15),
    (0, 0, -10),
]
FLY_VELOCITY = 3  # m/s

def filter_valid_points(points):
    """Remove zero / invalid LiDAR returns (all-zero rows)."""
    mask = np.any(points != 0, axis=1)
    return points[mask]


def transform_points(points, position, orientation):
    """Transform sensor-frame points into the world frame.

    Uses the LiDAR sensor's own pose (position + orientation) which is
    time-matched to the scan, avoiding the mismatch that occurs when the
    vehicle state is queried separately.

    Parameters
    ----------
    points      : (N, 3) float array in sensor local frame
    position    : (3,)  [x, y, z]  sensor world position, NED metres
    orientation : (4,)  [w, x, y, z] sensor world orientation quaternion
    """
    rot = Rotation.from_quat([orientation[1], orientation[2],
                              orientation[3], orientation[0]])
    R = rot.as_matrix()
    return (R @ points.T).T + position


def fit_plane(points, label=""):
    """Fit a plane to *points* (N,3) via SVD and print its slope.

    Returns (normal, slope_x, slope_y) or None if too few points.
    """
    if len(points) < 10:
        return None
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] != 0:
        slope_x = -normal[0] / normal[2]
        slope_y = -normal[1] / normal[2]
    else:
        slope_x = slope_y = float('nan')
    print(f"  {label}plane slope  x: {slope_x:+.4f}  y: {slope_y:+.4f}  "
          f"normal: [{normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f}]")
    return normal, slope_x, slope_y


def resolve_recording_dir(path):
    """If *path* is the parent recordings folder, pick the latest flight_* subfolder."""
    if os.path.isfile(os.path.join(path, "frame_00000.npz")):
        return path  # already a flight folder
    flights = sorted(glob.glob(os.path.join(path, "flight_*")))
    if flights:
        return flights[-1]
    return path


def run_replay(recording_dir):
    recording_dir = resolve_recording_dir(recording_dir)
    frames = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    if not frames:
        print(f"No frame_*.npz files found in {recording_dir}")
        return

    print(f"Replaying {len(frames)} frames from {recording_dir}")

    viewer = Viewer3D()
    global_map = np.zeros((0, 3), dtype=np.float32)

    # Timing log
    timing_log_path = os.path.join(recording_dir, "timing_log.txt")
    timing_log = open(timing_log_path, "w")
    timing_log.write(f"{'frame':>5}  {'lidar_ts':>22}  {'imu_ts':>22}  {'gps_ts':>22}  {'state_ts':>22}  "
                     f"{'imu-lidar_ms':>14}  {'gps-lidar_ms':>14}  {'state-lidar_ms':>16}\n")
    timing_log.write("-" * 160 + "\n")

    for i, path in enumerate(frames):
        data = np.load(path)
        points = filter_valid_points(data["points"])

        if len(points) == 0:
            continue

        # Log timestamp diffs if all timestamps are present
        if all(k in data.files for k in ["timestamp", "imu_timestamp", "gps_timestamp", "state_timestamp"]):
            lidar_ts = float(data["timestamp"])
            imu_ts   = float(data["imu_timestamp"])
            gps_ts   = float(data["gps_timestamp"])
            state_ts = float(data["state_timestamp"])
            imu_diff_ms   = (imu_ts   - lidar_ts) / 1e6
            gps_diff_ms   = (gps_ts   - lidar_ts) / 1e6
            state_diff_ms = (state_ts - lidar_ts) / 1e6
            timing_log.write(f"{i:05d}  {lidar_ts:>22.0f}  {imu_ts:>22.0f}  {gps_ts:>22.0f}  {state_ts:>22.0f}  "
                             f"{imu_diff_ms:>+14.3f}  {gps_diff_ms:>+14.3f}  {state_diff_ms:>+16.3f}\n")
            timing_log.flush()

        if len(points) == 0:
            continue

        # Points are in sensor local frame.
        # Step 1: apply LiDAR mount offset (sensor → vehicle body frame).
        lidar_pos = data["lidar_position"] if "lidar_position" in data.files else np.zeros(3)
        lidar_ori = data["lidar_orientation"] if "lidar_orientation" in data.files else np.array([1,0,0,0], dtype=float)
        body_pts = transform_points(points, lidar_pos, lidar_ori)

        # Step 2: apply vehicle world pose (body → world frame).
        position    = data["position"] if "position" in data.files else np.zeros(3)
        orientation = data["orientation"] if "orientation" in data.files else np.array([1,0,0,0], dtype=float)
        world_pts = transform_points(body_pts, position, orientation)

        # Per-scan ground-plane diagnostic
        fit_plane(world_pts, label=f"  frame {i:03d}")

        global_map = np.vstack([global_map, world_pts])

        # Visualize
        if i == 0:
            viewer.start(initial_points=global_map)
        else:
            viewer.update(global_map)

        # Status line
        extra = ""
        if "gps" in data.files:
            gps = data["gps"]
            extra = f" | GPS ({gps[0]:.6f}, {gps[1]:.6f}, {gps[2]:.1f})"
        print(f"\n  frame {i+1}/{len(frames)} | map pts {len(global_map):,}{extra}",
              end="", flush=True)

    viewer.update(global_map)
    timing_log.close()
    print(f"\nTiming log written to: {timing_log_path}")

    print(f"\n\nReplay complete â {len(global_map):,} points in map.")
    print("Close the Open3D window or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


def run_live():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    # Grab an initial scan so the viewer opens with something
    lidar = client.getLidarData()
    init_pts = np.array(lidar.point_cloud, dtype=np.float32).reshape((-1, 3))

    viewer = Viewer3D()
    viewer.start(initial_points=init_pts)
    global_map = np.zeros((0, 3), dtype=np.float32)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)

    min_interval = 1.0 / MAX_SCAN_HZ
    last_scan = 0.0
    scan_count = 0

    try:
        for i, wp in enumerate(WAYPOINTS):
            print(f"\nWaypoint {i+1}/{len(WAYPOINTS)}: {wp}")
            future = client.moveToPositionAsync(wp[0], wp[1], wp[2],
                                                velocity=FLY_VELOCITY)

            while not future._set_flag:
                now = time.time()
                if now - last_scan < min_interval:
                    time.sleep(0.01)
                    continue
                last_scan = now

                lidar = client.getLidarData()
                points = np.array(lidar.point_cloud, dtype=np.float32).reshape((-1, 3))
                points = filter_valid_points(points)
                if len(points) == 0:
                    continue

                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                ori = state.kinematics_estimated.orientation
                position    = np.array([pos.x_val, pos.y_val, pos.z_val])
                orientation = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])

                lpos = lidar.pose.position
                lori = lidar.pose.orientation
                lidar_position    = np.array([lpos.x_val, lpos.y_val, lpos.z_val])
                lidar_orientation = np.array([lori.w_val, lori.x_val, lori.y_val, lori.z_val])

                # sensor local → body → world
                body_pts  = transform_points(points, lidar_position, lidar_orientation)
                world_pts = transform_points(body_pts, position, orientation)
                global_map = np.vstack([global_map, world_pts])

                viewer.update(global_map)
                scan_count += 1
                print(f"\n  scans {scan_count} | map pts {len(global_map):,}",
                      end="", flush=True)

        print(f"\n\nMapping complete â {len(global_map):,} points, {scan_count} scans.")
        print("Close the Open3D window or Ctrl+C to exit.")
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        viewer.stop()
        print("Done.")


if __name__ == "__main__":
    if USE_RECORDING:
        run_replay(RECORDING_DIR)
    else:
        run_live()
