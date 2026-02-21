"""Record AirSim LiDAR + pose while flying waypoints.

Saves one .npz per frame into a timestamped directory.
Uses Viewer3D from sensorFeed.py for live display.
"""

import cosysairsim as airsim
import numpy as np
import time
import os
from sensorFeed import Viewer3D

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
MAX_SCAN_HZ = 4  # maximum recording frequency (Hz)
MIN_SCAN_INTERVAL = 1.0 / MAX_SCAN_HZ

out_dir = f"flight_data_{int(time.time())}"
os.makedirs(out_dir, exist_ok=True)
print(f"Saving frames to {out_dir}/  (max {MAX_SCAN_HZ} Hz)")

# --- Takeoff ---
print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# --- Fly waypoints and record ---
waypoints = [
    (0, 0, -10),
    (20, 0, -10),
    (0, 0, -10),
]

frame = 0
last_save_time = 0.0
for i, waypoint in enumerate(waypoints):
    print(f"Moving to waypoint {i+1}: {waypoint}...")
    future = client.moveToPositionAsync(
        waypoint[0], waypoint[1], waypoint[2], velocity=5
    )
    while not future._set_flag:
        lidarData = client.getLidarData()
        imuData = client.getImuData()
        state = client.getMultirotorState()

        points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))
        tvis.update(points)

        now = time.time()
        if now - last_save_time >= MIN_SCAN_INTERVAL:
            last_save_time = now
            pos = state.kinematics_estimated.position
            ori = state.kinematics_estimated.orientation
            np.savez(
                os.path.join(out_dir, f"frame_{frame:05d}.npz"),
                points=points,
                timestamp=np.array(float(lidarData.time_stamp)),
                position=np.array([pos.x_val, pos.y_val, pos.z_val]),
                orientation=np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val]),
                imu_angular_vel=np.array([imuData.angular_velocity.x_val,
                                          imuData.angular_velocity.y_val,
                                          imuData.angular_velocity.z_val]),
                imu_linear_acc=np.array([imuData.linear_acceleration.x_val,
                                         imuData.linear_acceleration.y_val,
                                         imuData.linear_acceleration.z_val]),
            )
            frame += 1

print(f"Done — saved {frame} frames to {out_dir}/")
tvis.stop()
