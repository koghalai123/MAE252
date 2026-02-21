import cosysairsim as airsim
import cv2
import numpy as np
import open3d as o3d
import time
import multiprocessing
import multiprocessing.shared_memory
from collections import deque
from scipy.spatial.transform import Rotation, Slerp


def _vis_process(shm_name, shape, dtype_str, lock, update_event, stop_event):
    """Visualizer loop running in a separate process."""
    import numpy as _np
    import open3d as _o3d

    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    buf = _np.ndarray(shape, dtype=dtype_str, buffer=shm.buf)

    pcd = _o3d.geometry.PointCloud()
    pcd.points = _o3d.utility.Vector3dVector(buf.copy())

    vis = _o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    while not stop_event.is_set():
        if update_event.is_set():
            with lock:
                pts = buf.copy()
            update_event.clear()
            pcd.points = _o3d.utility.Vector3dVector(pts)
            vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()
    shm.close()


class Viewer3D:
    """Open3D visualizer in a separate process — stays responsive at breakpoints."""

    MAX_POINTS = 200_000  # pre-allocated shared buffer size

    def __init__(self):
        self._shm = None
        self._buf = None
        self._proc = None
        self._lock = multiprocessing.Lock()
        self._update_event = multiprocessing.Event()
        self._stop_event = multiprocessing.Event()
        self._shape = (self.MAX_POINTS, 3)
        self._dtype = np.dtype(np.float64)

    def start(self, initial_points=None):
        nbytes = int(np.prod(self._shape) * self._dtype.itemsize)
        self._shm = multiprocessing.shared_memory.SharedMemory(create=True, size=nbytes)
        self._buf = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
        self._buf[:] = 0.0
        if initial_points is not None:
            n = min(len(initial_points), self.MAX_POINTS)
            self._buf[:n] = initial_points[:n].astype(self._dtype)
        self._update_event.set()
        self._proc = multiprocessing.Process(
            target=_vis_process,
            args=(self._shm.name, self._shape, self._dtype.str,
                  self._lock, self._update_event, self._stop_event),
            daemon=True,
        )
        self._proc.start()

    def update(self, points):
        n = min(len(points), self.MAX_POINTS)
        with self._lock:
            self._buf[:] = 0.0
            self._buf[:n] = points[:n].astype(self._dtype)
        self._update_event.set()

    def stop(self):
        self._stop_event.set()
        if self._proc is not None:
            self._proc.join(timeout=3)
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()



def get_lidar_data(client):
    """Get lidar point cloud, timestamp, and sensor pose."""
    try:
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            return None, None, None, None
    except Exception:
        return None, None, None, None

    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape((-1, 3))

    # Normalize timestamp to seconds
    ts = None
    if hasattr(lidar_data, 'time_stamp') and lidar_data.time_stamp:
        tsf = float(lidar_data.time_stamp)
        if tsf > 1e18:   ts = tsf / 1e9
        elif tsf > 1e15: ts = tsf / 1e6
        elif tsf > 1e12: ts = tsf / 1e3
        else:            ts = tsf
    if ts is None:
        ts = time.time()

    pos = quat = None
    if hasattr(lidar_data, 'pose'):
        p = lidar_data.pose.position
        o = lidar_data.pose.orientation
        pos  = (float(p.x_val), float(p.y_val), float(p.z_val))
        quat = (float(o.w_val), float(o.x_val), float(o.y_val), float(o.z_val))

    return points, ts, pos, quat






if __name__ == "__main__":
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")

    # Enable API control and arm
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    #
    lidarData = client.getLidarData()
    imuData = client.getImuData()
    gpsData = client.getGpsData()

    points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))

    tvis = Viewer3D()
    tvis.start(initial_points=points)
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")

    # Enable API control and arm
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    #
    lidarData = client.getLidarData()
    imuData = client.getImuData()
    gpsData = client.getGpsData()

    points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))

    tvis = Viewer3D()
    tvis.start(initial_points=points)

    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)

    # Define waypoints (x, y, z in NED coordinates)
    waypoints = [
        (0, 0, -10),      # Starting position - 10m up
        (20, 0, -10),     # Move 20m forward
        (20, 20, -10),    # Move 20m right
        (0, 20, -10),     # Move back 20m
        (0, 0, -10),      # Return to start
    ]
    for i, waypoint in enumerate(waypoints):
        print(f"Moving to waypoint {i+1}: {waypoint}...")
        future = client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2], 
            velocity=5
        )
        while not future._set_flag:  # non-blocking poll
            lidarData = client.getLidarData()
            imu_data = client.getImuData()
            gpsData = client.getGpsData()

            points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))
            tvis.update(points)
            #print("rendered")