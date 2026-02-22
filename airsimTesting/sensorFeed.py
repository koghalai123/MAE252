import cosysairsim as airsim
import cv2
import numpy as np
import open3d as o3d
import time
import multiprocessing
import multiprocessing.shared_memory
from collections import deque
from scipy.spatial.transform import Rotation, Slerp


# Marker shared-memory layout: 8 float64 values
#   [0:3] drone position,  [3] drone valid (1.0/0.0)
#   [4:7] target position, [7] target valid (1.0/0.0)
_MARKER_FLOATS = 8
_MARKER_RADIUS = 0.5
_FRONTIER_MAX_PTS = 200_000  # max frontier overlay points


def _make_sphere(color, radius=_MARKER_RADIUS):
    """Create a coloured triangle-mesh sphere."""
    import open3d as _o3d
    s = _o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    return s


def _vis_process(shm_name, shape, dtype_str, lock, update_event, stop_event,
                 marker_shm_name=None, frontier_shm_name=None,
                 frontier_shape=None):
    """Visualizer loop running in a separate process."""
    import numpy as _np
    import open3d as _o3d

    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    buf = _np.ndarray(shape, dtype=dtype_str, buffer=shm.buf)

    # Optional marker shared memory
    marker_shm = None
    marker_buf = None
    if marker_shm_name is not None:
        marker_shm = multiprocessing.shared_memory.SharedMemory(name=marker_shm_name)
        marker_buf = _np.ndarray((_MARKER_FLOATS,), dtype='float64', buffer=marker_shm.buf)

    # Optional frontier overlay shared memory
    frontier_shm = None
    frontier_buf = None
    if frontier_shm_name is not None and frontier_shape is not None:
        frontier_shm = multiprocessing.shared_memory.SharedMemory(name=frontier_shm_name)
        frontier_buf = _np.ndarray(frontier_shape, dtype=dtype_str, buffer=frontier_shm.buf)

    pcd = _o3d.geometry.PointCloud()
    # Filter zero-padded slots for initial display
    init_pts = buf.copy()
    mask = _np.any(init_pts != 0, axis=1)
    pcd.points = _o3d.utility.Vector3dVector(init_pts[mask])

    # Drone marker (red) and target marker (green)
    drone_sphere  = _make_sphere([1.0, 0.0, 0.0])
    target_sphere = _make_sphere([0.0, 1.0, 0.0])
    drone_visible  = False
    target_visible = False

    # Frontier overlay point cloud (orange)
    frontier_pcd = _o3d.geometry.PointCloud()

    vis = _o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(frontier_pcd)

    while not stop_event.is_set():
        if update_event.is_set():
            with lock:
                pts = buf.copy()
                mdata = marker_buf.copy() if marker_buf is not None else None
                fdata = frontier_buf.copy() if frontier_buf is not None else None
            update_event.clear()
            # Only render actual points, not zero-padded buffer slots
            mask = _np.any(pts != 0, axis=1)
            pts = pts[mask]
            pcd.points = _o3d.utility.Vector3dVector(pts)
            vis.update_geometry(pcd)

            # ── Update frontier overlay (orange) ──────────────────────
            if fdata is not None:
                fmask = _np.any(fdata != 0, axis=1)
                fpts = fdata[fmask]
                frontier_pcd.points = _o3d.utility.Vector3dVector(fpts)
                if len(fpts) > 0:
                    frontier_pcd.paint_uniform_color([1.0, 0.6, 0.0])
                vis.update_geometry(frontier_pcd)

            # ── Update drone sphere ───────────────────────────────────
            if mdata is not None and mdata[3] == 1.0:
                new_pos = mdata[0:3]
                # Remove and re-add at new position (Open3D has no translate-in-place for vis)
                if drone_visible:
                    vis.remove_geometry(drone_sphere, reset_bounding_box=False)
                drone_sphere = _make_sphere([1.0, 0.0, 0.0])
                drone_sphere.translate(new_pos)
                vis.add_geometry(drone_sphere, reset_bounding_box=False)
                drone_visible = True
            elif drone_visible and (mdata is None or mdata[3] != 1.0):
                vis.remove_geometry(drone_sphere, reset_bounding_box=False)
                drone_visible = False

            # ── Update target sphere ──────────────────────────────────
            if mdata is not None and mdata[7] == 1.0:
                new_pos = mdata[4:7]
                if target_visible:
                    vis.remove_geometry(target_sphere, reset_bounding_box=False)
                target_sphere = _make_sphere([0.0, 1.0, 0.0])
                target_sphere.translate(new_pos)
                vis.add_geometry(target_sphere, reset_bounding_box=False)
                target_visible = True
            elif target_visible and (mdata is None or mdata[7] != 1.0):
                vis.remove_geometry(target_sphere, reset_bounding_box=False)
                target_visible = False

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()
    shm.close()
    if marker_shm is not None:
        marker_shm.close()
    if frontier_shm is not None:
        frontier_shm.close()


class Viewer3D:
    """Open3D visualizer in a separate process — stays responsive at breakpoints."""

    MAX_POINTS = 7_000_000  # pre-allocated shared buffer size

    def __init__(self):
        self._shm = None
        self._buf = None
        self._marker_shm = None
        self._marker_buf = None
        self._frontier_shm = None
        self._frontier_buf = None
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

        # Marker shared memory (drone + target positions)
        marker_nbytes = _MARKER_FLOATS * np.dtype(np.float64).itemsize
        self._marker_shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=marker_nbytes)
        self._marker_buf = np.ndarray((_MARKER_FLOATS,), dtype=np.float64,
                                       buffer=self._marker_shm.buf)
        self._marker_buf[:] = 0.0

        # Frontier overlay shared memory
        self._frontier_shape = (_FRONTIER_MAX_PTS, 3)
        frontier_nbytes = int(np.prod(self._frontier_shape) * self._dtype.itemsize)
        self._frontier_shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=frontier_nbytes)
        self._frontier_buf = np.ndarray(self._frontier_shape, dtype=self._dtype,
                                         buffer=self._frontier_shm.buf)
        self._frontier_buf[:] = 0.0

        self._update_event.set()
        self._proc = multiprocessing.Process(
            target=_vis_process,
            args=(self._shm.name, self._shape, self._dtype.str,
                  self._lock, self._update_event, self._stop_event),
            kwargs=dict(marker_shm_name=self._marker_shm.name,
                        frontier_shm_name=self._frontier_shm.name,
                        frontier_shape=self._frontier_shape),
            daemon=True,
        )
        self._proc.start()

    def update(self, points, *, drone_pos=None, target_pos=None, frontier_points=None):
        n = min(len(points), self.MAX_POINTS)
        with self._lock:
            self._buf[:] = 0.0
            self._buf[:n] = points[:n].astype(self._dtype)
            if self._marker_buf is not None:
                if drone_pos is not None:
                    self._marker_buf[0:3] = np.asarray(drone_pos, dtype=np.float64)
                    self._marker_buf[3] = 1.0
                else:
                    self._marker_buf[3] = 0.0
                if target_pos is not None:
                    self._marker_buf[4:7] = np.asarray(target_pos, dtype=np.float64)
                    self._marker_buf[7] = 1.0
                else:
                    self._marker_buf[7] = 0.0
            if self._frontier_buf is not None:
                self._frontier_buf[:] = 0.0
                if frontier_points is not None and len(frontier_points) > 0:
                    nf = min(len(frontier_points), _FRONTIER_MAX_PTS)
                    self._frontier_buf[:nf] = frontier_points[:nf].astype(self._dtype)
        self._update_event.set()

    def stop(self):
        self._stop_event.set()
        if self._proc is not None:
            self._proc.join(timeout=3)
        if self._marker_shm is not None:
            self._marker_shm.close()
            self._marker_shm.unlink()
        if self._frontier_shm is not None:
            self._frontier_shm.close()
            self._frontier_shm.unlink()
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