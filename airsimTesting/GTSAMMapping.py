# --------------------------- Occupancy Grid ---------------------------------


class OccupancyGrid3D:
    """Simple 3D occupancy grid for fast visualization."""
    def __init__(self, voxel_size=1.0, grid_shape=(200,200,80), origin=(-50,-50,-50)):
        self.voxel_size = voxel_size
        self.grid_shape = grid_shape
        self.origin = np.array(origin, dtype=float)
        self.grid = np.zeros(grid_shape, dtype=np.uint8)

    def show_open3d(self):
        print("DEBUG: Entered show_open3d()")
        try:
            import open3d as o3d
            import numpy as np
        except ImportError:
            print("Open3D not installed. Run: pip install open3d")
            return
        print("DEBUG: Open3D and numpy imported successfully.")
        occupied = np.argwhere(self.grid > 0)
        print(f"DEBUG: Occupied voxels count = {len(occupied)}")
        if len(occupied) == 0:
            print("DEBUG: No occupied voxels to display.")
            return
        points = occupied * self.voxel_size + self.origin
        points = np.array(points, dtype=np.float32)
        print(f"DEBUG: Points min = {np.min(points, axis=0)}, max = {np.max(points, axis=0)}")
        print(f"DEBUG: Points shape = {points.shape}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("DEBUG: Calling o3d.visualization.draw_geometries...")
        try:
            o3d.visualization.draw_geometries([pcd], window_name="Occupancy Grid (Open3D)")
            print("DEBUG: draw_geometries call completed.")
        except Exception as e:
            print(f"DEBUG: draw_geometries failed: {e}")
        print("DEBUG: show_open3d() finished.")

    def world_to_grid(self, xyz):
        idx = np.floor((xyz - self.origin) / self.voxel_size).astype(int)
        return idx

    def add_points(self, points):
        if len(points) == 0:
            return
        idxs = self.world_to_grid(points)
        # filter points inside grid
        mask = ((idxs[:,0] >= 0) & (idxs[:,0] < self.grid_shape[0]) &
                (idxs[:,1] >= 0) & (idxs[:,1] < self.grid_shape[1]) &
                (idxs[:,2] >= 0) & (idxs[:,2] < self.grid_shape[2]))
        idxs = idxs[mask]
        for i in idxs:
            self.grid[i[0], i[1], i[2]] = 1

    def to_voxelgrid(self):
        # Convert occupied voxels to Open3D VoxelGrid with color
        occupied = np.argwhere(self.grid > 0)
        if len(occupied) == 0:
            return o3d.geometry.VoxelGrid()
        points = occupied * self.voxel_size + self.origin
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Color all voxels green
        colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        return vg

"""
GTSAM-based LIDAR + GPS mapping (AirSim)

Features
- Collects LIDAR scans from AirSim and computes relative motion with Open3D ICP
- Fuses ICP odometry + GPS (lat/lon -> local ENU or AirSim position fallback) in a GTSAM factor graph (iSAM2)
- Optional loop-closure by ICP against previous scans
- Builds an optimized dense pointcloud map using optimized poses

Dependencies: cosysairsim (AirSim), open3d, gtsam, numpy, opencv-python (optional)
Install examples:
  pip install open3d numpy opencv-python
  pip install gtsam  # if you use the GTSAM python binding distribution you have

Run while AirSim is running (example):
  python GTSAMMapping.py

You can also replay previously recorded data from a rosbag2 storage directory instead
of using live AirSim by setting the configuration variables near the bottom of this file:
  USE_ROSBAG = True
  ROSBAG_PATH = "/path/to/rosbag2_directory"
  ROSBAG_REALTIME = True   # optional, throttle to original timing

"""

import time
import argparse
import math
import copy

import cosysairsim as airsim
import numpy as np

try:
    import gtsam
    from gtsam import symbol
except Exception as e:
    raise ImportError("GTSAM python bindings are required (pip install gtsam or follow GTSAM install).")

try:
    import open3d as o3d
except Exception as e:
    raise ImportError("Open3D is required: pip install open3d")

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


# --------------------------- Utilities ---------------------------------

def pose3_from_pos_quat(pos, quat):
    """Create gtsam.Pose3 from position (x,y,z) and quaternion (w,x,y,z)"""
    print('DEBUG pose3_from_pos_quat: pos(type,width)=', type(pos), getattr(pos, 'shape', None), 'quat(type)=', type(quat))
    print('DEBUG pose3_from_pos_quat values (trim):', None if pos is None else np.array(pos).tolist()[:3], None if quat is None else list(quat)[:4])
    
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    rotation = gtsam.Rot3.Quaternion(w, x, y, z)
    translation = gtsam.Point3(float(pos[0]), float(pos[1]), float(pos[2]))
    return gtsam.Pose3(rotation, translation)


def transform_matrix_to_pose3(T):
    # Construct Pose3 directly from 4x4 matrix to avoid Rot3(matrix) issues
    return gtsam.Pose3(T)


def pose3_to_transform_matrix(pose3: gtsam.Pose3):
    R = pose3.rotation().matrix()
    t = pose3.translation()
    T = np.eye(4)
    T[:3, :3] = np.array(R)
    # Handle both gtsam.Point3 and numpy array
    if hasattr(t, 'x') and hasattr(t, 'y') and hasattr(t, 'z'):
        T[:3, 3] = np.array([t.x(), t.y(), t.z()])
    else:
        T[:3, 3] = np.array(t).reshape(3)
    return T


def latlon_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Rough conversion from lat/lon (deg) to local ENU meters using equirectangular approx.
    Good for small areas (tens of kilometers). Returns (east, north, up).
    """
    # WGS84 radius
    R = 6378137.0
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    mean_lat = math.radians((lat + ref_lat) / 2.0)
    north = dlat * R
    east = dlon * R * math.cos(mean_lat)
    up = alt - ref_alt
    return np.array([east, north, up], dtype=float)


# --------------------------- GTSAM Mapper ---------------------------------

class GTSAMMapper:
    """Collect scans, add ICP odometry + GPS priors into a GTSAM graph, optimize with ISAM2.

    Usage:
      mapper = GTSAMMapper(...)
      mapper.add_scan(points, airsim_pos, airsim_quat, gps_data)
      mapper.get_map()  # Open3D PointCloud of optimized map
    """

    def __init__(self,
                 voxel_size=0.2,
                 icp_max_dist=1.5,
                 gps_sigma=3.0,
                 icp_noise_scale=1.0,
                 loop_closure_search_every=10,
                 loop_closure_min_index_separation=15,
                 loop_closure_fitness_threshold=0.2,
                 max_map_points=800000,
                 use_icp=True):
        self.use_icp = use_icp
        self.voxel_size = voxel_size
        self.icp_max_dist = icp_max_dist
        self.gps_sigma = gps_sigma
        self.icp_noise_scale = icp_noise_scale

        # Loop-closure params
        self.lc_every = loop_closure_search_every
        self.lc_min_sep = loop_closure_min_index_separation
        self.lc_fitness_thresh = loop_closure_fitness_threshold

        self.max_map_points = max_map_points

        # Open3D containers
        self.scans = []            # list of raw Open3D pointclouds (downsampled)
        self.scan_poses_initial = []  # initial (pre-opt) Pose3 for each scan
        self.optimized_poses = []  # optimized Pose3 values after ISAM2
        self.raw_pose_matrices = []  # 4x4 drone world-pose matrix per scan, for ICP init

        # GTSAM containers
        print('DEBUG: creating gtsam.NonlinearFactorGraph()')
        self.graph = gtsam.NonlinearFactorGraph()
        print('DEBUG: creating gtsam.Values()')
        self.initial_estimates = gtsam.Values()
        print('DEBUG: creating gtsam.ISAM2()')
        self.isam = gtsam.ISAM2()
        print('DEBUG: created isam')
        self.pose_count = 0

        # GPS reference (for lat/lon -> local ENU). Set on first gps sample
        print('DEBUG: setting gps_ref = None')
        self.gps_ref = None

        # Map pointcloud (updated after optimization)
        print('DEBUG: creating o3d.geometry.PointCloud()')
        self.global_map = o3d.geometry.PointCloud()
        print('DEBUG: created global_map')

        # Pre-allocate a prior noise for the first pose (anchors the graph)
        print('DEBUG: creating prior_noise via gtsam.noiseModel.Isotropic.Sigma (diagonal broken)')
        # Diagonal.Sigmas is unstable in some gtsam Python builds; use isotropic fallback
        self.prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        print('DEBUG: set prior_noise (isotropic)')

    # ------------------ ICP helpers ------------------
    def _prepare_pcd(self, points_np):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        return pcd

    def _compute_icp(self, src_pcd, tgt_pcd, init=np.eye(4)):
        # estimate normals for point-to-plane ICP (helps when plane-like structure)
        radius = max(self.voxel_size * 2.0, 0.5)
        src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

        icp_res = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, self.icp_max_dist, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        return icp_res

    def _icp_noise_from_result(self, icp_result):
        # Build a conservative 6D diagonal noise model from ICP inlier_rmse
        rmse = float(icp_result.inlier_rmse)
        # rot_sigma and trans_sigma tuned by icp_noise_scale
        rot_sigma = max(0.05, rmse * 0.5) * self.icp_noise_scale
        trans_sigma = max(0.1, rmse * 1.5) * self.icp_noise_scale
        # Diagonal.Sigmas is unreliable in the current gtsam build; fall back to isotropic
        combined_sigma = float(max(rot_sigma, trans_sigma))
        return gtsam.noiseModel.Isotropic.Sigma(6, combined_sigma)

    # ------------------ Graph helpers ------------------
    def _add_pose_prior(self, key, pose3):
        # Use prior factor to anchor the first pose
        self.graph.add(gtsam.PriorFactorPose3(key, pose3, self.prior_noise))
        self.initial_estimates.insert(key, pose3)

    def _add_between_factor(self, key_i, key_j, relative_pose3, noise_model):
        self.graph.add(gtsam.BetweenFactorPose3(key_i, key_j, relative_pose3, noise_model))

    def _add_gps_prior_on_pose(self, key, gps_pos_world):
        # gps_pos_world is a 3-vector (x,y,z) in the same world frame as AirSim positions
        # Use a GPSFactor, which is a unary factor on the translation part of a Pose3
        gps_point = gtsam.Point3(float(gps_pos_world[0]), float(gps_pos_world[1]), float(gps_pos_world[2]))
        gps_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.gps_sigma)
        self.graph.add(gtsam.GPSFactor(key, gps_point, gps_noise))

    # ------------------ Main API ------------------
    def add_scan(self, points_np, airsim_pos=None, airsim_quat=None, gps_data=None):
        """Add a LIDAR scan (Nx3 numpy array) and optionally fuse GPS. Imitates basicMapping3D.py, but uses GTSAM for pose graph optimization."""
        if points_np is None or len(points_np) == 0:
            return

        # Transform scan from sensor to world frame using drone position and orientation
        if airsim_pos is not None and airsim_quat is not None:
            w, x, y, z = airsim_quat
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
            ])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = airsim_pos
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(points_np)
            scan_pcd.transform(T)
        else:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(points_np)

        # Add to global map
        self.global_map += scan_pcd

        # Downsample to prevent map from growing too large
        if len(self.global_map.points) > self.max_map_points:
            self.global_map = self.global_map.voxel_down_sample(self.voxel_size)

        # GTSAM pose graph logic
        pcd = self._prepare_pcd(points_np)
        gps_pos_world = None
        if gps_data is not None:
            try:
                lat = gps_data.gnss.geo_point.latitude
                lon = gps_data.gnss.geo_point.longitude
                alt = gps_data.gnss.geo_point.altitude
            except Exception:
                lat = getattr(gps_data, 'latitude', None)
                lon = getattr(gps_data, 'longitude', None)
                alt = getattr(gps_data, 'altitude', None)
            if lat is not None and lon is not None:
                if self.gps_ref is None:
                    self.gps_ref = (float(lat), float(lon), float(alt) if alt is not None else 0.0)
                ref_lat, ref_lon, ref_alt = self.gps_ref
                enu = latlon_to_enu(float(lat), float(lon), float(alt or 0.0), ref_lat, ref_lon, ref_alt)
                gps_pos_world = enu
        if gps_pos_world is None and airsim_pos is not None:
            gps_pos_world = np.array(airsim_pos, dtype=float)

        key = symbol('x', self.pose_count)

        if self.pose_count == 0:
            # Anchor first pose — requires valid pose data
            if airsim_pos is None or airsim_quat is None:
                return
            pose3 = pose3_from_pos_quat(airsim_pos, airsim_quat)
            self._add_pose_prior(key, pose3)
            self.scan_poses_initial.append(pose3)
            self.scans.append(pcd)
            self.optimized_poses.append(pose3)
            self.raw_pose_matrices.append(pose3_to_transform_matrix(pose3))
            self.pose_count += 1
            return

        # Compute relative pose and initial guess — either via ICP or pure odometry
        key_prev = symbol('x', self.pose_count - 1)

        if airsim_pos is not None and airsim_quat is not None:
            T_curr = pose3_to_transform_matrix(pose3_from_pos_quat(airsim_pos, airsim_quat))
        else:
            T_curr = None

        if self.use_icp:
            prev_pcd = self.scans[-1]
            init_trans = np.linalg.inv(self.raw_pose_matrices[-1]) @ T_curr if T_curr is not None else np.eye(4)
            icp_res = self._compute_icp(pcd, prev_pcd, init_trans)
            relative_pose = transform_matrix_to_pose3(icp_res.transformation)
            noise = self._icp_noise_from_result(icp_res)
            # Compose previous optimized pose with ICP relative for initial estimate
            try:
                prev_opt_pose = self.isam.calculateEstimate().atPose3(key_prev)
            except Exception:
                prev_opt_pose = self.scan_poses_initial[-1]
            initial_pose_guess = prev_opt_pose.compose(relative_pose)
        else:
            # Odometry-only: derive relative pose directly from consecutive drone poses
            if T_curr is not None:
                rel_T = np.linalg.inv(self.raw_pose_matrices[-1]) @ T_curr
                relative_pose = transform_matrix_to_pose3(rel_T)
                initial_pose_guess = transform_matrix_to_pose3(T_curr)
            else:
                # No odometry available for this scan; hold last pose
                relative_pose = transform_matrix_to_pose3(np.eye(4))
                initial_pose_guess = self.scan_poses_initial[-1]
            noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            icp_res = None

        self._add_between_factor(key_prev, key, relative_pose, noise)
        self.initial_estimates.insert(key, initial_pose_guess)

        # Add GPS prior if available
        if gps_pos_world is not None:
            self._add_gps_prior_on_pose(key, gps_pos_world)

        # Optional loop-closure
        if (self.pose_count % self.lc_every) == 0:
            for j in range(0, max(0, self.pose_count - self.lc_min_sep)):
                if abs(self.pose_count - j) < self.lc_min_sep:
                    continue
                candidate_pcd = self.scans[j]
                try:
                    est_values = self.isam.calculateEstimate()
                    pose_j = est_values.atPose3(symbol('x', j))
                    pose_curr = initial_pose_guess
                    pj = pose3_to_transform_matrix(pose_j)
                    pc = pose3_to_transform_matrix(pose_curr)
                    dist = np.linalg.norm(pj[:3, 3] - pc[:3, 3])
                    if dist > 50.0:
                        continue
                except Exception:
                    pass

                # Estimate normals for PointToPlane ICP
                radius = max(self.voxel_size * 2.0, 0.5)
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
                candidate_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

                lc_icp = o3d.pipelines.registration.registration_icp(
                    pcd, candidate_pcd, self.icp_max_dist * 1.5, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
                )
                if lc_icp.fitness > self.lc_fitness_thresh and lc_icp.inlier_rmse < (self.icp_max_dist * 0.5):
                    rel_pose_lc = transform_matrix_to_pose3(lc_icp.transformation)
                    noise_lc = self._icp_noise_from_result(lc_icp)
                    self._add_between_factor(symbol('x', j), key, rel_pose_lc, noise_lc)
                    print(f"Loop-closure added between x{j} and x{self.pose_count} (fitness={lc_icp.fitness:.3f}, rmse={lc_icp.inlier_rmse:.3f})")
                    break

        # Update ISAM
        self.isam.update(self.graph, self.initial_estimates)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        # Retrieve optimized pose
        current_est = self.isam.calculateEstimate()
        opt_pose_curr = current_est.atPose3(key)

        # Store scan and poses
        self.scans.append(pcd)
        self.scan_poses_initial.append(initial_pose_guess)
        self.optimized_poses.append(opt_pose_curr)
        if T_curr is not None:
            self.raw_pose_matrices.append(T_curr)
        elif icp_res is not None:
            self.raw_pose_matrices.append(self.raw_pose_matrices[-1] @ icp_res.transformation)
        else:
            self.raw_pose_matrices.append(self.raw_pose_matrices[-1])
        self.pose_count += 1

    def optimize_and_build_map(self, downsample_voxel=None):
        """Rebuild dense map from optimized poses + raw scans."""
        if self.pose_count == 0:
            return self.global_map

        values = self.isam.calculateEstimate()
        merged = o3d.geometry.PointCloud()

        for i, scan in enumerate(self.scans):
            key = symbol('x', i)
            try:
                pose_i = values.atPose3(key)
            except Exception:
                pose_i = self.scan_poses_initial[i]
            T = pose3_to_transform_matrix(pose_i)
            pc = copy.deepcopy(scan)
            pc.transform(T)
            merged += pc

        # Voxel downsample
        if downsample_voxel is None:
            downsample_voxel = self.voxel_size
        merged = merged.voxel_down_sample(downsample_voxel)

        # Trim map size
        if len(merged.points) > self.max_map_points:
            merged = merged.voxel_down_sample(downsample_voxel * 2.0)

        self.global_map = merged
        return merged

    def save_map(self, filename="gtsam_map.ply"):
        o3d.io.write_point_cloud(filename, self.global_map)
        print(f"Saved map to {filename}")


# --------------------------- AirSim data helpers -------------------------

def get_lidar_scan(client):
    try:
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            return None
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return points
    except Exception:
        return None


def get_gps_data(client):
    try:
        return client.getGpsData()
    except Exception:
        return None


def get_camera_image(client):
    if not CV2_AVAILABLE:
        return None
    try:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        if not responses:
            return None
        r = responses[0]
        img1d = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(r.height, r.width, 3)
        return img_rgb
    except Exception:
        return None


# ---------------------- rosbag replay helpers ----------------------

def rosbag_scan_generator(bag_path, realtime=False):
    """Yield tuples (points Nx3, pos(3,), quat(4,), timestamp, gps_object).
    Reads messages from a rosbag2 storage directory written with rosbag2_py.
    Expects topics: /lidar (sensor_msgs/PointCloud2), /odom (nav_msgs/Odometry),
    optional /gps/fix (sensor_msgs/NavSatFix). GPS messages are converted to a
    minimal object implementing ``latitude``, ``longitude`` and ``altitude``
    attributes so that :py:meth:`GTSAMMapper.add_scan` can handle them.
    """
    try:
        import rclpy
        from rclpy.serialization import deserialize_message
        import rosbag2_py
        from sensor_msgs.msg import PointCloud2, NavSatFix
        from nav_msgs.msg import Odometry
    except Exception as e:
        raise RuntimeError(f"rosbag replay requires ROS2 python packages: {e}")

    # simple helper class for GPS data
    class _BagGPS:
        def __init__(self, lat, lon, alt):
            self.latitude = float(lat)
            self.longitude = float(lon)
            self.altitude = float(alt)

    # initialize rclpy for deserialization (safe to call multiple times)
    try:
        rclpy.init()
    except Exception:
        pass

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('cdr', 'cdr')
    reader.open(storage_options, converter_options)

    available = {t.name: t.type for t in reader.get_all_topics_and_types()}
    has_lidar = '/lidar' in available
    has_odom = '/odom' in available
    has_gps = '/gps/fix' in available

    last_pos = None
    last_quat = None
    last_gps = None
    last_ts = None

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        t_sec = float(ts_ns) / 1e9

        if topic == '/odom' and has_odom:
            msg = deserialize_message(data, Odometry)
            px = float(msg.pose.pose.position.x)
            py = float(msg.pose.pose.position.y)
            pz = float(msg.pose.pose.position.z)
            last_pos = np.array([px, py, pz], dtype=np.float32)
            q = msg.pose.pose.orientation
            last_quat = (float(q.w), float(q.x), float(q.y), float(q.z))

        elif topic == '/gps/fix' and has_gps:
            msg = deserialize_message(data, NavSatFix)
            last_gps = _BagGPS(msg.latitude, msg.longitude, msg.altitude)

        elif topic == '/lidar' and has_lidar:
            msg = deserialize_message(data, PointCloud2)
            if msg.data and len(msg.data) > 0:
                pts = np.frombuffer(msg.data, dtype=np.float32)
                try:
                    pts = pts.reshape(-1, 3)
                except Exception:
                    pts = np.zeros((0, 3), dtype=np.float32)
            else:
                pts = np.zeros((0, 3), dtype=np.float32)

            # yield using last known odom/gps (may be None)
            yield pts, last_pos, last_quat, t_sec, last_gps

            if realtime and last_ts is not None:
                dt = t_sec - last_ts
                if dt > 0:
                    time.sleep(dt)

            last_ts = t_sec

    try:
        rclpy.shutdown()
    except Exception:
        pass


# ---------------------- Configuration Variables ----------------------
# Set USE_ROSBAG to True to replay from a rosbag instead of live AirSim.
# When True, ROSBAG_PATH must point to a valid rosbag2 storage directory.

USE_ROSBAG = True
ROSBAG_PATH = 'airsim_rosbag_1771706695'
ROSBAG_REALTIME = False
USE_ICP = False  # Set False to skip ICP and use GPS/odometry poses directly
# --------------------------- Main script ---------------------------------

def main(args):
    # decide between live AirSim mode and rosbag playback
    rosbag_path = ROSBAG_PATH if USE_ROSBAG else None
    rosbag_realtime = ROSBAG_REALTIME if USE_ROSBAG else False

    if rosbag_path:
        # playback mode: read scans from rosbag and feed them into mapper
        print(f"Replaying from rosbag: {rosbag_path} (realtime={rosbag_realtime})")
        use_icp = not args.no_icp and USE_ICP
        print(f"Scan matching mode: {'ICP' if use_icp else 'odometry-only'}")
        mapper = GTSAMMapper(voxel_size=args.voxel_size,
                             icp_max_dist=args.icp_max_distance,
                             gps_sigma=args.gps_sigma,
                             icp_noise_scale=args.icp_noise_scale,
                             loop_closure_search_every=args.lc_every,
                             loop_closure_min_index_separation=args.lc_min_sep,
                             loop_closure_fitness_threshold=args.lc_fitness_thresh,
                             max_map_points=args.max_map_points,
                             use_icp=use_icp)
        try:
            for points, pos_arr, quat, ts, gps in rosbag_scan_generator(rosbag_path, realtime=rosbag_realtime):
                if points is not None and len(points) > 0:
                    mapper.add_scan(points, airsim_pos=pos_arr, airsim_quat=quat, gps_data=gps)
            print("Rosbag playback finished; optimizing map")
        except KeyboardInterrupt:
            print("Interrupted during playback; optimizing partial map")

        final_map = mapper.optimize_and_build_map()
        print(f"Map contains {len(final_map.points):,} points — opening visualization")
        o3d.visualization.draw_geometries([final_map], window_name="GTSAM Map", width=800, height=600)
        return

    # --- live AirSim mode ---
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(1)

    mapper = GTSAMMapper(voxel_size=args.voxel_size,
                         icp_max_dist=args.icp_max_distance,
                         gps_sigma=args.gps_sigma,
                         icp_noise_scale=args.icp_noise_scale,
                         loop_closure_search_every=args.lc_every,
                         loop_closure_min_index_separation=args.lc_min_sep,
                         loop_closure_fitness_threshold=args.lc_fitness_thresh,
                         max_map_points=args.max_map_points)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1.5)

    # Simple waypoint set for demo (same pattern used in other scripts)
    waypoints = [
        (0, 0, -10),
        (10, 0, -10),
        (0, 0, -10),
    ]

    # No occupancy grid; visualize global point cloud map after mapping

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="GTSAM Map", width=800, height=600)
        vis.add_geometry(mapper.global_map)
        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        for i, wp in enumerate(waypoints):
            print(f"Moving to waypoint {i+1}/{len(waypoints)}: {wp}")
            client.moveToPositionAsync(wp[0], wp[1], wp[2], velocity=3)

            while True:
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                pos_arr = np.array([pos.x_val, pos.y_val, pos.z_val])
                ori = state.kinematics_estimated.orientation
                quat = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])

                points = get_lidar_scan(client)
                gps = get_gps_data(client)
                print(f"DEBUG: Lidar scan points shape: {None if points is None else points.shape}")
                if points is not None and len(points) > 0:
                    print(f"DEBUG: Lidar scan points sample: {points[:5]}")
                    mapper.add_scan(points, airsim_pos=pos_arr, airsim_quat=quat, gps_data=gps)
                    global_points = np.asarray(mapper.global_map.points)
                    print(f"DEBUG: Global map point count after add_scan: {len(global_points)}")
                    if len(global_points) > 0:
                        print(f"DEBUG: Global map points sample: {global_points[:5]}")
                        colors = np.tile(np.array([[1.0, 1.0, 0.0]]), (len(global_points), 1))
                        mapper.global_map.colors = o3d.utility.Vector3dVector(colors)
                        vis.clear_geometries()
                        vis.add_geometry(mapper.global_map)
                        vis.poll_events()
                        vis.update_renderer()

                if CV2_AVAILABLE:
                    cam_img = get_camera_image(client)
                    if cam_img is not None:
                        small = cv2.resize(cam_img, (640, 360))
                        cv2.imshow("Camera", small)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                dist = np.linalg.norm(pos_arr - np.array([wp[0], wp[1], wp[2]]))
                if dist < 1.0:
                    break
                time.sleep(0.08)

        print("Flight complete — running final optimization & saving map")
        mapper.optimize_and_build_map()
        mapper.save_map(args.output)

        print("DEBUG: Mapping complete. Visualization window open.")
        print("Press 'q' in the Open3D window to exit.")
        o3d.visualization.draw_geometries([mapper.global_map], window_name="GTSAM Map", width=800, height=600)

    except KeyboardInterrupt:
        print("Interrupted by user — saving partial map...")
        mapper.optimize_and_build_map()
        mapper.save_map(args.output.replace('.ply', '_partial.ply'))

    finally:
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTSAM LIDAR+GPS mapping (AirSim)')
    parser.add_argument('--voxel-size', type=float, default=0.2)
    parser.add_argument('--icp-max-distance', type=float, default=1.5)
    parser.add_argument('--gps-sigma', type=float, default=3.0)
    parser.add_argument('--icp-noise-scale', type=float, default=1.0)
    parser.add_argument('--lc-every', type=int, default=12)
    parser.add_argument('--lc-min-sep', type=int, default=20)
    parser.add_argument('--lc-fitness-thresh', type=float, default=0.25)
    parser.add_argument('--max-map-points', type=int, default=800000)
    parser.add_argument('--output', type=str, default='gtsam_map.ply')
    parser.add_argument('--no-icp', action='store_true', default=False,
                        help='Skip ICP; place scans using GPS/odometry poses only')
    args = parser.parse_args()
    main(args)
