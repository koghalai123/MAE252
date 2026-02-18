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

"""

import time
import argparse
import math
import copy

import cosysairsim as airsim
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise ImportError("Open3D is required: pip install open3d")

try:
    import gtsam
    from gtsam import symbol
except Exception as e:
    raise ImportError("GTSAM python bindings are required (pip install gtsam or follow GTSAM install).")

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


# --------------------------- Utilities ---------------------------------

def quaternion_to_rot3(quat):
    """Convert AirSim quaternion (w,x,y,z) to gtsam.Rot3"""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return gtsam.Rot3.Quaternion(w, x, y, z)


def pose3_from_pos_quat(pos, quat):
    """Create gtsam.Pose3 from position (x,y,z) and quaternion (w,x,y,z)"""
    R = quaternion_to_rot3(quat)
    t = gtsam.Point3(float(pos[0]), float(pos[1]), float(pos[2]))
    return gtsam.Pose3(R, t)


def transform_matrix_to_pose3(T):
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    return gtsam.Pose3(R, t)


def pose3_to_transform_matrix(pose3: gtsam.Pose3):
    R = pose3.rotation().matrix()
    t = pose3.translation()
    T = np.eye(4)
    T[:3, :3] = np.array(R)
    T[:3, 3] = np.array([t.x(), t.y(), t.z()])
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
                 max_map_points=800000):
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

        # GTSAM containers
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.isam = gtsam.ISAM2()
        self.pose_count = 0

        # GPS reference (for lat/lon -> local ENU). Set on first gps sample
        self.gps_ref = None

        # Map pointcloud (updated after optimization)
        self.global_map = o3d.geometry.PointCloud()

        # Pre-allocate a prior noise for the first pose (anchors the graph)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6]))
        self.prior_noise = prior_noise

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
        sigmas = np.array([rot_sigma, rot_sigma, rot_sigma, trans_sigma, trans_sigma, trans_sigma])
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    # ------------------ Graph helpers ------------------
    def _add_pose_prior(self, key, pose3):
        # Use prior factor to anchor the first pose
        self.graph.add(gtsam.PriorFactorPose3(key, pose3, self.prior_noise))
        self.initial_estimates.insert(key, pose3)

    def _add_between_factor(self, key_i, key_j, relative_pose3, noise_model):
        self.graph.add(gtsam.BetweenFactorPose3(key_i, key_j, relative_pose3, noise_model))

    def _add_gps_prior_on_pose(self, key, gps_pos_world):
        # gps_pos_world is a 3-vector (x,y,z) in the same world frame as AirSim positions
        # Create a Pose3 with GPS translation and identity rotation and add a PriorFactorPose3
        pose_from_gps = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(float(gps_pos_world[0]),
                                                                float(gps_pos_world[1]),
                                                                float(gps_pos_world[2])))
        # noise: large rotation sigma (so GPS doesn't constrain orientation), small translation sigma
        rot_sigma = 1e3
        trans_sigma = max(0.3, self.gps_sigma)
        sigmas = np.array([rot_sigma, rot_sigma, rot_sigma, trans_sigma, trans_sigma, trans_sigma])
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self.graph.add(gtsam.PriorFactorPose3(key, pose_from_gps, noise))

    # ------------------ Main API ------------------
    def add_scan(self, points_np, airsim_pos=None, airsim_quat=None, gps_data=None):
        """Add a LIDAR scan (Nx3 numpy array) and optionally fuse GPS.

        airsim_pos: (x,y,z) from getMultirotorState (same frame as lidar usage in other scripts)
        airsim_quat: (w,x,y,z) orientation from AirSim
        gps_data: AirSim getGpsData() object (optional)
        """
        if points_np is None or len(points_np) == 0:
            return

        # Prepare downsampled Open3D pointcloud for ICP/map
        pcd = self._prepare_pcd(points_np)

        # Compute a 'gps' position in world frame if provided (lat/lon -> ENU)
        gps_pos_world = None
        if gps_data is not None:
            try:
                # AirSim may provide gnss.geo_point or simple latitude/longitude fields
                lat = gps_data.gnss.geo_point.latitude
                lon = gps_data.gnss.geo_point.longitude
                alt = gps_data.gnss.geo_point.altitude
            except Exception:
                lat = getattr(gps_data, 'latitude', None)
                lon = getattr(gps_data, 'longitude', None)
                alt = getattr(gps_data, 'altitude', None)

            if lat is not None and lon is not None:
                if self.gps_ref is None:
                    # store reference to convert to local ENU
                    self.gps_ref = (float(lat), float(lon), float(alt) if alt is not None else 0.0)
                ref_lat, ref_lon, ref_alt = self.gps_ref
                enu = latlon_to_enu(float(lat), float(lon), float(alt or 0.0), ref_lat, ref_lon, ref_alt)
                # Use AirSim vehicle position sign convention: AirSim world used elsewhere is typically NED
                # We'll keep everything in ENU for GPS prior; airsim_pos should be consistent (we use airsim_pos if provided)
                gps_pos_world = enu

        # Fallback: use AirSim position (kinematics_estimated.position) as a coarse global measurement
        if gps_pos_world is None and airsim_pos is not None:
            gps_pos_world = np.array(airsim_pos, dtype=float)

        # New pose key
        key = symbol('x', self.pose_count)

        if self.pose_count == 0:
            # First pose: insert prior (anchor) using airsim_pos/quat if available, else identity
            if airsim_pos is not None and airsim_quat is not None:
                init_pose = pose3_from_pos_quat(airsim_pos, airsim_quat)
            elif gps_pos_world is not None:
                init_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(float(gps_pos_world[0]), float(gps_pos_world[1]), float(gps_pos_world[2])))
            else:
                init_pose = gtsam.Pose3()  # identity

            self._add_pose_prior(key, init_pose)
            self.isam.update(self.graph, self.initial_estimates)
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial_estimates = gtsam.Values()
            current_est = self.isam.calculateEstimate()

            self.scans.append(pcd)
            self.scan_poses_initial.append(init_pose)
            self.optimized_poses.append(current_est.atPose3(key))
            self.pose_count += 1
            return

        # Otherwise, compute ICP between previous scan and current to estimate relative transform
        prev_pcd = self.scans[-1]
        # Use previous optimized pose as initial guess for ICP alignment in world frame
        init_trans = np.eye(4)
        icp_res = self._compute_icp(pcd, prev_pcd, init_trans)

        relative_pose = transform_matrix_to_pose3(icp_res.transformation)

        # Add BetweenFactor from previous pose to this pose with noise from ICP
        key_prev = symbol('x', self.pose_count - 1)
        key_curr = key
        noise = self._icp_noise_from_result(icp_res)
        self._add_between_factor(key_prev, key_curr, relative_pose, noise)

        # Add initial estimate for current pose by composing previous optimized pose and ICP relative
        # Use last optimized pose if available from isam, else use last initial
        try:
            current_estimate_values = self.isam.calculateEstimate()
            prev_opt_pose = current_estimate_values.atPose3(key_prev)
        except Exception:
            prev_opt_pose = self.scan_poses_initial[-1]

        initial_pose_guess = prev_opt_pose.compose(relative_pose)
        self.initial_estimates.insert(key_curr, initial_pose_guess)

        # Add GPS prior if we have gps_pos_world available
        if gps_pos_world is not None:
            # If gps_ref is set (lat/lon converted), gps_pos_world is ENU relative to the ref origin.
            # If gps_pos_world was taken from airsim_pos fallback, it's in simulator world coords and matches pose units.
            self._add_gps_prior_on_pose(key_curr, gps_pos_world)

        # Optional: loop-closure — check against older scans every LC interval
        if (self.pose_count % self.lc_every) == 0:
            for j in range(0, max(0, self.pose_count - self.lc_min_sep)):
                # skip near neighbors
                if abs(self.pose_count - j) < self.lc_min_sep:
                    continue
                candidate_pcd = self.scans[j]
                # Quick downsample and bounding-box distance check to avoid heavy ICP
                # Estimate center distance
                try:
                    est_values = self.isam.calculateEstimate()
                    pose_j = est_values.atPose3(symbol('x', j))
                    pose_curr = initial_pose_guess
                    pj = pose3_to_transform_matrix(pose_j)
                    pc = pose3_to_transform_matrix(pose_curr)
                    dist = np.linalg.norm(pj[:3, 3] - pc[:3, 3])
                    if dist > 50.0:  # too far to consider
                        continue
                except Exception:
                    pass

                lc_icp = o3d.pipelines.registration.registration_icp(
                    pcd, candidate_pcd, self.icp_max_dist * 1.5, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
                )
                if lc_icp.fitness > self.lc_fitness_thresh and lc_icp.inlier_rmse < (self.icp_max_dist * 0.5):
                    # Good loop closure — add between factor
                    rel_pose_lc = transform_matrix_to_pose3(lc_icp.transformation)
                    noise_lc = self._icp_noise_from_result(lc_icp)
                    self._add_between_factor(symbol('x', j), key_curr, rel_pose_lc, noise_lc)
                    print(f"Loop-closure added between x{j} and x{self.pose_count} (fitness={lc_icp.fitness:.3f}, rmse={lc_icp.inlier_rmse:.3f})")
                    break

        # Update ISAM with the new factors and initial estimate
        self.isam.update(self.graph, self.initial_estimates)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        # Retrieve the optimized current pose
        current_est = self.isam.calculateEstimate()
        opt_pose_curr = current_est.atPose3(key_curr)

        # Store scan and poses
        self.scans.append(pcd)
        self.scan_poses_initial.append(initial_pose_guess)
        self.optimized_poses.append(opt_pose_curr)
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


# --------------------------- Main script ---------------------------------

def main(args):
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
        (10, 10, -10),
        (0, 10, -10),
        (0, 0, -10),
    ]

    # Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GTSAM Map (optimized)", width=900, height=600)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.05, 0.05, 0.05])
    render_opt.point_size = 2.0

    try:
        for i, wp in enumerate(waypoints):
            print(f"Moving to waypoint {i+1}/{len(waypoints)}: {wp}")
            client.moveToPositionAsync(wp[0], wp[1], wp[2], velocity=3)

            # gather data while moving
            while True:
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                pos_arr = np.array([pos.x_val, pos.y_val, pos.z_val])
                ori = state.kinematics_estimated.orientation
                quat = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])

                points = get_lidar_scan(client)
                gps = get_gps_data(client)

                if points is not None and len(points) > 0:
                    mapper.add_scan(points, airsim_pos=pos_arr, airsim_quat=quat, gps_data=gps)

                # Re-optimize map and update visualization occasionally
                if mapper.pose_count % 2 == 0 and mapper.pose_count > 0:
                    merged = mapper.optimize_and_build_map()
                    vis.clear_geometries()
                    vis.add_geometry(merged)
                    # coordinate frame
                    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
                    vis.add_geometry(coord)
                    vis.poll_events()
                    vis.update_renderer()

                # Optional camera overlay
                if CV2_AVAILABLE:
                    cam_img = get_camera_image(client)
                    if cam_img is not None:
                        small = cv2.resize(cam_img, (640, 360))
                        cv2.imshow("Camera", small)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                # check reached waypoint
                dist = np.linalg.norm(pos_arr - np.array([wp[0], wp[1], wp[2]]))
                if dist < 1.0:
                    break
                time.sleep(0.08)

        print("Flight complete — running final optimization & saving map")
        mapper.optimize_and_build_map()
        mapper.save_map(args.output)

        print("Press 'q' in the camera window (if open) to exit; close 3D window to finish")
        while True:
            vis.poll_events()
            vis.update_renderer()
            if CV2_AVAILABLE and cv2.waitKey(100) & 0xFF == ord('q'):
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user — saving partial map...")
        mapper.optimize_and_build_map()
        mapper.save_map(args.output.replace('.ply', '_partial.ply'))

    finally:
        vis.destroy_window()
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
    args = parser.parse_args()
    main(args)
