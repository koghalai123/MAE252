"""
Advanced 3D Mapping using ICP-based SLAM
Uses GPS for initial pose estimation and ICP for scan registration refinement
Open3D for point cloud processing and visualization
"""
import cosysairsim as airsim
import numpy as np
import time
import cv2
from collections import deque

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Open3D not installed. Install with: pip install open3d")
    print("Open3D is recommended for 3D mapping and visualization")
    OPEN3D_AVAILABLE = False
    exit(1)


class ICPSLAMMapper:
    """
    3D Mapping with ICP-based scan registration
    Uses GPS for initial pose guess, then refines with ICP scan matching
    """
    
    def __init__(self, voxel_size=0.15, icp_fitness_threshold=0.3):
        """
        Args:
            voxel_size: Size of voxel for downsampling (meters)
            icp_fitness_threshold: Minimum fitness score for ICP to be accepted
        """
        self.voxel_size = voxel_size
        self.icp_fitness_threshold = icp_fitness_threshold
        
        # Map storage
        self.global_map = o3d.geometry.PointCloud()
        self.local_maps = []  # Store recent local maps for ICP
        self.max_local_maps = 5
        
        # Pose tracking
        self.poses = []  # List of refined poses
        self.gps_poses = []  # GPS-based poses for comparison
        
        # Statistics
        self.scan_count = 0
        self.icp_success_count = 0
        self.total_drift = 0.0
        
    def add_scan(self, points, gps_position, orientation):
        """
        Add a lidar scan using ICP-based registration
        
        Args:
            points: Nx3 numpy array of points in sensor frame
            gps_position: (x, y, z) GPS position (used as initial guess)
            orientation: (w, x, y, z) quaternion orientation
        """
        if len(points) == 0 or len(points) < 100:
            return
        
        # Create point cloud from this scan
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(points)
        
        # Remove outliers
        scan_pcd, _ = scan_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample the scan
        scan_pcd = scan_pcd.voxel_down_sample(self.voxel_size)
        
        if len(scan_pcd.points) < 50:
            return
        
        # Compute normals for better ICP
        scan_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        
        # GPS-based transformation (initial guess)
        gps_transform = self._create_transform_matrix(gps_position, orientation)
        self.gps_poses.append(gps_transform)
        
        # First scan - just add it
        if self.scan_count == 0:
            scan_pcd.transform(gps_transform)
            # Don't replace reference, copy data to maintain visualizer connection
            self.global_map.points = scan_pcd.points
            self.global_map.colors = scan_pcd.colors if scan_pcd.has_colors() else o3d.utility.Vector3dVector([])
            self.global_map.normals = scan_pcd.normals if scan_pcd.has_normals() else o3d.utility.Vector3dVector([])
            self.local_maps.append(o3d.geometry.PointCloud(scan_pcd))
            self.poses.append(gps_transform)
            self.scan_count += 1
            return
        
        # ICP registration against recent local maps
        refined_transform = self._register_scan_icp(scan_pcd, gps_transform)
        
        # Transform scan with refined pose
        scan_pcd.transform(refined_transform)
        
        # Add to global map
        self.global_map += scan_pcd
        
        # Store for future ICP matching
        self.local_maps.append(scan_pcd)
        if len(self.local_maps) > self.max_local_maps:
            self.local_maps.pop(0)
        
        # Store pose
        self.poses.append(refined_transform)
        
        # Periodically downsample global map
        if self.scan_count % 10 == 0:
            self.global_map = self.global_map.voxel_down_sample(self.voxel_size)
            
        self.scan_count += 1
    
    def _register_scan_icp(self, scan_pcd, initial_transform):
        """
        Register scan against local map using ICP
        
        Args:
            scan_pcd: Current scan point cloud
            initial_transform: GPS-based initial transformation
            
        Returns:
            Refined transformation matrix
        """
        # Create target from recent local maps
        target = o3d.geometry.PointCloud()
        for local_map in self.local_maps[-3:]:  # Use last 3 scans
            target += local_map
        target = target.voxel_down_sample(self.voxel_size)
        
        if len(target.points) < 50:
            return initial_transform
        
        # Estimate normals for target
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        
        # Transform scan with initial guess
        source = o3d.geometry.PointCloud(scan_pcd)
        source.transform(initial_transform)
        
        # Point-to-plane ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=self.voxel_size * 3,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
        # Check if ICP succeeded
        if icp_result.fitness > self.icp_fitness_threshold:
            self.icp_success_count += 1
            # Combine ICP refinement with initial transform
            refined_transform = icp_result.transformation @ initial_transform
            
            # Calculate drift correction
            drift = np.linalg.norm(refined_transform[:3, 3] - initial_transform[:3, 3])
            self.total_drift += drift
            
            return refined_transform
        else:
            # ICP failed, use GPS pose
            return initial_transform
        
    def _create_transform_matrix(self, position, orientation):
        """Create 4x4 transformation matrix from position and quaternion"""
        # Convert quaternion to rotation matrix
        w, x, y, z = orientation
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        return T
    
    def get_map(self):
        """Get the current global map"""
        return self.global_map
    
    def colorize_by_height(self):
        """Color points by height (z coordinate)"""
        if len(self.global_map.points) == 0:
            return
        
        points = np.asarray(self.global_map.points)
        z_values = points[:, 2]
        
        # Normalize z values to 0-1 range
        z_min, z_max = z_values.min(), z_values.max()
        if z_max > z_min:
            z_norm = (z_values - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_values)
        
        # Create color map (blue = low, green = mid, red = high)
        colors = np.zeros((len(points), 3))
        colors[:, 2] = 1 - z_norm  # Blue decreases with height
        colors[:, 0] = z_norm      # Red increases with height
        colors[:, 1] = 1 - np.abs(z_norm - 0.5) * 2  # Green peaks at middle
        
        self.global_map.colors = o3d.utility.Vector3dVector(colors)
    
    def save_map(self, filename="map.ply"):
        """Save map to file"""
        # Final global downsampling
        final_map = self.global_map.voxel_down_sample(self.voxel_size * 0.8)
        
        # Remove statistical outliers
        final_map, _ = final_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        o3d.io.write_point_cloud(filename, final_map)
        print(f"Map saved to {filename}")
        print(f"Final map points: {len(final_map.points):,}")
        
    def get_statistics(self):
        """Get mapping statistics"""
        icp_success_rate = (self.icp_success_count / max(self.scan_count - 1, 1)) * 100
        avg_drift = self.total_drift / max(self.scan_count - 1, 1)
        
        return {
            'total_scans': self.scan_count,
            'total_points': len(self.global_map.points),
            'icp_success_rate': icp_success_rate,
            'average_drift_correction': avg_drift
        }


def get_lidar_scan(client):
    """Get lidar point cloud data"""
    try:
        lidar_data = client.getLidarData()
        
        if len(lidar_data.point_cloud) < 3:
            return None
        
        # Parse point cloud
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape((-1, 3))
        
        return points
    except Exception as e:
        return None


def get_camera_image(client):
    """Get camera image"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    
    if responses:
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb
    return None


def visualize_mapping(mapper, camera_img=None):
    """Create a visualization combining 3D map and camera view with SLAM stats"""
    vis_img = None
    
    if camera_img is not None:
        # Resize camera image for display
        vis_img = cv2.resize(camera_img, (640, 480))
        
        # Add mapping stats overlay
        stats = mapper.get_statistics()
        
        # Draw semi-transparent overlay
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
        vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(vis_img, "ICP-SLAM MAPPING", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_img, f"Points: {stats['total_points']:,}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"Scans: {stats['total_scans']}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"ICP Success: {stats['icp_success_rate']:.1f}%", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_img, f"Avg Drift Fix: {stats['average_drift_correction']:.3f}m", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return vis_img


def main():
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")
    
    # Enable API control
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(1)
    
    # Create ICP-SLAM mapper
    print("Initializing ICP-SLAM mapper...")
    print("Using GPS for initial pose + ICP for refinement")
    mapper = ICPSLAMMapper(voxel_size=0.15, icp_fitness_threshold=0.3)
    
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Define flight path for mapping
    waypoints = [
        (0, 0, -10),
        (10, 0, -10),
        (10, 10, -10),
        (0, 10, -10),
        (0, 0, -10),
        (5, 5, -15),  # Go higher for different perspective
        (0, 0, -10),
    ]
    
    print("Starting ICP-SLAM mapping flight...")
    print("Building map with scan registration...")
    print("ICP will refine GPS poses for accurate alignment")
    print("Press 'q' in camera view to stop early")
    
    # Create Open3D visualizer (non-blocking)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D ICP-SLAM Map", width=800, height=600)
    vis.add_geometry(mapper.global_map)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # Flag to reset view after first scan
    first_scan_received = False
    
    try:
        for i, waypoint in enumerate(waypoints):
            print(f"\nWaypoint {i+1}/{len(waypoints)}: {waypoint}")
            
            # Move to waypoint
            client.moveToPositionAsync(
                waypoint[0], waypoint[1], waypoint[2],
                velocity=3
            )
            
            # Collect scans while moving
            while True:
                # Get drone state
                state = client.getMultirotorState()
                position = state.kinematics_estimated.position
                pos_array = np.array([position.x_val, position.y_val, position.z_val])
                
                orientation = state.kinematics_estimated.orientation
                quat = np.array([orientation.w_val, orientation.x_val, 
                               orientation.y_val, orientation.z_val])
                
                # Get lidar scan
                points = get_lidar_scan(client)
                if points is not None and len(points) > 0:
                    prev_point_count = len(mapper.global_map.points)
                    mapper.add_scan(points, pos_array, quat)
                    
                    # Reset view after first scan to fix bounding box
                    if not first_scan_received and len(mapper.global_map.points) > 0:
                        first_scan_received = True
                        vis.reset_view_point(True)
                        print(f"3D view initialized - map now has {len(mapper.global_map.points)} points")
                    
                    # Only update visualization if points were added
                    if len(mapper.global_map.points) > prev_point_count:
                        # Update colors
                        mapper.colorize_by_height()
                        
                        # Update visualization
                        vis.update_geometry(mapper.global_map)
                    
                    vis.poll_events()
                    vis.update_renderer()
                
                # Show camera view
                camera_img = get_camera_image(client)
                if camera_img is not None:
                    vis_img = visualize_mapping(mapper, camera_img)
                    cv2.imshow("Camera + SLAM Info", vis_img)
                
                # Check if reached waypoint
                distance = np.sqrt(
                    (position.x_val - waypoint[0])**2 + 
                    (position.y_val - waypoint[1])**2 + 
                    (position.z_val - waypoint[2])**2
                )
                
                if distance < 1.0:
                    break
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                
                time.sleep(0.1)  # 10Hz scan rate
        
        print("\n" + "="*50)
        print("MAPPING COMPLETE!")
        print("="*50)
        
        stats = mapper.get_statistics()
        print(f"Total scans collected: {stats['total_scans']}")
        print(f"Total points in map: {stats['total_points']:,}")
        print(f"ICP success rate: {stats['icp_success_rate']:.1f}%")
        print(f"Average drift correction: {stats['average_drift_correction']:.3f}m")
        print("="*50)
        
        # Save map
        print("\nSaving final map...")
        mapper.save_map("airsim_icp_slam_map.ply")
        
        # Keep visualization open
        print("\nVisualization windows open.")
        print("Press 'q' in camera window to exit...")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            vis.poll_events()
            vis.update_renderer()
    
    except KeyboardInterrupt:
        print("\nMapping interrupted by user")
        if mapper.scan_count > 0:
            stats = mapper.get_statistics()
            print(f"\nPartial map statistics:")
            print(f"Scans collected: {stats['total_scans']}")
            print(f"ICP success rate: {stats['icp_success_rate']:.1f}%")
            mapper.save_map("airsim_icp_slam_map_partial.ply")
    
    finally:
        print("\nLanding...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
