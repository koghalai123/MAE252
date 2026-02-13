"""
3D Mapping using Lidar and Camera visualization
Uses Open3D for point cloud accumulation and visualization
"""
import cosysairsim as airsim
import numpy as np
import time
import cv2

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Open3D not installed. Install with: pip install open3d")
    print("Open3D is recommended for 3D mapping and visualization")
    OPEN3D_AVAILABLE = False
    exit(1)


class LidarMapper:
    """Accumulates lidar scans into a 3D map"""
    
    def __init__(self, voxel_size=0.1, max_points=1000000):
        """
        Args:
            voxel_size: Size of voxel for downsampling (meters)
            max_points: Maximum number of points to keep in map
        """
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.global_map = o3d.geometry.PointCloud()
        self.scan_count = 0
        
    def add_scan(self, points, position, orientation):
        """
        Add a lidar scan to the global map
        
        Args:
            points: Nx3 numpy array of points in sensor frame
            position: (x, y, z) drone position in world frame
            orientation: (w, x, y, z) quaternion orientation
        """
        if len(points) == 0:
            return
        
        # Create point cloud from this scan
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = o3d.utility.Vector3dVector(points)
        
        # Transform from sensor frame to world frame
        # Create transformation matrix from position and orientation
        transform = self._create_transform_matrix(position, orientation)
        scan_pcd.transform(transform)
        
        # Add to global map
        self.global_map += scan_pcd
        
        # Downsample to prevent map from growing too large
        if len(self.global_map.points) > self.max_points:
            self.global_map = self.global_map.voxel_down_sample(self.voxel_size)
        
        self.scan_count += 1
        
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
        o3d.io.write_point_cloud(filename, self.global_map)
        print(f"Map saved to {filename}")


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
    """Create a visualization combining 3D map and camera view"""
    vis_img = None
    
    if camera_img is not None:
        # Resize camera image for display
        vis_img = cv2.resize(camera_img, (640, 480))
        
        # Add mapping stats overlay
        map_points = len(mapper.global_map.points)
        scan_count = mapper.scan_count
        
        # Draw semi-transparent overlay
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(vis_img, "3D LIDAR MAPPING", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_img, f"Points: {map_points:,}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"Scans: {scan_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
    
    # Create mapper
    print("Initializing 3D mapper...")
    mapper = LidarMapper(voxel_size=0.2, max_points=500000)
    
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
    
    print("Starting 3D mapping flight...")
    print("Building map from lidar scans...")
    print("Press 'q' in camera view to stop early")
    
    # Create Open3D visualizer (non-blocking)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Map", width=800, height=600)
    vis.add_geometry(mapper.global_map)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
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
                    mapper.add_scan(points, pos_array, quat)
                    
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
                    cv2.imshow("Camera + Mapping Info", vis_img)
                
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
        
        print("\nMapping complete!")
        print(f"Total points in map: {len(mapper.global_map.points):,}")
        print(f"Total scans collected: {mapper.scan_count}")
        
        # Save map
        mapper.save_map("airsim_map.ply")
        
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
            mapper.save_map("airsim_map_partial.ply")
    
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
