"""
3D Mapping using Open3D SLAM, Lidar, and GPS
This script accumulates lidar scans, uses GPS for global alignment, and visualizes the map with Open3D.
RGB images are used for visualization only.
"""
import cosysairsim as airsim
import numpy as np
import time
import cv2
import open3d as o3d
import os

class OccupancyGridMapper:
    """Accumulates lidar scans into a 3D occupancy grid (voxel grid) with intermediate, occupied, and unoccupied states."""
    def __init__(self, voxel_size=2.0, grid_extent=60):
        self.voxel_size = voxel_size
        self.scan_count = 0
        self.origin = np.zeros(3)
        self.grid_extent = grid_extent
        self.grid_shape = (2*grid_extent, 2*grid_extent, 2*grid_extent)
        # 0: unknown/intermediate, 1: occupied, -1: free/unoccupied
        self.grid = np.zeros(self.grid_shape, dtype=np.int8)

    def add_scan(self, points, position, orientation):
        """
        Add a lidar scan to the occupancy grid, aligned using odometry (position + orientation).
        Args:
            points: Nx3 numpy array of points in sensor frame
            position: (x, y, z) in world frame (from odometry)
            orientation: (w, x, y, z) quaternion (from odometry)
        """
        if len(points) == 0:
            return
        # Transform points to world frame
        transform = self._create_transform_matrix(position, orientation)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # Nx4
        points_world = (transform @ points_h.T).T[:, :3]
        # Convert to voxel indices (centered at origin)
        indices = np.floor((points_world - self.origin) / self.voxel_size).astype(int) + self.grid_extent
        # Mark occupied
        for idx in indices:
            ix, iy, iz = idx
            if 0 <= ix < self.grid_shape[0] and 0 <= iy < self.grid_shape[1] and 0 <= iz < self.grid_shape[2]:
                self.grid[ix, iy, iz] = 1
        self.scan_count += 1

    def get_voxel_grid(self):
        """Return an Open3D VoxelGrid for visualization with alpha blending and state coloring."""
        # Colors: occupied = blue (opaque)
        pts = []
        colors = []
        for ix in range(self.grid_shape[0]):
            for iy in range(self.grid_shape[1]):
                for iz in range(self.grid_shape[2]):
                    state = self.grid[ix, iy, iz]
                    if state == 1:
                        # occupied
                        pts.append((np.array([ix, iy, iz]) - self.grid_extent) * self.voxel_size + self.origin + self.voxel_size/2)
                        colors.append([0.2, 0.4, 1.0])
        if not pts:
            return o3d.geometry.VoxelGrid()
        pts = np.array(pts)
        colors = np.array(colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        return voxel_grid

    def get_occupied_count(self):
        return np.sum(self.grid == 1)

    def _create_transform_matrix(self, position, orientation):
        w, x, y, z = orientation
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    def save_map(self, filename="occupancy_grid.ply"):
        # Save only occupied voxel centers as a point cloud
        pts = []
        for ix in range(self.grid_shape[0]):
            for iy in range(self.grid_shape[1]):
                for iz in range(self.grid_shape[2]):
                    if self.grid[ix, iy, iz] == 1:
                        pts.append((np.array([ix, iy, iz]) - self.grid_extent) * self.voxel_size + self.origin + self.voxel_size/2)
        if not pts:
            print("No voxels to save.")
            return
        pts = np.array(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Occupancy grid saved to {filename}")

def get_lidar_scan(client):
    try:
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            print("[DEBUG] No lidar points received.")
            return None
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape((-1, 3))
        print(f"[DEBUG] Lidar points shape: {points.shape}")
        return points
    except Exception as e:
        print(f"[DEBUG] Lidar exception: {e}")
        return None

def get_gps_position(client):
    gps_data = client.getGpsData()
    pos = gps_data.gnss.geo_point
    # Convert GPS to local NED (or use AirSim's local position)
    # Here, we use AirSim's local position for simplicity
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    return np.array([position.x_val, position.y_val, position.z_val])

def get_camera_image(client):
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
    vis_img = None
    if camera_img is not None:
        vis_img = cv2.resize(camera_img, (640, 480))
        map_voxels = mapper.get_occupied_count()
        scan_count = mapper.scan_count
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
        cv2.putText(vis_img, "Occupancy Grid Mapping", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_img, f"Voxels: {map_voxels:,}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"Scans: {scan_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis_img

def main():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(1)
    print("Initializing Open3D SLAM mapper...")
    mapper = OccupancyGridMapper(voxel_size=2.0, grid_extent=60)
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    waypoints = [
        (0, 0, -10),
        (10, 0, -10),
        (10, 10, -10),
        (0, 10, -10),
        (0, 0, -10),
        (5, 5, -15),
        (0, 0, -10),
    ]
    print("Starting Open3D SLAM mapping flight...")
    print("Press 'q' in camera view to stop early")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Map", width=800, height=600)
    voxel_grid = mapper.get_voxel_grid()
    vis.add_geometry(voxel_grid)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    try:
        for i, waypoint in enumerate(waypoints):
            print(f"\nWaypoint {i+1}/{len(waypoints)}: {waypoint}")
            client.moveToPositionAsync(
                waypoint[0], waypoint[1], waypoint[2],
                velocity=3
            )
            while True:
                state = client.getMultirotorState()
                position = state.kinematics_estimated.position
                orientation = state.kinematics_estimated.orientation
                pos_array = np.array([position.x_val, position.y_val, position.z_val])
                quat = np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
                points = get_lidar_scan(client)
                if points is not None and len(points) > 0:
                    mapper.add_scan(points, pos_array, quat)
                    # Save current camera parameters
                    ctr = vis.get_view_control()
                    cam_params = ctr.convert_to_pinhole_camera_parameters()
                    # Remove old voxel grid and add new one
                    vis.remove_geometry(voxel_grid, reset_bounding_box=False)
                    voxel_grid = mapper.get_voxel_grid()
                    vis.add_geometry(voxel_grid)
                    # Restore camera parameters
                    ctr.convert_from_pinhole_camera_parameters(cam_params)
                    vis.poll_events()
                    vis.update_renderer()
                camera_img = get_camera_image(client)
                if camera_img is not None:
                    vis_img = visualize_mapping(mapper, camera_img)
                    cv2.imshow("Camera + Mapping Info", vis_img)
                distance = np.sqrt(
                    (position.x_val - waypoint[0])**2 + 
                    (position.y_val - waypoint[1])**2 + 
                    (position.z_val - waypoint[2])**2
                )
                if distance < 1.0:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                time.sleep(0.1)
        print("\nMapping complete!")
        print(f"Total scans collected: {mapper.scan_count}")
        print(f"Total occupied voxels: {mapper.get_occupied_count():,}")
        mapper.save_map("occupancy_grid.ply")
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
            mapper.save_map("open3d_map_partial.ply")
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
