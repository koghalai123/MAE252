"""
3D Mapping with rtabmap using ROS2
Publishes lidar and GPS data to ROS topics for rtabmap SLAM
"""
import cosysairsim as airsim
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

class RtabmapMappingNode(Node):
    def __init__(self):
        super().__init__('rtabmap_mapping')
        
        # Publishers
        self.lidar_pub = self.create_publisher(PointCloud2, '/scan_cloud', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.odom_pub = self.create_publisher(Odometry, '/rtabmap/odom', 10)  # For odometry
        
        self.get_logger().info('Rtabmap Mapping Node initialized')
        
    def publish_lidar(self, points, timestamp_msg):
        """Publish lidar point cloud"""
        if len(points) == 0:
            return
            
        # Create PointCloud2 message
        header = Header()
        header.stamp = timestamp_msg
        header.frame_id = 'base_link'  # Adjust frame if needed
        
        # Convert points to list of tuples (x, y, z)
        cloud_points = [(p[0], p[1], p[2]) for p in points]
        
        pc2_msg = point_cloud2.create_cloud_xyz32(header, cloud_points)
        self.lidar_pub.publish(pc2_msg)
        
    def publish_gps(self, gps_data, timestamp_msg):
        """Publish GPS data"""
        msg = NavSatFix()
        msg.header.stamp = timestamp_msg
        msg.header.frame_id = 'base_link'
        msg.latitude = gps_data.gnss.geo_point.latitude
        msg.longitude = gps_data.gnss.geo_point.longitude
        msg.altitude = gps_data.gnss.geo_point.altitude
        # Add covariance if available
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        
        self.gps_pub.publish(msg)
        
    def publish_odom(self, position, orientation, timestamp_msg):
        """Publish odometry"""
        msg = Odometry()
        msg.header.stamp = timestamp_msg
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        
        msg.pose.pose.position.x = position[0]
        msg.pose.pose.position.y = position[1]
        msg.pose.pose.position.z = position[2]
        
        msg.pose.pose.orientation.w = orientation[0]
        msg.pose.pose.orientation.x = orientation[1]
        msg.pose.pose.orientation.y = orientation[2]
        msg.pose.pose.orientation.z = orientation[3]
        
        # Add covariance if available
        msg.pose.covariance = [0.1]*36  # Example covariance
        
        self.odom_pub.publish(msg)


def get_lidar_scan(client):
    """Get lidar point cloud data from AirSim"""
    try:
        lidar_data = client.getLidarData()
        
        if len(lidar_data.point_cloud) < 3:
            return None
        
        # Parse point cloud
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape((-1, 3))
        
        return points
    except Exception as e:
        print(f"Error getting lidar data: {e}")
        return None


def get_gps_data(client):
    """Get GPS data from AirSim"""
    try:
        gps_data = client.getGpsData()
        return gps_data
    except Exception as e:
        print(f"Error getting GPS data: {e}")
        return None


def main():
    # Initialize ROS2
    rclpy.init()
    
    # Create node
    node = RtabmapMappingNode()
    
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")
    
    # Enable API control
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(1)
    
    # Start mapping before taking off
    print("Starting mapping before takeoff...")
    for _ in range(10):  # 
        # Get drone state
        state = client.getMultirotorState()
        position = state.kinematics_estimated.position
        pos_array = np.array([position.x_val, position.y_val, position.z_val])
        
        orientation = state.kinematics_estimated.orientation
        quat = np.array([orientation.w_val, orientation.x_val, 
                       orientation.y_val, orientation.z_val])
        
        # Get timestamp
        timestamp_msg = node.get_clock().now().to_msg()
        
        # Publish pose/odometry
        node.publish_odom(pos_array, quat, timestamp_msg)
        
        # Get and publish lidar
        points = get_lidar_scan(client)
        if points is not None:
            node.publish_lidar(points, timestamp_msg)
        
        # Get and publish GPS
        gps_data = get_gps_data(client)
        if gps_data is not None:
            node.publish_gps(gps_data, timestamp_msg)
        
        # Spin ROS
        rclpy.spin_once(node, timeout_sec=0.1)
        
        time.sleep(0.1)  # 10Hz
    
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Define flight path
    waypoints = [
        (0, 0, -10),
        (10, 0, -10),
        (10, 10, -10),
        (0, 10, -10),
        (0, 0, -10),
    ]
    
    print("Starting mapping flight...")
    print("Publishing lidar and GPS data to ROS...")
    
    try:
        for i, waypoint in enumerate(waypoints):
            print(f"Waypoint {i+1}/{len(waypoints)}: {waypoint}")
            
            # Move to waypoint
            client.moveToPositionAsync(
                waypoint[0], waypoint[1], waypoint[2],
                velocity=3
            )
            
            # Collect and publish data while moving
            while True:
                # Get drone state
                state = client.getMultirotorState()
                position = state.kinematics_estimated.position
                pos_array = np.array([position.x_val, position.y_val, position.z_val])
                
                orientation = state.kinematics_estimated.orientation
                quat = np.array([orientation.w_val, orientation.x_val, 
                               orientation.y_val, orientation.z_val])
                
                # Get timestamp
                timestamp_msg = node.get_clock().now().to_msg()
                
                # Publish pose/odometry
                node.publish_odom(pos_array, quat, timestamp_msg)
                
                # Get and publish lidar
                points = get_lidar_scan(client)
                if points is not None:
                    node.publish_lidar(points, timestamp_msg)
                
                # Get and publish GPS
                gps_data = get_gps_data(client)
                if gps_data is not None:
                    node.publish_gps(gps_data, timestamp_msg)
                
                # Check if reached waypoint
                distance = np.sqrt(
                    (position.x_val - waypoint[0])**2 + 
                    (position.y_val - waypoint[1])**2 + 
                    (position.z_val - waypoint[2])**2
                )
                
                if distance < 1.0:
                    break
                
                # Spin ROS
                rclpy.spin_once(node, timeout_sec=0.1)
                
                time.sleep(0.1)  # 10Hz
        
        print("Mapping complete!")
        
        # Keep publishing for a bit
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        
        node.destroy_node()
        rclpy.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()