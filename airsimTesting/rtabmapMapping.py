"""
3D Mapping with rtabmap using ROS2
Publishes lidar and GPS data to ROS topics for rtabmap SLAM
"""


#Currently uses this launch file: https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_examples/launch/lidar3d_assemble.launch.py
# with this launch command: ros2 launch rtabmap_examples lidar3d_assemble.launch.py deskewing:=false

import cosysairsim as airsim
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, NavSatFix, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

class RtabmapMappingNode(Node):
    def __init__(self):
        super().__init__('rtabmap_mapping')
        
        # Publishers
        self.lidar_pub = self.create_publisher(PointCloud2, '/velodyne_points', 10)
        # publish directly to the deskewed topic so rtabmap's assembler/icp gets data
        self.lidar_deskewed_pub = self.create_publisher(PointCloud2, '/velodyne_points/deskewed', 10)
        self.odom_pub = self.create_publisher(Odometry, '/icp_odom', 10)  # For odometry
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        # publish IMU on both common conventions so icp_odometry and imu_to_tf see it
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.imu_pub_data = self.create_publisher(Imu, '/imu/data', 10)

        self.get_logger().info('Rtabmap Mapping Node initialized')

    def publish_gps(self, gps_data, timestamp_msg):
        """Publish GPS data"""
        msg = NavSatFix()
        msg.header.stamp = timestamp_msg
        msg.header.frame_id = 'velodyne'
        # AirSim GPS structure (supports both wrapped and simple forms)
        try:
            msg.latitude = gps_data.gnss.geo_point.latitude
            msg.longitude = gps_data.gnss.geo_point.longitude
            msg.altitude = gps_data.gnss.geo_point.altitude
        except Exception:
            msg.latitude = getattr(gps_data, 'latitude', 0.0)
            msg.longitude = getattr(gps_data, 'longitude', 0.0)
            msg.altitude = getattr(gps_data, 'altitude', 0.0)
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.gps_pub.publish(msg)

    def publish_imu(self, imu_data, orientation, timestamp_msg):
        """Publish full IMU message using orientation (quat) + AirSim IMU values if available"""
        msg = Imu()
        msg.header.stamp = timestamp_msg
        msg.header.frame_id = 'velodyne'

        # orientation is expected as [w, x, y, z]
        msg.orientation.w = float(orientation[0])
        msg.orientation.x = float(orientation[1])
        msg.orientation.y = float(orientation[2])
        msg.orientation.z = float(orientation[3])

        # Fill angular_velocity and linear_acceleration from AirSim IMU when available
        if imu_data is not None:
            try:
                la = imu_data.linear_acceleration
                av = imu_data.angular_velocity
                msg.linear_acceleration.x = float(la.x_val)
                msg.linear_acceleration.y = float(la.y_val)
                msg.linear_acceleration.z = float(la.z_val)
                msg.angular_velocity.x = float(av.x_val)
                msg.angular_velocity.y = float(av.y_val)
                msg.angular_velocity.z = float(av.z_val)
            except Exception:
                msg.linear_acceleration.x = msg.linear_acceleration.y = msg.linear_acceleration.z = 0.0
                msg.angular_velocity.x = msg.angular_velocity.y = msg.angular_velocity.z = 0.0
        else:
            msg.linear_acceleration.x = msg.linear_acceleration.y = msg.linear_acceleration.z = 0.0
            msg.angular_velocity.x = msg.angular_velocity.y = msg.angular_velocity.z = 0.0

        # leave covariance as default (unknown)
        self.imu_pub.publish(msg)
        # also publish on /imu/data for nodes that expect that topic
        try:
            self.imu_pub_data.publish(msg)
        except Exception:
            pass
        
    def publish_lidar(self, points, timestamp_msg):
        """Publish lidar point cloud"""
        if len(points) == 0:
            return
            
        # Create PointCloud2 message
        header = Header()
        header.stamp = timestamp_msg
        header.frame_id = 'velodyne'  # Match expected frame_id
        
        # Convert points to list of tuples (x, y, z)
        cloud_points = [(p[0], p[1], p[2]) for p in points]
        
        pc2_msg = point_cloud2.create_cloud_xyz32(header, cloud_points)
        # publish on both raw and deskewed topics (deskewed is used by assembler/icp)
        self.lidar_pub.publish(pc2_msg)
        try:
            self.lidar_deskewed_pub.publish(pc2_msg)
        except Exception:
            pass
        
    def publish_odom(self, position, orientation, timestamp_msg):
        """Publish odometry"""
        msg = Odometry()
        msg.header.stamp = timestamp_msg
        msg.header.frame_id = 'icp_odom'
        msg.child_frame_id = 'velodyne'
        
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
    """Get GPS data from AirSim (module-level helper)."""
    try:
        return client.getGpsData()
    except Exception as e:
        print(f"Error getting GPS data: {e}")
        return None


def get_imu_data(client, sensor_name: str = None):
    """Get IMU data from AirSim. Pass sensor_name if you configured multiple IMUs.
    Returns AirSim ImuData or None on error.
    """
    try:
        if sensor_name:
            return client.getImuData(sensor_name)
        return client.getImuData()
    except Exception as e:
        print(f"Error getting IMU data: {e}")
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
    for _ in range(10):
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
        # Get and publish IMU (angular velocity + linear acceleration + orientation)
        imu_data = get_imu_data(client)
        node.publish_imu(imu_data, quat, timestamp_msg)
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
        (10, -30, -10),
        (0, -30, -10),
        (0, 0, -10),
    ]
    
    print("Starting mapping flight...")
    print("Publishing lidar and odometry data to ROS...")
    
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
                # Get and publish IMU (angular velocity + linear acceleration + orientation)
                imu_data = get_imu_data(client)
                node.publish_imu(imu_data, quat, timestamp_msg)
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