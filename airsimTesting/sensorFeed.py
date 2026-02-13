"""
Autonomous waypoint navigation with camera and lidar feed display
"""
import cosysairsim as airsim
import cv2
import numpy as np
import time

def get_camera_image(client):
    """Get camera image from drone"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    
    if responses:
        response = responses[0]
        # Get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        # Reshape to RGB image
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb
    return None

def get_lidar_data(client):
    """Get lidar data and create 2D distance projection visualization"""
    try:
        lidar_data = client.getLidarData()
        
        if len(lidar_data.point_cloud) < 3:
            return None
    except Exception as e:
        # Lidar not configured or not available
        return None
    
    # Parse point cloud (x, y, z format)
    points = np.array(lidar_data.point_cloud, dtype=np.float32)
    points = points.reshape((-1, 3))
    
    # Create 2D projection (like a radar screen)
    img_size = 600
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = img_size // 2
    max_range = 50  # meters
    scale = (img_size // 2 - 50) / max_range  # pixels per meter
    
    # Draw range circles (every 10m)
    for radius in [10, 20, 30, 40, 50]:
        cv2.circle(img, (center, center), int(radius * scale), (50, 50, 50), 1)
        cv2.putText(img, f"{radius}m", (center + int(radius * scale) - 20, center - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Draw crosshairs
    cv2.line(img, (center, 0), (center, img_size), (50, 50, 50), 1)
    cv2.line(img, (0, center), (img_size, center), (50, 50, 50), 1)
    
    # Add direction labels
    cv2.putText(img, "FRONT", (center - 25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "BACK", (center - 20, img_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "LEFT", (15, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "RIGHT", (img_size - 60, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Project points onto 2D horizontal plane (ignore Z, use X-Y)
    for point in points:
        x, y, z = point
        
        # Calculate distance in horizontal plane
        distance = np.sqrt(x**2 + y**2)
        
        if distance > 0 and distance <= max_range:
            # Calculate angle
            angle = np.arctan2(y, x)  # angle in radians
            
            # Convert to screen coordinates
            px = int(center + distance * scale * np.sin(angle))
            py = int(center - distance * scale * np.cos(angle))
            
            if 0 <= px < img_size and 0 <= py < img_size:
                # Color based on distance: close = red, far = green
                if distance < 10:
                    color = (0, 0, 255)  # Red - close
                elif distance < 20:
                    color = (0, 165, 255)  # Orange
                elif distance < 30:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green - far
                
                cv2.circle(img, (px, py), 2, color, -1)
    
    # Draw drone at center
    cv2.circle(img, (center, center), 10, (255, 255, 255), -1)
    cv2.circle(img, (center, center), 8, (0, 255, 255), -1)
    # Forward direction arrow
    cv2.arrowedLine(img, (center, center), (center, center - 25), (0, 255, 255), 2, tipLength=0.3)
    
    # Calculate and display closest obstacle distance in each quadrant
    front_points = points[(points[:, 0] > 0)]  # X > 0
    back_points = points[(points[:, 0] < 0)]   # X < 0
    left_points = points[(points[:, 1] < 0)]   # Y < 0
    right_points = points[(points[:, 1] > 0)]  # Y > 0
    
    def get_min_distance(pts):
        if len(pts) > 0:
            distances = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            return np.min(distances)
        return max_range
    
    front_dist = get_min_distance(front_points)
    back_dist = get_min_distance(back_points)
    left_dist = get_min_distance(left_points)
    right_dist = get_min_distance(right_points)
    
    # Display distance measurements
    y_offset = img_size - 120
    cv2.rectangle(img, (5, y_offset - 5), (180, img_size - 5), (30, 30, 30), -1)
    cv2.putText(img, "DISTANCE SENSORS", (10, y_offset + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Front:  {front_dist:5.1f}m", (10, y_offset + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(img, f"Back:   {back_dist:5.1f}m", (10, y_offset + 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(img, f"Left:   {left_dist:5.1f}m", (10, y_offset + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(img, f"Right:  {right_dist:5.1f}m", (10, y_offset + 95),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    # Title and info
    cv2.putText(img, "LIDAR 2D DISTANCE MAP", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Points: {len(points)}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Color legend
    legend_x = img_size - 140
    cv2.rectangle(img, (legend_x, 10), (img_size - 10, 110), (30, 30, 30), -1)
    cv2.putText(img, "RANGE", (legend_x + 5, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(img, (legend_x + 15, 45), 4, (0, 0, 255), -1)
    cv2.putText(img, "< 10m", (legend_x + 25, 48),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.circle(img, (legend_x + 15, 63), 4, (0, 165, 255), -1)
    cv2.putText(img, "< 20m", (legend_x + 25, 66),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.circle(img, (legend_x + 15, 81), 4, (0, 255, 255), -1)
    cv2.putText(img, "< 30m", (legend_x + 25, 84),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.circle(img, (legend_x + 15, 99), 4, (0, 255, 0), -1)
    cv2.putText(img, "> 30m", (legend_x + 25, 102),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    return img

# Connect to AirSim
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")

# Enable API control and arm
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(1)

# Check if lidar is available
lidar_available = False
try:
    client.getLidarData()
    lidar_available = True
    print("Lidar detected!")
except:
    print("Lidar not available - will show camera only")

# Takeoff
print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

# Define waypoints (x, y, z in NED coordinates)
waypoints = [
    (0, 0, -10),      # Starting position - 10m up
    (20, 0, -10),     # Move 20m forward
    (20, 20, -10),    # Move 20m right
    (0, 20, -10),     # Move back 20m
    (0, 0, -10),      # Return to start
]

print("Starting waypoint navigation with sensor feeds...")
print("Press 'q' in any window to stop and land")

try:
    for i, waypoint in enumerate(waypoints):
        print(f"\nNavigating to waypoint {i+1}/{len(waypoints)}: {waypoint}")
        
        # Move to waypoint
        client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2], 
            velocity=5
        )
        
        # Display sensor feeds while moving
        while True:
            # Get camera image
            camera_img = get_camera_image(client)
            if camera_img is not None:
                cv2.imshow("Camera Feed", camera_img)
            
            # Get lidar visualization (if available)
            if lidar_available:
                lidar_img = get_lidar_data(client)
                if lidar_img is not None:
                    cv2.imshow("Lidar Top-Down View", lidar_img)
            
            # Check if reached waypoint
            pos = client.getMultirotorState().kinematics_estimated.position
            distance = np.sqrt(
                (pos.x_val - waypoint[0])**2 + 
                (pos.y_val - waypoint[1])**2 + 
                (pos.z_val - waypoint[2])**2
            )
            
            if distance < 1.0:  # Within 1 meter
                print(f"Reached waypoint {i+1}")
                break
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt
            
            time.sleep(0.05)  # 20Hz update
        
        # Hover at waypoint for 2 seconds
        print(f"Hovering at waypoint {i+1}...")
        hover_start = time.time()
        while time.time() - hover_start < 2.0:
            camera_img = get_camera_image(client)
            if camera_img is not None:
                cv2.imshow("Camera Feed", camera_img)
            
            if lidar_available:
                lidar_img = get_lidar_data(client)
                if lidar_img is not None:
                    cv2.imshow("Lidar Top-Down View", lidar_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt
            
            time.sleep(0.05)

    print("\nWaypoint navigation complete!")
    
except KeyboardInterrupt:
    print("\nStopping...")

finally:
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    cv2.destroyAllWindows()
    print("Done!")
