import cosysairsim as airsim
import time

# Connect to AirSim
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")

# Enable API control
print("Enabling API control...")
client.enableApiControl(True)
time.sleep(1)

# Arm the drone
print("Arming...")
client.armDisarm(True)
time.sleep(1)

# Take off
print("Taking off...")
client.takeoffAsync().join()
print("Takeoff complete!")
time.sleep(2)

# Move forward 10 meters
print("Moving forward...")
client.moveByVelocityAsync(5, 0, 0, 5).join()
print("Movement complete!")
time.sleep(2)

# Land
print("Landing...")
client.landAsync().join()
print("Landed!")

# Cleanup
client.armDisarm(False)
client.enableApiControl(False)
print("Done!")