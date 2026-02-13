"""
Keyboard teleoperation for CoSys-AirSim drone
Uses moveByManualAsync and RCData for proper control
Displays camera and lidar feeds
"""
import cosysairsim as airsim
import time
from pynput import keyboard
import cv2
import numpy as np

# Track which keys are currently pressed
keys_pressed = set()
running = True

# Current smooth control values
current_pitch = 0.0
current_roll = 0.0
current_yaw = 0.0
current_throttle = 0.5

# Smoother control values (reduced from 0.5 to 0.25 for smoother movement)
CONTROL_STRENGTH = 0.25
THROTTLE_ADJUST = 0.1
SMOOTHING_FACTOR = 0.15  # Lower = smoother but less responsive (0.1-0.3 range)

def on_press(key):
    global keys_pressed
    try:
        if hasattr(key, 'char') and key.char:
            keys_pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    global keys_pressed, running
    if key == keyboard.Key.esc:
        running = False
        return False
    try:
        if hasattr(key, 'char') and key.char:
            keys_pressed.discard(key.char)
    except AttributeError:
        pass

def get_control_values():
    """Calculate RC values based on currently pressed keys"""
    global current_pitch, current_roll, current_yaw, current_throttle
    
    # Target values based on key presses
    target_pitch = 0.0
    target_roll = 0.0
    target_yaw = 0.0
    target_throttle = 0.5
    
    # Calculate pitch (forward/backward)
    if 'w' in keys_pressed:
        target_pitch -= CONTROL_STRENGTH
    if 's' in keys_pressed:
        target_pitch += CONTROL_STRENGTH
    
    # Calculate roll (left/right)
    if 'a' in keys_pressed:
        target_roll -= CONTROL_STRENGTH
    if 'd' in keys_pressed:
        target_roll += CONTROL_STRENGTH
    
    # Calculate yaw (rotation)
    if 'q' in keys_pressed:
        target_yaw -= CONTROL_STRENGTH
    if 'e' in keys_pressed:
        target_yaw += CONTROL_STRENGTH
    
    # Calculate throttle (up/down)
    if 'z' in keys_pressed:
        target_throttle += THROTTLE_ADJUST
    if 'x' in keys_pressed:
        target_throttle -= THROTTLE_ADJUST
    
    # Smooth interpolation (exponential smoothing)
    current_pitch += (target_pitch - current_pitch) * SMOOTHING_FACTOR
    current_roll += (target_roll - current_roll) * SMOOTHING_FACTOR
    current_yaw += (target_yaw - current_yaw) * SMOOTHING_FACTOR
    current_throttle += (target_throttle - current_throttle) * SMOOTHING_FACTOR
    
    return current_pitch, current_roll, current_yaw, current_throttle

# Connect to AirSim
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")

# Enable API control and arm
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(1)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

# Setup manual mode
print("Setting up manual control mode...")
client.moveByManualAsync(
    vx_max=1E6,
    vy_max=1E6,
    z_min=-1E6,
    duration=1E10
)

# Start keyboard listener
print("\n=== CONTROLS ===")
print("W/S: Forward/Backward")
print("A/D: Left/Right")
print("Q/E: Rotate Left/Right")
print("Z/X: Up/Down")
print("ESC: Land and exit")
print("================\n")

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while running:
        # Get current control values based on pressed keys
        pitch, roll, yaw, throttle = get_control_values()
        
        # Send RC data continuously
        client.moveByRC(rcdata=airsim.RCData(
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            throttle=throttle,
            is_initialized=True,
            is_valid=True
        ))
        time.sleep(0.02)  # 50Hz update rate for smoother control
        
except KeyboardInterrupt:
    pass
finally:
    print("\nLanding...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Landed and released control")
