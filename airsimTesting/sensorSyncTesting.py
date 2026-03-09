#!/usr/bin/env python3
"""Diagnose scan-to-pose yaw mismatch with simPause + GT kinematics.

Hypothesis: getLidarData() returns the LAST completed scan cycle, which
may have been captured at a different yaw than the GT pose at the pause
instant.  This script:

1. Flies the drone on a turning path to maximise yaw rate.
2. For each simPause sample, reads:
   - GT kinematics (pose at frozen instant)
   - LiDAR data (point cloud from the last completed cycle)
   - Two consecutive GT reads to check if orientation changes mid-pause
3. Compares the LiDAR timestamp vs. the MR state timestamp to measure
   how stale the scan is.
4. Estimates how much yaw the drone rotated during that staleness window,
   using the GT angular velocity.
5. Optionally checks if consecutive scans' point clouds overlap correctly
   when transformed with the GT pose.

Key outputs per sample:
  - GT yaw at pause instant
  - LiDAR staleness (ms) = MR timestamp - LiDAR timestamp
  - GT angular velocity (yaw rate, deg/s)
  - Estimated yaw error = staleness × yaw_rate
"""

import cosysairsim as airsim
import numpy as np
import time
from scipy.spatial.transform import Rotation

def quat_to_ypr(q_wxyz):
    """Convert [w,x,y,z] quaternion to [yaw, pitch, roll] in degrees."""
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    # ZYX intrinsic = yaw-pitch-roll
    return r.as_euler('ZYX', degrees=True)

def _to_ns(ts):
    if ts > 1e18:
        return ts
    elif ts > 1e15:
        return ts * 1e3
    elif ts > 1e12:
        return ts * 1e6
    else:
        return ts * 1e9

# ── Connect ──────────────────────────────────────────────────────────────────
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(0.5)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# Rise to altitude
print("Rising to z=-8...")
client.moveToPositionAsync(0, 0, -8, 3).join()
time.sleep(0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Stationary — baseline (no rotation, scans should align perfectly)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  TEST 1 — STATIONARY (hovering, no rotation)")
print("  If yaw error is non-zero here, the problem is NOT motion-related.")
print("="*100)

header = (f"{'#':>3}  {'GT yaw (°)':>10}  {'lidar stale (ms)':>16}  "
          f"{'yaw rate (°/s)':>14}  {'est yaw err (°)':>15}  "
          f"{'GT pos (x,y,z)':>28}")
print(header)
print("-" * len(header))

prev_gt_ori = None

for i in range(10):
    client.simPause(True)

    lidar_data = client.getLidarData()
    gt = client.simGetGroundTruthKinematics()
    mr = client.getMultirotorState()

    client.simPause(False)

    gt_ori = np.array([gt.orientation.w_val, gt.orientation.x_val,
                       gt.orientation.y_val, gt.orientation.z_val])
    gt_pos = np.array([gt.position.x_val, gt.position.y_val, gt.position.z_val])
    gt_angvel = np.array([gt.angular_velocity.x_val, gt.angular_velocity.y_val,
                          gt.angular_velocity.z_val])

    ypr = quat_to_ypr(gt_ori)
    gt_yaw = ypr[0]

    lidar_ns = _to_ns(float(lidar_data.time_stamp))
    mr_ns = _to_ns(float(mr.timestamp))
    stale_ms = (mr_ns - lidar_ns) / 1e6

    # Yaw rate in deg/s (angular velocity Z in NED = yaw rate)
    yaw_rate = np.degrees(gt_angvel[2])

    est_yaw_err = stale_ms / 1000.0 * yaw_rate

    print(f"{i+1:3d}  {gt_yaw:10.3f}  {stale_ms:16.3f}  "
          f"{yaw_rate:14.3f}  {est_yaw_err:15.4f}  "
          f"({gt_pos[0]:7.2f},{gt_pos[1]:7.2f},{gt_pos[2]:7.2f})")

    prev_gt_ori = gt_ori
    time.sleep(0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Flying a turning path — maximise yaw rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  TEST 2 — TURNING FLIGHT (zigzag path to induce yaw rotation)")
print("  High yaw rate + LiDAR staleness = large yaw error.")
print("="*100)

# Issue async moves from the main thread — the drone keeps flying between
# simPause samples.  No separate thread needed (avoids asyncio loop issues).
waypoints = [
    (10,  10, -8, 4),
    (-5, -15, -8, 4),
    (15,  -5, -8, 4),
    (-10, 10, -8, 4),
    (10, -10, -8, 4),
]

# Kick off the first waypoint (async — returns immediately)
wp_idx = 0
flight_task = client.moveToPositionAsync(
    waypoints[wp_idx][0], waypoints[wp_idx][1],
    waypoints[wp_idx][2], waypoints[wp_idx][3])
time.sleep(0.5)  # let the drone start accelerating

print(header)
print("-" * len(header))

samples = []
for i in range(30):
    client.simPause(True)

    lidar_data = client.getLidarData()
    gt = client.simGetGroundTruthKinematics()
    mr = client.getMultirotorState()

    client.simPause(False)

    gt_ori = np.array([gt.orientation.w_val, gt.orientation.x_val,
                       gt.orientation.y_val, gt.orientation.z_val])
    gt_pos = np.array([gt.position.x_val, gt.position.y_val, gt.position.z_val])
    gt_angvel = np.array([gt.angular_velocity.x_val, gt.angular_velocity.y_val,
                          gt.angular_velocity.z_val])

    ypr = quat_to_ypr(gt_ori)
    gt_yaw = ypr[0]

    lidar_ns = _to_ns(float(lidar_data.time_stamp))
    mr_ns = _to_ns(float(mr.timestamp))
    stale_ms = (mr_ns - lidar_ns) / 1e6

    yaw_rate = np.degrees(gt_angvel[2])
    est_yaw_err = stale_ms / 1000.0 * yaw_rate

    n_pts = len(lidar_data.point_cloud) // 3

    print(f"{i+1:3d}  {gt_yaw:10.3f}  {stale_ms:16.3f}  "
          f"{yaw_rate:14.3f}  {est_yaw_err:15.4f}  "
          f"({gt_pos[0]:7.2f},{gt_pos[1]:7.2f},{gt_pos[2]:7.2f})")

    samples.append({
        "gt_yaw": gt_yaw, "stale_ms": stale_ms,
        "yaw_rate": yaw_rate, "est_yaw_err": est_yaw_err,
        "gt_ori": gt_ori, "gt_pos": gt_pos, "n_pts": n_pts,
    })

    # Advance to next waypoint if close to current target
    wp = waypoints[wp_idx]
    dist_to_wp = np.linalg.norm(gt_pos - np.array([wp[0], wp[1], wp[2]]))
    if dist_to_wp < 3.0 and wp_idx < len(waypoints) - 1:
        wp_idx += 1
        wp = waypoints[wp_idx]
        flight_task = client.moveToPositionAsync(wp[0], wp[1], wp[2], wp[3])

    time.sleep(0.4)

# ── Summary statistics ───────────────────────────────────────────────────────
stales = [s["stale_ms"] for s in samples]
yaw_rates = [abs(s["yaw_rate"]) for s in samples]
yaw_errs = [abs(s["est_yaw_err"]) for s in samples]

print(f"\n{'─'*80}")
print(f"  LiDAR staleness:  mean={np.mean(stales):.1f} ms, "
      f"max={np.max(stales):.1f} ms, min={np.min(stales):.1f} ms")
print(f"  |Yaw rate|:       mean={np.mean(yaw_rates):.1f} °/s, "
      f"max={np.max(yaw_rates):.1f} °/s")
print(f"  |Est. yaw error|: mean={np.mean(yaw_errs):.2f}°, "
      f"max={np.max(yaw_errs):.2f}°, min={np.min(yaw_errs):.2f}°")
print()

if np.max(yaw_errs) > 1.0:
    print("  ⚠  Significant yaw error detected!  The GT pose at pause time")
    print("     does not match the LiDAR scan's actual capture orientation.")
    print("     SOLUTION: Compensate by rotating the point cloud by the")
    print("     estimated yaw error, or use getMultirotorState() at the")
    print("     LiDAR timestamp instead of the pause-time GT.")
elif np.max(yaw_errs) > 0.2:
    print("  ⚡ Small yaw error — may be visible in tight spaces.")
else:
    print("  ✓  Yaw error is negligible — simPause approach should work well.")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Check if consecutive GT reads while paused give the same pose
#         (rules out "GT changes during pause" as a cause)
# ═══════════════════════════════════════════════════════════════════════════════
print("="*100)
print("  TEST 3 — Multiple GT reads within a single simPause")
print("  Checks whether the GT pose drifts during a pause.")
print("="*100)

client.moveToPositionAsync(5, 5, -8, 3)
time.sleep(2)

for i in range(5):
    client.simPause(True)

    gt1 = client.simGetGroundTruthKinematics()
    time.sleep(0.1)  # deliberate delay while paused
    gt2 = client.simGetGroundTruthKinematics()
    time.sleep(0.1)
    gt3 = client.simGetGroundTruthKinematics()

    client.simPause(False)

    pos1 = np.array([gt1.position.x_val, gt1.position.y_val, gt1.position.z_val])
    pos2 = np.array([gt2.position.x_val, gt2.position.y_val, gt2.position.z_val])
    pos3 = np.array([gt3.position.x_val, gt3.position.y_val, gt3.position.z_val])
    ori1 = np.array([gt1.orientation.w_val, gt1.orientation.x_val,
                      gt1.orientation.y_val, gt1.orientation.z_val])
    ori3 = np.array([gt3.orientation.w_val, gt3.orientation.x_val,
                      gt3.orientation.y_val, gt3.orientation.z_val])

    pos_drift = np.linalg.norm(pos3 - pos1)
    ori_diff = np.abs(quat_to_ypr(ori1) - quat_to_ypr(ori3))

    print(f"  Sample {i+1}: pos drift = {pos_drift:.6f} m, "
          f"yaw drift = {ori_diff[0]:.6f}°, "
          f"pitch drift = {ori_diff[1]:.6f}°, "
          f"roll drift = {ori_diff[2]:.6f}°")

    time.sleep(0.5)

print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Compare LiDAR timestamp to previous scan — detect duplicate scans
# ═══════════════════════════════════════════════════════════════════════════════
print("="*100)
print("  TEST 4 — Consecutive scan timestamps (detect duplicates / stale data)")
print("  If getLidarData() returns the same scan twice, subsequent reads during")
print("  rapid polling will have identical timestamps but different GT poses.")
print("="*100)

prev_lidar_ts = 0
dup_count = 0

client.moveToPositionAsync(-5, -5, -8, 3)
time.sleep(1)

for i in range(20):
    client.simPause(True)
    ld = client.getLidarData()
    gt = client.simGetGroundTruthKinematics()
    client.simPause(False)

    lidar_ts = float(ld.time_stamp)
    gt_yaw = quat_to_ypr(np.array([gt.orientation.w_val, gt.orientation.x_val,
                                     gt.orientation.y_val, gt.orientation.z_val]))[0]
    n_pts = len(ld.point_cloud) // 3

    is_dup = (lidar_ts == prev_lidar_ts)
    if is_dup:
        dup_count += 1
    tag = " ** DUPLICATE **" if is_dup else ""

    print(f"  {i+1:3d}  lidar_ts={lidar_ts:18.0f}  yaw={gt_yaw:8.3f}°  "
          f"pts={n_pts:6d}{tag}")

    prev_lidar_ts = lidar_ts
    time.sleep(0.15)  # fast polling — intentionally faster than LiDAR update rate

print(f"\n  Duplicate scans detected: {dup_count} / 20")
if dup_count > 0:
    print("  ⚠  Duplicate scans mean getLidarData() returns STALE data when polled")
    print("     faster than the sensor update rate.  The GT pose keeps changing but")
    print("     the point cloud is the same → yaw mismatch!")
    print("     FIX: Skip scans with duplicate timestamps.")
else:
    print("  ✓  No duplicates — each read returns a fresh scan.")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Compare simGetGroundTruthKinematics vs kinematics_estimated ORIENTATION
#         (This is the key difference between the old and new approaches)
# ═══════════════════════════════════════════════════════════════════════════════
print("="*100)
print("  TEST 5 — GT kinematics vs kinematics_estimated ORIENTATION comparison")
print("  Old approach used: getMultirotorState().kinematics_estimated.orientation")
print("  New approach uses: simGetGroundTruthKinematics().orientation")
print("  If these differ, that's why scans are oriented wrong.")
print("="*100)

# Fly a turning path to induce yaw changes
flight_task = client.moveToPositionAsync(10, 10, -8, 4)
time.sleep(1)

print(f"{'#':>3}  {'GT yaw':>8}  {'KE yaw':>8}  {'Δyaw (°)':>9}  "
      f"{'GT pitch':>8}  {'KE pitch':>8}  {'Δpitch':>8}  "
      f"{'GT roll':>8}  {'KE roll':>8}  {'Δroll':>8}  "
      f"{'pos diff':>8}")
print("-" * 110)

for i in range(20):
    client.simPause(True)

    gt = client.simGetGroundTruthKinematics()
    mr = client.getMultirotorState()

    client.simPause(False)

    # GT kinematics orientation (what simPause approach uses)
    gt_ori = np.array([gt.orientation.w_val, gt.orientation.x_val,
                       gt.orientation.y_val, gt.orientation.z_val])
    gt_pos = np.array([gt.position.x_val, gt.position.y_val, gt.position.z_val])

    # kinematics_estimated orientation (what old approach used)
    ke = mr.kinematics_estimated
    ke_ori = np.array([ke.orientation.w_val, ke.orientation.x_val,
                       ke.orientation.y_val, ke.orientation.z_val])
    ke_pos = np.array([ke.position.x_val, ke.position.y_val, ke.position.z_val])

    gt_ypr = quat_to_ypr(gt_ori)
    ke_ypr = quat_to_ypr(ke_ori)
    diff_ypr = gt_ypr - ke_ypr
    pos_diff = np.linalg.norm(gt_pos - ke_pos)

    print(f"{i+1:3d}  {gt_ypr[0]:8.3f}  {ke_ypr[0]:8.3f}  {diff_ypr[0]:+9.4f}  "
          f"{gt_ypr[1]:8.3f}  {ke_ypr[1]:8.3f}  {diff_ypr[1]:+8.4f}  "
          f"{gt_ypr[2]:8.3f}  {ke_ypr[2]:8.3f}  {diff_ypr[2]:+8.4f}  "
          f"{pos_diff:8.5f}")

    # Advance waypoint
    dist_to_wp = np.linalg.norm(gt_pos - np.array([10, 10, -8]))
    if dist_to_wp < 3.0:
        flight_task = client.moveToPositionAsync(-10, -10, -8, 4)

    time.sleep(0.3)

print()
print("If Δyaw/Δpitch/Δroll are non-zero, simGetGroundTruthKinematics returns")
print("a DIFFERENT orientation than kinematics_estimated — that's the bug.")
print()

# Also print raw quaternions for one sample to check for sign/convention issues
client.simPause(True)
gt = client.simGetGroundTruthKinematics()
mr = client.getMultirotorState()
client.simPause(False)

gt_ori = [gt.orientation.w_val, gt.orientation.x_val,
          gt.orientation.y_val, gt.orientation.z_val]
ke_ori = [mr.kinematics_estimated.orientation.w_val,
          mr.kinematics_estimated.orientation.x_val,
          mr.kinematics_estimated.orientation.y_val,
          mr.kinematics_estimated.orientation.z_val]

print(f"  Raw GT quat (w,x,y,z): ({gt_ori[0]:.6f}, {gt_ori[1]:.6f}, {gt_ori[2]:.6f}, {gt_ori[3]:.6f})")
print(f"  Raw KE quat (w,x,y,z): ({ke_ori[0]:.6f}, {ke_ori[1]:.6f}, {ke_ori[2]:.6f}, {ke_ori[3]:.6f})")
quat_diff = np.array(gt_ori) - np.array(ke_ori)
print(f"  Quat difference:       ({quat_diff[0]:.6f}, {quat_diff[1]:.6f}, {quat_diff[2]:.6f}, {quat_diff[3]:.6f})")
print(f"  Quat L2 diff:          {np.linalg.norm(quat_diff):.8f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: simContinueForFrames approach vs simPause approach
#         Instead of pausing mid-flight, we step the sim 1 frame at a time.
#         This should make the LiDAR scan timestamp match the GT kinematcs
#         timestamp much more closely.
# ═══════════════════════════════════════════════════════════════════════════════
print("="*100)
print("  TEST 6 — simContinueForFrames(1) vs simPause staleness comparison")
print("  We compare LiDAR staleness using two approaches:")
print("    A) simPause(True) → read → simPause(False)     [old approach]")
print("    B) simContinueForFrames(1) → read  (frame-step) [new approach]")
print("="*100)

# Fly an aggressive zigzag
zigzag_wps = [
    (15, -15, -8, 5),
    (-15, 15, -8, 5),
    (15, 15, -8, 5),
    (-15, -15, -8, 5),
]
wp_idx = 0

# ── Part A: simPause approach ────────────────────────────────────────────
print("\n  Part A: simPause approach (current)")
flight_task = client.moveToPositionAsync(*zigzag_wps[wp_idx])

print(f"{'#':>3}  {'staleness':>10}  {'yaw_rate':>8}  {'est err':>10}")
print("-" * 40)

pause_stalenesses = []
pause_errs = []

for i in range(30):
    time.sleep(0.2)
    client.simPause(True)
    lidar_data = client.getLidarData()
    gt = client.simGetGroundTruthKinematics()
    mr = client.getMultirotorState()
    client.simPause(False)

    if len(lidar_data.point_cloud) < 9:
        continue

    lidar_ts = float(lidar_data.time_stamp)
    mr_ts = float(mr.timestamp)
    stale_ms = (mr_ts - lidar_ts) / 1e6

    omega = np.array([gt.angular_velocity.x_val, gt.angular_velocity.y_val,
                      gt.angular_velocity.z_val])
    yaw_rate = np.degrees(np.linalg.norm(omega))
    est_err = yaw_rate * stale_ms / 1000.0

    pause_stalenesses.append(stale_ms)
    pause_errs.append(est_err)

    print(f"{i+1:3d}  {stale_ms:8.1f}ms  {yaw_rate:8.2f}  {est_err:10.4f}°")

    pos_now = np.array([gt.position.x_val, gt.position.y_val, gt.position.z_val])
    wp_pos = np.array(zigzag_wps[wp_idx][:3])
    if np.linalg.norm(pos_now - wp_pos) < 4.0:
        wp_idx = (wp_idx + 1) % len(zigzag_wps)
        flight_task = client.moveToPositionAsync(*zigzag_wps[wp_idx])

# ── Part B: simContinueForFrames approach ────────────────────────────────
print(f"\n  Part B: simContinueForFrames(1) approach (new)")
wp_idx = 0
client.simPause(True)  # start paused
flight_task = client.moveToPositionAsync(*zigzag_wps[wp_idx])

print(f"{'#':>3}  {'staleness':>10}  {'yaw_rate':>8}  {'est err':>10}")
print("-" * 40)

frame_stalenesses = []
frame_errs = []
last_lidar_ts = 0

for i in range(30):
    # Step exactly 1 render frame, then sim auto-pauses
    client.simContinueForFrames(1)
    time.sleep(0.05)  # small delay to let the frame complete

    lidar_data = client.getLidarData()
    gt = client.simGetGroundTruthKinematics()
    mr = client.getMultirotorState()

    if len(lidar_data.point_cloud) < 9:
        continue

    lidar_ts = float(lidar_data.time_stamp)
    mr_ts = float(mr.timestamp)

    # Skip if no new scan
    if lidar_ts == last_lidar_ts:
        # Step more frames to get a fresh scan
        for _ in range(10):
            client.simContinueForFrames(1)
            time.sleep(0.05)
            lidar_data = client.getLidarData()
            new_ts = float(lidar_data.time_stamp)
            if new_ts != last_lidar_ts:
                gt = client.simGetGroundTruthKinematics()
                mr = client.getMultirotorState()
                lidar_ts = new_ts
                mr_ts = float(mr.timestamp)
                break

    last_lidar_ts = lidar_ts
    stale_ms = (mr_ts - lidar_ts) / 1e6

    omega = np.array([gt.angular_velocity.x_val, gt.angular_velocity.y_val,
                      gt.angular_velocity.z_val])
    yaw_rate = np.degrees(np.linalg.norm(omega))
    est_err = yaw_rate * stale_ms / 1000.0

    frame_stalenesses.append(stale_ms)
    frame_errs.append(est_err)

    print(f"{i+1:3d}  {stale_ms:8.1f}ms  {yaw_rate:8.2f}  {est_err:10.4f}°")

    pos_now = np.array([gt.position.x_val, gt.position.y_val, gt.position.z_val])
    wp_pos = np.array(zigzag_wps[wp_idx][:3])
    if np.linalg.norm(pos_now - wp_pos) < 4.0:
        wp_idx = (wp_idx + 1) % len(zigzag_wps)
        flight_task = client.moveToPositionAsync(*zigzag_wps[wp_idx])

client.simPause(False)

print()
if pause_stalenesses and frame_stalenesses:
    print(f"  simPause approach:         staleness mean={np.mean(pause_stalenesses):.1f}ms, "
          f"max={np.max(pause_stalenesses):.1f}ms | "
          f"yaw err mean={np.mean(pause_errs):.3f}°, max={np.max(pause_errs):.3f}°")
    print(f"  simContinueForFrames(1):   staleness mean={np.mean(frame_stalenesses):.1f}ms, "
          f"max={np.max(frame_stalenesses):.1f}ms | "
          f"yaw err mean={np.mean(frame_errs):.3f}°, max={np.max(frame_errs):.3f}°")
    improvement = np.mean(pause_stalenesses) / max(np.mean(frame_stalenesses), 0.001)
    print(f"  Staleness improvement: {improvement:.1f}x")
print()

# Land
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Done.")