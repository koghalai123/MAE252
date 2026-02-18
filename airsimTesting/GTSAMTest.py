import gtsam
import numpy as np

# IF GETTING SEGFAULT, INSTALL OLD VERSION OF NUMPY: pip install numpy==1.26.4

def run_mapping_example():
    # 1. Create Graph
    graph = gtsam.NonlinearFactorGraph()
    
    # 2. Define Noise Models (Explicitly create them)
    # Using 'Diagonal.Sigmas' is safer than raw matrices for beginners
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    
    # 3. Add Factors
    # Symbol X(1) is just an integer key, but using Symbols is best practice
    X1 = gtsam.symbol('x', 1)
    X2 = gtsam.symbol('x', 2)
    
    # Add a prior: "I start at (0,0,0)"
    graph.add(gtsam.PriorFactorPose2(X1, gtsam.Pose2(0, 0, 0), prior_noise))
    
    # Add odometry: "I moved 2 meters on X axis"
    graph.add(gtsam.BetweenFactorPose2(X1, X2, gtsam.Pose2(2, 0, 0), odom_noise))
    
    # 4. Initial Estimates (Crucial: Must match the keys used in the graph)
    initials = gtsam.Values()
    initials.insert(X1, gtsam.Pose2(0, 0, 0))
    initials.insert(X2, gtsam.Pose2(2.1, 0.1, 0.05)) # A slightly noisy guess
    
    # 5. Optimize
    try:
        # Use Levenberg-Marquardt as it is the most robust
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initials, params)
        result = optimizer.optimize()
        
        print("Optimization Successful!")
        print(result)
    except Exception as e:
        print(f"An error occurred during optimization: {e}")

if __name__ == "__main__":
    run_mapping_example()