import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos)
        self.parent = parent

class Obstacle:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def collides(self, p):
        return np.linalg.norm(p - self.center) <= self.radius

    def line_collides(self, p1, p2, steps=10):
        # Check for collision along the line segment
        for t in np.linspace(0, 1, steps):
            pt = p1 + t * (p2 - p1)
            if self.collides(pt):
                return True
        return False

def dist(a, b):
    return np.linalg.norm(a - b)

def steer(from_node, to_pos, step_size):
    direction = to_pos - from_node.pos
    length = np.linalg.norm(direction)
    if length == 0:
        return from_node.pos
    direction = direction / length
    new_pos = from_node.pos + step_size * direction
    return new_pos

def get_nearest(tree, pos):
    dists = [dist(n.pos, pos) for n in tree]
    return tree[np.argmin(dists)]

def path_from_node(node):
    path = []
    while node is not None:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def plot_tree(tree, color='b', alpha=0.5, label=None):
    # Only add label to the first line for legend clarity
    first = True
    for node in tree:
        if node.parent is not None:
            if first and label is not None:
                plt.plot([node.pos[0], node.parent.pos[0]], [node.pos[1], node.parent.pos[1]], color=color, alpha=alpha, label=label)
                first = False
            else:
                plt.plot([node.pos[0], node.parent.pos[0]], [node.pos[1], node.parent.pos[1]], color=color, alpha=alpha)

def plot_path(path, color='r', linewidth=2, label=None):
    path = np.array(path)
    # Only plot if path has at least 2 points and is 2D
    if path.ndim != 2 or path.shape[0] < 2:
        return
    if label is not None:
        plt.plot(path[:,0], path[:,1], color=color, linewidth=linewidth, label=label)
    else:
        plt.plot(path[:,0], path[:,1], color=color, linewidth=linewidth)

def plot_obstacles(obstacles):
    for obs in obstacles:
        circle = plt.Circle(obs.center, obs.radius, color='k', alpha=0.3)
        plt.gca().add_patch(circle)

def collision_free(p1, p2, obstacles):
    for obs in obstacles:
        if obs.line_collides(p1, p2):
            return False
    return True

def rrt(start, goal, bounds, step_size=0.1, max_iter=500, goal_sample_rate=0.01, plot_callback=None, obstacles=None):
    tree = [Node(start)]
    for i in range(max_iter):
        if np.random.rand() < goal_sample_rate:
            sample = goal
        else:
            sample = np.random.uniform(bounds[:,0], bounds[:,1])
        nearest = get_nearest(tree, sample)
        new_pos = steer(nearest, sample, step_size)
        if obstacles and not collision_free(nearest.pos, new_pos, obstacles):
            continue
        new_node = Node(new_pos, nearest)
        tree.append(new_node)
        if dist(new_pos, goal) < step_size and (not obstacles or collision_free(new_pos, goal, obstacles)):
            goal_node = Node(goal, new_node)
            tree.append(goal_node)
            if plot_callback: plot_callback(tree, path_from_node(goal_node))
            return path_from_node(goal_node), tree
        if plot_callback and i % 20 == 0:
            plot_callback(tree, None)
    if plot_callback: plot_callback(tree, None)
    return None, tree

def birrt(start, goal, bounds, step_size=0.1, max_iter=500, goal_sample_rate=0.05, plot_callback=None, obstacles=None):
    tree_a = [Node(start)]
    tree_b = [Node(goal)]
    for i in range(max_iter):
        if np.random.rand() < goal_sample_rate:
            sample = tree_b[0].pos
        else:
            sample = np.random.uniform(bounds[:,0], bounds[:,1])
        nearest_a = get_nearest(tree_a, sample)
        new_pos_a = steer(nearest_a, sample, step_size)
        if obstacles and not collision_free(nearest_a.pos, new_pos_a, obstacles):
            continue
        new_node_a = Node(new_pos_a, nearest_a)
        tree_a.append(new_node_a)

        nearest_b = get_nearest(tree_b, new_pos_a)
        new_pos_b = steer(nearest_b, new_pos_a, step_size)
        if obstacles and not collision_free(nearest_b.pos, new_pos_b, obstacles):
            continue
        new_node_b = Node(new_pos_b, nearest_b)
        tree_b.append(new_node_b)

        if dist(new_node_a.pos, new_node_b.pos) < step_size and (not obstacles or collision_free(new_node_a.pos, new_node_b.pos, obstacles)):
            path_a = path_from_node(new_node_a)
            path_b = path_from_node(new_node_b)
            if plot_callback: plot_callback(tree_a + tree_b, path_a + path_b[::-1])
            return path_a + path_b[::-1], tree_a, tree_b
        if plot_callback and i % 20 == 0:
            plot_callback(tree_a + tree_b, None)
        tree_a, tree_b = tree_b, tree_a
    if plot_callback: plot_callback(tree_a + tree_b, None)
    return None, tree_a, tree_b

def optimize_path(path, obstacles):
    """
    Greedy shortcutting: repeatedly try to connect farther nodes directly if collision-free.
    Returns a new optimized path.
    """
    if path is None or len(path) < 2:
        return path
    optimized = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Try to find the farthest node we can connect to directly
        j = len(path) - 1
        while j > i + 1:
            if collision_free(path[i], path[j], obstacles):
                break
            j -= 1
        optimized.append(path[j])
        i = j
    return optimized

# Example usage and plotting with obstacles
if __name__ == "__main__":
    bounds = np.array([[0, 10], [0, 10]])
    start = np.array([1, 1])
    goal = np.array([9, 9])
    obstacles = [
        Obstacle(center=[5, 5], radius=1.5),
        Obstacle(center=[3, 7], radius=1.0),
        Obstacle(center=[7, 3], radius=1.0),
        Obstacle(center=[2, 2], radius=0.7),
        Obstacle(center=[8, 2], radius=0.7),
        Obstacle(center=[2, 8], radius=0.7),
        Obstacle(center=[8, 8], radius=0.7),
        Obstacle(center=[5, 8], radius=0.8),
        Obstacle(center=[8, 5], radius=0.8),
        Obstacle(center=[5, 2], radius=0.8),
        Obstacle(center=[2, 5], radius=0.8),
    ]

    nodes_per_frame = 10  # Change this to control video speed

    def rrt_with_frames(start, goal, bounds, step_size, max_iter, goal_sample_rate, obstacles, nodes_per_frame):
        tree = [Node(start)]
        frames = []
        path = None
        for i in range(max_iter):
            if np.random.rand() < goal_sample_rate:
                sample = goal
            else:
                sample = np.random.uniform(bounds[:,0], bounds[:,1])
            nearest = get_nearest(tree, sample)
            new_pos = steer(nearest, sample, step_size)
            if obstacles and not collision_free(nearest.pos, new_pos, obstacles):
                continue
            new_node = Node(new_pos, nearest)
            tree.append(new_node)
            if dist(new_pos, goal) < step_size and (not obstacles or collision_free(new_pos, goal, obstacles)):
                goal_node = Node(goal, new_node)
                tree.append(goal_node)
                path = path_from_node(goal_node)
            if i % nodes_per_frame == 0 or (path is not None and len(frames)==0):
                frames.append((list(tree), path))
            if path is not None:
                break
        # Add final frame
        frames.append((list(tree), path))
        return path, tree, frames

    def birrt_with_frames(start, goal, bounds, step_size, max_iter, goal_sample_rate, obstacles, nodes_per_frame):
        # Initialize two trees: one from the start, one from the goal
        tree_a = [Node(start)]
        tree_b = [Node(goal)]
        frames = []  # For animation frames
        path = None  # Will hold the final path if found
        node_add_count = 0  # Count how many nodes have been added (for animation pacing)
        for i in range(max_iter):
            # With some probability, bias the sample toward the other tree's root (goal bias)
            if np.random.rand() < goal_sample_rate:
                sample = tree_b[0].pos
            else:
                sample = np.random.uniform(bounds[:,0], bounds[:,1])
            # 1. Extend tree_a toward the sample
            nearest_a = get_nearest(tree_a, sample)
            new_pos_a = steer(nearest_a, sample, step_size)
            # Check for collision
            if obstacles and not collision_free(nearest_a.pos, new_pos_a, obstacles):
                # Swap trees and try again next iteration
                tree_a, tree_b = tree_b, tree_a
                continue
            # Add new node to tree_a
            new_node_a = Node(new_pos_a, nearest_a)
            tree_a.append(new_node_a)
            node_add_count += 1
            # Save a frame for animation every nodes_per_frame additions
            if node_add_count % nodes_per_frame == 0:
                frames.append((list(tree_a + tree_b), path))
            # 2. Try to connect tree_b to the new node in tree_a
            nearest_b = get_nearest(tree_b, new_node_a.pos)
            new_pos_b = steer(nearest_b, new_node_a.pos, step_size)
            # Check for collision
            if obstacles and not collision_free(nearest_b.pos, new_pos_b, obstacles):
                # Swap trees and try again next iteration
                tree_a, tree_b = tree_b, tree_a
                continue
            # Add new node to tree_b
            new_node_b = Node(new_pos_b, nearest_b)
            tree_b.append(new_node_b)
            node_add_count += 1
            # Save a frame for animation every nodes_per_frame additions
            if node_add_count % nodes_per_frame == 0:
                frames.append((list(tree_a + tree_b), path))
            # 3. Check if the two trees are connected (i.e., nodes are close enough)
            if dist(new_node_a.pos, new_node_b.pos) < step_size and (not obstacles or collision_free(new_node_a.pos, new_node_b.pos, obstacles)):
                # If so, extract the path from start to goal by joining the two trees
                path_a = path_from_node(new_node_a)
                path_b = path_from_node(new_node_b)
                path = path_a + path_b[::-1]
                # Save the final frame with the solution path
                frames.append((list(tree_a + tree_b), path))
                break
            # 4. Swap the trees for the next iteration (this alternates which tree grows)
            tree_a, tree_b = tree_b, tree_a
        # Add a final frame if not already added
        if not frames or frames[-1][1] is None:
            frames.append((list(tree_a + tree_b), path))
        return path, tree_a, tree_b, frames

    def make_anim(frames, filename, title):
        fig = plt.figure()
        def draw(frame):
            tree, path = frame
            plt.clf()
            plt.xlim(bounds[0])
            plt.ylim(bounds[1])
            plot_obstacles(obstacles)
            plot_tree(tree, label='Tree')
            if path is not None:
                plot_path(path, label='Path')
            plt.scatter(*start, c='g', s=100, label='Start')
            plt.scatter(*goal, c='r', s=100, label='Goal')
            plt.title(title)
            # Only show legend if at least one path/tree is present
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(loc='best')
        ani = animation.FuncAnimation(fig, draw, frames=frames, repeat=False)
        # Save as mp4
        writer = animation.FFMpegWriter(fps=10)
        ani.save(filename, writer=writer)
        # Save as gif (filename with .gif extension)
        gif_filename = filename.rsplit('.', 1)[0] + '.gif'
        ani.save(gif_filename, writer='pillow', fps=10)
        plt.close(fig)


    print("Running RRT with obstacles and saving animation...")
    path, tree, frames = rrt_with_frames(start, goal, bounds, step_size=0.1, max_iter=2000, goal_sample_rate=0.05, obstacles=obstacles, nodes_per_frame=nodes_per_frame)
    make_anim(frames, "rrt_animation.mp4", "RRT with Obstacles")
    print("Saved RRT animation to rrt_animation.mp4")

    # Save image of the final path before optimization (RRT)
    if path is not None:
        plt.figure()
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plot_obstacles(obstacles)
        plot_tree(tree, label='Tree')
        plot_path(path, color='r', linewidth=2, label='Path (Unoptimized)')
        plt.scatter(*start, c='g', s=100, label='Start')
        plt.scatter(*goal, c='r', s=100, label='Goal')
        plt.title("RRT Final Path (Before Optimization)")
        plt.legend(loc='best')
        plt.savefig("rrt_path_before_optimization.png", dpi=150)
        plt.close()
        print("Saved rrt_path_before_optimization.png")

    # Optimize the path (RRT)
    opt_path = optimize_path(path, obstacles)
    # Save image of the optimized path, also show unoptimized path for comparison
    plt.figure()
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plot_obstacles(obstacles)
    plot_tree(tree, label='Tree')
    plot_path(path, color='r', linewidth=2, label='Path (Unoptimized)')
    plot_path(opt_path, color='m', linewidth=2, label='Path (Optimized)')
    plt.scatter(*start, c='g', s=100, label='Start')
    plt.scatter(*goal, c='r', s=100, label='Goal')
    plt.title("RRT Optimized Path (After Greedy Shortcutting)")
    plt.legend(loc='best')
    plt.savefig("rrt_path_after_optimization.png", dpi=150)
    plt.close()
    print("Saved rrt_path_after_optimization.png")

    print("Running Bi-RRT with obstacles and saving animation...")
    path, tree_a, tree_b, frames = birrt_with_frames(start, goal, bounds, step_size=0.1, max_iter=500, goal_sample_rate=0.05, obstacles=obstacles, nodes_per_frame=nodes_per_frame)
    make_anim(frames, "birrt_animation.mp4", "Bi-RRT with Obstacles")
    print("Saved Bi-RRT animation to birrt_animation.mp4")

    # Save image of the final path before optimization (Bi-RRT)
    if path is not None:
        plt.figure()
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plot_obstacles(obstacles)
        plot_tree(tree_a + tree_b, label='Tree')
        plot_path(path, color='r', linewidth=2, label='Path (Unoptimized)')
        plt.scatter(*start, c='g', s=100, label='Start')
        plt.scatter(*goal, c='r', s=100, label='Goal')
        plt.title("Bi-RRT Final Path (Before Optimization)")
        plt.legend(loc='best')
        plt.savefig("birrt_path_before_optimization.png", dpi=150)
        plt.close()
        print("Saved birrt_path_before_optimization.png")

    # Optimize the path (Bi-RRT)
    opt_path = optimize_path(path, obstacles)
    # Save image of the optimized path, also show unoptimized path for comparison
    plt.figure()
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plot_obstacles(obstacles)
    plot_tree(tree_a + tree_b, label='Tree')
    plot_path(path, color='r', linewidth=2, label='Path (Unoptimized)')
    plot_path(opt_path, color='m', linewidth=2, label='Path (Optimized)')
    plt.scatter(*start, c='g', s=100, label='Start')
    plt.scatter(*goal, c='r', s=100, label='Goal')
    plt.title("Bi-RRT Optimized Path (After Greedy Shortcutting)")
    plt.legend(loc='best')
    plt.savefig("birrt_path_after_optimization.png", dpi=150)
    plt.close()
    print("Saved birrt_path_after_optimization.png")