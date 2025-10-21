"""Task 1: Point Cloud Registration using ICP"""
import numpy as np
from scipy.spatial import KDTree
import trimesh
from matplotlib import pyplot as plt

def visualize_registration_dynamic(
    source,
    sphere,
    cylinder,
    sphere_transforms,
    cylinder_transforms,
    show_visualization=True,
):
    """
    Visualizes the dynamic registration of the source and target point clouds step-by-step using matplotlib.

    Args:
        source (np.array): Nx3 array representing the source point cloud.
        sphere (np.array): Nx3 array representing the sphere point cloud.
        cylinder (np.array): Nx3 array representing the cylinder point cloud.
        sphere_transforms (list): List of transformation matrices applied to the sphere in each step.
        cylinder_transforms (list): List of transformation matrices applied to the cylinder in each step.
        show_visualization (bool): If True, displays the dynamic registration process.
    """
    if not show_visualization:
        return

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the source point cloud (fixed reference)
    ax.scatter(
        source[:, 0], source[:, 1], source[:, 2], c="r", label="Source (Red)", alpha=0.5
    )

    # Set the plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Define the update function for animation
    def update_plot(i):
        ax.cla()  # Clear the previous plot



        # Plot the source point cloud (fixed reference)
        ax.scatter(
            source[:, 0],
            source[:, 1],
            source[:, 2],
            c="r",
            label="Source (Red)",
            alpha=0.5,
        )

        #old version
        # Apply the current transformation to the sphere
        transformed_sphere = (
            sphere_transforms[i][:3, :3] @ sphere.T
        ).T + sphere_transforms[i][:3, 3]

        # Apply the current transformation to the cylinder
        transformed_cylinder = (
            cylinder_transforms[i][:3, :3] @ cylinder.T
        ).T + cylinder_transforms[i][:3, 3]

        # Plot the transformed point clouds
        ax.scatter(
            transformed_sphere[:, 0],
            transformed_sphere[:, 1],
            transformed_sphere[:, 2],
            c="b",
            label="Sphere (Blue)",
            alpha=0.5,
        )
        ax.scatter(
            transformed_cylinder[:, 0],
            transformed_cylinder[:, 1],
            transformed_cylinder[:, 2],
            c="g",
            label="Cylinder (Green)",
            alpha=0.5,
        )

        # Update plot labels and limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Redraw the plot
        plt.draw()
        plt.pause(0.01)  # Pause for animation effect

    # Run the update function for each transformation
    # sphere_transforms = sphere.copy()
    # cylinder_transforms = cylinder.copy()

    #MODIFICATION: use min_transforms instead of just sphere_transforms as otherwise the updateplot runs of out frames for cylinder
    min_transforms = min(len(sphere_transforms), len(cylinder_transforms))
    for i in range(min_transforms):
        update_plot(i)
        print(i)

    # Show the final frame
    # plt.show()
    plt.close(ax.figure)


def compute_centroid(points):
    """Computes the centroid of a set of 3D points.

    Args:
        points (np.array): A Nx3 array of 3D points.

    Returns:
        np.array: A 1x3 array representing the centroid of the points.
    """

    ### YOUR CODE HERE ###
    N = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_z = sum(p[2] for p in points)
    centroid = [sum_x / N, sum_y / N, sum_z / N]
    ### END OF YOUR CODE ###

    return centroid


def generate_random_transformation():
    # Generate a random rotation matrix using trimesh
    random_rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]

    # Generate a random translation vector between -0.5 and 0.5
    random_translation = np.random.uniform(-0.5, 0.5, size=3)

    # Construct the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = random_rotation
    transformation_matrix[:3, 3] = random_translation

    return transformation_matrix


def visualize_registration(source, target0, target1):
    source_pcd = trimesh.points.PointCloud(source, colors=[255, 0, 0])
    target0_pcd = trimesh.points.PointCloud(target0, colors=[0, 255, 0])
    target1_pcd = trimesh.points.PointCloud(target1, colors=[0, 0, 255])

    # Create a scene
    scene = trimesh.Scene()
    scene.add_geometry(source_pcd)
    scene.add_geometry(target0_pcd)
    scene.add_geometry(target1_pcd)
    scene.show()


def icp_point_to_point(source, target, max_iterations=100, tolerance=1e-10):
    """
    Point-to-Point ICP algorithm.
    Args:
        source: Nx3 array of source points
        target: Nx3 array of target points
        max_iterations: maximum number of iterations to run ICP
        tolerance: convergence criteria for early stopping
    Returns:
        source_transformed: Transformed source points after registration
        error: Registration error
    """
    tree = KDTree(target)
    source_transformed = source.copy()
    all_transforms = [np.eye(4)]
    prev_error = np.inf
    for i in range(max_iterations):
        ### YOUR CODE HERE ###
        # Find the nearest neighbors of the source points in the target point cloud. Hint: check tree.query() function
        distances, indices = tree.query(source_transformed, 1)
        matched_points = target[indices]

        # Compute the centroid of the source and target points.
        source_center = compute_centroid(source_transformed)
        target_center = compute_centroid(matched_points)

        # Demean the source and target points.
        source_demeaned = source_transformed - source_center
        target_demeaned = matched_points - target_center

        # Compute the covariance matrix between the source and target points.
        covariance = source_demeaned.T @ target_demeaned

        # Compute the Singular Value Decomposition of the covariance matrix.
        U, S, Vt = np.linalg.svd(covariance)

        # Compute the rotation matrix R.
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute the translation vector t.
        t = target_center - R @ source_center
        
        # Update the source points using the computed rotation matrix and translation vector.
        source_transformed = (source_transformed @ R.T) + t 

        # Compute the transformation matrix
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3,  3] = t
        transformation_matrix = T_iter @ all_transforms[-1]
        
        # Update the transformation matrix list
        all_transforms.append(transformation_matrix)
        
        # Compute the error as the mean of the distances between the source and target points.
        error = float(np.mean(distances ** 2))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    # target_len = max_iterations + 1  # include initial identity
    # if len(all_transforms) < target_len:
    #     all_transforms.extend([all_transforms[-1]] * (target_len - len(all_transforms)))

        ### END OF YOUR CODE ###

    print(f"Point-to-Point ICP converged after {i+1} iterations with error {error}")
    return source_transformed, error, all_transforms


def main():
    # Load the point clouds and normals
    sphere_pc = np.load("data/sphere_data.npz")["points"]
    cylinder_pc = np.load("data/cylinder_data.npz")["points"]
    lollipop_pc = np.load("data/lollipop_data.npz")["points"]

    for i in range(10):
        visualize_registration(lollipop_pc, sphere_pc, cylinder_pc)
        random_transformation_matrix = generate_random_transformation()
        cylinder_pc_homo = np.hstack((cylinder_pc, np.ones((cylinder_pc.shape[0], 1))))
        sphere_pc_homo = np.hstack((sphere_pc, np.ones((sphere_pc.shape[0], 1))))
        transformed_cylinder = np.dot(cylinder_pc_homo, random_transformation_matrix)[
            :, :3
        ]
        
        transformed_sphere = np.dot(sphere_pc_homo, random_transformation_matrix)[:, :3]
        
        # # Apply Point-to-Point ICP to register the sphere to the lollipop model
        # sphere_transformed_ptp, error_sphere_ptp, all_transforms_sphere = (
        #     icp_point_to_point(transformed_sphere, lollipop_pc)
        # )

        # # Apply Point-to-Point ICP to register the cylinder to the lollipop model
        # cylinder_transformed_ptp, error_cylinder_ptp, all_transforms_cylinder = (
        #     icp_point_to_point(transformed_cylinder, lollipop_pc)
        # )

        # --- 1) Register the sphere to the full lollipop ---
        sphere_transformed_ptp, error_sphere_ptp, all_transforms_sphere = (
            icp_point_to_point(transformed_sphere, lollipop_pc)
        )

        # --- 2) Estimate candy center & radius from the aligned sphere ---
        sphere_center = np.mean(sphere_transformed_ptp, axis=0)
        # median radius is robust to partial sampling/noise
        sphere_radius = np.median(np.linalg.norm(sphere_transformed_ptp - sphere_center, axis=1))

        # --- 3) Mask out *all* target points near the candy (keep only stick) ---
        # pick a small margin relative to the sphere radius (tune 0.03–0.10)
        margin = 0.2 * sphere_radius
        dist_to_center = np.linalg.norm(lollipop_pc - sphere_center, axis=1)
        stick_mask = dist_to_center > (sphere_radius + margin)
        lollipop_stick = lollipop_pc[stick_mask]

        # --- 4) Register the cylinder ONLY to stick points ---
        cylinder_transformed_ptp, error_cylinder_ptp, all_transforms_cylinder = (
            icp_point_to_point(transformed_cylinder, lollipop_stick)
        )

        # Print comparison of results
        print(f"Point-to-Point ICP Sphere registration error: {error_sphere_ptp}")
        print(f"Point-to-Point ICP Cylinder registration error: {error_cylinder_ptp}")

        # Visualize results
        visualize_registration_dynamic(
            lollipop_pc,
            transformed_sphere,
            transformed_cylinder,
            all_transforms_sphere,
            all_transforms_cylinder,
            show_visualization=True,
        )
        
        visualize_registration(
            lollipop_pc, cylinder_transformed_ptp, sphere_transformed_ptp
        )


if __name__ == "__main__":
    main()
