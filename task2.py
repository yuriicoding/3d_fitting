"""Task 2: Primitive parameter otpimizaiton with RANSAC"""
import numpy as np
from scipy.optimize import least_squares
import trimesh


def create_cylinder_mesh(center, direction, radius, height, color=[0, 1, 0]):
    """
    Create a cylinder mesh in trimesh centered at `center` and aligned to `direction`.

    Args:
        center (np.ndarray): The center point of the cylinder.
        direction (np.ndarray): A vector indicating the cylinder's orientation.
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        color (list): RGB color of the cylinder.

    Returns:
        trimesh.Trimesh: A trimesh object representing the cylinder.
    """
    # Create a cylinder aligned with the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)

    # Normalize the direction vector
    direction = np.array(direction)
    direction /= np.linalg.norm(direction)

    # Compute the rotation matrix to align the cylinder's Z-axis with the given direction vector
    z_axis = np.array([0, 0, 1])  # The default axis of the cylinder
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)

    # Apply rotation to the cylinder
    cylinder.apply_transform(rotation_matrix)

    # Translate the cylinder to the desired center position
    cylinder.apply_translation(center)

    # Apply color to the cylinder mesh
    cylinder.visual.face_colors = np.array(color + [1.0]) * 255  # Color the mesh faces

    return cylinder


def create_sphere_mesh(center, radius, color=[1, 0, 0]):
    """
    Create a sphere mesh in trimesh centered at `center`.

    Args:
        center (np.ndarray): The center of the sphere.
        radius (float): The radius of the sphere.
        color (list): RGB color of the sphere.

    Returns:
        trimesh.Trimesh: A trimesh object representing the sphere.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation(center)
    sphere.visual.face_colors = np.array(color + [1.0]) * 255  # Color the mesh faces

    return sphere

def sphere_residuals(params, points):
        """
        Compute the residuals for fitting points to a sphere.

        Args:
            params (np.array): A 1x4 array representing the sphere parameters [x0, y0, z0, r].
            points (np.array): Nx3 array of points to fit the sphere to.

        Returns:
            np.array: Residuals representing the difference between distances and the radius.
        """
        # Sphere parameters
        x0, y0, z0, r = params
        center = np.array([x0, y0, z0])

        ### Your code here ###
        # Compute the residuals: distance of each point to sphere surface
        residual = np.zeros(points.shape[0])
        ### End of your code ###
        return residual

def sphere_jac(params, points):
    """
    Compute the Jacobian matrix for fitting points to a sphere.

    Args:
        params (np.array): A 1x4 array representing the sphere parameters [x0, y0, z0, r].
        points (np.array): Nx3 array of points to fit the sphere to.

    Returns:
        np.array: Jacobian matrix (Nx4), with partial derivatives of the residuals with respect to [x, y, z, r].
    """
    # Sphere parameters
    x0, y0, z0, r = params
    center = np.array([x0, y0, z0])

    ### Your code here ###
    # Initialize the Jacobian matrix (Nx4)
    J = np.zeros((points.shape[0], 4))

    # Partial derivatives w.r.t x0, y0, z0 (center coordinates)
    J[:, 0] = 0
    J[:, 1] = 0
    J[:, 2] = 0
    
    # Partial derivative w.r.t r (radius)
    J[:, 3] = 0

    ### End of your code ###
    return J
    
def fit_sphere(points):
    # Initial guess for the sphere parameters: [x0, y0, z0, r]
    x = np.array([0.0, 0.0, 0.0, 0.3])  # Start with center at origin, small radius

    for i in range(1000):
        ### Your code here ###
        # Compute the residuals and Jacobian matrix
        residual = np.zeros(points.shape[0])
        J = np.zeros((points.shape[0], 4))

        # Compute the update step with Gauss-Newton method
        delta = np.zeros(4)

        # Update the parameters
        x = x - delta  

        ### End of your code ###
        
        # Check for convergence
        if np.linalg.norm(delta) < 1e-4:
            print(f"Converged after {i+1} iterations")
            break

    return x

def cylinder_residuals(params, points):
        x0, y0, z0 = params[:3]
        dx, dy, dz = params[3:6]
        r = params[6]
        axis = np.array([dx, dy, dz])

        ### Your code here ###
        # Normalise the axis vector
        axis_normalized = np.zeros_like(axis)

        ### Comment on line below: What does h represent and why it is calculated in this way?###
        h = np.ptp(np.dot(points - np.array([x0, y0, z0]), axis_normalized))
        
        # Compute the projection of each point onto the cylinder axis
        projection = np.zeros_like(points)
        
        # Compute the distance of each point to the cylinder axis
        dist_to_axis = np.zeros(points.shape[0])
        
        # Compute the distance of each point to the cylinder bottom/top surfaces
        # Hint: if dist_to_height is used as a residual, shall we penalise the points that are right on the cylinder surface but with a postiive dist_to_height measures?
        dist_to_height = np.zeros(points.shape[0])
        
        # Compute the residuals: distance of each point to the cylinder surface
        # Take the absolute value of the point to axis or height RESIDUE, whichever is greater
        dist_to_cyl = np.zeros(points.shape[0])
        ### End of your code ###
        
        return dist_to_cyl
    
def fit_cylinder(points):
    initial_guess = [0, 0, 0, 0, 0, 1, 0.05]
    ### Bonus: Implement the Gauss-Newton optimization function to fit the cylinder parameters ###
    result = least_squares(cylinder_residuals, initial_guess, args=(points,))
    return result.x  # Estimated cylinder parameters



def ransac_lollipop_fitting(points, max_iterations=100, threshold=0.01):
    best_sphere_params = None
    best_cylinder_params = None
    best_inliers_count = 0
    best_cylinder_height = 0

    sphere_points = points
    
    for _ in range(max_iterations):
        ### Your code here ###
        # Sample 10 points from the sphere points
        sphere_sample = np.zeros((10, 3))
        
        # Fit a sphere to the sampled points
        sphere_params = np.zeros(4)
        
        # Compute the residuals of the sphere fit from all points
        sphere_distances = np.zeros(points.shape[0])
        
        # Identify inliers (points with distance to sphere within the threshold) based on the residuals
        sphere_inliers = np.zeros((0, 3))
        
        # If the number of inliers is less than 50, continue to the next iteration
        if len(sphere_inliers) < 50:
            continue
        
        # Select the points that are not inliers to the sphere and use them to fit the cylinder
        cylinder_points = np.zeros((0, 3))
        
        # Sample 10 points from the cylinder points
        cylinder_sample = np.zeros((10, 3))
        
        # Fit a cylinder to the sampled points
        cylinder_params = np.zeros(7)
        
        # Compute the residuals of the cylinder fit from all points
        dist_to_cyl = np.zeros(points.shape[0])
        
        # Compute the total number of inliers for both sphere and cylinder fits
        total_inliers_count = 0
        
        # Update the best parameters if the current inliers count is higher
        
        if total_inliers_count > best_inliers_count:
            best_sphere_params = np.zeros(4)
            best_cylinder_params = np.zeros(7)
            best_inliers_count = total_inliers_count
            best_cylinder_height = 0
        
        # Update the sphere points to exclude the inliers to the cylinder
        sphere_points = np.zeros((0, 3))
        ### End of your code ###
        
    return best_sphere_params, best_cylinder_params, best_cylinder_height


def main():
    # Load the lollipop point cloud
    lollipop_pc = np.load("data/lollipop_data.npz")["points"]

    # Apply RANSAC to fit sphere and cylinder
    best_sphere_params, best_cylinder_params, cylinder_height = ransac_lollipop_fitting(
        lollipop_pc
    )

    # Extract the parameters
    sphere_center, sphere_radius = best_sphere_params[:3], best_sphere_params[3]
    cylinder_center, cylinder_axis, cylinder_radius = (
        best_cylinder_params[:3],
        best_cylinder_params[3:6],
        best_cylinder_params[6],
    )

    # Create Point Cloud object for visualization
    sphere_mesh = create_sphere_mesh(sphere_center, sphere_radius)
    cylinder_mesh = create_cylinder_mesh(
        cylinder_center, cylinder_axis, cylinder_radius, height=cylinder_height
    )

    # Convert point cloud to trimesh for visualization
    lollipop_cloud = trimesh.points.PointCloud(
        vertices=lollipop_pc, colors=[0, 0, 255, 255]
    )

    scene = trimesh.Scene([lollipop_cloud])
    scene.show()
    
    # Visualize using trimesh's scene
    scene = trimesh.Scene([lollipop_cloud, sphere_mesh, cylinder_mesh])
    scene.show()


if __name__ == "__main__":
    main()
