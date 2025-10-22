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

        # Vectorized distances from each point to the center
        dists = np.linalg.norm(points - center, axis=1)

        # Residuals: how far each point is from the sphere surface
        residual = dists - r
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
    # Differences from center to each point (N x 3)
    # = [x_i-x0, y_i-y0, z_i-z0]
    diff = points - center  
    dists = np.linalg.norm(diff, axis=1)

    # Avoid division by zero for points exactly at the center
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)

    # Jacobian initialization
    J = np.zeros((points.shape[0], 4))

    # Partials w.r.t center coordinates: -(diff / ||diff||)
    J[:, 0] = -diff[:, 0] / safe_dists
    J[:, 1] = -diff[:, 1] / safe_dists
    J[:, 2] = -diff[:, 2] / safe_dists

    # Partial w.r.t radius: -1
    J[:, 3] = -1.0

    ### End of your code ###
    return J
    
def fit_sphere(points):
    # Initial guess for the sphere parameters: [x0, y0, z0, r]
    x = np.array([0.0, 0.0, 0.0, 0.3])  # Start with center at origin, small radius

    for i in range(1000):
        ### Your code here ###
        # Compute the residuals and Jacobian matrix
        residual = sphere_residuals(x, points)
        J = sphere_jac(x, points)

        # Compute the update step with Gauss-Newton method
        delta, *_ = np.linalg.lstsq(J, residual, None)

        # Update the parameters
        x = x - delta

        # Keep radius non-negative (robustness)
        if x[3] < 0:
            x[3] = 1e-12  

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
        axis_norm = np.linalg.norm(axis)
        axis_normalized = axis / axis_norm

        ### Comment on line below: What does h represent and why it is calculated in this way?###
        # h represents the height of the cylinder — how tall it is along its main axis.
        # a cylinder extends in a the direction of the axis and has a certain height h
        # points - np.array([x0, y0, z0]) gives the positions of all points relative to the cylinder’s base point (the starting point of the axis)
        # np.dot(..., axis_normalized) projects each of those points onto the axis direction — so for each point, we get how far along the axis it lies
        # np.ptp() means peak to peak (max - min). It tells us the total spread of those projected points — the distance between the lowest and highest points along the axis.
        # That’s exactly the height of the cylinder that best spans all the data points.

        h = np.ptp(np.dot(points - np.array([x0, y0, z0]), axis_normalized))
        
        # Compute the projection of each point onto the cylinder axis
        vectors = points - np.array([x0, y0, z0])
        scalar_proj = np.dot(vectors, axis_normalized)
        projection = (scalar_proj[:, None] * axis_normalized[None, :])
        # projection = np.array([x0, y0, z0]) + np.outer(scalar_proj, axis_normalized)
        
        # Compute the distance of each point to the cylinder axis
        radial_vec = vectors - projection
        dist_to_axis = np.linalg.norm(radial_vec, axis=1)
        
        # Compute the distance of each point to the cylinder bottom/top surfaces
        # Hint: if dist_to_height is used as a residual, shall we penalise the points that are right on the cylinder surface but with a postiive dist_to_height measures?
        dist_bottom_plane = -scalar_proj
        dist_top_plane = scalar_proj - h
        dist_to_height = np.maximum(0.0, np.maximum(dist_bottom_plane, dist_top_plane))
        # dist_to_height = np.maximum(np.abs(scalar_proj) - 0.5*h, 0)
        
        # Compute the residuals: distance of each point to the cylinder surface
        # Take the absolute value of the point to axis or height RESIDUE, whichever is greater
        side_residual = dist_to_axis - r
        dist_to_cyl = np.maximum(np.abs(side_residual), dist_to_height)
        ### End of your code ###
        
        return dist_to_cyl



def cylinder_jac(params, points):
    """
    Compute the Jacobian matrix for fitting points to a cylinder.

    Args:
        params (np.array): A 1x7 array [x0, y0, z0, dx, dy, dz, r].
        points (np.array): Nx3 array of 3D points.

    Returns:
        np.array: Jacobian matrix (Nx7),
                  partial derivatives of the residuals w.r.t [x0, y0, z0, dx, dy, dz, r].
    """

    x0, y0, z0, dx, dy, dz, r = params
    c = np.array([x0, y0, z0])
    d = np.array([dx, dy, dz])

    # Normalize axis direction (safely)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        d_norm = 1e-12
    d = d / d_norm

    # Compute vectors from axis base point to each data point
    v = points - c  # shape (N,3)

    # Perpendicular component from axis to point
    perp = v - proj  # shape (N,3)
    perp_len = np.linalg.norm(perp, axis=1)
    safe_len = np.maximum(perp_len, 1e-12)

    # Initialize Jacobian (N x 7)
    J = np.zeros((points.shape[0], 7))

    # --- Derivatives with respect to (x0, y0, z0) ---
    # Shifting the base point moves v -> v - dc, thus affects perp directly.
    # ∂res/∂c = - (perp / ||perp||)
    J[:, 0:3] = -perp / safe_len[:, None]

    # --- Derivatives with respect to (dx, dy, dz) ---
    # Axis direction affects projection: ∂proj/∂d = t_i * I - (v_i ⋅ d) * something
    # Approximation: the direction change effect on perp ≈ -(v ⋅ d) component orthogonalized.
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = 1.0
        # derivative of projection wrt axis direction
        d_proj = np.outer(t, e_j) + np.outer(np.dot(v, e_j), d) * 0.0  # simplified form
        # approximate partial: derivative of perp wrt d = -d_proj
        d_perp = -np.outer(t, e_j)
        # derivative of residual wrt d_j = (perp ⋅ d_perp) / ||perp||
        J[:, 3 + j] = np.sum(perp * d_perp, axis=1) / safe_len

    # --- Derivative with respect to radius ---
    J[:, 6] = -1.0

    return J


     
    
def fit_cylinder(points):
    initial_guess = [0, 0, 0, 0, 0, 1, 0.05]
    ### Bonus: Implement the Gauss-Newton optimization function to fit the cylinder parameters ###
    result = least_squares(cylinder_residuals, initial_guess, args=(points,))
    return result.x  # Estimated cylinder parameters



def ransac_lollipop_fitting(points, max_iterations= 100, threshold=0.001):
    best_sphere_params = None
    best_cylinder_params = None
    best_inliers_count = 0
    best_cylinder_height = 0

    sphere_points = points
    
    for _ in range(max_iterations):
        ### Your code here ###
        # Sample 10 points from the sphere points
        k_sph = min(10, sphere_points.shape[0])
        random_sph = np.random.choice(sphere_points.shape[0], k_sph, False)
        sphere_sample = sphere_points[random_sph]
        
        # Fit a sphere to the sampled points
        sphere_params = fit_sphere(sphere_sample)
        
        # Compute the residuals of the sphere fit from all points
        sphere_distances = np.abs(sphere_residuals(sphere_params, points))
        
        # Identify inliers (points with distance to sphere within the threshold) based on the residuals
        sphere_inliers_mask = (sphere_distances <= threshold)
        sphere_inliers = points[sphere_inliers_mask]
        
        # If the number of inliers is less than 50, continue to the next iteration
        if len(sphere_inliers) < 50:
            continue
        
        # Select the points that are not inliers to the sphere and use them to fit the cylinder
        cylinder_points = points[~sphere_inliers_mask]
        k_cyl = min(10, cylinder_points.shape[0])
        random_cyl = np.random.choice(cylinder_points.shape[0], k_cyl, False)
        
        # Sample 10 points from the cylinder points
        cylinder_sample = cylinder_points[random_cyl]
        
        # Fit a cylinder to the sampled points
        cylinder_params = fit_cylinder(cylinder_sample)
        
        # Compute the residuals of the cylinder fit from all points
        dist_to_cyl = np.abs(cylinder_residuals(cylinder_params, points))
        
        # Compute the total number of inliers for both sphere and cylinder fits
        cylinder_inliers_mask = (dist_to_cyl <= threshold)
        exclusive_cylinder_inliers_mask = cylinder_inliers_mask & (~sphere_inliers_mask)
        total_inliers_count = np.sum(sphere_inliers_mask) + np.sum(exclusive_cylinder_inliers_mask)
        
        # Update the best parameters if the current inliers count is higher
        if total_inliers_count > best_inliers_count:
            best_inliers_count = int(total_inliers_count)
            best_sphere_params = sphere_params
            best_cylinder_params = cylinder_params

            # Height from inlier axial spread
            c = cylinder_params[:3]
            a = cylinder_params[3:6]
            a = a / (np.linalg.norm(a))
            t_in = (points[exclusive_cylinder_inliers_mask] - c) @ a
            best_cylinder_height = float(np.ptp(t_in)) if t_in.size else 0.0
            if t_in.size:
                t_min = float(np.min(t_in))
                mid_along_axis = t_min + 0.5 * best_cylinder_height
                best_cylinder_params[:3] = c + mid_along_axis * a
        
        # Update the sphere points to exclude the inliers to the cylinder
        sphere_points = points[dist_to_cyl > threshold]
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
