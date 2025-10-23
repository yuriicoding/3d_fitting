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
        dists = np.linalg.norm(points - center, axis=1)
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

    # Initialize the Jacobian matrix (Nx4)
    J = np.zeros((points.shape[0], 4))

    diff = points - center  
    dists = np.linalg.norm(diff, axis=1)
    

    # Partials w.r.t center coordinates: -(diff / ||diff||)
    J[:, 0] = -diff[:, 0] / dists
    J[:, 1] = -diff[:, 1] / dists
    J[:, 2] = -diff[:, 2] / dists

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
        # np.dot(..., axis_normalized) projects each of those points onto the axis direction — 
        # so for each point, we get how far along the axis it lies
        # np.ptp() means peak to peak (max - min). It tells us the total spread of those projected points — 
        # the distance between the lowest and highest points along the axis.
        # That’s exactly the height of the cylinder that best spans all the data points.

        h = np.ptp(np.dot(points - np.array([x0, y0, z0]), axis_normalized))
        
        # Compute the projection of each point onto the cylinder axis
        vectors = points - np.array([x0, y0, z0])
        scalar_proj = np.dot(vectors, axis_normalized)
        projection = (scalar_proj[:, None] * axis_normalized[None, :])
        
        # Compute the distance of each point to the cylinder axis
        radial_vec = vectors - projection
        dist_to_axis = np.linalg.norm(radial_vec, axis=1)
        
        # Compute the distance of each point to the cylinder bottom/top surfaces
        # Hint: if dist_to_height is used as a residual, shall we penalise the points that are right on the cylinder surface but with a postiive dist_to_height measures?
        dist_bottom_plane = -scalar_proj
        dist_top_plane = scalar_proj - h
        dist_to_height = np.maximum(0.0, np.maximum(dist_bottom_plane, dist_top_plane))
        
        # Compute the residuals: distance of each point to the cylinder surface
        # Take the absolute value of the point to axis or height RESIDUE, whichever is greater
        side_residual = dist_to_axis - r
        dist_to_cyl = np.maximum(np.abs(side_residual), dist_to_height)
        ### End of your code ###
        
        return dist_to_cyl



def cylinder_jac(params, points):

    x0, y0, z0, dx, dy, dz, r = params
    c = np.array([x0, y0, z0], dtype=float)
    d = np.array([dx, dy, dz], dtype=float)

    d_norm = np.linalg.norm(d)
    u = d / d_norm

    v   = points - c
    t   = v @ u
    proj = t[:, None] * u[None, :]
    perp = v - proj
    perp_len = np.linalg.norm(perp, axis=1)

    h = np.ptp(t)

    side_residual = perp_len - r
    dist_bottom_plane = -t
    dist_top_plane    =  t - h
    dist_to_height    = np.maximum(0.0, np.maximum(dist_bottom_plane, dist_top_plane))

    use_side = (np.abs(side_residual) >= dist_to_height)
    N = points.shape[0]
    J = np.zeros((N, 7), dtype=float)

    idx = use_side
    sgn = np.sign(side_residual[idx])
    J[idx, 0:3] = (-perp[idx] / perp_len[idx, None]) * sgn[:, None]
    Ju = ( - (t[idx, None] * (perp[idx] / perp_len[idx, None])) ) * sgn[:, None]
    J[idx, 3:6] = Ju / d_norm
    J[idx, 6] = -sgn

    jdx = ~use_side
    bottom = jdx & (dist_bottom_plane >= dist_top_plane) & (dist_bottom_plane > 0)
    J[bottom, 0:3] =  u[None, :]
    J[bottom, 3:6] = -(perp[bottom] / d_norm)

    top = jdx & (dist_top_plane >= dist_bottom_plane) & (dist_top_plane > 0)
    J[top, 0:3] = -u[None, :]
    J[top, 3:6] =  (perp[top] / d_norm)

    return J




def fit_cylinder(points):

    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.05], dtype=float)

    for _ in range(1000):

        r = cylinder_residuals(x, points)
        J = cylinder_jac(x, points)

        delta, *_ = np.linalg.lstsq(J, r, rcond=None)
        x = x - delta

        axis = x[3:6]
        nrm = np.linalg.norm(axis)
        x[3:6] = axis / nrm

        if np.linalg.norm(delta) < 1e-4:
            break

    return x

     
    
# def fit_cylinder(points):
#     initial_guess = [0, 0, 0, 0, 0, 1, 0.05]
#     ### Bonus: Implement the Gauss-Newton optimization function to fit the cylinder parameters ###
#     result = least_squares(cylinder_residuals, initial_guess, args=(points,))
#     return result.x  # Estimated cylinder parameters



def ransac_lollipop_fitting(points, max_iterations= 100, threshold=0.01):
    best_sphere_params = None
    best_cylinder_params = None
    best_inliers_count = 0

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
            center = cylinder_params[:3]
            axis = cylinder_params[3:6]
            axis = axis / (np.linalg.norm(axis))
            cylinder_inliers = (points[exclusive_cylinder_inliers_mask] - center) @ axis

            if cylinder_inliers.size:
                best_cylinder_height = float(np.ptp(cylinder_inliers))
                t_min = float(np.min(cylinder_inliers))
                mid_along_axis = t_min + 0.5 * best_cylinder_height
                best_cylinder_params[:3] = center + mid_along_axis * axis
            else:
                best_cylinder_height = 0.0
        
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
