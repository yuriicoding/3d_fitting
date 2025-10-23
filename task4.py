"""Task 4: Constructive Solid Geometry (CSG) Operations with Primitives"""
import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
import mcubes


def determine_sphere_sdf(query_points, sphere_params):
    """Query sphere sdf for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        sphere_params (torch.tensor): Kx4 tensor of sphere parameters (center and radius).

    Returns:
        torch.tensor: Signed distance field of each sphere primitive with respect to each query point. NxK tensor.
    """
    
    ### Your code here ###
    # Determine the SDF value of each query point with respect to each sphere, you may reuse the function from task 3 with modifications
    center = sphere_params[:3]
    radius = sphere_params[3]
    distances = torch.norm(query_points - center, dim=1, keepdim=True)
    sphere_sdf = distances - radius
    ### End of your code ###

    return sphere_sdf

def determine_box_sdf(query_points, box_params):
    """Query box SDF for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        box_params (torch.tensor): 6 tensor of box parameters (center and half-extents).

    Returns:
        torch.tensor: SDF field of box primitive with respect to each query point.
    """
    ### Your code here ###
    # TODO: Implement box SDF calculation
    # Hint: Transform points to box local space, then calculate distance to axis-aligned box
    center = box_params[:3]
    half_extents = box_params[3:]
    local_points = query_points - center
    q = local_points.abs() - half_extents
    outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=1, keepdim=True)
    inside_dist = torch.clamp(torch.max(q, dim=1, keepdim=True).values, max=0.0)
    box_sdf = outside_dist + inside_dist
    ### End of your code ###

    return box_sdf


def determine_torus_sdf(query_points, torus_params):
    """Query torus SDF for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        torus_params (torch.tensor): 5 tensor of torus parameters (center, major_radius, minor_radius).

    Returns:
        torch.tensor: SDF field of torus primitive with respect to each query point.
    """
    ### Your code here ###
    # TODO: Implement torus SDF calculation
    # Hint: Project points to xz-plane for major radius, then calculate distance to torus
    # Major radius is the distance from center to the ring center
    # Minor radius is the thickness of the ring
    center = torus_params[:3]
    big_R = torus_params[3]
    small_R = torus_params[4]

    points = query_points - center
    q = torch.stack([torch.norm(points[:, [0, 2]], dim=1) - big_R, points[:, 1]], dim=1)
    torus_sdf = torch.norm(q, dim=1, keepdim=True) - small_R
    ### End of your code ###
    return torus_sdf


def csg_union(sdf1, sdf2):
    """Perform CSG union operation (minimum of two SDFs).
    
    Args:
        sdf1 (torch.tensor): SDF values for first shape
        sdf2 (torch.tensor): SDF values for second shape
        
    Returns:
        torch.tensor: Union SDF values
    """
    ### Your code here ###
    # TODO: Implement CSG union operation
    union_sdf = torch.minimum(sdf1, sdf2)
    ### End of your code ###
    return union_sdf


def csg_subtraction(sdf1, sdf2):
    """Perform CSG subtraction operation (sdf1 - sdf2).
    
    Args:
        sdf1 (torch.tensor): SDF values for shape to subtract from
        sdf2 (torch.tensor): SDF values for shape to subtract
        
    Returns:
        torch.tensor: Subtraction SDF values
    """
    ### Your code here ###
    # TODO: Implement CSG subtraction operation
    subtraction_sdf = torch.maximum(sdf1, -sdf2)
    ### End of your code ###
    return subtraction_sdf


def create_query_grid(resolution=64, bounds=(-2, 2)):
    """Create a 3D grid of query points.
    
    Args:
        resolution (int): Grid resolution
        bounds (tuple): Bounds for the grid (min, max)
        
    Returns:
        torch.tensor: Query points of shape (resolution^3, 3)
    """
    x = torch.linspace(bounds[0], bounds[1], resolution)
    y = torch.linspace(bounds[0], bounds[1], resolution)
    z = torch.linspace(bounds[0], bounds[1], resolution)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    return query_points


def extract_mesh_from_sdf(query_points, sdf_values, resolution=128, bounds=(-2, 2)):
    """Extract mesh from SDF using PyMCubes marching cubes.
    
    Args:
        query_points (torch.tensor): Query points
        sdf_values (torch.tensor): SDF values
        resolution (int): Grid resolution
        bounds (tuple): Grid bounds
        
    Returns:
        trimesh.Trimesh: Extracted mesh
    """
    # Reshape SDF values to 3D grid
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution).cpu().numpy()
    
    # Use PyMCubes marching cubes to extract mesh
    vertices, faces = mcubes.marching_cubes(sdf_grid, 0.0)
    
    # Scale vertices to correct coordinate system
    scale = (bounds[1] - bounds[0]) / (resolution - 1)
    vertices = vertices * scale + bounds[0]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def main():
    """Main function to demonstrate CSG operations with sphere, box, and torus."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define grid resolution
    resolution = 128
    # Define three different primitives
    sphere_params = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).to(device)  # A: center (0,0,0), radius 1.0
    box_params = torch.tensor([0.8, 0.0, 0.0, 0.6, 0.6, 0.6], dtype=torch.float32).to(device)  # B: center (0.8,0,0), half-extents (0.6,0.6,0.6)
    torus_params = torch.tensor([0.0, 0.6, 0.0, 0.8, 0.3], dtype=torch.float32).to(device)  # C: center (0,0.6,0), major_radius 0.8, minor_radius 0.3
    
    print("Primitive configuration:")
    print(f"A (Sphere): center {sphere_params[:3].cpu().numpy()}, radius {sphere_params[3].item():.2f}")
    print(f"B (Box): center {box_params[:3].cpu().numpy()}, half-extents {box_params[3:].cpu().numpy()}")
    print(f"C (Torus): center {torus_params[:3].cpu().numpy()}, major_radius {torus_params[3].item():.2f}, minor_radius {torus_params[4].item():.2f}")
    
    # Create query grid
    query_points = create_query_grid(resolution=resolution, bounds=(-2, 2)).to(device)
    print(f"Query grid shape: {query_points.shape}")
    
    # Calculate individual primitive SDFs
    print("Calculating SDFs...")
    sdf_a = determine_sphere_sdf(query_points, sphere_params)
    sdf_b = determine_box_sdf(query_points, box_params)
    sdf_c = determine_torus_sdf(query_points, torus_params)
    
    # Perform CSG operations
    print("Performing CSG operations...")
    sdf_union_ab = csg_union(sdf_a, sdf_b)
    sdf_subtract_ac = csg_subtraction(sdf_a, sdf_c)
    
    # Implement the union of the union of A and B with C
    sdf_union_abc = csg_union(sdf_union_ab, sdf_c)
    # Implement the union of A and B subtracted by C
    sdf_union_ab_subtract_c = csg_subtraction(sdf_union_ab, sdf_c)
    
    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and save meshes
    print("Extracting meshes...")
    operations = {
        "A_union_B": sdf_union_ab,
        "A_subtract_C": sdf_subtract_ac,
        "A_union_B_union_C": sdf_union_abc,
        "A_union_B_subtract_C": sdf_union_ab_subtract_c,
    }
    
    #END OF BONUS
    
    for op_name, sdf_values in operations.items():
        mesh = extract_mesh_from_sdf(query_points, sdf_values, resolution=resolution)
        if mesh is not None:
            mesh_path = os.path.join(output_dir, f"{op_name}.obj")
            mesh.export(mesh_path)
            print(f"Saved {op_name} to {mesh_path}")
    
    # Save individual primitive meshes
    mesh_a = extract_mesh_from_sdf(query_points, sdf_a)
    mesh_b = extract_mesh_from_sdf(query_points, sdf_b)
    mesh_c = extract_mesh_from_sdf(query_points, sdf_c)
    
    if mesh_a is not None:
        mesh_a.export(os.path.join(output_dir, "primitive_A.obj"))
    if mesh_b is not None:
        mesh_b.export(os.path.join(output_dir, "primitive_B.obj"))
    if mesh_c is not None:
        mesh_c.export(os.path.join(output_dir, "primitive_C.obj"))



    #BONUS
    ring_torus = torch.tensor([-1.00, 0.20, 0.0, 0.35, 0.12], dtype=torch.float32).to(device)
    connector_box = torch.tensor([-0.55, 0.20, 0.0, 0.15, 0.12, 0.12], dtype=torch.float32).to(device)
    shoulder_sphere = torch.tensor([-0.40, 0.20, 0.0, 0.22], dtype=torch.float32).to(device)
    shaft_box = torch.tensor([-0.05, 0.20, 0.0, 0.80, 0.12, 0.12], dtype=torch.float32).to(device)
    tooth1_box = torch.tensor([0.35, 0.05, 0.0, 0.08, 0.12, 0.12], dtype=torch.float32).to(device)
    tooth2_box = torch.tensor([0.55, 0.12, 0.0, 0.08, 0.16, 0.12], dtype=torch.float32).to(device)
    tooth3_box = torch.tensor([0.75, 0.00, 0.0, 0.06, 0.20, 0.12], dtype=torch.float32).to(device)
    groove_box = torch.tensor([0.15, 0.32, 0.0, 0.20, 0.04, 0.10], dtype=torch.float32).to(device)

    sdf_ring = determine_torus_sdf(query_points, ring_torus)
    sdf_conn = determine_box_sdf(query_points, connector_box)
    sdf_shoulder = determine_sphere_sdf(query_points, shoulder_sphere)
    sdf_shaft = determine_box_sdf(query_points, shaft_box)
    sdf_tooth1 = determine_box_sdf(query_points, tooth1_box)
    sdf_tooth2 = determine_box_sdf(query_points, tooth2_box)
    sdf_tooth3 = determine_box_sdf(query_points, tooth3_box)
    sdf_groove = determine_box_sdf(query_points, groove_box)

    key_step1 = csg_union(sdf_ring, sdf_conn)
    key_step2 = csg_union(key_step1, sdf_shoulder)
    key_step3 = csg_union(key_step2, sdf_shaft)
    key_step4 = csg_subtraction(key_step3, sdf_tooth1)
    key_step5 = csg_subtraction(key_step4, sdf_tooth2)
    key_step6 = csg_subtraction(key_step5, sdf_tooth3)
    key_final = csg_subtraction(key_step6, sdf_groove)

    mesh = extract_mesh_from_sdf(query_points, key_final, resolution=resolution)
    if mesh is not None:
        mesh.export(os.path.join(output_dir, "KEY_FINAL.obj"))

if __name__ == "__main__":
    main()
