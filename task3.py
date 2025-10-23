"""Task 3: Shape abstraction with neural networks"""
# Part of the code in adopted from DualSDF repository.
import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from dgcnn import DGCNNFeat


def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def determine_sphere_sdf(query_points, sphere_params):
    """Query sphere sdf for a set of points.

    Args:
        query_points (torch.tensor): Nx3 tensor of query points.
        sphere_params (torch.tensor): Kx4 tensor of sphere parameters (center and radius).

    Returns:
        torch.tensor: Signed distance field of each sphere primitive with respect to each query point. NxK tensor.
    """

    ### Your code here ###
    # Determine the SDF value of each query point with respect to each sphere ###
    points = query_points.unsqueeze(1)
    center = sphere_params[:, :3]
    radius = sphere_params[:, 3]
    distance = torch.linalg.norm(points - center, dim=2)
    sphere_sdf = distance - radius
    ### End of your code ###
    return sphere_sdf


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        out_ch = 1024
        feat_ch = 512

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2


class SphereNet(nn.Module):
    def __init__(self, num_spheres=256):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        sphere_params = self.decoder(features)
        
        ### Comment on the following 4 lines, why do we have to do it?###

        # Sigmoid compresses the raw decoder outputs into the range [0, 1], preventing extremely large or negative values. 
        # This keeps training numerically stable and ensures the radius starts positive.
        # Sphere multiplier and adder then remap those normalized values into meaningful ranges:
        # Sphere centers → [-0.5, 0.5], so they stay near the object’s coordinate space.
        # Sphere radii → [0.1, 0.5], ensuring every sphere is visible, positive, and not oversized.
        #These transformations regularize the parameter space: they retsrict learning 
        # to a plausible region, speed up convergence, and prevent exploding gradients.
        #If they are removed, the decoder outputs become unbounded - sphere centers may drift far away, 
        # radii may become negative or excessively large, 
        # and hence the network might collapse or diverge because gradients through the SDF and bsmin become unstable.

        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder

        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params


def visualise_spheres(sphere_params, reference_model, save_path=None):
    sphere_params = sphere_params.cpu().detach().numpy()
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])
    scene = trimesh.Scene()
    if reference_model is not None:
        scene.add_geometry(reference_model)
    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(center)
        scene.add_geometry(sphere)
    if save_path is not None:
        scene.export(save_path)
    scene.show()


def visualise_sdf(points, values):
    """Visualise the SDF values as a point cloud."""
    # Use trimesh to create a point cloud from the SDF values
    inside_points = points[values < 0]
    outside_points = points[values > 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    outside_points = trimesh.points.PointCloud(outside_points)
    inside_points.colors = [0, 0, 1, 1]  # Blue color for inside points
    outside_points.colors = [1, 0, 0, 1]  # Red color for outside points
    scene = trimesh.Scene()
    scene.add_geometry([inside_points, outside_points])
    scene.show()


def main():
    dataset_path = "./data"
    name = "shiba"

    mesh_model = trimesh.load(os.path.join(dataset_path, f"{name}_model.obj"))
    pcd_model = trimesh.load(
        os.path.join(dataset_path, f"{name}_surface_pointcloud.ply")
    )
    points, values = (
        np.load(f"data/{name}.npz")["points"],
        np.load(f"data/{name}.npz")["values"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)
    surface_pointcloud = torch.from_numpy(pcd_model.vertices).float().to(device)

    model = SphereNet(num_spheres=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 500
    for i in range(num_epochs):
        optimizer.zero_grad()
        sphere_sdf, sphere_params = model(
            surface_pointcloud.unsqueeze(0).transpose(2, 1), points
        )
        ### Explain why the following line is necessary and what does it do###
        # Since each query point sphere_sdf has distances to many spheres, 
        # we need the minimum distance to represent the union of spheres. 
        # The function bsmin computes a smooth, differentiable version of the min, 
        # so gradients can get to multiple spheres during training instead of just the closest one. 
        # This makes learning stable and allows all spheres to adjust their positions and sizes together.
        sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)

        ### Your code here ###
        ### Determine the loss function to train the model, i.e. the mean squared error between gt sdf field and predicted sdf field. ###
        loss_fn = torch.nn.MSELoss()
        mseloss = loss_fn(sphere_sdf.squeeze(), values.squeeze())

        #Bonus 
        # Penalize small spheres (encourage r >= r_min)
        # r = sphere_params[..., 3]
        # r_min = 0.10
        # small_sphere_penalty = torch.relu(r_min - r).mean()
        # mseloss += 0.2 * small_sphere_penalty
        ### End of your code ###

        loss = mseloss
        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{name}_sphere_params.npy"), sphere_params.cpu().detach().numpy())
    
    print(sphere_params)

    visualise_spheres(sphere_params, reference_model=pcd_model, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))


if __name__ == "__main__":
    main()
