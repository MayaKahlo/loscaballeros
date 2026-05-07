import json
import os
import student
import helpers
import cv2
import numpy as np
import networkx as nx
import open3d as o3d
import matplotlib.pyplot as plt
import xatlas
import pandas as pd
from scipy.spatial.transform import Rotation

# generate planes from point cloud
# goal: group points that are (1) nearby each other and (2) form a plane

# import point cloud from output file
pcd = o3d.io.read_point_cloud("online.ply")
plane_transforms = []

print('original points', len(pcd.points))

# estimate surface normals
o3d.geometry.PointCloud.estimate_normals(pcd)
assert (pcd.has_normals())

# using all defaults
oboxes = pcd.detect_planar_patches(
    normal_variance_threshold_deg=70,
    coplanarity_deg=85,
    outlier_ratio=0.5,
    min_plane_edge_length=0,
    min_num_points=500,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))

print("Detected {} patches".format(len(oboxes)))

# visualize planes and their bounding boxes
geometries = []
inlier_indices = []
for index, obox in enumerate(oboxes):
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
    mesh.paint_uniform_color(obox.color)
    geometries.append(mesh)
    geometries.append(obox)

    # track inliers
    indices = obox.get_point_indices_within_bounding_box(pcd.points)
    inlier_indices.extend(indices)

    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
        obox, scale=[1, 1, 0.0001]
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"plane_{index}.obj", mesh)

    ''''position = obox.center.tolist()
    rot = Rotation.from_matrix(obox.R)
    euler = rot.as_euler('xyz').tolist()
    scale = (obox.extent).tolist()
    plane_transforms.append({
        "id": index,
        "name": f"Plane_{index}",
        "position": {"x": position[0], "y": position[1], "z": position[2]},
        "rotation": {"x": euler[0], "y": euler[1], "z": euler[2]},
        "scale": {"x": scale[0], "y": scale[1], "z": scale[2]}
    })

    

# geometries.append(pcd)

# write plane transforms out
with open("plane_transforms.json", "w") as f:
    json.dump(plane_transforms, f, indent=4)

'''

o3d.visualization.draw_geometries(geometries,
                                  zoom=0.62,
                                  front=[0.4361, -0.2632, -0.8605],
                                  lookat=[2.4947, 1.7728, 1.5541],
                                  up=[-0.1726, -0.9630, 0.2071])

# remove all plane inliers from the point cloud
no_planes_pcd = pcd.select_by_index(inlier_indices, invert=True)

# use dbscan on remaining points
print('nonplanar points', len(no_planes_pcd.points))
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(no_planes_pcd.cluster_dbscan(eps=0.08, min_points=10, print_progress=True))

max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

# iterate through points in each cluster
for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    cluster_points = np.asarray(no_planes_pcd.points)[cluster_indices]
    print(f"Cluster {i} has {len(cluster_points)} points")

    # generate a triangle mesh
    cluster = o3d.geometry.PointCloud()
    cluster.points = o3d.utility.Vector3dVector(cluster_points)
    o3d.geometry.PointCloud.estimate_normals(cluster)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cluster)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    
    # generate a uv mapped texture from triangle mesh
    # get faces (triangle indices) in correct format
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    
    # ensure contiguous arrays
    vertices = np.ascontiguousarray(vertices)
    faces = np.ascontiguousarray(faces)
    
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    xatlas.export(f"{i}_output.obj", vertices[vmapping], indices, uvs)
    


# Visualize clusters
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Black for noise
no_planes_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# center the viewer so that it is in the middle of points
no_planes_pcd.translate(-no_planes_pcd.get_center())
o3d.visualization.draw_geometries([no_planes_pcd])

'''
# iteratively fit a plane to the point cloud, remove inliers, and repeat until no more planes can be found
while True:
    # important that since there are so many points, ransac needs to fit a lot of points for a plane to count
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n = 100000,
                                             num_iterations = 10)
    [a, b, c, d] = plane_model

    print('inliers', len(inliers))

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1.0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    break
'''
'''
# cluster using dbscan 
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.08, min_points=10, print_progress=True))

max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

# Visualize clusters
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Black for noise
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# center the viewer so that it is in the middle of points
pcd.translate(-pcd.get_center())
o3d.visualization.draw_geometries([pcd])

'''