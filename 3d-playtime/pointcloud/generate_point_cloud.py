import os
import student
import helpers
import cv2
import numpy as np
import networkx as nx
import open3d as o3d
import matplotlib.pyplot as plt


# Import images from data directory
data = 'NotreDame'
image_dir = os.path.join('data', data)
image_paths = sorted([
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpeg', '.jpg'))
])
print(f"\n  Loading {len(image_paths)} images...")

images = []
scale = 0.5
for path in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    images.append(img)

print(f"    Resized to {images[0].shape[1]}x{images[0].shape[0]}")

sift_points = []
# Run SIFT on all images
sift = cv2.SIFT_create(nfeatures=10000)
for img in images:
    kp1, des1 = sift.detectAndCompute(img, None)
    sift_points.append((kp1, des1))

# Match keypoint descriptors between all image pairs
bf = cv2.BFMatcher(cv2.NORM_L2)
track_graph = nx.Graph()
matches = {}
for i in range(len(sift_points)):
    for j in range(i + 1, len(sift_points)):
        des1 = sift_points[i][1]
        des2 = sift_points[j][1]
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # use raw points that count as good matches
        points1 = np.array([sift_points[i][0][m.queryIdx].pt for m in good_matches])
        points2 = np.array([sift_points[j][0][m.trainIdx].pt for m in good_matches])

        if len(good_matches) >= 8:
            print(f"\n  Image pair ({i}, {j}): {len(good_matches)} good matches")

            try:
                ## Estimate fundamental matrix using RANSAC for each image pair
                F_est, inliers1, inliers2, residual, inlier_idxs = \
                    student.ransac_fundamental_matrix(points1, points2, 100)
                print(f"    RANSAC inliers: {len(inliers1)} / {len(points1)}  "
                    f"residual={residual:.2f}")

                ## Remove matches that are outliers to best fundamental matrix
                ## Remove all matches if number of matches is less than 20
                if len(inliers1) >= 20:
                    matches[(i, j)] = []
                    for idx_idx, idx in enumerate(inlier_idxs):
                        m = good_matches[idx]
                        matches[(i, j)].append((inliers1[idx_idx], inliers2[idx_idx], m.queryIdx, m.trainIdx))
                        track_graph.add_edge((i, m.queryIdx), (j, m.trainIdx))
            except np.linalg.LinAlgError:
                print(f"skipping pair ({i}, {j}) due to linear algebra error in RANSAC")
            except AttributeError:
                print(f"sKipping pair ({i}, {j}) due to no inliers found")

            
# Construct tracks of matching keypoints across multiple images
# Each track will be a list of tuples like [(image_index, inlier_value)]
tracks = list(nx.connected_components(track_graph))
print(f"\n  Found {len(tracks)} tracks across all images")
# Exclude tracks with repeat images
nonrepeat_tracks = []
camera_image_points = {}
track_id_to_nodes = {}
for track_id, track in enumerate(tracks):
    # Get all of the images for all nodes in this track
    images_in_track = set(node[0] for node in track)
    # if any images were repeats, this conditional will fail
    if len(images_in_track) == len(track) and len(images_in_track) > 1:
        nonrepeat_tracks.append(track)
        track_id_to_nodes[track_id] = track

print(f"\n {len(nonrepeat_tracks)} tracks did not contain repeat images and are valid.")

# Structure from motion https://dl.acm.org/doi/pdf/10.1145/1179352.1141964 

# Choose an initial image pair with the most matches that can't be modeled by a single homography
max_tuple = max(matches, key=lambda x: len(matches[x]))
print(f"Image pair with the maximum number of matches is: {max_tuple[0]} and {max_tuple[1]} with {len(matches[max_tuple])} matches")
used_cameras = set([max_tuple[0], max_tuple[1]])

## initialize K with best guess
k = np.array([[images[0].shape[1] + images[0].shape[0], 0, images[0].shape[1] / 2],
              [0, images[0].shape[1] + images[0].shape[0], images[0].shape[0] / 2],
              [0, 0, 1]])
## compute essential matrix E
inliers1 = [match[0] for match in matches[max_tuple]]
inliers1 = np.array(inliers1)
camera_image_points[max_tuple[0]] = inliers1
inliers2 = [match[1] for match in matches[max_tuple]]
inliers2 = np.array(inliers2)
camera_image_points[max_tuple[1]] = inliers2
E, _ = cv2.findEssentialMat(inliers1, inliers2, k, method=cv2.RANSAC)
_, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, k)
## set first image camera to origin
R1 = np.eye(3)
t1 = np.zeros((3, 1))
P1 = k @ np.hstack((R1, t1))
## set other image camera to k[R | t]
R2 = R
t2 = t
P2 = k @ np.hstack((R2, t2))
## triangulate points to get 3d reconstruction
points_3d_hom = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
## convert to 3d
points_3d = points_3d_hom[:3, :] / points_3d_hom[3, :]
print(f"\n  Initial reconstruction has {points_3d.shape[1]} points")

# Map tracks to 3D points for the initial pair
track_to_3d = {}
for idx, match in enumerate(matches[max_tuple]):
    # match is (inlier1, inlier2, queryIdx, trainIdx)
    query_kp = (max_tuple[0], match[2])
    train_kp = (max_tuple[1], match[3])
    # Find which track contains both keypoints
    for track_id, nodes in track_id_to_nodes.items():
        if query_kp in nodes and train_kp in nodes:
            track_to_3d[track_id] = points_3d[:, idx]
            break

## intermediate check
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T.astype(np.float32))
o3d.io.write_point_cloud(f"{data}_basic_output.ply", pcd)

# Iteratively add cameras with the most number of tracks that are visible in the current reconstruction
remaining_camera = set(range(len(images))) - used_cameras
cameras = [None] * len(images)
cameras[max_tuple[0]] = (R1, t1)
cameras[max_tuple[1]] = (R2, t2)

for camera in remaining_camera:
    object_points = []
    image_points = []

    # Find image points in this camera that are currently in the 3d cloud
    ## iterate through matches between this camera and all others
    for other_camera in used_cameras:
        pair = (min(camera, other_camera), max(camera, other_camera))
        if pair in matches:
            ## for each match, check if the track is in the current reconstruction
            for match in matches[pair]:
                if camera < other_camera:
                    this_img_pt = match[0]
                    this_kp = (camera, match[2])
                    other_kp = (other_camera, match[3])
                else:
                    this_img_pt = match[1]
                    this_kp = (camera, match[3])
                    other_kp = (other_camera, match[2])
                
                # Find the track that contains both keypoints
                for track_id, nodes in track_id_to_nodes.items():
                    if this_kp in nodes and other_kp in nodes and track_id in track_to_3d:
                        object_points.append(track_to_3d[track_id])
                        image_points.append(this_img_pt)
                        break
    camera_image_points[camera] = np.array(image_points)

    # run pnp
    if len(object_points) >= 6:
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        R, t, inliers = student.pnp_ransac(object_points, image_points, k)
        print(f"Added camera {camera} with {len(inliers)} inliers")
        
        # Store the camera pose
        cameras[camera] = (R, t)
        used_cameras.add(camera)
        
        # triangulate new 3D points visible in this camera
        for track_id, nodes in track_id_to_nodes.items():
            # add points on new tracks not seen by already added cameras
            if track_id not in track_to_3d: 
                cameras_seeing_track = [node[0] for node in nodes if node[0] in used_cameras]
                if len(cameras_seeing_track) >= 2:
                    # Triangulate using this camera and the first existing camera that sees it
                    cam1 = camera
                    cam2 = next(c for c in cameras_seeing_track if c != camera)
                    
                    try:
                        kp1 = next(node for node in nodes if node[0] == cam1)
                        kp2 = next(node for node in nodes if node[0] == cam2)
                    except StopIteration:
                        continue
                    
                    pt1 = sift_points[cam1][0][kp1[1]].pt
                    pt2 = sift_points[cam2][0][kp2[1]].pt
                    
                    M1 = student.build_camera_matrix(k, *cameras[cam1])
                    M2 = student.build_camera_matrix(k, *cameras[cam2])
                    
                    point_3d = student.triangulate(M1, M2, np.array([pt1]), np.array([pt2]))[0]
                    
                    # ensure point is in front of both cameras
                    valid = True
                    for cam in [cam1, cam2]:
                        R, t = cameras[cam]
                        point_cam = R @ point_3d + t.flatten()
                        if point_cam[2] <= 0:
                            valid = False
                            break
                    
                    if valid:
                        track_to_3d[track_id] = point_3d
        
        # Update points_3d array
        all_points_3d = np.array([track_to_3d[track_id] for track_id in sorted(track_to_3d.keys())])
        print(f"Reconstruction now has {all_points_3d.shape[0]} points")

        '''
        # Run batch sparse bundle adjustment on each camera addition
        used_camera_indices = sorted(used_cameras)
        ba_cameras = [cameras[i] for i in used_camera_indices]
        camera_idx_to_ba_idx = {cam_idx: ba_idx for ba_idx, cam_idx in enumerate(used_camera_indices)}

        ba_tracks = {}
        ba_image_points = [dict() for _ in used_camera_indices]
        for ba_point_idx, track_id in enumerate(sorted(track_to_3d.keys())):
            observations = []
            for cam_idx, kp_idx in track_id_to_nodes[track_id]:
                if cam_idx in camera_idx_to_ba_idx:
                    ba_cam_idx = camera_idx_to_ba_idx[cam_idx]
                    observations.append((ba_cam_idx, kp_idx))
                    ba_image_points[ba_cam_idx][kp_idx] = sift_points[cam_idx][0][kp_idx].pt
            if observations:
                ba_tracks[ba_point_idx] = observations

        _, refined_points = student.bundle_adjustment(
            ba_cameras,
            all_points_3d,
            ba_tracks,
            ba_image_points,
            k
        )

        for ba_point_idx, track_id in enumerate(sorted(track_to_3d.keys())):
            track_to_3d[track_id] = refined_points[ba_point_idx]
        '''
        
    else:
        print(f"Skipped camera {camera}, only {len(object_points)} points")

# Final output
if track_to_3d:
    points_3d = np.array(list(track_to_3d.values()))
    tracks = {}
    image_points_list = [[] for _ in range(len(images))]
    
    final_points = np.array(list(track_to_3d.values()))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points.astype(np.float32))

    o3d.io.write_point_cloud(f"{data}_final_output.ply", pcd)
    print(f"Saved final point cloud to {data}_final_output.ply")

    

    