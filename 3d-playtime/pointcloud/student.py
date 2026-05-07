"""
=====================================
CSCI 1430 - Brown University
Camera Geometry - student.py
=====================================

  Part A -- Calibrated Geometry:
    Task 1: Camera Projection  (4 functions)
      camera_center(M)                              (~1 line)
      project(M, points3d)                          (~3 lines)
      reprojection_error(M, points3d, points2d)     (~1 line)
      estimate_camera_matrix(points2d, points3d)    (~15 lines)

    Task 2: Dense Stereo via Plane Sweeping  (4 functions)
      back_project(M, points2d, lambdas)            (~6 lines)
      compute_depth_homography(M_ref, M_other, lam) (~10 lines)
      compute_ncc(ref_gray, warped_gray, win_size)  (~12 lines)
      plane_sweep_stereo(...)                       (~30 lines)

  Part B -- Uncalibrated Geometry:
    Task 3: Fundamental Matrix  (1 function)
      estimate_fundamental_matrix(points1, points2) (~20 lines)

    Task 4: RANSAC  (1 function)
      ransac_fundamental_matrix(matches1, matches2, num_iters)  (~30 lines)

    Task 5: Uncalibrated Stereo Disparity  (1 function)
      compute_disparity_map(rect_left_gray, rect_right_gray, win_size, max_disparity)  (~30 lines)

  Extra Credit:
    normalize_coordinates(points)
    compute_sampson_distance(F, points1, points2)
    decompose_projection_matrix(M)
    estimate_relative_pose(F, K1, K2, inliers1, inliers2)
    compute_plane_homography(M_ref, M_other, n, d)
"""

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# =============================================================================
#  Part A — Calibrated Geometry
# =============================================================================

# Task 1: Camera Projection
#
# Four functions that implement the projection matrix M: 
# - camera_center(): extract the camera center 
# - project(): forward-project 3D→2D 
# - reprojection_error(): measure reprojection error
# - estimate_camera_matrix(): estimate M from 2D-3D correspondences via the Direct Linear Transform (DLT).

def camera_center(M):
    """
    Extract the camera center in world coordinates from a 3x4 projection
    matrix.  Recall M = [A | m4], so:

        C = -A^{-1} m4

    :param M: 3x4 projection matrix
    :return: length-3 numpy array, the camera center in world coordinates
    """
    return (-np.linalg.inv(M[:, :3])@M[:, 3:]).flatten()


def project(M, points3d):
    """
    Forward-project 3D world points to 2D image coordinates using M.

        [su, sv, s]^T = M @ [X, Y, Z, 1]^T
        (u, v) = (su / s, sv / s)

    :param M: 3x4 projection matrix
    :param points3d: N x 3 array of 3D world coordinates
    :return: N x 2 array of 2D image coordinates (u, v)
    """
    '''
    # add a column of ones to the 3d world points
    points3d_homogenous = np.hstack((points3d, np.ones((points3d.shape[0], 1))))

    # perform matrix multiplication
    points3d_multiplied = (M @ points3d_homogenous.T).T

    # perspective division
    u = (points3d_multiplied[:, 0] / points3d_multiplied[:, 2]).reshape(-1, 1)
    v = (points3d_multiplied[:, 1] / points3d_multiplied[:, 2]).reshape(-1, 1)

    return np.hstack((u, v))
    '''
    N = points3d.shape[0]
    A = np.zeros((N, 2))

    for point in range(N):
        point3d_homogenous = np.hstack((points3d[point, :], [1]))
        point3d_multiplied = (M@point3d_homogenous.T).T

        # perspective division
        u = (point3d_multiplied[0] / point3d_multiplied[2])
        v = (point3d_multiplied[1] / point3d_multiplied[2])

        A[point, 0] = u
        A[point, 1] = v
    
    return A





def reprojection_error(M, points3d, points2d):
    """
    Per-point L2 reprojection error: how far each projected 3D point
    lands from its observed 2D position.

        error_i = || project(M, P_i) - p_i ||_2

    Use our project() function.

    :param M: 3x4 projection matrix
    :param points3d: N x 3 array of 3D world coordinates
    :param points2d: N x 2 array of observed 2D image coordinates
    :return: length-N array of L2 reprojection errors
    """
    return np.linalg.norm(project(M, points3d) - points2d, ord=2, axis=1)


def estimate_camera_matrix(points2d, points3d):
    """
    Estimate the 3x4 camera matrix M from 2D-3D correspondences via 
    the direct linear transform (DLT).

    Build the 2N x 12 matrix A from the homogeneous system Am = 0.  Each
    correspondence (u,v) <-> (X,Y,Z) gives two rows:

        [X Y Z 1  0 0 0 0  -uX -uY -uZ -u]
        [0 0 0 0  X Y Z 1  -vX -vY -vZ -v]

    Solve via SVD.

    After solving, compute and return the residual as the sum of squared reprojection
    errors using our reprojection_error() function.

    For extra credit: apply coordinate normalization to both points2d and
    points3d before building A, then un-normalize M afterward.
    See normalize_coordinates().

    :param points2d: N x 2 array of 2D image coordinates
    :param points3d: N x 3 array of corresponding 3D world coordinates
    :return: M, the 3x4 camera matrix
             residual, the sum of squared reprojection error (scalar)
    """

    N = points3d.shape[0]
    A = np.zeros((2 * N, 12))

    orig_points2d = points2d.copy()
    orig_points3d = points3d.copy()

    # noramlize points
    (points2d, T_2d) = normalize_coordinates(points2d)
    (points3d, T_3d) = normalize_coordinates(points3d)


    # iterate through N
    for point in range(N):
        idx = point * 2

        # handle first eight columns
        padded_point = np.hstack((points3d[point, :], 1))
        A[idx, :4] = padded_point
        A[idx+1, 4:8] = padded_point

        # handle last four columns
        A[idx, 8:12] = -points2d[point, 0] * padded_point
        A[idx+1, 8:12] = -points2d[point, 1] * padded_point

    # with matrix A, solve via SVD
    _, _, M = np.linalg.svd(A)
    M = M[-1, :]
    M = M.reshape((3, 4))

    # normalize M
    M = np.linalg.inv(T_2d)@M@T_3d

    return M, np.sum(reprojection_error(M, orig_points3d, orig_points2d)**2)


        


# Task 2: Dense Stereo via Plane Sweeping
#
# Four functions that turn calibrated cameras into a dense depth map.
# - back_project(): inverts the projection to create 3D points.  
# - compute_depth_homography(): builds the depth-dependent homography H(lambda).  
# - compute_ncc(): measures local similarity.
# - plane_sweep_stereo(): sweeps candidate depths and picks the best NCC.


def back_project(M, points2d, lambdas):
    """
    Back-project 2D image points to 3D world coordinates at given depths (lambdas).
    This is the INVERSE of project().

        C   = camera_center(M)
        ray = A^{-1} @ [u, v, 1]^T    (direction in world coordinates)
        P   = C + lam * ray

    :param M: 3x4 projection matrix
    :param points2d: N x 2 array of 2D image coordinates
    :param lambdas: length-N array of lambda (depth) parameters
    :return: N x 3 array of 3D world coordinates
    """

    N = points2d.shape[0]
    world_coords = np.zeros((N, 3))
    C = camera_center(M)
    A = M[:, :3]

    for point in range(N):
        homogenous_2d = np.hstack((points2d[point, :], 1))
        ray = np.linalg.inv(A)@homogenous_2d.T
        P = C + lambdas[point] * ray
        world_coords[point, :3] = P

    return world_coords


def compute_depth_homography(M_ref, M_other, lam):
    """
    Compute the depth-dependent homography between two views at given depth (lambda).

        B = A_other @ A_ref^{-1}
        a = A_other @ C_ref + t_other
        e3 = [0, 0, 1]^T
        H(lam) = lam * B + np.outer(a, e3)

    :param M_ref:   3x4 projection matrix of reference camera
    :param M_other: 3x4 projection matrix of other camera
    :param lam:     scalar lambda (depth) parameter
    :return: H, 3x3 homography matrix
    """

    A_other = M_other[:, :3]
    t_other = M_other[:, 3].reshape((3, 1))
    A_ref = M_ref[:, :3]
    C_ref = camera_center(M_ref).reshape((3, 1))
    
    B = A_other @ np.linalg.inv(A_ref)
    a = A_other @ C_ref + t_other
    e3 = np.array([0, 0, 1]).T

    ret = lam * B + np.outer(a, e3)
    return ret


def compute_ncc(ref_gray, warped_gray, win_size):
    """
    Compute per-pixel windowed normalized cross correlation (NCC) between 
    reference and warped image. NCC is exactly as it is named: we subtract 
    the mean of each patch and divide through by the standard deviation to
    remove brightness variations before correlating. E is the correlation 
    surface over a local window.

        NCC = (E[ref*warped] - E[ref]*E[warped]) / (std[ref]*std[warped] + eps)

    Implementation tips:
    - We can use cv2.boxFilter(img, -1, (win_size, win_size)) to compute a local mean.
    - Variance: var = E[x^2] - E[x]^2
    - Std:      std = sqrt(max(var, 0) + eps).

    :param ref_gray:    H x W float32, reference image
    :param warped_gray: H x W float32, warped image
    :param win_size:    int, window size
    :return: H x W float32, NCC scores in [-1, 1]
    """

    E_ref = cv2.boxFilter(ref_gray, -1, (win_size, win_size))
    E_ref_2 = cv2.boxFilter(ref_gray ** 2, -1, (win_size, win_size))
    E_warped = cv2.boxFilter(warped_gray, -1, (win_size, win_size))
    E_warped_2 = cv2.boxFilter(warped_gray ** 2, -1, (win_size, win_size))
    E_ref_warped = cv2.boxFilter(ref_gray * warped_gray, -1, (win_size, win_size))

    var_ref = E_ref_2 - E_ref**2
    var_warped = E_warped_2 - E_warped**2

    eps = 1e-6
    std_ref = np.sqrt(np.maximum(var_ref, 0) + eps)
    std_warped = np.sqrt(np.maximum(var_warped, 0) + eps)

    return (E_ref_warped - E_ref * E_warped) / (std_ref * std_warped + eps)


def plane_sweep_stereo(ref_gray, other_grays, M_ref, Ms_other, lambdas,
                       win_size=11, ncc_threshold=0.4):
    """
    Multi-view plane-sweep stereo.

    For each candidate depth lam in lambdas:
      1. For the other view, compute H(lam) and warp with
         cv2.warpPerspective(img, H, (w,h), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
      2. Compute NCC between ref and warped view
      3. If the NCC beats the current best at a pixel, update.
      
      4. [Better] Warp not just the first but _all_ images in other_grays
         Then, compute the average NCC across these views.

    Tips: 
    - We recommend _visualizing the warped image_ to make sure it makes sense.
    Can we (as a human) visually match them? What will NCC do?
    - NCC will always return some number; make sure we guard against incorrect values.
    
    :param ref_gray:    H x W float32, reference grayscale image
    :param other_grays: list of H x W float32, other-view images
    :param M_ref:       3x4 projection matrix of reference camera
    :param Ms_other:    list of 3x4 projection matrices
    :param lambdas:     1D array of lambda (depth) candidates
    :param win_size:    NCC window size
    :param ncc_threshold: minimum NCC to accept a lambda (depth) (default 0.4)
    :return: lambda_map, H x W float32 lambda (depth) map (NaN where unreliable)
    """

    W = other_grays[0].shape[1]
    H = other_grays[0].shape[0]

    # set all best nccs to worst possible value (-1)
    best_nccs = np.ones((H, W))
    best_nccs = best_nccs * -1

    best_lambdas = np.full((H, W), np.nan)

    # iterate through lambdas
    for lam in lambdas:
        ncc = np.zeros((H, W))
        for idx in range(len(Ms_other)):
            # compute depth homography
            H_lam = compute_depth_homography(M_ref, Ms_other[idx], lam)
            # warp
            warped = cv2.warpPerspective(other_grays[idx], H_lam, (W, H), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            # compute average NCC
            ncc += compute_ncc(ref_gray, warped, win_size)

        average_ncc = ncc / len(Ms_other)

        # iterate through pixels in window
        for row in range(H):
            for col in range(W):
                # check that this ncc is within the expected range
                curr_ncc = average_ncc[row, col]
                if curr_ncc > -1 and curr_ncc < 1 and curr_ncc > ncc_threshold:
                    if curr_ncc > best_nccs[row, col]:
                        best_nccs[row, col] = curr_ncc
                        best_lambdas[row, col] = lam

    return best_lambdas
                
                    




# =============================================================================
#  Part B — Uncalibrated Geometry
# =============================================================================

# Task 3: Fundamental Matrix Estimation

def estimate_fundamental_matrix(points1, points2):
    """
    Estimate the fundamental matrix F from point correspondences using
    the 8-point algorithm with SVD and rank-2 enforcement.

    Steps:
      1. Build the N x 9 data matrix A where each row is
         [u'*u, u'*v, u', v'*u, v'*v, v', u, v, 1]
      2. Solve Af = 0 via SVD: f is the last row of V^T.  Reshape to 3x3.
      3. Enforce rank 2: decompose F with SVD, zero out the smallest
         singular value, and reconstruct.

    The residual is: sum_i (x_i'^T F x_i)^2

    For extra credit: apply coordinate normalization before step 1 and
    un-normalize F afterward.  See normalize_coordinates().

    :param points1: N x 2 array of 2D points in image 1
    :param points2: N x 2 array of 2D points in image 2
    :return: F_matrix, the 3x3 fundamental matrix
             residual, the sum of squared algebraic error
    """
    # make copy for res
    orig_points1 = points1.copy()
    orig_points2 = points2.copy()

    # normalize points
    (points1, T_1) = normalize_coordinates(points1)
    (points2, T_2) = normalize_coordinates(points2)

    # construct A
    A = np.zeros((points1.shape[0], 9))
    for idx in range(points1.shape[0]):
        u, v = points1[idx]
        u_pr, v_pr = points2[idx]
        row = np.array([u_pr*u, u_pr*v, u_pr, v_pr*u, v_pr*v, v_pr, u, v, 1])
        A[idx] = row
    
    # solve via SVD
    _, _, V = np.linalg.svd(A)
    f = V[-1, :]
    f = f.reshape((3, 3))

    # Enforce rank 2: decompose F with SVD, zero out the smallest singular value, and reconstruct.
    U, s, decomposed = np.linalg.svd(f)
    s_rank = s[:2]
    f = U[:, :2] @ np.diag(s_rank) @ decomposed[:2, :]

    # unnormalize f
    f = T_2.T@f@T_1

    # compute residual
    res_total = 0
    for idx in range(orig_points1.shape[0]):
        # 1x3 3x3 3x1 = 1x1
        point1 = np.hstack((orig_points1[idx], [1]))
        point2 = np.hstack((orig_points2[idx], [1]))
        res_total += (point2.T @ f @ point1) ** 2

    return f, res_total

# Task 4: RANSAC 
# Robust estimation: find F despite noisy feature matches.

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC.

    For each iteration:
      1. Randomly sample 8 correspondences
      2. Estimate F using estimate_fundamental_matrix() on the sample
      3. Compute algebraic error |x'^T F x| for ALL correspondences
         (or Sampson distance if we implemented compute_sampson_distance() extra credit)
      4. Count inliers (error < threshold; start around 0.005 for algebraic error)
      5. Keep the F with the most inliers

    After the loop, re-estimate F from ALL inliers of the best model.

    For visualization, don't forget to append to inlier_counts and inlier_residuals each
    iteration (for the RANSAC convergence visualization).

    :param matches1: N x 2 array of 2D points in image 1
    :param matches2: N x 2 array of 2D points in image 2
    :param num_iters: number of RANSAC iterations
    :return: best_Fmatrix, best_inliers1, best_inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)

    # var initializations
    global inlier_counts, inlier_residuals
    max_inliers = 0
    best_inliers1 = None
    best_inliers2 = None
    inlier_idxs = None

    for iter in range(num_iters):
        # pick 8 random points to est fundamental matrix
        sample_idxs = np.random.choice(matches1.shape[0], 8, replace=False)
        samples1 = matches1[sample_idxs]
        samples2 = matches2[sample_idxs]
        F, res = estimate_fundamental_matrix(samples1, samples2)

        # get list of errors for all points
        errors = compute_sampson_distance(F, matches1, matches2)
        '''for idx in range(matches1.shape[0]):
            sample1_hom = np.hstack((matches1[idx], [1]))
            sample2_hom = np.hstack((matches2[idx], [1]))
            errors += [np.abs(sample2_hom.T @ F @ sample1_hom)]'''
        # convert to np array
        errors = np.array(errors)

        # get idx for all inliers
        inlier_idx = errors < 2
        # determine if this is max inliers
        if np.sum(inlier_idx) > max_inliers:
            max_inliers = np.sum(inlier_idx)
            best_inliers1 = matches1[inlier_idx]
            best_inliers2 = matches2[inlier_idx]
            inlier_idxs = np.where(inlier_idx)[0]
            
        # update globally scoped vars
        inlier_counts += [np.sum(inlier_idx)]
        inlier_residuals += [res]

    # recalculate fundamental matrix
    best_Fmatrix, best_inlier_residual = estimate_fundamental_matrix(best_inliers1, best_inliers2)

    return best_Fmatrix, best_inliers1, best_inliers2, best_inlier_residual, inlier_idxs


def triangulate(M1, M2, points1, points2):
    """
    Triangulate 3D points from two views using the Linear Triangulation Method (DLT).
    
    For each correspondence (p1, p2) between images 1 and 2, we solve:
        A @ X = 0
    where X = [X, Y, Z, 1]^T is the homogeneous 3D point.
    
    Build A as:
        [p1_x * M1[2,:] - M1[0,:]
         p1_y * M1[2,:] - M1[1,:]
         p2_x * M2[2,:] - M2[0,:]
         p2_y * M2[2,:] - M2[1,:]]
    
    Solve via SVD: X is the last row of V^T.
    
    :param M1: 3x4 projection matrix of camera 1
    :param M2: 3x4 projection matrix of camera 2
    :param points1: N x 2 array of 2D points in image 1
    :param points2: N x 2 array of 2D points in image 2
    :return: N x 3 array of triangulated 3D points
    """
    N = points1.shape[0]
    points_3d = np.zeros((N, 3))
    
    for i in range(N):
        # Build the 4x4 matrix A for this correspondence
        p1 = points1[i]
        p2 = points2[i]
        
        A = np.array([
            p1[0] * M1[2, :] - M1[0, :],
            p1[1] * M1[2, :] - M1[1, :],
            p2[0] * M2[2, :] - M2[0, :],
            p2[1] * M2[2, :] - M2[1, :]
        ])
        
        # Solve via SVD
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]  # Last row of V^T
        
        # Convert from homogeneous to Euclidean coordinates
        points_3d[i] = X[:3] / X[3]
    
    return points_3d


def pnp_ransac(object_points, image_points, K, num_iters=100, reprojection_error_threshold=8.0):
    """
    Estimate camera pose (R, t) from 3D-to-2D correspondences using PnP + RANSAC.
    
    Uses cv2.solvePnPRansac which:
      1. Randomly samples 4 correspondences
      2. Solves PnP for camera pose
      3. Reprojects all points and counts inliers
      4. Keeps the pose with most inliers
    
    :param object_points: N x 3 array of 3D world points
    :param image_points: N x 2 array of 2D image points
    :param K: 3x3 camera intrinsic matrix
    :param num_iters: number of RANSAC iterations
    :param reprojection_error_threshold: max reprojection error in pixels for inlier
    :return: R (3x3 rotation), t (3x1 translation), inlier_indices (M,)
    """
    # Use OpenCV's PnP RANSAC solver
    success, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        iterationsCount=num_iters,
        reprojectionError=reprojection_error_threshold,
        confidence=0.99
    )
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    
    if inlier_indices is not None:
        inlier_indices = inlier_indices.flatten()
    else:
        inlier_indices = np.arange(len(object_points))
    
    return R, t, inlier_indices


def build_camera_matrix(K, R, t):
    """
    Build a 3x4 projection matrix from K, R, t.
    
    M = K @ [R | t]
    
    :param K: 3x3 intrinsic matrix
    :param R: 3x3 rotation matrix
    :param t: 3x1 or 3, translation vector
    :return: 3x4 projection matrix
    """
    if t.shape == (3,):
        t = t.reshape(3, 1)
    Rt = np.hstack((R, t))
    return K @ Rt


def bundle_adjustment(cameras, points_3d, tracks, image_points_list, K, max_iters=10):
    """
    Refine camera poses and 3D points using bundle adjustment via Levenberg-Marquardt.
    
    Uses scipy.optimize.least_squares to minimize reprojection error:
        min sum_i || p_i - project(M_i, X) ||^2
    
    :param cameras: list of (R, t) tuples, one per camera
    :param points_3d: M x 3 array of 3D points to optimize
    :param tracks: dict mapping point_idx -> [(camera_idx, keypoint_idx), ...]
    :param image_points_list: list of dicts mapping keypoint_idx -> 2D image coordinates for each camera
    :param K: 3x3 camera intrinsic matrix
    :param max_iters: max iterations for optimization
    :return: refined_cameras, refined_points_3d
    """
    from scipy.optimize import least_squares
    
    num_cameras = len(cameras)
    num_points = points_3d.shape[0]
    
    params = []
    
    for R, t in cameras:
        rvec, _ = cv2.Rodrigues(R)
        params.extend(rvec.flatten())
        params.extend(t.flatten())
    
    params.extend(points_3d.flatten())
    params = np.array(params)
    
    def residuals(params):
        """Compute reprojection errors for all tracks."""
        errors = []
        
        # Unpack cameras
        idx = 0
        cameras_unpacked = []
        for i in range(num_cameras):
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            R, _ = cv2.Rodrigues(rvec)
            cameras_unpacked.append((R, tvec))
            idx += 6
        
        # Unpack 3D points
        points_3d_unpacked = params[idx:].reshape(-1, 3)
        
        # Compute reprojection errors for each track
        for point_idx, observations in tracks.items():
            point_3d = points_3d_unpacked[point_idx]
            
            for cam_idx, kp_idx in observations:
                R, t = cameras_unpacked[cam_idx]
                M = build_camera_matrix(K, R, t.reshape(3, 1))
                
                # Project 3D point
                proj_2d = project(M, point_3d.reshape(1, 3))[0]
                
                # Observed 2D point
                obs_2d = image_points_list[cam_idx][kp_idx]
                
                # Reprojection error
                error = proj_2d - obs_2d
                errors.extend(error)
        
        return np.array(errors)
    
    # Run least squares optimization
    result = least_squares(residuals, params, max_nfev=max_iters * 100, verbose=0)
    optimized_params = result.x
    
    # Unpack refined cameras
    idx = 0
    refined_cameras = []
    for i in range(num_cameras):
        rvec = optimized_params[idx:idx+3]
        tvec = optimized_params[idx+3:idx+6]
        R, _ = cv2.Rodrigues(rvec)
        refined_cameras.append((R, tvec.reshape(3, 1)))
        idx += 6
    
    # Unpack refined 3D points
    refined_points_3d = optimized_params[idx:].reshape(-1, 3)
    
    return refined_cameras, refined_points_3d


# Task 5: Uncalibrated Stereo Disparity

def compute_disparity_map(rect_left_gray, rect_right_gray, win_size=11,
                          max_disparity=64):
    """
    Compute a disparity map from a rectified stereo pair using NCC.
    The stencil code will perform the rectification once F is computed.
    
    After epipolar rectification the epipolar lines are horizontal, so the
    matching point in the right image is always on the same row, shifted
    by some disparity d. For each candidate d in (-max_disparity,
    +max_disparity), shift the right image by d pixels and compute
    per-pixel NCC against the left image using compute_ncc(). Track the
    disparity with the best (highest) NCC at each pixel.

    This is structurally almost identical to plane_sweep_stereo -- a loop over
    candidates, computing NCC at each one -- but here we shift the image
    horizontally instead of warping with a homography.

    :param rect_left_gray:  H x W float32, rectified left image (grayscale)
    :param rect_right_gray: H x W float32, rectified right image (grayscale)
    :param win_size:        NCC window size (default 11)
    :param max_disparity:   search range is (-max_disparity, +max_disparity)
    :return: H x W float32 disparity map (NaN where no valid match)
    """

    W = rect_left_gray.shape[1]
    H = rect_left_gray.shape[0]

    # default disparity map to nan
    disparity_map = np.full(shape = (H, W), fill_value = np.nan)
    best_nccs = np.full(shape = (H, W), fill_value = -1.0)

    # iterate through ds
    for d in range(-max_disparity, max_disparity+1):
        # shift image
        shifted = np.roll(rect_right_gray, d, axis=1)

        # compute ncc
        ncc = compute_ncc(rect_left_gray, shifted, win_size=win_size)

        # iterate through pixels
        for row in range(H):
            for col in range(W):
                # check for max ncc
                curr_ncc = ncc[row, col]
                if curr_ncc > 0.2 and curr_ncc > best_nccs[row, col]:
                    best_nccs[row, col] = curr_ncc
                    disparity_map[row, col] = d    

    return disparity_map


###############################################################################
# EXTRA CREDIT SECTION

# Below this line are functions for extra credit 
# that we can evaluate in the autograder.

###############################################################################
# Extra Credit: Coordinate Normalization

def normalize_coordinates(points):
    """
    EXTRA CREDIT: Hartley normalization — zero mean, average distance sqrt(D)
    from the centroid, where D is the dimensionality (2 or 3).

    This improves numerical conditioning of any DLT-style estimation
    (camera matrix M via estimate_camera_matrix, or fundamental matrix F
    via estimate_fundamental_matrix).

    Build T = T_scale @ T_offset where:
      T_offset translates the centroid to the origin
      T_scale  scales so the average distance from the origin is sqrt(D)
              (sqrt(2) for 2D points, sqrt(3) for 3D points)
      s = sqrt(D) / mean_distance, where
          mean_distance = mean(||p_i - centroid||)

    For 2D input (N x 2): returns (normalized_points [N x 3], T [3x3])
    For 3D input (N x 3): returns (normalized_points [N x 4], T [4x4])

    :param points: N x D array of points (D = 2 or 3)
    :return: (normalized_points, T)
    """

    # find centroid - one value per dimension
    centroid = np.mean(points, axis=0)

    # get T_offset by subtracting all values 
    points_offset = points.copy()
    for index, coord in enumerate(centroid):
        points_offset[:, index] = points_offset[:, index] - coord

    # find average distance from origin
    # because points are offset from origin, their distance is sqrt(point**2)
    mean_distance = np.mean(np.sqrt(np.sum(points_offset**2, axis=1)))
    s = np.sqrt(len(centroid)) / mean_distance

    if len(centroid) == 2:
        T_offset = np.array([
            [1, 0, -centroid[0]],
            [0, 1, -centroid[1]],
            [0, 0, 1]
        ])
        T_scale = np.array([
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1]
        ])
    else:
        T_offset = np.array([
            [1, 0, 0, -centroid[0]],
            [0, 1, 0, -centroid[1]],
            [0, 0, 1, -centroid[2]],
            [0, 0, 0, 1]
        ])
        T_scale = np.array([
            [s, 0, 0, 0],
            [0, s, 0, 0],
            [0, 0, s, 0],
            [0, 0, 0, 1]
        ])

    T = T_scale@T_offset
    ones = np.ones((points.shape[0], 1))
    homogenous_coordinates = np.hstack((points,ones))
    normalized_coordinates = (T @ homogenous_coordinates.T).T

    # make normalized coords not homogenous
    # divide by and remove last column
    divide = normalized_coordinates[:, -1].reshape(-1, 1)

    normalized_coordinates = (normalized_coordinates[:, :-1] / divide)

    return (normalized_coordinates, T)


# Extra Credit: Sampson Distance

def compute_sampson_distance(F, points1, points2):
    """
    EXTRA CREDIT: Per-point Sampson distance — the first-order approximation
    to geometric reprojection error for the fundamental matrix.

    Steps:
      1. Build homogeneous coordinates: x = [u, v, 1]^T, x' = [u', v', 1]^T
      2. Compute Fx, F^T x'
      3. Compute numerator: (x'^T F x)^2
      4. Compute denominator: (Fx)[0]^2 + (Fx)[1]^2 + (F^T x')[0]^2 + (F^T x')[1]^2
      5. Return numerator / denominator

    If we implement this, we can use it as the distance metric in RANSAC
    instead of the algebraic error |x'^T F x|, for a more principled
    inlier/outlier threshold in pixel^2 units rather than algebraic error units.

    :param F: 3x3 fundamental matrix
    :param points1: N x 2 array of 2D points in image 1
    :param points2: N x 2 array of 2D points in image 2
    :return: length-N array of Sampson distances
    """
    # make homogenous coords
    ones = np.ones((points1.shape[0], 1))
    points1 = np.hstack((points1, ones))
    points2 = np.hstack((points2, ones))

    distances = []
    for point1, point2 in zip(points1, points2):
        Fx1 = F@point1
        Fx2 = F.T@point2

        numerator = (point2.T @ F @ point1) ** 2
        denominator = Fx1[0] ** 2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2

        distances += [numerator/denominator]

    return distances

# Extra Credit: Decompose M into K[R|t]

def decompose_projection_matrix(M):
    """
    EXTRA CREDIT: Factor M into K, R, t via RQ decomposition of M[:,:3].

    Use cv2.RQDecomp3x3(M[:,:3]), and normalize so that K[2,2] = 1.
    Compute t = K^{-1} @ M[:,3].

    :param M: 3x4 projection matrix
    :return: K (3x3 upper-triangular), R (3x3 rotation), t (length-3)
    """
    raise NotImplementedError("TODO (Extra Credit): implement decompose_projection_matrix")


# Extra Credit: Essential Matrix and Relative Pose

def estimate_relative_pose(F, K1, K2, inliers1, inliers2):
    """
    EXTRA CREDIT: From F and intrinsic matrices, compute E and decompose
    to (R, t).

    Steps:
      1. E = K2^T @ F @ K1
      2. Enforce essential matrix constraint: SVD, set singular values
         to [s, s, 0] where s = (S[0]+S[1])/2
      3. Decompose E into 4 candidate (R, t) pairs using the W matrix
      4. Cheirality check: pick the (R, t) where most points are in
         front of both cameras

    :param F: 3x3 fundamental matrix
    :param K1: 3x3 intrinsic matrix of camera 1
    :param K2: 3x3 intrinsic matrix of camera 2
    :param inliers1: N x 2 inlier points in image 1
    :param inliers2: N x 2 inlier points in image 2
    :return: R (3x3), t (length-3)
    """
    raise NotImplementedError("TODO (Extra Credit): implement estimate_relative_pose")


# Extra Credit: Oriented Plane Homography (toward multi-view stereo)

def compute_plane_homography(M_ref, M_other, n, d):
    """
    EXTRA CREDIT: Compute the homography induced by a general oriented plane.
    We can then use this in our plane sweep algorithm. But, beware - we now
    have to search over a larger space of parameters, which will take a long time.

        s = d - n^T C_ref
        m = A_ref^{-T} n      (hint: use np.linalg.solve(A_ref.T, n))
        H(n, d) = s * B + np.outer(a, m)

    where B = A_other @ A_ref^{-1} and a = A_other @ C_ref + t_other.

    When the plane is fronto-parallel (n parallel to A_ref^T @ e3),
    this reduces to the H(lam) from compute_depth_homography().

    :param M_ref:   3x4 projection matrix of the reference camera
    :param M_other: 3x4 projection matrix of the other camera
    :param n:       length-3 array, plane normal (world coordinates)
    :param d:       scalar, plane offset (n^T P = d)
    :return: H, a 3x3 numpy array
    """
    raise NotImplementedError("TODO (Extra Credit): implement compute_plane_homography")


# /////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    """Two-panel RANSAC diagnostic:
    1. Inlier count vs. iteration
    2. Residual vs. iteration
    """
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title("RANSAC Convergence")

    plt.subplot(2, 1, 1)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()

    plt.tight_layout()
    plt.show()
