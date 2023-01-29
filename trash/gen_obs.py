from rlbench.backend.observation import Observation
import numpy as np

front_camera_extrinsics    = [[ 1,    0,    0,    0.13],
                          [ 0,   -1,    0,    0   ],
                          [ 0,    0,   -1,    2.1],
                          [ 0,    0,    0,    1.  ]]
left_shoulder_camera_extrinsics  = [[ 7.96326711e-04,  5.55111512e-17, -9.99999683e-01,  1.37000000e+00],
                           [-9.99999366e-01, -7.96326711e-04, -7.96326458e-04,  0.00000000e+00],
                           [-7.96326458e-04,  9.99999683e-01, -6.34136230e-07,  1.12000000e+00],
                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
right_shoulder_camera_extrinsics = [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.30000000e-01],
                            [ 0.00000000e+00, -7.96326711e-04, -9.99999683e-01,  1.50000000e+00],
                            [ 0.00000000e+00,  9.99999683e-01, -7.96326711e-04,  1.12000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
wrist_camera_extrinsics = [[-9.99998732e-01,  1.59265292e-03,  2.16840434e-19,  1.30000000e-01],
                            [ 1.26827206e-06,  7.96325701e-04,  9.99999683e-01, -1.50000000e+00],
                            [ 1.59265241e-03,  9.99998415e-01, -7.96326711e-04,  1.12000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

front_camera_intrinsics = [[171.17577561,   0,         212],
                         [  0,         171.17577561, 120],
                         [  0,           0,            1]]
left_shoulder_camera_intrinsics = [[171.17577561,   0,         212],
                           [  0,         171.17577561, 120],
                           [  0,           0,            1]]
right_shoulder_camera_intrinsics = [[171.17577561,   0,         212],
                            [  0,         171.17577561, 120],
                            [  0,           0,            1]]
wrist_camera_intrinsics = [[171.17577561,   0,         212],
                            [  0,         171.17577561, 120],
                            [  0,           0,            1]]

front_camera_extrinsics = np.array(front_camera_extrinsics)
left_shoulder_camera_extrinsics = np.array(left_shoulder_camera_extrinsics)
right_shoulder_camera_extrinsics = np.array(right_shoulder_camera_extrinsics)
wrist_camera_extrinsics = np.array(wrist_camera_extrinsics)
front_camera_intrinsics = np.array(front_camera_intrinsics)
left_shoulder_camera_intrinsics = np.array(left_shoulder_camera_intrinsics)
right_shoulder_camera_intrinsics = np.array(right_shoulder_camera_intrinsics)
wrist_camera_intrinsics = np.array(wrist_camera_intrinsics)

camera_near = 0.16
camera_far = 2
obs = []
misc = {"front_camera_extrinsics": front_camera_extrinsics,
        "front_camera_intrinsics": front_camera_intrinsics,
        "left_shoulder_camera_extrinsics": left_shoulder_camera_extrinsics,
        "left_shoulder_camera_intrinsics": left_shoulder_camera_intrinsics,
        "right_shoulder_camera_extrinsics": right_shoulder_camera_extrinsics,
        "right_shoulder_camera_intrinsics": right_shoulder_camera_intrinsics,
        "wrist_camera_extrinsics": wrist_camera_extrinsics,
        "wrist_camera_intrinsics": wrist_camera_intrinsics,
        "front_camera_near": camera_near,
        "front_camera_far": camera_far,
        "left_shoulder_camera_near": camera_near,
        "left_shoulder_camera_far": camera_far,
        "right_shoulder_camera_near": camera_near,
        "right_shoulder_camera_far": camera_far,
        "wrist_camera_near": camera_near,
        "wrist_camera_far": camera_far
}
for n in range(5):
    obs.append(Observation(misc=misc))

# save the observations to a pickle file
import pickle
with open("data/episode1/low_dim_obs.pkl", "wb") as f:
    pickle.dump(obs, f)