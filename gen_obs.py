from rlbench.backend.observation import Observation
import numpy as np

top_camera_extrinsics    = [[ 1,    0,    0,    0.53],
                          [ 0,   -1,    0,    0   ],
                          [ 0,    0,   -1,    2.52],
                          [ 0,    0,    0,    1.  ]]
front_camera_extrinsics  = [[ 7.96326711e-04,  5.55111512e-17, -9.99999683e-01,  1.37000000e+00],
                           [-9.99999366e-01, -7.96326711e-04, -7.96326458e-04,  0.00000000e+00],
                           [-7.96326458e-04,  9.99999683e-01, -6.34136230e-07,  1.12000000e+00],
                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
side_1_camera_extrinsics = [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  5.30000000e-01],
                            [ 0.00000000e+00, -7.96326711e-04, -9.99999683e-01,  1.50000000e+00],
                            [ 0.00000000e+00,  9.99999683e-01, -7.96326711e-04,  1.12000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
side_2_camera_extrinsics = [[-9.99998732e-01,  1.59265292e-03,  2.16840434e-19,  5.30000000e-01],
                            [ 1.26827206e-06,  7.96325701e-04,  9.99999683e-01, -1.50000000e+00],
                            [ 1.59265241e-03,  9.99998415e-01, -7.96326711e-04,  1.12000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

top_camera_intrinsics = [[320.95457927,   0,         212],
                         [  0,         320.95457927, 120],
                         [  0,           0,            1]]
front_camera_intrinsics = [[320.95457927,   0,         212],
                           [  0,         320.95457927, 120],
                           [  0,           0,            1]]
side_1_camera_intrinsics = [[320.95457927,   0,         212],
                            [  0,         320.95457927, 120],
                            [  0,           0,            1]]
side_2_camera_intrinsics = [[320.95457927,   0,         212],
                            [  0,         320.95457927, 120],
                            [  0,           0,            1]]

top_camera_extrinsics = np.array(top_camera_extrinsics)
front_camera_extrinsics = np.array(front_camera_extrinsics)
side_1_camera_extrinsics = np.array(side_1_camera_extrinsics)
side_2_camera_extrinsics = np.array(side_2_camera_extrinsics)
top_camera_intrinsics = np.array(top_camera_intrinsics)
front_camera_intrinsics = np.array(front_camera_intrinsics)
side_1_camera_intrinsics = np.array(side_1_camera_intrinsics)
side_2_camera_intrinsics = np.array(side_2_camera_intrinsics)

camera_near = 0.16
camera_far = 2
obs = []
misc = {"top_camera_extrinsics": top_camera_extrinsics,
        "top_camera_intrinsics": top_camera_intrinsics,
        "front_camera_extrinsics": front_camera_extrinsics,
        "front_camera_intrinsics": front_camera_intrinsics,
        "side_1_camera_extrinsics": side_1_camera_extrinsics,
        "side_1_camera_intrinsics": side_1_camera_intrinsics,
        "side_2_camera_extrinsics": side_2_camera_extrinsics,
        "side_2_camera_intrinsics": side_2_camera_intrinsics,
        "top_camera_near": camera_near,
        "top_camera_far": camera_far,
        "front_camera_near": camera_near,
        "front_camera_far": camera_far,
        "side_1_camera_near": camera_near,
        "side_1_camera_far": camera_far,
        "side_2_camera_near": camera_near,
        "side_2_camera_far": camera_far
}
for n in range(4):
    obs.append(Observation(misc=misc))

# save the observations to a pickle file
import pickle
with open("data/episode0/obs.pkl", "wb") as f:
    pickle.dump(obs, f)