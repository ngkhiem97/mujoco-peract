import numpy as np
import cv2
from scipy import stats
from PIL import Image

CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128

def get_camera_intrinsic_matrix(sim, camera_name):
    """
    Obtains camera intrinsic matrix.
    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * CAMERA_HEIGHT / np.tan(fovy * np.pi / 360)
    K = np.array([[-f, 0, CAMERA_WIDTH / 2], [0, -f, CAMERA_HEIGHT / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    
    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array([
        [1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., -1., 0.],
        [0., 0., 0., 1.]]
    )
    R = R @ camera_axis_correction
    return R

def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.
    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation
    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def clip_float_values(float_array, min_value, max_value):
    """Clips values to the range [min_value, max_value].

    First checks if any values are out of range and prints a message.
    Then clips all values to the given range.

    Args:
        float_array: 2D array of floating point values to be clipped.
        min_value: Minimum value of clip range.
        max_value: Maximum value of clip range.

    Returns:
        The clipped array.

    """
    if float_array.min() < min_value or float_array.max() > max_value:
        float_array = np.clip(float_array, min_value, max_value)
    return float_array

DEFAULT_RGB_SCALE_FACTOR = 256000.0

def float_array_to_rgb_image(float_array,
                             scale_factor=DEFAULT_RGB_SCALE_FACTOR,
                             drop_blue=False):
    """Convert a floating point array of values to an RGB image.

    Convert floating point values to a fixed point representation where
    the RGB bytes represent a 24-bit integer.
    R is the high order byte.
    B is the low order byte.
    The precision of the depth image is 1/256 mm.

    Floating point values are scaled so that the integer values cover
    the representable range of depths.

    This image representation should only use lossless compression.

    Args:
        float_array: Input array of floating point depth values in meters.
        scale_factor: Scale value applied to all float values.
        drop_blue: Zero out the blue channel to improve compression, results in 1mm
        precision depth values.

    Returns:
        24-bit RGB PIL Image object representing depth values.
    """
    # Scale the floating point array.
    scaled_array = np.floor(float_array * scale_factor + 0.5)
    # Convert the array to integer type and clip to representable range.
    min_inttype = 0
    max_inttype = 2**24 - 1
    scaled_array = clip_float_values(scaled_array, min_inttype, max_inttype)
    int_array = scaled_array.astype(np.uint32)
    # Calculate:
    #   r = (f / 256) / 256  high byte
    #   g = (f / 256) % 256  middle byte
    #   b = f % 256          low byte
    rg = np.divide(int_array, 256)
    r = np.divide(rg, 256)
    g = np.mod(rg, 256)
    image_shape = int_array.shape
    rgb_array = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    rgb_array[..., 0] = r
    rgb_array[..., 1] = g
    if not drop_blue:
        # Calculate the blue channel and add it to the array.
        b = np.mod(int_array, 256)
        rgb_array[..., 2] = b
    image_mode = 'RGB'
    image = Image.fromarray(rgb_array, mode=image_mode)
    return image

def process_depth_img(model, depth_img):
    znear = 0.01
    zfar = 50.0
    div_near = 1/(znear*model.stat.extent)
    div_far = 1/(zfar*model.stat.extent)
    s = div_far-div_near
    depth_img = 1/(s*depth_img + div_near)
    # limit measurement range
    dplim_upper = 2
    dplim_lower = 0.16
    depth_img[depth_img<=dplim_lower]=dplim_lower
    depth_img[depth_img>=dplim_upper]=dplim_upper
    # add noise
    image_noise_1=stats.distributions.norm.rvs(0,0.00005,size=depth_img.shape)
    image_noise_2=np.random.normal(0,0.00015,size=depth_img.shape)
    depth_img = depth_img + image_noise_1 + image_noise_2
    depth_img = cv2.normalize(depth_img, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)      
    # convert to rgb presentation
    depth_img = float_array_to_rgb_image(depth_img, 2**24 - 1)
    return depth_img