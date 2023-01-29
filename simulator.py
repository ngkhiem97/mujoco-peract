import mujoco_py
from mujoco_py import MjSim, MjViewer, MjRenderContext
import numpy as np
from collections import deque
from VelocityController import *
import threading
import cv2
import datetime
from scipy import stats
from PIL import Image

ACTION_SPACE = 3
TWIST_SPACE = 6
ROBOT_INIT_POS = [-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335]
INIT_VEL = [0, 0, 0, 0, 0, 0]
DOF = 7
INC_POS_VEL = 0.15
INC_ANG_VEL = 15/180*np.pi

CAMERAS = ["top", "front", "side-1", "side-2"]
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128

DEFAULT_RGB_SCALE_FACTOR = 256000.0
          
'''
convert a unit quaternion to angle/axis representation
'''                                                                                                            
def quat2axang(q): 
    s = np.linalg.norm(q[1:4])
    if s >= 10*np.finfo(float).eps:#10*np.finfo(q.dtype).eps:
        vector = q[1:4]/s
        theta = 2*np.arctan2(s,q[0])
    else:
        vector = np.array([0,0,1])
        theta = 0
    
    axang = np.hstack((vector,theta))
    return axang
    
    
'''
multiply two quaternions (numpy arrays)
'''
def quatmultiply(q1, q2):

    # scalar = s1*s2 - dot(v1,v2)
    scalar = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]

    # vector = s1*v2 + s2*v1 + cross(v1,v2)
    vector = np.array([q1[0]*q2[1], q1[0]*q2[2], q1[0]*q2[3]]) + \
             np.array([q2[0]*q1[1], q2[0]*q1[2], q2[0]*q1[3]]) + \
             np.array([ q1[2]*q2[3]-q1[3]*q2[2], \
                        q1[3]*q2[1]-q1[1]*q2[3], \
                        q1[1]*q2[2]-q1[2]*q2[1]])

    rslt = np.hstack((scalar, vector))
    return rslt

class Simulator:
    def __init__(self, model_path):
        # simulation variables
        with open(model_path, 'r') as f:
            self.model = mujoco_py.load_model_from_xml(f.read())
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.offscreen = MjRenderContext(self.sim, 0, quiet = True)
        # controller variables
        self.velocity_ctrl = VelocityController()
        self.queue_states = deque(maxlen=10)
        self.queue_cameras = deque(maxlen=10)
        self.lock_states = threading.Lock()
        self.lock_cameras = threading.Lock()
        self.lock_main = threading.Lock()
        self.action = np.zeros(ACTION_SPACE)
        self.twist = np.zeros(TWIST_SPACE)
        self.sim.data.qpos[0:DOF] = ROBOT_INIT_POS
        self.velocity = np.array(INIT_VEL)
        self.gripper = float(self.sim.data.ctrl[DOF])
        self.v_tgt = np.zeros(DOF)
        self.is_save = False
        # viewer variables
        self.view_thread = threading.Thread(target=self.view, args=())
        self.view_thread.start()

    def view(self):
        while True:
            cmd_received = False
            velocity = self.velocity
            gripper = self.gripper
            keypressed = cv2.waitKey(1)
            if keypressed == 27: # Esc key to stop
                break
            elif keypressed == ord('\\'):#\, up
                cmd_received = True
                velocity = velocity + np.array([0,0,INC_POS_VEL,0,0,0])
            elif keypressed == 13:       #return, down
                cmd_received = True
                velocity = velocity + np.array([0,0,-INC_POS_VEL,0,0,0])
            elif keypressed == ord('i'): # forward
                cmd_received = True
                velocity = velocity + np.array([INC_POS_VEL,0,0,0,0,0])
            elif keypressed == ord('k'): # backward
                cmd_received = True
                velocity = velocity + np.array([-INC_POS_VEL,0,0,0,0,0])
            elif keypressed == ord('j'): # left
                cmd_received = True
                velocity = velocity + np.array([0,INC_POS_VEL,0,0,0,0])
            elif keypressed == ord('l'): # right
                cmd_received = True
                velocity = velocity + np.array([0,-INC_POS_VEL,0,0,0,0])
            elif keypressed == ord('q'): # roll
                cmd_received = True
                ang_vel = velocity[3:6]
                ang_vel = quatmultiply(np.array([np.cos(INC_ANG_VEL/2),np.sin(INC_ANG_VEL/2),0,0]),ang_vel)
                axang = quat2axang(ang_vel)
                axang = axang[3]*axang[0:3]
                velocity = np.array([velocity[0],velocity[1],velocity[2],axang[0],axang[1],axang[2]])
            elif keypressed == ord('a'): # -roll
                cmd_received = True
                ang_vel = velocity[3:6]
                ang_vel = quatmultiply(np.array([np.cos(-INC_ANG_VEL/2),np.sin(-INC_ANG_VEL/2),0,0]),ang_vel)
                axang = quat2axang(ang_vel)
                axang = axang[3]*axang[0:3]
                velocity = np.array([velocity[0],velocity[1],velocity[2],axang[0],axang[1],axang[2]])
            elif keypressed == ord('w'): # pitch
                cmd_received = True
                ang_vel = velocity[3:6]
                ang_vel = quatmultiply(np.array([np.cos(INC_ANG_VEL/2),0,np.sin(INC_ANG_VEL/2),0]),ang_vel)
                axang = quat2axang(ang_vel)
                axang = axang[3]*axang[0:3]
                velocity = np.array([velocity[0],velocity[1],velocity[2],axang[0],axang[1],axang[2]])
            elif keypressed == ord('s'): # -pitch
                cmd_received = True
                ang_vel = velocity[3:6]
                ang_vel = quatmultiply(np.array([np.cos(-INC_ANG_VEL/2),0,np.sin(-INC_ANG_VEL/2),0]),ang_vel)
                axang = quat2axang(ang_vel)
                axang = axang[3]*axang[0:3]
                velocity = np.array([velocity[0],velocity[1],velocity[2],axang[0],axang[1],axang[2]])
            elif keypressed == ord('e'): # yaw
                cmd_received = True
                ang_vel = velocity[3:6]
                ang_vel = quatmultiply(np.array([np.cos(INC_ANG_VEL/2),0,0,np.sin(INC_ANG_VEL/2)]),ang_vel)
                axang = quat2axang(ang_vel)
                axang = axang[3]*axang[0:3]
                velocity = np.array([velocity[0],velocity[1],velocity[2],axang[0],axang[1],axang[2]])
            elif keypressed == ord('d'): # -yaw
                cmd_received = True
                ang_vel = velocity[3:6]
                ang_vel = quatmultiply(np.array([np.cos(-INC_ANG_VEL/2),0,0,np.sin(-INC_ANG_VEL/2)]),ang_vel)
                axang = quat2axang(ang_vel)
                axang = axang[3]*axang[0:3]   
                velocity = np.array([velocity[0],velocity[1],velocity[2],axang[0],axang[1],axang[2]])
            elif keypressed == ord('c'): # gripper close
                cmd_received = True
                gripper += 0.1
                gripper = np.clip(gripper, 0, 1.5)
            elif keypressed == ord('v'): # gripper open
                cmd_received = True
                gripper -= 0.1
                gripper = np.clip(gripper, 0, 1.5)
            elif keypressed == ord('p'): # pause robot
                cmd_received = True
                velocity = np.array(INIT_VEL)
            elif keypressed == ord('y'): # save the simulator model
                self.save()
            if cmd_received:
                self.velocity = velocity
                self.gripper = gripper
            self.push_states()
            self.update_cameras()
        cv2.destroyAllWindows()

    def get_camera_intrinsic_matrix(self, camera_name):
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
        cam_id = self.sim.model.camera_name2id(camera_name)
        fovy = self.sim.model.cam_fovy[cam_id]
        f = 0.5 * CAMERA_HEIGHT / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, CAMERA_WIDTH / 2], [0, f, CAMERA_HEIGHT / 2], [0, 0, 1]])
        return K


    def get_camera_extrinsic_matrix(self, camera_name):
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
        cam_id = self.sim.model.camera_name2id(camera_name)
        camera_pos = self.sim.data.cam_xpos[cam_id]
        camera_rot = self.sim.data.cam_xmat[cam_id].reshape(3, 3)
        R = self.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.]]
        )
        R = R @ camera_axis_correction
        return R

    def make_pose(self, translation, rotation):
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

    def save(self):
        time = datetime.datetime.now()
        # save the simulator model
        filename = 'saves/models/robot_'+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.xml'
        file = open(filename, 'w')
        self.sim.save(file, 'xml')
        # save the camera images and matrices
        self.is_save = True
            
    def update_cameras(self):
        if len(self.queue_cameras) == 0:
            return
        self.lock_cameras.acquire()
        rgb_imgs, depth_imgs = self.queue_cameras.popleft()
        self.lock_cameras.release()
        for title, camera_id in zip(CAMERAS, [0, 1, 2, 3]):
            cv2.imshow(title+"_rgb", rgb_imgs[camera_id])
            # cv2.imshow(title+"_depth", depth_imgs[camera_id])
        if self.is_save == False:
            return
        time = datetime.datetime.now()
        for camera_id in range(4):
            cv2.imwrite('saves/cameras/'+CAMERAS[camera_id].replace("-", "_")+"/images/rgb_"+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.png', rgb_imgs[camera_id])
            depth_imgs[camera_id].save('saves/cameras/'+CAMERAS[camera_id].replace("-", "_")+"/images/depth_"+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.png')
            np.save('saves/cameras/'+CAMERAS[camera_id].replace("-", "_")+"/matrices/ex_"+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.npy', self.get_camera_extrinsic_matrix("realsense-"+CAMERAS[camera_id]))
            np.save('saves/cameras/'+CAMERAS[camera_id].replace("-", "_")+"/matrices/in_"+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.npy', self.get_camera_intrinsic_matrix("realsense-"+CAMERAS[camera_id]))
        self.is_save = False

    
    def push_states(self):
        self.lock_states.acquire()
        joint_vel = self.velocity_ctrl.get_joint_vel_worldframe(self.velocity, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
        self.queue_states.append([joint_vel, self.gripper])
        self.lock_states.release()

    def update_states(self):
        if len(self.queue_states) == 0:
            return
        self.lock_states.acquire()
        joint_vel, gripper = self.queue_states.popleft()
        self.v_tgt[0:DOF] = np.squeeze(joint_vel[0:DOF])
        self.sim.data.ctrl[DOF:DOF+2] = [gripper, gripper]
        self.lock_states.release()

    def push_cameras(self):
        rgb_imgs = []
        depth_imgs = []
        for camera_id in range(4):
            self.offscreen.render(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, camera_id=camera_id)
            pixels = self.offscreen.read_pixels(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, depth=True)
            rgb_img, depth_img = pixels[0:2]
            rgb_img = rgb_img[:, ::-1, ::-1]
            depth_img = self.process_depth_img(depth_img[:, ::-1])
            rgb_imgs.append(rgb_img)
            depth_imgs.append(depth_img)
        self.lock_cameras.acquire()
        self.queue_cameras.append([rgb_imgs, depth_imgs])
        self.lock_cameras.release()

    def clip_float_values(self, float_array, min_value, max_value):
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

    def float_array_to_rgb_image(self, float_array,
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
        scaled_array = self.clip_float_values(scaled_array, min_inttype, max_inttype)
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

    def process_depth_img(self, depth_img):
        znear = 0.01
        zfar = 50.0
        div_near = 1/(znear*self.model.stat.extent)
        div_far = 1/(zfar*self.model.stat.extent)
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
        depth_img = self.float_array_to_rgb_image(depth_img, 2**24 - 1)
        return depth_img

    def start(self):
        ct = 0       
        while True:  
            self.sim.data.qfrc_applied[0:DOF] = self.sim.data.qfrc_bias[0:DOF]
            self.sim.data.qvel[0:DOF] = self.v_tgt[0:DOF]
            if (ct*self.sim.model.opt.timestep/0.01).is_integer(): # update the target velocity every 0.01 seconds
                self.update_states()
            ct = ct + 1                
            self.sim.step()
            self.viewer.render()
            if ct%17 == 1:
                self.push_cameras()

if __name__ == "__main__":
    sim = Simulator("robot_2023_1_17_12_17_30.xml")
    sim.start()