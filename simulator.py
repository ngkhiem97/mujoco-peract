import mujoco_py
from mujoco_py import MjSim, MjViewer, MjRenderContext
import numpy as np
from collections import deque
import VelocityController
import threading
import cv2
import datetime

ACTION_SPACE = 3
TWIST_SPACE = 6
ROBOT_INIT_POS = [-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335]
INIT_VEL = [0, 0, 0, 0, 0, 0]
DOF = 7
INC_POS_VEL = 0.15
INC_ANG_VEL = 15/180*np.pi
          
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

class Gen3Env:
    def __init__(self, model_path):
        # simulation variables
        with open(model_path, 'r') as f:
            self.model = mujoco_py.load_model_from_xml(f.read())
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.offscreen = MjRenderContext(self.sim, 0, quiet = True)

        # controller variables
        self.velocity_ctrl = VelocityController()
        self.queue = deque(maxlen=10)
        self.queue_img = deque(maxlen=10)
        self.action = np.zeros(ACTION_SPACE)
        self.twist = np.zeros(TWIST_SPACE)
        self.set_robot_pos(ROBOT_INIT_POS)
        self.velocity = np.array(INIT_VEL)
        self.gripper = float(self.sim.data.ctrl[DOF])

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
                time = datetime.datetime.now()
                filename = 'saves/robot_'+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.xml'
                file = open(filename, 'w')
                self.sim.save(file, 'xml')
                print("saved the simulator model")
            if cmd_received:
                self.velocity = velocity
                self.gripper = gripper
            self.lock_ctrl.acquire()
            self.sim.data.ctrl[self.nv:self.nv+1] = [gripper, gripper] 
            joint_vel = self.velocity_ctrl.get_joint_vel_worldframe(self.velocity, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(joint_vel)
            self.lock.release()

            self.show_color_image("top", self.queue_img)
            self.show_color_image("front", self.queue_img_front)
            self.show_color_image("side_1", self.queue_img_side1)
            self.show_color_image("side_2", self.queue_img_side2)
        cv2.destroyAllWindows()

    def set_robot_pos(self, pos):
        ''' set the initial position of the robot '''
        self.sim.data.qpos[0:DOF] = pos