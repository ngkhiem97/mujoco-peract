import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContext
from mujoco_py import cymj
from mujoco_py.utils import remove_empty_lines
from mujoco_py.builder import build_callback_fn
import numpy as np
from collections import deque
import socket
import threading
import json
import xml.etree.ElementTree as ET
import os
import cv2
from scipy import stats
from pyhull.convex_hull import ConvexHull
from scipy.spatial import ConvexHull as ConvexHull2D
from stl.mesh import Mesh
from VelocityController import *
import time
import argparse

import datetime
from mujoco_camera_util import *

class SimulatorVelCtrl: #a communication wrapper for MuJoCo
                   
    def init(self, modelFile, nv, action):
        self.lock = threading.Lock()
        self.lock1 = threading.Lock()
        self.lockcv2 = threading.Lock()
        self.lockcv2Front = threading.Lock()
        self.lockcv2Side1 = threading.Lock()
        self.lockcv2Side2 = threading.Lock()
        self.velocityCtrl = VelocityController()
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(modelFile, parser=parser)
        root = tree.getroot()
        print(root.tag)
        for child in root:
            if child.tag == "mujoco":
                worldbody = child.find('worldbody')
                folder = '/tmp/mujoco_objects/'
                if not os.path.exists(folder):
                    os.makedirs(folder) 
                obj1 = ET.SubElement(worldbody, "body")
                pos_table = np.array([0.475, 0.0, 0.8])
                size_table = np.array([0.6,0.3,0.015])            
                self.generate_obj_table(obj1,pos_table,size_table) 
                obj3 = ET.SubElement(worldbody, "body")
                self.generate_obj_box(obj3, pos_table)
                self.modelStr = ET.tostring(child, encoding='utf8', method='xml').decode("utf-8") 
                with open(os.path.join(folder + 'model' + '.xml'), 'w') as f:
                    f.write(self.modelStr)
            elif child.tag == "learning":
                for node in child:
                    if node.tag == "init": 
                        statesStr = node.get('qpos')                       
                        self.initQpos = [float(n) for n in statesStr.split(",")]
                        gpCtrlStr = node.get('gpCtrl')
                        self.initGripper = float(gpCtrlStr)
                    else:
                        continue
        self.model = mujoco_py.load_model_from_xml(self.modelStr)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.offscreen = MjRenderContext(self.sim, 0, quiet = True)
        self.goal = self.sim.data.get_site_xpos('target0')
        self.nv = nv
        self.v_tgt = np.zeros(self.nv)
        self.queue = deque(maxlen=10)
        self.queue_img = deque(maxlen=10)
        self.queue_img_front = deque(maxlen=10)
        self.queue_img_side1 = deque(maxlen=10)
        self.queue_img_side2 = deque(maxlen=10)
        self.action = action
        self.thViewer = threading.Thread(target=self.cv2_viewer, args=())
        self.thViewer.start()
        
    def generate_obj_box(self, obj, pos_table):
        pos_box = np.array([0.45, 0.0, pos_table[2]+0.015+0.025])
        obj.set('name', 'ConSurf')                
        obj.set('pos', '{} {} {}'.format(pos_box[0], pos_box[1], pos_box[2]))# 0.035))
        geom = ET.SubElement(obj, "geom")
        geom.set('name', 'ConSurf') 
        geom.set('type','box')
        geom.set('size','0.035 0.015 0.025')
        geom.set('rgba','0.999 0.999 0.999 1')    
        geom.set('friction','0.45 0.005 0.0001')
        geom.set('solimp','0.95 0.95 0.01')
        geom.set('solref','0.01 1')
        geom.set('condim','3')
        geom.set('priority','2')
        geom.set('margin','0.0')
        
        
        joint_x = ET.SubElement(obj, "joint")
        joint_x.set('name','object0:joint_x')
        joint_x.set('pos','0 0 0')
        joint_x.set('type','slide')        
        joint_x.set('axis','1 0 0')
        joint_x.set('damping','0.01')
        
        joint_y = ET.SubElement(obj, "joint")
        joint_y.set('name','object0:joint_y')
        joint_y.set('pos','0 0 0')
        joint_y.set('type','slide')        
        joint_y.set('axis','0 1 0')
        joint_y.set('damping','0.01')
        
        joint_z = ET.SubElement(obj, "joint")
        joint_z.set('name','object0:joint_z')
        joint_z.set('pos','0 0 0')
        joint_z.set('type','slide')        
        joint_z.set('axis','0 0 1')
        joint_z.set('damping','0.01')
        
        joint_r = ET.SubElement(obj, "joint")
        joint_r.set('name','object0:joint_r')
        joint_r.set('pos','0 0 0')
        joint_r.set('type','hinge')        
        joint_r.set('axis','1 0 0')
        joint_r.set('damping','0.01')
        
        joint_p = ET.SubElement(obj, "joint")
        joint_p.set('name','object0:joint_p')
        joint_p.set('pos','0 0 0')
        joint_p.set('type','hinge')        
        joint_p.set('axis','0 1 0')
        joint_p.set('damping','0.01')
        
        joint_yw = ET.SubElement(obj, "joint")
        joint_yw.set('name','object0:joint_yw')
        joint_yw.set('pos','0 0 0')
        joint_yw.set('type','hinge')        
        joint_yw.set('axis','0 0 1')
        joint_yw.set('damping','0.01')
        
        
        
        site = ET.SubElement(obj, "site")
        site.set('name', 'object0')
        site.set('size', '0.0001 0.0001')
        site.set('rgba', '0 1 0 1')
        site.set('type', 'cylinder')
        site.set('pos', '{} {} {}'.format(0,0,-0.025))
        
        return pos_box
    
    def generate_obj_table(self, obj, pos_table, size_table):
        #table top
        obj.set('name', 'Table')                
        
        obj.set('pos', '{} {} {}'.format(pos_table[0],pos_table[1],pos_table[2]))
        
        geom = ET.SubElement(obj, "geom")
        geom.set('size','{} {} {}'.format(size_table[0],size_table[1],size_table[2]))
        geom.set('type','box')
        geom.set('rgba','0.8 0.2 0.1 1')
        geom.set('friction','0.25 0.005 0.0001')
        geom.set('solimp','0.95 0.95 0.01')
        geom.set('solref','0.01 1')
        geom.set('condim','4')
        geom.set('margin','0.0')
        site = ET.SubElement(obj, "site")
        site.set('name', 'target0')
        site.set('size', '0.02 0.0001')
        site.set('rgba', '0 1 0 1')
        site.set('type', 'cylinder')
        site.set('pos', '{} {} {}'.format(-0.1,-0.25,size_table[2]))
        
        #table legs                
        geom1 = ET.SubElement(obj, "geom")
        geom1.set('size','{} {}'.format(0.0135,0.2))
        geom1.set('pos','{} {} {}'.format(-(size_table[0]-0.025), -(size_table[1]-0.025), -0.2))
        geom1.set('type','cylinder')
        geom1.set('rgba','0.8 0.2 0.1 1')
        
        geom2 = ET.SubElement(obj, "geom")
        geom2.set('size','{} {}'.format(0.0135,0.2))
        geom2.set('pos','{} {} {}'.format(-(size_table[0]-0.025), (size_table[1]-0.025), -0.2))
        geom2.set('type','cylinder')
        geom2.set('rgba','0.8 0.2 0.1 1')
        
        geom3 = ET.SubElement(obj, "geom")
        geom3.set('size','{} {}'.format(0.0135,0.2))
        geom3.set('pos','{} {} {}'.format((size_table[0]-0.025), -(size_table[1]-0.025), -0.2))
        geom3.set('type','cylinder')
        geom3.set('rgba','0.8 0.2 0.1 1')
        
        geom4 = ET.SubElement(obj, "geom")
        geom4.set('size','{} {}'.format(0.0135,0.2))
        geom4.set('pos','{} {} {}'.format((size_table[0]-0.025), (size_table[1]-0.025), -0.2))
        geom4.set('type','cylinder')
        geom4.set('rgba','0.8 0.2 0.1 1')
        
        '''
        generate spheres above the table
        '''                
        for i in range(100):
            tmp_pos = np.array([np.random.uniform(-size_table[0], size_table[0]), np.random.uniform(-size_table[1], size_table[1]), size_table[2]])
            geom = ET.SubElement(obj, "geom")
            geom.set('name','spheret{}'.format(i))
            geom.set('type','sphere')
            geom.set('size','{}'.format(np.random.uniform(0.00005,0.0002)))
            geom.set('pos','{} {} {}'.format(tmp_pos[0],tmp_pos[1],tmp_pos[2]))
            geom.set('rgba','1.0 1.0 1.0 1')
            geom.set('friction','0.25 0.005 0.0001')
            geom.set('solimp','0.95 0.95 0.01')
            geom.set('solref','0.01 1')
            geom.set('condim','4')
            geom.set('margin','0.0') 
            
    def generate_obj_lamina(self, obj, child, pos_table, folder):
        '''
        generate the mesh
        '''
        name = 'lamina'
            
        points = 0.1*np.array([np.random.rand(),np.random.rand(),0])
        for i in range(24):
            points = np.vstack((points,0.1*np.array([np.random.rand(),np.random.rand(),0])))
        
        Points2D = points[:,:-1]
        mins, maxs = np.min(Points2D, axis=0), np.max(Points2D, axis=0)
        Points2D = Points2D- (maxs + mins)/2
        
        points = np.vstack((points,points+np.array([0,0,0.01])))
        
        mins, maxs = np.min(points, axis=0), np.max(points, axis=0)
        points = points- (maxs + mins)/2
        
        #print(points)
        hull2D = ConvexHull2D(Points2D)
        hull = ConvexHull(points)
        #print(hull)
        shape = Mesh(np.zeros(len(hull.vertices), dtype=Mesh.dtype))
        for i, vertex in enumerate(hull.vertices):
            shape.vectors[i] = hull.points[vertex][::-1]  # Turn it inside out
            #print(shape.vectors[vertex])
        size = np.max(hull.points, axis=0)
        #print(size)
        shape.save(folder + name + '.stl')
        
        '''
        deal with the mesh file assignment
        '''          
        asset = child.find('asset')
        mesh = ET.SubElement(asset, "mesh")
        mesh.set('name',name)
        mesh.set('file',folder+name+'.stl')
        '''
        deal with the object
        '''  
        
        obj.set('name', name)                
        #obj.set('pos', '0.375 0.0 {}'.format(pos_table[2]+0.035))
        obj.set('pos', '{} {} {}'.format(0.355+0.1*np.random.rand(), 0.1*np.random.rand()-0.05, pos_table[2]+0.035))
        #inertial = ET.SubElement(obj, "inertial")
        #inertial.set('pos','0 0 0')
        #inertial.set('mass','0.0458')                
        #inertial.set('diaginertia','4.11145e-05 2.69913e-05 1.81232e-05')
        geom = ET.SubElement(obj, "geom")
        geom.set('type','mesh')
        geom.set('mesh',name)
        geom.set('mass','0.1')
        geom.set('rgba','0.999 0.999 0.999 1')    
        geom.set('friction','0.1 0.005 0.0001')
        geom.set('solimp','0.95 0.95 0.01')
        geom.set('solref','0.01 1')
        geom.set('condim','4')
        geom.set('margin','0.0')
        joint = ET.SubElement(obj, "joint")
        joint.set('name','obj_J')
        joint.set('type','free')
        site = ET.SubElement(obj, "site")
        site.set('name', 'object0')
        site.set('size', '0.06 0.0001')
        site.set('rgba', '0 1 0 1')
        site.set('type', 'cylinder')
        site.set('pos', '{} {} {}'.format(0,0,-0.005))
        
        '''
        generate spheres below object
        '''
        
        for i in range(100):
            tmp_pos = np.array([0.1*(np.random.rand()-0.5),0.1*(np.random.rand()-0.5)])
            while not self.isInHull(tmp_pos, hull2D): 
                tmp_pos = np.array([0.1*(np.random.rand()-0.5),0.1*(np.random.rand()-0.5)])
            else:
                geom = ET.SubElement(obj, "geom")
                geom.set('name','sphere{}'.format(i))
                geom.set('type','sphere')
                geom.set('size','{}'.format(np.random.uniform(0.0001,0.0004)))
                geom.set('pos','{} {} -0.005'.format(tmp_pos[0],tmp_pos[1]))
                geom.set('rgba','0.2 1.0 0.2 1')    
                geom.set('friction','0.25 0.005 0.0001')
                geom.set('solimp','0.95 0.95 0.01')
                geom.set('solref','0.01 1')
                geom.set('condim','4')
                geom.set('margin','0.0')  
    
    def _get_obs(self):
        '''
        from OpenAI Gym to match with it
        '''
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            names = [n for n in self.sim.model.joint_names if n.startswith('robot')]
            robot_qpos = np.array([self.sim.data.get_joint_qpos(name) for name in names])
            robot_qvel = np.array([self.sim.data.get_joint_qvel(name) for name in names])
        else:
            robot_qpos = np.zeros(0)
            robot_qvel = np.zeros(0)            
            
        #if self.has_object:
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = np.squeeze(object_pos.copy())
        
        obs = np.concatenate([
            object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_pos, grip_velp, gripper_state, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
        
    def isInHull(self, point, hull):
        tolerance=1e-12
        flag = all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)
        return flag
        
    def update_v(self):
        self.lock.acquire()
        if len(self.queue) != 0:
            v_tgt_new = self.queue.popleft()            
            for i in range(self.nv):
                self.v_tgt[i] = v_tgt_new[i]
        self.lock.release()

    def cv2_viewer(self):
        inc_pos_v = 0.15
        inc_ang_v = 15/180*np.pi
        
        qtgt = np.array([-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335])
        
        self.lock1.acquire()
        for i in range(7):
            self.sim.data.qpos[i] = qtgt[i]
        self.lock1.release()          
        
        gqtgt = float(self.sim.data.ctrl[self.nv])
        self.initQpos = [self.sim.data.qpos[i] for i in range(self.sim.model.nq)]
        
        # initial speeds
        ang_v = np.array([1,0,0,0])
        pos_v = np.array([0,0,0])
        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])
        
        while True:
            twist_ee = self.twist_ee
            is_cmd_received = False
            keypressed = cv2.waitKey(1)
            
            if keypressed == 27:
                break
            elif keypressed == ord('\\'):#92: #\, up
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,0,inc_pos_v,0,0,0])
            elif keypressed == 13: #return, down
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,0,-inc_pos_v,0,0,0])
            elif keypressed == ord('i'): # forward
                is_cmd_received = True
                twist_ee = twist_ee + np.array([inc_pos_v,0,0,0,0,0])
            elif keypressed == ord('k'): # backward
                is_cmd_received = True
                twist_ee = twist_ee + np.array([-inc_pos_v,0,0,0,0,0])
            elif keypressed == ord('j'): # left
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,inc_pos_v,0,0,0,0])
            elif keypressed == ord('l'): # right
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,-inc_pos_v,0,0,0,0])
                
            elif keypressed == ord('q'): # roll
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(inc_ang_v/2),np.sin(inc_ang_v/2),0,0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('a'): # -roll
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(-inc_ang_v/2),np.sin(-inc_ang_v/2),0,0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('w'): # pitch
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(inc_ang_v/2),0,np.sin(inc_ang_v/2),0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('s'): # -pitch
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(-inc_ang_v/2),0,np.sin(-inc_ang_v/2),0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('e'): # yaw
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(inc_ang_v/2),0,0,np.sin(inc_ang_v/2)]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('d'): # -yaw
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(-inc_ang_v/2),0,0,np.sin(-inc_ang_v/2)]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]                
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('c'): # gripper close
                is_cmd_received = True
                tmp = gqtgt + 0.1
                if tmp < 1.5:
                    gqtgt = tmp
                else:
                    gqtgt = 1.5
            elif keypressed == ord('v'): # gripper open
                is_cmd_received = True
                tmp = gqtgt - 0.1
                if tmp > 0:
                    gqtgt = tmp
                else:
                    gqtgt = 0
            elif keypressed == ord('p'): # pause robot
                is_cmd_received = True
                ang_v = np.array([1,0,0,0])
                pos_v = np.array([0,0,0])
                twist_ee = np.array([0, 0, 0, 0, 0, 0])
            elif keypressed == ord('y'): # save the simulator model
                print("save the simulator model")
                time = datetime.datetime.now()
                filename = 'saves/robot_'+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.xml'
                file = open(filename, 'w')
                self.sim.save(file, 'xml')
                print("save the simulator model done")

                print("cam_pos")
                print(self.sim.model.cam_pos[0])
                print(self.sim.model.cam_pos[1])
                print(self.sim.model.cam_pos[2])
                print(self.sim.model.cam_pos[3])

                print("cam_quat")
                print(self.sim.model.cam_quat[0])
                print(self.sim.model.cam_quat[1])
                print(self.sim.model.cam_quat[2])
                print(self.sim.model.cam_quat[3])

                print("cam_poscom0")
                print(self.sim.model.cam_poscom0[0])
                print(self.sim.model.cam_poscom0[1])
                print(self.sim.model.cam_poscom0[2])
                print(self.sim.model.cam_poscom0[3])

                print("cam_pos0")
                print(self.sim.model.cam_pos0[0])
                print(self.sim.model.cam_pos0[1])
                print(self.sim.model.cam_pos0[2])
                print(self.sim.model.cam_pos0[3])

                print("cam_mat0")
                print(self.sim.model.cam_mat0[0])
                print(self.sim.model.cam_mat0[1])
                print(self.sim.model.cam_mat0[2])
                print(self.sim.model.cam_mat0[3])

                print("cam_ipd")
                print(self.sim.model.cam_ipd[0])
                print(self.sim.model.cam_ipd[1])
                print(self.sim.model.cam_ipd[2])
                print(self.sim.model.cam_ipd[3])

                print("extrinsic")
                print(get_camera_extrinsic_matrix(self.sim, "realsense-top"))
                print(get_camera_extrinsic_matrix(self.sim, "realsense-front"))
                print(get_camera_extrinsic_matrix(self.sim, "realsense-side-1"))
                print(get_camera_extrinsic_matrix(self.sim, "realsense-side-2"))

                # width=424, height=240
                print("intrinsic")
                print(get_camera_intrinsic_matrix(self.sim, "realsense-top", 240, 424))
                print(get_camera_intrinsic_matrix(self.sim, "realsense-front", 240, 424))
                print(get_camera_intrinsic_matrix(self.sim, "realsense-side-1", 240, 424))
                print(get_camera_intrinsic_matrix(self.sim, "realsense-side-2", 240, 424))

                # export the camera as image
                print("export the camera as image")
           
            if is_cmd_received:
                self.twist_ee = twist_ee
                #gripper
                self.lock1.acquire()
                self.sim.data.ctrl[self.nv] = gqtgt 
                self.sim.data.ctrl[self.nv+1] = gqtgt
                self.lock1.release()
            #joint
            self.lock.acquire()                    
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt)                    
            self.lock.release()
            
            # self.show_image(title)
            self.show_color_image("top", self.queue_img)
            self.show_color_image("front", self.queue_img_front)
            self.show_color_image("side_1", self.queue_img_side1)
            self.show_color_image("side_2", self.queue_img_side2)
        cv2.destroyAllWindows()
  
    def show_depth_image(self, title, queue_img):
        znear = 0.01
        zfar = 50.0

        image_data = []  
        self.lockcv2.acquire()
        if len(queue_img) != 0:
            image_data = queue_img.popleft() 
        self.lockcv2.release()
        if len(image_data) != 0:
            div_near = 1/(znear*self.model.stat.extent)
            div_far = 1/(zfar*self.model.stat.extent)
            s = div_far-div_near
            image_data = 1/(s*image_data + div_near)
            
            #limit measurement range
            dplim_upper = 2
            dplim_lower = 0.16
            image_data[image_data<=dplim_lower]=0
            image_data[image_data>=dplim_upper]=1.0
            
            #add noise
            # image_noise_1=stats.distributions.norm.rvs(0,0.00005,size=image_data.shape)
            # image_noise_2=np.random.normal(0,0.00015,size=image_data.shape)
            # image_data = image_data + image_noise_1 + image_noise_2

            norm_image = cv2.normalize(image_data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)            
            cv2.imshow(title, norm_image)

            # export the camera as image
            time = datetime.datetime.now()
            filename = 'data/episode0/'+title+'_depth/robot_'+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.png'
            cv2.imwrite(filename, norm_image*255)

    def show_color_image(self, title, queue_img):
        image_data = []
        self.lockcv2.acquire()
        if len(queue_img) != 0:
            image_data = queue_img.popleft()
        self.lockcv2.release()
        if len(image_data) != 0:
            cv2.imshow(title, image_data)

            # export the camera as image
            time = datetime.datetime.now()
            filename = 'data/episode0/'+title+'_rgb/robot_'+str(time.year)+'_'+str(time.month)+'_'+str(time.day)+'_'+str(time.hour)+'_'+str(time.minute)+'_'+str(time.second)+'.png'
            cv2.imwrite(filename, image_data)

    def start(self):
        ct = 0       
        while True:  
            self.lock1.acquire()          
            for i in range(self.nv):
                self.sim.data.qfrc_applied[i] = self.sim.data.qfrc_bias[i]
                self.sim.data.qvel[i] = self.v_tgt[i]
            if  (ct*self.sim.model.opt.timestep/0.01).is_integer():
                self.update_v()
            ct = ct + 1                
            self.sim.step()
            self.lock1.release()
            self.viewer.render()
            if ct%17 == 1:
                self.push_rgb_image_to_queue(self.queue_img,  0,self.lockcv2)
                self.push_rgb_image_to_queue(self.queue_img_front, 1, self.lockcv2Front)
                self.push_rgb_image_to_queue(self.queue_img_side1, 2, self.lockcv2Side1)
                self.push_rgb_image_to_queue(self.queue_img_side2, 3, self.lockcv2Side2)

    def push_rgb_image_to_queue(self, queue_img, camera_id, lockcv2):
        self.offscreen.render(width=424, height=240, camera_id=camera_id)
        a=self.offscreen.read_pixels(width=424, height=240, depth=True)
        rgb_img = a[0]
        rgb_img = rgb_img[:, ::-1, ::-1]
        lockcv2.acquire()
        queue_img.append(rgb_img)
        lockcv2.release()

    def push_depth_image_to_queue(self, queue_img, camera_id, lockcv2):
        self.offscreen.render(width=424, height=240, camera_id=camera_id)
        a=self.offscreen.read_pixels(width=424, height=240, depth=True)
        depth_img = a[1]
        depth_img = depth_img[:, ::-1]
        lockcv2.acquire()
        queue_img.append(depth_img)
        lockcv2.release()
            
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

'''
from OpenAI gym
''' 
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler
    
if __name__ == "__main__":
    # get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='none')
    args = parser.parse_args()
    action = args.action
    
    simulator = SimulatorVelCtrl()
    simulator.init("gen3_robotiq_2f_85_cellphone_table_grasp.xml", 7, action)
    simulator.start()