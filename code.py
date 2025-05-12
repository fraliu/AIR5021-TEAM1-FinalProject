import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import sys
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# 全局变量用于跟踪PyBullet连接状态
pybullet_connected = False

class DualArmPouringEnv:
    def __init__(self, render=True, max_steps=1000):
        global pybullet_connected
        self.max_steps = max_steps
        self.current_step = 0
        
        # 连接物理引擎
        print("连接到物理引擎...")
        # 确保不存在已有的连接
        if pybullet_connected:
            try:
                p.disconnect()
                print("断开之前的连接")
                pybullet_connected = False
            except:
                pass

        # 连接到物理引擎
        if render:
            try:
                self.client = p.connect(p.GUI)
                pybullet_connected = True
                print("已连接到GUI模式物理引擎")
            except Exception as e:
                print(f"无法连接到GUI模式: {e}，尝试DIRECT模式")
                self.client = p.connect(p.DIRECT)
                pybullet_connected = True
                print("已连接到DIRECT模式物理引擎")
        else:
            self.client = p.connect(p.DIRECT)
            pybullet_connected = True
            print("已连接到DIRECT模式物理引擎")
        
        # 设置PyBullet资源路径 - 在连接后设置
        print("设置PyBullet资源路径...")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 设置重力
        p.setGravity(0, 0, -9.8)
        print("已设置重力")
        
        # 设置相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # 加载场景
        self._load_scene()
        
        # 加载机械臂
        self._load_robots()
        
        # 加载水壶和杯子
        self._load_objects()
        
        # 初始化机械臂位置
        self._reset_robot_positions()
        
        # 动作空间和观察空间
        self.action_dim = 12  # 两个机械臂，每个有6个自由度
        self.observation_dim = 27  # 位置、速度和状态标志
        
        # 增强可视化效果
        if render:
            self._reset_visualization()
        
    def _reset_visualization(self):
        """重置可视化参数，确保环境元素正确显示"""
        # 调整默认相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,    # 距离更近一些
            cameraYaw=30,          # 改变水平角度
            cameraPitch=-30,       # 改变垂直角度
            cameraTargetPosition=[0, 0, 0.7]  # 将焦点移到桌面附近的中心
        )
        
        # 设置GUI选项
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # 添加一个坐标系在原点
        #p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [1, 0, 0], lineWidth=2.0)
        #p.addUserDebugLine([0, 0, 0], [0, 0.2, 0], [0, 1, 0], lineWidth=2.0)
        #p.addUserDebugLine([0, 0, 0], [0, 0, 0.2], [0, 0, 1], lineWidth=2.0)
        
        # 在关键位置添加文本标签，帮助识别物体
        p.addUserDebugText("机械臂1", [-0.8, 0, 0.9], [1, 0, 0], textSize=1.5)
        p.addUserDebugText("机械臂2", [0.8, 0, 0.9], [0, 0, 1], textSize=1.5)
        p.addUserDebugText("水壶", [-0.3, -0.3, 0.9], [1, 0, 0], textSize=1.5)
        p.addUserDebugText("杯子", [0.3, -0.3, 0.9], [0, 0, 1], textSize=1.5)
        
    def _load_scene(self):
        # 加载平面
        try:
            self.plane_id = p.loadURDF("plane.urdf")
            print("成功加载地面平面")
        except Exception as e:
            print(f"加载地面平面时出错: {e}")
            # 尝试创建一个简单的地面作为替代
            self.plane_id = p.createCollisionShape(p.GEOM_PLANE)
            p.createMultiBody(0, self.plane_id)
            print("创建了替代地面")
        
        # 尝试加载桌子，如果失败则创建一个简单的替代物
        try:
            self.table_id = p.loadURDF("table/table.urdf", 
                                     basePosition=[0, 0, 0],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                     globalScaling=1.0)
            print("成功加载桌子模型")
        except Exception as e:
            print(f"加载桌子模型时出错: {e}")
            # 创建一个简单的箱体作为桌子
            table_shape = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=[3.8, 1.8, 0.025]
            )
            table_visual = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[3.8, 0.8, 0.025], 
                rgbaColor=[0.8, 0.88, 0.8, 1]
            )
            self.table_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=table_shape,
                baseVisualShapeIndex=table_visual,
                basePosition=[0, 0, 0.5]
            )
            print("创建了替代桌子")
                                 
    def _load_robots(self):
        # 使用Kuka机器人替代UR5，因为Kuka是PyBullet默认提供的
        # 第一个机械臂 - 用于拿水壶
        self.arm1_id = p.loadURDF("kuka_iiwa/model.urdf", 
                                basePosition=[-1.0, 0, 0.6],  # 调整为更远的位置
                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)
        
        # 第二个机械臂 - 用于拿杯子
        self.arm2_id = p.loadURDF("kuka_iiwa/model.urdf", 
                                basePosition=[1.2, 0, 0.6],  # 调整为更远的位置
                                baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi]),
                                useFixedBase=True)
        
        # 获取机械臂关节信息
        self.arm1_joints = []
        self.arm2_joints = []
        
        for i in range(p.getNumJoints(self.arm1_id)):
            info = p.getJointInfo(self.arm1_id, i)
            if info[2] == p.JOINT_REVOLUTE:  # 只保留旋转关节
                self.arm1_joints.append(i)
                
        for i in range(p.getNumJoints(self.arm2_id)):
            info = p.getJointInfo(self.arm2_id, i)
            if info[2] == p.JOINT_REVOLUTE:  # 只保留旋转关节
                self.arm2_joints.append(i)
                
        print(f"找到机械臂1的关节数量: {len(self.arm1_joints)}")
        print(f"找到机械臂2的关节数量: {len(self.arm2_joints)}")
                
        # 为机械臂添加夹具
        self._attach_grippers()
        
    def _attach_grippers(self):
        # 在现实场景中，需要添加更复杂的夹具模型
        # 这里简化为一个简单的固定连接
        
        # 为简化实现，我们使用可视化形状作为夹具
        self.gripper1_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02], rgbaColor=[1.0, 0.3, 0.3, 1])  # 更鲜艳的红色
        self.gripper2_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02], rgbaColor=[0.3, 0.3, 1.0, 1])  # 更鲜艳的蓝色
            
        # 创建夹具的碰撞形状
        self.gripper1_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02])
        self.gripper2_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02])
            
        # 连接夹具到机械臂末端
        # 对于Kuka机器人，末端关节通常是索引6
        arm1_end_effector = 6  # 机械臂末端连接点
        arm2_end_effector = 6
        
        # 验证末端关节索引是否有效
        if len(self.arm1_joints) > 0:
            arm1_end_effector = self.arm1_joints[-1]
        if len(self.arm2_joints) > 0:
            arm2_end_effector = self.arm2_joints[-1]
            
        print(f"使用末端关节索引 - 机械臂1: {arm1_end_effector}, 机械臂2: {arm2_end_effector}")
        
        # 计算夹具的位置和方向
        pos1 = [0, 0, 0.1]  # 相对于末端的位置
        orn1 = p.getQuaternionFromEuler([0, 0, 0])
        
        pos2 = [0, 0, 0.1]
        orn2 = p.getQuaternionFromEuler([0, 0, 0])
        
        # 创建并连接夹具
        self.gripper1_body = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=self.gripper1_col,
            baseVisualShapeIndex=self.gripper1_id,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )
        
        self.gripper2_body = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=self.gripper2_col,
            baseVisualShapeIndex=self.gripper2_id,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 创建一个固定约束将夹具连接到机械臂末端
        try:
            self.gripper1_constraint = p.createConstraint(
                parentBodyUniqueId=self.arm1_id,
                parentLinkIndex=arm1_end_effector,
                childBodyUniqueId=self.gripper1_body,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=pos1,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=orn1,
                childFrameOrientation=[0, 0, 0, 1]
            )
            
            self.gripper2_constraint = p.createConstraint(
                parentBodyUniqueId=self.arm2_id,
                parentLinkIndex=arm2_end_effector,
                childBodyUniqueId=self.gripper2_body,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=pos2,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=orn2,
                childFrameOrientation=[0, 0, 0, 1]
            )
            print("成功创建夹具约束")
        except Exception as e:
            print(f"创建夹具约束时出错: {e}")
        
    def _load_objects(self):
        # 创建水壶模型
        teapot_height = 0.2
        teapot_radius = 0.06
        spout_length = 0.1
        
        # 创建水壶主体（简化为圆柱体）
        teapot_body = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=teapot_radius, 
            height=teapot_height
        )
        
        # 创建水壶嘴（简化为细长圆柱体）
        teapot_spout = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=0.01, 
            height=spout_length
        )
        
        # 创建水壶手柄（简化为细长方块）
        teapot_handle = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[0.01, 0.04, 0.04]
        )
        
        # 创建可视化形状 - 使用更鲜艳的颜色
        teapot_visual_body = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=teapot_radius, 
            length=teapot_height, 
            rgbaColor=[1.0, 0.0, 0.0, 1.0]  # 鲜艳的红色
        )
        
        teapot_visual_spout = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=0.01, 
            length=spout_length, 
            rgbaColor=[0.8, 0.0, 0.0, 1.0]  # 暗红色
        )
        
        teapot_visual_handle = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.01, 0.04, 0.04], 
            rgbaColor=[0.8, 0.0, 0.0, 1.0]  # 暗红色
        )
        
        # 创建水壶多体对象
        self.teapot_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=teapot_body,
            baseVisualShapeIndex=teapot_visual_body,
            basePosition=[-0.25, -0.3, 0.7],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            linkMasses=[0.1, 0.1],
            linkCollisionShapeIndices=[teapot_spout, teapot_handle],
            linkVisualShapeIndices=[teapot_visual_spout, teapot_visual_handle],
            linkPositions=[[teapot_radius + spout_length/2, 0, 0], 
                        [-teapot_radius - 0.01, 0, 0]],
            linkOrientations=[p.getQuaternionFromEuler([0, math.pi/2, 0]), 
                            p.getQuaternionFromEuler([0, 0, 0])],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]]
        )
        
        # 创建杯子（简化为圆柱体）
        cup_height = 0.15
        cup_radius = 0.04
        
        cup_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=cup_radius, 
            height=cup_height
        )
        
        cup_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=cup_radius, 
            length=cup_height, 
            rgbaColor=[1.0, 1.0, 0, 0.6]  # 鲜艳的蓝色，半透明
        )
        
        self.cup_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=cup_collision,
            baseVisualShapeIndex=cup_visual,
            basePosition=[0.3, -0.3, 0.7],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # 添加水的可视化效果（初始为空）
        self.water_level = 0.0
        self.water_color = [0.0, 0.7, 1.0, 0.6]  # 更鲜艳的蓝色，半透明
        self.update_water_level()

    def update_water_level(self):
        if hasattr(self, 'water_visual_id'):
            p.removeBody(self.water_visual_id)
            
        if self.water_level > 0:
            water_height = self.water_level
            cup_height = 0.15
            cup_radius = 0.04
            
            water_collision = -1  # water has no collision, just visual effect
            
            water_visual = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=cup_radius-0.005,  # slightly smaller than cup
                length=water_height,
                rgbaColor=self.water_color
            )
            
            # Get cup position and orientation
            cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup_id)
            
            # Calculate water position in cup (aligned at bottom, height adjusted)
            water_pos = [
                cup_pos[0],
                cup_pos[1],
                cup_pos[2] - (cup_height/2 - water_height/2)
            ]
            
            self.water_visual_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=water_collision,
                baseVisualShapeIndex=water_visual,
                basePosition=water_pos,
                baseOrientation=cup_orn
            )
        
    def _reset_robot_positions(self):
        # 设置机械臂初始位置
        initial_pos1 = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/2, 0]  # 调整为更自然的姿态
        initial_pos2 = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/4, 0]
        
        for i, joint in enumerate(self.arm1_joints):
            if i < len(initial_pos1):
                p.resetJointState(self.arm1_id, joint, initial_pos1[i])
            
        for i, joint in enumerate(self.arm2_joints):
            if i < len(initial_pos2):
                p.resetJointState(self.arm2_id, joint, initial_pos2[i])
    
    def reset(self):
        self.current_step = 0
        self._reset_robot_positions()
        
        # 重置物体位置
        p.resetBasePositionAndOrientation(
            self.teapot_id,
            [-0.5, -0.3, 0.7],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        p.resetBasePositionAndOrientation(
            self.cup_id,
            [0.5, -0.3, 0.7],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # 重置水位
        self.water_level = 0.0
        self.update_water_level()
        
        # 增强可视化
        if self.client == p.GUI:
            self._reset_visualization()
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        获取环境观测向量
        返回一个包含机械臂状态、物体位置/方向和抓取状态的向量
        """
        # 获取机械臂状态
        arm1_states = []
        arm2_states = []
        
        for joint in self.arm1_joints:
            state = p.getJointState(self.arm1_id, joint)
            arm1_states.extend([state[0], state[1]])  # 位置和速度
            
        for joint in self.arm2_joints:
            state = p.getJointState(self.arm2_id, joint)
            arm2_states.extend([state[0], state[1]])
            
        # 获取物体状态
        teapot_pos, teapot_orn = p.getBasePositionAndOrientation(self.teapot_id)
        cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup_id)
        
        # 检查抓取状态 - 新增部分
        arm1_end_pos = p.getLinkState(self.arm1_id, self.arm1_joints[-1])[0]
        arm2_end_pos = p.getLinkState(self.arm2_id, self.arm2_joints[-1])[0]
        
        # 判断是否抓住物体 (距离小于阈值)
        holding_teapot = int(np.linalg.norm(np.array(arm1_end_pos) - np.array(teapot_pos)) < 0.1)
        holding_cup = int(np.linalg.norm(np.array(arm2_end_pos) - np.array(cup_pos)) < 0.1)
        
        # 检查水壶是否在倒水的位置和角度
        teapot_euler = p.getEulerFromQuaternion(teapot_orn)
        spout_to_cup = np.linalg.norm(
            np.array([
                teapot_pos[0] + 0.1 * np.cos(teapot_euler[2]), 
                teapot_pos[1] + 0.1 * np.sin(teapot_euler[2]), 
                teapot_pos[2]
            ]) - 
            np.array([cup_pos[0], cup_pos[1], cup_pos[2] + 0.075])
        )
        is_pouring = int((teapot_euler[0] > 0.5) and (spout_to_cup < 0.1))
        
        # 将状态中的关键部分提取为27维向量 (原24维 + 3个新增标志)
        # 机械臂1和机械臂2各取最多6个关节的位置和速度 (12维)
        # 水壶和杯子的3D位置 (6维)
        # 水壶和杯子的姿态欧拉角 (6维)
        # 是否拿住水壶 (1维)
        # 是否拿住杯子 (1维)
        # 是否在倒水 (1维)
        
        # 截取前12维的机械臂状态
        if len(arm1_states) + len(arm2_states) > 12:
            arm_states = (arm1_states + arm2_states)[:12]
        else:
            arm_states = arm1_states + arm2_states
            # 如果不足12维，补0
            arm_states = arm_states + [0] * (12 - len(arm_states))
        
        # 提取物体位置和欧拉角
        cup_euler = p.getEulerFromQuaternion(cup_orn)
        
        # 组合成27维向量 (添加抓取状态)
        observation = np.array(
            arm_states +
            list(teapot_pos) +
            list(cup_pos) +
            list(teapot_euler) +
            list(cup_euler) + 
            [holding_teapot, holding_cup, is_pouring]  # 新增3个二进制标志
        )
        
        # 确保向量长度为27
        if len(observation) > 27:
            observation = observation[:27]
        elif len(observation) < 27:
            observation = np.concatenate([observation, np.zeros(27 - len(observation))])
            
        return observation.astype(np.float32)

        

        
    def close(self):
        global pybullet_connected
        if pybullet_connected:
            try:
                p.disconnect(self.client)
                pybullet_connected = False
                print("断开PyBullet连接")
            except:
                print("断开PyBullet连接时出错")
        
    def get_gripper_position(self, arm_id, link_index=6):
        """获取夹具位置"""
        return p.getLinkState(arm_id, link_index)[0]

    def demonstration(self):
        """演示两个机械臂协作完成倒水任务"""
        self.reset()
        
        print("开始演示...")
        print("1. 机械臂1移向水壶...")
        
        # 机械臂1移向水壶
        teapot_pos, teapot_orn = p.getBasePositionAndOrientation(self.teapot_id)
        approach_pos = [teapot_pos[0] - 0.4, teapot_pos[1], teapot_pos[2]]
        
        # 使用IK移动机械臂
        arm1_end_effector = self.arm1_joints[-1]
        #target_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])  # 确保夹具顶部朝上
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])  # 确保夹具顶部朝上
        # 计算IK
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm1_id,
            endEffectorLinkIndex=arm1_end_effector,
            targetPosition=approach_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # 设置关节位置
        for i, joint in enumerate(self.arm1_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm1_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(200):
            p.stepSimulation()
            time.sleep(0.01)  # 控制模拟速度
        
        print("2. 机械臂1接近水壶...")
        # 降低机械臂1到抓取位置
        grab_pos = [teapot_pos[0] - 0.23, teapot_pos[1], teapot_pos[2]]
        
        # 计算IK
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm1_id,
            endEffectorLinkIndex=arm1_end_effector,
            targetPosition=grab_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # 设置关节位置
        for i, joint in enumerate(self.arm1_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm1_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        print("3. 抓取水壶...")
        # 创建约束来模拟抓取水壶
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.arm1_id,
            parentLinkIndex=arm1_end_effector,
            childBodyUniqueId=self.teapot_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.1],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=target_orn
        )
        
        print("4. 举起水壶...")
        # 举起水壶
        lift_pos = [teapot_pos[0] + 0.4, teapot_pos[1], teapot_pos[2] + 0.2]
        
        # 计算IK
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm1_id,
            endEffectorLinkIndex=arm1_end_effector,
            targetPosition=lift_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # 设置关节位置
        for i, joint in enumerate(self.arm1_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm1_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        
        print("5. 机械臂2接近杯子...")
        # 机械臂2移向杯子
        cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup_id)
        approach_pos = [cup_pos[0] + 0.3, cup_pos[1], cup_pos[2] ]
        
        # 使用IK移动机械臂
        arm2_end_effector = self.arm2_joints[-1]
        target_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])  # 确保夹具顶部朝上
        
        # 计算IK
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm2_id,
            endEffectorLinkIndex=arm2_end_effector,
            targetPosition=approach_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # 设置关节位置
        for i, joint in enumerate(self.arm2_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm2_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(200):
            p.stepSimulation()
            time.sleep(0.01)
        
        print("6. 机械臂2接近杯子...")
        # 降低机械臂2到抓取位置
        grab_pos = [cup_pos[0] + 0.18, cup_pos[1] , cup_pos[2]]
        
        # 计算IK
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm2_id,
            endEffectorLinkIndex=arm2_end_effector,
            targetPosition=grab_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # 设置关节位置
        for i, joint in enumerate(self.arm2_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm2_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        
        print("7. 抓取杯子...")
        # 创建约束来模拟抓取杯子
        cup_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.arm2_id,
            parentLinkIndex=arm2_end_effector,
            childBodyUniqueId=self.cup_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.1],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]),
            childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
            
        )
        
        # 确保杯子顶部朝上
        p.resetBasePositionAndOrientation(self.cup_id, cup_pos, p.getQuaternionFromEuler([0, 0, 0]))
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
        print("8. 移动物体到倒水位置...")
        # 移动水壶和杯子到合适的倒水位置
        pour_teapot_pos = [0.55, 0, 1.2]
        pour_cup_pos = [0.2, 0, 0.8]
        target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
        # 移动机械臂1（水壶）
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm1_id,
            endEffectorLinkIndex=arm1_end_effector,
            targetPosition=pour_teapot_pos,
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )

        for i, joint in enumerate(self.arm1_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm1_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )

        
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm2_id,
            endEffectorLinkIndex=arm2_end_effector,
            targetPosition=pour_cup_pos,
            targetOrientation=p.getQuaternionFromEuler([0, -np.pi/2, 0]),
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        for i, joint in enumerate(self.arm2_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm2_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(50):
            p.stepSimulation()
            time.sleep(0.01)
        
        print("9. 倾斜水壶倒水...")
        # 倾斜水壶倒水
        pour_orn = p.getQuaternionFromEuler([0, math.pi * 0.6, 0])
        
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm1_id,
            endEffectorLinkIndex=arm1_end_effector,
            targetPosition=pour_teapot_pos,
            targetOrientation=pour_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        for i, joint in enumerate(self.arm1_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm1_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟倒水
        for i in range(20):
            self.water_level = i * 0.005
            self.update_water_level()
            
            for _ in range(10):
                p.stepSimulation()
                time.sleep(0.01)
        
        print("10. 完成倒水，恢复姿态...")
        # 恢复水壶姿态
        normal_orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
        
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.arm1_id,
            endEffectorLinkIndex=arm1_end_effector,
            targetPosition=pour_teapot_pos,
            targetOrientation=normal_orn,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        for i, joint in enumerate(self.arm1_joints):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm1_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
        
        # 模拟物理让机械臂移动到位
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        
        print("演示完成！请按任意键关闭...")
        input()

# 简化的Gym环境包装器
class DualArmPouringGymEnv(gym.Env):
    """适合stable-baselines3的Gym环境封装器"""
    
    metadata = {'render.modes': ['human']}  # 必须提供的元数据
    
    def __init__(self, render=True, max_steps=1000):
        global pybullet_connected
        super(DualArmPouringGymEnv, self).__init__()
        
        # 如果已经有PyBullet连接，先断开
        if pybullet_connected:
            try:
                p.disconnect()
                pybullet_connected = False
                print("断开之前的PyBullet连接")
            except:
                pass
                
        # 创建基础环境
        self.env = DualArmPouringEnv(render=render, max_steps=max_steps)
        
        # 定义动作空间 (12维, 每个机械臂6维)
        self.action_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(12,), dtype=np.float32
        )
        
        # 更新观察空间为27维 (24维原始状态 + 3个状态标志)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(27,), dtype=np.float32
        )
    
    
    def close(self):
        self.env.close()
    
    def render(self, mode='human'):
        pass  # PyBullet已经处理了渲染
    
    def demonstration(self):
        """调用环境的演示函数"""
        self.env.demonstration()

# 主函数
def main():
    global pybullet_connected
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="双臂倒水机器人演示")
    parser.add_argument('--demo', action='store_true', help='显示预设的演示')
    args = parser.parse_args()
    
    # 确保开始前没有活跃的PyBullet连接
    if pybullet_connected:
        try:
            p.disconnect()
            pybullet_connected = False
            print("主函数开始时断开旧的PyBullet连接")
        except:
            pass
            
    # 演示模式
    if args.demo:
        env = DualArmPouringGymEnv(render=True)
        env.demonstration()
        env.close()
    else:
        print("未指定演示模式，退出程序。")

if __name__ == "__main__":
    main()