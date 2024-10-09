import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import spaces
from CSEnv_noRobot import CSEnv
import time as t

# Constants
deg = np.pi /180
rad = 180 / np.pi
cm = 0.01
m = 1
Frequency = 100
_rangeXLower = 35 * cm  # was 25
_rangeXLUpper = 55 * cm # was 65
_rangeY = 20 * cm
Frequency = 40
maxTime = 20
max_episode_steps= 200
graspScoreThreshold = 0.5
initialTimeStep = 5
distanceThreshold = 5 * cm
gripperOnGroundTreshold = 9 * cm
obsvConstant = 8
obsVector1 = np.empty(shape = (obsvConstant,), dtype = np.float32)

class GymEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 port, action_repeat = 1, imageObservation = True, numberOfPoints = 20):
        
        self._stepcounter = 0
        self._action_repeat = action_repeat
        self.imageObservation = imageObservation
        self.numberOfPoints = numberOfPoints
        self.distanceThreshold = distanceThreshold
        self.actionData = []
        self.obsvData = []
        self.rewardData = 0
        #super(GymEnv, self).__init__()  
        
        # Creating an instance of the main CoppeliaSim class
        self.cs = CSEnv(port)
        # Reseting
        self.reset()
        # Define Observation Space
        if self.imageObservation:
            self.observation_space = spaces.Dict({"image": spaces.Box(low=0, high=255, shape=(256,256), dtype=np.uint8),
                                              "vector": spaces.Box(low = -np.inf, high= np.inf, shape=(obsvConstant,), dtype=np.float32)})
        else:
            #obsvDimension = self._obsvShapePointPart + obsvConstant
            obsvDimension = obsvConstant
            self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (obsvDimension,), dtype = np.float32)
        
        # Define Action Space
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype = np.float32)



    def reset(self, seed = None, options = None):
    # def reset(self):
        super().reset(seed=seed)
        # Reseting the environment and returning back the observation

        self.terminated = 0 
        self._stepcounter = 0     
        self.success_counter = 0

        # Closing simulation
        self.cs.simStop()
        try:
            if self.objectHandle is not None:
                self.cs.removeObject(self.objectHandle)
        except:
            pass

        # Enabling Stepping mode, starting the simulation, and simulating for a few time steps
        self.cs.simStep()
        self.cs.simStart()
        for _ in range(initialTimeStep):
            self.cs.simStep()

        # Choosing a random object and importing it
        self.objectName, self.objectHandle = self.cs.ImportShape()
        
        # setting the location of the object at a radnom location
        _currPosZ = self.cs.getObjectPos(self.objectHandle)
        _randPosX = random.uniform(_rangeXLower,_rangeXLUpper)
        _randPosY = random.uniform(-_rangeY,_rangeY)
        #_PosZ = 0.08
        _PosZ = _currPosZ[2]
        self.cs.PlaceObject([_randPosX,_randPosY,_PosZ], self.objectHandle, self.cs.robotBaseFrameHandle)
        self.cs.simStep()
        _objPosition = self.cs.getObjectPos(self.objectHandle)
        _objOrn = self.cs.getObjectOrn(self.objectHandle)
        self.objectInitialPose = np.concatenate((_objPosition, _objOrn), axis = None)

        # geting depth info
        t.sleep(1)
        if self.imageObservation:
            _depthInfo = self.cs.getDepthImage(num_points = self.numberOfPoints)
            self._depthImage = _depthInfo['Image']
            self._ObjLength = _depthInfo['ObjectLength']
        else:
            _depthInfo = self.cs.getDepthImage(num_points = self.numberOfPoints)
            self._ObjLength = _depthInfo['ObjectLength']
            self._depthPoints = _depthInfo['Points']
            self._obsvShapePointPart = len(self._depthPoints)

        # returning back the observation when using  Gym
        # return self.getObservation()
    
        # In case using Gymnasium instead of Gym, the reset method should also return info
        info = {"is_success": self._is_success}
        #info = {}
        obs = self.getObservation()
        return obs, info

    def step(self, action):

        self._action = action
        self.actionData = action

        for i in range(self._action_repeat):
            
            self.cs.apply_action(action)
            self.Position = self.cs.Position
            self.cs.simStep()

            if self.cs.checkPhase(self.objectHandle) == 2:
                self.cs.closeGripper(self.objectHandle)

            if self._termination():
                break

            self._stepcounter += 1
        
        _observation = self.getObservation()
        self.obsvData = _observation
        done = self._termination()
        reward = self._reward()
        self.rewardData = reward

        infos = {"is_success": self._is_success()}
        observation, reward, terminated, info = _observation, reward, done, infos
        # return observation, reward, terminated, info    # Gym
        #0  & Component X of Gripper's absolute position & $-$\infty & \infty & meter \\ \hline 
        info = {}
        return observation, reward, terminated, False, info   # Gymnasium

    def getObservation(self):

        """
        Observation when ImageObservation is enabled: 
                     End Effector pos: 4
                     Object pos:       3
                     Relative:         3        # Omitted   Obse space = 8 + image 
                     Closest dis:      3        # Omitted
                     Objects length:   1
                     depth Image

        When it's disabled:
                     End Effector pos: 4
                     Object pos:       3
                     Relative:         3        # Omitted   Obse space = 8 + points
                     Closest dis:      3        # Omitted
                     Objects length:   1
                     Points:           2 * number of points or self._obsvShapePointPart
        """

        # if self.imageObservation:
        #     self.observation_space = spaces.Dict({"image": spaces.Box(low=0, high=255, shape=(256,256), dtype=np.uint8),
        #                                       "vector": spaces.Box(low = -np.inf, high= np.inf, shape=(14,), dtype=np.float32)})
        # else:
        #     obsvDimension = self._obsvShapePointPart + 14
        #     self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.numberOfPoints,), dtype = np.float32)
        # if self.imageObservation:
        #     obsVector = np.empty(shape = (obsvConstant,), dtype = np.float32)
        
        # else:
        #     #_shape = obsvConstant + self._obsvShapePointPart
        #     _shape = obsvConstant
        #     # obsVector = np.empty(shape = (_shape,), dtype = np.float32)

        
        _vector = obsVector1
        ## Object Length:
        _vector[obsvConstant-1] = self._ObjLength

        ## End effector pose
        _eePos = self.cs.getEEPos()
        _eeSpin = _eePos[-1] * rad
        _vector[0:4] = [_eePos[0],_eePos[1],_eePos[2],_eeSpin]

        # Object pose:
        _objPose = self.cs.getObjectPos(self.objectHandle)
        _vector[4:7] = [_objPose[0],_objPose[1],_objPose[2]]

        # # Relative
        # _relPos = np.array(np.subtract(_objPose,_eePos[0:3]))
        # self.obsVector[7:10] = [_relPos[0],_relPos[1],_relPos[2]]

        # # Closest dist
        # _closDist = self.cs.getClosestDist(self.objectHandle)
        # self.obsVector[10:13] = [_closDist[0],_closDist[1],_closDist[2]]

        # Wrapping up all parts
        if self.imageObservation:
            obsv = {"image": self._depthImage, "vector": _vector}
        
        else:
            #self.obsVector[obsvConstant:] = self._depthPoints
            obsv = _vector

        return obsv

    def _termination(self):

        _maxTimeStep = self._stepcounter > max_episode_steps 
        _endEffectorPos = self.cs.getEEPos()
        _objPosition = self.cs.getObjectPos(self.objectHandle)
        _objOrn = self.cs.getObjectOrn(self.objectHandle)
        _objCurrentPose = np.concatenate((_objPosition, _objOrn), axis = None)
        _gripperOnGround = 1 if _endEffectorPos[2] < gripperOnGroundTreshold else 0
        _gripperObjectCollision = 1 if self.cs.gripperObjectCollision(self.objectInitialPose, _objCurrentPose) else 0
        # _fingerFloorCollision = self.cs.gripperFingersCollision()
        
        # _objectPos = self.cs.getObjectPos(self.objectHandle)
        # _outRangeX = 1 if (_objectPos[0] < 0.35 or _objectPos[0] > 0.55) else 0
        # _outRangeY = 1 if np.abs(_objectPos[1]) > 0.2 else 0
        # _objectOutofBox = 1 if (_outRangeX or _outRangeY) else 0

        #_condition = self.cs.getSimTime() > maxTime or _gripperOnGround or _objectOutofBox or _fingerFloorCollision
        # _condition = self._stepcounter > max_episode_steps or _gripperOnGround or _objectOutofBox or _fingerFloorCollision
        _condition = _maxTimeStep or _gripperOnGround or _gripperObjectCollision
        result = True if _condition else False
        return result

    def _reward(self):
        
        # Distance Reward
        __eefPose = self.cs.getEEPos()
        __eefPose = __eefPose[0:3]
        dist = np.abs(np.linalg.norm(__eefPose - self.cs.getObjectPos(self.objectHandle)))

        #distanceReward = (1 - np.tanh(2*dist)) if dist >= distanceThreshold else 1
        distanceReward = (1-1/(1+np.exp(-2*dist))) if dist >= distanceThreshold else 1
        # Penalty
        _objPosition = self.cs.getObjectPos(self.objectHandle)
        _objOrn = self.cs.getObjectOrn(self.objectHandle)
        _objCurrentPose = np.concatenate((_objPosition, _objOrn), axis = None)

        _baseFloorCollision, _baseObjectCollision  = self.cs.gripperBaseCollision(self.objectHandle)

        _fingerFloorCollision = self.cs.gripperFingersCollision()

        _gripperOnGround = 1 if __eefPose[2] < gripperOnGroundTreshold else 0

        _gripperObjectCollision = 1 if self.cs.gripperObjectCollision(self.objectInitialPose, _objCurrentPose) else 0

        collision =  _baseFloorCollision or _baseObjectCollision or _fingerFloorCollision or _gripperOnGround or _gripperObjectCollision

        collisionPenalty = -1 if collision else 0

        # Time step penalty

        timePenalty = -1 if self.cs.getSimTime() > maxTime else 0

        # Successful Grasp

        successfulGrasp = 1 if self.cs.objectGraspped(self.objectHandle) else 0

        # Grasp Score

        graspScore = 1 if self.cs.graspScore(self.objectHandle, self._ObjLength) > graspScoreThreshold else 0
        # Weighting the reward:
        #   distance: %75
        #   successful: %10
        #   time: %5
        #   collision: %5
        reward = 10 * (0.75 * distanceReward + 0.05 * timePenalty + 0.1 * collisionPenalty + 0.1 * successfulGrasp)

        if self.cs.objectGraspped(self.objectHandle):
            reward += graspScore
        return reward

    def _is_success(self):

        # if self.success_counter > 100:
        #     return np.float32(1.0)

        # if self._reward() > 1:
        #     self.success_counter += 1
        # else:
        #     self.success_counter = 0

        # return np.float32(self.success_counter > 100)

        return 1 if self.cs.objectGraspped(self.objectHandle) else 0
        
    def close(self):
        self.cs.simStop()
        self.cs.removeObject(self.objectHandle)

