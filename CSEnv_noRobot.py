import numpy as np
import time as t
import sys
import os
import random
import array

#from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from coppeliasim_zmqremoteapi_client import *
#from zmqRemoteApi import RemoteAPIClient
from Utils.FindPerimeter import find_perimeter_and_uniform_points
from Utils.GripperStateMachine import GripperStateMachine as GSM
from Utils.SensorProcessor import SensorProcessor as SP

# Constants
deg = np.pi /180
rad = 180 / np.pi
cm = 0.01
rotStep = 5
xPosLower  =  45 * cm
xPosUpper  =  65 * cm
yPosLower  = -40 * cm
yPosUpper  =  40 * cm
zPosLower  =  12 * cm
zPosUpper  =  35 * cm
thetaLim   = 170 * deg
GripperJointLimits = np.array([67, 90, 43, -55])
maxGripperTime = 10
tol = 2
Frequency = 40
graspThresholdH = 0.05
graspThresholdV = 0.08
translationalStep = 2
objectTranslationalCollisionThreshold = 10 * cm
objectRotationalCollisionThreshold = 45 * deg

# Objects Directory
Dirlist = np.array([['/home/major/Desktop/Simulation/Main/Objects/Can.ttm','Can'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Hammer.ttm','Hammer'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Prism.ttm','Prism'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Mouse.ttm', 'Mouse'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Marker.ttm', 'Marker'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Drill.ttm','Drill'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Cube1.ttm','Cube1'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Can4.ttm','Can4'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Cylinder.ttm','Cylinder'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Clamp.ttm','Clamp'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Can2.ttm','Can2'],
                    ['/home/major/Desktop/Simulation/Main/Objects/HalfCylinder.ttm','HalfCylinder'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Bowl.ttm','Bowl'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Can3.ttm','Can3'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Pitcher.ttm','Pitcher'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Bottle2.ttm','Bottle2'],
                    ['/home/major/Desktop/Simulation/Main/Objects/BleachCleaner.ttm','BleachCleaner'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Box1.ttm','Box1'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Bottle1.ttm','Bottle1'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Banana.ttm','Banana'],
                    ['/home/major/Desktop/Simulation/Main/Objects/CubeCylinderCut.ttm','CubeCylinderCut'],
                    ['/home/major/Desktop/Simulation/Main/Objects/Mug.ttm','Mug']])

class CSEnv():

    def __init__(self,port):

        # Intial values
        
        # Connecting to remote API server
        print(f'running CS on port = {port}')
        client = RemoteAPIClient(host='localhost',port = port)
        
        #self.sim = client.require('sim')
        self.sim = client.getObject('sim')

        # Stopping the sim in case it was not stopped
        self.simStop()

        # Loading the simulation scene
        self.sim.loadScene("/home/major/Desktop/Simulation/Main/Scene_noRobot.ttt")

        # Stopping the sim in case it was not stopped
        self.simStop()

        ### Retrieving Handles
        
        # Floor and plane
        self.planeHandle = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.planeHandle, self.sim.handle_single, self.sim.getObject('/Plane'), 0) 
        for i in range(1,5):
            _temp = self.sim.getObject(f'/Plane/Plane[{i-1}]')
            self.sim.addItemToCollection(self.planeHandle, self.sim.handle_single, _temp, 0) 
        _temp = self.sim.getObject('/Floor')
        self.sim.addItemToCollection(self.planeHandle, self.sim.handle_single, _temp, 0) 

        # vision sensor
        self.visionSensorHandle = self.sim.getObject('/VisionSensor')
        
        # World frame
        self.worldFrameHandle = self.sim.getObject('/DefaultLights')

        # Robot Frame
        self.robotBaseFrameHandle = self.sim.getObject('/RobotFrame')

        ## Convex Hulls for reward function 
 
        # Gripper Base
        self.gripperBaseConvexHull = self.sim.getObject('/GripperBase_convexHull')
        
        # Gripper
        self.gripperBase = self.sim.getObject("/Robotiq3F")
        self.GripperJointHandle = np.zeros(9) 
        self.GripperSensorHandles = np.zeros(9)
        self.GripperTipsConvexHullHandles = np.zeros(9)
        self.GripperFingersConvexHull = np.zeros(9)
        self.GripperFingersConvexHullCollection = self.sim.createCollection(0)

        self.GripperPalm = self.sim.getObject("/Robotiq3F/PalmTip")
        self.GripperPalmSensor = self.sim.getObject("/Robotiq3F/Palm_sensor")
        self.attachPoint = self.sim.getObject("/Robotiq3F/AttachPoint") 

        for i in range(9):
            finger_number = (i // 3) + 1
            link_number = (i % 3) + 1

            JointAlias = f'/Robotiq3F/finger_{finger_number}_joint_{link_number}'
            SensorAlias = f'/Robotiq3F/finger{finger_number}_link{link_number}_sensor'
            ConvexHullAlias = f'/Robotiq3F/finger{finger_number}_link{link_number}_sensor2'
            FingerConvexHullAlias = f'/Robotiq3F/finger_{finger_number}_link_{link_number}_convexHull'

            self.GripperJointHandle[i] = self.sim.getObject(JointAlias)
            self.GripperSensorHandles[i] = self.sim.getObject(SensorAlias)
            self.GripperTipsConvexHullHandles[i] = self.sim.getObject(ConvexHullAlias)
            _temp = self.sim.getObject(FingerConvexHullAlias)
            self.GripperFingersConvexHull[i] = _temp
            self.sim.addItemToCollection(self.GripperFingersConvexHullCollection, self.sim.handle_single, _temp, 0)

    def simStart(self):
        while self.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.startSimulation()

    def simStep(self):
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.step()

    def setSimStepping(self):
        self.sim.setStepping(True)

    def simStop(self):
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()
    
    def getSimTime(self):
        return self.sim.getSimulationTime()
    
    # function to import a random object from database to the scene,
    # output is random object's name and handle
    def ImportShape(self):
        random_object_index = random.randint(0,len(Dirlist)-1)
        random_object_path = Dirlist[random_object_index,0]
        random_object_name = Dirlist[random_object_index,1]
        random_object_handle = self.sim.loadModel(random_object_path)
        return random_object_name,random_object_handle
    
    # Remove Object from scene
    def removeObject(self, ObjectHandle):
        self.sim.removeObjects([ObjectHandle])

    # Place the object with given handle at the given position
    def PlaceObject(self,_position,_objectHandle,_relativeFrameHandle):
        self.sim.setObjectPosition(_objectHandle, _position,_relativeFrameHandle)

    # Getting depth image, object's length, and represenation points from vision sensor
    def getDepthImage(self, num_points):
        img, res = self.sim.getVisionSensorDepth(self.visionSensorHandle,1)
        width, height = res[0], res[1]
        DepthValues = array.array('f')
        DepthValues.frombytes(img)
        DepthValuesArray = np.array(DepthValues).reshape((height, width))
        ObjectLength = 1.02 - np.min(DepthValuesArray)
        self.objectLength = ObjectLength
        DepthValuesArray *= 255
        DepthValuesArray = DepthValuesArray.astype(np.uint8)
        UniformPoints = find_perimeter_and_uniform_points(DepthValuesArray, num_points = num_points)
        UniformPoints = ((UniformPoints - 128) / 512).astype(np.float32).flatten()
        return {'Image':DepthValuesArray, 'ObjectLength' : ObjectLength, 'Points' : UniformPoints}
    
    def getEEPos(self):
        _pose = self.sim.getObjectPosition(self.GripperPalmSensor, self.robotBaseFrameHandle)
        _orn = self.sim.getObjectOrientation(self.GripperPalm, self.robotBaseFrameHandle)
        _pose.append(_orn[2])
        return np.array(_pose)
    
    def getObjectPos(self, ObjectHandle):
        return np.array(self.sim.getObjectPosition(ObjectHandle, self.robotBaseFrameHandle))
    
    def getObjectOrn(self, ObjectHandle):
        return np.array(self.sim.getObjectOrientation(ObjectHandle, self.robotBaseFrameHandle))

    def getClosestDist(self, ObjectHandle):        
        _, _distanceData, _ = self.sim.checkDistance(ObjectHandle, self.GripperPalm, 0)
        _distX = _distanceData[0] - _distanceData[3]
        _distY = _distanceData[1] - _distanceData[4]
        _distZ = _distanceData[2] - _distanceData[5]
        return np.array([_distX,_distY,_distZ])

    def checkCollision(self, Handle1, Handle2):
        _res, _ = self.sim.checkCollision(Handle1, Handle2)
        return _res

    def gripperBaseCollision(self, ObjectHandle):
        """
        check if collision has happened between the gripper base with the object or floor
        The output is a list: [collision with object, collision with floor]
        """
        collisionWithObject = self.checkCollision(self.gripperBaseConvexHull, ObjectHandle)

        collisionWithFloor = self.checkCollision(self.gripperBaseConvexHull, self.planeHandle)

        return [collisionWithFloor, collisionWithObject]

    def gripperFingersCollision(self):
        """
        check if collision has happened between the gripper fingers and the floor.
        The output is the result.
        """
        collisionWithFloor = self.checkCollision(self.GripperFingersConvexHullCollection, self.planeHandle)

        return collisionWithFloor

    def gripperObjectCollision(self, initialPose, currentPose):
        """
        check if an undersired collision has happened between the gripper and the object
        and the object has fallen
        """
        delta = np.abs(currentPose - initialPose)
        for i in range(3):
            if delta[i] > objectTranslationalCollisionThreshold:
                return True

        for i in range(3,6):
            if delta[i] > objectRotationalCollisionThreshold:
                return True    
        
        return False

    def apply_action(self, action):
        _currentPose = self.getEEPos()
        _newPos = _currentPose[0:3] + action[0:3] * cm * translationalStep
        self.Position = list(_newPos)
        #_newOrn = _currentPose[3] + action[3] * rotStep * deg
        self.sim.setObjectPosition(self.gripperBase, list(_newPos), self.robotBaseFrameHandle)
        #self.sim.setObjectOrientation(self.gripperBase,[0,_newOrn,0],self.gripperBase)
        
    def checkPhase(self, objHandle):

        # Phase 1 is reaching
        # Phase 2 is grasping
        phase = 1
        gripPose = self.getEEPos()
        gripPoseXY = np.array([gripPose[0],gripPose[1]])
        objPose = self.getObjectPos(objHandle)
        objPoseXY = np.array([objPose[0],objPose[1]])
        dist_xy = np.linalg.norm(np.subtract(gripPoseXY,objPoseXY)) < graspThresholdH
        dist_z = np.abs(gripPose[2] - objPose[2]/2) < graspThresholdV
        _type_1_phaseCondition = dist_xy & dist_z
        _type_2_phaseCondition = self.objectGraspped(objHandle)
        if _type_2_phaseCondition: 
            phase = 2
        
        return phase
    

    def closeGripper(self,ObjectHandle):

        # Defining objects of Gripper State Machine Class GSM
        GSM1 = GSM(finger = 1)
        GSM2 = GSM(finger = 2)
        GSM3 = GSM(finger = 3)

        # Wrapping them into a dict for easy use
        GSMdict = {
        'Finger1': GSM1,
        'Finger2': GSM2,
        'Finger3': GSM3
        }

        # Retriveing final states 
        FinalState = GSM1.final_states
        FinalStateValues = []

        for i in range(len(FinalState)):
            FinalStateValues.append(FinalState[i].value)

        # Defining objects of Sensor Processor Calss
        SP1 = SP(self._readGripperSensor(1, ObjectHandle))
        SP2 = SP(self._readGripperSensor(2, ObjectHandle))
        SP3 = SP(self._readGripperSensor(3, ObjectHandle))

        # Wrapping them into a Dict for easy use
        SPdict = {
            'Finger1': SP1,
            'Finger2': SP2,
            'Finger3': SP3
        }

        _startTime = self.sim.getSimulationTime()

        # Sending close commond
        for i in range(3):
            Finger = f'Finger{i+1}'
            GSMdict[Finger].send("Close")
            self._moveGripperFinger(i+1,GSMdict[Finger].action)
            self.simStep()
            t.sleep(1/Frequency)

        while (((self.sim.getSimulationTime() - _startTime) < maxGripperTime) or not (
            (GSM1.current_state.value in FinalStateValues) &
            (GSM2.current_state.value in FinalStateValues) &
            (GSM3.current_state.value in FinalStateValues)) 
        ):
            self.simStep()
            t.sleep(1/Frequency)

            for i in range(3):
                Finger = f'Finger{i+1}'
                _curr_sensor_data = self._readGripperSensor(i+1, ObjectHandle)
                result, index = SPdict[Finger].process_sensor_data(_curr_sensor_data)
                if (result and GSMdict[Finger].current_state_value not in FinalStateValues):
                    for j in range(len(index)):
                        Event = f"Event{index[j]+1}"
                        try:
                            GSMdict[Finger].send(Event)
                            self._moveGripperFinger(i+1,GSMdict[Finger].action)
                            self.simStep()
                        except GSMdict[Finger].TransitionNotAllowed:
                            pass
            self.simStep()
            t.sleep(1/Frequency)
            if ((self.sim.getSimulationTime() - _startTime) > maxGripperTime):
                break
        
        for i in range(1,4):
            self._moveGripperFinger(i,"0000")
        self.simStep()

    def _readGripperSensor(self, Finger, ObjectHandle):
        i = (Finger - 1) * 3
        C1, _,_,_,_ = self.sim.checkProximitySensor(self.GripperSensorHandles[i], ObjectHandle)
        C2, _,_,_,_ = self.sim.checkProximitySensor(self.GripperSensorHandles[i+1], ObjectHandle)
        C3, _,_,_,_ = self.sim.checkProximitySensor(self.GripperSensorHandles[i+2], ObjectHandle)
        L1 = (np.abs(self.sim.getJointPosition(self.GripperJointHandle[i]) * rad - GripperJointLimits[0]) <= tol)
        L2 = (np.abs(self.sim.getJointPosition(self.GripperJointHandle[i+1]) * rad - GripperJointLimits[1]) <= tol)
        _theta3 = self.sim.getJointPosition(self.GripperJointHandle[i+2]) * rad
        L3 = GripperJointLimits[2] - tol <= _theta3 <= GripperJointLimits[2] + tol
        L4 = GripperJointLimits[3] - tol <= _theta3 <= GripperJointLimits[3] + tol
        L1, L2, L3, L4 = int(L1), int(L2), int(L3), int(L4)
        return [C1, C2, C3, L1, L2, L3, L4]

    def _moveGripperFinger(self, FingerNumber, LinkVelocity):
        i = (FingerNumber - 1) * 3
        LinkVelocity = [int(bit) for bit in LinkVelocity]
        LinkVelocity[2] = -1 if LinkVelocity[3] == 1 else LinkVelocity[2]
        LinkVelocity.pop()
        LinkHandels = [self.GripperJointHandle[i], self.GripperJointHandle[i+1], self.GripperJointHandle[i+2]]
        self.sim.setJointTargetVelocity(LinkHandels[0], LinkVelocity[0] * deg * 30)
        self.sim.setJointTargetVelocity(LinkHandels[1], LinkVelocity[1] * deg * 60)
        self.sim.setJointTargetVelocity(LinkHandels[2], LinkVelocity[2] * deg * 30)

    def objectGraspped(self, ObjectHandle):

        res, _, _, _, _= self.sim.checkProximitySensor(self.GripperPalmSensor, ObjectHandle)

        return res
    
    def graspScore(self, ObjectHandle, ObjectLength):

        objectCenter = self.getObjectPos(ObjectHandle)

        contactPoints = np.array([])

        for i in range(len(self.GripperTipsConvexHullHandles)):

            res = self.checkCollision(self.GripperFingersConvexHull[i], ObjectHandle)

            if res:
                contactPoints = np.append(contactPoints, self.sim.getObjectPosition(self.GripperFingersConvexHull[i], self.robotBaseFrameHandle))

        centerOfContactPoints = np.mean(contactPoints, axis=0)

        dist = np.linalg.norm(centerOfContactPoints - objectCenter)

        if (0 < dist < ObjectLength/2):

            score = -2 * dist/ObjectLength + 1

        else:
            score = 0

        return score
