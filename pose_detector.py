

import cv2
import mediapipe as mp
from config.settings import DetectionConfig, DisplayConfig
from src.utils.geometry import calculate_distance, calculate_angle

class PoseDetector:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            model_complexity=DetectionConfig.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=DetectionConfig.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=DetectionConfig.POSE_MIN_TRACKING_CONFIDENCE
        )

        self.SHOULDER_LEFT = 11
        self.SHOULDER_RIGHT = 12
        self.ELBOW_LEFT = 13
        self.ELBOW_RIGHT = 14
        self.WRIST_LEFT = 15
        self.WRIST_RIGHT = 16
        self.HIP_LEFT = 23
        self.HIP_RIGHT = 24
        self.KNEE_LEFT = 25
        self.KNEE_RIGHT = 26
        self.ANKLE_LEFT = 27
        self.ANKLE_RIGHT = 28
        
    def draw_custom_pose_mesh(self, image, pose_landmarks):
        
        landmarks = pose_landmarks.landmark
        h, w = image.shape[:2]

        body_connections = [
            
            ([0, 1, 2, 3, 7], DisplayConfig.POSE_HEAD_COLOR),  
            
            ([11, 12, 23, 24], DisplayConfig.POSE_BODY_COLOR),  
            ([11, 23], DisplayConfig.POSE_BODY_COLOR),  
            ([12, 24], DisplayConfig.POSE_BODY_COLOR),  
            
            ([11, 13, 15], DisplayConfig.POSE_ARM_COLOR),  
            ([12, 14, 16], DisplayConfig.POSE_ARM_COLOR),  
            
            ([23, 25, 27], DisplayConfig.POSE_LEG_COLOR),  
            ([24, 26, 28], DisplayConfig.POSE_LEG_COLOR),  
        ]

        for connection_group, color in body_connections:
            for i in range(len(connection_group) - 1):
                start_idx = connection_group[i]
                end_idx = connection_group[i + 1]
                
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    
                    cv2.line(image, start_point, end_point, color, DisplayConfig.POSE_MESH_THICKNESS)

        additional_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  
            (0, 4), (4, 5), (5, 6), (6, 8),  
            (9, 10),  
            (11, 12),  
            (11, 13), (13, 15),  
            (12, 14), (14, 16),  
            (11, 23), (12, 24),  
            (23, 24),  
            (23, 25), (25, 27),  
            (24, 26), (26, 28),  
        ]

        for start_idx, end_idx in additional_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                
                cv2.line(image, start_point, end_point, DisplayConfig.POSE_BODY_COLOR, 
                        DisplayConfig.POSE_MESH_THICKNESS)

        joint_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  
        
        for joint_idx in joint_indices:
            if joint_idx < len(landmarks):
                joint_point = (int(landmarks[joint_idx].x * w), int(landmarks[joint_idx].y * h))
                cv2.circle(image, joint_point, DisplayConfig.POSE_JOINT_RADIUS, 
                          DisplayConfig.POSE_JOINT_COLOR, -1)

        for i, landmark in enumerate(landmarks):
            if i not in joint_indices:
                point = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(image, point, DisplayConfig.POSE_LANDMARK_RADIUS, 
                          DisplayConfig.POSE_BODY_COLOR, -1)
        
    def detect(self, frame):
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    def measure_arm_length(self, pose_landmarks, side='left'):
        
        landmarks = pose_landmarks.landmark
        
        if side == 'left':
            shoulder = landmarks[self.SHOULDER_LEFT]
            elbow = landmarks[self.ELBOW_LEFT]
            wrist = landmarks[self.WRIST_LEFT]
        else:
            shoulder = landmarks[self.SHOULDER_RIGHT]
            elbow = landmarks[self.ELBOW_RIGHT]
            wrist = landmarks[self.WRIST_RIGHT]
        
        upper_arm = calculate_distance(shoulder, elbow)
        forearm = calculate_distance(elbow, wrist)
        
        return upper_arm + forearm
    
    def measure_leg_length(self, pose_landmarks, side='left'):
        
        landmarks = pose_landmarks.landmark
        
        if side == 'left':
            hip = landmarks[self.HIP_LEFT]
            knee = landmarks[self.KNEE_LEFT]
            ankle = landmarks[self.ANKLE_LEFT]
        else:
            hip = landmarks[self.HIP_RIGHT]
            knee = landmarks[self.KNEE_RIGHT]
            ankle = landmarks[self.ANKLE_RIGHT]
        
        thigh = calculate_distance(hip, knee)
        calf = calculate_distance(knee, ankle)
        
        return thigh + calf
    
    def measure_shoulder_width(self, pose_landmarks):
        
        landmarks = pose_landmarks.landmark
        left_shoulder = landmarks[self.SHOULDER_LEFT]
        right_shoulder = landmarks[self.SHOULDER_RIGHT]
        
        return calculate_distance(left_shoulder, right_shoulder)
    
    def measure_hip_width(self, pose_landmarks):
        
        landmarks = pose_landmarks.landmark
        left_hip = landmarks[self.HIP_LEFT]
        right_hip = landmarks[self.HIP_RIGHT]
        
        return calculate_distance(left_hip, right_hip)
    
    def get_body_height(self, pose_landmarks):
        
        landmarks = pose_landmarks.landmark

        nose = landmarks[0]  
        left_ankle = landmarks[self.ANKLE_LEFT]
        right_ankle = landmarks[self.ANKLE_RIGHT]

        if left_ankle.y > right_ankle.y:
            ankle = left_ankle
        else:
            ankle = right_ankle
        
        return calculate_distance(nose, ankle)
    
    def calculate_joint_angle(self, pose_landmarks, joint):
        
        landmarks = pose_landmarks.landmark
        
        if joint == 'left_elbow':
            angle = calculate_angle(
                landmarks[self.SHOULDER_LEFT],
                landmarks[self.ELBOW_LEFT],
                landmarks[self.WRIST_LEFT]
            )
        elif joint == 'right_elbow':
            angle = calculate_angle(
                landmarks[self.SHOULDER_RIGHT],
                landmarks[self.ELBOW_RIGHT],
                landmarks[self.WRIST_RIGHT]
            )
        elif joint == 'left_knee':
            angle = calculate_angle(
                landmarks[self.HIP_LEFT],
                landmarks[self.KNEE_LEFT],
                landmarks[self.ANKLE_LEFT]
            )
        elif joint == 'right_knee':
            angle = calculate_angle(
                landmarks[self.HIP_RIGHT],
                landmarks[self.KNEE_RIGHT],
                landmarks[self.ANKLE_RIGHT]
            )
        else:
            angle = 0
        
        return angle
    
    def is_arm_raised(self, pose_landmarks, side='left', threshold=0.1):
        
        landmarks = pose_landmarks.landmark
        
        if side == 'left':
            shoulder = landmarks[self.SHOULDER_LEFT]
            wrist = landmarks[self.WRIST_LEFT]
        else:
            shoulder = landmarks[self.SHOULDER_RIGHT]
            wrist = landmarks[self.WRIST_RIGHT]
        
        return wrist.y < shoulder.y - threshold
    
    def is_sitting(self, pose_landmarks, threshold=0.4):
        
        left_knee_angle = self.calculate_joint_angle(pose_landmarks, 'left_knee')
        right_knee_angle = self.calculate_joint_angle(pose_landmarks, 'right_knee')
        
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        return avg_knee_angle < threshold
    
    def get_posture_analysis(self, pose_landmarks):
        
        analysis = {}

        analysis['left_arm_raised'] = self.is_arm_raised(pose_landmarks, 'left')
        analysis['right_arm_raised'] = self.is_arm_raised(pose_landmarks, 'right')

        analysis['is_sitting'] = self.is_sitting(pose_landmarks)

        analysis['left_elbow_angle'] = self.calculate_joint_angle(pose_landmarks, 'left_elbow')
        analysis['right_elbow_angle'] = self.calculate_joint_angle(pose_landmarks, 'right_elbow')
        analysis['left_knee_angle'] = self.calculate_joint_angle(pose_landmarks, 'left_knee')
        analysis['right_knee_angle'] = self.calculate_joint_angle(pose_landmarks, 'right_knee')

        analysis['shoulder_width'] = self.measure_shoulder_width(pose_landmarks)
        analysis['hip_width'] = self.measure_hip_width(pose_landmarks)
        analysis['body_height'] = self.get_body_height(pose_landmarks)
        
        return analysis
    
    def close(self):
        
        self.pose.close()
