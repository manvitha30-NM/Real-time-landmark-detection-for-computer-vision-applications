

import cv2
import mediapipe as mp
from config.settings import DetectionConfig
from src.utils.geometry import calculate_distance

class HandDetector:

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=DetectionConfig.HAND_MAX_NUM_HANDS,
            min_detection_confidence=DetectionConfig.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=DetectionConfig.HAND_MIN_TRACKING_CONFIDENCE
        )

        self.finger_tip_ids = [4, 8, 12, 16, 20]
        self.finger_pip_ids = [3, 6, 10, 14, 18]
        
    def detect(self, frame):
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    
    def count_fingers(self, hand_landmarks, handedness):
        
        landmarks = hand_landmarks.landmark
        count = 0
        hand_label = handedness.classification[0].label

        if hand_label == "Right":
            if landmarks[4].x < landmarks[3].x:
                count += 1
        else:  
            if landmarks[4].x > landmarks[3].x:
                count += 1

        for i in range(1, 5):
            tip_id = self.finger_tip_ids[i]
            pip_id = self.finger_pip_ids[i]

            if landmarks[tip_id].y < landmarks[pip_id].y:
                count += 1
        
        return count
    
    def get_hand_center(self, hand_landmarks):
        
        landmarks = hand_landmarks.landmark

        wrist = landmarks[0]

        key_landmarks = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        x_coords = [lm.x for lm in key_landmarks]
        y_coords = [lm.y for lm in key_landmarks]
        
        center = type('Point', (), {
            'x': sum(x_coords) / len(x_coords),
            'y:': sum(y_coords) / len(y_coords)
        })()
        
        return center
    
    def measure_distance(self, landmark1, landmark2):
        
        return calculate_distance(landmark1, landmark2)
    
    def get_finger_positions(self, hand_landmarks):
        
        landmarks = hand_landmarks.landmark
        finger_positions = {}
        
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger_name in enumerate(finger_names):
            finger_positions[finger_name] = landmarks[self.finger_tip_ids[i]]
        
        return finger_positions
    
    def is_hand_open(self, hand_landmarks):
        
        landmarks = hand_landmarks.landmark

        for i in range(1, 5):
            tip_id = self.finger_tip_ids[i]
            pip_id = self.finger_pip_ids[i]
            
            if landmarks[tip_id].y > landmarks[pip_id].y:
                return False
        
        return True
    
    def close(self):
        
        self.hands.close()
