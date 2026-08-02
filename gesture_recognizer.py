

import math
from config.settings import DetectionConfig
from src.utils.geometry import calculate_distance, calculate_angle

class GestureRecognizer:

    def __init__(self):
        self.gesture_history = []
        self.max_history = 5
        
    def recognize_gesture(self, hand_landmarks, handedness):
        
        landmarks = hand_landmarks.landmark
        finger_count = self._count_fingers(landmarks, handedness)
        
        gesture = None
        confidence = 0.0

        if finger_count == 0:
            gesture, confidence = self._check_fist(landmarks)
        elif finger_count == 1:
            gesture, confidence = self._check_point(landmarks)
        elif finger_count == 2:
            gesture, confidence = self._check_peace_or_victory(landmarks)
        elif finger_count == 3:
            gesture, confidence = self._check_three(landmarks)
        elif finger_count == 5:
            gesture, confidence = self._check_open_palm(landmarks)
        
        if not gesture:
            gesture, confidence = self._check_thumbs_up(landmarks)
        if not gesture:
            gesture, confidence = self._check_ok_sign(landmarks)
        if not gesture:
            gesture, confidence = self._check_rock_sign(landmarks)

        self._update_history(gesture)
        
        smoothed_gesture = self._smooth_gesture()
        
        return smoothed_gesture, confidence
    
    def _count_fingers(self, landmarks, handedness):
        count = 0
        hand_label = handedness.classification[0].label
        
        if hand_label == "Right":
            if landmarks[4].x < landmarks[3].x:
                count += 1
        else:
            if landmarks[4].x > landmarks[3].x:
                count += 1
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                count += 1
        
        return count
    
    def _check_fist(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        
        is_fist = True
        for tip, mcp in zip(finger_tips, finger_mcps):
            if landmarks[tip].y < landmarks[mcp].y:
                is_fist = False
                break
        
        thumb_curled = (landmarks[4].y > landmarks[3].y or  
                       calculate_distance(landmarks[4], landmarks[2]) < 0.08)  
        
        if is_fist and thumb_curled:
            return "Fist", 0.9
        return None, 0.0
    
    def _check_point(self, landmarks):
        if (landmarks[8].y < landmarks[6].y and  
            landmarks[12].y > landmarks[10].y and  
            landmarks[16].y > landmarks[14].y and  
            landmarks[20].y > landmarks[18].y):    
            
            return "Point", 0.8
        return None, 0.0
    
    def _check_peace_or_victory(self, landmarks):
        if (landmarks[8].y < landmarks[6].y and  
            landmarks[12].y < landmarks[10].y and  
            landmarks[16].y > landmarks[14].y and  
            landmarks[20].y > landmarks[18].y):    
            
            return "Peace", 0.9
        return None, 0.0
    
    def _check_three(self, landmarks):
        if (landmarks[8].y < landmarks[6].y and  
            landmarks[12].y < landmarks[10].y and  
            landmarks[16].y < landmarks[14].y and  
            landmarks[20].y > landmarks[18].y):    
            
            return "Three", 0.8
        return None, 0.0
    
    def _check_open_palm(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        is_open = True
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y > landmarks[pip].y:
                is_open = False
                break
        
        if is_open and landmarks[4].y < landmarks[3].y:
            return "Open Palm", 0.9
        return None, 0.0
    
    def _check_thumbs_up(self, landmarks):
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        thumb_extension = calculate_distance(thumb_tip, wrist)
        
        if (landmarks[4].y < landmarks[3].y and  
            thumb_extension > 0.15 and  
            landmarks[8].y > landmarks[6].y and  
            landmarks[12].y > landmarks[10].y and  
            landmarks[16].y > landmarks[14].y and  
            landmarks[20].y > landmarks[18].y):    
            
            return "Thumbs Up", 0.9
        return None, 0.0
    
    def _check_ok_sign(self, landmarks):
        thumb_index_dist = calculate_distance(landmarks[4], landmarks[8])
        
        if (thumb_index_dist < 0.05 and  
            landmarks[12].y < landmarks[10].y and  
            landmarks[16].y < landmarks[14].y and  
            landmarks[20].y < landmarks[18].y):    
            
            return "OK", 0.8
        return None, 0.0
    
    def _check_rock_sign(self, landmarks):
        if (landmarks[8].y < landmarks[6].y and  
            landmarks[12].y < landmarks[10].y and  
            landmarks[4].y > landmarks[3].y and  
            landmarks[20].y > landmarks[18].y):    
            
            return "Rock", 0.8
        return None, 0.0
    
    def _update_history(self, gesture):
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
    
    def _smooth_gesture(self):
        if not self.gesture_history:
            return None
        
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        if gesture_counts:
            return max(gesture_counts, key=gesture_counts.get)
        return None
    
    def recognize_complex_gesture(self, hand_landmarks_list, face_landmarks=None):
        if len(hand_landmarks_list) == 2:
            return self._check_two_handed_gestures(hand_landmarks_list)
        elif face_landmarks:
            return self._check_face_hand_gestures(hand_landmarks_list[0], face_landmarks)
        
        return None, 0.0
    
    def _check_two_handed_gestures(self, hand_landmarks_list):
        left_hand = hand_landmarks_list[0]
        right_hand = hand_landmarks_list[1]
        
        left_fingers = self._count_fingers(left_hand.landmark, 
                                          type('Handedness', (), {'classification': [type('Classification', (), {'label': 'Left'})()]})())
        right_fingers = self._count_fingers(right_hand.landmark,
                                           type('Handedness', (), {'classification': [type('Classification', (), {'label': 'Right'})()]})())
        
        if left_fingers == 1 and right_fingers == 1:
            return "Frame", 0.7
        elif left_fingers == 2 and right_fingers == 2:
            return "Two Peace", 0.7
        
        return None, 0.0
    
    def _check_face_hand_gestures(self, hand_landmarks, face_landmarks):
        hand_center = self._get_hand_center(hand_landmarks)
        face_center = self._get_face_center(face_landmarks)
        
        distance = calculate_distance(hand_center, face_center)
        
        if distance < 0.15:  
            return "Face Touch", 0.6
        
        return None, 0.0
    
    def _get_hand_center(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        key_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        
        x = sum(p.x for p in key_points) / len(key_points)
        y = sum(p.y for p in key_points) / len(key_points)
        z = sum(p.z for p in key_points) / len(key_points)
        
        return type('Point', (), {'x': x, 'y': y, 'z': z})()
    
    def _get_face_center(self, face_landmarks):
        landmarks = face_landmarks.landmark
        key_points = [landmarks[1], landmarks[33], landmarks[263], landmarks[13]]  
        
        x = sum(p.x for p in key_points) / len(key_points)
        y = sum(p.y for p in key_points) / len(key_points)
        z = sum(p.z for p in key_points) / len(key_points)
        
        return type('Point', (), {'x': x, 'y': y, 'z': z})()
