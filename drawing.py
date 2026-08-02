

import cv2
import mediapipe as mp
from .geometry import get_landmark_coords

class DrawingUtils:

    def __init__(self, config):
        self.config = config
        
    def draw_landmarks(self, image, landmarks, connections=None, color=None, landmark_radius=None):
        
        if color is None:
            color = self.config.COLOR_HAND
            
        if landmark_radius is None:
            landmark_radius = self.config.LANDMARK_RADIUS
            
        h, w = image.shape[:2]

        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = get_landmark_coords(landmarks[start_idx], image.shape)
                    end_point = get_landmark_coords(landmarks[end_idx], image.shape)
                    cv2.line(image, start_point, end_point, color, self.config.CONNECTION_THICKNESS)

        for landmark in landmarks:
            point = get_landmark_coords(landmark, image.shape)
            cv2.circle(image, point, landmark_radius, color, -1)
    
    def draw_text(self, image, text, position, color=None, scale=None):
        
        if color is None:
            color = self.config.COLOR_TEXT
            
        if scale is None:
            scale = self.config.FONT_SCALE
            
        cv2.putText(image, text, position, self.config.FONT, scale, color, self.config.THICKNESS)
    
    def draw_finger_count(self, image, count, position, hand_label=""):
        
        text = f"{count} Fingers"
        if hand_label:
            text += f" ({hand_label})"
        self.draw_text(image, text, position, self.config.COLOR_HAND, 1.1)
    
    def draw_blink_indicator(self, image, is_blinking, position):
        
        color = (0, 0, 255) if is_blinking else (255, 255, 255)  
        text = "EYES CLOSED" if is_blinking else "EYES OPEN"
        self.draw_text(image, text, position, color, 0.8)
    
    def draw_head_direction(self, image, direction, position):
        
        direction_text = f"Head: {direction}"
        self.draw_text(image, direction_text, position, self.config.COLOR_FACE, 0.8)
    
    def draw_gesture(self, image, gesture, confidence, position):
        
        text = f"{gesture} ({confidence:.1f}%)"
        self.draw_text(image, text, position, self.config.COLOR_HAND, 0.9)
    
    def draw_distance_measurement(self, image, distance, point1, point2, label=""):

        cv2.line(image, point1, point2, (255, 255, 0), 2)

        mid_x = (point1[0] + point2[0]) // 2
        mid_y = (point1[1] + point2[1]) // 2
        
        text = f"{distance:.2f}"
        if label:
            text = f"{label}: {distance:.2f}"
            
        self.draw_text(image, text, (mid_x, mid_y - 10), (255, 255, 0), 0.6)
