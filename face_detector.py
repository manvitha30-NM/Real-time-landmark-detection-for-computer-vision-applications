

import cv2
import mediapipe as mp
import numpy as np
from config.settings import DetectionConfig, DisplayConfig
from src.utils.geometry import calculate_aspect_ratio, calculate_angle, calculate_distance

class FaceDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=DetectionConfig.FACE_MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=DetectionConfig.FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=DetectionConfig.FACE_MIN_TRACKING_CONFIDENCE
        )

        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

        self.NOSE_TIP = 1
        self.CHIN = 175
        self.LEFT_EYE_CORNER = 33
        self.RIGHT_EYE_CORNER = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291

        self.blink_counter = 0
        self.is_blinking = False
        
    def draw_custom_face_mesh(self, image, face_landmarks):
        
        landmarks = face_landmarks.landmark
        h, w = image.shape[:2]

        mesh_drawing_spec = self.mp_drawing.DrawingSpec(
            color=DisplayConfig.COLOR_FACE, 
            thickness=DisplayConfig.FACE_MESH_THICKNESS,
            circle_radius=DisplayConfig.FACE_MESH_LANDMARK_RADIUS
        )

        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mesh_drawing_spec,
            connection_drawing_spec=mesh_drawing_spec
        )

        eye_drawing_spec = self.mp_drawing.DrawingSpec(
            color=DisplayConfig.EYE_CONTOUR_COLOR,
            thickness=DisplayConfig.EYE_CONTOUR_THICKNESS,
            circle_radius=1
        )
        
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=eye_drawing_spec
        )
        
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=eye_drawing_spec
        )

        lip_drawing_spec = self.mp_drawing.DrawingSpec(
            color=DisplayConfig.COLOR_FACE,
            thickness=1,
            circle_radius=1
        )
        
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=lip_drawing_spec
        )

        eyebrow_drawing_spec = self.mp_drawing.DrawingSpec(
            color=DisplayConfig.COLOR_FACE,
            thickness=1,
            circle_radius=1
        )
        
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=eyebrow_drawing_spec
        )
        
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=eyebrow_drawing_spec
        )
        
    def detect(self, frame):
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results
    
    def detect_blink(self, face_landmarks):
        
        landmarks = face_landmarks.landmark
        import math

        LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  
        RIGHT_EYE_EAR = [362, 387, 386, 263, 373, 380]  

        try:
            
            p1 = (landmarks[33].x, landmarks[33].y)   
            p2 = (landmarks[160].x, landmarks[160].y) 
            p3 = (landmarks[158].x, landmarks[158].y) 
            p4 = (landmarks[133].x, landmarks[133].y) 
            p5 = (landmarks[153].x, landmarks[153].y) 
            p6 = (landmarks[144].x, landmarks[144].y) 

            vertical1 = math.dist(p2, p6)  
            vertical2 = math.dist(p3, p5)  
            horizontal = math.dist(p1, p4)  

            left_ear = (vertical1 + vertical2) / (2.0 * horizontal)
        except:
            left_ear = 0.3

        try:
            
            p1 = (landmarks[362].x, landmarks[362].y)   
            p2 = (landmarks[387].x, landmarks[387].y) 
            p3 = (landmarks[386].x, landmarks[386].y) 
            p4 = (landmarks[263].x, landmarks[263].y) 
            p5 = (landmarks[373].x, landmarks[373].y) 
            p6 = (landmarks[380].x, landmarks[380].y) 

            vertical1 = math.dist(p2, p6)  
            vertical2 = math.dist(p3, p5)  
            horizontal = math.dist(p1, p4)  

            right_ear = (vertical1 + vertical2) / (2.0 * horizontal)
        except:
            right_ear = 0.3

        ear = (left_ear + right_ear) / 2.0

        if ear < 0.25:
            self.is_blinking = True
        else:
            self.is_blinking = False

        return self.is_blinking, ear
    
    def get_eye_state_text(self, left_closed, right_closed):
        
        if left_closed and right_closed:
            return "Both Eyes Closed"
        elif left_closed and not right_closed:
            return "Left Closed, Right Open"
        elif not left_closed and right_closed:
            return "Left Open, Right Closed"
        else:
            return "Both Eyes Open"
    
    def get_eye_direction(self, face_landmarks):
        
        landmarks = face_landmarks.landmark
        
        try:
            
            left_ear = landmarks[234]   
            right_ear = landmarks[454]  
            nose_tip = landmarks[1]     

            face_width = right_ear.x - left_ear.x
            face_center_x = (left_ear.x + right_ear.x) / 2

            nose_offset_x = nose_tip.x - face_center_x
            horizontal_score = nose_offset_x / face_width

            horizontal_score *= 3

            if horizontal_score > 0.04:   
                return "Right Center"
            elif horizontal_score < -0.04:  
                return "Left Center"
            else:
                return "Center Center"
                
        except IndexError:
            return "Center Center"
    
    def get_head_direction(self, face_landmarks):
        
        landmarks = face_landmarks.landmark

        nose_tip = landmarks[self.NOSE_TIP]
        chin = landmarks[self.CHIN]
        left_eye = landmarks[self.LEFT_EYE_CORNER]
        right_eye = landmarks[self.RIGHT_EYE_CORNER]
        left_mouth = landmarks[self.LEFT_MOUTH]
        right_mouth = landmarks[self.RIGHT_MOUTH]

        face_center = type('Point', (), {
            'x': (left_eye.x + right_eye.x + left_mouth.x + right_mouth.x) / 4,
            'y': (left_eye.y + right_eye.y + left_mouth.y + right_mouth.y) / 4
        })()

        eye_center_x = (left_eye.x + right_eye.x) / 2
        mouth_center_x = (left_mouth.x + right_mouth.x) / 2

        if nose_tip.x < eye_center_x - DetectionConfig.HEAD_DIRECTION_THRESHOLD:
            horizontal = "Left"
        elif nose_tip.x > eye_center_x + DetectionConfig.HEAD_DIRECTION_THRESHOLD:
            horizontal = "Right"
        else:
            horizontal = "Center"

        if nose_tip.y < face_center.y - DetectionConfig.HEAD_DIRECTION_THRESHOLD:
            vertical = "Up"
        elif nose_tip.y > face_center.y + DetectionConfig.HEAD_DIRECTION_THRESHOLD:
            vertical = "Down"
        else:
            vertical = "Center"
        
        return f"{horizontal} {vertical}"
    
    def measure_eye_distance(self, face_landmarks):
        
        landmarks = face_landmarks.landmark
        left_eye = landmarks[self.LEFT_EYE_CORNER]
        right_eye = landmarks[self.RIGHT_EYE_CORNER]
        
        return calculate_distance(left_eye, right_eye)
    
    def measure_face_width(self, face_landmarks):
        
        landmarks = face_landmarks.landmark
        left_edge = landmarks[234]  
        right_edge = landmarks[454]  
        
        return calculate_distance(left_edge, right_edge)
    
    def draw_eye_gaze_visualization(self, image, face_landmarks):
        
        landmarks = face_landmarks.landmark
        h, w = image.shape[:2]

        left_eye_center = self._get_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_eye_center = self._get_eye_center(landmarks, self.RIGHT_EYE_INDICES)

        eye_direction = self.get_eye_direction(face_landmarks)

        left_pos = (int(left_eye_center.x * w), int(left_eye_center.y * h))
        right_pos = (int(right_eye_center.x * w), int(right_eye_center.y * h))

        arrow_length = 30
        arrow_color = (0, 255, 255)  
        
        if "Right" in eye_direction:
            
            cv2.arrowedLine(image, left_pos, (left_pos[0] + arrow_length, left_pos[1]), arrow_color, 2)
            cv2.arrowedLine(image, right_pos, (right_pos[0] + arrow_length, right_pos[1]), arrow_color, 2)
        elif "Left" in eye_direction:
            
            cv2.arrowedLine(image, left_pos, (left_pos[0] - arrow_length, left_pos[1]), arrow_color, 2)
            cv2.arrowedLine(image, right_pos, (right_pos[0] - arrow_length, right_pos[1]), arrow_color, 2)
        
        if "Up" in eye_direction:
            
            cv2.arrowedLine(image, left_pos, (left_pos[0], left_pos[1] - arrow_length), arrow_color, 2)
            cv2.arrowedLine(image, right_pos, (right_pos[0], right_pos[1] - arrow_length), arrow_color, 2)
        elif "Down" in eye_direction:
            
            cv2.arrowedLine(image, left_pos, (left_pos[0], left_pos[1] + arrow_length), arrow_color, 2)
            cv2.arrowedLine(image, right_pos, (right_pos[0], right_pos[1] + arrow_length), arrow_color, 2)
    
    def _get_eye_center(self, landmarks, eye_indices):
        
        x_coords = [landmarks[i].x for i in eye_indices[:6]]
        y_coords = [landmarks[i].y for i in eye_indices[:6]]
        
        return type('Point', (), {
            'x': sum(x_coords) / len(x_coords),
            'y': sum(y_coords) / len(y_coords)
        })()
    
    def _calculate_gaze_direction(self, eye_center, iris):
        
        gaze_x = iris.x - eye_center.x
        gaze_y = iris.y - eye_center.y
        
        return gaze_x, gaze_y
    
    def close(self):
        
        self.face_mesh.close()
