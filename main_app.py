

import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detectors.hand_detector import HandDetector
from src.detectors.face_detector import FaceDetector
from src.detectors.pose_detector import PoseDetector
from src.gestures.gesture_recognizer import GestureRecognizer
from src.utils.drawing import DrawingUtils
from config.settings import DetectionConfig, DisplayConfig, CameraConfig

class ComputerVisionApp:

    def __init__(self):
        
        self.hand_detector = HandDetector()
        self.face_detector = FaceDetector()
        self.pose_detector = PoseDetector()
        self.gesture_recognizer = GestureRecognizer()

        self.drawing_utils = DrawingUtils(DisplayConfig)

        self.cap = cv2.VideoCapture(CameraConfig.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CameraConfig.FPS)

        self.running = True
        self.show_hand_landmarks = False
        self.show_face_landmarks = False
        self.show_face_mask = False
        self.show_pose_landmarks = False
        self.show_measurements = False
        self.show_confidence_panel = False
        self.show_fps_indicator = False

        self.detection_status = {
            'face': False,
            'hands': False,
            'pose': False
        }

        self.active_warnings = []
        self.warning_cooldowns = {
            'face_not_visible': 0,
            'low_light': 0
        }

        self.face_detection_history = []
        self.face_detection_threshold = 5  

        self.fps_history = []
        self.max_fps_history = 30

        self.confidence_scores = {
            'face': 0.0,
            'hands': 0.0,
            'gesture': 0.0
        }
        self.confidence_history = {
            'face': [],
            'hands': [],
            'gesture': []
        }
        self.max_confidence_history = 10
        
    def run(self):
        
        print("Computer Vision Application Started")
        print("Press 'q' to quit")
        print("Press 'h' to toggle hand landmarks")
        print("Press 'f' to toggle face landmarks")
        print("Press 'm' to toggle face mask")
        print("Press 'p' to toggle pose landmarks")
        print("Press 'd' to toggle measurements")
        print("Press 'c' to toggle confidence panel")
        print("Press 'x' to toggle FPS indicator")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            if CameraConfig.MIRROR_IMAGE:
                frame = cv2.flip(frame, 1)

            head_direction, eye_status, eye_direction, posture = self.process_frame(frame)

            self.draw_ui(frame, head_direction, eye_status, eye_direction, posture)

            cv2.imshow("Real-Time Landmark Detection System", frame)

            self.handle_keyboard()
        
        self.cleanup()
    
    def process_frame(self, frame):
        
        h, w = frame.shape[:2]

        self.detection_status = {
            'face': False,
            'hands': False,
            'pose': False
        }

        self.update_warnings()

        head_direction = ""
        eye_status = ""
        eye_direction = ""
        posture = None

        hand_results = self.hand_detector.detect(frame)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            self.detection_status['hands'] = True
            self.process_hand_detection(frame, hand_results)
            
            self.update_confidence_score('hands', self.calculate_hand_confidence(hand_results))
        else:
            self.update_confidence_score('hands', 0.0)

        face_results = self.face_detector.detect(frame)
        face_detected = bool(face_results.multi_face_landmarks)

        self.face_detection_history.append(face_detected)
        if len(self.face_detection_history) > self.face_detection_threshold:
            self.face_detection_history.pop(0)

        face_consistently_missing = len(self.face_detection_history) >= self.face_detection_threshold and not any(self.face_detection_history)
        
        if face_detected:
            self.detection_status['face'] = True
            for face_landmarks in face_results.multi_face_landmarks:
                self.process_face_detection(frame, face_landmarks)

                is_blinking, ear = self.face_detector.detect_blink(face_landmarks)
                head_direction = self.face_detector.get_head_direction(face_landmarks)
                eye_status = "EYES CLOSED" if is_blinking else "EYES OPEN"
                eye_direction = self.face_detector.get_eye_direction(face_landmarks)  

                self.update_confidence_score('face', self.calculate_face_confidence(face_landmarks))
        else:
            self.update_confidence_score('face', 0.0)

            if face_consistently_missing and self.show_face_landmarks and self.warning_cooldowns['face_not_visible'] == 0:
                self.add_warning('Face not visible', 'warning')
                self.warning_cooldowns['face_not_visible'] = 30

        pose_results = self.pose_detector.detect(frame)
        if pose_results.pose_landmarks:
            self.detection_status['pose'] = True
            self.process_pose_detection(frame, pose_results)
            posture = self.pose_detector.get_posture_analysis(pose_results.pose_landmarks)

        self.check_lighting_conditions(frame)

        self.update_fps_tracking()

        return head_direction, eye_status, eye_direction, posture
    
    def process_hand_detection(self, frame, hand_results):
        
        total_fingers = 0
        hand_info = []
        
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):

            if self.show_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame, 
                    hand_landmarks.landmark,
                    connections=self.hand_detector.mp_hands.HAND_CONNECTIONS,
                    color=DisplayConfig.COLOR_HAND
                )

            finger_count = self.hand_detector.count_fingers(hand_landmarks, handedness)
            hand_label = handedness.classification[0].label

            total_fingers += finger_count

            hand_info.append(f"{hand_label}: {finger_count}")

            h, w = frame.shape[:2]
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h)

            self.drawing_utils.draw_finger_count(frame, finger_count, (x - 60, y - 40), hand_label)

            gesture, confidence = self.gesture_recognizer.recognize_gesture(hand_landmarks, handedness)
            if gesture and confidence > DetectionConfig.GESTURE_CONFIDENCE_THRESHOLD:
                self.drawing_utils.draw_gesture(frame, gesture, confidence * 100, (x - 60, y - 70))

            if self.show_measurements:
                self.draw_hand_measurements(frame, hand_landmarks)

        if total_fingers > 0:
            h, w = frame.shape[:2]
            total_text = f"Total: {total_fingers}/10 fingers"
            if len(hand_info) > 1:
                total_text += f" ({', '.join(hand_info)})"

            text_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 85  

            overlay = frame.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - 25), 
                         (text_x + text_size[0] + 10, text_y + 10), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, total_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def process_face_detection(self, frame, face_landmarks):

        if self.show_face_landmarks and self.show_face_mask:
            self.face_detector.draw_custom_face_mesh(frame, face_landmarks)

        if self.show_face_landmarks:
            self.face_detector.draw_eye_gaze_visualization(frame, face_landmarks)

        if self.show_measurements:
            self.draw_face_measurements(frame, face_landmarks)
    
    def process_pose_detection(self, frame, pose_results):

        if self.show_pose_landmarks:
            self.pose_detector.draw_custom_pose_mesh(frame, pose_results.pose_landmarks)

        if self.show_measurements:
            posture = self.pose_detector.get_posture_analysis(pose_results.pose_landmarks)
            self.draw_pose_measurements(frame, pose_results.pose_landmarks, posture)
    
    def draw_hand_measurements(self, frame, hand_landmarks):
        
        h, w = frame.shape[:2]

        finger_positions = self.hand_detector.get_finger_positions(hand_landmarks)

        thumb_index_dist = self.hand_detector.measure_distance(
            finger_positions['thumb'], 
            finger_positions['index']
        )

        thumb_pos = (int(finger_positions['thumb'].x * w), int(finger_positions['thumb'].y * h))
        index_pos = (int(finger_positions['index'].x * w), int(finger_positions['index'].y * h))
        
        self.drawing_utils.draw_distance_measurement(
            frame, thumb_index_dist, thumb_pos, index_pos, "Thumb-Index"
        )
    
    def draw_face_measurements(self, frame, face_landmarks):
        
        h, w = frame.shape[:2]

        eye_distance = self.face_detector.measure_eye_distance(face_landmarks)

        left_eye_pos = (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h))
        right_eye_pos = (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h))
        
        self.drawing_utils.draw_distance_measurement(
            frame, eye_distance, left_eye_pos, right_eye_pos, "Eye Distance"
        )
    
    def draw_pose_measurements(self, frame, pose_landmarks, posture):
        
        h, w = frame.shape[:2]

        left_shoulder = pose_landmarks.landmark[self.pose_detector.SHOULDER_LEFT]
        right_shoulder = pose_landmarks.landmark[self.pose_detector.SHOULDER_RIGHT]
        
        left_shoulder_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_pos = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        
        self.drawing_utils.draw_distance_measurement(
            frame, posture['shoulder_width'], left_shoulder_pos, right_shoulder_pos, "Shoulders"
        )
    
    def draw_ui(self, frame, head_direction, eye_status, eye_direction, posture):
        
        h, w = frame.shape[:2]

        margin = 20  
        panel_spacing = 20  

        positions = self.calculate_panel_positions(h, w, margin, panel_spacing)

        colors = {
            'cyan': (255, 255, 0),
            'yellow': (0, 255, 255),
            'light_blue': (255, 200, 100),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'purple': (255, 0, 255),
            'orange': (0, 165, 255),
            'white': (255, 255, 255),
            'panel_bg': (20, 20, 20),
            'dark_line': (40, 40, 40)
        }

        self.draw_title_panel(frame, margin, margin, colors)

        self.draw_active_modules_badge(frame, positions['active'], colors)

        if self.show_fps_indicator:
            self.draw_fps_optimization_indicator(frame, positions['fps'], colors)

        self.draw_detection_status_panel(frame, positions['status'], colors)

        self.draw_warning_messages(frame, positions['warnings'], colors)

        if posture:
            self.draw_posture_panel(frame, positions['posture'], colors, posture)

        if head_direction or eye_status:
            self.draw_face_status_panel(frame, positions['face_status'], colors, head_direction, eye_status, eye_direction)

        if self.show_confidence_panel:
            self.draw_confidence_panel(frame, positions['confidence'], colors)

        self.draw_controls_panel(frame, positions['controls'], colors)

        hint_text = "Press keys to toggle modules"
        hint_size = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        hint_x = (w - hint_size[0]) // 2
        cv2.putText(frame, hint_text, (hint_x, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (120, 120, 120), 1)
    
    def calculate_panel_positions(self, h, w, margin, spacing):
        
        positions = {}

        title_height = 40
        current_y = margin + title_height + spacing  

        positions['posture'] = {'x': margin, 'y': current_y}
        posture_height = 110
        current_y += posture_height + spacing

        positions['face_status'] = {'x': margin, 'y': current_y}
        face_status_height = 145
        current_y += face_status_height + spacing

        positions['confidence'] = {'x': margin, 'y': current_y}

        positions['warnings'] = {'x': w // 2 - 140, 'y': margin + 10}  

        positions['active'] = {'x': w - 200 - margin, 'y': margin}

        positions['fps'] = {'x': w - 140 - margin, 'y': margin + 45}

        positions['status'] = {'x': w - 180 - margin, 'y': margin + 105}

        positions['controls'] = {'x': w - 150 - margin, 'y': h - 170 - margin}
        
        return positions
    
    def draw_title_panel(self, frame, x, y, colors):
        
        title_text = "Real-Time Landmark Detection System"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        title_w = title_size[0] + 40
        title_h = 40

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + title_w, y + title_h), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x, y), (x + title_w, y + title_h), 
                     colors['purple'], 2)
        cv2.putText(frame, title_text, (x + 20, y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['purple'], 2)
    
    def draw_active_modules_badge(self, frame, pos, colors):
        
        features = []
        if self.show_hand_landmarks:
            features.append("Hands")
        if self.show_face_landmarks:
            features.append("Face")
        if self.show_pose_landmarks:
            features.append("Pose")
        
        if features:
            status_text = f"Active: {', '.join(features)}"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            badge_w = text_size[0] + 24
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (pos['x'], pos['y']), (pos['x'] + badge_w, pos['y'] + 32), 
                         colors['panel_bg'], -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (pos['x'], pos['y']), (pos['x'] + badge_w, pos['y'] + 3), 
                         colors['purple'], -1)
            cv2.rectangle(frame, (pos['x'], pos['y']), (pos['x'] + badge_w, pos['y'] + 32), 
                         colors['purple'], 1)
            cv2.putText(frame, status_text, (pos['x'] + 12, pos['y'] + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['purple'], 2)
    
    def draw_controls_panel(self, frame, pos, colors):

        controls = [
            ("q", "Quit"),
            ("h", "Hands"),
            ("f", "Face"),
            ("p", "Pose"),
            ("m", "Mask"),
            ("c", "Confidence"),
            ("x", "FPS")
        ]
        
        panel_w = 140
        panel_h = len(controls) * 22 + 30  

        overlay = frame.copy()
        cv2.rectangle(overlay, (pos['x'], pos['y']), 
                     (pos['x'] + panel_w, pos['y'] + panel_h), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (pos['x'], pos['y']), 
                     (pos['x'] + panel_w, pos['y'] + panel_h), 
                     colors['orange'], 1)

        cv2.putText(frame, "CONTROLS", (pos['x'] + 10, pos['y'] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['orange'], 1)

        line_y = pos['y'] + 40
        for key, action in controls:
            
            cv2.putText(frame, f"[{key}]", (pos['x'] + 10, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['orange'], 1)
            
            cv2.putText(frame, action, (pos['x'] + 45, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['white'], 1)
            line_y += 22  
    
    def draw_posture_panel(self, frame, pos, colors, posture):

        panel_width = 270  
        panel_height = 110

        overlay = frame.copy()
        cv2.rectangle(overlay, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.rectangle(frame, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['green'], 1)

        cv2.putText(frame, "POSTURE", (pos['x'] + 12, pos['y'] + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors['green'], 2)

        cv2.line(frame, (pos['x'] + 10, pos['y'] + 38), 
                (pos['x'] + panel_width - 10, pos['y'] + 38), colors['dark_line'], 1)

        line_y = pos['y'] + 60
        
        cv2.putText(frame, "Left Arm", (pos['x'] + 12, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['green'], 1)
        arm_status = "Lowered" if not posture['right_arm_raised'] else "Raised"
        cv2.putText(frame, arm_status, (pos['x'] + 95, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        
        line_y += 26
        
        cv2.putText(frame, "Right Arm", (pos['x'] + 12, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['green'], 1)
        arm_status = "Lowered" if not posture['left_arm_raised'] else "Raised"
        cv2.putText(frame, arm_status, (pos['x'] + 105, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
    
    def draw_face_status_panel(self, frame, pos, colors, head_direction, eye_status, eye_direction):

        panel_width = 270
        panel_height = 145

        overlay = frame.copy()
        cv2.rectangle(overlay, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.rectangle(frame, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['cyan'], 1)

        cv2.putText(frame, "FACE STATUS", (pos['x'] + 12, pos['y'] + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors['cyan'], 2)

        cv2.line(frame, (pos['x'] + 10, pos['y'] + 38), 
                (pos['x'] + panel_width - 10, pos['y'] + 38), colors['dark_line'], 1)

        line_y = pos['y'] + 58

        cv2.putText(frame, "Head", (pos['x'] + 12, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['cyan'], 1)
        cv2.putText(frame, head_direction, (pos['x'] + 75, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        
        line_y += 26

        cv2.putText(frame, "Eyes", (pos['x'] + 12, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['yellow'], 1)
        cv2.putText(frame, eye_status, (pos['x'] + 75, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
        
        line_y += 26

        cv2.putText(frame, "Gaze", (pos['x'] + 12, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['light_blue'], 1)
        cv2.putText(frame, eye_direction, (pos['x'] + 75, line_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)
    
    def handle_keyboard(self):
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.running = False
        elif key == ord('h'):
            self.show_hand_landmarks = not self.show_hand_landmarks
        elif key == ord('f'):
            self.show_face_landmarks = not self.show_face_landmarks
        elif key == ord('m'):
            self.show_face_mask = not self.show_face_mask
        elif key == ord('p'):
            self.show_pose_landmarks = not self.show_pose_landmarks
        elif key == ord('d'):
            self.show_measurements = not self.show_measurements
        elif key == ord('c'):
            self.show_confidence_panel = not self.show_confidence_panel
        elif key == ord('x'):
            self.show_fps_indicator = not self.show_fps_indicator
    
    def cleanup(self):
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nApplication closed")
    
    def update_fps_tracking(self):
        
        import time
        current_time = time.time()
        
        if not hasattr(self, 'last_frame_time'):
            self.last_frame_time = current_time
            return

        fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time

        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
    
    def get_fps_performance(self):
        
        if not self.fps_history:
            return 0, "Unknown", (255, 255, 255)  
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        if avg_fps >= 25:
            performance = "Smooth"
            color = (0, 255, 0)  
        elif avg_fps >= 15:
            performance = "Moderate"
            color = (0, 255, 255)  
        else:
            performance = "Lagging"
            color = (0, 0, 255)  
        
        return avg_fps, performance, color
    
    def calculate_face_confidence(self, face_landmarks):

        visible_landmarks = sum(1 for lm in face_landmarks.landmark if lm.z > 0)
        total_landmarks = len(face_landmarks.landmark)

        visibility_ratio = visible_landmarks / total_landmarks

        key_landmarks = [1, 4, 6, 10, 33, 263, 362, 386]  
        key_visible = sum(1 for idx in key_landmarks if face_landmarks.landmark[idx].z > 0)
        key_ratio = key_visible / len(key_landmarks)

        confidence = (visibility_ratio * 0.6) + (key_ratio * 0.4)
        return min(confidence * 100, 100.0)  
    
    def calculate_hand_confidence(self, hand_results):
        
        if not hand_results.multi_hand_landmarks:
            return 0.0
        
        total_confidence = 0.0
        hand_count = 0
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            
            visible_landmarks = sum(1 for lm in hand_landmarks.landmark if lm.z > 0)
            total_landmarks = len(hand_landmarks.landmark)
            visibility_ratio = visible_landmarks / total_landmarks

            finger_tips = [4, 8, 12, 16, 20]  
            tip_visible = sum(1 for idx in finger_tips if hand_landmarks.landmark[idx].z > 0)
            tip_ratio = tip_visible / len(finger_tips)

            hand_confidence = (visibility_ratio * 0.7) + (tip_ratio * 0.3)
            total_confidence += hand_confidence
            hand_count += 1
        
        avg_confidence = total_confidence / hand_count if hand_count > 0 else 0.0
        return min(avg_confidence * 100, 100.0)  
    
    def update_confidence_score(self, detection_type, confidence):

        self.confidence_history[detection_type].append(confidence)
        if len(self.confidence_history[detection_type]) > self.max_confidence_history:
            self.confidence_history[detection_type].pop(0)

        if self.confidence_history[detection_type]:
            self.confidence_scores[detection_type] = sum(self.confidence_history[detection_type]) / len(self.confidence_history[detection_type])
        else:
            self.confidence_scores[detection_type] = 0.0
    
    def draw_fps_optimization_indicator(self, frame, pos, colors):

        avg_fps, performance, perf_color = self.get_fps_performance()

        panel_width = 140
        panel_height = 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     perf_color, 1)

        fps_text = f"{avg_fps:.1f} FPS"
        cv2.putText(frame, fps_text, 
                   (pos['x'] + 10, pos['y'] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)

        cv2.putText(frame, performance, 
                   (pos['x'] + 10, pos['y'] + 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, perf_color, 1)
    
    def draw_confidence_panel(self, frame, pos, colors):

        panel_width = 270  
        panel_height = 80  

        overlay = frame.copy()
        cv2.rectangle(overlay, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['purple'], 1)

        cv2.putText(frame, "CONFIDENCE", 
                   (pos['x'] + 10, pos['y'] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['purple'], 1)

        confidence_text = f"Face: {self.confidence_scores['face']:.0f}%  Hands: {self.confidence_scores['hands']:.0f}%  Gesture: {self.confidence_scores['gesture']:.0f}%"

        avg_confidence = (self.confidence_scores['face'] + self.confidence_scores['hands'] + self.confidence_scores['gesture']) / 3
        if avg_confidence >= 75:
            color = colors['green']
        elif avg_confidence >= 50:
            color = colors['yellow']
        else:
            color = colors['red']
        
        cv2.putText(frame, confidence_text, 
                   (pos['x'] + 10, pos['y'] + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def update_warnings(self):

        for key in self.warning_cooldowns:
            if self.warning_cooldowns[key] > 0:
                self.warning_cooldowns[key] -= 1

        self.active_warnings = [w for w in self.active_warnings if w.get('age', 0) < 60]

        for warning in self.active_warnings:
            warning['age'] = warning.get('age', 0) + 1
    
    def add_warning(self, message, severity='warning'):

        for warning in self.active_warnings:
            if warning['message'] == message:
                warning['age'] = 0  
                return

        self.active_warnings.append({
            'message': message,
            'severity': severity,
            'age': 0
        })
    
    def check_lighting_conditions(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = gray.mean()

        if avg_brightness < 50 and self.warning_cooldowns['low_light'] == 0:
            self.add_warning('Low light detected', 'warning')
            self.warning_cooldowns['low_light'] = 60  

    def draw_detection_status_panel(self, frame, pos, colors):

        panel_width = 180
        panel_height = 100

        overlay = frame.copy()
        cv2.rectangle(overlay, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (pos['x'], pos['y']), 
                     (pos['x'] + panel_width, pos['y'] + panel_height), 
                     colors['purple'], 1)

        cv2.putText(frame, "DETECTION STATUS", 
                   (pos['x'] + 10, pos['y'] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['purple'], 1)

        status_items = [
            ('Face', self.detection_status['face']),
            ('Hands', self.detection_status['hands']),
            ('Pose', self.detection_status['pose'])
        ]
        
        for i, (name, is_active) in enumerate(status_items):
            y_pos = pos['y'] + 40 + (i * 18)

            circle_color = colors['green'] if is_active else colors['red']
            cv2.circle(frame, (pos['x'] + 15, y_pos), 4, circle_color, -1)

            status_text = f"{name}: {'Active' if is_active else 'Not Detected'}"
            text_color = colors['green'] if is_active else colors['red']
            cv2.putText(frame, status_text, 
                       (pos['x'] + 25, y_pos + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
    
    def draw_warning_messages(self, frame, pos, colors):
        
        if not self.active_warnings:
            return

        box_width = 280
        box_height = 40
        box_spacing = 10

        priority_order = ['Face not visible', 'Low light detected']
        sorted_warnings = sorted(self.active_warnings, 
                               key=lambda w: priority_order.index(w['message']) if w['message'] in priority_order else 999)
        
        for i, warning in enumerate(sorted_warnings):
            y_pos = pos['y'] + (i * (box_height + box_spacing))

            box_x = pos['x']

            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, y_pos), 
                         (box_x + box_width, y_pos + box_height), 
                         colors['panel_bg'], -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

            cv2.rectangle(frame, (box_x, y_pos), 
                         (box_x + box_width, y_pos + box_height), 
                         colors['yellow'], 2)

            cv2.putText(frame, warning['message'], 
                       (box_x + 15, y_pos + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'], 1)

if __name__ == "__main__":
    app = ComputerVisionApp()
    app.run()
