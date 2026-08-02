

import math
import numpy as np

def calculate_distance(point1, point2):
    
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def calculate_2d_distance(point1, point2):
    
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_angle(point1, point2, point3):

    v1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
    
    v2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return math.degrees(angle)

def calculate_aspect_ratio(landmarks, eye_indices):

    A = calculate_2d_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = calculate_2d_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])

    C = calculate_2d_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

    ear = (A + B) / (2.0 * C)
    
    return ear

def get_landmark_coords(landmark, frame_shape):
    
    h, w = frame_shape[:2]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return (x, y)

def calculate_center_point(landmarks):
    
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    z_coords = [lm.z for lm in landmarks]
    
    center = type('Point', (), {
        'x': sum(x_coords) / len(x_coords),
        'y': sum(y_coords) / len(y_coords),
        'z': sum(z_coords) / len(z_coords)
    })()
    
    return center
