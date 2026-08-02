# Computer Vision Application

A comprehensive computer vision application featuring hand, face, and pose detection with advanced gesture recognition and measurement capabilities.

## Features

### Hand Detection
- ✅ Hand landmark detection
- ✅ Finger counting
- ✅ Hand gesture recognition
- ✅ Distance measurements between landmarks

### Face Detection
- ✅ Face landmark detection with mesh visualization
- ✅ Eye direction detection
- ✅ Blink detection
- ✅ Head direction estimation
- ✅ Eye distance measurement

### Pose Detection
- ✅ Full body pose landmark detection
- ✅ Body measurements (shoulder width, hip width, height)
- ✅ Joint angle calculations
- ✅ Posture analysis (sitting/standing detection)
- ✅ Arm position detection

### Gesture Recognition
- ✅ Basic gestures (fist, point, peace, open palm, thumbs up)
- ✅ Complex gestures (OK sign, rock sign)
- ✅ Two-handed gestures
- ✅ Face-hand interaction gestures

### Distance Measurements
- ✅ Landmark-to-landmark distance calculation
- ✅ Visual measurement display
- ✅ Multiple measurement types
- ✅ Thumb-index finger distance measurement
- ✅ Eye distance measurement
- ✅ Shoulder width measurement

### User Interface & Visualization
- ✅ Advanced UI panel system with multiple information panels
- ✅ Real-time FPS tracking and performance indicators
- ✅ Confidence scoring for all detection types
- ✅ Warning system for detection issues and lighting conditions
- ✅ Color-coded status indicators
- ✅ Interactive controls panel
- ✅ Active modules badge
- ✅ Detection status panel

### Performance & Optimization
- ✅ Real-time processing optimization
- ✅ Configurable confidence thresholds
- ✅ Efficient landmark processing
- ✅ Minimal CPU usage
- ✅ FPS performance classification (Smooth/Moderate/Lagging)
- ✅ Detection stability tracking
- ✅ Warning cooldown system

### Advanced Features
- ✅ Face mask visualization toggle
- ✅ Eye gaze direction arrows
- ✅ Posture analysis with arm position detection
- ✅ Joint angle calculations for elbows and knees
- ✅ Body height estimation
- ✅ Hip width measurement
- ✅ Sitting/standing posture detection
- ✅ Three-finger gesture recognition
- ✅ Gesture history smoothing for stability

## Project Structure

```
├── main_app.py              # Main application entry point
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration settings
├── src/
│   ├── __init__.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── hand_detector.py     # Hand detection and finger counting
│   │   ├── face_detector.py     # Face detection and eye tracking
│   │   └── pose_detector.py     # Pose detection and analysis
│   ├── gestures/
│   │   ├── __init__.py
│   │   └── gesture_recognizer.py # Gesture recognition system
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py          # Distance and angle calculations
│       └── drawing.py           # Visualization utilities
└── README.md
```

## Installation

1. Install required dependencies:
```bash
pip install opencv-python mediapipe numpy
```

2. Run the application:
```bash
python main_app.py
```

## Usage

### Keyboard Controls
- **q**: Quit application
- **h**: Toggle hand landmarks display
- **f**: Toggle face landmarks display
- **p**: Toggle pose landmarks display
- **m**: Toggle face mask visualization
- **d**: Toggle distance measurements
- **c**: Toggle confidence panel display
- **x**: Toggle FPS indicator display

### Features Display

The application displays comprehensive information through multiple UI panels:

#### Information Panels
- **Title Panel**: Application name and branding
- **Active Modules Badge**: Shows which detection modules are currently active
- **Detection Status Panel**: Real-time status of face, hands, and pose detection
- **Posture Panel**: Arm position analysis (raised/lowered status)
- **Face Status Panel**: Head direction, eye status (open/closed), and gaze direction
- **Confidence Panel**: Confidence scores for face, hands, and gesture detection
- **Controls Panel**: Interactive keyboard controls reference
- **Warning Messages**: Alerts for detection issues and environmental conditions

#### Real-time Features
- **Hand Features**: Finger count per hand, total finger count, recognized gestures with confidence scores, hand landmarks
- **Face Features**: Eye direction (Left/Right/Center), blink detection (Eyes Open/Closed), head direction, face mesh visualization, eye gaze arrows
- **Pose Features**: Body posture analysis, arm positions (raised/lowered), joint angles, body measurements
- **Measurements**: Visual distance measurements between landmarks when enabled
- **Performance Metrics**: FPS counter with performance classification (Smooth/Moderate/Lagging)
- **Warning System**: Low light detection, face not visible warnings

## Configuration

All settings are configurable in `config/settings.py`:

### Detection Parameters
- Hand detection confidence thresholds and maximum hands (2)
- Face detection confidence thresholds and maximum faces (1)
- Pose detection confidence thresholds and model complexity
- Blink detection eye aspect ratio thresholds
- Finger counting thresholds for thumb and finger detection
- Head direction detection sensitivity
- Gesture recognition confidence thresholds

### Display Settings
- Color schemes for different landmark types (hands, face, pose)
- Font settings, text sizes, and text display parameters
- Landmark visualization radii and connection thickness
- Face mesh opacity and contour colors
- Eye contour visualization settings
- Pose mesh colors for different body parts
- UI panel colors and transparency settings

### Camera Settings
- Camera ID selection (default: 0)
- Frame resolution (1280x720 default)
- Frame rate configuration (30 FPS default)
- Image mirroring for natural interaction

## Technical Details

### Hand Detection
- Uses MediaPipe Hands solution
- Supports up to 2 hands simultaneously
- 21 landmarks per hand with handedness detection
- Real-time finger counting with thumb-specific logic
- Hand center calculation and distance measurements
- Hand state detection (open/closed)

### Face Detection
- Uses MediaPipe Face Mesh solution with refined landmarks
- 468 facial landmarks for precise tracking
- Eye aspect ratio (EAR) calculation for blink detection
- Head pose estimation using key facial points (nose, eyes, mouth)
- Eye gaze direction detection with visual arrows
- Face mesh visualization with customizable opacity
- Eye contour highlighting and lip detection

### Pose Detection
- Uses MediaPipe Pose solution with configurable complexity
- 33 body landmarks covering full body
- Real-time joint angle calculations (elbows, knees)
- Posture analysis (sitting/standing detection)
- Arm position detection (raised/lowered)
- Body measurements (shoulder width, hip width, height)
- Custom pose mesh with color-coded body parts

### Gesture Recognition
- Rule-based gesture classification system
- Supported gestures: Fist, Point, Peace, Three, Open Palm, Thumbs Up, OK, Rock
- Gesture history smoothing for stability
- Confidence scoring for all recognized gestures
- Support for complex two-handed gestures
- Face-hand interaction detection
- Finger counting integration for gesture validation

### UI System
- Advanced panel-based layout system
- Real-time FPS tracking with performance classification
- Confidence score calculation and smoothing
- Warning system with cooldown management
- Color-coded status indicators
- Responsive panel positioning to avoid overlaps

## Performance

- Optimized for real-time processing at 30 FPS
- Configurable confidence thresholds for accuracy/speed balance
- Efficient landmark processing with minimal CPU usage
- FPS performance classification (Smooth: 25+ FPS, Moderate: 15-24 FPS, Lagging: <15 FPS)
- Detection stability tracking to reduce false positives
- Warning cooldown system to prevent message spam
- Smart panel positioning to avoid UI overlaps

## Troubleshooting

### Common Issues
1. **Camera not detected**: Check camera ID in settings (default is 0)
2. **Low performance**: Reduce frame rate, disable some features, or lower resolution
3. **Detection not working**: Adjust confidence thresholds in settings
4. **Face not visible warning**: Ensure proper lighting and face positioning
5. **Low light warning**: Improve lighting conditions or adjust brightness threshold
6. **Gesture recognition inaccurate**: Check hand visibility and reduce motion blur

### Performance Optimization Tips
- Disable unused detection modules using keyboard shortcuts
- Lower camera resolution if experiencing lag
- Adjust confidence thresholds for your specific environment
- Ensure adequate lighting for better detection accuracy
- Keep face and hands within camera frame for best results

### Known Limitations
- Maximum 2 hands detected simultaneously
- Single face detection at a time
- Performance depends on CPU capabilities and camera quality
- Detection accuracy varies with lighting conditions
- Fast movements may cause tracking instability

### Dependencies
- Python 3.7+
- OpenCV 4.x
- MediaPipe 0.10.x
- NumPy

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
