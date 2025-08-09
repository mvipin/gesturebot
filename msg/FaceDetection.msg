# Face detection result

std_msgs/Header header

# Face identification
int32 face_id
float64 confidence
bool is_present

# Bounding box
int32 bbox_x
int32 bbox_y
int32 bbox_width
int32 bbox_height

# Face properties
geometry_msgs/Point face_center
float64 face_size             # normalized size
geometry_msgs/Vector3 face_orientation

# Key facial landmarks (if available)
geometry_msgs/Point left_eye
geometry_msgs/Point right_eye
geometry_msgs/Point nose_tip
geometry_msgs/Point mouth_center

# Face analysis
float64 estimated_age         # -1 if not available
string estimated_emotion      # empty if not available
float64 emotion_confidence
