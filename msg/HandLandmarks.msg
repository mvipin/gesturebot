# Hand landmark detection result

std_msgs/Header header

# Hand identification
string handedness              # "Left" or "Right"
float64 confidence

# Landmark points (21 landmarks per hand)
geometry_msgs/Point[] landmarks

# Hand properties
geometry_msgs/Point wrist_position
geometry_msgs/Vector3 palm_normal
float64 hand_size

# Gesture analysis
geometry_msgs/Vector3 pointing_direction
bool is_pointing
geometry_msgs/Point[5] fingertip_positions  # thumb, index, middle, ring, pinky
