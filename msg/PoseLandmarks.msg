# Pose landmark detection result

std_msgs/Header header

# Pose detection confidence
float64 confidence
bool is_present

# Landmark points (33 landmarks for full body pose)
geometry_msgs/Point[] landmarks

# Key body points
geometry_msgs/Point nose
geometry_msgs/Point left_shoulder
geometry_msgs/Point right_shoulder
geometry_msgs/Point left_elbow
geometry_msgs/Point right_elbow
geometry_msgs/Point left_wrist
geometry_msgs/Point right_wrist
geometry_msgs/Point left_hip
geometry_msgs/Point right_hip
geometry_msgs/Point left_knee
geometry_msgs/Point right_knee
geometry_msgs/Point left_ankle
geometry_msgs/Point right_ankle

# Pose analysis
geometry_msgs/Vector3 body_orientation
float64 body_height           # normalized height
geometry_msgs/Point center_of_mass
