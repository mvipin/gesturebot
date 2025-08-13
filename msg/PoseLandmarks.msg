# Pose landmark detection results for multiple poses

std_msgs/Header header

# Timestamp from MediaPipe processing
int64 timestamp_ms

# Flattened array of all pose landmarks (33 points per pose)
# To get landmarks for pose N: landmarks[N*33:(N+1)*33]
geometry_msgs/Point[] landmarks

# Number of detected poses
int32 num_poses
