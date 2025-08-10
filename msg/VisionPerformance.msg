# Vision pipeline performance metrics for optimization analysis

std_msgs/Header header

# Node identification
string feature_name

# Frame processing counts (for the measurement period)
int32 frames_processed_period     # Frames processed in this 5-second period
float32 current_fps              # Effective frames per second

# Pipeline stage timing (milliseconds, averaged over measurement period)
float32 avg_preprocessing_time   # ROS message → MediaPipe submission
float32 avg_mediapipe_time      # MediaPipe processing duration
float32 avg_postprocessing_time # MediaPipe callback → ROS publishing

# Total pipeline timing
float32 avg_total_pipeline_time # Sum of all pipeline stages
