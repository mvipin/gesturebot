# Performance metrics for vision processing nodes

std_msgs/Header header

# Feature identification
string feature_name
bool enabled
int32 priority

# Processing statistics
int64 frames_processed
float64 avg_processing_time    # milliseconds
float64 current_fps
float64 cpu_usage             # percentage
float64 memory_usage          # MB

# System health indicators
bool processing_healthy
string status_message
