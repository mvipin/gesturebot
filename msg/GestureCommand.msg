# Navigation command derived from gesture recognition

std_msgs/Header header

# Command identification
string command_name           # e.g., "move_forward", "turn_left", "stop"
int32 command_id
float64 confidence

# Command parameters
geometry_msgs/Twist velocity_command
float64 duration             # seconds, 0 for continuous
int32 priority               # 0=emergency, 1=high, 2=normal

# Source information
string source_gesture        # original gesture name
string source_node          # node that generated command
bool requires_confirmation   # true for potentially dangerous commands
