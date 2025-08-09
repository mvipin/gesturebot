#!/bin/bash
# GestureBot Vision Workspace Setup Script
# Sets up the camera_ws ROS 2 workspace with all dependencies and configurations.
# Updated for gesturebot package structure with vision_core and nodes organization.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Workspace directory (go up from scripts to workspace root)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo -e "${BLUE}Setting up GestureBot Vision workspace at: ${WORKSPACE_DIR}${NC}"
echo -e "${BLUE}Package directory: ${PACKAGE_DIR}${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if [[ $(uname -m) == "aarch64" ]] && [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model; then
        print_status "Running on Raspberry Pi - optimizing for Pi 5"
        export RPI_PLATFORM=true
    else
        print_warning "Not running on Raspberry Pi - some features may not work"
        export RPI_PLATFORM=false
    fi
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."

    # Check if ROS 2 Jazzy is already installed
    if dpkg -l | grep -q "ros-jazzy"; then
        print_status "ROS 2 Jazzy installation detected - skipping ROS 2 installation"
        ROS_DISTRO_DETECTED="jazzy"
    else
        print_warning "ROS 2 Jazzy not detected - installing ROS 2 packages"
        sudo apt update

        # ROS 2 Jazzy dependencies (only install if not present)
        sudo apt install -y \
            ros-jazzy-desktop \
            ros-jazzy-cv-bridge \
            ros-jazzy-image-transport \
            ros-jazzy-nav2-bringup \
            ros-jazzy-slam-toolbox \
            ros-jazzy-navigation2 \
            ros-jazzy-nav2-msgs

        ROS_DISTRO_DETECTED="jazzy"
    fi
    
    # Computer vision dependencies
    sudo apt install -y \
        python3-opencv \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-psutil \
        python3-pip
    
    # Camera dependencies (Pi specific)
    if [[ "$RPI_PLATFORM" == "true" ]]; then
        # Check if rpicam tools are available (may be built from source)
        if command -v rpicam-still >/dev/null 2>&1; then
            print_status "rpicam tools already available: $(which rpicam-still)"
        else
            print_warning "rpicam tools not found - attempting package installation"
            # Try different possible package names
            if apt list --installed 2>/dev/null | grep -q libcamera-tools; then
                print_status "libcamera-tools already installed"
            else
                sudo apt install -y libcamera-tools || print_warning "libcamera-tools installation failed"
            fi
        fi

        # Install v4l-utils (useful for camera debugging)
        sudo apt install -y v4l-utils || print_warning "v4l-utils installation failed"

        # Verify camera functionality
        if command -v rpicam-still >/dev/null 2>&1; then
            print_status "✅ Camera tools verified: rpicam-still available"
        else
            print_warning "⚠️  rpicam-still not found - camera capture may not work"
        fi
    fi
    
    print_status "System dependencies installed"
}

# Install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."

    # Detect if virtual environment is active
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_status "Virtual environment detected: $VIRTUAL_ENV"
        PIP_CMD="pip"
        INSTALL_FLAGS=""
    else
        print_status "Using system-wide installation with --user flag"
        PIP_CMD="pip3"
        INSTALL_FLAGS="--user"
    fi

    # MediaPipe
    $PIP_CMD install $INSTALL_FLAGS mediapipe

    # Additional Python packages
    $PIP_CMD install $INSTALL_FLAGS \
        opencv-python \
        numpy \
        scipy \
        matplotlib \
        psutil

    print_status "Python dependencies installed using $PIP_CMD $INSTALL_FLAGS"
}

# Create workspace directories
create_workspace_structure() {
    print_status "Creating workspace structure..."
    
    cd "$WORKSPACE_DIR"
    
    # Create standard ROS 2 workspace directories
    mkdir -p build install log
    
    # Create models directory for MediaPipe models
    mkdir -p src/gesturebot/models
    
    # Create results directory for test outputs
    mkdir -p src/gesturebot/test/results
    
    # Create additional directories for organized structure
    mkdir -p src/gesturebot/src/vision_core
    mkdir -p src/gesturebot/src/nodes
    mkdir -p src/gesturebot/config
    mkdir -p src/gesturebot/launch
    mkdir -p src/gesturebot/scripts
    
    print_status "Workspace structure created"
}

# Download MediaPipe models
download_models() {
    print_status "Downloading MediaPipe models..."
    
    MODELS_DIR="$WORKSPACE_DIR/src/gesturebot/models"
    
    # EfficientDet model for object detection
    if [[ ! -f "$MODELS_DIR/efficientdet.tflite" ]]; then
        print_status "Downloading EfficientDet model..."
        wget -q --show-progress \
            "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite" \
            -O "$MODELS_DIR/efficientdet.tflite"
    else
        print_status "EfficientDet model already exists"
    fi
    
    # Hand landmark model
    if [[ ! -f "$MODELS_DIR/hand_landmarker.task" ]]; then
        print_status "Downloading Hand Landmarker model..."
        wget -q --show-progress \
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
            -O "$MODELS_DIR/hand_landmarker.task"
    else
        print_status "Hand Landmarker model already exists"
    fi
    
    # Gesture recognizer model
    if [[ ! -f "$MODELS_DIR/gesture_recognizer.task" ]]; then
        print_status "Downloading Gesture Recognizer model..."
        wget -q --show-progress \
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" \
            -O "$MODELS_DIR/gesture_recognizer.task"
    else
        print_status "Gesture Recognizer model already exists"
    fi
    
    print_status "MediaPipe models downloaded"
}

# Build the workspace
build_workspace() {
    print_status "Building ROS 2 workspace..."

    cd "$WORKSPACE_DIR"

    # Source ROS 2 (detect distribution)
    if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
        source /opt/ros/jazzy/setup.bash
        print_status "Using ROS 2 Jazzy"
    elif [[ -f "/opt/ros/humble/setup.bash" ]]; then
        source /opt/ros/humble/setup.bash
        print_status "Using ROS 2 Humble"
    else
        print_error "No supported ROS 2 distribution found"
        exit 1
    fi

    # Create comprehensive .colcon_ignore to exclude virtual environment
    cat > .colcon_ignore << 'EOF'
# Exclude virtual environment and non-ROS directories
../gesturebot_env/
gesturebot_env/
.git/
.vscode/
__pycache__/
*.pyc
.pytest_cache/
node_modules/
.env/
venv/
env/
EOF

    # Also create one in parent directory
    cat > ../.colcon_ignore << 'EOF'
# Exclude virtual environment from colcon scanning
gesturebot_env/
.git/
.vscode/
EOF

    # Build the package with very restrictive paths
    print_status "Building gesturebot package..."

    # Use COLCON_IGNORE environment variable as additional protection
    export COLCON_IGNORE="gesturebot_env:../gesturebot_env"

    # Build only from the src directory with explicit exclusions
    colcon build \
        --packages-select gesturebot \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --packages-ignore-regex ".*gesturebot_env.*" \
        --base-paths ./src

    if [[ $? -eq 0 ]]; then
        print_status "Workspace built successfully"
    else
        print_error "Workspace build failed"
        exit 1
    fi
}

# Set up environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create setup script
    cat > "$WORKSPACE_DIR/setup_env.sh" << 'EOF'
#!/bin/bash
# GestureBot Vision Environment Setup

# Source ROS 2 (auto-detect distribution)
if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
    source /opt/ros/jazzy/setup.bash
elif [[ -f "/opt/ros/humble/setup.bash" ]]; then
    source /opt/ros/humble/setup.bash
else
    echo "Error: No supported ROS 2 distribution found"
    exit 1
fi

# Source workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/install/setup.bash"

# Set environment variables (updated paths)
export GESTUREBOT_WORKSPACE="$SCRIPT_DIR"
export GESTUREBOT_MODELS="$SCRIPT_DIR/src/gesturebot/models"
export GESTUREBOT_CONFIG="$SCRIPT_DIR/src/gesturebot/config"
export PYTHONPATH="$SCRIPT_DIR/src/gesturebot/src:$PYTHONPATH"

echo "GestureBot Vision environment loaded"
echo "Workspace: $SCRIPT_DIR"
echo "Models: $GESTUREBOT_MODELS"
echo "Config: $GESTUREBOT_CONFIG"
EOF
    
    chmod +x "$WORKSPACE_DIR/setup_env.sh"
    
    # Create convenient aliases (updated package name)
    cat > "$WORKSPACE_DIR/aliases.sh" << 'EOF'
#!/bin/bash
# GestureBot Vision Aliases

# Launch commands (updated package name)
alias launch_vision="ros2 launch gesturebot vision_system.launch.py"
alias launch_object_detection="ros2 launch gesturebot vision_system.launch.py enable_object_detection:=true enable_gesture_recognition:=false"
alias launch_gesture_recognition="ros2 launch gesturebot vision_system.launch.py enable_object_detection:=false enable_gesture_recognition:=true"

# Testing commands (updated package and test locations)
alias test_mediapipe="cd $GESTUREBOT_WORKSPACE/src/gesturebot/test && python3 test_mediapipe_models.py"
alias test_camera="cd $GESTUREBOT_WORKSPACE/src/gesturebot/test && python3 test_camera_integration.py"
alias test_vision_nodes="cd $GESTUREBOT_WORKSPACE/src/gesturebot/test && python3 test_vision_nodes.py"
alias test_all="cd $GESTUREBOT_WORKSPACE/src/gesturebot/test && python3 run_all_tests.py"
alias quick_test="cd $GESTUREBOT_WORKSPACE/src/gesturebot/test && python3 quick_test.py"

# Monitoring commands
alias monitor_performance="ros2 topic echo /vision/*/performance"
alias monitor_gestures="ros2 topic echo /vision/gestures"
alias monitor_objects="ros2 topic echo /vision/objects"

# Navigation commands (updated package name)
alias start_nav_bridge="ros2 run gesturebot gesture_navigation_bridge.py"

# Development commands
alias build_gesturebot="cd $GESTUREBOT_WORKSPACE && colcon build --packages-select gesturebot"
alias test_gesturebot="cd $GESTUREBOT_WORKSPACE && colcon test --packages-select gesturebot"

echo "GestureBot Vision aliases loaded"
EOF
    
    chmod +x "$WORKSPACE_DIR/aliases.sh"
    
    print_status "Environment setup complete"
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    cd "$WORKSPACE_DIR"
    source install/setup.bash
    
    # Test package installation (updated package name)
    if ros2 pkg list | grep -q gesturebot; then
        print_status "Package installed successfully"
    else
        print_error "Package not found in ROS 2"
        exit 1
    fi
    
    # Test Python imports (updated import paths)
    python3 -c "
import sys
sys.path.append('$WORKSPACE_DIR/src/gesturebot/src')
try:
    from vision_core.base_node import MediaPipeBaseNode
    from vision_core.message_converter import MessageConverter
    print('✅ Python package import successful')
except ImportError as e:
    print(f'❌ Python package import failed: {e}')
    exit(1)
" || {
        print_error "Python package import failed"
        exit 1
    }
    
    # Test MediaPipe
    python3 -c "import mediapipe as mp; print('✅ MediaPipe import successful')" || {
        print_error "MediaPipe import failed"
        exit 1
    }
    
    print_status "Installation test passed"
}

# Verify camera functionality
verify_camera_functionality() {
    print_status "Verifying camera functionality..."

    if [[ "$RPI_PLATFORM" == "true" ]]; then
        # Test rpicam-still functionality
        if command -v rpicam-still >/dev/null 2>&1; then
            print_status "Testing rpicam-still capture..."

            # Test basic camera capture (no preview, quick timeout)
            if rpicam-still --output /tmp/test_capture.jpg --timeout 1000 --nopreview >/dev/null 2>&1; then
                if [[ -f /tmp/test_capture.jpg ]]; then
                    print_status "✅ Camera capture test successful"
                    rm -f /tmp/test_capture.jpg
                else
                    print_warning "⚠️  Camera capture completed but no image file created"
                fi
            else
                print_warning "⚠️  Camera capture test failed - check camera connection"
                print_warning "    Try: rpicam-hello --list-cameras"
            fi
        else
            print_warning "⚠️  rpicam-still not available - camera functionality may be limited"
        fi

        # Check libcamera tools
        if command -v libcamera-hello >/dev/null 2>&1; then
            print_status "✅ libcamera tools available"
        fi

        # List available cameras
        if command -v rpicam-hello >/dev/null 2>&1; then
            print_status "Available cameras:"
            rpicam-hello --list-cameras 2>/dev/null || print_warning "No cameras detected"
        fi
    else
        print_status "Not on Raspberry Pi - skipping camera verification"
    fi
}

# Run comprehensive tests
run_comprehensive_tests() {
    print_status "Running comprehensive tests..."

    cd "$WORKSPACE_DIR/src/gesturebot/test"

    # Run quick test first
    print_status "Running quick validation test..."
    if python3 quick_test.py; then
        print_status "Quick test passed"
    else
        print_warning "Quick test failed - check system setup"
    fi

    # Run full test suite if quick test passes
    print_status "Running full test suite..."
    if python3 run_all_tests.py; then
        print_status "All tests passed"
    else
        print_warning "Some tests failed - check test results"
    fi
}

# Print usage instructions
print_usage() {
    echo ""
    echo -e "${BLUE}=== GestureBot Vision Setup Complete ===${NC}"
    echo ""
    echo "To use the system:"
    echo ""
    echo "1. Source the environment:"
    echo "   source $WORKSPACE_DIR/setup_env.sh"
    echo ""
    echo "2. Load convenient aliases:"
    echo "   source $WORKSPACE_DIR/aliases.sh"
    echo ""
    echo "3. Test system functionality:"
    echo "   quick_test"
    echo ""
    echo "4. Launch vision system:"
    echo "   launch_vision"
    echo ""
    echo "5. Monitor system:"
    echo "   monitor_performance"
    echo "   monitor_gestures"
    echo ""
    echo "6. Run comprehensive tests:"
    echo "   test_all"
    echo ""
    echo "Package structure:"
    echo "   Models: $WORKSPACE_DIR/src/gesturebot/models/"
    echo "   Source: $WORKSPACE_DIR/src/gesturebot/src/"
    echo "   Tests:  $WORKSPACE_DIR/src/gesturebot/test/"
    echo "   Config: $WORKSPACE_DIR/src/gesturebot/config/"
    echo ""
    echo "For more information, see:"
    echo "   $WORKSPACE_DIR/src/gesturebot/README.md"
    echo "   $WORKSPACE_DIR/src/gesturebot/test/README.md"
    echo ""
}

# Main setup function
main() {
    print_status "Starting GestureBot Vision workspace setup..."

    check_raspberry_pi
    install_system_dependencies
    install_python_dependencies
    create_workspace_structure
    download_models
    build_workspace
    setup_environment
    test_installation
    verify_camera_functionality

    # Optional: Run comprehensive tests
    if [[ "${1:-}" == "--with-tests" ]]; then
        run_comprehensive_tests
    fi

    print_usage

    print_status "Setup complete! 🚀"
}

# Run main function with all arguments
main "$@"
