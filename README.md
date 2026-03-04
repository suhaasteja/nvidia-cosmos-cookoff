# CosmosCleanerBot 🤖

**An intelligent autonomous robot cleaner powered by NVIDIA Cosmos AI and Isaac Sim**

CosmosCleanerBot is a complete robotics simulation project that combines NVIDIA's Cosmos vision-language models with Isaac Sim physics simulation to create an intelligent autonomous cleaning robot. The project demonstrates real-world scene reconstruction, AI-powered navigation, and interactive robot control through a web-based dashboard.

---

## 🎯 Project Overview

This project showcases three major capabilities:

1. **AI-Powered Vision & Reasoning**: Uses NVIDIA Cosmos Reason2 VLM to analyze camera feeds and make navigation decisions
2. **3D Scene Reconstruction**: Converts real-world video into photorealistic 3D environments using 3D Gaussian Splatting
3. **Interactive Simulation**: Full physics-based robot simulation in NVIDIA Isaac Sim with web-based control

### Key Features

- 🧠 **Vision-Language Navigation**: Real-time obstacle detection and path planning using Cosmos Reason2
- 🎥 **Video-to-3D Pipeline**: Transform video footage into interactive 3D scenes with COLMAP + 3DGUT
- 🎮 **Web Dashboard**: Real-time telemetry, camera feed, and mission control interface
- 🤖 **Differential Drive Robot**: Accurate physics simulation with camera sensors
- 📹 **Video Recording**: Capture and download robot camera footage
- 🗺️ **Waypoint Navigation**: Interactive map-based mission planning

---

## 🏗️ Architecture

```
CosmosCleanerBot/
├── Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets/  # Isaac Sim robot assets & control
│   ├── CosmosCleanerBot_Camera.usd              # Robot USD with camera
│   ├── launch_scene.py                          # Main Isaac Sim application
│   ├── dashboard.py                             # Streamlit dashboard (alternative)
│   └── templates/index.html                     # Flask web interface
│
├── jkchoi/                                      # Cosmos Reason2 API integration
│   ├── scripts/
│   │   ├── app.py                               # FastAPI bridge for Cosmos VLM
│   │   ├── cosmos_openai.py                     # OpenAI-compatible client
│   │   └── cosmos_housing.py                    # Vision reasoning examples
│   └── commands/                                # Docker & server commands
│
├── video_to_3dgrut.ipynb                        # Complete video-to-3D workflow
├── README_3DGUT.md                              # 3D reconstruction documentation
└── runs/                                        # Training outputs & models
```

---

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with 16GB+ VRAM (tested on L40S 46GB) running on Brev instance
- **Software**: 
  - NVIDIA Isaac Sim 4.0+
  - Docker (for Cosmos Reason2 NIM)
  - Python 3.10+
  - Conda

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/yourusername/nvidia-cosmos-cookoff.git
cd nvidia-cosmos-cookoff
```

### 2. Set Up Isaac Sim Robot

```bash
cd Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets

# Install dependencies
pip install flask opencv-python pillow numpy

# Launch the robot simulation with web interface
python launch_scene.py
```

The web dashboard will be available at `http://localhost:5000`

### 3. Set Up Cosmos Reason2 (Optional - for AI Navigation)

```bash
# Pull and run the Cosmos Reason2 NIM container
docker run -d --gpus all \
  -p 8000:8000 \
  nvcr.io/nvidia/cosmos-reason2:latest

# Set up the API bridge
cd jkchoi/scripts
export BRIDGE_API_KEY="your-secret-key"
export NIM_BASE_URL="http://127.0.0.1:8000/v1"

# Run the bridge server
uvicorn app:app --host 0.0.0.0 --port 8080
```

### 4. Video-to-3D Reconstruction (Optional)

See [`README_3DGUT.md`](README_3DGUT.md) for the complete workflow or run the Jupyter notebook:

```bash
jupyter notebook video_to_3dgrut.ipynb
```

---

## 📖 Usage Guide

### Web Dashboard Controls

The Flask-based web interface provides:

- **🕹️ Manual Control**: D-pad or keyboard (WASD/Arrow keys) for direct robot control
- **🗺️ Waypoint Navigation**: Click on the map to set waypoints, then start autonomous mission
- **📷 Live Camera Feed**: Real-time RGB camera view from robot
- **📊 Telemetry**: Position, velocity, orientation, and odometry data
- **📹 Recording**: Start/stop video recording and download footage
- **⚙️ Speed Settings**: Adjust navigation speed and angular gain

### Keyboard Shortcuts

- `W` / `↑` - Move forward
- `S` / `↓` - Move backward  
- `A` / `←` - Turn left
- `D` / `→` - Turn right
- `Space` - Emergency stop

### Mission Planning

1. **Set Waypoints**: Click on the map or use "Add Current Position"
2. **Start Mission**: Click "Start Mission" to begin autonomous navigation
3. **Monitor Progress**: Watch telemetry and camera feed
4. **Stop/Clear**: Use "Stop Mission" or "Clear All" as needed

---

## 🧠 AI Navigation with Cosmos Reason2

The project includes integration with NVIDIA Cosmos Reason2 vision-language model for intelligent navigation:

### How It Works

1. **Camera Capture**: Robot's RGB camera captures forward view every N frames
2. **Vision Analysis**: Image sent to Cosmos Reason2 with odometry data
3. **Reasoning**: Model analyzes scene and detects obstacles
4. **Action Decision**: Returns navigation command (move forward, turn, stop)
5. **Execution**: Robot controller applies the command

### API Bridge Architecture

```
Robot Camera → JPEG Frame → FastAPI Bridge → Cosmos Reason2 NIM
                                ↓
                          JSON Response
                                ↓
                    {"reasoning": "...", "action": {...}}
                                ↓
                        Robot Controller
```

### Enabling AI Navigation

Uncomment the LLM navigation code in `launch_scene.py` and configure:

```python
LLM_ENABLED  = True
LLM_API_URL  = "http://localhost:8080/v1/reason2/action"
LLM_API_KEY  = "your-secret-key"
```

---

## 🎬 Video-to-3D Reconstruction Pipeline

Transform real-world videos into interactive 3D scenes for robot simulation.

### Workflow

1. **Video Extraction**: Extract frames at 5 fps with FFmpeg
2. **COLMAP Reconstruction**: Structure-from-Motion camera pose estimation
3. **3DGUT Training**: 3D Gaussian Splatting model training
4. **USD Export**: Export to Universal Scene Description for Isaac Sim

### Example Command

```bash
# Extract frames
ffmpeg -i interior_video.mp4 -vf fps=5 -qscale:v 1 -qmin 1 frames/frame_%04d.jpg

# Run COLMAP (automated in notebook)
colmap feature_extractor --database_path colmap/database.db --image_path frames/

# Train 3DGUT
conda activate 3dgrut
python train.py --config-name apps/colmap_3dgut.yaml path=colmap/ out_dir=runs/
```

See [`README_3DGUT.md`](README_3DGUT.md) for detailed instructions.

---

## 🔧 Technical Details

### Robot Specifications

- **Type**: Differential drive wheeled robot
- **Wheel Base**: 0.48m
- **Wheel Radius**: 0.10m
- **Camera**: OmniVision OV9782 RGB (1280x800 @ 20Hz)
- **DOF Names**: `Revolute_left`, `Revolute_right`

### Control System

- **Controller**: `DifferentialController` from Isaac Sim
- **Update Rate**: 60 Hz physics simulation
- **Navigation Speed**: 0.5 m/s (configurable)
- **Angular Gain**: 0.6 (configurable)

### Web Server

- **Framework**: Flask
- **Endpoints**: 15+ REST API endpoints
- **WebSocket**: Real-time telemetry updates
- **Video Encoding**: JPEG @ 80 quality (configurable)

---

## 📚 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Simulation** | NVIDIA Isaac Sim | Physics-based robot simulation |
| **AI Vision** | NVIDIA Cosmos Reason2 | Vision-language reasoning |
| **3D Reconstruction** | 3DGUT + COLMAP | Video-to-3D scene conversion |
| **Robot Control** | Differential Controller | Wheeled robot kinematics |
| **Web Interface** | Flask + HTML5 Canvas | Real-time dashboard |
| **API Bridge** | FastAPI | Cosmos VLM integration |
| **Video Processing** | FFmpeg + OpenCV | Frame extraction & recording |

---

## 📁 Project Components

### 1. Isaac Sim Robot Simulation

**Location**: `Nvidia-Cosmos-Cookoff-CosmosCleaner-Assets/`

- `launch_scene.py` - Main simulation application with Flask server
- `CosmosCleanerBot_Camera.usd` - Robot asset with camera sensor
- `templates/index.html` - Web dashboard UI
- `dashboard.py` - Alternative Streamlit dashboard

### 2. Cosmos Reason2 Integration

**Location**: `jkchoi/scripts/`

- `app.py` - FastAPI bridge for vision-language reasoning
- `cosmos_openai.py` - OpenAI-compatible client example
- `validate_api_key.py` - API key validation utilities

### 3. 3D Reconstruction

**Location**: Root directory

- `video_to_3dgrut.ipynb` - Complete video-to-3D workflow
- `README_3DGUT.md` - Detailed reconstruction documentation

---

## 🎓 Learning Resources

- [NVIDIA Isaac Sim Documentation](https://developer.nvidia.com/isaac/sim)
- [Cosmos Reason2 Model Card](https://docs.nvidia.com/cosmos/latest/reason2/index.html)
- [3DGUT reference 1](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/)
- [3DGUT reference 2](https://github.com/nv-tlabs/3dgrut)
- [COLMAP Documentation](https://colmap.github.io/)
- [Differential Drive Kinematics](https://en.wikipedia.org/wiki/Differential_wheeled_robot)

---

## 🐛 Troubleshooting

### Robot Not Moving

- Check wheel DOF names match USD file: `Revolute_left`, `Revolute_right`
- Verify `wheel_base` and `wheel_radius` parameters
- Check for joint axis inversions (right wheel may need negation)

### Camera Not Showing

- Ensure camera prim exists: `Camera_OmniVision_OV9782_Color`
- Wait for 30 warmup frames after world reset
- Check camera is initialized: `camera.initialize()` and `camera.add_rgb_to_frame()`

### Web Dashboard Not Loading

- Check Flask server started successfully (ports 5000-5002)
- Verify firewall allows local connections
- Check browser console for JavaScript errors

### Cosmos Reason2 Errors

- Ensure NIM container is running: `docker ps`
- Verify API key is set correctly
- Check image size < 2MB (configurable in `app.py`)
- Validate image format (JPEG or PNG only)

---

## 🤝 Contributing

This project was created for the NVIDIA Cosmos Cookoff. Contributions and improvements are welcome!

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/nvidia-cosmos-cookoff.git

# Install development dependencies
pip install -r requirements-dev.txt  # if available

# Run tests
pytest tests/  # if available
```

---

## 📄 License

See individual component licenses:

- **Isaac Sim**: [NVIDIA Omniverse License](https://www.nvidia.com/en-us/omniverse/download/)
- **Cosmos Reason2**: [NVIDIA AI Foundation Models License](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
- **3DGUT**: Check [repository license](https://github.com/nv-tlabs/3dgrut)
- **COLMAP**: BSD License

---

## 🙏 Acknowledgments

- NVIDIA for Isaac Sim and Cosmos AI models
- NVIDIA TLabs for 3DGUT implementation
- COLMAP team for structure-from-motion tools
- The robotics and computer vision communities

---

## 📞 Contact

For questions or issues related to this project, please open an issue on GitHub.

**Built with ❤️ for the NVIDIA Cosmos Cookoff**
