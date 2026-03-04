# CosmosCleanerBot - Video to 3D Reconstruction

**Transform real-world videos into interactive 3D simulations using NVIDIA Cosmos and 3D Gaussian Splatting**

This project demonstrates a complete pipeline for converting video footage into high-quality 3D reconstructions that can be rendered in interactive simulations. The workflow uses COLMAP for camera pose estimation and NVIDIA's 3DGUT (3D Gaussian with Unscented Transforms) for scene reconstruction.

## 🎯 Project Overview

CosmosCleanerBot showcases how to:
- Extract frames from video footage
- Perform camera calibration and pose estimation with COLMAP
- Train 3D Gaussian Splatting models using 3DGUT
- Export to USD format for use in NVIDIA Isaac Sim
- Render photorealistic views of reconstructed scenes

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (tested on L40S with 46GB VRAM)
- Ubuntu 22.04 or similar Linux distribution
- Conda package manager
- FFmpeg for video processing
- COLMAP for structure-from-motion

### Environment Setup

This setup was tested on NVIDIA Brev (https://docs.nvidia.com/brev/latest/about-brev.html)

```bash
# Clone the 3DGUT repository
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# Install environment (creates Python 3.11 conda env + Jupyter kernel)
chmod +x install_env.sh
./install_env.sh 3dgrut

# Install COLMAP
sudo apt-get update && sudo apt-get install -y colmap

# Install Xvfb for headless COLMAP execution
sudo apt-get install -y xvfb
```

## 📋 Complete Workflow

### Step 1: Video to Images

Extract frames from your video at 5 fps with high quality:

```bash
ffmpeg -i input_video.mp4 -vf fps=5 -qscale:v 1 -qmin 1 output_folder/frame_%04d.jpg
```

### Step 2: COLMAP Reconstruction

Set up COLMAP workspace and run structure-from-motion:

```bash
# Create workspace
mkdir -p colmap_workspace/sparse

# Feature extraction
xvfb-run -a -s "-screen 0 1920x1080x24" colmap feature_extractor \
  --database_path colmap_workspace/database.db \
  --image_path path/to/frames \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --SiftExtraction.use_gpu 1

# Feature matching
xvfb-run -a -s "-screen 0 1920x1080x24" colmap exhaustive_matcher \
  --database_path colmap_workspace/database.db \
  --SiftMatching.use_gpu 1

# Sparse reconstruction
xvfb-run -a -s "-screen 0 1920x1080x24" colmap mapper \
  --database_path colmap_workspace/database.db \
  --image_path path/to/frames \
  --output_path colmap_workspace/sparse

# Convert to text format for 3DGUT
colmap model_converter \
  --input_path colmap_workspace/sparse/0 \
  --output_path colmap_workspace/sparse/0 \
  --output_type TXT
```

### Step 3: Train 3DGUT Model

```bash
conda activate 3dgrut

python train.py \
  --config-name apps/colmap_3dgut.yaml \
  path=path/to/colmap_workspace \
  out_dir=runs \
  experiment_name=my_scene \
  num_iterations=30000
```

### Step 4: Render and Export

```bash
# Render video
python render.py \
  --checkpoint runs/my_scene/ckpt_last.pt \
  --output renders/output.mp4

# Export to USD for Isaac Sim
python export_usd.py \
  --checkpoint runs/my_scene/ckpt_last.pt \
  --output exports/scene.usdz
```

## 📓 Jupyter Notebook

The complete workflow is demonstrated in `video_to_3dgrut.ipynb`:

1. Open the notebook in Jupyter or your IDE
2. Select the `3dgrut` kernel
3. Run cells sequentially to:
   - Verify GPU and system resources
   - Extract video frames
   - Run COLMAP pipeline
   - Train 3DGUT model
   - Visualize and export results

## 🎬 Example Outputs

- **Trained Model**: `runs/my_scene/ckpt_last.pt` - PyTorch checkpoint
- **USD Export**: `runs/my_scene/export_last.usdz` - For Isaac Sim integration
- **Rendered Video**: `renders/output.mp4` - Photorealistic novel views

## 🔧 Hardware Requirements

**Tested Configuration:**
- GPU: NVIDIA L40S (46GB VRAM)
- CPU: AMD EPYC 9254 (8 cores)
- RAM: 144GB
- CUDA: 12.7
- Driver: 565.57.01

**Minimum Requirements:**
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 32GB+
- Storage: 50GB+ for datasets and outputs

## 📚 Key Technologies

- **[NVIDIA 3DGUT](https://github.com/nv-tlabs/3dgrut)**: 3D Gaussian Splatting with Unscented Transforms
- **[COLMAP](https://colmap.github.io/)**: Structure-from-Motion and Multi-View Stereo
- **[FFmpeg](https://ffmpeg.org/)**: Video processing and frame extraction
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **USD**: Universal Scene Description for 3D interchange

## 🎓 Resources

- [NVIDIA Developer Blog: Rendering Real-World Scenes](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/)
- [3DGUT Paper](https://arxiv.org/abs/2405.09531)
- [COLMAP Documentation](https://colmap.github.io/tutorial.html)
- [NVIDIA Brev Documentation](https://docs.nvidia.com/brev/latest/about-brev.html)

## 📝 Notes

- Frame extraction rate (fps) affects reconstruction quality and training time
- COLMAP requires sufficient image overlap for successful reconstruction
- Training iterations can be adjusted based on scene complexity
- GPU memory requirements scale with scene size and resolution

## 🤝 Contributing

This project is part of the NVIDIA Cosmos Cookoff. Contributions and improvements are welcome!

## 📄 License

See individual component licenses:
- 3DGUT: Check [repository license](https://github.com/nv-tlabs/3dgrut)
- COLMAP: BSD License
