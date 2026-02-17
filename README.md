# 3D Gaussian Splatting Tutorial

Train 3DGUT models on NeRF Synthetic dataset and export to USD.

## Setup

Run in terminal:

```bash
# Clone repo
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# Install environment (creates Python 3.11 conda env + Jupyter kernel)
./install_env.sh 3dgrut
```

## Run

1. Open `gs.ipynb` in your IDE
2. Select `3dgrut` kernel (top-right dropdown)
3. Run all cells

## Outputs

- `runs/lego_3dgut_tutorial/lego-*/ckpt_last.pt` - Trained model
- `runs/lego_3dgut_tutorial/lego-*/export_last.usdz` - USD export (28MB)
- `lego_render.mp4` - Rendered video

## Training Command

```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml \
    path=data/nerf_synthetic/lego \
    out_dir=runs \
    experiment_name=lego_3dgut
```
