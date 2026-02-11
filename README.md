# Bandgap network-based OCT Denoiser

Eric Tang (tangericm) \
eric.tang22@gmail.com

## Environment setup

IDE: Visual Studio Code

Miniconda Python v3.14.2 environment
```
conda create --name OCTDenoiser python=3.14
conda activate OCTDenoiser
```

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

### Train a model
```
python model_train.py
```

### Run inference with a trained checkpoint
```
python model_predict.py
```

### Hyperparameter tuning
```
python tune.py
```

## Project Structure

```
OCT-Denoiser/
├── model_train.py           # Main training entry point
├── model_predict.py         # Standalone inference script
├── preprocess.py            # OCT signal processing pipeline
├── tune.py                  # Optuna hyperparameter search
├── configs/default.py       # TrainConfig and FolderSpec dataclasses
├── engine/                  # Training, evaluation, inference, losses, metrics
├── data/                    # Dataset and DataModule
├── networks/                # Model definitions (ResUNet Pseudo-3D)
└── utils/                   # I/O, seeding, plotting, run management
```

## Configuration

All configuration uses Python dataclasses in `configs/default.py`:
- `FolderSpec` — per-dataset specification (also used directly by `BscanProcessor`)
- `TrainConfig` — training hyperparameters, loss weights, early stopping, ROI bounds
