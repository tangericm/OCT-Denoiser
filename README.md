# Bandgap network-based OCT Denoiser

Eric Tang (tangericm) \
eric.tang22@gmail.com

## Enviroment setup

IDE: Visual Studio Code

Miniconda Python v3.14.2 environment
```
conda create --name OCTDenoiser python=3.14
conda activate OCTDenoiser 
```

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install numpy matplotlib tifffile
pip install -r requirements.txt
pip freeze > requirements.txt
```