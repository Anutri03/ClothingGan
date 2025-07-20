# ClothingGAN Installation Guide

This guide provides detailed instructions for setting up the ClothingGAN project, with a focus on resolving the PyCUDA OpenGL support issue.

## Prerequisites

- Python 3.7 or higher
- CUDA toolkit (compatible with your GPU)
- NVIDIA drivers
- OpenGL development libraries

## Installation Options

### Option 1: Using Conda (Recommended)

1. Create a new conda environment:
   ```bash
   conda create -n clothinggan python=3.10
   conda activate clothinggan
   ```

2. Install PyTorch with CUDA support:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. Install PyCUDA with OpenGL support from conda-forge:
   ```bash
   conda install -c conda-forge pycuda
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Building PyCUDA from Source with OpenGL Support

If the conda installation doesn't work, you can build PyCUDA from source with OpenGL support:

1. Install system dependencies:

   **For Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     libgl1-mesa-dev \
     libglu1-mesa-dev \
     freeglut3-dev \
     libglew-dev \
     libglfw3-dev \
     python3-dev
   ```

   **For Windows:**
   - Install Visual Studio with C++ development tools
   - Install the Windows SDK

2. Clone and build PyCUDA:
   ```bash
   git clone https://github.com/inducer/pycuda.git
   cd pycuda
   python configure.py --cuda-root=/usr/local/cuda
   python setup.py install
   ```

3. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Using the Gradio Interface (Alternative)

If you continue to have issues with PyCUDA and OpenGL, you can use the Gradio interface instead:

1. Install the basic requirements:
   ```bash
   pip install torch torchvision torchaudio
   pip install gradio
   ```

2. Run the Gradio app:
   ```bash
   python gradio_app.py
   ```

## Troubleshooting

### PyCUDA OpenGL Support Issues

If you see the error "PyCUDA was compiled without GL extension support", try these solutions:

1. **Check OpenGL installation:**
   ```bash
   python -c "from OpenGL.GL import *; print('OpenGL is working')"
   ```

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ```

3. **Check PyCUDA installation:**
   ```bash
   python -c "import pycuda; print(pycuda.VERSION_TEXT)"
   ```

4. **For Windows users:**
   - Make sure you have the latest NVIDIA drivers
   - Install the Visual C++ Redistributable packages
   - Ensure your PATH includes the CUDA bin directory

### Common Issues

1. **CUDA version mismatch:**
   - Make sure your PyTorch CUDA version matches your system CUDA version
   - You may need to install a specific version: `pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html`

2. **OpenGL compatibility:**
   - Some systems may require additional OpenGL libraries
   - For Linux: `sudo apt-get install mesa-utils`

3. **Memory issues:**
   - If you encounter CUDA out of memory errors, try reducing batch size or image resolution

## Running the Application

Once installation is complete, you can run the interactive application:

```bash
python interactive.py
```

Or use the Gradio interface:

```bash
python gradio_app.py
``` 