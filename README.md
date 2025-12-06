# CMY Reaction Diffusion Camera Shader

A real-time WebGL shader that creates stunning visual effects by combining reaction-diffusion patterns with live camera input using the CMY (Cyan, Magenta, Yellow) color model.

## Overview

This project implements the Gray-Scott reaction-diffusion model across three separate layers (Cyan, Magenta, Yellow). The camera input is inverted and mapped to control each layer:

- **Red channel (inverted) → Cyan layer**: Areas with more red in the camera feed increase cyan concentration
- **Green channel (inverted) → Magenta layer**: Areas with more green increase magenta concentration
- **Blue channel (inverted) → Yellow layer**: Areas with more blue increase yellow concentration

The three layers are then combined using subtractive color mixing, creating a full-color reaction-diffusion image that responds to your camera feed in real-time.

## Features

- 🎨 Real-time reaction-diffusion simulation using Gray-Scott model
- 📹 Live camera input integration
- 🌈 Subtractive CMY color mixing
- ⚙️ Interactive parameter controls
- 🎯 High-performance WebGL implementation with ping-pong rendering

## How It Works

### Reaction-Diffusion Model

The simulation uses the Gray-Scott model, which describes the reaction and diffusion of two chemicals (A and B):

```
∂A/∂t = Da∇²A - AB² + f(1-A)
∂B/∂t = Db∇²B + AB² - (k+f)B
```

Where:
- `Da`, `Db`: Diffusion rates for chemicals A and B
- `f`: Feed rate
- `k`: Kill rate
- `∇²`: Laplacian operator

### Input Sources

The shader supports three input types:

1. **Camera**: Live webcam feed using WebMediaDevices API
2. **Video File**: Upload a video file from your computer (loops automatically)
3. **Image File**: Upload a static image as input

### Input Processing

1. **Capture/Load**: Get input from camera, video file, or image file
2. **Inversion**: RGB channels are inverted to get CMY values
3. **Layer Mapping**: Each CMY channel feeds into its corresponding reaction-diffusion layer
4. **Influence**: Input data affects the B chemical concentration in each layer
5. **Compositing**: Three layers are combined using subtractive color mixing

### Color Mixing

Subtractive color mixing (CMY model):
```
Red   = 1 - (Cyan + Magenta)
Green = 1 - (Cyan + Yellow)
Blue  = 1 - (Magenta + Yellow)
```

## Usage

### Running Locally

1. Simply open `index.html` in a modern web browser that supports WebGL and camera access (Chrome, Firefox, Safari, Edge)

2. Grant camera permissions when prompted

3. The simulation will start automatically

### Hosting

For development, you can use any local server:

```bash
# Using Python 3
python -m http.server 8000

# Using Node.js http-server
npx http-server

# Using PHP
php -S localhost:8000
```

Then navigate to `http://localhost:8000`

## Controls

### Input Source
- **Input Source Selector**: Choose between Camera, Video File, or Image File
  - **Camera**: Uses your webcam (requires permission)
  - **Video File**: Upload a video from your computer (MP4, WebM, etc.)
  - **Image File**: Upload a static image (PNG, JPG, etc.)

### Reaction-Diffusion Parameters
- **Feed Rate** (0.01 - 0.1): Controls how quickly chemical A is added to the system
- **Kill Rate** (0.01 - 0.1): Controls how quickly chemical B is removed
- **Diffusion A** (0.1 - 2.0): Diffusion rate for chemical A
- **Diffusion B** (0.1 - 2.0): Diffusion rate for chemical B
- **Camera Influence** (0.0 - 1.0): How strongly the input affects the simulation

### Actions
- **Reset Simulation**: Reinitialize the simulation with random seed (also retries camera if failed)
- **Hide/Show Controls**: Toggle the control panel visibility

## Parameter Presets

Try these parameter combinations for different effects:

### Coral Pattern
- Feed: 0.055
- Kill: 0.062
- Diff A: 1.0
- Diff B: 0.5

### Spots
- Feed: 0.014
- Kill: 0.054
- Diff A: 1.0
- Diff B: 0.5

### Waves
- Feed: 0.014
- Kill: 0.045
- Diff A: 1.0
- Diff B: 0.5

### Maze
- Feed: 0.029
- Kill: 0.057
- Diff A: 1.0
- Diff B: 0.5

## Technical Details

### Architecture

- **Ping-Pong Rendering**: Alternates between two framebuffers for each layer to avoid read/write conflicts
- **Multi-pass Pipeline**:
  1. Camera processing pass (invert and extract CMY)
  2. Reaction-diffusion simulation pass (3 layers, 2 iterations per frame)
  3. Compositing pass (combine layers with subtractive mixing)

### Shaders

1. **Camera Process Shader**: Inverts video input and separates CMY channels
2. **Reaction-Diffusion Shader**: Implements Gray-Scott model with Laplacian diffusion
3. **Compositing Shader**: Combines three layers using subtractive color mixing
4. **Init Shader**: Creates random initial conditions

### Performance

- Simulation resolution: 512x512 pixels
- Updates: 2 simulation steps per frame
- Target: 60 FPS on modern hardware

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari (iOS may require HTTPS)
- ⚠️ Requires WebGL support
- 📷 Camera access optional (can use video/image files instead)

## Troubleshooting

### Camera not working
- **Quick solution**: Select "Video File" or "Image File" from the Input Source dropdown and upload a file instead
- Ensure you granted camera permissions when prompted
- Check if another application is using the camera
- Try using `http://localhost:8000` instead of opening the file directly (file://)
- Try using HTTPS (required on some browsers for camera access)
- Check browser console (F12) for detailed error messages

### Poor performance
- Close other GPU-intensive applications
- Try a different browser
- Reduce browser window size
- Check if hardware acceleration is enabled

### Black screen
- Check browser console for WebGL errors
- Ensure WebGL is supported and enabled
- Try updating graphics drivers

## License

MIT License - Feel free to use and modify for your projects

## Acknowledgments

- Gray-Scott model research by Pearson (1993)
- Inspired by reaction-diffusion work by Karl Sims and others
- WebGL and GLSL shader programming community

## Future Enhancements

- [ ] Additional reaction-diffusion models
- [ ] Recording/export functionality
- [ ] Mobile touch controls
- [ ] Multiple pattern presets
- [ ] Save/load parameter configurations
- [ ] GPU performance optimizations
