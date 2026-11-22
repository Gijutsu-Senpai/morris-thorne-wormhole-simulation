# Morris–Thorne Wormhole Visualization

A GPU-accelerated Python/GLSL visualization of light-ray propagation through a wormhole-inspired spatial geometry, using two OpenEXR environment maps to represent the two sides of the scene.

> **Project status:** Educational / visualization project. The source documents the implementation used to create the associated video. It is not a validated physical simulation of an experimentally observed wormhole.

## Video

**YouTube:** [https://youtu.be/DsKurZOjRbw](https://youtu.be/DsKurZOjRbw)

## What this project does

The renderer combines Python, ModernGL, Pygame, NumPy, and a GLSL fragment shader to:

1. Load two RGB OpenEXR environment maps.
2. Create an OpenGL 3.3 rendering context.
3. Generate a viewing ray for each output pixel.
4. Express the ray in spherical coordinates associated with the wormhole geometry.
5. Numerically propagate eligible rays through the shader's effective geometry.
6. Sample the environment map corresponding to the side on which the ray finishes.
7. Write the rendered frames to an MP4 using the `h264_nvenc` encoder.

Python controls the animation timeline, camera, assets, and output. The fragment shader performs the per-pixel ray initialization, propagation, and environment lookup on the GPU.

## Repository contents

```text
.
├── README.md
├── wormhole_simulation.py
├── wormhole_simulation.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── assets/
    └── README.md           # Instructions for obtaining external EXR maps
```

The EXR files are not included by default. See [Environment-map assets](#environment-map-assets) and [`assets/README.md`](assets/README.md).

## Scientific background and scope

The project is inspired by the Morris–Thorne traversable wormhole framework. In the standard zero-redshift, ultrastatic form, a commonly used spatial line element is:

\[
ds^2=-dt^2+dl^2+(l^2+b_0^2)
\left(d\theta^2+\sin^2\theta\,d\phi^2\right)
\]

where:

- \(l\) is a radial proper-distance coordinate.
- \(b_0\) is the throat-radius parameter.
- \(\theta\) is the polar angle.
- \(\phi\) is the azimuthal angle.

The supplied shader uses an animated effective-radius expression of the form:

\[
R_{\mathrm{eff}}^2=l^2+1.8b_0^2w
\]

where \(w\) is the animation control passed to the shader as `u_warp`.

**Important distinction:** The `u_warp` animation modifies the effective geometry between rendered frames to create the growth and collapse effect. The shader does not solve a fully time-dependent spacetime geodesic problem while each ray is travelling. The animated geometry is a custom visualization device, not a standard dynamic Morris–Thorne solution.

### Numerical integration note

The GLSL function is named `rk45` in the supplied source, but its stages and weighted update do not implement a complete or validated Runge–Kutta–Fehlberg 4(5) algorithm. This repository preserves the supplied implementation and refers to it as a **custom, truncated Runge–Kutta-style update**. It should not be treated as an error-controlled RK45 solver.

The visualization is intended to illustrate lensing and ray propagation in a wormhole-inspired geometry. It is not a validated numerical prediction, proof of wormhole existence, or demonstration that traversable wormholes are physically feasible.

## Requirements

### Python packages

The source imports:

- `pygame`
- `moderngl`
- `numpy`
- `imageio`
- `OpenEXR`
- `Imath`
- `tqdm`

It also uses Python standard-library modules `math`.

Install the packages with:

```bash
python -m pip install -r requirements.txt
```

Package versions should be pinned only after testing in the environment used for the release. The initial requirements file lists the direct dependencies without claiming that particular versions have been validated.

### System requirements

- Python 3.
- A graphics card and driver supporting OpenGL 3.3 or later.
- Pygame/ModernGL-compatible OpenGL context creation.
- An ImageIO/FFmpeg setup that exposes the `h264_nvenc` encoder.
- A compatible NVIDIA GPU and driver for NVIDIA NVENC encoding when using the configured codec.

The renderer uses the GPU for the fragment shader. Video encoding through `h264_nvenc` is a separate requirement from OpenGL rendering.

## Environment-map assets

The current source expects these files in its working directory:

| Configuration variable | Expected filename | Purpose |
|---|---|---|
| `EXR_UNIVERSE_A` | `starmap_2020_16k.exr` | Environment sampled for the positive-\(l\) side |
| `EXR_UNIVERSE_B` | `hdrmaps_com_free_10K.exr` | Environment sampled for the negative-\(l\) side |

The project does not include these assets by default. Obtain them from their original sources and follow the applicable licence and attribution terms. Do not redistribute them in this repository unless their licences explicitly permit redistribution.

If you store the files in an `assets/` directory, update the two path variables in the configuration section of the Python file accordingly.

## Installation and execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the EXR files

Place both required EXR files in the working directory, or update `EXR_UNIVERSE_A` and `EXR_UNIVERSE_B` to point to their locations.

### 5. Run the script

```bash
python wormhole_simulation.py
```

The configured output filename is `wormhole_test1.mp4`, written to the current working directory.

### Jupyter Notebook

Open `wormhole_simulation.ipynb` in Jupyter Notebook, JupyterLab, or a compatible notebook environment. Ensure the dependencies and EXR assets are available to the notebook's kernel before running the implementation cell.

## Configuration reference

The main configuration values are defined near the top of `wormhole_simulation.py`.

| Parameter | Default | What it controls |
|---|---:|---|
| `WIDTH` | `640` | Output frame width in pixels |
| `HEIGHT` | `480` | Output frame height in pixels |
| `FPS` | `12` | Output frame rate; also used to convert camera speed to per-frame movement |
| `THROAT_START` | `0.0` | Declared in the source but currently unused |
| `THROAT_MAX` | `1.5` | Maximum throat parameter passed to the shader |
| `GROW_DURATION` | `1.5` | Duration of the growth phase, in seconds |
| `TRAVEL_DURATION` | `27.0` | Duration of the travel phase, in seconds |
| `COLLAPSE_DURATION` | `1.5` | Duration of the collapse phase, in seconds |
| `EXR_UNIVERSE_A` | `starmap_2020_16k.exr` | Environment map for the positive-\(l\) side |
| `EXR_UNIVERSE_B` | `hdrmaps_com_free_10K.exr` | Environment map for the negative-\(l\) side |
| `OUTPUT_FILE` | `wormhole_test1.mp4` | Output video filename |
| `BASE_SPEED` | `1.2` | Base factor in the camera's distance-dependent speed expression |
| `CAMERA_OFFSET` | `1.25` | Camera's transverse offset from the wormhole axis |

### Animation phases

The frame counts are derived from `FPS` and the phase durations:

- `GROW_FRAMES = int(FPS * GROW_DURATION)`
- `TRAVEL_FRAMES = int(FPS * TRAVEL_DURATION)`
- `COLLAPSE_FRAMES = int(FPS * COLLAPSE_DURATION)`

The growth phase uses `warp_growth(t)`, the travel phase fixes `warp` to `1.0`, and the collapse phase uses `warp_collapse(t)`.

### Shader uniforms

| Uniform | Purpose |
|---|---|
| `u_resolution` | Output dimensions used to form normalized pixel coordinates |
| `u_cam_pos` | Camera position in `(l, theta, phi)` coordinates |
| `u_cam_forward` | Camera forward direction |
| `u_cam_right` | Camera right direction |
| `u_cam_up` | Camera up direction |
| `u_throat` | Throat parameter used in the effective-radius expressions |
| `u_warp` | Animated multiplier for the throat contribution |
| `u_max_dist` | Radial-coordinate threshold used to stop ray integration |
| `u_envA`, `u_envB` | Samplers for the two environment maps |

The shader also contains fixed implementation choices, including a maximum of 40,000 integration-loop iterations and a step-size expression clamped between `0.001` and `0.05`. These are currently constants in the GLSL source rather than top-level Python configuration parameters.

## Troubleshooting

### `FileNotFoundError` or EXR loading errors

Check that both EXR files exist at the paths specified by `EXR_UNIVERSE_A` and `EXR_UNIVERSE_B`. The loader expects RGB channels named `R`, `G`, and `B`.

### OpenGL context or shader compilation errors

Check that the graphics driver supports OpenGL 3.3 or later and that Pygame can create an OpenGL context on the machine.

### `h264_nvenc` encoder not found

The configured video writer requests NVIDIA's NVENC H.264 encoder. Check that the installed FFmpeg build exposes `h264_nvenc`, and that the NVIDIA driver/GPU supports it. The current script does not automatically fall back to a software encoder.

### Output differs across machines

GPU model, graphics driver, OpenGL implementation, library versions, and encoder settings can affect rendering or video output. Use the same environment maps and record the software/hardware environment when comparing results.

## Reproducibility and known limitations

- The code is preserved from the supplied implementation; the numerical method has not been corrected or independently validated.
- The geometry is modified by an animation parameter between frames rather than evolved as a fully time-dependent spacetime during ray propagation.
- The numerical integration uses a heuristic step size and a fixed iteration limit.
- The EXR loader assumes separate RGB channels and reads them as 32-bit floats.
- The output pipeline reads pixels from an off-screen framebuffer and sends them directly to ImageIO. Verify image orientation and encoding on your own system.
- No claim is made that the rendered imagery is a physically exact prediction.

## Acknowledgements and references

The theoretical inspiration is the Morris–Thorne wormhole model:

Morris, M. S., & Thorne, K. S. (1988). *Wormholes in spacetime and their use for interstellar travel: A tool for teaching general relativity*. American Journal of Physics, 56, 395–412.

Please also acknowledge the creators and sources of any environment maps or other external assets used with the project, in accordance with their licences.

## How to cite this project

If you use or adapt this code, please cite the GitHub repository and include its URL, release tag (if available), and access date. A formal citation record can be added when the repository has a stable public URL and release.

## Licence

The source code in this repository is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the complete licence text.

The MIT License applies to the original code in this repository. It does **not** automatically grant rights to third-party environment maps, textures, or other external assets. Those remain subject to their respective owners' licences and attribution requirements.
