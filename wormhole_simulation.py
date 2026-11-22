"""
Morris–Thorne Wormhole Visualization
====================================

GPU-accelerated, environment-map-based visualization of light-ray propagation
through a wormhole-inspired spatial geometry.

This script:
    1. Loads two RGB OpenEXR environment maps.
    2. Creates an OpenGL context through Pygame and ModernGL.
    3. Runs a GLSL fragment shader that initializes and propagates viewing rays.
    4. Samples the environment map associated with the ray's final universe.
    5. Writes rendered frames to an H.264 video using NVIDIA NVENC.

Important scientific and implementation notes:
    - This is a visualization, not an observational model of a real wormhole.
    - The shader uses a custom, truncated Runge–Kutta-style update. Although
      the original function is named ``rk45``, its implementation is not a
      complete or validated Runge–Kutta–Fehlberg 4(5) method.
    - The animated ``u_warp`` parameter modifies the effective spatial
      geometry for the growth/collapse effect. It is a custom animation
      mechanism, not a standard time-dependent Morris–Thorne solution.
    - The script preserves the supplied implementation and its numerical
      behaviour. Comments explain the current code; they do not certify its
      numerical accuracy.

Run this file from a directory containing both EXR assets:
    python wormhole_simulation.py
"""

import pygame
import moderngl
import numpy as np
import math
import imageio
import OpenEXR
import Imath
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================
# These values control the output video, animation timeline,
# environment-map inputs, and camera motion.
#
# WIDTH / HEIGHT:
#     Pixel dimensions of each rendered frame.
#
# THROAT_START:
#     Declared in the original code but not currently referenced.
#
# THROAT_MAX:
#     Maximum value supplied to the shader as the throat parameter.
#     The shader combines this with the radial coordinate to form an
#     effective radius. It is a model parameter, not a physical unit
#     unless a unit convention is separately established.
#
# FPS:
#     Output frame rate and the time-step conversion used by camera motion.
#
# GROW_DURATION / TRAVEL_DURATION / COLLAPSE_DURATION:
#     Durations of the three animation phases, in seconds.
#
# EXR_UNIVERSE_A / EXR_UNIVERSE_B:
#     Equirectangular environment maps sampled when a ray exits on the
#     positive- or negative-l side, respectively.
#
# BASE_SPEED:
#     Base factor used by the camera's distance-dependent travel-speed rule.
#
# CAMERA_OFFSET:
#     Transverse camera offset from the wormhole axis.
# ==========================================================

WIDTH, HEIGHT = 640, 480

THROAT_START = 0.0
THROAT_MAX = 1.5

FPS = 12

GROW_DURATION = 1.5
TRAVEL_DURATION = 24 + 2*GROW_DURATION
COLLAPSE_DURATION = GROW_DURATION

# Convert phase durations into integer frame counts.
GROW_FRAMES = int(FPS * GROW_DURATION)
TRAVEL_FRAMES = int(FPS * TRAVEL_DURATION)
COLLAPSE_FRAMES = int(FPS * COLLAPSE_DURATION)

TOTAL_FRAMES = GROW_FRAMES + TRAVEL_FRAMES + COLLAPSE_FRAMES

# Environment maps for the two sides/universes of the visualization.
EXR_UNIVERSE_B = "Starting_Universe.exr"
EXR_UNIVERSE_A = "Ending_Universe.exr"

OUTPUT_FILE = "wormhole_test1.mp4"

BASE_SPEED = 1.2
CAMERA_OFFSET = 1.25


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def smoothstep(x):
    """Cubic smooth interpolation: 0→1 with eased start and end.

    The function is used to make camera/geometry animation transitions
    less abrupt than linear interpolation.
    """
    return x*x*(3-2*x)


def ease_out(x):
    """Quadratic ease-out curve.

    Present in the original source but not currently called elsewhere.
    """
    return 1 - (1-x)*(1-x)


def lerp(a,b,t):
    """Linearly interpolate from ``a`` to ``b`` by fraction ``t``."""
    return a+(b-a)*t


def warp_growth(t):
    """Return the eased growth envelope used during the opening phase."""
    return smoothstep(t*t)


def warp_collapse(t):
    """Return the eased collapse envelope used during the closing phase."""
    return (1 - smoothstep(t))**2


# ==========================================================
# OPENEXR ENVIRONMENT-MAP LOADER
# ==========================================================

def load_exr(path):
    """Load the R, G and B channels of an OpenEXR image into a float array.

    Args:
        path: Path to an RGB OpenEXR file.

    Returns:
        Contiguous NumPy array with shape (height, width, 3), dtype float32.
        The rows are vertically flipped before being returned.

    Notes:
        This loader assumes the EXR contains separate R, G and B channels
        that can be read as 32-bit floating-point values. It does not apply
        exposure, tone mapping, or colour-space conversion.
    """

    # Open the EXR and read its data-window dimensions. The data window
    # defines the pixel region stored in the file.
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header['dataWindow']

    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Request each colour channel as 32-bit float data.
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    r = np.frombuffer(exr.channel('R', FLOAT), dtype=np.float32)
    g = np.frombuffer(exr.channel('G', FLOAT), dtype=np.float32)
    b = np.frombuffer(exr.channel('B', FLOAT), dtype=np.float32)

    # Reconstruct each one-dimensional channel buffer as a 2D image.
    r = r.reshape(height,width)
    g = g.reshape(height,width)
    b = b.reshape(height,width)

    # Combine channels into an H×W×3 RGB array.
    img = np.stack([r,g,b],axis=-1)

    # Flip rows to match the vertical convention expected by the shader.
    img = np.flipud(img)

    # Ensure the array has contiguous memory for texture upload.
    return np.ascontiguousarray(img)


# ==========================================================
# GLSL FRAGMENT SHADER
# ==========================================================
# The shader runs once per output pixel. It constructs a viewing ray,
# expresses that ray in the wormhole's spherical coordinates, propagates
# the ray through the effective geometry, and samples one of the two
# environment maps according to the side on which the ray finishes.
#
# Coordinate/state conventions:
#     l   : radial proper-distance-like coordinate along the wormhole axis
#     th  : polar angle
#     ph  : azimuthal angle
#     vl  : derivative of l with respect to the ray parameter
#     vth : derivative of th with respect to the ray parameter
#     vph : derivative of ph with respect to the ray parameter
#
# The function called rk45 below is retained exactly as supplied. Its
# stages and update are incomplete for a standard RK45 method; see the
# repository documentation for this limitation.
# ==========================================================

FRAGMENT_SHADER = """
#version 330

uniform vec2 u_resolution;
uniform vec3 u_cam_pos;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_throat;
uniform float u_max_dist;
uniform float u_warp;

uniform sampler2D u_envA;
uniform sampler2D u_envB;

out vec4 f_color;

#define PI 3.14159265359

// Convert a 3D direction into equirectangular UV coordinates and sample
// the corresponding environment texture.
vec3 sample_env(sampler2D tex, vec3 dir)
{
    dir = normalize(dir);
    float phi = atan(dir.z, dir.x);
    float u = (phi + PI) / (2.0 * PI);
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return texture(tex, vec2(u, v)).rgb;
}

// State of one ray in spherical coordinates, including its coordinate
// velocities. The state is integrated numerically in the fragment shader.
struct State { float l; float th; float ph; float vl; float vth; float vph; };

// Evaluate the first-order system corresponding to the spatial geodesic
// equations used by this visualization.
State derivatives(State s)
{
    // Effective squared areal-radius term. u_warp animates the throat
    // contribution during the growth and collapse phases.
    float r2 = s.l*s.l + (u_throat*u_throat)*1.8*u_warp;

    float sin_th = sin(s.th);
    sin_th = max(abs(sin_th),0.001) * sign(sin_th);

    float cos_th = cos(s.th);
    cos_th = max(abs(cos_th),0.001) * sign(cos_th);

    State d;
    d.l = s.vl;
    d.th = s.vth;
    d.ph = s.vph;

    d.vl  = s.l * (s.vth*s.vth + sin_th*sin_th * s.vph*s.vph);
    d.vth = -(2.0*s.l/r2)*s.vl*s.vth + sin_th*cos_th*s.vph*s.vph;
    d.vph = -(2.0*s.l/r2)*s.vl*s.vph - 2.0*(cos_th/sin_th)*s.vth*s.vph;

    return d;
}

// Advance the ray state by step h using the original custom multi-stage
// update. The function name is preserved from the source; this is not a
// complete adaptive RK45 / Runge–Kutta–Fehlberg 4(5) implementation.
State rk45(State s, float h)
{
    State k1 = derivatives(s);

    State s2=s;
    s2.l+=0.25*h*k1.l;
    s2.th+=0.25*h*k1.th;
    s2.ph+=0.25*h*k1.ph;
    s2.vl+=0.25*h*k1.vl;
    s2.vth+=0.25*h*k1.vth;
    s2.vph+=0.25*h*k1.vph;
    State k2=derivatives(s2);

    State s3=s;
    s3.l+=h*(3./32.*k1.l+9./32.*k2.l);
    s3.th+=h*(3./32.*k1.th+9./32.*k2.th);
    s3.ph+=h*(3./32.*k1.ph+9./32.*k2.ph);
    s3.vl+=h*(3./32.*k1.vl+9./32.*k2.vl);
    s3.vth+=h*(3./32.*k1.vth+9./32.*k2.vth);
    s3.vph+=h*(3./32.*k1.vph+9./32.*k2.vph);
    State k3=derivatives(s3);

    s.l+=h*(16./135.*k1.l+6656./12825.*k3.l);
    s.th+=h*(16./135.*k1.th+6656./12825.*k3.th);
    s.ph+=h*(16./135.*k1.ph+6656./12825.*k3.ph);
    s.vl+=h*(16./135.*k1.vl+6656./12825.*k3.vl);
    s.vth+=h*(16./135.*k1.vth+6656./12825.*k3.vth);
    s.vph+=h*(16./135.*k1.vph+6656./12825.*k3.vph);

    return s;
}

void main()
{
    // Convert fragment coordinates to normalized screen coordinates.
    // The x coordinate is scaled to account for the viewport aspect ratio.
    vec2 uv=((gl_FragCoord.xy+vec2(0.5))/u_resolution.xy)*2.0-1.0;
    uv.x*=u_resolution.x/u_resolution.y;

    // Construct a world-space viewing ray from the camera basis vectors.
    vec3 rd=normalize(u_cam_forward+uv.x*u_cam_right+uv.y*u_cam_up);

    // Camera radial coordinate and effective radius at the camera location.
    float cl=u_cam_pos.x;
    float rl=sqrt(cl*cl+(u_throat*u_throat)*1.8*u_warp);

    // Local spherical-coordinate basis vectors at the camera position.
    vec3 er=vec3(sin(u_cam_pos.y)*cos(u_cam_pos.z),
                 sin(u_cam_pos.y)*sin(u_cam_pos.z),
                 cos(u_cam_pos.y));

    vec3 eth=vec3(cos(u_cam_pos.y)*cos(u_cam_pos.z),
                  cos(u_cam_pos.y)*sin(u_cam_pos.z),
                 -sin(u_cam_pos.y));

    vec3 eph=vec3(-sin(u_cam_pos.z),cos(u_cam_pos.z),0.0);

    // Initialize the ray state by projecting its direction onto the
    // local spherical basis. Angular components are scaled by the
    // effective radius and sin(theta), as in spherical coordinates.
    State s;
    s.l=cl; s.th=u_cam_pos.y; s.ph=u_cam_pos.z;

    float sin_th0=sin(u_cam_pos.y);
    sin_th0 = max(abs(sin_th0),0.001) * sign(sin_th0);

    s.vl=dot(rd,er);
    s.vth=dot(rd,eth)/rl;
    s.vph=dot(rd,eph)/(rl*sin_th0);

    // Angular-momentum-like quantity used for the initial ray
    // classification. The expression is retained from the original code.
    float L2 = rl*rl*rl*rl * (s.vth*s.vth + sin_th0*sin_th0*s.vph*s.vph);
    float r2 = cl*cl + (u_throat*u_throat)*u_warp;

    // Rays that fail this threshold are sent directly to the environment
    // on the camera's current side, without running the integration loop.
    if(L2 > r2*r2)
    {
        vec3 col = (cl > 0.0) ? sample_env(u_envA, rd)
                              : sample_env(u_envB, rd);
        col = pow(col,vec3(0.4545));
        f_color = vec4(col,1.0);
        return;
    }

    // Propagate the ray. The step size grows with |l| but is clamped
    // between 0.001 and 0.05. Integration stops after 40,000 iterations
    // or when the ray moves beyond u_max_dist.
    for(int i=0;i<40000;i++)
    {
        float h = clamp(0.001 + 0.002*abs(s.l), 0.001, 0.05);
        s=rk45(s,h);
        if(abs(s.l)>u_max_dist) break;
    }

    // Convert the final angular coordinates back into a unit direction.
    vec3 final_dir=vec3(sin(s.th)*cos(s.ph),
                        sin(s.th)*sin(s.ph),
                        cos(s.th));

    // Select the environment according to the sign of the final l value.
    vec3 col=(s.l>0.0)?sample_env(u_envA,final_dir)
                      :sample_env(u_envB,final_dir);

    // Apply the shader's power-law colour transform and write the pixel.
    col=pow(col,vec3(0.4545));
    f_color=vec4(col,1.0);
}
"""


# ==========================================================
# MAIN RENDERING FUNCTION
# ==========================================================

def main():
    """Load assets, initialize OpenGL, render all frames, and encode video."""

    # Load both EXR environment maps before initializing the renderer.
    envA = load_exr(EXR_UNIVERSE_A)
    envB = load_exr(EXR_UNIVERSE_B)

    # Create a hidden 1×1 Pygame OpenGL window. It provides the OpenGL
    # context required by ModernGL; the rendered frames are read from an
    # off-screen framebuffer rather than displayed in a visible window.
    pygame.init()
    pygame.display.set_mode((1,1), pygame.OPENGL | pygame.HIDDEN)

    ctx = moderngl.create_context()

    # Compile a minimal full-screen vertex shader and the fragment shader
    # defined above. The vertex shader passes a full-screen quad to the
    # fragment shader, which performs the per-pixel rendering.
    prog=ctx.program(
        vertex_shader="#version 330\nin vec2 in_vert;void main(){gl_Position=vec4(in_vert,0,1);}",
        fragment_shader=FRAGMENT_SHADER)

    # Full-screen quad covering normalized device coordinates [-1, 1].
    quad=ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],dtype='f4'))
    vao=ctx.simple_vertex_array(prog,quad,'in_vert')

    # Upload the two float RGB environment maps as GPU textures.
    texA=ctx.texture(envA.shape[1::-1],3,envA.tobytes(),dtype='f4')
    texB=ctx.texture(envB.shape[1::-1],3,envB.tobytes(),dtype='f4')

    # Bind environment textures to texture units 0 and 1.
    texA.use(0)
    texB.use(1)

    prog['u_envA']=0
    prog['u_envB']=1

    # Create an off-screen colour texture and framebuffer at output size.
    color_tex = ctx.texture((WIDTH, HEIGHT), 3)
    fbo = ctx.framebuffer(color_tex)

    # Configure ImageIO to encode frames using H.264 through NVIDIA NVENC.
    writer=imageio.get_writer(OUTPUT_FILE,fps=FPS,codec='h264_nvenc')

    # Initial camera position along the wormhole's l coordinate.
    cam_l=-12.5

    # Render each frame in the complete growth → travel → collapse timeline.
    for frame in tqdm(range(TOTAL_FRAMES), desc="Rendering Wormhole"):

        # ------------------------------------------------------
        # PHASE 1: WORMHOLE GROWTH
        # ------------------------------------------------------
        if frame < GROW_FRAMES:

            t = frame/GROW_FRAMES
            throat = THROAT_MAX
            warp = warp_growth(t)

            orbit = 0
            radius = CAMERA_OFFSET

        # ------------------------------------------------------
        # PHASE 2: CAMERA TRAVERSAL
        # ------------------------------------------------------
        elif frame < GROW_FRAMES+TRAVEL_FRAMES:

            t = (frame-GROW_FRAMES)/TRAVEL_FRAMES
            throat = THROAT_MAX
            warp = 1.0

            # Increase the camera's travel speed with distance according
            # to the original distance-dependent expression.
            dist=abs(cam_l)
            speed=BASE_SPEED*(0.3+0.7*(dist/15.0))
            cam_l+=speed/FPS

            # Smoothly accumulate a full orbit during traversal.
            orbit = smoothstep(t)*6.0

            # Reduce transverse camera radius as the camera approaches
            # the central region, producing a tighter spiral-like path.
            spiral = smoothstep(1.0 - min(dist/8.0,1.0))
            radius = lerp(CAMERA_OFFSET, CAMERA_OFFSET*0.4, spiral)

        # ------------------------------------------------------
        # PHASE 3: WORMHOLE COLLAPSE
        # ------------------------------------------------------
        else:

            t=(frame-(GROW_FRAMES+TRAVEL_FRAMES))/COLLAPSE_FRAMES
            throat = THROAT_MAX
            warp = warp_collapse(t)

            dist=abs(cam_l)
            speed=BASE_SPEED*(0.3+0.7*(dist/15.0))
            cam_l+=speed/FPS

            orbit = 6.0
            radius = CAMERA_OFFSET*0.4

        # Convert the animated transverse offset and axial coordinate
        # into a camera position in Cartesian coordinates.
        x = radius*math.cos(orbit)
        y = radius*math.sin(orbit)

        r = math.sqrt(cam_l*cam_l + throat*throat)
        z = math.sqrt(max(r*r - x*x - y*y,0.0001))

        # Spherical camera angles supplied to the shader.
        cam_th = math.acos(z/r)
        cam_ph = math.atan2(y,x)

        cam_pos = np.array([x,y,cam_l],dtype='f4')
        target = np.array([0,0,0],dtype='f4')

        # Build an orthonormal camera basis looking toward the origin.
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)

        right = np.cross(forward,[0,1,0])
        right /= np.linalg.norm(right)

        up = np.cross(right,forward)

        # Update shader uniforms for the current frame.
        prog['u_resolution'].value=(WIDTH,HEIGHT)
        prog['u_cam_pos'].value=(cam_l,cam_th,cam_ph)
        prog['u_cam_forward'].value=tuple(forward)
        prog['u_cam_right'].value=tuple(right)
        prog['u_cam_up'].value=tuple(up)
        prog['u_throat'].value=throat
        prog['u_warp'].value=warp
        prog['u_max_dist'].value = 30 + abs(cam_l)

        # Render the full-screen quad into the off-screen framebuffer.
        fbo.use()
        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)

        # Read RGB pixels from the framebuffer and reshape into an image.
        data = fbo.read(components=3)
        img = np.frombuffer(data,dtype=np.uint8).reshape((HEIGHT,WIDTH,3))

        # Append this frame to the encoded output video.
        writer.append_data(img)

    # Close the video writer and shut down Pygame after successful rendering.
    writer.close()
    pygame.quit()

    print("Saved:",OUTPUT_FILE)


# Run the renderer when this file is executed directly, but not when imported.
if __name__ == "__main__":
    main()
