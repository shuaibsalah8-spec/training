import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import argparse


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    """Compute the Mandelbrot set escape counts for a rectangular region.
    Returns a 2D array of iteration counts (0..maxiter).
    """
    # Create a complex grid
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=np.complex128)
    div_time = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(maxiter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = np.abs(Z) > 2.0
        newly_escaped = escaped & mask
        div_time[newly_escaped] = i
        mask &= ~escaped
        # minor speed-up: break if all escaped
        if not mask.any():
            break
    div_time[div_time == 0] = maxiter
    return div_time


def make_colormap():
    # gentle viridis-like ramp but custom for visual richness
    colors = [
        (0.0, "#0d1b2a"),
        (0.16, "#1b263b"),
        (0.42, "#415a77"),
        (0.6425, "#778da9"),
        (0.8575, "#e0e1dd"),
        (1.0, "#ffb703"),
    ]
    cdict = [(t, c) for t, c in colors]
    return LinearSegmentedColormap.from_list("custom_fractal", [c for t, c in colors])


def animate_fractal(
    center=(-0.743643887037151, 0.13182590420533),
    zoom_factor=0.96,
    frames=180,
    width=800,
    height=600,
    maxiter=300,
    cmap=None,
    save_path=None,
):
    """Create and display (and optionally save) a zooming Mandelbrot animation."""
    if cmap is None:
        cmap = make_colormap()

    # initial viewing window size
    span = 3.5
    cx, cy = center

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_axis_off()

    # Precompute frames (this keeps CPU busy but ensures a smooth animation)
    ims = []
    current_span = span
    # We'll compute frame-by-frame and yield images to animation.FuncAnimation

    # Instead of precomputing all frames (memory heavy), we'll compute on the fly
    im = ax.imshow(np.zeros((height, width)), cmap=cmap, origin='lower', extent=[0, 1, 0, 1])

    # We'll cache the most recent computation to reduce flicker
    def update(frame):
        nonlocal current_span
        current_span *= zoom_factor
        xmin = cx - current_span / 2
        xmax = cx + current_span / 2
        ymin = cy - (current_span * height / width) / 2
        ymax = cy + (current_span * height / width) / 2

        # adapt maxiter slowly so deeper zooms get more iterations
        local_maxiter = int(maxiter * (1 + frame / (frames * 0.6)))

        arr = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, local_maxiter)

        # smooth coloring: normalize and apply log to iteration counts for smoother bands
        with np.errstate(divide='ignore'):
            log_counts = np.log(arr + 1)
        # normalize to 0..1
        norm = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min() + 1e-9)

        im.set_data(norm)
        ax.set_title(f"Zoom frame {frame+1}/{frames}  span={current_span:.2e}  iters={local_maxiter}", fontsize=10)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)

    if save_path:
        # Try to save. This requires ffmpeg (for mp4) or imagemagick (for gif) installed.
        ext = save_path.split('.')[-1].lower()
        print(f"Saving animation to {save_path} (this may take a while)...")
        if ext in ('mp4', 'm4v'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30, metadata=dict(artist='fractal_endless'), bitrate=1800)
            ani.save(save_path, writer=writer)
        elif ext in ('gif',):
            ani.save(save_path, writer='imagemagick', fps=30)
        else:
            raise ValueError('Unsupported save extension. Use mp4 or gif.')
        print('Saved.')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Endless fractal (Mandelbrot) zoom animation')
    parser.add_argument('--frames', type=int, default=180, help='Number of frames in the animation')
    parser.add_argument('--width', type=int, default=800, help='Frame width in pixels')
    parser.add_argument('--height', type=int, default=600, help='Frame height in pixels')
    parser.add_argument('--maxiter', type=int, default=300, help='Base maximum iterations')
    parser.add_argument('--zoom', type=float, default=0.96, help='Multiplicative zoom per frame (0.99=slow)')
    parser.add_argument('--center', type=str, default='-0.743643887037151,0.13182590420533',
                        help='Center coordinates as "real,imag"')
    parser.add_argument('--save', type=str, default=None, help='Optional output file (mp4 or gif)')

    args = parser.parse_args()
    cx, cy = map(float, args.center.split(','))

    animate_fractal(center=(cx, cy), zoom_factor=args.zoom, frames=args.frames, width=args.width,
                    height=args.height, maxiter=args.maxiter, cmap=None, save_path=args.save)
