import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


def mandelbrot(xmin, xmax, ymin, ymax, width, height, maxiter):
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    divtime = np.zeros(C.shape, dtype=float)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(1, maxiter + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = np.abs(Z) > 2.0
        just_escaped = escaped & mask
        divtime[just_escaped] = i - np.log(np.log(np.abs(Z[just_escaped])))/np.log(2)
        mask &= ~escaped
        if not mask.any():
            break
    divtime[divtime == 0] = maxiter
    return divtime


def make_animation(center, start_span, zoom_per_frame, frames, width, height, base_iter, cmap_name, output):
    cx, cy = center
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.axis('off')
    im = ax.imshow(np.zeros((height, width)), origin='lower', cmap=cmap_name, vmin=0, vmax=1)

    span = start_span

    def update(frame):
        nonlocal span
        span *= zoom_per_frame
        xmin = cx - span/2
        xmax = cx + span/2
        ymin = cy - (span * height/width)/2
        ymax = cy + (span * height/width)/2
        maxiter = int(base_iter * (1 + frame / (frames*0.6)))

        arr = mandelbrot(xmin, xmax, ymin, ymax, width, height, maxiter)
        with np.errstate(divide='ignore'):
            img = np.log(arr + 1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        im.set_data(img)
        ax.set_title(f"frame {frame+1}/{frames}  span={span:.2e}  iters={maxiter}", fontsize=8)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=40)

    if output:
        ext = output.rsplit('.', 1)[-1].lower()
        if ext in ('mp4', 'm4v'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=25, metadata=dict(artist='fractal'), bitrate=2000)
            ani.save(output, writer=writer)
        elif ext == 'gif':
            ani.save(output, writer='imagemagick', fps=25)
        else:
            raise ValueError('Unsupported output format. Use mp4 or gif.')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Endless Mandelbrot zoom animation')
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--center', type=str, default='-0.743643887037151,0.13182590420533')
    parser.add_argument('--start_span', type=float, default=3.5)
    parser.add_argument('--zoom', type=float, default=0.96)
    parser.add_argument('--base_iter', type=int, default=300)
    parser.add_argument('--cmap', type=str, default='turbo')
    parser.add_argument('--output', type=str, default='fractal_output.mp4')

    args = parser.parse_args()
    cx, cy = map(float, args.center.split(','))

    make_animation((cx, cy), args.start_span, args.zoom, args.frames, args.width, args.height, args.base_iter, args.cmap, args.output)
