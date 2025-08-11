# script.py/start.py
# Endless fractal zoom animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        divtime[just_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break
    divtime[divtime == 0] = maxiter
    return divtime

def create_animation(output_file):
    center = (-0.743643887037151, 0.13182590420533)
    span = 3.5
    zoom_factor = 0.96
    frames = 120
    width, height = 640, 480
    base_iter = 300

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis('off')
    im = ax.imshow(np.zeros((height, width)), origin='lower', cmap='turbo', vmin=0, vmax=1)

    def update(frame):
        nonlocal span
        span *= zoom_factor
        xmin = center[0] - span/2
        xmax = center[0] + span/2
        ymin = center[1] - (span * height / width)/2
        ymax = center[1] + (span * height / width)/2
        maxiter = int(base_iter * (1 + frame / (frames*0.6)))
        arr = mandelbrot(xmin, xmax, ymin, ymax, width, height, maxiter)
        img = np.log(arr + 1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        im.set_data(img)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=40)
    ani.save(output_file, writer='ffmpeg', fps=25)

if __name__ == "__main__":
    create_animation("fractal_output.mp4")
