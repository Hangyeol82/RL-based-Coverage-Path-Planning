import os
import numpy as np


def show_path_plot(planner):
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = os.path.join("/tmp", "matplotlib")
        os.makedirs(mpl_cache, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = "/tmp"

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.collections import LineCollection

    if not planner.path:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = ListedColormap(["white", "black"])
    ax.imshow((planner.true_grid == 1).astype(int), cmap=cmap, origin="upper")

    points = np.array([(p[1], p[0]) for p in planner.path], dtype=float)
    if len(points) >= 2:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        progress = np.arange(len(segments), dtype=float)
        line = LineCollection(segments, cmap="turbo", linewidths=2.0, alpha=0.95, zorder=3)
        line.set_array(progress)
        line.set_clim(0.0, max(1.0, float(len(segments) - 1)))
        ax.add_collection(line)
        cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Path Progress")

    ax.scatter(points[0, 0], points[0, 1], color="tab:blue", s=40, label="Start", zorder=4)
    ax.scatter(points[-1, 0], points[-1, 1], color="tab:green", s=40, label="End", zorder=4)

    ax.set_title("C* Coverage Trajectory")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, planner.cols - 0.5)
    ax.set_ylim(planner.rows - 0.5, -0.5)
    ax.grid(which="major", color="lightgray", linestyle=":", linewidth=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
