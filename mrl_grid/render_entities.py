import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def get_agent_img(color, size, zorder=10):
    img = Circle((0, 0), radius=size, color=color, zorder=zorder)
    return img

def get_seen_cell_img(color, pos, zorder=1):
    """Returns a matplotlib patch object for a seen cell."""
    x, y = pos
    img = plt.Rectangle((x-0.5, y-0.5), 1, 1, facecolor=color, zorder=zorder)
    return img

def get_wall_img(color, pos):
    """Returns a matplotlib patch object for a wall."""
    x, y = pos
    img = plt.Rectangle((y-0.5, x-0.5), 1, 1, facecolor=color)
    return img

def get_trail_img(color, old_pos, new_pos, curve_no, zorder=2):
    """Returns a matplotlib patch object for a trail segment."""
    old_x, old_y = old_pos
    new_x, new_y = new_pos

    ctrl_shift = 0.05 * curve_no
    if old_x == new_x:  # Horizontal movement
        ctrl_shift *= -1 if old_pos < new_pos else 1
        ctrl_x1, ctrl_y1 = ((old_y + new_y) / 2, old_x - ctrl_shift)
        ctrl_x2, ctrl_y2 = ((old_y + new_y) / 2, new_x - ctrl_shift)
    else:  # Vertical movement
        ctrl_shift *= -1 if old_pos < new_pos else 1
        ctrl_x1, ctrl_y1 = (old_y - ctrl_shift, (old_x + new_x) / 2)
        ctrl_x2, ctrl_y2 = (new_y - ctrl_shift, (old_x + new_x) / 2)

    # Data for the trail curve
    path_data = [
        (mpath.Path.MOVETO, (old_y, old_x)),
        (mpath.Path.CURVE4, (ctrl_x1, ctrl_y1)),
        (mpath.Path.CURVE4, (ctrl_x2, ctrl_y2)),
        (mpath.Path.CURVE4, (new_y, new_x)),
    ]

    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    img = mpatches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=0.4, zorder=zorder)

    return img
