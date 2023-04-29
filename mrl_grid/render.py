import matplotlib.pyplot as plt
import os

class WorldRenderer:
    """
    WorldRenderer renders a multi-agent gridworld instance using Matplotlib.
    """

    def __init__(self, title, world, fps):
        aspect_ratio = world.rows / world.cols
        self.fig, self.ax = plt.subplots(figsize=(8 * aspect_ratio, 8))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.world = world
        self.fps = fps
        self.fig.canvas.manager.set_window_title(title)
        self.title = title
        self.background = None

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.set_xlim(- 0.5, self.world.rows - 0.5)
        self.ax.set_ylim(self.world.cols - 0.5, - 0.5)

        # Create grid layout.
        for i in range(self.world.rows):
            self.ax.axvline(i - 0.5, color='black', linewidth=0.3)
        for j in range(self.world.cols):
            self.ax.axhline(j - 0.5, color='black', linewidth=0.3)

        # Add the walls to the grid.
        for wall in self.world.walls:
            self.ax.add_patch(wall.image)
    
        self.last_seen_cell_index = -1
        self.last_trail_index = -1

    def render_static_elements(self, filters):
        # Render new seen_cells.
        # if seen_cells is True in filters, render seen_cells
        if "seen_cells" not in filters:
            for i, cell in enumerate(self.world.seen_cells[self.last_seen_cell_index + 1:], self.last_seen_cell_index + 1):
                x, y = cell.pos
                cell.image.set_xy((y - 0.5, x - 0.5))
                self.ax.add_patch(cell.image)
                self.last_seen_cell_index = i

        # Render new trails.
        if "trails" not in filters:
            for i, trail_segment in enumerate(self.world.trails[self.last_trail_index + 1:], self.last_trail_index + 1):
                self.ax.add_patch(trail_segment.image)
                self.last_trail_index = i

        # Save the background (static elements).
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def render(self, filters=[], save_img=False, episode=None):
        """
        Render the world or update the image being shown.
        """
        self.render_static_elements(filters)

        # Restore the background (static elements).
        self.fig.canvas.restore_region(self.background)

        # Add agents to the grid at their current positions.
        for agent in self.world.agents:
            x, y = agent.pos
            agent.image.center = (y, x)
            self.ax.add_patch(agent.image)

        # Add agents to the grid at their current positions.
        for agent in self.world.agents:
            x, y = agent.pos
            agent.image.center = (y, x)
            self.ax.add_patch(agent.image)
            self.ax.draw_artist(agent.image)


        self.ax.set_aspect('equal', adjustable='box')

        # Request the window be redrawn.
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if save_img:
            self.save_image(episode, filters)

        # Pause for a moment to control the frame rate.
        plt.pause(1 / self.fps)

    def save_image(self, episode, filters):
        image_folder = "images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # get world size
        rows = self.world.rows
        cols = self.world.cols

        image_type = ""
        if "seen_cells" in filters:
            image_type = "_trails"
        elif "trails" in filters:
            image_type = "_seen_cells"

        image_name = f"{rows}x{cols}_ep{episode}{image_type}.png"
        image_path = os.path.join(image_folder, image_name)
        plt.savefig(image_path)

    def render_image(self, episode):

        # Render & save three types of images
        self.render(filters=["trails"], save_img=True, episode=episode)
        self.remove_patches()
        self.add_patches(filters=["seen_cells"])
        self.render(filters=["seen_cells"], save_img=True, episode=episode)
        self.remove_patches()
        self.add_patches(filters=[])
        self.render(filters=[], save_img=True, episode=episode)

    def remove_patches(self):
        # Remove seen_cells and trails patches.
        for cell in self.world.seen_cells:
            if cell.image in self.ax.collections:
                cell.image.remove()
            elif cell.image in self.ax.patches:
                cell.image.remove()

        for trail_segment in self.world.trails:
            if trail_segment.image in self.ax.collections:
                self.ax.collections.remove(trail_segment.image)
            elif trail_segment.image in self.ax.patches:
                trail_segment.image.remove()

        # Restore the background (static elements).
        # self.fig.canvas.restore_region(self.background)
    
    def add_patches(self, filters=[]):
        
        if "seen_cells" not in filters:
            # Add the seen_cells to the grid.
            for cell in self.world.seen_cells:
                self.ax.add_patch(cell.image)

        if "trails" not in filters:
            # Add the trails to the grid.
            for trail_segment in self.world.trails:
                self.ax.add_patch(trail_segment.image)

    def show(self):
        plt.ion()

    def close(self):
        plt.close()
