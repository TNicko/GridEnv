import mrl_grid.render_entities as re

AGENT_COLORS = [
    {
    'color': '#e8074a',
    'color_cell': '#e68aa5',
    'color_trail': '#CC315F',
    },
    {
    'color': '#003366',
    'color_cell': '#6da4db',
    'color_trail': '#2465A7',
    },
    {
    'color': '#FF6600',
    'color_cell': '#faa66e',
    'color_trail': '#BA662E',
    },
]

class World(object):
    """World object that contains all entities in the environment"""
    def __init__(self, rows, cols):
        self._agents = []
        self.seen_cells = []
        self.trails = []
        self.walls = []
        self.cells = 0

        self.rows = rows
        self.cols = cols
        
    @property
    def entities(self):
        "return all entities"
        return self._agents + self.seen_cells + self.trails + self.walls
    
    @property
    def agents(self):
        "return all agents"
        return self._agents

    @agents.setter
    def agents(self, value):
        self._agents = value

    def cell_visited(self, pos, agent):
        "mark new cell as visited"
        cell = SeenCell(pos, agent)
        agent.color_cell = AGENT_COLORS[agent.agent_id]['color_cell']
        agent.cells_covered += 1
        self.seen_cells.append(cell)

    def add_trail(self, old_pos, new_pos, agent):
        "add new trail segment"
        if old_pos == new_pos:
            return

        # check if trail already exists 
        max_curve_trail = max(
            (trail for trail in self.trails if tuple(sorted((trail.old_pos, trail.new_pos))) == tuple(sorted((old_pos, new_pos)))),
            key=lambda t: t.curve_no,
            default=None,
        )  
        if max_curve_trail:
            new_trail = TrailSegment(old_pos, new_pos, agent.agent_id, curve_no=max_curve_trail.curve_no + 1)
        else:
            new_trail = TrailSegment(old_pos, new_pos, agent.agent_id, curve_no=0)

        self.trails.append(new_trail)

    def get_coverage(self):
        total_covered_cells = len(self.seen_cells)
        overall_coverage = round((total_covered_cells / self.cells) * 100)

        individual_coverage = {}
        for agent in self._agents:
            agent_coverage = round((agent.cells_covered / self.cells) * 100)
            individual_coverage[agent.agent_id] = agent_coverage

        return overall_coverage, individual_coverage

    def get_steps_taken(self):
        steps_taken = {}
        for agent in self._agents:
            steps_taken[agent.agent_id] = agent.steps_taken

        return steps_taken
    
    def get_cell(self, pos):
        for cell in self.seen_cells:
            if cell.pos == pos:
                return cell
        return None
    
    def check_agent(self, pos):
        for agent in self._agents:
            if agent.pos == pos:
                return agent
        return None
    
    def check_wall(self, pos):
        for wall in self.walls:
            if wall.pos == pos:
                return wall
        return None

    def is_cell_visited(self, pos):
        return any(cell.pos == pos for cell in self.seen_cells)

    def all_cells_visited(self):
        return len(self.seen_cells) == self.cells

class Entity(object):
    def __init__(self):
        self.name = ''
        self.size = 0.050
        self.movable = False
        self.collide = True
        self.color = None
        self.pos = None

class Agent(Entity):
    def __init__(self, agent_id, init_pos):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.pos = init_pos
        self.movable = True
        self.action = None
        self.size = 0.3
        self.color = AGENT_COLORS[self.agent_id]['color']
        self.color_cell = AGENT_COLORS[self.agent_id]['color_cell']
        self.color_trail = AGENT_COLORS[self.agent_id]['color_trail']
        self.steps_taken = 0
        self.cells_covered = 0
        self.image = re.get_agent_img(self.color, self.size)
        self.collided = False

    def get_new_pos(self, action):
        x, y = self.pos

        if action == 0: x -= 1  # up
        if action == 1: x += 1  # down
        if action == 2: y -= 1  # left
        if action == 3: y += 1  # right
        if action == 4: pass    # wait

        return (x, y)

class SeenCell(Entity):
    def __init__(self, pos, agent):
        super(SeenCell, self).__init__()
        self.collide = False
        self.agent_id = agent.agent_id
        self.color = agent.color_cell
        self.pos = pos
        self.image = re.get_seen_cell_img(self.color, self.pos)
        self.seen_counter = 0

class TrailSegment(Entity):
    def __init__(self, old_pos, new_pos, agent_id, curve_no):
        super(TrailSegment, self).__init__()
        self.collide = False
        self.agent_id = agent_id
        self.new_pos = new_pos
        self.old_pos = old_pos
        self.color = AGENT_COLORS[agent_id]["color_trail"]
        self.curve_no = curve_no
        self.image = re.get_trail_img(self.color, self.old_pos, self.new_pos, self.curve_no)

class Wall(Entity):
    def __init__(self, pos):
        super(Wall, self).__init__()
        self.pos = pos
        self.color = '#000000'
        self.image = re.get_wall_img(self.color, self.pos)