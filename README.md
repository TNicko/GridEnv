# MultiGridEnv: A Custom Gym Environment for Multi-Agent Gridworld Navigation

`MultiGridEnv` is a custom gym environment designed for multi-agent gridworld navigation tasks with partial observability. In this environment, multiple agents can navigate through a gridworld to explore and cover as many cells as possible while avoiding collisions with each other and walls.

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjBiZDMzMDI1NjU4ZDk2M2IyZDkzMzRkYTM1ODc2MTAwZDU2MjMyMCZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/et3ByoDiNEFXLdoFOU/giphy.gif)

## Documentation

### Initializing environment

Inputs:
- grid_map
- view_radius
Optional:
- traversal limit factor

#### grid_map
Create a grid map as a list of lists containing integers that represent the grid world. The integers represent different objects on the grid: 
- 0 = empty cell
- 1 = agent
- 2 = wall

`maps.py` contain a list of different grid_maps as examples.Example grid_map:
```python
    grid_map = [
    [0, 0, 0, 2, 0],
    [0, 1, 0, 2, 0],
    [0, 0, 0, 2, 0],
    [2, 2, 2, 2, 1],
    [0, 0, 0, 0, 0]
]
```

#### view_radius
Integer defining the radius of the agents observation area. A value of 1 means the agent can only see 1 grid cell out from each direction. Examples:
- 1 = 3x3 agent view area
- 2 = 5x5 agent view area

#### traversal limit factor
A float factor that determines the maximum number of cells an agent can visit before the episode terminates. If `None`, there is no traversal limit. 


