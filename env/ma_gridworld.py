"""
Multi-Agent Dynamic Grid World Environment
Created by: Ardianto Wibowo
"""

import time
import numpy as np
import tkinter as tk
import random

# np.random.seed(1)

class Env(tk.Tk):
    def __init__(self, num_agents=2, num_targets_per_agent=3, num_obstacles=2, eps_moving_obstacles=20, eps_moving_targets=5, is_agent_silent=False, is_single_target=False, sensory_size=3, gpixels=50, gheight=20, gwidth=20, is_sensor_active=True, 
                 min_obstacle_distance_from_target=1, max_obstacle_distance_from_target=3, min_obstacle_distance_from_agents=2, is_totally_random=False, animation_speed = 0.005,reward_normal=-1, reward_obstacle=-5, reward_target=50):
        
        super(Env, self).__init__()
        
        if(is_totally_random):
            np.random.seed(int(time.time()))
        else:
            np.random.seed(1)
        
        self.title('Multi-Agent Dynamic Environment with Sensory Information')
        
        self.UNIT = gpixels
        self.HEIGHT = gheight
        self.WIDTH = gwidth
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.is_agent_silent = is_agent_silent
        self.is_single_target = is_single_target  # New variable for single target option
        self.single_target = None  # Initialize single target reference
        self.single_target_direction = (0, 0)  # Initialize direction for single target
        self.geometry('{0}x{1}'.format(self.HEIGHT * self.UNIT, self.HEIGHT * self.UNIT))
        self.grid_colors = [[(255, 255, 255)] * self.WIDTH for _ in range(self.HEIGHT)]
        self.texts = []
        self.episode_count = 0  # Initialize episode counter
        self.sensory_size = sensory_size
        self.sensory_grid_on = is_sensor_active
        self.min_obstacle_distance_from_target = min_obstacle_distance_from_target  # Minimum grid distance from the target
        self.max_obstacle_distance_from_target = max_obstacle_distance_from_target  # Maximum grid distance from the target
        self.min_obstacle_distance_from_agents = min_obstacle_distance_from_agents  # Minimum distance from agents
        self.is_totally_random = is_totally_random
        self.animation_speed = animation_speed
        self.reward_normal = reward_normal
        self.reward_obstacle = reward_obstacle
        self.reward_target = reward_target

         # Colors for agents and their corresponding targets
        self.agent_colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]

        # Multi-agent setup
        self.num_agents = num_agents
        self.num_targets_per_agent = num_targets_per_agent
        self.num_obstacles = num_obstacles  # Number of obstacles is now dynamic
        self.eps_moving_obstacles = eps_moving_obstacles
        self.eps_moving_targets = eps_moving_targets
        self.agents = []
        self.agent_targets = [[] for _ in range(num_agents)] 
        self.initial_agent_positions = []
        self.messages = []
        self.obstacles = []  # List to hold obstacle objects
        self.obstacle_positions = []  # Store the positions of obstacles
        self.obstacle_directions = [(1, 0), (0, 1)] * (num_obstacles // 2) + [(-1, 0), (0, -1)] * ((num_obstacles + 1) // 2)
        self.first_agent_reached = False
        self.mega_bonus_given = False
        self.win_flag = False
        self.locked = [False] * self.num_agents
        self.win = [False] * self.num_agents 
        self.next_state_comms = [[] for _ in range(self.num_agents)]
        self.bonus_reward=0
        
        self.init_agents()
        self.canvas = self._build_canvas()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Ensure cleanup when closing window


    def init_agents(self):
        self.agents = []
        self.target_objs = [[] for _ in range(self.num_agents)]  # Store multiple target objects per agent
        self.target_directions = [[] for _ in range(self.num_agents)]  # Direction for each target

        # Use previous position logic to position agents along the edges
        positions = [
            [self.UNIT / 2, self.UNIT / 2],  # Agent 1: Top-left
            [(self.WIDTH - 0.5) * self.UNIT, self.UNIT / 2],  # Agent 2: Top-right
            [(self.WIDTH - 0.5) * self.UNIT, (self.HEIGHT - 0.5) * self.UNIT],  # Agent 3: Bottom-right
            [self.UNIT / 2, (self.HEIGHT - 0.5) * self.UNIT]  # Agent 4: Bottom-left
        ]

        # Function to calculate evenly spaced positions along an edge and snap to the grid center
        def evenly_spaced_positions(start, end, num_agents):
            positions = []
            step_x = (end[0] - start[0]) / (num_agents + 1)
            step_y = (end[1] - start[1]) / (num_agents + 1)
            for i in range(1, num_agents + 1):
                pos_x = start[0] + i * step_x
                pos_y = start[1] + i * step_y
                snapped_x = (pos_x // self.UNIT) * self.UNIT + self.UNIT / 2
                snapped_y = (pos_y // self.UNIT) * self.UNIT + self.UNIT / 2
                positions.append([snapped_x, snapped_y])
            return positions

        # Calculate the edge distribution for agents
        num_edges = 4
        edge_distribution = [0] * num_edges
        remaining_agents = self.num_agents - 4
        for i in range(remaining_agents):
            edge_distribution[i % num_edges] += 1

        # Add agents to the edges based on calculated distribution
        top_edge_positions = evenly_spaced_positions(positions[0], positions[1], edge_distribution[0])
        positions += top_edge_positions
        right_edge_positions = evenly_spaced_positions(positions[1], positions[2], edge_distribution[1])
        positions += right_edge_positions
        bottom_edge_positions = evenly_spaced_positions(positions[2], positions[3], edge_distribution[2])
        positions += bottom_edge_positions
        left_edge_positions = evenly_spaced_positions(positions[3], positions[0], edge_distribution[3])
        positions += left_edge_positions

        self.initial_agent_positions = positions.copy()
        
        for i in range(self.num_agents):
            agent = {'id': i, 'coords': self.initial_agent_positions[i]}
            self.agents.append(agent)

        # Initialize targets based on `is_single_target`
        if self.is_single_target:
            # Set a single target at the center with a shared direction
            center_x = (self.WIDTH // 2) * self.UNIT + self.UNIT / 2
            center_y = (self.HEIGHT // 2) * self.UNIT + self.UNIT / 2
            self.single_target = [center_x, center_y]
            self.single_target_direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])#target movement direction (if active)
    
        else:
            # Default behavior: each agent has its own targets with random positions and directions
            self.agent_targets = [[] for _ in range(self.num_agents)]
            self.target_directions = [[] for _ in range(self.num_agents)]
            for i in range(self.num_agents):
                for _ in range(self.num_targets_per_agent):
                    target_position = self.get_random_target_position()
                    self.agent_targets[i].append(target_position)
                    direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)]) #obstacles movement direction (if active)
                    self.target_directions[i].append(direction)
        self.messages = [None] * len(self.agents)


    def get_random_target_position(self):
        x = np.random.randint(1, self.WIDTH - 1) * self.UNIT + self.UNIT / 2
        y = np.random.randint(1, self.HEIGHT - 1) * self.UNIT + self.UNIT / 2
        return [x, y]


    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=self.HEIGHT * self.UNIT, width=self.WIDTH * self.UNIT)
        for r in range(0, self.HEIGHT * self.UNIT, self.UNIT):
            for c in range(0, self.WIDTH * self.UNIT, self.UNIT):
                x0, y0, x1, y1 = c, r, c + self.UNIT, r + self.UNIT
                grid_color = self.grid_colors[r // self.UNIT][c // self.UNIT]
                canvas.create_rectangle(x0, y0, x1, y1, fill=self.rgb_to_hex(grid_color), outline='black')

        # Draw agents and their targets on the canvas
        self.target_objs = [[] for _ in range(self.num_agents)]
        
        for i, agent in enumerate(self.agents):
            agent_center_x, agent_center_y = agent['coords']
            agent_color = self.agent_colors[i % len(self.agent_colors)]
            agent['image_obj'] = canvas.create_oval(agent_center_x - self.UNIT / 4, agent_center_y - self.UNIT / 4,
                                                    agent_center_x + self.UNIT / 4, agent_center_y + self.UNIT / 4,
                                                    fill=agent_color, outline='black')
            
            if self.is_single_target:
                # Draw the single target in the center or starting position
                if i == 0:  # Only add one target object
                    target_x, target_y = self.single_target
                    triangle_points = [
                        target_x, target_y - self.UNIT / 4,
                        target_x - self.UNIT / 4, target_y + self.UNIT / 4,
                        target_x + self.UNIT / 4, target_y + self.UNIT / 4
                    ]
                    target_obj = canvas.create_polygon(triangle_points, fill="green", outline='black')
                    self.target_objs[0].append(target_obj)
            else:
                # Draw multiple targets for each agent
                for target_position in self.agent_targets[i]:
                    target_x, target_y = target_position
                    triangle_points = [
                        target_x, target_y - self.UNIT / 4,
                        target_x - self.UNIT / 4, target_y + self.UNIT / 4,
                        target_x + self.UNIT / 4, target_y + self.UNIT / 4
                    ]
                    target_obj = canvas.create_polygon(triangle_points, fill=agent_color, outline='black')
                    self.target_objs[i].append(target_obj)


        # Place obstacles
        center_x = (self.WIDTH // 2) * self.UNIT + self.UNIT / 2
        center_y = (self.HEIGHT // 2) * self.UNIT + self.UNIT / 2

        for i in range(self.num_obstacles):
            while True:
                x_offset = np.random.randint(-self.max_obstacle_distance_from_target, self.max_obstacle_distance_from_target + 1) * self.UNIT
                y_offset = np.random.randint(-self.max_obstacle_distance_from_target, self.max_obstacle_distance_from_target + 1) * self.UNIT

                if abs(x_offset) >= self.min_obstacle_distance_from_target * self.UNIT or abs(y_offset) >= self.min_obstacle_distance_from_target * self.UNIT:
                    x = center_x + x_offset
                    y = center_y + y_offset
                    pos = (x, y)

                    # Check if the position is valid
                    distance_from_targets = all(
                        np.linalg.norm(np.array(pos) - np.array(target_position)) >= self.min_obstacle_distance_from_target * self.UNIT
                        for agent_targets in self.agent_targets for target_position in agent_targets
                    )

                    valid_position_from_agents = all(
                        np.linalg.norm(np.array(pos) - np.array(agent['coords'])) >= self.min_obstacle_distance_from_agents * self.UNIT
                        for agent in self.agents
                    )

                    if (self.UNIT / 2 <= x <= (self.WIDTH - 0.5) * self.UNIT and
                        self.UNIT / 2 <= y <= (self.HEIGHT - 0.5) * self.UNIT and
                        distance_from_targets and valid_position_from_agents and pos not in self.obstacle_positions):
                        
                        self.obstacle_positions.append(pos)
                        obstacle = canvas.create_rectangle(pos[0] - self.UNIT / 4, pos[1] - self.UNIT / 4,
                                                           pos[0] + self.UNIT / 4, pos[1] + self.UNIT / 4,
                                                           fill='red', outline='black')
                        self.obstacles.append(obstacle)
                        break

        canvas.pack()
        return canvas

    def reset(self):
        self.update()
        time.sleep(0.5)
        for agent, initial_pos in zip(self.agents, self.initial_agent_positions):
            x, y = initial_pos
            self.canvas.coords(agent['image_obj'], x - self.UNIT / 4, y - self.UNIT / 4, x + self.UNIT / 4, y + self.UNIT / 4)
            agent['coords'] = [x, y]
        self.update_grid_colors()
        self.messages = [None] * len(self.agents)
        self.locked = [False] * self.num_agents
        self.win = [False] * self.num_agents
        self.next_state_comms = [[0] for _ in range(self.num_agents)]
        self.episode_count += 1
        if self.episode_count % self.eps_moving_obstacles == 0:
            self.move_obstacles()
        if self.episode_count % self.eps_moving_targets == 0:
            self.move_targets()

        # Collect initial observations
        observations = []
        for agent in self.agents:
            state = self.coords_to_state(agent['coords'])
            sensory_grid = self.get_sensory_grid(agent['coords'])
            communication_observation = [] if self.is_agent_silent else []
            observation = [state, False, sensory_grid, communication_observation]
            observations.append(observation)

        return observations

    def step(self, actions):
        rewards = []
        dones = []
        next_states = []
        self.update_grid_colors()
        all_reached = True 


        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if self.locked[idx]:  # If the agent is already locked (reached target or hit obstacle)
                rewards.append(0)
                dones.append(True)
                next_state_obs = self.coords_to_state(agent['coords'])
                sensory_grid = self.get_sensory_grid(agent['coords'])
                next_states.append([next_state_obs, self.win[idx], sensory_grid, self.next_state_comms[idx]])
                print(f"Agent {idx} is locked. Done status: {self.locked[idx]}, win status: {self.win[idx]}")
                all_reached = all_reached and self.win[idx]
                continue

            state = agent['coords']
            base_action = np.array([0, 0])
            physical_action = action[0]

            if physical_action == 1:  # up
                if state[1] > self.UNIT:
                    base_action[1] -= self.UNIT
            elif physical_action == 2:  # down
                if state[1] < (self.HEIGHT - 1) * self.UNIT:
                    base_action[1] += self.UNIT
            elif physical_action == 3:  # left
                if state[0] > self.UNIT:
                    base_action[0] -= self.UNIT
            elif physical_action == 4:  # right
                if state[0] < (self.WIDTH - 1) * self.UNIT:
                    base_action[0] += self.UNIT
            else:
                base_action = np.array([0, 0])  # Agent remains stationary

            # Apply action and update position
            new_coords = [state[0] + base_action[0], state[1] + base_action[1]]
            self.canvas.coords(agent['image_obj'], new_coords[0] - self.UNIT / 4, new_coords[1] - self.UNIT / 4,
                            new_coords[0] + self.UNIT / 4, new_coords[1] + self.UNIT / 4)
            next_state_grid = self.coords_to_state(new_coords)  # Convert to grid coordinates

            # Check if the agent reaches the single target (if `is_single_target` is True)
            if self.is_single_target:
                target_grid = self.coords_to_state(self.single_target)
                if next_state_grid == target_grid:
                    rewards.append(self.reward_target)
                    self.win[idx] = True
                    self.locked[idx] = True
                    dones.append(True)
                    print(f"Agent {idx} reached the single target! Next state: {next_state_grid}")
                else:
                    rewards.append(self.reward_normal)
                    dones.append(False)
                    all_reached = False 
            else:
                # Check if the agent reaches any of its own targets (for multiple targets setup)
                own_targets_grid = [self.coords_to_state(self.canvas.coords(target)) for target in self.target_objs[idx]]
                if next_state_grid in own_targets_grid:
                    rewards.append(self.reward_target)
                    self.win[idx] = True
                    self.locked[idx] = True
                    dones.append(True)
                    print(f"Agent {idx} reached one of its own targets! Next state: {next_state_grid}")

                # Check if agent hits another agent's target
                elif any(next_state_grid in [self.coords_to_state(self.canvas.coords(target)) for target in self.target_objs[other_idx]]
                        for other_idx in range(self.num_agents) if other_idx != idx):
                    rewards.append(self.reward_obstacle)
                    self.locked[idx] = True
                    dones.append(True)
                    all_reached = False 
                    print(f"Agent {idx} hit another agent's target! Next state: {next_state_grid}")
                else:
                    rewards.append(self.reward_normal)
                    dones.append(False)
                    all_reached = False 

            # Check if the agent hits an obstacle
            if next_state_grid in [self.coords_to_state(self.canvas.coords(obstacle)) for obstacle in self.obstacles]:
                rewards[-1] = self.reward_obstacle
                self.locked[idx] = True
                dones[-1] = True
                all_reached = False 
                print(f"Agent {idx} hit an obstacle! Next state: {next_state_grid}")

            # Get the next state and sensory grid for observation
            next_state_obs = self.coords_to_state(new_coords)
            sensory_grid = self.get_sensory_grid(new_coords)
            
            if not self.is_agent_silent:
                # Clear the communications for the current agent to avoid overwriting
                self.next_state_comms[idx] = []

                for other_agent in self.agents:
                    if other_agent == agent:
                        continue

                    other_agent_message = actions[other_agent['id']][1]
                    if other_agent_message:
                        self.next_state_comms[idx].append(other_agent_message)  # Collect messages from all other agents
            else:
                self.next_state_comms[idx] = []

            next_state_observation = [next_state_obs, self.win[idx], sensory_grid, self.next_state_comms[idx]]
            next_states.append(next_state_observation)

            # Update the agent's position in terms of coordinates
            agent['coords'] = new_coords

        # Apply bonus reward if all agents arrived simultaneously at the target
        if all(self.win) and all_reached:
            for idx in range(len(rewards)):
                rewards[idx] += self.bonus_reward  # Bonus reward
            print("All agents reached the target at the same time! Bonus reward applied.")


        # Highlight sensory grids for all agents (including locked ones)
        if self.sensory_grid_on:
            self.highlight_all_sensory_grids()

        return next_states, rewards, dones


    def highlight_all_sensory_grids(self):
        # Clear previous highlights
        self.update_grid_colors()

        if not self.sensory_grid_on:
            return  # Skip highlighting if the sensory grid is turned off

        half_size = self.sensory_size // 2

        # Highlight each agent's sensory grid
        for agent in self.agents:
            x, y = self.coords_to_state(agent['coords'])
            if self.locked[agent['id']] and self.win[agent['id']]:
                color = (144, 238, 144)  # Light green color (agent hit target)
            elif self.locked[agent['id']]:
                color = (255, 182, 193)  # Light red color (agent hit obstacle)
            else:
                color = (173, 216, 230)  # Light blue color (default)

            for r in range(y - half_size, y + half_size + 1):
                for c in range(x - half_size, x + half_size + 1):
                    if 0 <= r < self.HEIGHT and 0 <= c < self.WIDTH:
                        self.grid_colors[r][c] = color

        self._update_canvas_colors()

    def coords_to_state(self, coords):
        # Convert canvas coordinates to grid indices, keeping the agent in the center
        x = int((coords[0]) // self.UNIT)
        y = int((coords[1]) // self.UNIT)
        return [x, y]

    def get_sensory_grid(self, coords):
        if not self.sensory_grid_on:
            return None  # No sensory grid data if the sensory grid is turned off

        x, y = self.coords_to_state(coords)
        half_size = self.sensory_size // 2
        sensory_grid = []

        for r in range(y - half_size, y + half_size + 1):
            row = []
            for c in range(x - half_size, x + half_size + 1):
                if 0 <= r < self.HEIGHT and 0 <= c < self.WIDTH:
                    grid_content = self.get_grid_content(c, r)
                    row.append(grid_content)
                else:
                    row.append(None)  # Outside of bounds
            sensory_grid.append(row)

        return sensory_grid

    def get_grid_content(self, x, y):
        """
        Checks the content of a grid cell at (x, y).
        """
        # Check if any agent is in this grid cell
        for agent in self.agents:
            agent_coords = self.coords_to_state(agent['coords'])
            if agent_coords == [x, y]:
                return 'agent'

        # Check if any obstacle is in this grid cell
        for obstacle in self.obstacles:
            obstacle_coords = self.coords_to_state(self.canvas.coords(obstacle))
            if obstacle_coords == [x, y]:
                return 'obstacle'

        # Check if any target is in this grid cell
        for agent_idx, target_list in enumerate(self.target_objs):  # Loop over each agent's list of targets
            for target_idx, target_obj in enumerate(target_list):  # Loop over each target for that agent
                target_coords = self.coords_to_state(self.canvas.coords(target_obj))
                if target_coords == [x, y]:
                    return f'target_{agent_idx}_{target_idx}'  # Label target by agent and target index

        # If none of the above, return 'empty'
        return 'empty'



    def update_grid_colors(self, color=(255, 255, 255)):
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                self.grid_colors[r][c] = color
        self._update_canvas_colors()


    def _update_canvas_colors(self):
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                grid_color = self.grid_colors[r][c]
                rect_id = (r * self.WIDTH) + c + 1
                self.canvas.itemconfig(rect_id, fill=self.rgb_to_hex(grid_color))


    def rgb_to_hex(self, rgb):
        # Convert RGB tuple to hex string for Tkinter color filling
        return '#%02x%02x%02x' % rgb
    

    def move_obstacles(self):
        new_positions = set()  # Set to track new positions of obstacles

        for i, obstacle in enumerate(self.obstacles):
            direction = self.obstacle_directions[i % len(self.obstacle_directions)]
            x_move, y_move = direction[0] * self.UNIT, direction[1] * self.UNIT

            current_coords = self.obstacle_positions[i]
            new_x = current_coords[0] + x_move
            new_y = current_coords[1] + y_move

            new_pos = (new_x, new_y)
            
            distance_from_targets = [
                np.linalg.norm(np.array(new_pos) - np.array(target))
                for agent_targets in self.agent_targets for target in agent_targets
            ]

            # Check distance from agents
            valid_position_from_agents = all(
                np.linalg.norm(np.array(new_pos) - np.array(agent['coords'])) >= self.min_obstacle_distance_from_agents * self.UNIT
                for agent in self.agents
            )

            # Check for grid boundaries, obstacle overlap, target overlap, and whether the new position respects the min/max distance from the target
            if (
                self.UNIT / 2 <= new_x <= (self.WIDTH - 0.5) * self.UNIT and
                self.UNIT / 2 <= new_y <= (self.HEIGHT - 0.5) * self.UNIT and
                all(dist >= self.min_obstacle_distance_from_target * self.UNIT for dist in distance_from_targets) and
                new_pos not in new_positions and
                new_pos not in self.obstacle_positions
            ):
                self.canvas.move(obstacle, x_move, y_move)
                new_positions.add(new_pos)
                self.obstacle_positions[i] = new_pos
            else:
                self.obstacle_directions[i] = (-direction[0], -direction[1])  # Reverse direction if new position is invalid

    def move_targets(self):
        print(f"Moving targets for episode count {self.episode_count}")
        
        if self.is_single_target:
            # Move single shared target
            x_move, y_move = self.single_target_direction[0] * self.UNIT, self.single_target_direction[1] * self.UNIT
            new_x = self.single_target[0] + x_move
            new_y = self.single_target[1] + y_move

            # Ensure single target remains within boundaries
            if self.UNIT / 2 <= new_x <= (self.WIDTH - 0.5) * self.UNIT and \
            self.UNIT / 2 <= new_y <= (self.HEIGHT - 0.5) * self.UNIT:
                self.single_target = [new_x, new_y]
                target_obj = self.target_objs[0][0]  # The single target object on the canvas
                triangle_points = [
                    new_x, new_y - self.UNIT / 4,
                    new_x - self.UNIT / 4, new_y + self.UNIT / 4,
                    new_x + self.UNIT / 4, new_y + self.UNIT / 4
                ]
                self.canvas.coords(target_obj, *triangle_points)
                print(f"Single target moved to {self.single_target}")
            else:
                # Reverse direction if out of bounds
                self.single_target_direction = (-self.single_target_direction[0], -self.single_target_direction[1])
                print(f"Single target reversed direction to {self.single_target_direction}")
        else:
            # Default multi-target logic for each agent
            new_positions = set()
            for agent_idx, agent_targets in enumerate(self.agent_targets):
                for target_idx, target_position in enumerate(agent_targets):
                    direction = self.target_directions[agent_idx][target_idx]
                    x_move, y_move = direction[0] * self.UNIT, direction[1] * self.UNIT
                    new_x = target_position[0] + x_move
                    new_y = target_position[1] + y_move
                    new_pos = (new_x, new_y)

                    if self.UNIT / 2 <= new_x <= (self.WIDTH - 0.5) * self.UNIT and \
                    self.UNIT / 2 <= new_y <= (self.HEIGHT - 0.5) * self.UNIT and \
                    new_pos not in new_positions:
                            
                        self.agent_targets[agent_idx][target_idx] = [new_x, new_y]
                        target_obj = self.target_objs[agent_idx][target_idx]
                        triangle_points = [
                            new_x, new_y - self.UNIT / 4,
                            new_x - self.UNIT / 4, new_y + self.UNIT / 4,
                            new_x + self.UNIT / 4, new_y + self.UNIT / 4
                        ]
                        self.canvas.coords(target_obj, *triangle_points)
                        new_positions.add(new_pos)
                    else:
                        self.target_directions[agent_idx][target_idx] = (-direction[0], -direction[1])
                        
    def render(self):
        time.sleep(self.animation_speed)
        self.update()

    def destroy_environment(self):
        self.destroy()

    # Method to handle closing event
    def on_closing(self):
        self.destroy_environment()

if __name__ == "__main__":
    env = Env()
    env.mainloop()  # Running the main loop for the Tkinter GUI
