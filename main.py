"""
Multi-Agent Dynamic Grid World Environment
Created by: Ardianto Wibowo
"""

import numpy as np
import sys
import random

# Add the path to the 'env' folder to sys.path
sys.path.append('env')

from env.ma_gridworld import Env

class SearchAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.targets_seen = []
        self.noise_level = 0.5

    def analyse_sensor_data(self, agent_id, coordinate_observation, sensor_data_observation):
        for i in range(len(sensor_data_observation)):
            for j in range(len(sensor_data_observation[i])):
                data = sensor_data_observation[i][j]
                location = [coordinate_observation[0] + j - len(sensor_data_observation[i])//2,
                           coordinate_observation[1] + i - len(sensor_data_observation)//2]
                if data != None and 'target_' + str(agent_id) in data:
                    if location not in self.targets_seen:
                        self.targets_seen.append(location)
                # Remove collected targets from targets_seen
                if location[0] == coordinate_observation[0] and location[1] == coordinate_observation[1]:
                    if location in self.targets_seen:
                        self.targets_seen.remove(location)

    def analyse_communication(self, agent_id, comm_observation):
        for comm in comm_observation:
            origin_location = comm[0]
            sensor_data_observation = comm[1]
            for i in range(len(sensor_data_observation)):
                for j in range(len(sensor_data_observation[i])):
                    data = sensor_data_observation[i][j]
                    location = [origin_location[0] + j - len(sensor_data_observation[i])//2,
                               origin_location[1] + i - len(sensor_data_observation)//2]
                    if data != None and 'target_' + str(agent_id) in data:
                        if location not in self.targets_seen:
                            self.targets_seen.append(location)
                            print(f"agent {agent_id} track target in location {location}")

    def select_action(self, coordinate_observation):
        if len(self.targets_seen) > 0:
            # find the closest target
            closest_target = None
            closest_target_distance = 999999
            for target_coordinate in self.targets_seen:
                horizontal_distance = target_coordinate[0] - coordinate_observation[0]
                vertical_distance = target_coordinate[1] - coordinate_observation[1]
                distance = abs(horizontal_distance) + abs(vertical_distance)
                if distance < closest_target_distance:
                    closest_target_distance = distance
                    closest_target = target_coordinate
            horizontal_distance = closest_target[0] - coordinate_observation[0]
            vertical_distance = closest_target[1] - coordinate_observation[1]
            if abs(horizontal_distance) >= abs(vertical_distance):
                if horizontal_distance < 0:
                    return 3
                elif horizontal_distance > 0:
                    return 4
                return 0
            else:
                if vertical_distance < 0:
                    return 1
                elif vertical_distance > 0:
                    return 2
                return 0
        else:
            return np.random.choice(num_actions-1) + 1 # example of random value as a physical action


# Draft Start
# Understand tuple and list
# understand random package
# you can make max range for random distance greater than the sensors range
def add_coordinate_noise(coordinate, sensor_data, agent_id, enable=False):
    # if not enable return
    if not enable:
        return coordinate, sensor_data

    # read current agent noise level to a local variable noise_level
    noise_level = agents[agent_id].noise_level
    if noise_level > random.random():

        # read current agent x location  to a local variable location_x
        location_x = coordinate[0]
        location_y = coordinate[1]

        # choose a random direction left (-1) or right (1)
        available_x_directions = []
        if location_x > 0:
            available_x_directions.append(-1)
        if location_x < env.WIDTH - 1:
            available_x_directions.append(1)
        direction_x = random.choice(available_x_directions)

        # add distance for coordinate x to get new x instead of zero
        if direction_x == -1:
            new_x = random.randint(0, location_x - 1)
        else:
            new_x = random.randint(location_x + 1, env.WIDTH - 1)

        # choose a random direction top (-1) or butttom (1)
        available_y_directions = []
        if location_y > 0:
            available_y_directions.append(-1)
        if location_y < env.HEIGHT - 1:
            available_y_directions.append(1)
        direction_y = random.choice(available_y_directions)

        # add distance for coordinate y to get new y instead of zero
        if direction_y == -1:
            new_y = random.randint(0, location_y - 1)
        else:
            new_y = random.randint(location_y + 1, env.HEIGHT - 1)

        ## Fix sensor data
        # loop on all rows of sensor data rows
        for i in range(len(sensor_data)):
            # loop on all columns of sensor data rows
            for j in range(len(sensor_data[i])):
                # set get the location of cell data
                location = [new_x + j - len(sensor_data[i]) // 2,
                            new_y + i - len(sensor_data) // 2]

                if sensor_data[i][j] is not None:
                    # check if cell outside boarders
                    if location[0] < 0 or location[0] >= env.WIDTH or location[1] < 0 or location[1] >= env.HEIGHT:
                        sensor_data[i][j] = None

        return ((new_x, new_y), sensor_data)

    return coordinate, sensor_data


def add_sensor_data_noise(sensor_data, agent_id, enable=True):
    # if not enable return
    if not enable:
        return sensor_data

    # read current agent noise level to a local variable noise_level
    noise_level = agents[agent_id].noise_level

    if noise_level > random.random():
        # loop on all rows of sensor data rows
        for i in range(len(sensor_data)):
            # loop on all columns of sensor data rows
            for j in range(len(sensor_data[i])):
                # check if cell contain a target
                if sensor_data[i][j] is not None and 'target_' in sensor_data[i][j]:
                    # get agent id from data
                    target_agent_id = int(sensor_data[i][j].split("_")[1])
                    if agent_id != target_agent_id:
                        # Create list of all agents ids
                        target_agent_list = list(range(len(agents)))
                        # Create target agent id from targets list
                        target_agent_list.remove(target_agent_id)
                        # remove current agent from list if
                        # Get random agent from the target list
                        new_target_agent = random.choice(target_agent_list)
                        new_target_id = random.randint(0, env.num_targets_per_agent - 1)
                        sensor_data[i][j] = 'target_' + str(new_target_agent) + "_" + str(new_target_id)

    return sensor_data

def get_action(agent_id, observation, num_actions, agents, env):
    """
    This method provide a random action chosen recognized by the ma-gridworld environment:
    1: up, 2: down, 3: left, 4: right, 0: stay
    """

    coordinate_observation = tuple(observation[0])  # Keep observation as (x, y) tuple

    #optional observation data may be used, depend on the agent needs.
    win_state_observation = observation[1]
    sensor_data_observation = observation[2]
    comm_observation = observation[3]

    agents[agent_id].analyse_sensor_data(agent_id, coordinate_observation, sensor_data_observation)
    agents[agent_id].analyse_communication(agent_id, comm_observation)
    physical_action = agents[agent_id].select_action(coordinate_observation)

    if env.is_agent_silent:
        comm_action = [] # communication action is set to be zero if agent silent
    else:
        # Draft Start
        coordinate_observation, sensor_data_observation = add_coordinate_noise(coordinate_observation, sensor_data_observation, agent_id)
        sensor_data_observation = add_sensor_data_noise(sensor_data_observation, agent_id)
        # Draft End

        comm_action = [coordinate_observation, sensor_data_observation] # example of random value as a communication action

    return (physical_action, comm_action)


results = []
def run(num_episodes, max_steps_per_episode, agents, num_actions, env):
    import pandas as pd
    import matplotlib.pyplot as plt

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")

        # Reset environment
        observations = env.reset()

        # حذف الأهداف القديمة وإنشاء أهداف جديدة
        for target_list in env.target_objs:
            for target in target_list:
                env.canvas.delete(target)

        env.agent_targets = [[] for _ in range(env.num_agents)]
        env.target_objs = [[] for _ in range(env.num_agents)]

        for i in range(env.num_agents):
            for _ in range(env.num_targets_per_agent):
                target_position = env.get_random_target_position()
                env.agent_targets[i].append(target_position)

                target_x, target_y = target_position
                triangle_points = [
                    target_x, target_y - env.UNIT / 4,
                              target_x - env.UNIT / 4, target_y + env.UNIT / 4,
                              target_x + env.UNIT / 4, target_y + env.UNIT / 4
                ]
                target_obj = env.canvas.create_polygon(
                    triangle_points,
                    fill=env.agent_colors[i % len(env.agent_colors)],
                    outline='black'
                )
                env.target_objs[i].append(target_obj)

        print(f"Reinitialized targets for episode {episode + 1}: {env.agent_targets}")

        # Tracking variables
        done = [False] * env.num_agents
        step_count = 0
        agent_steps = {agent_id: 0 for agent_id in range(env.num_agents)}  # Track steps per agent
        agent_goals_achieved = {agent_id: 0 for agent_id in range(env.num_agents)}  # Track goals achieved per agent

        while not all(done) and step_count < max_steps_per_episode:
            actions = []
            for agent_id in range(env.num_agents):
                if not done[agent_id]:  # Only process agents still active
                    observation = observations[agent_id]
                    action = get_action(agent_id, observation, num_actions, agents, env)
                    actions.append(action)
                    agent_steps[agent_id] += 1  # Increment step count
                else:
                    actions.append((0, []))  # No action for agents marked as done

            observations, rewards, done = env.step(actions)

            # Check if goals are achieved and update counters
            for agent_id in range(env.num_agents):
                if rewards[agent_id] > 0:  # Positive reward implies a goal achieved
                    agent_goals_achieved[agent_id] += 1

            step_count += 1
            env.render()

        # Log results for this episode
        for agent_id in range(env.num_agents):
            results.append({
                "Agent ID": agent_id,
                "Noise Level": getattr(agents[agent_id], 'noise_level', 0),  # Default noise level to 0 if not available
                "Steps Taken": agent_steps[agent_id],
                "Goals Achieved": agent_goals_achieved[agent_id],
                "Episode": episode + 1
            })

        print(f"Episode {episode + 1} finished after {step_count} steps.\n")

    # Save results to Excel
    df = pd.DataFrame(results)

    # Create summary statistics grouped by noise level
    summary = df.groupby("Noise Level").agg({
        "Steps Taken": ["mean", "std"],  # Mean and standard deviation for steps
        "Goals Achieved": ["mean", "std", "sum"]  # Mean, std, and total for goals
    }).reset_index()

    # Rename columns for clarity
    summary.columns = ["Noise Level", "Avg Steps Taken", "Steps Std Dev", "Avg Goals Achieved", "Goals Std Dev",
                       "Total Goals"]

    # Write results and summary to Excel
    with pd.ExcelWriter("experiment_results.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name="Detailed Results")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        print("Results saved to 'experiment_results.xlsx' with a summary sheet.")

    # Plot results
    # Plot steps taken vs noise level
    plt.figure(figsize=(12, 8))

    # Plotting average steps taken for each noise level
    plt.plot(
        summary["Noise Level"],
        summary["Avg Steps Taken"],
        marker="o",
        label="Average Steps Taken"
    )

    # Adding labels and title
    plt.xlabel("Noise Level")
    plt.ylabel("Average Steps Taken")
    plt.title("Average Steps Taken vs Noise Level")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig("steps_vs_noise_level.png")
    plt.show()

    print("Plot saved to 'steps_vs_noise_level.png'.")




if __name__ == "__main__":

    gsize=15 #grid size (square)
    gpixels=30 #grid cell size in pixels

    is_sensor_active = True #True:  Activate the sensory observation data
    sensory_size = 3 #'is_sensor_active' must be True. The value must be odd, if event will be converted to one level odd number above

    num_agents = 8 #the number of agents will be run in paralel
    num_obstacles = 0 #the number of obstacles
    is_single_target = False #True: all agents have a single target, False: each agent has their own target
    num_targets_per_agent = 10 #'is_single_target' must be true to have an effect

    is_agent_silent = False #True: communication among agents is allowed

    num_episodes=1 #the number of episode will be run
    max_steps_per_episode=500 #each episode will be stopped when max_step is reached

    eps_moving_targets = 10 #set this value greater than 'num_episodes' to keep the targets in a stationary position
    eps_moving_obstacles = 10 #set this value greater than 'num_episodes' to keep the obstacles in a stationary position

    render = True #True: render the animation into the screen (so far, it is still can not be deactivated)

    min_obstacle_distance_from_target = 1 #min grid distance of each obstacles relative to targets
    max_obstacle_distance_from_target = 5 #max grid distance of each obstacles relative to targets
    min_obstacle_distance_from_agents = 1 #min grid distance of each obstacles relative to agents

    reward_normal = -1 #reward value of normal steps
    reward_obstacle = -5 #reward value when hit an obstacle
    reward_target = 50 #reward value when reach the target

    is_totally_random = True #True: target and obstacles initial as well as movement position is always random on each call, False: only random at the beginning.
    animation_speed = 0.2 #smaller is faster
    is_destroy_environment = True #True: automatically close the animation after all episodes end.

    for noise_level in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for trial in range(5):
            # Initialize environment
            env = Env(
                num_agents=num_agents, num_targets_per_agent=num_targets_per_agent, num_obstacles=num_obstacles,
                eps_moving_obstacles=eps_moving_obstacles, eps_moving_targets=eps_moving_targets,
                is_agent_silent=is_agent_silent, is_single_target=is_single_target, sensory_size=sensory_size,
                gpixels=gpixels, gheight=gsize, gwidth=gsize, is_sensor_active=is_sensor_active,
                min_obstacle_distance_from_target=min_obstacle_distance_from_target,
                max_obstacle_distance_from_target=max_obstacle_distance_from_target,
                min_obstacle_distance_from_agents=min_obstacle_distance_from_agents,
                is_totally_random=is_totally_random, animation_speed=animation_speed,
                reward_normal=reward_normal, reward_obstacle=reward_obstacle, reward_target=reward_target
            )

            num_actions = len(env.action_space)

            # Initialize Q-learning agents
            agents = [SearchAgent(num_actions) for _ in range(num_agents)]
            for agent in agents:
                agent.noise_level = noise_level

            # Run episodes
            run(num_episodes, max_steps_per_episode, agents, num_actions, env)

            if is_destroy_environment:
                env.destroy_environment()
