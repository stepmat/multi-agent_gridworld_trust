""" Ver10
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

    def __init__(self, agent_id, noise_level, init_trust_level, min_trust_level, num_actions):

        self.num_actions = num_actions
        self.targets_seen = []
        self.noise_level = noise_level
        self.agent_id = agent_id
        self.init_trust_level = init_trust_level  # round(random.uniform(0.3, 1.0), 2)
        self.min_trust_level = min_trust_level

        self.memory = []  # to store verified data

        self.true_interaction = {}
        self.false_interaction = {}

        # self.trust_values = {other_id: self.init_trust_level for other_id in range(num_agents) if other_id != self.agent_id}
        if isinstance(self.init_trust_level, dict):
            self.trust_values = self.init_trust_level
        else:
            self.trust_values = {other_id: self.init_trust_level for other_id in range(num_agents) if other_id != self.agent_id}

        print(f"Agent {self.agent_id}: Noise level {self.noise_level:.2f}, Initialized agents with trust level {self.trust_values}")


    def reset(self):
        self.memory = []

        self.true_interaction = {}
        self.false_interaction = {}

        self.targets_seen = []

        # self.trust_values = {other_id: self.init_trust_level for other_id in range(num_agents) if other_id != self.agent_id}
        if isinstance(self.init_trust_level, dict):
            self.trust_values = self.init_trust_level
        else:
            self.trust_values = {other_id: self.init_trust_level for other_id in range(num_agents) if other_id != self.agent_id}



    def analyse_sensor_data(self, agent_id, coordinate_observation, sensor_data_observation):
        for i in range(len(sensor_data_observation)):
            for j in range(len(sensor_data_observation[i])):
                data = sensor_data_observation[i][j]
                location = [coordinate_observation[0] + j - len(sensor_data_observation[i])//2,
                           coordinate_observation[1] + i - len(sensor_data_observation)//2]


                if data != None and 'target_' + str(agent_id) in data:
                    if location not in [target["location"] for target in self.targets_seen]:
                        self.targets_seen.append({
                            "location": location,
                            "verified": False,
                            "source": "self"
                        })
                        print(f"Agent {agent_id}: Added target at {location} to targets_seen")

                #    update the meemory agent_id
                if data is not None and 'target_' + str(agent_id) in data:  # Check for target
                    self.update_memory(location, data, agent_id)  #  update_memory



    def analyse_communication(self, agent_id, comm_observation, apply_trust_threshold=False):

        for comm in comm_observation:
            origin_location = comm[0]
            sensor_data_observation = comm[1]
            reported_by = comm[2]

            if apply_trust_threshold and self.trust_values[reported_by] < self.min_trust_level:
                print(f"Agent {agent_id}: Non trusted communication from Agent {reported_by}")
            else:
                for i in range(len(sensor_data_observation)):
                    for j in range(len(sensor_data_observation[i])):
                        data = sensor_data_observation[i][j]
                        location = [origin_location[0] + j - len(sensor_data_observation[i])//2,
                                   origin_location[1] + i - len(sensor_data_observation)//2]
                        if data != None and 'target_' + str(agent_id) in data:
                            if reported_by == 0 and self.agent_id == 3:
                                print("monitor")
                            if location not in [target["location"] for target in self.targets_seen]:

                                self.targets_seen.append({
                                    "location": location,
                                    "verified": False,
                                    "source": f"agent_{reported_by}"
                                })
                                print(f"Agent {agent_id}: Added target at {location} from Agent {reported_by}")
                                # 12/17/2024

    def select_action(self, coordinate_observation, agent_id):
        print(f"Agent {agent_id}: Current targets seen: {self.targets_seen}")

        self.targets_seen = list({tuple(target["location"]): target for target in self.targets_seen}.values())

        if len(self.targets_seen) > 0:
            closest_target = None
            closest_target_distance = float('inf')
            for target in self.targets_seen:
                target_coordinate = target["location"]
                horizontal_distance = target_coordinate[0] - coordinate_observation[0]
                vertical_distance = target_coordinate[1] - coordinate_observation[1]
                distance = abs(horizontal_distance) + abs(vertical_distance)
                print(f"Agent {agent_id}: Target {target_coordinate} at distance {distance}")

                if distance < closest_target_distance:
                    closest_target_distance = distance
                    closest_target = target_coordinate

            if closest_target:
                # Check the validity of the target
                grid_width = 15
                grid_height = 15
                if not (0 <= closest_target[0] < grid_width and 0 <= closest_target[1] < grid_height):
                    print(f"Invalid target location {closest_target}, skipping target.")
                    closest_target = None

                if closest_target and closest_target_distance == 0:
                    print(f"Agent {agent_id}: Reached target at {closest_target}, removing from targets_seen.")
                    self.targets_seen = [t for t in self.targets_seen if tuple(t["location"]) != tuple(closest_target)]
                    return 0

                if closest_target:
                    print(f"Agent {agent_id}: Moving towards target at {closest_target}")
                    horizontal_distance = closest_target[0] - coordinate_observation[0]
                    vertical_distance = closest_target[1] - coordinate_observation[1]
                    if abs(horizontal_distance) >= abs(vertical_distance):
                        return 3 if horizontal_distance < 0 else 4
                    else:
                        return 1 if vertical_distance < 0 else 2
        else:
            print(f"Agent {agent_id}: No valid targets, attempting random action.")
            possible_actions = [1, 2, 3, 4]
            return np.random.choice(possible_actions)


    def update_trust(self, other_agent_id, interaction_success, use_indirect=True, alpha1=.8, alpha2=.8, apply_trust_threshold=True):
        if not (apply_trust_threshold and self.trust_values[other_agent_id] < self.min_trust_level):

            if other_agent_id not in self.true_interaction:
                self.true_interaction[other_agent_id] = 0
                self.false_interaction[other_agent_id] = 0

            if interaction_success:
                self.true_interaction[other_agent_id] = self.true_interaction[other_agent_id] + 1
            else:
                self.false_interaction[other_agent_id] = self.false_interaction[other_agent_id] + 1

            if other_agent_id in self.trust_values:
                # privouse trust
                previous_trust = self.trust_values[other_agent_id]

                # the result of interaction
                interaction_result = 1.0 if interaction_success else 0.0

                witness_report = (self.true_interaction[other_agent_id] + 1) / (
                        self.true_interaction[other_agent_id] + self.false_interaction[other_agent_id] + 2)


                # Only direct trust
                updated_trust = alpha1 * previous_trust + (1 - alpha1) * (self.true_interaction[other_agent_id] / (self.true_interaction[other_agent_id] + self.false_interaction[other_agent_id]))


                previous_trust = updated_trust


                if use_indirect:
                    # Calculate Indirect trust
                    indirect_trust = 0.0
                    total_weight = 0.0

                    for intermediary_agent, intermediary_trust in self.trust_values.items():
                        if intermediary_agent != other_agent_id:
                            indirect_trust += (
                                    intermediary_trust * agents[intermediary_agent].trust_values[other_agent_id]
                            )
                            total_weight += intermediary_trust
                    indirect_trust = indirect_trust / total_weight

                    # direct and indirect trust
                    updated_trust = alpha2 * previous_trust + (1 - alpha2) * indirect_trust

                # Set the trust value to be between 0 and 1
                self.trust_values[other_agent_id] = max(0.0, min(1.0, updated_trust))

                trust_type = "Direct + Indirect" if use_indirect else "Direct"

                print(
                    f"Agent {self.agent_id} update trust for {other_agent_id}: :The previous trust value is: {previous_trust} : Updated trust: {updated_trust}  The interaction_result is {interaction_result}")


    def update_memory(self, location, data, agent_id):
        print(f"Agent {agent_id} is updating memory for location {location} with data {data}")
        if location not in [record["location"] for record in self.memory]:
            # add
            self.memory.append({"location": location, "data": data})
            print(f"Agent {agent_id} Updated Memory: {location} -> {data}")


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

    print(f"Observation for Agent {agent_id}: {observation}")

    # تحقق من وجود بيانات الاتصال
    if comm_observation:
        for comm in comm_observation:
            origin_location = comm[0]
            data = comm[1]
            reported_by = comm[2]  # الوكيل الذي أرسل البيانات

            # تحليل البيانات المستلمة من وكيل معين
            #agents[agent_id].analyse_communication([comm], agent_id)

            for i in range(len(sensor_data_observation)):
                for j in range(len(sensor_data_observation[i])):
                    data = sensor_data_observation[i][j]
                    location = [
                        coordinate_observation[0] + j - len(sensor_data_observation[i]) // 2,
                        coordinate_observation[1] + i - len(sensor_data_observation) // 2
                    ]
                    for target in agents[agent_id].targets_seen:
                        if not target["verified"]:
                            if location == target["location"]:
                                if data != None and 'target_' + str(agent_id) in data:
                                    target["verified"] = True
                                    if 'agent' in target['source']:
                                        reported_by = int(target['source'].split("_")[1])
                                        print(f"Reach reported target location in {location} report by {reported_by}")
                                        # Update trust towards the reported agent
                                        agents[agent_id].update_trust(reported_by, interaction_success=True)
                                else:
                                    if 'agent' in target['source']:
                                        target["verified"] = True
                                        reported_by = int(target['source'].split("_")[1])

                                        self_target_seen = False
                                        for self_target in agents[agent_id].targets_seen:
                                            if self_target["location"] == location and self_target["source"] == "self":
                                                self_target_seen = True
                                        if self_target_seen:
                                            agents[agent_id].update_trust(reported_by, interaction_success=True)
                                        else:
                                            agents[agent_id].update_trust(reported_by, interaction_success=False)

    else:
        print(f"Agent {agent_id}: No communication data available this step.")

    agents[agent_id].analyse_sensor_data(agent_id, coordinate_observation, sensor_data_observation)
    agents[agent_id].analyse_communication(agent_id, comm_observation)
    physical_action = agents[agent_id].select_action(coordinate_observation, agent_id)


    if env.is_agent_silent:
        comm_action = [] # communication action is set to be zero if agent silent
    else:
        coordinate_observation, sensor_data_observation = add_coordinate_noise(coordinate_observation, sensor_data_observation, agent_id)
        sensor_data_observation = add_sensor_data_noise(sensor_data_observation, agent_id)

        comm_action = [coordinate_observation, sensor_data_observation, agent_id] # example of random value as a communication action

    return (physical_action, comm_action)


results = []

def run(num_episodes, max_steps_per_episode, agents, num_actions, env):
    import pandas as pd
    import matplotlib.pyplot as plt

    # تخزين القيم المبدئية لدرجة الثقة لكل وكيل
    initial_trust_levels = {}

    # تسجيل القيم المبدئية عند التهيئة
    for agent_id, agent in enumerate(agents):
        print(f"Agent {agent_id}: Noise level {agent.noise_level:.2f}, Initialized agents with trust level {agent.trust_values}")
        initial_trust_levels[agent_id] = agent.trust_values.copy()  # حفظ نسخة من القيم المبدئية

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")

        # Reset environment
        observations = env.reset()

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

        # Log final trust values at the end of each episode
        for agent_id, agent in enumerate(agents):
            trust_values = agent.trust_values  # Dictionary of trust values
            initial_trust_for_agent = initial_trust_levels.get(agent_id, {})  # القيم المبدئية المحفوظة

            results.append({
                "Agent ID": agent_id,
                "Noise Level": agent.noise_level,
                **{f"Initial Trust Against Agent {other_id}": initial_trust_for_agent.get(other_id, 0) for other_id in range(env.num_agents)},
                **{f"Trust Against Agent {other_id}": 0.0 if agent_id == other_id else trust_values[other_id] for other_id in range(env.num_agents)},
                "Steps": agent_steps[agent_id],
                "Episode": episode + 1
            })

        for agent_id, agent in enumerate(agents):
            print(f"Agent {agent_id}: Final trust values: {agent.trust_values}")

        # Reset agents for the next episode
        for i in range(env.num_agents):
            agents[i].reset()

        # Delete previous goals and create new ones
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

        print(f"Episode {episode + 1} finished after {step_count} steps.\n")

    # Transfer the results into DataFrame
    df = pd.DataFrame(results)


    # إنشاء ملخص البيانات مع إضافة Initial Trust Level و Steps
    summary = df.groupby(["Agent ID", "Noise Level"]).agg({
        **{col: "mean" for col in df.columns if "Trust Against" in col or "Initial Trust Against" in col},  # متوسط الثقة لكل وكيل
        "Steps": "mean"  # إضافة حساب متوسط الخطوات
    }).reset_index()

    # Write to Excel
    with pd.ExcelWriter("final_trust_results.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name="Detailed Results")  # الصفحة الأولى
        summary.to_excel(writer, index=False, sheet_name="Summary")  # الصفحة الثانية
    print("Results saved to 'final_trust_results.xlsx'.")



if __name__ == "__main__":
    gsize=15 #grid size (square)
    gpixels=30 #grid cell size in pixels

    is_sensor_active = True #True:  Activate the sensory observation data
    sensory_size = 3 #'is_sensor_active' must be True. The value must be odd, if event will be converted to one level odd number above

    num_agents = 5 #the number of agents will be run in paralel
    num_obstacles = 0 #the number of obstacles
    is_single_target = False #True: all agents have a single target, False: each agent has their own target
    num_targets_per_agent = 10 #'is_single_target' must be true to have an effect

    is_agent_silent = False #True: communication among agents is allowed

    num_episodes=30 #the number of episode will be run
    max_steps_per_episode=1000 #each episode will be stopped when max_step is reached

    eps_moving_targets = 100000 #set this value greater than 'num_episodes' to keep the targets in a stationary position
    eps_moving_obstacles = 100000 #set this value greater than 'num_episodes' to keep the obstacles in a stationary position

    render = True #True: render the animation into the screen (so far, it is still can not be deactivated)

    min_obstacle_distance_from_target = 1 #min grid distance of each obstacles relative to targets
    max_obstacle_distance_from_target = 5 #max grid distance of each obstacles relative to targets
    min_obstacle_distance_from_agents = 1 #min grid distance of each obstacles relative to agents

    reward_normal = -1 #reward value of normal steps
    reward_obstacle = -5 #reward value when hit an obstacle
    reward_target = 50 #reward value when reach the target

    is_totally_random = True #True: target and obstacles initial as well as movement position is always random on each call, False: only random at the beginning.
    animation_speed = 0.1 #smaller is faster
    is_destroy_environment = True #True: automatically close the animation after all episodes end.


    for settings in [[(.0, {1:.6, 2:.6, 3:.8, 4:.9}, .5), (.0, {0: .5, 2:.6, 3:.8, 4:.9}, .5), (.5, {0: .5, 1:.6, 3:.8, 4:.9}, .5), (.9, {0: .5, 1:.6, 2:.6, 4:.9}, .5), (.8, {0: .5, 1:.6, 2:.6, 3:.8}, .5)]]:

        for trial in range(1):
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

            agents = [SearchAgent(index, settings[index][0], settings[index][1], settings[index][2], num_actions) for index in range(num_agents)]

            # Run episodes
            run(num_episodes, max_steps_per_episode, agents, num_actions, env)

            if is_destroy_environment:
                env.destroy_environment()



