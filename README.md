Welcome to Multi-Agent Grid World Dynamic Environment!

This works is part of my Ph.D project. This gridworld environment have several features as follows: 
1. Designed for both single-agent and multi-agent (with communication) reinforcement learning
2. Provides agent sensory information using grid surrounding agents (apart of only agent coordinates data)
3. Provides target and obstacles movement mechanism to simulate a dynamic environment
4. Provides a single target for all agents or a unique target for each agent, where each agent can have multiple target.


The observation data from this environment consist of 4 sequences of variables as follows: 
1. Agent coordinates: top-left is [0,0]
2. Agent win-state flag: indicate the done status of each agent. If hit the obstacle, the win-state flag set to be False, otherwise set to be True (reach the target)
3. Agent sensory information: consist of matrics data, depend on the sensory size in the configuration setting. Each grid cells can contain 'None': out of environment boundary, 'Empty': contain an empty cell, 'Obstacle': contains an obstacle,  and 'Target': contains target of any agent.
4. Other agents communication data.

As an addition, this environment is developed by adopted two other environments: 
1. Single-agent gridworld environment (https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world)
2. Multi-agent particle environment (https://github.com/proroklab/VectorizedMultiAgentSimulator)
