# Welcome to the Multi-Agent Grid World Dynamic Environment!

This environment is a significant part of my Ph.D. research project. It is specifically designed to support both single-agent and multi-agent reinforcement learning applications, with a focus on discrete observation and action space, dynamic, and communication-enabled scenarios. Below are some of its key features:

## Key Features

1. **Flexible Agent Support:**
   - Supports both single-agent and multi-agent setups.
   - Enables communication between agents, allowing for coordinated multi-agent reinforcement learning.

2. **Enhanced Sensory Information:**
   - Provides agents with a surrounding grid of sensory data, beyond just coordinate information.
   - Sensory information allows agents to perceive objects (e.g., obstacles, targets) within a specified range.

3. **Dynamic Environment Elements:**
   - Includes moving targets and obstacles to simulate a dynamic environment.
   - Supports both shared and unique targets for agents:
     - **Single Target Mode**: All agents aim for a common target.
     - **Multiple Targets Mode**: Each agent has its own unique target or multiple targets.

## Observation Data Structure

The environment provides agents with structured observations, consisting of the following four data sequences:

1. **Agent Coordinates**:
   - Specifies the agent’s position on the grid, with the top-left corner represented as `[0,0]`.

2. **Agent Win-State Flag**:
   - Indicates the agent's "done" status:
     - **True**: The agent has successfully reached its target.
     - **False**: The agent has hit an obstacle, marking a failed state.

3. **Agent Sensory Information**:
   - A grid matrix surrounding the agent, with content based on the defined sensory range.
   - Each grid cell may contain:
     - `'None'`: Outside the environment boundaries.
     - `'Empty'`: An unoccupied cell.
     - `'Obstacle'`: A cell containing an obstacle.
     - `'Target'`: A cell containing a target, either for the observing agent or another agent.

4. **Communication Data from Other Agents**:
   - Provides information shared by other agents, facilitating cooperative strategies.

## Development Background

This environment is inspired by and adapted from two existing environments:
1. Single-agent grid world environment:https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world
2. Vectorized Multi-Agent Simulator (VMAS): https://github.com/proroklab/VectorizedMultiAgentSimulator

