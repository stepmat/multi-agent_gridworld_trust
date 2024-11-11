<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Grid World Dynamic Environment</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1 {
            color: #333;
        }
        h2 {
            color: #555;
            margin-top: 20px;
        }
        ul {
            margin-top: 10px;
            margin-left: 20px;
        }
        p {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <h1>Welcome to the Multi-Agent Grid World Dynamic Environment!</h1>
    <p>
        This environment is a significant part of my Ph.D. research project. It is specifically designed to support both single-agent and multi-agent reinforcement learning applications, with a focus on dynamic, communication-enabled scenarios. Below are some of its key features:
    </p>
    
    <h2>Key Features</h2>
    <ul>
        <li><strong>Flexible Agent Support:</strong>
            <ul>
                <li>Supports both single-agent and multi-agent setups.</li>
                <li>Enables communication between agents, allowing for coordinated reinforcement learning.</li>
            </ul>
        </li>
        <li><strong>Enhanced Sensory Information:</strong>
            <ul>
                <li>Provides agents with a surrounding grid of sensory data, beyond just coordinate information.</li>
                <li>Sensory information allows agents to perceive objects (e.g., obstacles, targets) within a specified range.</li>
            </ul>
        </li>
        <li><strong>Dynamic Environment Elements:</strong>
            <ul>
                <li>Includes moving targets and obstacles to simulate a dynamic environment.</li>
                <li>Supports both shared and unique targets for agents:
                    <ul>
                        <li><strong>Single Target Mode:</strong> All agents aim for a common target.</li>
                        <li><strong>Multiple Targets Mode:</strong> Each agent has its own target or multiple unique targets.</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>

    <h2>Observation Data Structure</h2>
    <p>The environment provides agents with structured observations, consisting of the following four data sequences:</p>
    <ul>
        <li><strong>Agent Coordinates:</strong>
            <ul>
                <li>Specifies the agent’s position on the grid, with the top-left corner represented as <code>[0,0]</code>.</li>
            </ul>
        </li>
        <li><strong>Agent Win-State Flag:</strong>
            <ul>
                <li>Indicates the agent's "done" status:</li>
                <ul>
                    <li><strong>True:</strong> The agent has successfully reached its target.</li>
                    <li><strong>False:</strong> The agent has hit an obstacle, marking a failed state.</li>
                </ul>
            </ul>
        </li>
        <li><strong>Agent Sensory Information:</strong>
            <ul>
                <li>A grid matrix surrounding the agent, with content based on the defined sensory range.</li>
                <li>Each grid cell may contain:</li>
                <ul>
                    <li><code>'None'</code>: Outside the environment boundaries.</li>
                    <li><code>'Empty'</code>: An unoccupied cell.</li>
                    <li><code>'Obstacle'</code>: A cell containing an obstacle.</li>
                    <li><code>'Target'</code>: A cell containing a target, either for the observing agent or another agent.</li>
                </ul>
            </ul>
        </li>
        <li><strong>Communication Data from Other Agents:</strong>
            <ul>
                <li>Provides information shared by other agents, facilitating cooperative strategies.</li>
            </ul>
        </li>
    </ul>

    <h2>Development Background</h2>
    <p>This environment is inspired by and adapted from two existing environments, integrating their features and expanding them to support more advanced multi-agent capabilities and dynamic conditions.</p>
</body>
</html>
