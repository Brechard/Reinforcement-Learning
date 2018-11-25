import gym
import numpy as np
import random

gym.envs.register(

    id='MountainCarMyEasyVersion-v0',

    entry_point='gym.envs.classic_control:MountainCarEnv',

    max_episode_steps=100000,  # MountainCar-v0 uses 200

)

env = gym.make("MountainCarMyEasyVersion-v0")

GAMMA = 0.9

n_position_state = 10
position_state_array = np.linspace(-1.2, 0.6, num=n_position_state)

n_velocity_state = 10
velocity_state_array = np.linspace(-0.07, 0.07, num=n_velocity_state)

# Row = position, Column = velocity
# policy_matrix = np.random.randint(low=0, high=3, size=(n_position_state, n_velocity_state))

# action = policy_matrix[observation[0], observation[1]]
possible_actions = [0, 1, 2]

# Rows = state, Column = actions
q_table = np.zeros((n_position_state * n_velocity_state, len(possible_actions)))

times_action_executed = np.zeros((n_position_state * n_velocity_state, len(possible_actions)))


def get_row_of_state(observation):
    """
        Get the row that corresponds to a particular observation, considering that the q_table has been designed in the
        way that every row is represents a state.
        :param observation: two positions array, where position 0 has the position and position 1 the velocity
        :return: state row in the q_table
        """
    # Discretize the continuous value of the observations, action and position
    # Position 0 = position, Position 1 = velocity
    obs = (np.digitize(observation[0], position_state_array) - 1,
           np.digitize(observation[1], velocity_state_array) - 1)

    row = obs[0] * n_velocity_state + obs[1]
    if row > n_velocity_state * n_position_state:
        print("error")
    return row


# observation, Row = velocity, Column = position
# q_table = {}
#
# print("a")
# actions = [0, 1, 2]
#
# observation = [0, 0]
#
#
#
#
# # At the beggining we use the epsilon greedy strategy


def q_learning(times_repeat):
    # In the beginning, this rate must be at its highest value, because we don’t know anything about
    # the values in Q-table.
    # Therefore we set it to 1 so that it is only exploration and we choose a random state
    epsilon = 1

    current_state = get_row_of_state(env.reset())

    # print(current_state)
    epsilons = []
    epsilons.append(epsilon)
    done = False
    timesteps = 0

    for i in range(times_repeat):
        epsilon = calculate_until_finish(current_state, done, epsilon, False)

    calculate_until_finish(current_state, done, epsilon, True)

    return epsilon


def calculate_until_finish(current_state, done, epsilon, draw):
    timesteps = 0
    epsilon = 1
    # print("AAAAAAAAAAAAAAAA")
    done = False
    env.reset()
    # while not done:
    for i in range(1000):
        timesteps += 1
        if draw:
            env.render()
            action = np.argmax(q_table[current_state])

        else:

            if random.random() > epsilon:  # Exploitation, choose the best action
                action = np.argmax(q_table[current_state])
            else:  # Exploration, we choose a random action
                action = random.choice(possible_actions)

        # action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        # print("Timestep", timesteps, "execute action", action, "epsilon", epsilon, "reward", reward)

        new_state = get_row_of_state(observation)

        current_q_value = q_table[current_state][action]

        # Update the counters table
        times_action_executed[current_state][action] += 1

        # λ^n = n^−α
        learning_rate = times_action_executed[current_state][action] ** -0.9

        # Update the q values table
        q_table[current_state][action] = current_q_value + learning_rate * \
                                         (reward + GAMMA * max(q_table[new_state]) - current_q_value)

        # Add the value to the table for the convergence plot
        # q_table_all_values[current_state][action].append(value)

        current_state = new_state

        epsilon -= 1 / 100
    print("Done in", timesteps, "timesteps")
    return epsilon


q_learning(10000)
