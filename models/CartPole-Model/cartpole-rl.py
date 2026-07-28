import random
from collections import deque

import gymnasium as gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Hyperparameters for the DQN agent and training process
HYPERPARAMS = {
    "env_name":            "CartPole-v1",
    "gamma":                0.99,   # discount factor - how much we care about future rewards
    "lr":                   1e-3,
    "epsilon_start":        1.0,    # fully random at the start 
    "epsilon_end":          0.01,   # floor exploration rate
    "epsilon_decay":        0.995,  # decay rate per episode
    "buffer_size":          50_000,
    "batch_size":           64,
    "target_update_every":  10,     # sync target network every N episodes
    "num_episodes":         300,
    "max_steps_per_episode": 500,
}

# simple feedforward neural network to approximate the Q-function
def build_q_network(state_dim: int, num_actions: int, lr: float) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(num_actions, activation="linear"),  # raw Q-values, no softmax
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

# Replay buffer to store experiences for experience replay
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# DQN Agent that interacts with the environment and learns from experiences
class DQNAgent:
    def __init__(self, state_dim, num_actions, hp):
        self.num_actions = num_actions
        self.gamma = hp["gamma"]
        self.epsilon = hp["epsilon_start"]
        self.epsilon_end = hp["epsilon_end"]
        self.epsilon_decay = hp["epsilon_decay"]
        self.batch_size = hp["batch_size"]

        self.q_network = build_q_network(state_dim, num_actions, hp["lr"]) # main Q-network
        self.target_network = build_q_network(state_dim, num_actions, hp["lr"]) # frozen copy of the main network for stability
        self.target_network.set_weights(self.q_network.get_weights()) # synchronize weights initially

        self.buffer = ReplayBuffer(hp["buffer_size"])

    # action selection using epsilon-greedy policy
    def act(self, state, explore: bool = True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.q_network.predict(state[np.newaxis, :], verbose=0)[0] # change shape to (1, state_dim) for prediction
        return int(np.argmax(q_values)) 

    # store experience in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    # learn from a batch of experiences sampled from the replay buffer
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Bellman target, using the frozen target network for stability
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)  # no future reward if done

        # only the taken action's Q-value moves toward the target
        current_q = self.q_network.predict(states, verbose=0)
        current_q[np.arange(self.batch_size), actions] = targets

        self.q_network.fit(states, current_q, epochs=1, verbose=0)

    def update_target_network(self): # update the target network weights
        self.target_network.set_weights(self.q_network.get_weights())

    def decay_epsilon(self): # decay exploration rate after each episode
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# Training loop for the DQN agent
def train(hp):
    env = gym.make(hp["env_name"]) 
    state_dim = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)  # gym returns numpy.int64; Keras needs plain int

    agent = DQNAgent(state_dim, num_actions, hp)
    episode_rewards = []

    for episode in range(hp["num_episodes"]):
        state, _ = env.reset()
        total_reward = 0

        for step in range(hp["max_steps_per_episode"]):
            action = agent.act(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action) # if the episode ends, terminated or truncated will be true
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        if episode % hp["target_update_every"] == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)
        if episode % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(f"Episode {episode:4d} | reward: {total_reward:6.1f} | "
                  f"avg(10): {avg_last_10:6.1f} | epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards

# Evaluate the trained agent over a number of episodes without exploration
def evaluate(agent, env_name, num_episodes=10, render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    scores = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, explore=False)  # no exploration during eval
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        scores.append(total_reward)

    env.close()
    print(f"\nEvaluation over {num_episodes} episodes: "
          f"mean={np.mean(scores):.1f}, min={np.min(scores):.1f}, max={np.max(scores):.1f}")
    return scores


if __name__ == "__main__":
    trained_agent, history = train(HYPERPARAMS)
    evaluate(trained_agent, HYPERPARAMS["env_name"], num_episodes=10)

# Save the model
trained_agent.save("cartpole_dqn_model.h5") 
 
# Evaluation over 10 episodes: mean=177.8, min=162.0, max=189.0
