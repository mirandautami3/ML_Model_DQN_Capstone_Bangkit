import tensorflow as tf
import numpy as np
import random
from collections import deque
import osmnx as ox
import networkx as nx
import folium
from matplotlib import cm
import matplotlib.colors as mcolors
from main import distance_matrix, valid_demands, valid_nodes
# Define DQN model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Define VRP Agent
class VRPAgent:
    def __init__(self, num_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.replay_buffer = ReplayBuffer()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, unvisited_nodes):
        state_tensor = tf.convert_to_tensor([[state]], dtype=tf.float32)
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(unvisited_nodes))
        else:
            q_values = self.model(state_tensor).numpy()[0]
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[list(unvisited_nodes)] = q_values[list(unvisited_nodes)]
            return np.argmax(masked_q_values)

    def store_experience(self, state, action, reward, next_state, done):
        state = np.atleast_2d(state)
        next_state = np.atleast_2d(next_state)
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = tf.convert_to_tensor(np.array(states).squeeze(axis=1), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states).squeeze(axis=1), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_q_values = self.target_model(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(action_masks * q_values, axis=1)
            loss = tf.keras.losses.MSE(targets, q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_vrp_agent(agent, num_vehicles, capacity, customer_demands, distance_matrix, num_episodes=150, batch_size=32, update_target_every=10):
    best_routes = []
    best_total_distance = float('inf')

    for episode in range(num_episodes):
        total_distance = 0
        cumulative_reward = 0
        episode_routes = []
        unvisited_nodes = set(range(1, len(customer_demands) + 1))  # Adjusted for valid_nodes (exclude depot)

        for vehicle_idx in range(num_vehicles):
            route = [0]  # Start from depot (node index 0)
            load = 0
            current_node = 0

            while unvisited_nodes:
                action = agent.choose_action(current_node, unvisited_nodes)

                if action in unvisited_nodes:
                    dist = distance_matrix[current_node][action]
                    if load + customer_demands[action - 1] <= capacity:
                        route.append(action)
                        load += customer_demands[action - 1]
                        total_distance += dist
                        unvisited_nodes.remove(action)

                        reward = max(10 / (dist + 1e-5), 1)
                        reward += 10

                        cumulative_reward += reward
                        agent.store_experience(np.array([current_node]), action, reward, np.array([action]), False)

                        current_node = action
                    else:
                        reward = -100
                        agent.store_experience(np.array([current_node]), action, reward, np.array([current_node]), True)
                        break
                else:
                    reward = -50
                    agent.store_experience(np.array([current_node]), action, reward, np.array([current_node]), True)
                    break

            if unvisited_nodes or vehicle_idx < num_vehicles - 1:
                route.append(0)
                total_distance += distance_matrix[current_node][0]
                reward = 100
                cumulative_reward += reward
            elif vehicle_idx == num_vehicles - 1:
                print(f"Last route ends at node {current_node} without returning to depot.")

            episode_routes.append(route)

        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_routes = episode_routes

        print(f"Episode {episode + 1}/{num_episodes} - Total Distance: {total_distance} - Reward: {cumulative_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

        agent.train(batch_size=batch_size)
        if episode % update_target_every == 0:
            agent.update_target_model()

    print(f"Best Total Distance: {best_total_distance}")
    return best_routes

# Initialize and train the VRP Agent
capacity = 100
num_vehicles = 1
# Initialize and train the VRP Agent
num_actions = len(distance_matrix[0])