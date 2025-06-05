import tensorflow as tf
import numpy as np
import random
from collections import deque
import osmnx as ox
import networkx as nx
import folium
from matplotlib import cm
import matplotlib.colors as mcolors
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
from scipy.spatial import cKDTree
import json


# Function to parse coordinates from a JSON file
def get_locations_from_json_file(file_path):
    """
    Extracts locations from a JSON file provided by the cloud team.

    Args:
        file_path (str): Path to the JSON file containing location data.

    Returns:
        list: List of all locations [(lat, lon), ...].
        tuple: Depot location (lat, lon).
    """
    try:
        # Open and parse the JSON file
        with open(file_path, "r") as f:
            payload = json.load(f)

        if "locations" not in payload:
            raise ValueError("Invalid JSON: 'locations' field is missing")

        locations = payload["locations"]
        depot_location = None
        customer_locations = []

        # Extract depot and customer locations
        for loc in locations:
            name = loc.get("name", "Unnamed Location")
            latitude = loc.get("latitude")
            longitude = loc.get("longitude")

            if latitude is None or longitude is None:
                raise ValueError(f"Invalid location {name}: Missing latitude or longitude")

            if name.lower() == "depot":
                depot_location = (latitude, longitude)
            else:
                customer_locations.append((latitude, longitude))

        if not depot_location:
            raise ValueError("Depot location is missing in the JSON file")

        all_locations = [depot_location] + customer_locations
        return all_locations, depot_location

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing JSON file: {e}")


# Example: Load locations from a JSON file
json_file_path = "locations.json"  # Replace with the path provided by the cloud team
try:
    locations, depot_location = get_locations_from_json_file(json_file_path)
    print("All Locations:", locations)
    print("Depot Location:", depot_location)
except Exception as e:
    print(f"Failed to load locations: {e}")

# Fetch road network with reduced distance for faster fetching

def fetch_road_network(depot_location, dist=20000, network_type='drive'):
    try:
        print("Fetching road network...")
        return ox.graph_from_point(
            depot_location,
            dist=dist,
            network_type=network_type,
        
        )
    except Exception as e:
        print(f"Error fetching road network: {e}")
        return None

# Function to map locations to nearest nodes
def map_to_nearest_nodes(locations, graph, distance_threshold=2000):
    valid_nodes = []
    valid_demands = []
    print("Mapping locations to the road network...")

    # Precompute graph node coordinates for faster nearest node lookups
    graph_nodes = {node: (data['y'], data['x']) for node, data in graph.nodes(data=True)}
    graph_node_coords = np.array(list(graph_nodes.values()))
    graph_node_ids = np.array(list(graph_nodes.keys()))

    # Build a KDTree for fast nearest-neighbor search
    tree = cKDTree(graph_node_coords)

    # Batch query locations for nearest nodes
    location_coords = np.array(locations)
    distances, indices = tree.query(location_coords)

    for idx, (dist, nearest_idx) in enumerate(zip(distances, indices)):
        print(f"Location {idx}: Distance to nearest node = {dist:.2f} meters")
        if dist > distance_threshold:  # Skip locations too far from the network
            print(f"Warning: Location {idx} is far from the road network! Skipping this location.")
            continue

        nearest_node = graph_node_ids[nearest_idx]
        valid_nodes.append(nearest_node)
        if idx > 0:  # Exclude depot from demands
            valid_demands.append(10)

    return valid_nodes, valid_demands

G = fetch_road_network(depot_location)
if G:
    valid_nodes, valid_demands = map_to_nearest_nodes(locations, G)
    print("Valid nodes:", valid_nodes)
    print("Valid demands:", valid_demands)

# Build distance matrix for valid nodes
def build_distance_matrix(valid_nodes, graph):
    distance_matrix = np.zeros((len(valid_nodes), len(valid_nodes)))
    paths = {}
    for i, start_node in enumerate(valid_nodes):
        for j, end_node in enumerate(valid_nodes):
            if i != j:
                try:
                    path = nx.shortest_path(graph, start_node, end_node, weight='length')
                    distance = nx.shortest_path_length(graph, start_node, end_node, weight='length')
                    distance_matrix[i][j] = distance
                    paths[(i, j)] = path
                except nx.NetworkXNoPath:
                    print(f"Warning: No path between nodes {start_node} and {end_node}")
                    distance_matrix[i][j] = float('inf')
                    paths[(i, j)] = []
    return distance_matrix, paths

# Build distance matrix and paths
distance_matrix, paths = build_distance_matrix(valid_nodes, G)

# Print summary
print(f"Valid nodes: {valid_nodes}")
print(f"Distance matrix:\n{distance_matrix}")

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
agent = VRPAgent(num_actions=num_actions)
routes = train_vrp_agent(agent, num_vehicles=1, capacity=100, customer_demands=valid_demands, distance_matrix=distance_matrix)

