import numpy as np
import matplotlib.pyplot as plt

class NetworkNode:
    def __init__(self, name, states, parents=None):
        self.name = name
        self.states = states  # Probability distribution (numpy array)
        self.parents = parents if parents is not None else []
        self.children = []
        self.messages = {}  # Messages from parents
        self.noise_params = {}  # Individual noise parameters

        # Initialize connections
        for parent in self.parents:
            parent.children.append(self)

    def receive_message(self, parent_name, message):
        self.messages[parent_name] = message

    def compute_belief(self):
        if not self.parents:
            belief = self.states.copy()
        else:
            belief = self.states.copy()
            for parent in self.parents:
                message = self.messages.get(parent.name, np.ones_like(self.states))
                belief *= message
            belief /= np.sum(belief)  # Normalize
        return belief

    def apply_noise(self):
        level = self.noise_params.get('level', 0.05)
        noise = np.random.normal(0, level, self.states.shape)
        self.states += noise
        self.states = np.clip(self.states, 0, 1)
        self.states /= np.sum(self.states)

    def update(self):
        self.apply_noise()
        belief = self.compute_belief()
        # Send belief to children
        for child in self.children:
            child.receive_message(self.name, belief)

class ProbabilisticNetwork:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node: NetworkNode):
        self.nodes[node.name] = node

    def propagate_beliefs(self, iterations=1):
        for _ in range(iterations):
            for node in self.nodes.values():
                node.update()

class AdaptiveNoiseControl:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def adjust_parameters(self, node: NetworkNode):
        avg_value = np.mean(node.states)
        node.noise_params['level'] = 0.01 if avg_value > self.threshold else 0.05

def network_simulation():
    initial_states = np.array([0.5, 0.5])  # Binary states [True, False]

    # Create nodes
    node_A = NetworkNode('A', initial_states.copy())
    node_B = NetworkNode('B', initial_states.copy(), parents=[node_A])
    node_C = NetworkNode('C', initial_states.copy(), parents=[node_A])
    node_D = NetworkNode('D', initial_states.copy(), parents=[node_B, node_C])

    network = ProbabilisticNetwork()
    for node in [node_A, node_B, node_C, node_D]:
        network.add_node(node)

    adaptive_control = AdaptiveNoiseControl(threshold=0.5)

    iterations = 5
    for i in range(iterations):
        print(f"\nIteration {i+1}")
        # Adjust noise parameters
        for node in network.nodes.values():
            adaptive_control.adjust_parameters(node)
        # Propagate beliefs
        network.propagate_beliefs(iterations=1)
        # Print node beliefs
        for node_name, node in network.nodes.items():
            belief = node.compute_belief()
            print(f"Node {node_name} belief: {belief}")

# Run the network simulation
network_simulation()


def network_simulation_with_visualization():
    initial_states = np.array([0.5, 0.5])  # Binary states [True, False]

    # Create nodes
    node_A = NetworkNode('A', initial_states.copy())
    node_B = NetworkNode('B', initial_states.copy(), parents=[node_A])
    node_C = NetworkNode('C', initial_states.copy(), parents=[node_A])
    node_D = NetworkNode('D', initial_states.copy(), parents=[node_B, node_C])

    network = ProbabilisticNetwork()
    for node in [node_A, node_B, node_C, node_D]:
        network.add_node(node)

    adaptive_control = AdaptiveNoiseControl(threshold=0.5)

    iterations = 10
    beliefs_history = {node_name: [] for node_name in network.nodes.keys()}

    for i in range(iterations):
        # Adjust noise parameters
        for node in network.nodes.values():
            adaptive_control.adjust_parameters(node)
        # Propagate beliefs
        network.propagate_beliefs(iterations=1)
        # Record node beliefs
        for node_name, node in network.nodes.items():
            belief = node.compute_belief()
            beliefs_history[node_name].append(belief.copy())

    # Plot the beliefs over iterations
    for node_name, beliefs in beliefs_history.items():
        beliefs_array = np.array(beliefs)
        plt.figure()
        plt.plot(range(iterations), beliefs_array[:, 0], label=f'{node_name} State 0')
        plt.plot(range(iterations), beliefs_array[:, 1], label=f'{node_name} State 1')
        plt.title(f'Belief Evolution for Node {node_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Belief')
        plt.legend()
        plt.show()

# Run the network simulation with visualization
network_simulation_with_visualization()
