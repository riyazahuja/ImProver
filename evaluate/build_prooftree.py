import math
import json
import matplotlib.pyplot as plt
import networkx as nx
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *


def build(nodes):
    adj = [None for _ in range(len(nodes))]
    enum_nodes = list(enumerate(nodes))

    def build_helper(idx):
        node = nodes[idx]
        children = []
        for j, other in [(i, x) for i, x in enum_nodes if x is not node]:
            after = node['after']
            before = other['before']
            if set(before).issubset(set(after)):
                children.append(j)
        adj[idx] = children

    for i in range(len(nodes)):
        build_helper(i)

    return adj

def build_graph(data):
    G = nx.DiGraph()
    positions = {}
    labels = {}

    # Store depth of each node for calculating the maximum depth
    node_depth = {}

    def set_positions_depth(index, depth):
        n = len(data)
        theta = (2 * math.pi) / n
        r = 5 + 3 * depth  # Increment radius based on depth to spread out nodes
        positions[index] = (r * math.cos(index * theta), r * math.sin(index * theta))
        node_depth[index] = depth
        for child_index in data[index][1]:
            if child_index < len(data):  # Ensure child index is valid
                set_positions_depth(child_index, depth + 1)

    # Initialize root node position and depth
    set_positions_depth(0, 0)  # Assuming the first node is the root

    for index, (node_info, children_indices) in enumerate(data):
        node_label = f"{node_info['node']}\nbefore: {node_info['before']}\nafter: {node_info['after']}"
        G.add_node(index, label=node_label)
        labels[index] = node_label
        for child_index in children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index)

    max_depth = max(node_depth.values()) if node_depth else 0
    print("Maximum depth of the tree:", max_depth)
    return G, positions, labels, max_depth

# Visualization function
def visualize_tree(G,positions,labels):
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=6, font_color="black", node_color="skyblue", edge_color="gray")
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    plt.show()



def getProofTree(thm, visualize = False):
    contents = []
    for step in thm.proof:
        contents.append({'node': step.tactic, 'before': step.prevState, 'after': step.nextState})


    adj = build(contents)
    data = list(zip(contents, adj))

    G,positions,labels,depth = build_graph(data)

    if visualize:
        visualize_tree(G,positions,labels)
    return G,positions,labels,depth
    


