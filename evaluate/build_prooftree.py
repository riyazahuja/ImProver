import math
import json
import matplotlib
matplotlib.use('agg')
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

    n = len(data)
    theta = (2 * math.pi) / n  # Calculate the angle between nodes
    r = 10  # Set radius for the circle

    for index, (node_info, children_indices) in enumerate(data):
        node_label = f"{node_info['node']}\nbefore: {node_info['before']}\nafter: {node_info['after']}"
        G.add_node(index, label=node_label)
        labels[index] = node_label
        # Set positions in a circle
        positions[index] = (r * math.cos(index * theta), r * math.sin(index * theta))

        # Add edges based on children indices
        for child_index in children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index)

    # Calculate the depth of the graph as the longest path
    if G.number_of_nodes() > 0:
        lengths = nx.single_source_shortest_path_length(G, 0)  # Assuming the root node is 0
        depth = max(lengths.values()) if lengths else 0
    else:
        depth = 0

    return G, positions, labels, depth

# Visualization function
def visualize_tree(G, positions, labels):
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=8, font_color="black", node_color="skyblue", edge_color="gray")
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    plt.show()

# Main function to handle the proof tree creation and visualization
def getProofTree(thm:AnnotatedTheorem, visualize=False):
    contents = []
    for step in thm.proof:
        contents.append({'node': step.tactic, 'before': step.prevState, 'after': step.nextState})

    # Logic to trim off unused goals
    new_contents = []
    for node in contents:
        old_before = node['before']
        old_after = node['after']
        delete = [goal for goal in old_before if goal in old_after]
        new_before = [goal for goal in old_before if goal not in delete]
        new_after = [goal for goal in old_after if goal not in delete]
        new_contents.append({'node': node['node'], 'before': new_before, 'after': new_after})
    contents = new_contents        

    adj = build(contents)
    data = list(zip(contents, adj))
    G, positions, labels, depth = build_graph(data)

    if visualize:
        visualize_tree(G, positions, labels)
    return G, positions, labels, depth

def save_tree(G,positions,labels,save_path):
    matplotlib.use('agg')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=8, font_color="black", node_color="skyblue", edge_color="gray")
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', bbox_inches='tight')  # Save the figure as a PNG file


if __name__ == '__main__':
    src = 'Tests3'
    file = 'Tests3/Basic.lean'
    f = getAnnotatedFile(src, file)
    thms = f.theorems
    thm_annotated = thms[0]

    G, p, l, depth = getProofTree(thm_annotated, visualize=True)
    print("Depth of the proof tree:", depth)
