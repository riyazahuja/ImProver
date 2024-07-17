import math
import json
import matplotlib
#matplotlib.use('agg')
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

            curr_before = node['before']
            after_ctx = [node['mctxAfter'].get(b4txt,'') for b4txt in curr_before if b4txt in node['mctxAfter'].keys()]
            kids = node['kids']
            match_before = j in kids

            inside = any([item in ctx for ctx in after_ctx for item in before])

            if set(before).issubset(set(after)) or inside or match_before:
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
        after = [f'{iden} : {node_info["mctxAfter"][iden]}' for iden in node_info['before'] if iden in node_info['mctxAfter'].keys()]
        node_label = f"idx={index}\n{node_info['node']}"#\nbefore: {node_info['before']}\nafter: {node_info['after']}\nmctxAfter:{after}\ninfo_children: {node_info['kids']}"
        G.add_node(index, label=node_label)
        labels[index] = node_label
        # Set positions in a circle
        positions[index] = (r * math.cos(index * theta), r * math.sin(index * theta))

        # Add edges based on children indices
        for child_index in children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index)
    #print(data)
    children = list(set([child for (_,children) in data for child in children]))
    roots = [index for index in range(len(data)) if index not in children]#[child for (_,children) in data for child in children]]
    
    def find_reachable_nodes(G, root_idx=0):
        reachable_nodes = set()
        queue = [root_idx]

        while queue:
            node = queue.pop(0)
            if node not in reachable_nodes:
                reachable_nodes.add(node)
                queue.extend(list(G.successors(node)))
    main_tree = find_reachable_nodes(G)


    for root in roots:
        if root != 0:
            child_idx = max([node for node in main_tree if node < root])
            #ensures we have an actual tree
            G.add_edge(child_idx,root)
        

    

    return G, positions, labels

def depth(G,root_idx = 0):
    # Calculate the depth of the graph as the longest path
    if G.number_of_nodes() > 0:
        lengths = nx.single_source_shortest_path_length(G, root_idx)  # Assuming the root node is 0
        depth = max(lengths.values()) if lengths else 0
    else:
        depth = 0
    return depth

# Visualization function
def visualize_tree(G, positions, labels):
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=8, font_color="black", node_color="skyblue", edge_color="gray")
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    plt.show()

def post_process_graph(G):
    # Remove bidirectional edges
    for u, v in list(G.edges):
        if G.has_edge(v, u):
            G.remove_edge(u, v)
            G.remove_edge(v, u)

    # Ensure each node has a unique path from root
    root = 0
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in list(G.successors(node)):
            if neighbor in visited:
                G.remove_edge(node, neighbor)
            else:
                dfs(neighbor)

    dfs(root)

    return G

# Main function to handle the proof tree creation and visualization
def getProofTree(thm:AnnotatedTheorem, visualize=False):
    contents = []

    for idx,step in enumerate(thm.proof):

        children = [child for child in step.children if child not in [kid for other in [thm.proof[i] for i in range(len(thm.proof)) if i>idx ] for kid in other.children]]

        contents.append({'node': step.tactic,
                          'before': step.goalsBefore,
                            'after': step.goalsAfter,
                            'mctxBefore':step.mctxBefore,
                            'mctxAfter':step.mctxAfter,
                            'kids':children})

    # Logic to trim off unused goals
    # new_contents = []
    # for node in contents:
    #     old_before = node['before']
    #     old_after = node['after']
    #     delete = [goal for goal in old_before if goal in old_after]
    #     new_before = [goal for goal in old_before if goal not in delete]
    #     new_after = [goal for goal in old_after if goal not in delete]
    #     new_contents.append({'node': node['node'], 'before': new_before, 'after': new_after})
    # contents = new_contents        

    adj = build(contents)
    data = list(zip(contents, adj))
    G, positions, labels = build_graph(data)

    G = post_process_graph(G)

    if visualize:
        visualize_tree(G, positions, labels)
    return G, positions, labels

def save_tree(G,positions,labels,save_path):
    matplotlib.use('agg')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=5, font_color="black", node_color="skyblue", edge_color="gray")
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', bbox_inches='tight')  # Save the figure as a PNG file


if __name__ == '__main__':
    repo = getRepo('Tests','configs/config_test.json')
    files = {file.file_name:file for file in repo.files}
    #f = files['Basic.lean']
    #f = files['Solutions_S01_Implication_and_the_Universal_Quantifier.lean']
    f = files['Solutions_S01_Implication_and_the_Universal_Quantifier.lean']
    thms = f.theorems
    thm = thms[0]

    G, p, l = getProofTree(thm, visualize=True)
    #print("Depth of the proof tree:", depth)
