import math
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
import zss
from zss import Node
import Levenshtein


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

            insideArr = [item in ctx for ctx in after_ctx for item in before]
            inside = any(insideArr)
            if inside:
                print(f'OLD : {idx} NEW: {j}')
            #print(f'================\ncurr_before : {curr_before}\nafter_ctx : {after_ctx}\n inside : {insideArr}\n================')

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
    theta = (2 * math.pi) / n 
    r = 10 

    for index, (node_info, children_indices) in enumerate(data):
        node_label = f"{node_info['node']}"
        G.add_node(index, label=node_label)
        labels[index] = node_label

        positions[index] = (r * math.cos(index * theta), r * math.sin(index * theta))


        for child_index in children_indices:
            if child_index < len(data):  
                G.add_edge(index, child_index)     

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
        return reachable_nodes

    main_tree = list(find_reachable_nodes(G))


    for root in roots:
        if root != 0:
            child_idx = max([node for node in main_tree if node < root])
            #ensures we have an actual tree
            G.add_edge(child_idx,root)
        

    

    return G, positions, labels

# Visualization function
def visualize_tree(G, positions, labels, show_mod=False):
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=8, font_color="black", node_color="skyblue", edge_color="gray")
    if show_mod:
        mod_edges = [(u, v) for u, v in G.edges() if G[u][v].get('spawned') or G[u][v].get('bifurcation')]
        nx.draw_networkx_edges(G, pos=positions, edgelist=mod_edges, edge_color='green', arrows=True, arrowstyle='-|>', arrowsize=12, width=1.5)
    
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    plt.show()

def post_process_graph(G):
    # Remove bidirectional edges
    # for u, v in list(G.edges):
    #     if G.has_edge(v, u):
    #         G.remove_edge(u, v)
    #         G.remove_edge(v, u)

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
def getProofTree(thm:AnnotatedTheorem, visualize=False,show_mod=True):
    G, positions, labels = build_graph2(thm.proof_tree)
    if visualize:
        visualize_tree(G, positions, labels, show_mod=show_mod)
    return G, positions, labels


def depth(G,root_idx = 0):
    # Calculate the depth of the graph as the longest path
    if G.number_of_nodes() > 0:
        lengths = nx.single_source_shortest_path_length(G, root_idx)  # Assuming the root node is 0
        depth = max(lengths.values()) if lengths else 0
    else:
        depth = 0
    return depth

def breadth(G):
    # Calculate the breadth of the graph as the number of leaf nodes
    leaf_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
    return len(leaf_nodes)

def tree_edit_distance(G1, G2,normalize=True):
    def nx_to_zss(G, node, label_func):
        zss_node = Node(label_func(node))
        for child in G.successors(node):
            zss_node.addkid(nx_to_zss(G, child, label_func))
        return zss_node

    def label_fn(G):
        return lambda n: G.nodes[n]['label']
    
    zss_tree1 = nx_to_zss(G1, 0, label_fn(G1))
    zss_tree2 = nx_to_zss(G2, 0, label_fn(G2))

    def insert_cost(node):
        return 1

    def remove_cost(node):
        return 1

    def update_cost(node1, node2):
        dist = Levenshtein.distance(node1.label, node2.label)
        max_dist = max(Levenshtein.distance('',node1.label),Levenshtein.distance('',node2.label))
        #print(f'{node1.label} -> {node2.label} : {dist} / {max_dist} = {dist/max_dist}')
        return dist/max_dist
    #dist = zss.simple_distance(zss_tree1,zss_tree2)
    dist = zss.distance(zss_tree1, zss_tree2, get_children=Node.get_children, insert_cost=insert_cost, remove_cost=remove_cost, update_cost=update_cost)
    if normalize:
        num_nodes = G1.number_of_nodes()+G2.number_of_nodes()
        #print(num_nodes)
        dist = dist/num_nodes
    return dist


def build_graph2(data):
    G = nx.DiGraph()
    positions = {}
    labels = {}

    n = len(data)
    theta = (2 * math.pi) / n  if n!= 0 else math.pi/4# Calculate the angle between nodes
    r = 10  # Set radius for the circle

    for index, (tactic, children_indices,spawned_children_indices) in enumerate(data):
        G.add_node(index, label=tactic)
        labels[index] = tactic
        # Set positions in a circle
        positions[index] = (r * math.cos(index * theta), r * math.sin(index * theta))

        # Add edges based on children indices
        
        pure_bifurcation = True if len(children_indices) > 1 and len(spawned_children_indices)==0 else False
        
        for child_index in children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index,bifurcation=pure_bifurcation)    
        
        for child_index in spawned_children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index, spawned=True)   

    return G, positions, labels



def save_tree(G,positions,labels,save_path,show_mod=False):
    matplotlib.use('agg')
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True, node_size=100, arrows=True, arrowstyle='-|>', arrowsize=12, font_size=8, font_color="black", node_color="skyblue", edge_color="gray")
    if show_mod:
        mod_edges = [(u, v) for u, v in G.edges() if G[u][v].get('spawned') or G[u][v].get('bifurcation')]
        nx.draw_networkx_edges(G, pos=positions, edgelist=mod_edges, edge_color='green', arrows=True, arrowstyle='-|>', arrowsize=12, width=1.5)
    
    plt.title('Proof Tree Visualization')
    plt.axis('off')  # Turn off the axis
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format='png', bbox_inches='tight')  # Save the figure as a PNG file


if __name__ == '__main__':
    repo = getRepo('Tests','configs/config_test.json')
    files = {file.file_name:file for file in repo.files}
    f = files['Basic.lean']
    #f = files['Solutions_S02_Functions.lean']
    
    #f = files['Tests/IMO/alphaproof/P1.lean']
    #f = files['Solutions_S01_Sets.lean']
    thms = f.theorems
    thm1 = thms[3]
    thm2 = thms[4]
    print(thm1.proof_tree)

    save_tree(*getProofTree(thm1,visualize=False),save_path='ex.png',show_mod=True)
    save_tree(*getProofTree(thm2,visualize=False),save_path='ex2.png',show_mod=True)