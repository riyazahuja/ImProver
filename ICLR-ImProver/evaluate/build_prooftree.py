import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
import zss
from zss import Node
import Levenshtein


"""
Generate prooftree from metaprogramming outputs, also some basic functions that can be used to make metrics that employ the proof tree.
"""


def visualize_tree(G, positions, labels, show_mod=False):
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_size=100,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        font_size=8,
        font_color="black",
        node_color="skyblue",
        edge_color="gray",
    )
    if show_mod:
        mod_edges = [
            (u, v)
            for u, v in G.edges()
            if G[u][v].get("spawned") or G[u][v].get("bifurcation")
        ]
        nx.draw_networkx_edges(
            G,
            pos=positions,
            edgelist=mod_edges,
            edge_color="green",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.5,
        )
    plt.title("Proof Tree Visualization")
    plt.axis("off")  # Turn off the axis
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


def build_graph2(data):
    G = nx.DiGraph()
    positions = {}
    labels = {}
    n = len(data)
    theta = (2 * math.pi) / n if n != 0 else math.pi / 4
    r = 10
    for index, (tactic, children_indices, spawned_children_indices) in enumerate(data):
        G.add_node(index, label=tactic)
        labels[index] = tactic
        positions[index] = (r * math.cos(index * theta), r * math.sin(index * theta))
        pure_bifurcation = (
            True
            if len(children_indices) > 1 and len(spawned_children_indices) == 0
            else False
        )
        for child_index in children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index, bifurcation=pure_bifurcation)
        for child_index in spawned_children_indices:
            if child_index < len(data):  # Ensure child index is valid
                G.add_edge(index, child_index, spawned=True)
    return G, positions, labels


def save_tree2(G, positions, labels, save_path, show_mod=False, partition=None):
    matplotlib.use("agg")
    plt.figure(figsize=(12, 8))

    # Assign colors to nodes based on their community in the partition
    if partition is not None:
        color_map = plt.get_cmap("tab20")
        node_colors = {}
        for idx, community in enumerate(partition):
            for node in community:
                node_colors[node] = color_map(idx % 20)
        node_colors = [node_colors[node] for node in G.nodes()]
    else:
        node_colors = "skyblue"

    nx.draw(
        G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_size=100,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        font_size=8,
        font_color="black",
        node_color=node_colors,
        edge_color="gray",
    )

    if show_mod:
        mod_edges = [
            (u, v)
            for u, v in G.edges()
            if G[u][v].get("spawned") or G[u][v].get("bifurcation")
        ]
        nx.draw_networkx_edges(
            G,
            pos=positions,
            edgelist=mod_edges,
            edge_color="green",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.5,
        )

    plt.title("Proof Tree Visualization")
    plt.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format="png", bbox_inches="tight")


def save_tree(G, positions, labels, save_path, show_mod=False):
    matplotlib.use("agg")
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_size=500,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        font_size=10,
        font_color="black",
        node_color="skyblue",
        edge_color="gray",
    )
    if show_mod:
        mod_edges = [
            (u, v)
            for u, v in G.edges()
            if G[u][v].get("spawned") or G[u][v].get("bifurcation")
        ]
        nx.draw_networkx_edges(
            G,
            pos=positions,
            edgelist=mod_edges,
            edge_color="green",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.5,
        )

    plt.title("Proof Tree Visualization")
    plt.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, format="png", bbox_inches="tight")


def getProofTree(thm: AnnotatedTheorem, visualize=False, show_mod=False):
    G, positions, labels = build_graph2(thm.proof_tree)
    if visualize:
        visualize_tree(G, positions, labels, show_mod=show_mod)
    return G, positions, labels


def depth(G, root_idx=0):
    if G.number_of_nodes() > 0:
        lengths = nx.single_source_shortest_path_length(G, root_idx)
        depth = max(lengths.values()) if lengths else 0
    else:
        depth = 0
    return depth


def breadth(G):
    leaf_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
    return len(leaf_nodes)


def calculate_modularity(G: nx.DiGraph):
    G = G.copy()
    G.remove_edges_from(get_modular_edges(G))

    # Get all connected components in the modified graph
    connected_components = list(nx.connected_components(G.to_undirected()))

    # Find the maximum number of nodes in any connected component
    max_component_size = max(len(component) for component in connected_components)

    return max_component_size


def calculate_efficiency(G):
    # Calculate the global efficiency of the subgraph
    undirected = G.to_undirected()
    efficiency = nx.global_efficiency(undirected)

    return efficiency


def get_modular_edges(G):
    return [
        (u, v)
        for u, v in G.edges()
        if G[u][v].get("spawned") or G[u][v].get("bifurcation")
    ]


def tree_edit_distance(G1, G2, normalize=True):
    def nx_to_zss(G, node, label_func):
        zss_node = Node(label_func(node))
        for child in G.successors(node):
            zss_node.addkid(nx_to_zss(G, child, label_func))
        return zss_node

    def label_fn(G):
        return lambda n: G.nodes[n]["label"]

    zss_tree1 = nx_to_zss(G1, 0, label_fn(G1))
    zss_tree2 = nx_to_zss(G2, 0, label_fn(G2))

    def insert_cost(node):
        return 1

    def remove_cost(node):
        return 1

    def update_cost(node1, node2):
        dist = Levenshtein.distance(node1.label, node2.label)
        max_dist = max(
            Levenshtein.distance("", node1.label), Levenshtein.distance("", node2.label)
        )
        return dist / max_dist

    dist = zss.distance(
        zss_tree1,
        zss_tree2,
        get_children=Node.get_children,
        insert_cost=insert_cost,
        remove_cost=remove_cost,
        update_cost=update_cost,
    )
    if normalize:
        num_nodes = G1.number_of_nodes() + G2.number_of_nodes()
        dist = dist / num_nodes
    return dist


if __name__ == "__main__":
    repo = getRepo("Tests", "configs/config_MIL.json")
    files = {file.file_path: file for file in repo.files}
    fs = [
        files[name]
        for name in files.keys()
        if "Solution" in name and "C04" in name and "S01" in name
    ]

    f = fs[0]
    thms = f.theorems
    for i in range(len(thms)):
        save_tree(
            *getProofTree(thms[i], visualize=False), save_path=f".trees/MIL/{i}.png"
        )
