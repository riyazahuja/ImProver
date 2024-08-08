import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
import zss
from zss import Node
import Levenshtein


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


def save_tree(G, positions, labels, save_path, show_mod=False, partition=None):
    matplotlib.use("agg")
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


def calculate_modularity(G: nx.DiGraph, resolution=1):
    G = G.to_undirected()

    def all_partitions(nodes):
        nodes = list(nodes)
        if len(nodes) == 0:
            yield []
            return
        for i in range(2 ** (len(nodes) - 1)):
            parts = [set(), set()]
            for item in nodes:
                parts[i & 1].add(item)
                i >>= 1
            for b in all_partitions(parts[1]):
                yield [parts[0]] + b

    def partition_modularity(G, partition):
        return nx.algorithms.community.modularity(G, partition, resolution=resolution)

    nodes = list(G.nodes)
    max_mod = float("-inf")
    best_partition = None

    for partition in all_partitions(nodes):
        if len(partition) == 1:  # Skip the trivial partition
            continue
        mod = partition_modularity(G, partition)
        if mod > max_mod:
            max_mod = mod
            best_partition = partition

    return max_mod, best_partition


# def calculate_modularity_efficient(G: nx.DiGraph, resolution=1):
#     G = G.to_undirected()

#     def all_partitions(nodes):
#         nodes = list(nodes)
#         if len(nodes) == 0:
#             yield []
#             return
#         for i in range(2 ** (len(nodes) - 1)):
#             parts = [set(), set()]
#             for item in nodes:
#                 parts[i & 1].add(item)
#                 i >>= 1
#             for b in all_partitions(parts[1]):
#                 yield [parts[0]] + b

#     def partition_modularity(G, partition):
#         return nx.algorithms.community.modularity(G, partition, resolution=resolution)

#     nodes = list(G.nodes)
#     max_mod = float("-inf")
#     best_partition = None

#     for partition in all_partitions(nodes):
#         if len(partition) == 1:  # Skip the trivial partition
#             continue
#         mod = partition_modularity(G, partition)
#         if mod > max_mod:
#             max_mod = mod
#             best_partition = partition

#     return max_mod, best_partition


# def calculate_modularity_spectral(G: nx.DiGraph, resolution=1):
#     G = G.to_undirected()
#     # partition = []

#     def modularity_matrix(G):
#         """Calculate the modularity matrix of the graph G."""
#         A = nx.adjacency_matrix(G).todense()
#         k = np.sum(A, axis=1)
#         m = np.sum(k) / 2
#         B = A - np.outer(k, k) / (2 * m)
#         return B, m

#     def leading_eigenvector(B):
#         """Compute the leading eigenvector of the modularity matrix B."""
#         eigenvalues, eigenvectors = np.linalg.eigh(B)
#         idx = np.argmax(eigenvalues)
#         return eigenvectors[:, idx]

#     def partition_from_eigenvector(v):
#         """Partition nodes based on the sign of the leading eigenvector."""
#         return [set(np.where(v > 0)[0]), set(np.where(v <= 0)[0])]

#     def refined_modularity_matrix(B, group):
#         """Construct the refined modularity matrix B^{(g)} for a given group."""
#         B_g = B[np.ix_(group, group)]
#         for i in range(B_g.shape[0]):
#             B_g[i, i] -= np.sum(B_g[i, :])
#         return B_g

#     def recursive_partition(G, nodes, B, m):
#         # nodes = sorted(nodes)
#         print(f"N:{nodes}\nB:{B}\nm:{m}")
#         """Recursively partition the graph to maximize modularity."""
#         if len(nodes) <= 1:
#             return [nodes]

#         leading_v = leading_eigenvector(B)
#         partition = partition_from_eigenvector(leading_v)
#         # partition = [set([nodes[i] for i in part]) for part in partition]
#         print(f"partition:{partition}")

#         s = np.array([1 if i in partition[0] else -1 for i in range(len(nodes))])
#         delta_Q = (s.T @ B @ s) / (4 * m)
#         print(f"dQ:{delta_Q}")
#         if delta_Q <= 0:
#             print(f"output:{[nodes]}\n======")
#             return [nodes]

#         sub_partitions = []
#         for part in partition:
#             if len(part) < 2:
#                 sub_partitions.append(part)
#                 continue

#             subgraph = G.subgraph(part).copy()
#             sub_nodes = list(subgraph.nodes)
#             print(f"subnodes:{sub_nodes}")
#             B_sub = refined_modularity_matrix(B, sub_nodes)
#             print("--[RECURSION]--")
#             sub_partitions.extend(recursive_partition(subgraph, sub_nodes, B_sub, m))
#             print("--===========--")
#         print(f"output:{sub_partitions}\n======")
#         return sub_partitions

#     B, m = modularity_matrix(G)
#     nodes = sorted(list(G.nodes))
#     best_partition = recursive_partition(G, nodes, B, m)
#     max_mod = nx.algorithms.community.modularity(G, best_partition)
#     return max_mod, best_partition

#     B, m = modularity_matrix(G)
#     best_partition = recursive_partition(G, B, m)
#     max_mod = nx.algorithms.community.modularity(G, best_partition)
#     return max_mod, best_partition

#     # def get_best_partition(group=set(range(len(G))),partitions = []):
#     #     B,m = modularity_matrix(G,group)
#     #     leading_v=leading_eigenvector(B)
#     #     partition = partition_from_eigenvector(leading_v)
#     #     s = np.array([1 if i in partition[0] else -1 for i in range(len(G))])
#     #     dQ = (s.T @ B @ s) / (4 * m)
#     #     if dQ <= 0:
#     #         partitions.append(group)
#     #     else:
#     #         for part in partitions:
#     #             get_best_partition(part,)

#     B, m = modularity_matrix(G)
#     print(B)
#     leading_v = leading_eigenvector(B)
#     print(leading_v)
#     initial_partition = partition_from_eigenvector(leading_v)
#     print(initial_partition)
#     best_partition, _ = _recursive_partition(G, initial_partition, B, m)
#     print(best_partition)

#     print(nx.algorithms.community.modularity(G, [set([0, 1, 2, 3, 4, 5])]))
#     # max_mod = nx.algorithms.community.modularity(G, best_partition)
#     # return max_mod, best_partition


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
    repo = getRepo("Tests", "configs/config_test.json")
    files = {file.file_name: file for file in repo.files}
    f = files["Basic.lean"]
    thms = f.theorems
    thm1 = thms[3]
    thm2 = thms[4]

    save_tree(*getProofTree(thm1, visualize=False), save_path="ex.png", show_mod=True)
    save_tree(*getProofTree(thm2, visualize=False), save_path="ex2.png", show_mod=True)
