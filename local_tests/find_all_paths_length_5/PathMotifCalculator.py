from datetime import datetime
import networkx as nx


class PathMotifCalculator:

    def __init__(self, graph: nx.Graph, directed: bool):
        self._graph = graph
        self._colors = nx.get_node_attributes(graph, "color")
        self._directed = directed

    # -------- PRIVATE API -------- #

    def _check_path_intersection(self, path1, path2):
            return (
        path1[0][0] == path2[1][0] or
        path1[1][0] == path2[0][0] or
        path1[1][0] == path2[1][0]
    )
    
    def iter_neighbors_with_direction(self, G, node):
        if self._directed:
            # Outgoing edges
            for nbr in G.successors(node):
                yield nbr, True
            
            # Incoming edges
            for nbr in G.predecessors(node):
                yield nbr, False
        else:
            # Undirected graph
            for nbr in G.neighbors(node):
                yield nbr, None
    
    def _concatenate_path(self, center, path1, path2, directed=False):
        path = []
        edge_directions = None if not directed else []
        for node, direction in reversed(path1):
            path.append(self._colors[node])
        path.append(self._colors[center])
        for node, direction in path2:
            path.append(self._colors[node])
            
        if directed:
            for node, direction in reversed(path1):
                edge_directions.append(direction)
            for node, direction in path2:
                edge_directions.append(not direction)
            
        return path, edge_directions

    def _get_all_length_5_paths_with_node_as_center(self, node: int):
        length_two_paths_by_neighbour = []
        for first_neighbour, fisrt_direction in self.iter_neighbors_with_direction(self._graph, node):
            neighbor_paths = []
            for second_neighbour, second_direction in self.iter_neighbors_with_direction(self._graph, first_neighbour):
                if node == second_neighbour:
                    continue
                neighbor_paths.append([(first_neighbour, fisrt_direction), (second_neighbour, second_direction)])
            length_two_paths_by_neighbour.append(neighbor_paths)
        
        for first_neighbour_paths_index in range(len(length_two_paths_by_neighbour)):
            for second_neighbour_paths_index in range(first_neighbour_paths_index + 1, len(length_two_paths_by_neighbour)):
                for first_path in length_two_paths_by_neighbour[first_neighbour_paths_index]:
                    for second_path in length_two_paths_by_neighbour[second_neighbour_paths_index]:
                        if not self._check_path_intersection(first_path, second_path):
                            yield self._concatenate_path(node, first_path, second_path)
                                
    def get_motif_number_from_path(self, colors, directions):
        color_num = 0
        color_num_reverse = 0
        for i in range(len(colors)):
            color_num += (colors[i] << (i * 8))
            color_num_reverse += (colors[len(colors) - 1 - i] << (i * 8))
        motif_number = min(color_num, color_num_reverse)
        if directions is not None:
            if (color_num_reverse < color_num):
                directions = list(reversed(directions))
            for i in range(len(directions)):
                motif_number += ((1 if directions[i] else 0) << (len(colors) * 8 + i))
        return motif_number

    def _order_by_degree(self):
        gnx = self._graph
        return sorted(gnx, key=lambda n: len(list(nx.all_neighbors(gnx, n))), reverse=True)

    def _count_all_length_5_paths(self):
        motifs = {}
        calculated_nodes = 0
        sorted_nodes = self._order_by_degree()
        for node in sorted_nodes:
            for colors, directions in self._get_all_length_5_paths_with_node_as_center(node):
                motif_num = self.get_motif_number_from_path(colors, directions)
                if motif_num not in motifs:
                    motifs[motif_num] = 0
                motifs[motif_num] += 1
            calculated_nodes += 1
            if calculated_nodes % 100 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculated motifs for {calculated_nodes} nodes")
        return motifs

    # -------- PUBLIC API -------- #

    def build(self):
        return self._count_all_length_5_paths()
