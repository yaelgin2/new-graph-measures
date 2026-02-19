from datetime import datetime
import networkx as nx


class PathMotifCalculator:

    def __init__(self, graph: nx.Graph):
        self._G = graph
        self._colors = nx.get_node_attributes(graph, "color")

    # -------- PRIVATE API -------- #

    def _check_path_intersection(self, path1, path2):
            return (
        path1[0] == path2[1] or
        path1[1] == path2[0] or
        path1[1] == path2[1]
    )
    
    def _get_all_length_5_paths_with_node_as_center(self, node: int):
        length_two_paths_by_neighbour = []
        for first_neighbour in self._G.neighbors(node):
            neighbor_paths = []
            for second_neighbour in self._G.neighbors(first_neighbour):
                if node == second_neighbour:
                    continue
                neighbor_paths.append([first_neighbour, second_neighbour])
            length_two_paths_by_neighbour.append(neighbor_paths)
        
        for first_neighbour_paths_index in range(len(length_two_paths_by_neighbour)):
            for second_neighbour_paths_index in range(first_neighbour_paths_index + 1, len(length_two_paths_by_neighbour)):
                for first_path in length_two_paths_by_neighbour[first_neighbour_paths_index]:
                    for second_path in length_two_paths_by_neighbour[second_neighbour_paths_index]:
                        if not self._check_path_intersection(first_path, second_path):
                            paths_colors = []
                            for first_path_node in first_path:
                                paths_colors.append(self._G.nodes[first_path_node]["color"])
                            paths_colors.append(self._G.nodes[node]["color"])
                            for second_path_node in second_path:
                                paths_colors.append(self._G.nodes[second_path_node]["color"])
                            yield paths_colors
                                
    
    def get_motif_number_from_colors(self, colors):
        color_num = 0
        color_num_reverse = 0
        for i in range(len(colors)):
            color_num += (colors[i] << (i * 8))
            color_num_reverse += (colors[len(colors) - 1 - i] << (i * 8))
        return min(color_num, color_num_reverse)

    def _count_all_length_5_paths(self):
        motifs = {}
        calculated_nodes = 0
        for node in self._G.nodes:
            for colors in self._get_all_length_5_paths_with_node_as_center(node):
                motif_num = self.get_motif_number_from_colors(colors)
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
