import os
from itertools import permutations, combinations

import graphviz

from graphMeasures.scripts.create_isomorphs.map_isomorphisms import OUT_FOLDER_PATH

MOTIF_NUMBER = 25
MOTIF_SIZE = 3
IS_DIRECTED = True
OUT_FOLDER_PATH = r"graphMeasures\scripts\draw_motifs\motifs_plot_by_number"

os.environ['PATH'] = r"C:\Program Files\Graphviz\bin;" + os.environ['PATH']

def draw_motif(motif_number, motif_size, is_directed, out_folder, is_colored):
    if is_colored:
        colors = motif_number % (1 << (8 * motif_size))
        motif_number = motif_number >> (8 * motif_size)
    # Create a directed graph object
    colors_string = str(colors)

    edge_iter = permutations if is_directed else combinations
    graph_type = graphviz.Digraph if is_directed else graphviz.Graph

    dot = graph_type(comment="motif " + str(motif_number))

    colors_choice = ["red", "blue", "green", "yellow", "cyan", "magenta"]
    colors_map = {}
    # Add nodes
    for i in range(motif_size):
        if colors is not None:
            color_index = colors%(1<<8)

            if not color_index in colors_map:
                for j in colors_choice:
                    if j not in colors_map.values():
                        colors_map[color_index] = j

            color = colors_map[color_index]

            dot.node(str(i), str(i), color=color)
            colors >>= 8
        else:
            dot.node(str(i), str(i))

    # Add edges
    for i, (x, y) in enumerate(edge_iter(range(motif_size), 2)):
        if (2 ** i) & motif_number:
            dot.edge(str(x), str(y))  # Edge from A to B with a label

    folder_path = os.path.join(out_folder, "%scolored" % ("" if colors is not None else "un"))
    file_name = os.path.join(folder_path , "%d_%sdirected_" %
                             (motif_number, "" if is_directed else "un") +
                             "motif_" + str(motif_size) + "_" + colors_string + ".gv")
    # Render the graph to a file (e.g., 'round-table.gv.pdf') and open it
    dot.render(file_name, view=True, format='pdf')

def main():
    draw_motif(133143986176, 4, False, OUT_FOLDER_PATH, True)

if __name__ == '__main__':
    main()