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
    colors_string = ""
    colors = None
    if is_colored:
        colors = motif_number % (1 << (8 * motif_size))
        motif_number = motif_number >> (8 * motif_size)
        # Create a directed graph object
        colors_string = str(colors)

    edge_iter = permutations if is_directed else combinations
    graph_type = graphviz.Digraph if is_directed else graphviz.Graph

    dot = graph_type(comment="motif " + str(motif_number))

    # Add nodes
    for i in range(motif_size):
        dot.node(str(i), str(i))

    # Add edges
    for i, (x, y) in enumerate(edge_iter(range(motif_size), 2)):
        if (2 ** i) & motif_number:
            dot.edge(str(x), str(y))  # Edge from A to B with a label

    file_name = os.path.join(out_folder , "%d_%sdirected_" % (motif_number, "" if is_directed else "un") +
                             "motif_" + str(motif_size) + ".gv")
    # Render the graph to a file (e.g., 'round-table.gv.pdf') and open it
    dot.render(file_name, view=True, format='pdf')

def main():
    draw_motif(30, 4, False, OUT_FOLDER_PATH, False)

if __name__ == '__main__':
    main()