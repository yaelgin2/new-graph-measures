import os
import pickle
from itertools import product

from graphMeasures.scripts.draw_motifs.draw_motif import draw_motif

MOTIF_NUMBER = 13
MOTIF_SIZE = 4
IS_DIRECTED = False

os.environ['PATH'] = r"C:\Program Files\Graphviz\bin;" + os.environ['PATH']
OUT_FOLDER_PATH = r"graphMeasures\scripts\draw_motifs\motifs_plot_by_number"
ISOMORPHS_FOLDER_PATH = r"graphMeasures\feature_calculators\node_features_calculators\calculators\motif_variations"

def draw_all_colored_motifs(motifs_size, is_directed):
    file_name = "%d_%sdirected.pkl" % (motifs_size, "" if is_directed else "un")
    with open(os.path.join(ISOMORPHS_FOLDER_PATH, file_name), "rb") as motifs_file:
        motifs = pickle.load(motifs_file)
    for motif in set(motifs.values()):
        for colors in product([0,1,2,3], repeat=motifs_size):
            color = 0
            for i in range(motifs_size):
                color += colors[i] << (8*i)
            draw_motif(motif, motifs_size, is_directed, OUT_FOLDER_PATH, color)

def main():
    draw_all_colored_motifs(3, True)
    draw_all_colored_motifs(3, False)
    draw_all_colored_motifs(4, True)
    draw_all_colored_motifs(4, False)

if __name__ == '__main__':
    main()