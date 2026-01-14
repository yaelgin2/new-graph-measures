import json
from collections import defaultdict

import networkx as nx
from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.feature_calculators.accelerated_feature_calculators import motifs
from graphMeasures.loggers import PrintLogger

CONFIGURATION = {
    "directed_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_directed.pkl",
    "undirected_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_undirected.pkl",
    "directed_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_directed.pkl",
    "undirected_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_undirected.pkl",

    "colored_directed_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_undirected_colored.pkl",
    "colored_directed_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_undirected_colored.pkl"
}

def read_graph_file(filename):
    graph = nx.Graph()
    with open(filename) as f:
        graph_json = json.load(f)
    for node in graph_json['nodes']:
        graph.add_node(node['id'])
        graph.nodes[node['id']]['color'] = node['color']
    for edge in graph_json['links']:
        graph.add_edge(edge['source'], edge['target'])
    return graph

def main():
    G = read_graph_file(r'C:\Users\ginzb\Documents\new-graph-measures\local_tests\input\G.json')
    g_calculator = MotifsNodeCalculator(graph=G, colores_loaded=True, configuration=CONFIGURATION,
                                        level=3, calc_edges=False, logger=PrintLogger())
    g_motifs = g_calculator.build()
    print("Done with g")

    count_false_positives = 0

    for i in range(1, 50):
        S = read_graph_file(fr'C:\Users\ginzb\Documents\new-graph-measures\local_tests\input\S_{str(i)}.json')
        s_calculator = MotifsNodeCalculator(graph=S, colores_loaded=True, configuration=CONFIGURATION,
                                            level=3, calc_edges=False, logger=PrintLogger())
        s_motifs = s_calculator.build()
        print(f"Done with s{i}")
        in_g = True
        motif_apps_s = defaultdict(list)
        for node, motifs in s_motifs.items():
            for motif, apps in motifs.items():
                motif_apps_s[motif].append(apps)

        for motif, apps in s_motifs.items():
            apps = sorted(apps)
            g_apps = sorted([(g_motifs[node][motif] if motif in g_motifs[node] else 0) for node in G.nodes()])
            for j, app in enumerate(apps):
                if app > g_apps[j]:
                    in_g = False
                    break

        if not in_g and i <= 10:
            print(f"FAILED ON {i}")
        elif in_g and i > 10:
            count_false_positives += 1

    print("False positives: ", count_false_positives)

if __name__ == '__main__':
    main()