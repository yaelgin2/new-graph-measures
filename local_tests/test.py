import os
from graphMeasures import FeatureManager

# set of features to be calculated
feats = ["motif4", "louvain"]

# path to the graph's edgelist or nx.Graph object
# graph = os.path.join("examples", "example_graph.txt")
graph = "examples\\example_graph.txt"

# The path in which one would like to save the pickled features calculated in the process.
dir_path = "..\\local_tests\\out"

configuration = "configuration\\config.json"
colors = "examples\\example_colors.json"

# More options are shown here. For information about them, refer to the file.
ftr_calc = FeatureManager(graph, feats, configuration, colors, dir_path=dir_path, acc=False, directed=False,
                             gpu=True, device=0, verbose=True, should_zscore=False)

# Calculates the features. If one do not want the features to be saved,
# one should set the parameter 'should_dump' to False (set to True by default).
# If the features was already saved, you can set force_build to be True.
ftr_calc.calculate_features(force_build=True)
features = ftr_calc.get_features() # return pandas Dataframe with the features
print(features)