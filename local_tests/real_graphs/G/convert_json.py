import json

edges_file = "G.edges"
labels_file = "G.node_labels"
output_file = "graph.json"

def write_graph_json(edges_path, labels_path, output_path):
    with open(output_path, "w") as out:
        out.write('{\n')

        # Write nodes
        out.write('"nodes": [\n')
        first = True
        with open(labels_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                node_id, color_id = line.split()

                if not first:
                    out.write(',\n')
                first = False

                out.write(
                    f'  {{"id": {int(node_id)}, "color": {int(color_id)}}}'
                )

        out.write('\n],\n')

        # Write links
        out.write('"links": [\n')
        first = True
        with open(edges_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                src, dst = line.split()

                if not first:
                    out.write(',\n')
                first = False

                out.write(
                    f'  {{"source": {int(src)}, "target": {int(dst)}}}'
                )

        out.write('\n]\n')
        out.write('}\n')


if __name__ == "__main__":
    write_graph_json(edges_file, labels_file, output_file)
    print(f"Graph JSON written to {output_file}")
