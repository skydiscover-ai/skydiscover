from utils import *
from simulator import *
from broadcast import BroadCastTopology
from pathlib import Path
import networkx as nx
import subprocess
import argparse
import json
import sys
import os


def N_dijkstra(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(source_node)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology


def N_direct(src, dsts, G, num_partitions):
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        edge = G[src][dst]
        for j in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, [[src, dst, edge]])

    return bc_topology


def MULTI_MDST(src, dsts, G, num_partitions):
    # Construct MDST path based on original graph
    h = G.copy()
    MDST_graphs = []
    while len(list(h.edges())) > 0:
        _, MDST_graph = MDST(src, dsts, h, 1)
        print("MDST graph: ", MDST_graph.edges.data())
        MDST_graphs.append(MDST_graph)
        h.remove_edges_from(list(MDST_graph.edges()))

    print("Number of MDSTs: ", len(MDST_graphs))


def Min_Steiner_Tree(src, dsts, G, num_partitions, hop_limit=3000):
    source_v, dest_v = src, dsts

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(source_v)) + list(nx.selfloop_edges(h)))

    nodes, edges = list(h.nodes), list(h.edges)
    num_nodes, num_edges = len(nodes), len(edges)
    id_to_name = {nodes.index(n) + 1: n for n in nodes}

    config_loc = "write.set"
    write_loc = "test.stplog"
    param_loc = "test.stp"

    with open(config_loc, "w") as f:
        f.write('stp/logfile = "use_probname"')
        f.close()

    scipstp_bin = os.environ.get("SCIPSTP_BIN", "scipstp")
    command = f" {scipstp_bin}"
    command += f" -f {param_loc} -s {config_loc} -l {write_loc}"

    def construct_stp():
        section_begin = '33D32945 STP File, STP Format Version 1.0\n\nSECTION Comment\nName "Relay: cloud regions"\nCreator "SkyDiscover"\n'
        section_begin += f'Remark "Cloud region problem adapted from relay"\nEND\n\nSECTION Graph\n'
        section_begin += f"Nodes {num_nodes}\nEdges {num_edges}\nHopLimit {hop_limit}\n"

        Edge_info = []
        cnt = 0
        for edge in edges:
            s, d = nodes.index(edge[0]) + 1, nodes.index(edge[1]) + 1
            cost = h[edge[0]][edge[1]]["cost"]
            cnt += 1
            Edge_info.append(f"A {s} {d} {cost}\n")
            if cnt == num_edges:
                Edge_info.append("END\n")

        s = nodes.index(source_v) + 1
        v = [nodes.index(i) + 1 for i in dest_v]
        terminal_info = [f"T {i}\n" for i in v]
        terminal_info.append("END\n\nEOF")
        section_terminal = f"""\nSECTION Terminals\nRoot {s}\nTerminals {len(dest_v)}\n"""

        with open(param_loc, "w") as f:
            f.write(section_begin)
            for edge in Edge_info:
                f.write(edge.lstrip())
            f.write(section_terminal)
            for t in terminal_info:
                f.write(t)
            f.close()
        return

    def read_result(loc):
        di_stree_graph = nx.DiGraph()
        with open(loc, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("E") and len(line.split()) == 3:
                    l = line.split()
                    src_r, dst_r = id_to_name[int(l[1])], id_to_name[int(l[2])]
                    di_stree_graph.add_edge(src_r, dst_r, **G[src_r][dst_r])

        # overlays = [node for node in di_stree_graph.nodes if node not in [source_v]+dest_v]
        return di_stree_graph

    construct_stp()  # construct problem to a file
    process = subprocess.Popen(command, shell=True)  # run the steiner tree solver
    process.wait()
    solution_graph = read_result(loc=write_loc)

    print(
        f"Number of overlays added: {len(solution_graph.nodes) - (1 + len(dsts))}, {[node for node in solution_graph.nodes if node not in [src]+dsts]}"
    )
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    os.remove(config_loc)
    os.remove(write_loc)
    os.remove(param_loc)

    return append_src_dst_paths(src, dsts, solution_graph, bc_topology)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonfile", help="input json file")
    parser.add_argument("-a", "--algo", type=str, nargs="?", const="")
    parser.add_argument("-n", "--num-vms", type=int, nargs="?", const="")
    args = vars(parser.parse_args())
    print("Args:", args)

    print(f"\n==============> Baseline generation")
    with open(args["jsonfile"], "r") as f:
        config_name = args["jsonfile"].split("/")[1].split(".")[0]
        config = json.loads(f.read())

    # generate default graph with node and edge info
    # G = make_nx_graph(throughput_path="profiles/aws_throughput_11_8.csv")
    G = make_nx_graph(num_vms=int(args["num_vms"]))

    # src, dst
    source_node = config["source_node"]
    terminal_nodes = config["dest_nodes"]

    print(f"source_v = '{source_node}'")
    print(f"dest_v = {terminal_nodes}")
    # baseline path generations
    if args["algo"] is None:
        algorithms = [
            "Ndirect",
            "MDST",
            # "HST",
        ]
    else:
        algorithms = [args["algo"]]
    print(f"Algorithms: {algorithms}\n")

    directory = f"paths/{config_name}"
    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    num_partitions = config["num_partitions"]
    for algo in algorithms:
        outf = f"{directory}/{algo}.json"
        print(f"Generate {algo} paths into {outf}")
        if algo == "Ndirect":
            bc_t = N_direct(source_node, terminal_nodes, G, num_partitions)
        elif algo == "MDST":
            bc_t, mdgraph = MDST(source_node, terminal_nodes, G, num_partitions)
        elif algo == "MULTI-MDST":
            bc_t = MULTI_MDST(source_node, terminal_nodes, G, num_partitions)
        elif algo == "HST":
            bc_t = Min_Steiner_Tree(source_node, terminal_nodes, G, num_partitions)
        elif algo == "Ndijkstra":
            bc_t = N_dijkstra(source_node, terminal_nodes, G, num_partitions)
        else:
            raise NotImplementedError(algo)

        bc_t.set_num_partitions(config["num_partitions"])  # simple baseline, don't care about partitions, simply set it

        with open(outf, "w") as outfile:
            outfile.write(
                json.dumps(
                    {
                        "algo": algo,
                        "source_node": bc_t.src,
                        "terminal_nodes": bc_t.dsts,
                        "num_partitions": bc_t.num_partitions,
                        "generated_path": bc_t.paths,
                    }
                )
            )

    # put the evaluate logic here
    input_dir = "paths"  # input paths
    output_dir = "evals"  # eval results
    with open(sys.argv[1], "r") as f:
        config_name = sys.argv[1].split("/")[1].split(".")[0]
        config = json.loads(f.read())

    input_dir += f"/{config_name}"
    output_dir += f"/{config_name}"
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    simulator = BCSimulator(int(args["num_vms"]), output_dir)
    for algo in algorithms:
        path = f"{input_dir}/{algo}.json"
        simulator.evaluate_path(path, config)  # path of algorithm output, basic config to evaluate

    # nx.draw(mdgraph, with_labels=True)
    # plt.show()
    # h.render(filename="Ndirect")
