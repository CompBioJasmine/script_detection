#!/usr/bin/python
import argparse
import csv
import stat
import networkx as nx
import numpy as np
from networkx.algorithms import *
from networkx.algorithms.connectivity import local_node_connectivity

def get_average_per_ip(lines, temporal):
    source_ips = {}
    dest_ips = {}
    for i in range(len(lines)):
        source_ips[i] = lines[i][5]
        dest_ips[i] = lines[i][6]
    ip_feature_dict = {}
    t_f = csv.reader(temporal)
    lines1 = list(t_f)
    for i in range(len(lines1)):
        source = source_ips[i]
        dest = dest_ips[i]
        value = lines1[i]
        for i in range(len(value)):
            value[i] = float(value[i])
        if source not in ip_feature_dict:
            ip_feature_dict[source] = [np.asarray(value)]
        else:
            np.append(ip_feature_dict[source], [np.asarray(value)])
        if dest not in ip_feature_dict:
            ip_feature_dict[dest] = [np.asarray(value)]
        else:
            np.append(ip_feature_dict[dest], [np.asarray(value)])
    ips_w_avg = {}

    for ip in ip_feature_dict.keys():
        avg = np.mean(ip_feature_dict[ip], axis=0)
        ips_w_avg[ip] = avg

    return ips_w_avg

def create_graph(lines):
    G = nx.MultiDiGraph()
    edges = []
    nodes = set()
    for i in range(len(lines)):
        edges.append((lines[i][5], lines[i][6]))
        nodes.add(lines[i][5])
        nodes.add(lines[i][6])
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def graph_features(graph):
    features = {}
    feats = []
    for node in list(graph.nodes):
        feats = []
        #for neighbor in graph[node]:
        #    feats.append(local_node_connectivity(graph, node, neighbor))
        feats.append(graph.degree(node))
        feats.append(graph.in_degree(node))
        feats.append(graph.out_degree(node))
        features[node] = feats
    return features

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfileips', type=str, required=True,
                        help="The netflow data.")
    parser.add_argument('--inputfileoldfeatures', type=str, required=True,
                        help="Temporal features.")
    parser.add_argument('--outputfile', type=str, required=True,
                        help="An empty file to write to.")

    FLAGS = parser.parse_args()
    with open(FLAGS.inputfileips, "r") as ips, open(FLAGS.inputfileoldfeatures, "r") as temporal:
        ip_f = csv.reader(ips)
        lines = list(ip_f)
        avgs = get_average_per_ip(lines,temporal)
        graph_object = create_graph(lines)
        graph_f = graph_features(graph_object)


        for ip in avgs.keys():
            for i in avgs[ip]:
                i = truncate(i, 3)

    with open(FLAGS.outputfile, "wb") as out:
        wr = csv.writer(out)
        for ip in avgs.keys():
            l = avgs[ip].tolist()
            for i in graph_f[ip]:
                l.append(i)
            wr.writerow(l)
        out.close()

main()
