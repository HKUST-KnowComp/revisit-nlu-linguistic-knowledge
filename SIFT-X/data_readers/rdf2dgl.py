# some methods/classes adpated from https://github.com/dmlc/dgl/blob/ca48787ae49e9d73ebf17b1c882cb66fb66fb828/python/dgl/contrib/data/knowledge_graph.py

from collections import Counter
from dataclasses import dataclass
import pickle
import sys
from typing import Dict, Optional

from dgl import DGLGraph
from rdflib import Literal
import numpy as np
import torch
import copy
import random
from tqdm import tqdm


class RDFReader(object):
    __graph = None
    __freq = {}

    def __init__(self, graph):
        self.__graph = graph
        self.__freq = Counter(self.__graph.predicates())

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequency
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, rel):
        if rel not in self.__freq:
            return 0
        return self.__freq[rel]

class Node:
    def __init__(self, d):
        self.data = d
        self.left = None
        self.right = None
        
def sortedArrayToBST(arr):
     
    if not arr:
        return None
 
    mid = (len(arr)) // 2
     
    root = Node(arr[mid])
     
    root.left = sortedArrayToBST(arr[:mid])
     
    root.right = sortedArrayToBST(arr[mid+1:])
    return root

def preOrder(node, link_list):
    if not node:
        return link_list
    
    l1 = preOrder(node.left, [])
    l2 = preOrder(node.right, [])
    if node.left is not None:
        link_list.append((node.data, node.left.data))
    if node.right is not None:
        link_list.append((node.data, node.right.data))
    link_list.extend(l1)
    link_list.extend(l2)
    return link_list

def rdf2dgl(rdf_graph, metadata, relation2id, graph_type, count, map_num=-1, bidirectional=True):
#     assert set(relation2id.values()) == set(range(len(relation2id)))
    count_edge = 0
    overlap = 0
    overlap_node = 0
    edge_relation = {}    
    with RDFReader(rdf_graph) as reader:
        relations = reader.relationList()
        subjects = reader.subjectSet()
        objects = reader.objectSet()

        nodes = sorted(list(subjects.union(objects)))
        assert [int(node) for node in nodes] == list(range(len(nodes)))  # to make sure the metadata-node alignment is correct
        num_node = len(nodes)
        assert num_node == len(metadata)
        num_rel = len(relations)
        num_rel = 2 * num_rel # * 2 for bi-directionality

        if num_node == 0:
            g = DGLGraph()
            g.gdata = {'metadata': metadata}
            if graph_type == "parsed":
                result = {'G': g, "Total_edge": count_edge, "Edge_type": edge_relation, "Total_node": num_node, "Select_node": overlap_node}
            elif graph_type == "skeleton":
                result = {'G': g, "Total_edge": count_edge, "Select_edge": overlap, "Total_node": num_node, "Select_node": overlap_node}
            else:
                result = {'G': g, "Total_edge": count_edge, "Select_edge": overlap, "Edge_type": edge_relation, "Total_node": num_node, "Select_node": 0}
            return result

        assert num_node < np.iinfo(np.int32).max
        
        count_edge = reader.__len__()
#         if bidirectional:
#             count_edge = count_edge * 2
            
        edge_list = []    
        if graph_type == "parsed":
            node_list = []
            for i, (s, p, o) in enumerate(reader.triples()):
                assert int(s) < num_node and int(o) < num_node
#                 rel = relation2id[p]
                rel_index = [*range(len(relation2id))]
                rel = random.choice(rel_index)
                try:
                    edge_relation[str(p)] += 1                
                except:
                    edge_relation[str(p)] = 1
                edge_list.append((int(s), int(o), rel))
                if bidirectional:
                    edge_list.append((int(o), int(s), rel + len(relation2id)))
                if int(o) not in node_list:
                    node_list.append(int(o))
                if int(s) not in node_list:
                    node_list.append(int(s))
        
            overlap = reader.__len__()
            overlap_node = len(node_list)

        elif graph_type == "skeleton":
            node_list = []

            for i, (s, p, o) in enumerate(reader.triples()):
                assert int(s) < num_node and int(o) < num_node
                rel = relation2id[p]
                if rel <= 4 or rel == 42:
                    overlap += 1
                    edge_list.append((int(s), int(o), rel))
                    sub_list = [int(s), int(o)]
                    if int(o) not in node_list:
                        node_list.append(int(o))
                    if int(s) not in node_list:
                        node_list.append(int(s))
                if bidirectional:
                    if rel <= 4 or rel == 42:
                        edge_list.append((int(o), int(s), rel + len(relation2id)))
            overlap_node = len(node_list)
            if len(edge_list) == 0:
                g = DGLGraph()
                g.gdata = {'metadata': []}
                result = {'G': g, "Total_edge": count_edge, "Select_edge": 0, "Total_node": num_node, "Select_node": 0}
                return result
        
        elif graph_type == "balanced":
            
            arr = [*range(num_node)]
            root = sortedArrayToBST(arr)
            link = []
            preOrder(root, link)

            for i in link:
                    rel_index = [*range(len(relation2id))]
                    if bidirectional:
                        rel_index = [*range(2 * len(relation2id))]
                    rel_1 = random.choice(rel_index)
                    edge_list.append((i[0], i[1], rel_1))
                    if bidirectional:
                        rel_2 = random.choice(rel_index)
                        edge_list.append((i[1], i[0], rel_2))
            overlap_node = num_node
            overlap = num_node - 1
                                
        elif graph_type == "sequential":
            
            for i in range(num_node - 1):
                rel_index = [*range(len(relation2id))]
                if bidirectional:
                    rel_index = [*range(2 * len(relation2id))]
                rel_1 = random.choice(rel_index)
                edge_list.append((i, i+1, rel_1))
                if bidirectional:
                    rel_2 = random.choice(rel_index)
                    edge_list.append((i+1, i, rel_2))
            overlap_node = num_node
            overlap = num_node - 1
            
        edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
        edge_list = np.array(edge_list, dtype=np.int)
        
    edge_src, edge_dst, edge_type = edge_list.transpose()

    # normalize by dst degree
    _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)

    node_ids = torch.arange(0, num_node, dtype=torch.long).view(-1, 1)
    edge_type = torch.from_numpy(edge_type)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

    g = DGLGraph()
    g.add_nodes(num_node)
    g.add_edges(edge_src, edge_dst)
    g.ndata.update({'id': node_ids})
    g.edata.update({'type': edge_type, 'norm': edge_norm})

    g.gdata = {'metadata': metadata}  # we add this field in DGLGraph
    if graph_type == "parsed":
        result = {'G': g, "Total_edge": count_edge, "Edge_type": edge_relation, "Total_node": num_node, "Select_node": overlap_node}
    elif graph_type == "skeleton":
        result = {'G': g, "Total_edge": count_edge, "Select_edge": overlap, "Total_node": num_node, "Select_node": overlap_node}
    else:
        result = {'G': g, "Total_edge": count_edge, "Select_edge": overlap, "Edge_type": edge_relation, "Total_node": num_node, "Select_node": overlap_node}
    return result

def relations_in(rdf_graphs):
    all_relations = set()
    for rdf_graph in tqdm(rdf_graphs):
        with RDFReader(rdf_graph) as reader:
            all_relations |= set(reader.relationList())
    return all_relations
