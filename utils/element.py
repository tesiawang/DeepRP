# import numpy as np

# define the class of a mesh node
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.pos_x = -1
        self.pos_y = -1
        self.is_in_topo = -1
        self.neigh_nodes = [] # a list with various length which is initialized and fixed in topo.py
        self.in_flows = [] # a list with various length
        self.out_flows = []

    def change_pos(self, pos_x):
        self.pos_x = pos_x

        
class Link:
    def __init__(self, link_id) -> None:
       self.link_id = link_id
       self.source_node = -1
       self.dest_node = -1
       self.link_capacity = -1 
       self.snr = -1000
       self.traversed_flow_set = [] # a list with various length
       self.belong_to_cliques = []


class Flow:
    def __init__(self, flow_id) -> None:
        self.flow_id = flow_id
        self.source_node = -1
        self.dest_node = -1
        self.rate = -1
        self.routing_path_nodes = [] # a list with various length
        self.routing_path_links = []
       

class Clique:
    def __init__(self, clique_id) -> None:
        self.clique_id = clique_id
        self.fair_share = -1
        self.link_set = [] # a list with various length
        self.flow_set = [] 
        self.policy = []
        # self.neigh_cliques = []



