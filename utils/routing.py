import numpy as np
import networkx as nx
import math
import random
from utils import element


def shortest_path_rerouting(nodes, node_adj_mat, flows):
    '''
    Rerouting algorithm: do not generate flows
    Flow S-D pairs are already fixed in shortest_path_routing
    
    @output: only return the routing_path_lst for further recording...
    '''
    deployed_id = []
    # physical_links = []
    del_idx = []
    deployed_nodes = []     # deployed_nodes = [nodes[0], ... nodes[4], nodes[7], ...nodes[11]]
    new_node_adj_mat = node_adj_mat.copy()
    
    for n in range(len(nodes)):
        if nodes[n].is_in_topo == True:
            deployed_id.append(n) # [2,10,14,20,...]
            deployed_nodes.append(nodes[n])
        else:
            # NOTE: node_adj_mat is a numpy array, which is not mutable
            # so here it will not affect its value out of the function scope
            new_node_adj_mat[n,:] = 0
            new_node_adj_mat[:,n] = 0

    rows, cols = np.where(new_node_adj_mat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)

    #### How to add edge weights: G.add_weighted_edges_from([(1,2,0.5),...])
    routing_path_nodes_lst = []
    for f in range(len(flows)):
        n_s = flows[f].source_node
        n_d = flows[f].dest_node
        
        path_len, path = nx.bidirectional_dijkstra(G, n_s, n_d)
        routing_path_nodes_lst.append(path)
    return routing_path_nodes_lst


def shortest_path_routing(nodes, node_adj_mat, num_flow, random_seed):
    '''
    ROUTING: Generate the routing path for each flow
    E.g., flow 1: the source node is node 1, the destination node is node 5
    the node routing path should be node 1 --> node 3 --> node 5
    the equivalent link routing path is link 1-3, link 3-5
    
    modify adjacency matrix for node topology: 
    - there is a link --> 1 (the same cost for each link); no
    - link --> Inf (set the cost as infinity)
    '''

    # FIX the RANDOM_SEED
    random.seed(random_seed)

    deployed_id = []
    # physical_links = []
    del_idx = []
    deployed_nodes = []     # deployed_nodes = [nodes[0], ... nodes[4], nodes[7], ...nodes[11]]
    
    for n in range(len(nodes)):
        if nodes[n].is_in_topo == True:
            deployed_id.append(n) # [2,10,14,20,...]
            deployed_nodes.append(nodes[n])
        else:
            del_idx.append(n)

    tmp_mat = np.delete(node_adj_mat, del_idx, axis=0) # delete rows
    dep_node_adj_mat = np.delete(tmp_mat, del_idx, axis=1) # delete cols`
    
    # call a specfic routing algorithm
    # NOTE: it is likely that a deployed node is isolated with all the other nodes
    # In this case, it is not included in the graph G
    # len(G.nodes) can be smaller than len(deployed_id)

    rows, cols = np.where(dep_node_adj_mat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)

    # set edge weights for dijsktra algorithm
    # num_dep_node = len(deployed_id)
    # cost = np.zeros([num_dep_node, num_dep_node])
    # for i in range(num_dep_node):  # i or j is only the index for deploy_nodes (a list of objects), instead of the real node_id
    #     for j in range(num_dep_node):
    #         if dep_node_adj_mat[i,j] == 1:
    #             cost[i,j] = 1
    #         else:
    #             cost[i,j] = math.inf

    #### How to add edge weights: G.add_weighted_edges_from([(1,2,0.5),...])
    routing_path_nodes_lst = []
    flows = [element.Flow(i) for i in range(num_flow)] 

    for f in range(num_flow):

        s = random.randint(0, len(deployed_nodes) - 1)
        iter = 0
        while s not in G.nodes: # to make sure s is in G (not an isolated node)
            s = random.randint(0, len(deployed_nodes) - 1) # randomly select a source node from deployed nodes, s is not the real node id!!!
            iter += 1
            if iter > 2*len(deployed_nodes):
                print("CANNOT select a source node. Check G.nodes: maybe G.nodes has no component!!")

        path_len, all_path = nx.single_source_dijkstra(G=G, source=s) # generate all the reachable paths to other nodes

        # select a REACHABLE destination (here we can also refer to path_len)
        d = random.sample(all_path.keys(), 1)[0]
        iter = 0
        while d == s:
            d = random.sample(all_path.keys(), 1)[0]
            iter += 1
            if iter > 2*len(deployed_nodes):
                print("CANNOT select a destination node. Check G.node and all the paths from the source!!")
        
        # set flow source and destination
        flows[f].source_node = deployed_nodes[s].node_id
        flows[f].dest_node = deployed_nodes[d].node_id

        # set flow path
        path = [deployed_nodes[idx].node_id for idx in all_path[d]]
        routing_path_nodes_lst.append(path)

    return routing_path_nodes_lst, flows



def rerouting_dij_with_link_capa(nodes, links, node_adj_mat, flows):
    '''
    Rerouting algorithm: do not generate flows
    Flow S-D pairs are already fixed in shortest_path_routing
    
    @output: only return the routing_path_lst for further recording...
    '''
    deployed_id = []
    # physical_links = []
    del_idx = []
    deployed_nodes = []     # deployed_nodes = [nodes[0], ... nodes[4], nodes[7], ...nodes[11]]
    new_node_adj_mat = node_adj_mat.copy()
    
    for n in range(len(nodes)):
        if nodes[n].is_in_topo == True:
            deployed_id.append(n) # [2,10,14,20,...]
            deployed_nodes.append(nodes[n])
        else:
            # NOTE: node_adj_mat is a numpy array, which is not mutable
            # so here it will not affect its value out of the function scope
            new_node_adj_mat[n,:] = 0
            new_node_adj_mat[:,n] = 0


    rows, cols = np.where(new_node_adj_mat == 1)
     # add edge weight:
    
    edge_weights = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        # links[i].link_capacity must be positive!!
        for i in range(len(links)):
            if links[i].source_node == r and links[i].dest_node == c:
                edge_weights.append(1/(links[i].link_capacity)) 
        
    weighted_edges =  zip(rows.tolist(), cols.tolist(), edge_weights)
    
    
    G = nx.Graph()
    G.add_weighted_edges_from(ebunch_to_add = weighted_edges)

    #### How to add edge weights: G.add_weighted_edges_from([(1,2,0.5),...])
    routing_path_nodes_lst = []
    for f in range(len(flows)):
        n_s = flows[f].source_node
        n_d = flows[f].dest_node
        
        path_len, path = nx.bidirectional_dijkstra(G, n_s, n_d)
        routing_path_nodes_lst.append(path)
        
        
    return routing_path_nodes_lst




def routing_dij_with_link_capa(nodes, links, node_adj_mat, num_flow, random_seed):
    '''
    ROUTING: Generate the routing path for each flow, link cost = 1/link_capacity
    '''

    # use the node_adj_mat directly, set the rows and cols to zeros
    # FIX the RANDOM_SEED
    random.seed(random_seed)

    ## maybe link cost is not needed here
    deployed_id = []
    # physical_links = []
    del_idx = []
    deployed_nodes = []     # deployed_nodes = [nodes[0], ... nodes[4], nodes[7], ...nodes[11]]
    relay_id = []
    new_node_adj_mat = node_adj_mat.copy()
    

    for n in range(len(nodes)):
        if nodes[n].is_in_topo == True:
            deployed_id.append(n) # [2,10,14,20,...]
            deployed_nodes.append(nodes[n])
        else:
            relay_id.append(n)
            new_node_adj_mat[n,:] = 0
            new_node_adj_mat[:,n] = 0

    rows, cols = np.where(new_node_adj_mat == 1)

    edge_weights = []
    # add edge weight:
    for r, c in zip(rows.tolist(), cols.tolist()):
        # links[i].link_capacity must be positive!!
        for i in range(len(links)):
            if links[i].source_node == r and links[i].dest_node == c:
                edge_weights.append(1/(links[i].link_capacity)) 
        


    weighted_edges =  zip(rows.tolist(), cols.tolist(), edge_weights)
    
    
    G = nx.Graph()
    G.add_weighted_edges_from(ebunch_to_add = weighted_edges)

    #### How to add edge weights: G.add_weighted_edges_from([(1,2,0.5),...])
    routing_path_nodes_lst = []
    flows = [element.Flow(i) for i in range(num_flow)] 

    for f in range(num_flow):
        
        s = random.sample(deployed_id, 1)[0]
        while s not in G.nodes: # to make sure s is in G (not an isolated node)
            s = random.sample(deployed_id, 1)[0]# randomly select a source node from deployed nodes
        
        path_len, all_path = nx.single_source_dijkstra(G=G, source=s) # generate all the reachable paths to other nodes


        # select a REACHABLE destination (here we can also refer to path_len)
        d = random.sample(all_path.keys(), 1)[0]
        while d == s:
            d = random.sample(all_path.keys(), 1)[0]
        
        # set flow source and destination
        flows[f].source_node = s
        flows[f].dest_node = d

        # set flow path
        path = [node_id for node_id in all_path[d]]
        routing_path_nodes_lst.append(path)

    return routing_path_nodes_lst, flows



def record_path(routing_path_nodes_lst, nodes, links, flows, num_link, num_flow):
    '''
    Make changes to links, flows, and nodes accroding to the routing path of flows 
    '''

    ###### set flow property 
    for f in range(num_flow):
        path_len = len(routing_path_nodes_lst[f])
        flows[f].routing_path_nodes = routing_path_nodes_lst[f]
        routing_path_links = []
        for n in range(path_len - 1): # indexing the node
            for i in range(num_link): # indexing the flow
                n_s = links[i].source_node
                n_d = links[i].dest_node

                if flows[f].routing_path_nodes[n] == n_s and flows[f].routing_path_nodes[n+1] == n_d:
                    routing_path_links.append(i) # E.g., [link 3--> link 5 --> link 10]
               
        flows[f].routing_path_links = routing_path_links


    ##### set link property
    for i in range(num_link):
        traversed_flow_set = []
        traversed_flow_set = [f for f in range(num_flow) if i in flows[f].routing_path_links]
        links[i].traversed_flow_set = traversed_flow_set


    ##### set node property
    in_flows = []
    out_flows = []
    for n in range(len(nodes)):
        in_flows = []
        out_flows = []
        for f in range(num_flow):
            if n in flows[f].routing_path_nodes and flows[f].source_node == n:
                out_flows.append(f)
            
            if n in flows[f].routing_path_nodes and flows[f].dest_node == n:
                in_flows.append(f)

            if n in flows[f].routing_path_nodes and flows[f].source_node != n and flows[f].dest_node == n:
                out_flows.append(f)
                in_flows.append(f)

        nodes[n].out_flows = out_flows
        nodes[n].in_flows = in_flows

    return nodes, links, flows


def record_path_single_flow(routing_path_nodes, f, links, flows, nodes):
    '''
    Make changes to nodes, links, flows accroding to the rerouting of one flow f
    '''
    num_link = len(links)
    
    ###### set flow property only for flow f
    path_len = len(routing_path_nodes)
    flows[f].routing_path_nodes = routing_path_nodes

    routing_path_links = []
    for n in range(path_len - 1): # indexing the node
        for i in range(num_link): # indexing the flow
            n_s = links[i].source_node
            n_d = links[i].dest_node
            if flows[f].routing_path_nodes[n] == n_s and flows[f].routing_path_nodes[n+1] == n_d:
                routing_path_links.append(i) # E.g., [link 3--> link 5 --> link 10]
            
    flows[f].routing_path_links = routing_path_links


    ##### set link property: can be much simplified (only care about the change from the old path to the new path)
    for i in range(num_link):
        traversed_flow_set = []
        traversed_flow_set = [f for f in range(len(flows)) if i in flows[f].routing_path_links]
        links[i].traversed_flow_set = traversed_flow_set


    ##### set node property: can be much simplified (only care about the change from the old path to the new path)
    in_flows = []
    out_flows = []
    for n in range(len(nodes)):
        in_flows = []
        out_flows = []
        for f in range(len(flows)):
            if n in flows[f].routing_path_nodes and flows[f].source_node == n:
                out_flows.append(f)
            
            if n in flows[f].routing_path_nodes and flows[f].dest_node == n:
                in_flows.append(f)

            if n in flows[f].routing_path_nodes and flows[f].source_node != n and flows[f].dest_node == n:
                out_flows.append(f)
                in_flows.append(f)

        nodes[n].out_flows = out_flows
        nodes[n].in_flows = in_flows

   

    return nodes, links, flows


