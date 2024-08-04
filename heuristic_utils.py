import numpy as np
from utils import clique as cq, routing as rt, bottleneck as bo
from copy import copy



def add_relay_with_grad(nodes, links, cliques, grads):
    '''
    Add one relay with the guidance of clique gradients
    - add one relay based on max connectivity to already deployed node within the high-impact clique
    - change this relay's property: is_in_topo = True
    '''

    # ------------- sort the cliques by clique_gradient descendingly ------------- #
    sorted_clique_id = np.argsort(-np.array(grads)) # descending order [3,0,1,2] for 4 cliques, the clique with index 3 has the largest gradient

    q_max_grad = sorted_clique_id[0]
    link_set = cliques[q_max_grad].link_set # only busy links with flow are in the link_set
    print("**** The high-impact cliques are Clique {} ****".format(q_max_grad))


    clique_deployed_nodes = []
    for i in link_set:
        if links[i].source_node not in clique_deployed_nodes:
            clique_deployed_nodes.append(links[i].source_node)
        if links[i].dest_node not in clique_deployed_nodes:
            clique_deployed_nodes.append(links[i].dest_node)
    
    # the candidate nodes include the nodes that are in proximity of the deployed nodes
    cand_nodes = []
    for n in clique_deployed_nodes:
        cand_nodes += nodes[n].neigh_nodes

    # -------------------- add one relay based on max connectivity ------------------- #
    relay_degree = {}

    for n in cand_nodes:
        # cand_nodes may have duplicated nodes, we need to rule out them
        if n not in relay_degree and nodes[n].is_in_topo == False:
            # calculate the candidtae relay's degree to other deployed nodes
            neigh_set = nodes[n].neigh_nodes # a list of node ids
            conn_degree = 0
            for nn in neigh_set:
                if nodes[nn].is_in_topo == True:
                    conn_degree += 1
            relay_degree[n] = conn_degree


    if relay_degree == {}:
        # there is no relays around this clique, we need to resort to another clique with a lower impact
        relay_id = -1
    else: 
        relay_id = max(relay_degree, key=relay_degree.get)
        nodes[relay_id].is_in_topo = True # add this relay into the topology

    print("=============Finish adding the relay===========")
    print("The added relay id is {}".format(relay_id))
    
    return nodes, relay_id


def add_relay_without_grad(nodes):
    '''
    Add one relay without the guidance of clique gradients
    - add one relay based on max connectivity to already deployed node
    - change this relay's property: is_in_topo = True
    '''
   
    relay_degree = {}
    for n in range(len(nodes)):
        if n not in relay_degree and nodes[n].is_in_topo == False:
            # calculate the node's degree to other deployed nodes
            conn_degree = 0
            for nn in nodes[n].neigh_nodes:
                if nodes[nn].is_in_topo == True:
                    conn_degree += 1
            
            relay_degree[n] = conn_degree

    relay_id = max(relay_degree, key=relay_degree.get)
    nodes[relay_id].is_in_topo == True

    print("=============Finish adding the relay===========")
    print("The added relay id is {}".format(relay_id))

    return nodes, relay_id


def load_balance_local_rerouting(nodes, links, flows, cliques, relay_id, dist_thre, angle_thre):
    '''
    Based on the shortest-path-routing, do local rerouting:
    - Only if local rerouting happens, change the property of nodes('in_flows, out_flows'), links('belong_to_cliques'), flows ('routing_path_...')
    - get the new cliques
    - Load-Balance-Factor: (max_share - min_share)/max_share, A hill-climbing algorithm to minimize LBF
    '''

    # ------------------ Find the feasible node pairs to reroute ----------------- #
    node_pair = []
    num_common_flow = []
    for ns in nodes[relay_id].neigh_nodes:
        for nd in nodes[relay_id].neigh_nodes:
            if ns != nd:
                common_flow = [f for f in range(len(flows)) if f in nodes[ns].out_flows and f in nodes[nd].in_flows]
                node_pair.append((ns,nd))
                num_common_flow.append(len(common_flow))


    # ------ find the node pair with the max number of flows to be rerouted ------ #
    idx = num_common_flow.index(max(num_common_flow))
    pair = node_pair[idx]
    cand_flows = [f for f in range(len(flows)) if f in nodes[pair[0]].out_flows and f in nodes[pair[1]].in_flows]


    # ---------------------------- get the current LBF --------------------------- #
    clique_fair_shares = get_clique_fair_shares(links, flows, cliques)
    LBF_new = (max(clique_fair_shares) - min(clique_fair_shares)) / max(clique_fair_shares)


    # -- if reroute a flow in cand_flows can reduce the LBF metric, then reroute - #
    for f in cand_flows:
        LBF = LBF_new

        routing_path_nodes = flows[f].routing_path_nodes
        ns_idx = routing_path_nodes.index(pair[0])
        nd_idx = routing_path_nodes.index(pair[1])
        del routing_path_nodes[ns_idx + 1 : nd_idx] 
        routing_path_nodes.insert(ns_idx + 1, relay_id)

        # create a new copy, and then rebind 'new_links' to the new copy
        new_links = copy(links)
        new_flows = copy(flows)
        new_nodes = copy(nodes)

        ########Only try rerouting which does not happen yet#########

        # change flows' object attributes
        new_nodes, new_links, new_flows = rt.record_path_single_flow(routing_path_nodes, f, new_links, new_flows, new_nodes)

        # get new link conflict graph
        new_busy_link_adj_mat, new_busy_links = cq.get_link_conflict_graph(new_nodes, new_links, dist_thre, angle_thre, len(new_links))

        # get new maximal cliques
        new_cliques, new_links, _ = cq.get_maximal_cliques(new_links, new_busy_links, new_busy_link_adj_mat, len(new_flows))

        # compute the new LBF: all the clique fair share --> (max() - min())/max()
        new_clique_fair_shares = get_clique_fair_shares(new_links, new_flows, new_cliques)
        LBF_new = (max(new_clique_fair_shares) - min(new_clique_fair_shares)) / max(new_clique_fair_shares)
        
        print("Evaluating whether to reroute flow {}".format(f))
        print("The previous LBF is {:3f}".format(LBF))
        print("The new LBF with rerouting is {:3f}".format(LBF))

        # Check if LBF is reduced, if so, reroute this flow; and reset the current links, flows, and cliques
        # Rebind the reference 'nodes' to 'new_nodes'

        if LBF_new < LBF:
            nodes = new_nodes
            links = new_links
            flows = new_flows
            cliques = new_cliques

            print("=========Finishing rerouting flows, and get the new links, nodes, cliques=============")
        else:
            print("Do not reroute this flow {} as there will be no gains in LBF!".format(f))


    print("=========Finishing load-balanced rerouting=============")
    
    return nodes, links, flows, cliques


def get_clique_fair_shares(links, flows, cliques):
    '''
    Compute all the fair shares for cliques
    '''

    num_flow = len(flows)
    num_link = len(links)
    num_clique = len(cliques)

    clique_fair_share = np.zeros(num_clique)
    flow_rates = np.zeros(num_flow)
    link_curr_takeup = np.zeros(num_link)
    con_sf_on_link = np.zeros(num_link)
    uncon_sf_on_link = np.zeros(num_link)
    converged_flows = np.zeros(num_flow)
    clique_remain_time_res = np.zeros( num_clique)
    clique_remain_weight = np.zeros( num_clique)

    for i in range(len(links)):
        link_curr_takeup[i] = sum([flow_rates[f] for f in links[i].traversed_flow_set]) # total flow rates on this link i
        con_sf_on_link[i] = sum([1 for f in links[i].traversed_flow_set if converged_flows[f] == 1])
        uncon_sf_on_link[i] = len(links[i].traversed_flow_set) - con_sf_on_link[i]

    for k in range(num_clique):
        clique_remain_time_res[k] = 1 - sum([link_curr_takeup[i]/links[i].link_capacity for i in cliques[k].link_set])
        clique_remain_weight[k] = sum([uncon_sf_on_link[i]/links[i].link_capacity for i in cliques[k].link_set])
        clique_fair_share[k] = clique_remain_time_res[k]/clique_remain_weight[k]

    return clique_fair_share.tolist()



