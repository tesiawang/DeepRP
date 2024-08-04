import numpy as np
from heapq import heapify, heappush, heappop
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io


def BG_clique(num_clique, num_link, num_flow, links, flows, cliques):
    '''
    BG_clique this function does max-min rate allocation and bottleneck structuring.
    - Make changes to cliques 'fair_share', 'policy'
    - Make changes to flows 'rate'

    directed edges: clique --> flow, the link is the bottleneck clique for the flow
    directed edges: flow --> clique , the flow traverses these cliques (including backward edges for gradient calculation)

    @output flow_rates, (1,num_flow)
    @output bottleneck_cliques, (1,num_clique), 1: saturated or forming a bottleneck clique
    @output clique_fair_share, (1,num_clique)
    @output bo_graph, a digraph object substance (this is a very useful object class in matlab)
    @ouput directed_edge_mat, (num_clique + num_flow, num_clique + num_flow)
    '''
    # initialization
    clique_fair_share = np.zeros(num_clique)
    flow_rates = np.zeros(num_flow)

    # link_init_time_res = np.ones([1, num_link])
    link_curr_takeup = np.zeros(num_link)
    con_sf_on_link = np.zeros(num_link)
    uncon_sf_on_link = np.zeros(num_link)
    # total_sf_on_link = len(links(l).traversed_flow_set)

    clique_remain_time_res = np.zeros( num_clique)
    clique_remain_weight = np.zeros( num_clique)
    bo_cliques = np.zeros(num_clique)
    converged_flows = np.zeros(num_flow)
    directed_edge_mat = np.zeros([num_clique + num_flow, num_clique + num_flow]) # edges between clique and flows

    policy_dict = {}

    num_busy_link = 0
    busy_links = []
    for i in range(num_link):
        if len(links[i].traversed_flow_set) > 0:
            busy_links.append(links[i])
            num_busy_link += 1


    # Initialization: compute the clique fair share before iterations
    for i in range(num_link):
        link_curr_takeup[i] = sum([flow_rates[f] for f in links[i].traversed_flow_set]) # total flow rates on this link i
        con_sf_on_link[i] = sum([1 for f in links[i].traversed_flow_set if converged_flows[f] == 1])
        uncon_sf_on_link[i] = len(links[i].traversed_flow_set) - con_sf_on_link[i]

        ##  Another expression when traversed_flow_set is a 1*num_flow set:
        # link_curr_time_res[i] = link_init_time_res[i] - sum([flow_rates[f] for f in range(num_flow) if links[i].traversed_flow_set[f] == 1])

    for k in range(num_clique):
        clique_remain_time_res[k] = 1 - sum([link_curr_takeup[i]/links[i].link_capacity for i in cliques[k].link_set])
        clique_remain_weight[k] = sum([uncon_sf_on_link[i]/links[i].link_capacity for i in cliques[k].link_set])
        clique_fair_share[k] = clique_remain_time_res[k]/clique_remain_weight[k]
        # here, clique_remain_weight must be non-zero.

    iteration = 0

    while sum(converged_flows) < num_flow : # not all the flows are converged

        # print(clique_fair_share)

        ## get the cliques with global minimum fair share
        remain_fair_share = []
        min_indices = []

        remain_fair_share = clique_fair_share[bo_cliques == 0] # non-bottlenecked cliques
        non_zero_idx = np.nonzero(remain_fair_share)[0]
        min_fair_share = min(remain_fair_share[non_zero_idx]) # min_fair_share = 3.33

        # if min_fair_share is non-empty, then min_indices is non-empty
        for k in range(num_clique):
            if bo_cliques[k] == 0 and clique_fair_share[k] == min_fair_share and min_fair_share > 0:
                min_indices.append(k) # e.g., min_indices = [1, 3, 5]

        ## allocate rates and form bottleneck structures
        for k in min_indices:
            for f in cliques[k].flow_set:
                if converged_flows[f] == 0:
                    flow_rates[f] = clique_fair_share[k]
                    directed_edge_mat[k, f + num_clique] = 1 # clique k --> flow f
                    
                    for i in flows[f].routing_path_links:
                        cli_idx = links[i].belong_to_cliques
                        directed_edge_mat[f + num_clique, cli_idx] = 1 # flow f --> clique k
                    
                    # add backward edge for gradient computing 
                    directed_edge_mat[f + num_clique, k] = 0 # here: set backward path as 0 for plotting

                
            # compute the scheduling policy in the bottleneck clique:
            # policy = np.zeros(1, len(cliques[k].link_set))
            nom = sum([uncon_sf_on_link[i] for i in cliques[k].link_set])
            link_remain_capa = [links[i].link_capacity - link_curr_takeup[i] for i in cliques[k].link_set]
            denom = sum([uncon_sf_on_link[cliques[k].link_set[idx]] * link_remain_capa[idx] for idx in range(len(cliques[k].link_set))])
            theta = clique_fair_share[k] * (nom/denom)
            policy = [theta * uncon_sf_on_link[i] for i in cliques[k].link_set]


            # mark clique k is saturated/bottlenecked 
            bo_cliques[k] = 1
            policy_dict[k] = policy # the dimension should be the same as the clique's 'link_set'

        
        # mark flow f as converged after recording all the edges for cliques in min_indices
        # if there is an edge from some clique to a flow, then this flow is converged
        for k in min_indices:
            f_ids = np.nonzero(directed_edge_mat[k,:])[0] - num_clique
            converged_flows[f_ids] = 1


        # print("STOP")
        for i in range(num_link):
            link_curr_takeup[i] = sum([flow_rates[f] for f in links[i].traversed_flow_set])
            con_sf_on_link[i] = sum([1 for f in links[i].traversed_flow_set if converged_flows[f] == 1])
            uncon_sf_on_link[i] = len(links[i].traversed_flow_set) - con_sf_on_link[i]


        for k in range(num_clique):
            ## update fair share value only for non-bottlenecked cliques
            if bo_cliques[k] == 0:
                clique_remain_time_res[k] = 1 - sum([link_curr_takeup[i]/links[i].link_capacity for i in cliques[k].link_set])
                clique_remain_weight[k] = sum([uncon_sf_on_link[i]/links[i].link_capacity for i in cliques[k].link_set])
                
                if clique_remain_weight[k] != 0:
                    clique_fair_share[k] = clique_remain_time_res[k]/clique_remain_weight[k]
                else: # the clique is 'passively' saturated after the flow rates are allocated
                    # 'uncon_sf_on_link' = 0 --> there is no unconverged flow in clique k
                    clique_fair_share[k] = 0

        # print("STOP")

        iteration = iteration + 1
        if iteration > num_clique:
            raise Exception("There is an error!")
            
        
    # show_graph_with_labels(directed_edge_mat)

    # change objects' property
    for q in range(num_clique):
        if bo_cliques[q] == 1:
            cliques[q].policy = policy_dict[q]
            cliques[q].fair_share = clique_fair_share[q]
        else:
            cliques[q].policy = np.zeros(len(cliques[q].link_set)).tolist()

    for f in range(num_flow):
        flows[f].rate = flow_rates[f]

    print("===========Finish the construction of bottleneck structures============")

    return flow_rates, bo_cliques, directed_edge_mat, cliques, flows 


def cal_grad(directed_edge_mat, cliques, flows, pert_clique_id, num_clique, num_flow, bo_cliques):

    '''
    CAL_GRADIENT this function calucates a bottleneck gradient (delta throughput/delta capacity) for one link pert_link_id (where perturbation happens)
    - To calculate all the gradients, add a loop to call the function 
    - do not change objects' property

    @input direct_edge_mat, adjencency matrix, (num_link + num_flow, num_link + num_flow)
    @input pert_clique_id, add positive perturbation (delta_capa) at this link
    @output gradient, (delta_throughput/delta_capa)
    '''

    # use MinHeap to implement

    # initilization:
    bo_heap = []
    heapify(bo_heap)

    delta_share = np.zeros(num_clique)
    delta_rate = np.zeros(num_flow)
    visited = np.zeros(num_clique + num_flow)

    for k in range(num_clique):
        if bo_cliques[k] == 0:
            visited[k] = 1 # do not need to visit non-bottlenecked cliques
    
    # add perturbation 
    # delta_link_capa = 1
    delta_R = sum(cliques[pert_clique_id].policy) # link_set = [1,3,5], policy = [0, 0.3, 0.4]
    suc_arr = np.where(directed_edge_mat[pert_clique_id, :] == 1)[0]
    delta_share[pert_clique_id] = delta_R/len(suc_arr)
    # this suc_arr must be non-empty!!! 
    
    # add the perturbation source in the heap
    # item = (key1, key2, vertexID)
    heappush(bo_heap, (cliques[pert_clique_id].fair_share, delta_share[pert_clique_id], pert_clique_id))

    
    # termination: the heap is empty
    # termination: sum(visited) < num_clique + num_flow
    while len(bo_heap) > 0:
        ele = ()
        exit_flag = 0

        for _ in range(num_clique + num_flow): # (num_clique + num_flow) is the max size for bo_heap
            if len(bo_heap) > 0:
                ele = heappop(bo_heap)
                ver_id = ele[2]
                if visited[ver_id] == 0:
                    break
            else:
                exit_flag = 1
                break
        if exit_flag == 1:
            break

        # mark the popped node as visited
        visited[ele[2]] = 1

        if ele[2] >= num_clique: # ele is a flow
            flow_id = ele[2] - num_clique

            # get successors (cliques)
            suc_cliques = np.where(directed_edge_mat[ele[2], :]==1)[0]

            for k in suc_cliques:
                # propagation function
                if bo_cliques[k] == 1 and visited[k] == 0:
                    capa_change = sum([cliques[k].policy[cliques[k].link_set.index(i)] for i in cliques[k].link_set if i in cliques[pert_clique_id].link_set])
                    nor_capa_change = capa_change/delta_R      # make sure that delta_R is not zero

                    pre_flows = np.where(directed_edge_mat[:, k]==1)[0]
                    total_rate_cost = 0
                    for f in pre_flows:
                        tmp = delta_rate[f-num_clique] * sum([cliques[k].policy[cliques[k].link_set.index(i)] for i in cliques[k].link_set if i in flows[f-num_clique].routing_path_links])
                        total_rate_cost += tmp

                    suc_flows = np.where(directed_edge_mat[k, :]==1)[0]

                    if sum([1 for f in suc_flows if visited[f] == 0]) > 0:
                        delta_share[k] = (nor_capa_change - total_rate_cost)/sum([1 for f in suc_flows if visited[f] == 0])

                        # add to heap
                        heappush(bo_heap, (cliques[k].fair_share, delta_share[k], k))
        else: # ele is a clique
            clique_id = ele[2]

            # get successors (cliques)
            suc_flows = np.where(directed_edge_mat[ele[2], :]==1)[0]

            for f in suc_flows:
                # propagation function
                pre_cliques = np.where(directed_edge_mat[:, f]==1)[0]
                delta_rate[f-num_clique] = min([delta_share[kk] for kk in pre_cliques])

                # add to heap
                heappush(bo_heap, (flows[f-num_clique].rate, delta_rate[f-num_clique], f))
            
    gradient = sum(delta_rate)

    return gradient
