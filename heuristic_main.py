import numpy as np
import matplotlib.pyplot as plt

from utils import net_init, clique as cq, routing as rt, bottleneck as bo, capacity as capa
import heuristic_utils
import argparse
import config
from copy import copy
import scipy.io




def heur_method(para_config, args):

    # ---------------------------------------------------------------------------- #
    #                         parameters and initilization                         #
    # ---------------------------------------------------------------------------- #
    rangeX = para_config.rangeX
    rangeY = para_config.rangeY
    
    comm_dist_thre = para_config.comm_dist_thre
    drawFigure = args.drawFigure
    random_seed = args.random_seed

    num_flow = para_config.num_flow
    interf_dist_thre = para_config.interf_dist_thre
    angle_thre = para_config.angle_thre

    bandw = para_config.bandw # should use a large bandwidth
    carrier_freq = para_config.carrier_freq
    p_tr =  para_config.p_tr 
    noise_density =  para_config.noise_density
    
    num_add_re = para_config.max_ep_len
    num_can_relay = para_config.num_can_relay
    num_node = para_config.num_node
    routing_scheme = para_config.routing
    
    num_topo = 1
    
    
    
    # ---------------------------------------------------------------------------- #
    #                            main iterative process                            #
    # ---------------------------------------------------------------------------- #

    # ------ create network topology and traffic, create nodes, links, flows ----- #
    # nodes, links, num_node, num_link, node_adj_mat = net_init.topo(avg_dist, thre_dist, rangeX, rangeY, topo_type, drawFigure, random_seed)
    topo_list = net_init.ppp_topo(comm_dist_thre, rangeX, rangeY, drawFigure, random_seed, num_topo, num_node, num_can_relay)
    nodes, links, num_node, num_link, node_adj_mat = topo_list[0].nodes, topo_list[0].links, topo_list[0].num_node, topo_list[0].num_link, topo_list[0].node_adj_mat

    # ------------------------------- link capacity ------------------------------ #
    links = capa.cal_link_capacity(nodes, links, bandw, carrier_freq, p_tr, noise_density)

    # Generate flows and routing at the same time
    # -- generate routing: only consider deployed_nodes and their adjacency mat. - #
    routing_path_nodes_lst, flows = rt.routing_dij_with_link_capa(nodes, links, node_adj_mat, num_flow, random_seed)
    
    # print("The initial routing path:")
    # print(routing_path_nodes_lst)
    
    nodes, links, flows = rt.record_path(routing_path_nodes_lst, nodes, links, flows, num_link, num_flow)


    # -------------------------------- get cliques ------------------------------- #
    # link conflict graph is created based on the busy_links (links with flow traversing them)
    busy_link_adj_mat, busy_links = cq.get_link_conflict_graph(nodes, links, interf_dist_thre, angle_thre, num_link)
    cliques, links, num_clique = cq.get_maximal_cliques(links, busy_links, busy_link_adj_mat, num_flow)


    # ----------------------- initial bottleneck structure ---------------------- #
    flow_rates, bo_cliques, directed_edge_mat, cliques, flows= bo.BG_clique(num_clique, num_link, num_flow, links, flows, cliques)


    # ------------------- initial network saturation throughput ------------------ #
    
    # print("The intiial flow rates are {}".format(flow_rates))
    throughput_record = []

    curr_throughput = sum(flow_rates)
    print("The current throughput is {:.3f} Gbps".format(curr_throughput/1e9))

    throughput_record.append(curr_throughput)
    

    # ---------------------------------------------------------------------------- #
    #                    Iterative search process to add relays                    #
    # ---------------------------------------------------------------------------- #
    num_relay = 0 # the number of already added relay
    while num_relay < num_add_re:

        throughput_before = curr_throughput

        # ----------------------------- compute gradients ---------------------------- #
        grad_vec = []
        for pert_clique_id in range(len(cliques)):
            if bo_cliques[pert_clique_id] == 1:
                # note: num_clique will change! but num_node, num_link, num_flow are fixed after initialization, nodes, links, flows' properties may change!
                gradient = bo.cal_grad(directed_edge_mat, cliques, flows, pert_clique_id, len(cliques), num_flow, bo_cliques)
            else:
                gradient = 0
            grad_vec.append(gradient)

        # print("The clique gradient vector is {}".format(grad_vec))


        # ----------------- add a relay based on gradients and degrees---------------- #
        nodes, relay_id = heuristic_utils.add_relay_with_grad(nodes, links, cliques, grad_vec)
        # this relay id is in the topology now...

        if relay_id >= 0:
            # a valid relay is added
            print("=================================================================")
            print("The relay id is {}".format(relay_id))

            old_routing_path_nodes_lst = copy(routing_path_nodes_lst)
            # --------------------------- re-routing algorithm --------------------------- #
            if routing_scheme == 'SPR':
                routing_path_nodes_lst = rt.rerouting_dij_with_link_capa(nodes,links, node_adj_mat, flows)
                # print("The new rerouting path is:")
                # print(routing_path_nodes_lst)
                # print("**** Is there any change for routes **** {} ****".format(old_routing_path_nodes_lst == routing_path_nodes_lst))

                nodes, links, flows = rt.record_path(routing_path_nodes_lst, nodes, links, flows, num_link, num_flow)
                busy_link_adj_mat, busy_links = cq.get_link_conflict_graph(nodes, links, interf_dist_thre, angle_thre, num_link)
                cliques, links, num_clique = cq.get_maximal_cliques(links, busy_links, busy_link_adj_mat, num_flow)

            elif routing_scheme == 'LBR':
                nodes, links, flows, cliques = heuristic_utils.load_balance_local_rerouting(nodes, links, flows, cliques, relay_id, interf_dist_thre, angle_thre)

            else:
                print("Input a correct routing scheme!")
    


        # -------------------- re-compute the bottleneck structure ------------------- #
        old_flow_rates = copy(flow_rates)
        flow_rates, bo_cliques, directed_edge_mat, cliques, flows = bo.BG_clique(len(cliques), len(links), num_flow, links, flows, cliques)
        

        
        # ---------------------------------------------------------------------------- #
        #                         logging and printing results                         #
        # ---------------------------------------------------------------------------- #
        # print(flow_rates/1e9)
        # print("**** Is there any change for flow rates **** {} ****".format(old_flow_rates == flow_rates))

        curr_throughput = sum(flow_rates)
        gain = curr_throughput - throughput_before
        throughput_record.append(curr_throughput)
        
        num_relay += 1
        print("============Finish one iteration of network augmentation, and the added relay is {}=======".format(relay_id))
        print("The current throughput is {:.3f} Gbps".format(curr_throughput/1e9))
        print("The throughput gain is {:.3f} Gbps".format(gain/1e9))
        


    return max(throughput_record)



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type = int, default = 2, help = "")
    parser.add_argument("--drawFigure", type = int, default = 1, help = "")
    parser.add_argument("--log_wandb", type = int, default = 0, help = "")
    # parser.add_argument("--num_flow", type = int, default = 50, help = "")
    # parser.add_argument('--comm_dist_thre',type = float,default = 300, help = "" )
    # parser.add_argument('--num_node',type = int, default = 100, help = "" )
      
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    para_config = config.ParaConfig()

    num_add_re_lst = [3,6,9,12,15,18,21,24,27,30]
    num_flow_lst = [50,100,150,200,250,300,350,400,450,500]
    comm_dist_thre_lst  = [50,80,100,150,200,250,300] # m
    e2e_range = [100,200,300,400,500,600,800,1000]


    # -------------------------------- Throughput -------------------------------- #
    #### Setting 1:
    # Vary num_add_re, Fixed num_flow, Fixed num_node, Fixed comm_dist_thre
    para_config.comm_dist_thre = 200 # m
    para_config.num_flow = 200
    para_config.num_can_relay = 100
    para_config.num_node = 150 
    para_config.routing = "SPR"
    # num_node: generate num_node nodes in total
    # num_can_relay: sample num_can_relay points as candidate relays. ( # (num_node - num_can_relay) nodes are deployed)
    # max_ep_len = num_add_re: add num_add_re relays at most

    max_throu_lst = []
    for num_add_re in num_add_re_lst:
        para_config.max_ep_len = num_add_re # add num_add_re relays at most
        max_throu = heur_method(para_config, args)
        max_throu_lst.append(max_throu)


    scipy.io.savemat('heur_throu_vary_Nr.mat', {"num_add_relay": num_add_re_lst, 
                                                "heur_throu": max_throu_lst})


    # #### Setting 2:
    # # Fix num_add_re, Vary num_flow, Fixed num_node, Fixed comm_dist_thre
    # para_config.comm_dist_thre = 200
    # para_config.num_node = 100
    # para_config.max_ep_len = 30
    # for num_flow in num_flow_lst:
    #     para_config.num_flow = num_flow
    #     heur_method(para_config, args)

    # #### Setting 3:
    # # Fix num_add_re, Fix num_flow, Vary num_node, Fix comm_dist_thre
    # para_config.comm_dist_thre = 200
    # para_config.num_flow = 200
    # para_config.max_ep_len = 30

    # for num_node in num_node_lst:
    #     para_config.num_node = num_node
    #     heur_method(para_config, args)

    # #### Setting 3:
    # # Fix num_add_re, Fix num_flow, Fix num_node, Vary comm_dist_thre
    # para_config.num_flow = 200
    # para_config.max_ep_len = 30
    # para_config.num_node = 100

    # for comm_dist_thre in comm_dist_thre_lst:
    #     para_config.comm_dist_thre = comm_dist_thre
    #     heur_method(para_config=para_config, args=args)

    
    # #### Setting 4:
    # # Fix num_add_re, Fix num_flow, Fix num_node, Fix comm_dist_thre, S-D pair range (m)
    # para_config.num_flow = 200
    # para_config.max_ep_len = 30
    # para_config.num_node = 100
    # para_config.comm_dist_thre = 200
    # for e2e_dist in e2e_range:
    #     para_config.e2e_dist = e2e_dist
    #     heur_method(para_config=para_config, args=args)



