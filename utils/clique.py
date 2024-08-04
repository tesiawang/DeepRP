import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils import element

def get_link_conflict_graph(nodes, links, dist_thre, angle_thre, num_link):

    '''
    Construct the link conflict graph based on the protocol model
    @input dist_thre, a threshold for interfering distance calculated by path loss model in mm-wave or THz
    @input angle_thre, 1/2 main lobe width based on beamforming ability, in radians [0, pi]
    @output link_adj_mat, num_link * num_link 0-1 array

    only consider the effective links with flows on it
    '''
    num_busy_link = 0
    busy_links = []
    for i in range(num_link):
        if len(links[i].traversed_flow_set) > 0:
            busy_links.append(links[i])
            num_busy_link += 1

    busy_link_adj_mat = np.zeros([num_busy_link, num_busy_link])

    # cross-link interference
    for i in range(num_busy_link):

        # get real node id
        n_si = busy_links[i].source_node
        n_di = busy_links[i].dest_node

        for j in range(num_busy_link):
            
            n_sj = busy_links[j].source_node
            n_dj = busy_links[j].dest_node

            # cross-link interference
            if j != i and n_si != n_sj and n_di != n_dj:

                # half duplex
                if n_si == n_dj or n_sj == n_di:
                    busy_link_adj_mat[i,j] = 1
                    busy_link_adj_mat[j,i] = 1

                else:
                    # compute the distance between link j's source to link i's destination
                    d_ji = np.sqrt(pow(nodes[n_sj].pos_x - nodes[n_di].pos_x,2) + pow(nodes[n_sj].pos_y - nodes[n_di].pos_y,2))
                    
                    if d_ji < dist_thre:
                        # compute the divation angle between link j's source to link i's destination
                        tmp_d = np.sqrt(pow(nodes[n_sj].pos_x - nodes[n_dj].pos_x,2) + pow(nodes[n_sj].pos_y - nodes[n_dj].pos_y,2))
                        tmp_dd = np.sqrt(pow(nodes[n_dj].pos_x - nodes[n_di].pos_x,2) + pow(nodes[n_dj].pos_y - nodes[n_di].pos_y,2))

                        if d_ji == 0 or tmp_d == 0:
                            print("Invalid number")
                        else:
                            cos_theta = (pow(d_ji,2) + pow(tmp_d,2) - pow(tmp_dd,2))/(2*d_ji*tmp_d)

                        theta_ji = np.arccos(cos_theta) # radian in [0,pi]

                        if theta_ji < angle_thre:
                            busy_link_adj_mat[i,j] = 1
                            busy_link_adj_mat[j,i] = 1

            # analog beamforming constraint: a node can only transmit/receive one beam at a time
            # can we form multiple beams at the same time by hybrid beamforming???
            if j != i and (n_si == n_sj or n_di == n_dj): 
                busy_link_adj_mat[i,j] = 1
                busy_link_adj_mat[j,i] = 1


    return busy_link_adj_mat, busy_links


def get_maximal_cliques(links, busy_links, busy_link_adj_mat, num_flow):
    '''
    Get all the maximal cliques from a link conflict graph.
    - Make changes to link property: "belong_to_cliques"
    - Create and return clique objects

    @input link_adj_mat, a num_link*num_link 0-1 numpy array
    @output cliques, a list of clique objects

    '''

    rows, cols = np.where(busy_link_adj_mat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    
    G = nx.Graph()
    G.add_edges_from(edges)

    # get a list of all maximal cliques
    c_lst = list(nx.find_cliques(G))
    cliques = []
    
    for i in range(len(c_lst)):
        cliques.append(element.Clique(i))
        real_link_ids = [busy_links[idx].link_id for idx in c_lst[i]]
        cliques[i].link_set = real_link_ids # c_lst[i] is the index of busy_links, not the real link idx

        ### set cliques property
        flow_ind_vec = np.zeros(num_flow)
        for l_id in cliques[i].link_set:
            # find the flows that traverse this link
            f_set = links[l_id].traversed_flow_set
            # add these flows into clique's property
            flow_ind_vec[f_set] = 1
        cliques[i].flow_set = list(np.where(flow_ind_vec == 1)[0])


    # IMPORTANT: if there is an all-zero row in busy_link_adj_mat
    # we consider this isolated link as a clique itself...
    clique_curr_id = len(c_lst)
    for row in range(len(busy_links)):
        if sum(busy_link_adj_mat[row,:]) == 0:
            link_id = busy_links[row].link_id
            cliques.append(element.Clique(clique_curr_id))
            
            cliques[clique_curr_id].link_set = [link_id]
            cliques[clique_curr_id].flow_set = links[link_id].traversed_flow_set 
            clique_curr_id += 1


    ### set links property
    for i in range(len(links)):
        belong_to_cliques = [q for q in range(len(cliques)) if i in cliques[q].link_set]

        links[i].belong_to_cliques = belong_to_cliques
        

    return cliques, links, len(cliques)



# def show_graph_with_labels(adjacency_matrix, mylabels):
#     rows, cols = np.where(adjacency_matrix == 1)
#     edges = zip(rows.tolist(), cols.tolist())
#     gr = nx.Graph()
#     gr.add_edges_from(edges)
#     nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
#     plt.show()