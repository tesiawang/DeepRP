import torch
import wandb
import numpy as np
import random
import argparse
from torch_geometric.data import Data
import config
import time

from ppo_agent import PPO
import heuristic_utils
from utils import net_init, clique as cq, routing as rt, bottleneck as bo, capacity as capa



class Environment:
    def __init__(self, topo_list, traffic_list, para_config, args):
        self.rangeX = para_config.rangeX
        self.rangeY = para_config.rangeY
        
        self.comm_dist_thre = para_config.comm_dist_thre
        self.interf_dist_thre = para_config.comm_dist_thre * 1.2
        self.angle_thre = para_config.angle_thre
    
        self.bandw = para_config.bandw
        self.carrier_freq =  para_config.carrier_freq 
        self.p_tr =  para_config.p_tr 
        self.noise_density =  para_config.noise_density;  # about -74dbm (with -174dbm/hz)

        self.nodes = []
        self.links = []
        self.flows = []
        self.cliques = []

        self.clique_grad = [] # need update
        self.node_adj_mat = [] # num_node * num_node, do not need update
        self.curr_throu = -1 # need update

        #### important paratmers for environment ####
        # the envionment main
        self.topo_list = topo_list # [topology 1, topology 2, ...]
        self.traffic_list = traffic_list # the number of traffic [10,5,3,...]

        self.node_feature_dim = args.node_feature_dim
        self.routing = para_config.routing # SPR for shortest_path_routing, LBF for load-balanced routing
        self.random_seed = args.random_seed
        self.is_normalized = args.is_normalized




    def reset(self, topo_idx, traffic_idx):
        '''
        set the initial state and action mask
        '''

        
        print("=================Reset()==================")


        ## set the seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        
        # use the topo_idx and traffic_idx to initialize/reset the environment
        self.nodes = self.topo_list[topo_idx].nodes
        self.links = self.topo_list[topo_idx].links
        self.num_node = len(self.nodes)
        self.num_link = len(self.links)
        self.num_flow = self.traffic_list[traffic_idx]
        self.node_adj_mat = self.topo_list[topo_idx].node_adj_mat


        ### use random_seed to generate random flow distribution
        ### self.flows = net_init.generate_traffic(self.num_flow, self.nodes, random_seed) 

        # ------------------------------- link capacity ------------------------------ #
        self.links = capa.cal_link_capacity(self.nodes, self.links, self.bandw, self.carrier_freq, self.p_tr, self.noise_density)

        # generate routing
        routing_path_nodes_lst, self.flows = rt.routing_dij_with_link_capa(self.nodes, self.links, self.node_adj_mat, self.num_flow, self.random_seed)
        print("The initial routing paths are:")
        print(routing_path_nodes_lst)


        self.nodes, self.links, self.flows = rt.record_path(routing_path_nodes_lst, self.nodes, self.links, self.flows, self.num_link, self.num_flow)
        
        # get cliques
        busy_link_adj_mat, busy_links = cq.get_link_conflict_graph(self.nodes, self.links, self.interf_dist_thre, self.angle_thre, self.num_link)
        self.cliques, self.links, self.num_clique = cq.get_maximal_cliques(self.links, busy_links, busy_link_adj_mat, self.num_flow)

        # bottleneck structure and throughput
        flow_rates, bo_cliques, directed_edge_mat, self.cliques, self.flows = bo.BG_clique(self.num_clique, self.num_link, self.num_flow, self.links, self.flows, self.cliques)
        self.curr_throu = sum(flow_rates)


        # gradients
        clique_grads = []
        for pert_clique_id in range(len(self.cliques)):
            if bo_cliques[pert_clique_id] == 1:
                gradient = bo.cal_grad(directed_edge_mat, self.cliques, self.flows, pert_clique_id, len(self.cliques), self.num_flow, bo_cliques)
            else:
                gradient = 0
            clique_grads.append(gradient)

        print("The initial clique gradients are")
        print(clique_grads)
        


        ###### generate the state as a Data object: [state.x, state.edge_index, state.edge_attr]

        # --------------------------- generate node feature -------------------------- #
        x = torch.zeros([self.num_node, self.node_feature_dim])
        for n in range(self.num_node):
            # 0-st col: is deployed or not (or is relay or not)
            if self.nodes[n].is_in_topo == True:
                x[n,0] = 1
            else:
                x[n,0] = 0
            x[n,1] = len(np.nonzero(self.node_adj_mat[n,:])[0]) # 1-st col: the degree of the node
            x[n,2] = len(self.nodes[n].in_flows) # 2-nd col: the num of incoming flows
            x[n,3] = len(self.nodes[n].out_flows) # 3-rd col: the num of outgoing flows
    

        # ---------------------------- generate edge index --------------------------- #
        rows, cols = np.where(self.node_adj_mat == 1)
        edge_index = torch.tensor(list(zip(rows.tolist(), cols.tolist())) ,dtype= torch.long)
        edge_attr = torch.zeros(edge_index.shape[0])


        # ---------------------------- generate edge attr ---------------------------- #
        idx = 0
        for e in edge_index: # a tuple (n_s, n_d)
            n_s = e[0]
            n_d = e[1]
            link_id = [i for i in range(self.num_link) if self.links[i].source_node == n_s and self.links[i].dest_node == n_d]
            edge_attr[idx] = self.links[link_id[0]].link_capacity
            idx += 1
        
        
        if bool(self.is_normalized) == 1:
        #### normalization for x, edge_attr
            # Standardize 'x'
            x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-7)
            x = x_normalized
            
            # Standardize 'edge_attr': edge weights cannot be negative!!!
            edge_attr_normalized = (edge_attr - edge_attr.min())/(edge_attr.max() - edge_attr.min() + 1e-7)
            edge_attr = edge_attr_normalized
        
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # ------------------------- generate the action mask ------------------------- #
        # if the action_dim = num_node, mask all the deployed nodes and relays that are not in high-grad cliques
        action_mask = torch.zeros(len(self.nodes), dtype = torch.bool) # true -> allow action, false -> do not allow action
        
        ### to start with, we do not allow any of the actions
        for n in range(len(self.nodes)):
            if self.nodes[n].is_in_topo == False:
                action_mask[n] = True # if the node is not in the topology (to be added)
                # then we allow this action to happen 
        
        #### DRL with gradients
        # sorted_clique_id = np.argsort(-np.array(clique_grads)) # descending order [3,0,1,2] for 4 cliques, the clique with index 3 has the largest gradient
        # q_max_grad = sorted_clique_id[0]
        # link_set = self.cliques[q_max_grad].link_set
        # clique_deployed_nodes = []
        # for i in link_set:
        #     if self.links[i].source_node not in clique_deployed_nodes:
        #         clique_deployed_nodes.append(self.links[i].source_node)
        #     if self.links[i].dest_node not in clique_deployed_nodes:
        #         clique_deployed_nodes.append(self.links[i].dest_node)

        # # for n in range(len(self.nodes)):
        # #     if self.nodes[n].is_in_topo == False and self.nodes[n] in clique_deployed_nodes:
        # #         action_mask[n] = True


        # -------------------- use the Data class to define state -------------------- #
        state = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)


        return state, action_mask


    def step(self, action):
        

        print("===========Step()============")

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # ------------------------ action: activate one relay ------------------------ #
        activate_relay_id = action
        print("The newly added relay id is {}".format(activate_relay_id))
        self.nodes[activate_relay_id].is_in_topo = True


        # ------------------------------ local rerouting ----------------------------- #
        if self.routing == 'SPR':  ### !!!shortest path re-routing is different from the initial path re-routing algorithm!!!
            routing_path_nodes_lst = rt.rerouting_dij_with_link_capa(self.nodes, self.links, self.node_adj_mat, self.flows)
            print("The new rerouting path is:")
            print(routing_path_nodes_lst)

            self.nodes, self.links, self.flows = rt.record_path(routing_path_nodes_lst, self.nodes, self.links, self.flows, self.num_link, self.num_flow)
            busy_link_adj_mat, busy_links = cq.get_link_conflict_graph(self.nodes, self.links, self.interf_dist_thre, self.angle_thre, self.num_link)
            self.cliques, self.links, self.num_clique = cq.get_maximal_cliques(self.links, busy_links, busy_link_adj_mat, self.num_flow)


        elif self.routing == 'LBR':
            self.nodes, self.links, self.flows, self.cliques = heuristic_utils.load_balance_local_rerouting(self.nodes, self.links, self.flows, self.cliques, activate_relay_id, self.interf_dist_thre, self.angle_thre)
        
        
        else:
            print("Input a correct routing scheme!")


        # ------------- compute rewards and update the current throughput ------------ #
        flow_rates, bo_cliques, directed_edge_mat, self.cliques, self.flows = bo.BG_clique(len(self.cliques), len(self.links), len(self.flows), self.links, self.flows, self.cliques)
        reward = sum(flow_rates) - self.curr_throu
        self.curr_throu = sum(flow_rates)


        # ---------------------------- update state vector --------------------------- #
        x = torch.zeros([self.num_node, self.node_feature_dim])
        for n in range(self.num_node):
            # 0-st col: is deployed or not (or is relay or not)
            if self.nodes[n].is_in_topo == True:
                x[n,0] = 1
            else:
                x[n,0] = 0
            x[n,1] = len(np.nonzero(self.node_adj_mat[n,:])[0]) # 1-st col: the degree of the node
            x[n,2] = len(self.nodes[n].in_flows) # 2-nd col: the num of incoming flows
            x[n,3] = len(self.nodes[n].out_flows) # 3-rd col: the num of outgoing flows
        
        # Standardize 'x'
        if bool(self.is_normalized) == 1:
            x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-7)
            x = x_normalized
        
        
        # ------------------------- generate the action mask ------------------------- #
        # gradients
        clique_grads = []
        for pert_clique_id in range(len(self.cliques)):
            if bo_cliques[pert_clique_id] == 1:
                gradient = bo.cal_grad(directed_edge_mat, self.cliques, self.flows, pert_clique_id, len(self.cliques), self.num_flow, bo_cliques)
            else:
                gradient = 0
            clique_grads.append(gradient)

        print("The updated clique gradients are")
        print(clique_grads)


        # if the action_dim = num_node, mask all the deployed nodes and relays that are not in high-grad cliques
        action_mask = torch.zeros(len(self.nodes), dtype = torch.bool) # true -> allow the action, false -> do not allow action
        
        for n in range(len(self.nodes)):
            if self.nodes[n].is_in_topo == False:
                action_mask[n] = True # if the node is not in the topology (to be added)
                # then we allow this action to happen 
        
        # In the beginning: we do not allow any action to happen
        # sorted_clique_id = np.argsort(-np.array(clique_grads)) # descending order [3,0,1,2] for 4 cliques, the clique with index 3 has the largest gradient
        # q_max_grad = sorted_clique_id[0]
        # link_set = self.cliques[q_max_grad].link_set

        # clique_deployed_nodes = []
        # for i in link_set:
        #     if self.links[i].source_node not in clique_deployed_nodes:
        #         clique_deployed_nodes.append(self.links[i].source_node)
        #     if self.links[i].dest_node not in clique_deployed_nodes:
        #         clique_deployed_nodes.append(self.links[i].dest_node)


        # -------------------- use the Data class to define state -------------------- #
        state = Data(x=x, edge_index=self.edge_index.t().contiguous(), edge_attr=self.edge_attr)


        # ------ check whether there is any other available relay in the network ----- #
        done = True
        for n in range(len(self.nodes)):
            if self.nodes[n].is_in_topo == False: 
                done = False

        return state, reward, done, action_mask


def wandb_init(para_config, args):
    
    # save the wandb config dict

    global_hyp = dict()
    for k, v in para_config.__dict__.items():
        if type(v) in [int, float, str, bool] and not k.startswith('_'):
            global_hyp.update({k: v})

    for k, v in args.__dict__.items():
        if type(v) in [int, float, str, bool] and not k.startswith('_'):
            global_hyp.update({k: v})

    curr_time = time.strftime('_%d-%H-%M-%S')
    wandb.init(project='DeepRP', config=global_hyp, name='PPO' + curr_time)
    


def train(para_config, args):
    
    if bool(args.log_wandb) == True:
        wandb.login()
        wandb_init(para_config, args)
        print("successfully set wandb")
    
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    # ---------------------------- set the agent ---------------------------- #
    ppo_agent = PPO(args.node_feature_dim, para_config.num_node, args.hidden_dim, args.num_layer, para_config.action_dim, \
                 args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip, bool(args.log_wandb), args.node_embed_dim)
    

    # --------------- create topology list and traffic pattern list -------------- #
    topo_list = net_init.ppp_topo(para_config.comm_dist_thre, para_config.rangeX, para_config.rangeY, \
                                 bool(args.drawFigure), args.random_seed, para_config.num_topo, para_config.num_node, para_config.num_relay)

    traffic_list = [random.randint(para_config.num_flow_min, para_config.num_flow_max) for _ in range(len(topo_list))]

    env = Environment(topo_list, traffic_list, para_config=para_config, args=args)


    # ----------------------------- training process ----------------------------- #
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0
    topo_idx = 0
    traffic_idx = 0
    
    while time_step <= para_config.max_training_timesteps:
         
        # reset the environment for the new episode
        state, action_mask = env.reset(topo_idx, traffic_idx) # reset for different network topologies and traffic patterns
        current_ep_reward = 0

        # -------------- Start one complete episode for relay adding process: the max # of added relay < max_ep_len ------------- #
        for t in range(1, para_config.max_ep_len+1):
            
            action = ppo_agent.select_action(state, action_mask) # this step is on the gpu
            state, reward, done, action_mask = env.step(action) # this step is on the cpu
            print("One action is taken: the added relay id is {}".format(action))

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward

            # update PPO agent every 'update_step' timesteps
            if time_step % args.update_step == 0:
                ppo_agent.update() # this step is on the gpu
            if bool(args.log_wandb) == True:
                wandb.log({"instant_reward": reward})

            print("Timestep: {}, Instant reward: {:4f}".format(time_step, reward))
            if done:
                break
            
        print("Episode: {}, Timestep: {}, Current_episode_reward: {:4f}".format(i_episode, time_step, current_ep_reward))

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        # log_running_reward += current_ep_reward
        # log_running_episodes += 1
        
        if bool(args.log_wandb) == True:
            wandb.log({"current_ep_reward": current_ep_reward})
            wandb.log({"print_running_reward": print_running_reward})

        i_episode += 1
        topo_idx += 1
        traffic_idx += 1
        # break: if the whole training process is over
        if topo_idx >= len(topo_list):
            break
    wandb.finish()


########## PPO parameters ##########
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--K_epochs", type = int, default = 1, help = "update policy for K epochs in one PPO update")
    parser.add_argument("--eps_clip", type = float, default = 0.2, help = "clip parameter for PPO")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount factor")
    parser.add_argument("--lr_actor", type = float, default = 0.001, help = "learning rate for actor network")
    parser.add_argument("--lr_critic", type = float, default = 0.003, help = "learning rate for critic network")
    parser.add_argument("--num_layer", type = int, default = 2, help = "number of GCNs for node embedding")
    parser.add_argument("--hidden_dim_1", type = int, default = 64, help = "hidden dimension of GCN layer")
    parser.add_argument("--hidden_dim_2", type = int, default = 128, help = "hidden dimension of GCN layer")
    parser.add_argument("--hidden_dim_3", type = int, default = 256, help = "hidden dimension of GCN layer")
    parser.add_argument("--node_feature_dim", type = int, default = 4, help = "dimension of node orginal feature")
    parser.add_argument("--node_embed_dim", type = int, default = 4, help = "dimension of node embedding")
    parser.add_argument("--update_step", type=int, default=20, help = "")
    
    # other configs
    parser.add_argument("--random_seed", type = int, default = 0, help = "")
    parser.add_argument("--drawFigure", type = int, default = 1, help = "")
    parser.add_argument("--log_wandb", type = int, default = 1, help = "")
    parser.add_argument("--same_topo", type = int, default = 1, help = "")
    parser.add_argument("--is_normalized", type = int, default = 1, help = "")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    para_config = config.ParaConfig()
    
    train(para_config, args)