import random
import numpy as np
import math
import argparse

class ParaConfig():
    '''
    This is the parameter configuration class for the backhaul mesh networks
    The initialization parameters need to be modified according to the specific requirements of the simulation
    '''
    def __init__(self,
        env_name = "Mesh",
        comm_dist_thre = 150,   # transmission distance (m) based on channels
        angle_thre =  math.pi/12,   # interference angle (pi) based on beamwidth 
        rangeX = 1000,    # range of node distribution along x-axis (m)
        rangeY = 1000,    # range of node distribution along y-axis (m)
        bandw = 1e9,     # 1G bandwidth
        carrier_freq = 0.1e12,      # 0.1 Thz
        num_node = 200, #  here, num_node = num_deployed_node + num_relay
        num_can_relay = 100,    # relay density = 0.0001/m2
        num_topo = 1000,
        num_flow = 50,
        num_flow_min = 30,
        num_flow_max = 50,
        routing = "SPR",
        max_training_timesteps = int(3e6),   # break training loop if timeteps > max_training_timesteps
        max_ep_len = 10,      #  max timesteps in one episode = max_num_relay
        
        ### need to be modified ###
        p_tr =  6,      # 6W per mesh node, about 37.8 dbm, 0.3 W per mesh node, about 24 dbm 
        noise_density =  4e-21    # about -74dbm (with -174dbm/hz) 
    ):
        # ------------------------------ Network set-up ------------------------------ #
        self.env_name = env_name
        self.comm_dist_thre = comm_dist_thre
        self.angle_thre = angle_thre
        
        self.interf_dist_thre = comm_dist_thre 
        self.rangeX = rangeX
        self.rangeY = rangeY
        self.bandw = bandw
        self.carrier_freq = carrier_freq
        self.p_tr = p_tr
        self.noise_density = noise_density
        self.num_node = num_node 
        self.num_topo = num_topo
        self.num_can_relay = num_can_relay
        self.num_flow = num_flow
        self.num_flow_min = num_flow_min
        self.num_flow_max = num_flow_max
        self.routing = routing
        
        # ------------------------------ Training set-up ------------------------------ #
        self.max_training_timesteps = max_training_timesteps
        self.max_ep_len = max_ep_len
        self.print_freq = max_ep_len          # print avg reward in the interval (in num timesteps)
        self.log_freq = max_ep_len            # log avg reward in the interval (in num timesteps)
        self.save_model_freq = int(1e5)          # save model frequency (in num timesteps)
        self.action_dim = self.num_node
        