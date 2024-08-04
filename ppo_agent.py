import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import argparse
import os
import random
import time
import wandb

################################## set device ##################################

print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.action_mask = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_mask[:]


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            
            # if self.masks == False (do not allow the action), set the logits to infinity...
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        
        return -p_log_p.sum(-1)
    

class GraphEmbedNet(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, node_embed_dim, num_node, num_layer):
        super(GraphEmbedNet, self).__init__()
        self.num_node = num_node
        self.node_feature_dim = node_feature_dim

        self.gcn_list = []
        for i in range(num_layer):
            if i == 0:
                self.gcn_list.append(GCNConv(node_feature_dim, hidden_dim))
            elif i == num_layer-1:
                self.gcn_list.append(GCNConv(hidden_dim, node_embed_dim))
            else:
                self.gcn_list.append(GCNConv(hidden_dim, hidden_dim))
        print("num of gcn layer:{}".format(len(self.gcn_list)))
        self.gcn_list = nn.ModuleList(self.gcn_list)


    def forward(self, state):
        x = state.x.to(device)
        edge_index = state.edge_index.to(device)
        edge_weight = state.edge_attr.to(device)

        for gcn in self.gcn_list:
            x = gcn(x, edge_index, edge_weight)
            x = nn.functional.relu(x)
        
        # flatten x as state embeddings...
        x_emb = torch.flatten(x)

        return x_emb


# --------------------------- Define Actor Network --------------------------- #
class Actor(nn.Module):
    def __init__(self, graph_embed_net, node_embed_dim, hidden_dim, action_dim, num_node): # hidden dime = 128, action_dim = number of candidate relays
        super(Actor, self).__init__()
        self.graph_embed_net  = graph_embed_net
        self.lin_input = nn.Linear(num_node * node_embed_dim, hidden_dim)
        self.lin_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_output = nn.Linear(hidden_dim, action_dim)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x_emb = self.graph_embed_net(state)
        x_emb = nn.functional.relu(self.lin_input(x_emb))
        x_emb = nn.functional.relu(self.lin_hidden(x_emb))
        x_emb = nn.functional.relu(self.lin_hidden(x_emb))
        logits = self.lin_output(x_emb)

        # return the logits before softmax; prepare for logit-based masking
        return logits


# --------------------------- Define Critic Network -------------------------- #
class Critic(nn.Module):
    def __init__(self, graph_embed_net, node_embed_dim, hidden_dim, num_node): # hidden_dim = 256 (larger than the actor network's hidden layer)
        super(Critic, self).__init__()
        self.graph_embed_net  = graph_embed_net
        self.lin_input = nn.Linear(num_node * node_embed_dim, hidden_dim)
        self.lin_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_output = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x_emb = self.graph_embed_net(state)
        x_emb = nn.functional.relu(self.lin_input(x_emb))
        x_emb = nn.functional.relu(self.lin_hidden(x_emb))
        value = nn.functional.relu(self.lin_output(x_emb))
        return value


# ----------------------------- Define Actor-Critic ----------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, node_embed_dim, action_dim, num_node, num_layer):
        '''
        hidden_dim_1: hidden layer dimension for the graph embedding network
        hidden_dim_2: hidden layer dimension for the actor network
        hidden_dim_3: hidden layer dimension for the critic network
        '''
        
        super(ActorCritic, self).__init__()

        # graph embedding
        self.graph_embed_net = GraphEmbedNet(node_feature_dim, hidden_dim_1, node_embed_dim, num_node, num_layer).to(device)

        # actor: choose which relay
        self.actor = Actor(self.graph_embed_net, node_embed_dim, hidden_dim_2, action_dim, num_node).to(device)
        
        # critic: output a scalar value
        self.critic = Critic(self.graph_embed_net, node_feature_dim, hidden_dim_3, num_node).to(device)

    
    def act(self, state, action_mask):
        logits = self.actor(state) # logits before softmax
        dist = CategoricalMasked(logits=logits, masks=action_mask)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action, action_mask):

        logits = self.actor(state)
        dist = CategoricalMasked(logits=logits, masks=action_mask)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


# ----------------------- PPO-algorithm-based DRL agent ---------------------- #
class PPO:
    def __init__(self, node_feature_dim, num_node, hidden_dim_1, hidden_dim_2, hidden_dim_3, num_layer, action_dim, \
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, log_wandb, node_embed_dim):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(node_feature_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, node_embed_dim, action_dim, num_node, num_layer)
        self.policy_old = ActorCritic(node_feature_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, node_embed_dim, action_dim, num_node, num_layer)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        # use a different learning rate for the critic
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        self.log_wandb = log_wandb


    def select_action(self, state, action_mask):
        with torch.no_grad():
            state = state.to(device)
            action, action_logprob, state_val = self.policy_old.act(state, action_mask)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.action_mask.append(action_mask)
        return action.item()
    

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # get buffered data
        old_states = self.buffer.states
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_action_masks = torch.stack(self.buffer.action_mask, dim=0).detach().to(device)
        
        # calculate advantages
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0).detach().to(device))
        advantages = rewards.detach() - old_state_values.detach()

        # optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs = []
            state_values = []
            dist_entropy = []
            for i in range(len(old_states)):
                tmp_logprobs, tmp_state_values, tmp_dist_entropy = self.policy.evaluate(old_states[i], old_actions[i], old_action_masks[i])
                logprobs.append(tmp_logprobs)
                state_values.append(tmp_state_values)
                dist_entropy.append(tmp_dist_entropy)


            # match state_values tensor dimensions with rewards tensor
            logprobs = torch.squeeze(torch.stack(logprobs, dim=0))
            state_values = torch.squeeze(torch.stack(state_values, dim=0))
            dist_entropy = torch.squeeze(torch.stack(dist_entropy, dim=0))

            # find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # find surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            # compute the mean loss for wandb logging
            mean_value_loss = (0.5 * self.MseLoss(state_values, rewards)).mean()
            mean_actor_loss = (loss - mean_value_loss).mean()
            mean_dist_entropy = dist_entropy.mean()
            if self.log_wandb == True:
                wandb.log({"actor_loss": mean_actor_loss, "value_loss":mean_value_loss, "policy_entropy":mean_dist_entropy})


        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))