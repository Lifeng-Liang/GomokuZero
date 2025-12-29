import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyValueNet(nn.Module):
    """
    policy-value network module
    """
    def __init__(self, board_width, board_height):
        super(PolicyValueNet, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, 
                                 board_width*board_height)
        
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        
        return x_act, x_val

class NetWrapper(object):
    """
    policy-value network wrapper
    """
    def __init__(self, board_width, board_height, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            
        print(f"[{self.__class__.__name__}] Using device: {self.device}")
            
        self.policy_value_net = PolicyValueNet(board_width, board_height).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(),
                                          weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device, weights_only=True)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.from_numpy(np.array(state_batch)).to(self.device)
        self.policy_value_net.eval()
        with torch.no_grad():
             log_act_probs, value = self.policy_value_net(state_batch)
             act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.available
        current_state = board.current_state()
        
        # Add batch dimension and convert to torch tensor
        state_batch = torch.from_numpy(current_state).unsqueeze(0).to(self.device)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        act_probs_zip = zip(legal_positions, act_probs[legal_positions])
        return act_probs_zip, value.item()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch = torch.as_tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        mcts_probs = torch.as_tensor(np.array(mcts_probs), dtype=torch.float32, device=self.device)
        winner_batch = torch.as_tensor(np.array(winner_batch), dtype=torch.float32, device=self.device)

        self.policy_value_net.train()
        self.optimizer.zero_grad()
        
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), 1.0 # entropy can be calculated if needed
        
    def save_model(self, model_file):
        """ save model params to file """
        # Always save in CPU format to ensure compatibility across different hardware
        model_params = self.policy_value_net.state_dict()
        cpu_params = {k: v.cpu() for k, v in model_params.items()}
        torch.save(cpu_params, model_file)
