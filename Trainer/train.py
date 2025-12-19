import random
import time
import numpy as np
import collections
from collections import deque
import sys
import os

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import Board, Game
from model import NetWrapper
from mcts import MCTSPlayer
import torch
import argparse

class TrainPipeline:
    def __init__(self, init_model=None, use_gpu=True):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        
        # GPU check
        self.use_gpu = use_gpu and torch.cuda.is_available()
        print(f"[TrainPipeline] GPU Requested: {use_gpu}, Available: {torch.cuda.is_available()}")
        if self.use_gpu:
            print(f"[TrainPipeline] Using Device: {torch.cuda.get_device_name(0)}")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             self.use_gpu = True
             print(f"[TrainPipeline] Using MPS (Metal Performance Shaders)")
        else:
            self.use_gpu = False
            print("[TrainPipeline] Using CPU for training")
        
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = NetWrapper(self.board_width,
                                               self.board_height,
                                               model_file=init_model,
                                               use_gpu=self.use_gpu)
        else:
            # start training from a new policy-value net
            self.policy_value_net = NetWrapper(self.board_width,
                                               self.board_height,
                                               use_gpu=self.use_gpu)
        
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                batch_start = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                batch_finish_play = time.time()
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    batch_finish_update = time.time()
                    print("Batch i:{}, episode_len:{}, play_time:{:.2f}s, update_time:{:.2f}s".format(
                        i+1, self.episode_len, batch_finish_play - batch_start, batch_finish_update - batch_finish_play))
                else:
                    print("Batch i:{}, episode_len:{}, play_time:{:.2f}s".format(
                        i+1, self.episode_len, batch_finish_play - batch_start))
                
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.policy_value_net.save_model('./current_policy.pth')
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gomoku AI')
    parser.add_argument('--model', type=str, help='load an existing model file')
    parser.add_argument('--cpu', action='store_true', help='force use CPU for training')
    args = parser.parse_args()
    
    # use_gpu defaults to True, but if --cpu is provided, it becomes False
    use_gpu = not args.cpu
    
    model_file = args.model
    if model_file is None:
        # Check for default checkpoint in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model = os.path.join(script_dir, 'current_policy.pth')
        if os.path.exists(default_model):
            model_file = default_model
            print(f"[Main] No model specified. Found default checkpoint: {model_file}. Resuming training...")
        else:
            print("[Main] No model specified and no default checkpoint found. Starting training from scratch.")
    else:
        print(f"[Main] Starting training from specified model: {model_file}")

    training_pipeline = TrainPipeline(init_model=model_file, use_gpu=use_gpu)
    training_pipeline.run()
