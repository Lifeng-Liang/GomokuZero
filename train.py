import sys
import os

# 确保当前目录在 sys.path 中，特别是在使用嵌入式 Python 时
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import random
import time
import numpy as np
import collections
from collections import deque
import torch
import argparse

import config
from lib.game import Board, Game
from lib.model import NetWrapper
from lib.mcts import MCTSPlayer

class TrainPipeline:
    def __init__(self, init_model=None, use_gpu=True):
        # params of the board and the game
        self.board_width = config.BOARD_WIDTH
        self.board_height = config.BOARD_HEIGHT
        self.n_in_row = config.N_IN_ROW
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
        self.check_freq = config.CHECK_FREQ
        self.game_batch_num = config.GAME_BATCH_NUM
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
        for state, mcts_prob, winner in play_data:
            # state shape is (4, H, W)
            # mcts_prob shape is (H*W,)
            prob_grid = mcts_prob.reshape(self.board_height, self.board_width)
            
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.rot90(state, i, axes=(1, 2))
                # For the policy, we need to match the rotation
                # Since state is already flip_ud, we rotate it directly
                # The policy grid was also constructed to match the flip_ud state
                equi_mcts_prob = np.rot90(prob_grid, i)
                extend_data.append((equi_state,
                                    equi_mcts_prob.flatten(),
                                    winner))
                # flip horizontally
                equi_state_flip = np.flip(equi_state, axis=2)
                equi_mcts_prob_flip = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state_flip,
                                    equi_mcts_prob_flip.flatten(),
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
                    save_path = config.get_model_path(self.board_width, self.board_height)
                    self.policy_value_net.save_model(save_path)
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
        # Check for default checkpoint using config
        default_model = config.get_model_path(config.BOARD_WIDTH, config.BOARD_HEIGHT)
             
        if os.path.exists(default_model):
            model_file = default_model
            print(f"[Main] No model specified. Found default checkpoint: {model_file}. Resuming training...")
        else:
            print("[Main] No model specified and no default checkpoint found. Starting training from scratch.")
    else:
        print(f"[Main] Starting training from specified model: {model_file}")

    training_pipeline = TrainPipeline(init_model=model_file, use_gpu=use_gpu)
    training_pipeline.run()
