import numpy as np
from numba import jit, njit

@njit(fastmath=True)
def check_winner_fast(width, height, states_array, n, last_move):
    """Optimized winner check: only check around the last move"""
    if last_move == -1:
        return False, -1
    
    player = states_array[last_move]
    if player == -1:
        return False, -1
        
    h = last_move // width
    w = last_move % width
    
    # 4 directions: horizontal, vertical, 2 diagonals
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dh, dw in directions:
        count = 1
        # search in one direction
        nh, nw = h + dh, w + dw
        while 0 <= nh < height and 0 <= nw < width and states_array[nh * width + nw] == player:
            count += 1
            nh += dh
            nw += dw
            
        # search in the opposite direction
        nh, nw = h - dh, w - dw
        while 0 <= nh < height and 0 <= nw < width and states_array[nh * width + nw] == player:
            count += 1
            nh -= dh
            nw -= dw
            
        if count >= n:
            return True, player
            
    return False, -1

@njit(fastmath=True)
def get_current_state_numba(width, height, states_array, current_player, last_move, move_count):
    """Numba optimized function to calculate current state"""
    # Create the state directly in the final required orientation (flipped if needed)
    # We'll stick to the original logic but make it faster
    square_state = np.zeros((4, width, height), dtype=np.float32)
    
    for m in range(width * height):
        player = states_array[m]
        if player == -1:
            continue
            
        h = m // width
        w = m % width
        
        # Original code used square_state[:, ::-1, :] which is a vertical flip
        # We can do it directly here by changing h to (height - 1 - h)
        h_idx = height - 1 - h
        
        if player == current_player:
            square_state[0][h_idx][w] = 1.0
        else:
            square_state[1][h_idx][w] = 1.0
            
    if last_move != -1:
        h = last_move // width
        w = last_move % width
        square_state[2][height - 1 - h][w] = 1.0
        
    if move_count % 2 == 0:
        square_state[3][:, :] = 1.0
        
    return square_state

class Board(object):
    """
    Board for the game
    """
    def __init__(self, width=8, height=8, n_in_row=5):
        self.width = width
        self.height = height
        self.states_array = np.full(width * height, -1, dtype=np.int32)
        self.n_in_row = n_in_row
        self.players = [1, 2]
        self.move_count = 0

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('Board width and height can not be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.available = list(range(self.width * self.height))
        self.states_array.fill(-1)
        self.last_move = -1
        self.move_count = 0

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        return get_current_state_numba(
            self.width, self.height, self.states_array, 
            self.current_player, self.last_move, self.move_count
        )

    def do_move(self, move):
        self.states_array[move] = self.current_player
        self.available.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move
        self.move_count += 1

    def undo_move(self, move, prev_last_move):
        """Undo a move (used in MCTS backtracking)"""
        self.states_array[move] = -1
        self.available.append(move) # This might break order, but MCTS doesn't care about available order
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = prev_last_move
        self.move_count -= 1

    def has_a_winner(self):
        if self.move_count < self.n_in_row * 2 - 1:
            return False, -1
        return check_winner_fast(self.width, self.height, self.states_array, self.n_in_row, self.last_move)

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.available):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def copy(self):
        """Return a copy of the board"""
        new_board = Board(self.width, self.height, self.n_in_row)
        new_board.current_player = self.current_player
        new_board.available = self.available[:]
        new_board.states_array = self.states_array.copy()
        new_board.last_move = self.last_move
        new_board.move_count = self.move_count
        return new_board

class Game(object):
    """game server"""
    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height
        
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states_array[loc]
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
            
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
                
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
