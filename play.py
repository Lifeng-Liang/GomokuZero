import sys
import os

# 确保当前目录在 sys.path 中，特别是在使用嵌入式 Python 时
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

import config
from lib.game import Board
try:
    from lib.model import NetWrapper
    from lib.mcts import MCTSPlayer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Game settings
WIDTH = config.BOARD_WIDTH
HEIGHT = config.BOARD_HEIGHT
N_IN_ROW = config.N_IN_ROW

class GomokuAI:
    def __init__(self, width=15, height=15, n_in_row=5, model_file=None):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.mcts_player = None
        
        if not HAS_TORCH:
            print("PyTorch not found. AI cannot run in AlphaZero mode.")
            return

        # Default model search
        if model_file is None:
            model_file = config.get_model_path(width, height)

        if os.path.exists(model_file):
            print(f"Loading model from {model_file}")
            # NetWrapper handles device selection internally (prefers GPU)
            policy_value_net = NetWrapper(width, height, model_file=model_file)
            self.mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
        else:
            print(f"Model file {model_file} not found. AI will use random moves (untrained).")
            policy_value_net = NetWrapper(width, height)
            self.mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)

    def get_move(self, board):
        if self.mcts_player:
            return self.mcts_player.get_action(board)
        else:
            import random
            return random.choice(board.available) if board.available else -1

    def reset(self):
        if self.mcts_player:
            self.mcts_player.reset_player()

# Initialize Board and AI
board = Board(width=WIDTH, height=HEIGHT, n_in_row=N_IN_ROW)
board.init_board()
ai = GomokuAI(width=WIDTH, height=HEIGHT)

if HAS_FLASK:
    app = Flask(__name__, static_folder=config.WEB_DIR)
    CORS(app)

    @app.route('/')
    def index():
        return send_from_directory(config.WEB_DIR, 'index.html')

    @app.route('/<path:path>')
    def static_proxy(path):
        return send_from_directory(config.WEB_DIR, path)

    @app.route('/config', methods=['GET'])
    def get_config():
        return jsonify({
            "width": WIDTH,
            "height": HEIGHT,
            "n_in_row": N_IN_ROW
        })

    @app.route('/reset', methods=['POST'])
    def reset():
        board.init_board()
        ai.reset()
        return jsonify({"status": "ok", "message": "Board reset"})

    @app.route('/move', methods=['POST'])
    def move():
        data = request.json
        player_move = data.get('move')
        
        if player_move is not None:
            if player_move in board.available:
                board.do_move(player_move)
            else:
                return jsonify({"status": "error", "message": "Invalid move"}), 400
                
        end, winner = board.game_end()
        if end:
            return jsonify({
                "status": "end",
                "winner": winner,
                "board": {str(i): int(board.states_array[i]) for i in range(len(board.states_array)) if board.states_array[i] != -1},
                "last_move": player_move
            })

        # AI move
        ai_move = ai.get_move(board)
            
        if ai_move != -1:
            board.do_move(ai_move)
            end, winner = board.game_end()
            return jsonify({
                "status": "ok" if not end else "end",
                "ai_move": int(ai_move),
                "winner": winner if end else None,
                "board": {str(i): int(board.states_array[i]) for i in range(len(board.states_array)) if board.states_array[i] != -1}
            })
        else:
             return jsonify({"status": "end", "winner": -1})

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
else:
    if __name__ == '__main__':
        print("Flask not found. Web server cannot be started.")
        print("However, the GomokuAI class is available for embedded use.")
