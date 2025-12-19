import sys
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add Trainer to path to import game modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Trainer')))

from game import Board
try:
    from model import NetWrapper
    from mcts import MCTSPlayer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

app = Flask(__name__, static_folder='static')
CORS(app)

# Game settings
WIDTH = 8
HEIGHT = 8
N_IN_ROW = 5

board = Board(width=WIDTH, height=HEIGHT, n_in_row=N_IN_ROW)
board.init_board()

# Load model if exists
model_file = os.path.join(os.path.dirname(__file__), '..', 'Trainer', 'current_policy.pth')
mcts_player = None

if HAS_TORCH:
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        policy_value_net = NetWrapper(WIDTH, HEIGHT, model_file=model_file)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
    else:
        print("Model file not found. AI will use random moves (untrained).")
        policy_value_net = NetWrapper(WIDTH, HEIGHT)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
else:
    print("PyTorch not found. AI cannot run in AlphaZero mode.")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static', path)

@app.route('/reset', methods=['POST'])
def reset():
    board.init_board()
    if mcts_player:
        mcts_player.reset_player()
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
            "board": {str(k): v for k, v in board.states.items()},
            "last_move": player_move
        })

    # AI move
    if mcts_player:
        ai_move = mcts_player.get_action(board)
    else:
        # Fallback to random if no torch
        import random
        ai_move = random.choice(board.available) if board.available else -1
        
    if ai_move != -1:
        board.do_move(ai_move)
        end, winner = board.game_end()
        return jsonify({
            "status": "ok" if not end else "end",
            "ai_move": int(ai_move),
            "winner": winner if end else None,
            "board": {str(k): v for k, v in board.states.items()}
        })
    else:
         return jsonify({"status": "end", "winner": -1})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
