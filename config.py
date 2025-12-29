import os

# 棋盘配置
BOARD_WIDTH = 15
BOARD_HEIGHT = 15
N_IN_ROW = 5

# 目录配置
MODEL_DIR = 'model'
LIB_DIR = 'lib'
WEB_DIR = 'web'

# 训练配置
GAME_BATCH_NUM = 100000
CHECK_FREQ = 50

def get_model_path(width=BOARD_WIDTH, height=BOARD_HEIGHT):
    """根据棋盘大小获取模型文件路径"""
    filename = f'current_{width}x{height}_policy.pth'
    return os.path.join(MODEL_DIR, filename)
