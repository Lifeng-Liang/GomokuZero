# Gomoku AI (AlphaZero Implementation) | 五子棋 AI (基于 AlphaZero)

> 本项目使用 **Gemini 3 Flash** 进行开发。代码我有简单浏览，不过没有进行任何修改，包括下面的README部分都是Gemini自己写的。
> 
> 这一版棋盘大小改为了15x15，优化了代码的效率，比最早一版快了1倍多吧，目录结构也调整了。model 里面有 8x8，10x10，15x15 几个模型，都训练了1、2万次，但是效果还是比较一般，相对的，棋盘越小，效果也会好一些。

本项目是一个包含 AlphaZero 风格 AI 训练器和网页端游戏界面的五子棋项目。

---

## 项目结构 | Project Structure

- **`train.py`**: 训练启动脚本。
- **`play.py`**: 对弈服务器启动脚本 (基于 Flask)。
- **`config.py`**: 项目全局配置文件，包含棋盘大小、训练参数等。
- **`lib/`**: 核心逻辑模块，包括游戏引擎、MCTS 算法和神经网络模型。
- **`model/`**: 存放训练好的 `.pth` 模型文件。
- **`web/`**: 网页端静态文件 (HTML/JS/CSS)。
- **`requirements.txt`**: 项目依赖清单。

---

## 环境要求 | Requirements

- Python 3.11+
- 依赖项请参考 `requirements.txt`

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 使用说明 | Usage

### 1. 配置项目 | Configuration
在 `config.py` 中可以修改棋盘大小、训练轮数等参数：
- `BOARD_WIDTH`, `BOARD_HEIGHT`: 棋盘宽高。
- `GAME_BATCH_NUM`: 训练的总批次数。
- `CHECK_FREQ`: 保存模型的频率。

### 2. 训练 AI | Training
在根目录下运行：
```bash
python train.py
```
**选项**:
- `--cpu`: 强制使用 CPU 运行。
- `--model <path>`: 加载特定路径下的模型继续训练。

### 3. 开始对弈 | Playing
在根目录下运行：
```bash
python play.py
```
启动后，在浏览器访问 `http://localhost:5000` 即可开始对弈。网页会自动根据 `config.py` 中的配置调整棋盘大小。

---

## 硬件加速 | Hardware Acceleration

训练器支持 **CUDA** (NVIDIA GPU) 和 **MPS** (Apple Silicon)。
对于较小的棋盘（如 8x8），使用 `--cpu` 可能会比 GPU 更快，因为调度开销较小。

---

## 检查 GPU 支持 | GPU Check
您可以运行以下命令检查环境的硬件加速支持情况：
```bash
python lib/check_gpu.py
```
