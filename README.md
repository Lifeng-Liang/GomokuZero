# Gomoku AI (AlphaZero Implementation) | 五子棋 AI (基于 AlphaZero)


> 本项目使用 **Gemini 3 Flash** 进行开发。代码我有简单浏览，不过没有进行任何修改，包括下面的README部分都是Gemini自己写的。
> 
> 目前代码里的棋盘大小是8x8，1次完整的训练是1500轮，current_policy.pth就是训练了1500轮的模型，这个模型的智力还比较一般。我自己的测试，训练时CPU会比GPU还快，可能是因为棋盘比较小，切换的损耗比较大吧。

A Gomoku (Five-in-a-Row) project featuring an AlphaZero-style AI trainer and a web-based game interface.
一个包含 AlphaZero 风格 AI 训练器和网页端游戏界面的五子棋项目。

---

## Project Structure | 项目结构

- **Trainer/**: Python implementation of the AlphaZero training algorithm using PyTorch.
  - 基于 PyTorch 的 AlphaZero 训练算法实现。
- **Player/**: Flask-based web server and frontend to play against the trained AI.
  - 基于 Flask 的 Web 服务器和前端界面，用于与训练好的 AI 对弈。

---

## Requirements | 环境要求

- Python 3.8+
- PyTorch (CUDA recommended for training | 建议安装 CUDA 版本进行训练)
- Flask & Flask-CORS
- NumPy

```bash
pip install flask flask-cors numpy torch
```

---

## Usage | 使用说明

### 1. Training the AI | 训练 AI

Navigate to the `Trainer` directory and run the training script:
进入 `Trainer` 目录并运行训练脚本：

```bash
cd Trainer
python train.py
```

**Options | 选项**:
- `--cpu`: Force use of CPU even if GPU is available. (强制使用 CPU 运行)
- `--model <path>`: Load an existing model to continue training. (加载特定路径下的模型)

**Model Resumption | 自动续训**:
The script automatically checks for `Trainer/current_policy.pth`. If found, it will resume training from that checkpoint by default. 
脚本会自动检测 `Trainer/current_policy.pth`。如果存在，将默认从该断点继续训练。

Example | 示例:
```bash
python train.py --cpu
```

### 2. Playing the Game | 开始对弈

Navigate to the `Player` directory and start the Flask server:
进入 `Player` 目录并启动 Flask 服务器：

```bash
cd Player
python app.py
```

Open your browser and visit `http://localhost:5000` to play.
打开浏览器访问 `http://localhost:5000` 即可开始对弈。

---

## Hardware Acceleration | 硬件加速

The trainer supports **CUDA** (NVIDIA GPU) and **MPS** (Apple Silicon).
训练器支持 **CUDA** (NVIDIA GPU) 和 **MPS** (Apple Silicon)。

If training on an 8x8 board feels slow on a GPU, try adding the `--cpu` flag. For small models, CPU can sometimes be faster due to lower scheduling overhead.
如果 8x8 棋盘在 GPU 上训练较慢，请尝试使用 `--cpu`。对于小型模型，CPU 有时由于调度开销较低而运行更快。

---

## GPU Check Tool | GPU 检测工具

You can run our utility to verify your hardware support:
您可以运行我们的工具来检查您的硬件支持情况：

```bash
python Trainer/check_gpu.py
```
