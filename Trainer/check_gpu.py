import torch
import sys
import os

# Add current directory to path for potential local module checks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_gpu():
    print("-" * 50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test a simple tensor operation on GPU
        try:
            x = torch.randn(3, 3).to("cuda")
            y = torch.randn(3, 3).to("cuda")
            z = x @ y
            print("Successfully executed a matrix multiplication on GPU!")
        except Exception as e:
            print(f"Error testing GPU operation: {e}")
    else:
        print("\n[!] CUDA is not available.")
        if "cpu" in torch.__version__:
            print("Note: You are currently using a CPU-only build of PyTorch.")
            print("To use a GPU, please install a CUDA-enabled version of PyTorch.")
            print("Example: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("Note: If you have an NVIDIA GPU, make sure you have the correct drivers and CUDA Toolkit installed.")
            
    print("-" * 50)

if __name__ == "__main__":
    check_gpu()
