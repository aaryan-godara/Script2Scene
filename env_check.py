import platform, sys, subprocess

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")

try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print(f"FFmpeg: {result.stdout.splitlines()[0]}")
except Exception:
    print("FFmpeg: Not Found")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"PyTorch: Not installed ({e})")
