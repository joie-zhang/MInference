import torch
from flash_attn import flash_attn_func
import time

# Check CUDA availability first
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# Create sample inputs
batch_size = 2
seq_len = 1024
n_heads = 8
head_dim = 64

# Generate random query, key, value tensors
q = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)

# Time the forward pass
try:
    start_time = time.time()
    out = flash_attn_func(q, k, v)
    end_time = time.time()
    
    print("\nFlash Attention is working!")
    print(f"Output shape: {out.shape}")
    print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    
except Exception as e:
    print("\nError running Flash Attention:")
    print(e)

from minference import MInferenceConfig
supported_attn_types = MInferenceConfig.get_available_attn_types()
supported_kv_types = MInferenceConfig.get_available_kv_types()
print(supported_attn_types)
print(supported_kv_types)
from minference import get_support_models
get_support_models()