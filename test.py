import torch
import torch_directml

# List DirectML devices (should show your AMD GPU)
print(torch_directml.device_count())

# Create a DirectML device and tensor
dml = torch_directml.device(0)
a = torch.tensor([1, 2, 3], device=dml)
print(a)

if torch_directml.is_available():
    print("HEER")
