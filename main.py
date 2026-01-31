import torch
print(torch.version.cuda)       # should now print CUDA version
print(torch.cuda.is_available()) # should be True
print(torch.cuda.get_device_name(0)) # should print MX110
