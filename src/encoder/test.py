import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print('Device count:', torch.cuda.device_count())
print('Current device:', torch.cuda.current_device())
print('Device name', torch.cuda.get_device_name(0))
