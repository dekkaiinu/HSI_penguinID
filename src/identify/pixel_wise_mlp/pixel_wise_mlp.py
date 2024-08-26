import numpy as np
import torch

def pixel_wise_mlp(hs_pixel: np.ndarray, model: torch.nn.Module, device: torch.device):
    model.to(device)
    model.eval()

    input = torch.from_numpy(hs_pixel).type(torch.FloatTensor)
    input = input.to(device)
        
    output_tensor = model(input)
    output = output_tensor.data.cpu().numpy()

    return output