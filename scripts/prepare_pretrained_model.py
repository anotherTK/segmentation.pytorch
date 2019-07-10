
import os
import torch


def trim(model_state, keys):
    for key in keys:
        if key in model_state.keys():
            model_state.pop(key)

    return model_state


if __name__ == "__main__":
    
    model = torch.load('work_dirs/pretrained/encnet_jpu_resnet101_pcontext.pth', map_location='cpu')

    model_state = trim(model['model'], ['head.encmodule.selayer', 'head.conv6', 'auxlayer']
    
    model['model'] = model_state
    torch.save(model, 'work_dirs/pretrained/encnet_jpu_resnet101_pcontext_trimed.pth')
