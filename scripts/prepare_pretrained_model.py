
import os
import torch


def trim(model_state, keys):
    true_keys = model_state.keys()
    poped_keys = []
    for key in keys:
        for true_key in true_keys:
            if key in true_key and true_key not in poped_keys:
                model_state.pop(true_key)
                poped_keys.append(true_key)

    return model_state


if __name__ == "__main__":
    
    model = torch.load('work_dirs/pretrained/encnet_jpu_resnet101_pcontext.pth', map_location='cpu')

    model_state = trim(model['model'], ['head.encmodule.selayer', 'head.conv6', 'auxlayer'])
    
    model['model'] = model_state
    torch.save(model, 'work_dirs/pretrained/encnet_jpu_resnet101_pcontext_trimed.pth')
