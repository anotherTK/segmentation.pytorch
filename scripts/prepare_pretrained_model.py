
import os
import torch


def trim(model_state, keys):
    true_keys = list(model_state.keys())
    poped_keys = []
    for key in keys:
        for true_key in true_keys:
            if key in true_key and true_key not in poped_keys:
                model_state.pop(true_key)
                poped_keys.append(true_key)

    return model_state


if __name__ == "__main__":
    
    model = torch.load('work_dirs/pretrained/encnet_jpu_res101_pcontext.pth.tar', map_location='cpu')

    model_state = trim(model['state_dict'], ['head.encmodule.selayer', 'head.conv6', 'auxlayer'])
    
    trimed_model = {}
    trimed_model['model'] = model_state
    torch.save(trimed_model, 'work_dirs/pretrained/encnet_jpu_resnet101_pcontext_trimed.pth')
