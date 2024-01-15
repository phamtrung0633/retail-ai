import os
import yaml
import torch
import dvc.api

from torchvision import models

STAGE_NAME = 'prepare'

def prepare():
    params = dvc.api.params_show()

    arch = params[STAGE_NAME]['arch']

    if arch not in models.list_models():
        print(f'An invalid model was specified, it must be one of:\n{models.list_models()}')
        return
    
    print(f"Preparing to download '{arch}'...")

    model = models.get_model(arch, weights = 'DEFAULT')
    path = params['model']['path']

    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)

    path = os.path.join(path, params['model']['name'])

    print(f'Saving model to {os.path.join(os.getcwd(), path)}...')

    torch.save(model.state_dict(), path)

    with open(params['config'], mode = 'w') as out:
        yaml.dump({'arch': arch}, out)

if __name__ == '__main__':
    prepare()