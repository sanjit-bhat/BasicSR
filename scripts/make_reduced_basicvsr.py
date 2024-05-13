import argparse
import torch

parser = argparse.ArgumentParser()
base_path = '/home/sanjit/BasicSR/experiments/pretrained_models/BasicVSR'
default_infile = base_path + '/BasicVSR_REDS4-543c8261.pth'
default_outfile = base_path + '/BasicVSR_trim.pth'
parser.add_argument('--infile', default=default_infile)
parser.add_argument('--outfile', default=default_outfile)
args = parser.parse_args()

model = torch.load(args.infile)
params = model['params']
last_layers = ['upconv1.weight', 'upconv1.bias',
               'upconv2.weight', 'upconv2.bias',
               'conv_hr.weight', 'conv_hr.bias',
               'conv_last.weight', 'conv_last.bias']
for layer in last_layers:
    params.pop(layer)
torch.save(model, args.outfile)
