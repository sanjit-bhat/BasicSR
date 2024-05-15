import logging
import torch
from os import path as osp

from basicsr.models import build_model
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    # create model
    model = build_model(opt)

    # Now open extra weights
    model2 = torch.load("/home/sanjit/ArbSR/experiment/ArbRCAN/model/model_150.pt")

    # TODO: Check that fusion is loading weights
    with torch.no_grad():
        upsampler = model.net_g.sa_upsample
        upsampler.body[0].weight.copy_(model2["sa_upsample.body.0.weight"])
        upsampler.body[0].bias.copy_(model2["sa_upsample.body.0.bias"])
        upsampler.body[2].weight.copy_(model2["sa_upsample.body.2.weight"])
        upsampler.body[2].bias.copy_(model2["sa_upsample.body.2.bias"])
        upsampler.routing[0].weight.copy_(model2["sa_upsample.routing.0.weight"])
        upsampler.routing[0].bias.copy_(model2["sa_upsample.routing.0.bias"])
        upsampler.offset.weight.copy_(model2["sa_upsample.offset.weight"])
        upsampler.offset.bias.copy_(model2["sa_upsample.offset.bias"])
        upsampler.weight_compress.copy_(model2["sa_upsample.weight_compress"])
        upsampler.weight_expand.copy_(model2["sa_upsample.weight_expand"])
        upsampler.tail.weight.copy_(model2["tail.1.weight"])
        upsampler.tail.bias.copy_(model2["tail.1.bias"])


    # Save weights
    torch.save({"params": model.net_g.state_dict()}, open("test.pt", "bw"))


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
