from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from libs.pytorch_lightning.util import load_pretrained_dict
from mind.dataframe import load_behaviours_df
from mind.main.dataset import get_test_dataset, MINDCollateVal
from mind.params import Params, DataParams, ModuleParams
from models.PvP import PvP, ContentsEncoder


def load_model(ckpt_path: str, params: ModuleParams):
    state_dict = load_pretrained_dict(ckpt_path)

    model = PvP(
        pretrained_model_name=params.pretrained_model_name,
        sa_pretrained_model_name=params.sa_pretrained_model_name,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().cuda()

    return model


@torch.no_grad()
def get_loader(params: DataParams, encoder: ContentsEncoder):
    tokenizer = AutoTokenizer.from_pretrained(params.pretrained_model_name)
    dateset = get_test_dataset(
        base_dir=params.mind_path,
        tokenizer=tokenizer,
    )

    encoder = encoder.eval()
    inputs = dateset.uniq_news_inputs
    feats = {
        k: encoder.forward(move_data_to_device(v, torch.device('cuda'))).squeeze().cpu()
        for k, v in tqdm(inputs.items(), desc='Encoding val candidates')
    }
    dateset.news_feature_map = feats

    loader = DataLoader(
        dateset,
        batch_size=64,
        collate_fn=MINDCollateVal(is_test=True),
        shuffle=False,
        pin_memory=True,
    )

    return loader


@torch.no_grad()
def main():
    ckpt_path = '/mnt/ssdstuff/akirasosa/experiments/010_mind_PvP/1739370894/checkpoints/epoch=2-step=35869.ckpt'
    params = Params.load('./params/main/002.yaml')

    model = load_model(ckpt_path, params.module_params)
    loader = get_loader(params.data_params, model.encoder)

    preds = []
    for batch in tqdm(loader):
        batch = move_data_to_device(batch, torch.device('cuda'))
        logits = model.forward(batch)
        logits = logits.cpu().numpy().reshape(-1)
        preds.append(logits)
    preds = np.concatenate(preds)

    out_dir = Path('../tmp')
    out_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    main()
