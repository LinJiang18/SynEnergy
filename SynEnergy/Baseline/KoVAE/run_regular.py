import re
import torch
import torch.utils.data as Data
import numpy as np
import tensorflow as tf
import random
import argparse

import os, sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = Path(__file__).resolve().parent

from models.kovae import KoVAE
import torch.optim as optim
import logging
from utils.utils_data import extreme_npy_loading, inverse_MinMax_NTC
from utils.utils import agg_losses, log_losses, set_seed_device


def parse_data_name_from_npy(path: Path) -> str:
    """Parse `{name}_seq_{T}_sample_{N}` -> `name`; otherwise use file stem."""
    m = re.match(r"^(.+)_seq_\d+_sample_\d+$", path.stem)
    if m:
        return m.group(1)
    return path.stem


def resolve_npy_path(raw: str) -> Path:
    """Resolve .npy path: absolute, or first match among cwd / KoVAE / Baseline."""
    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        out = raw_path.resolve()
        if out.is_file():
            return out
        raise FileNotFoundError(f".npy not found: {out}")

    tried = []
    bases = [Path.cwd(), PROJECT_DIR, PROJECT_DIR.parent]
    for base in bases:
        cand = (base / raw_path).resolve()
        tried.append(cand)
        if cand.is_file():
            return cand

    # Repo layout: Baseline/TimeVAE/data/Extreme/<file> (helps if `..` depth is off by one)
    fallback = (PROJECT_DIR.parent / "TimeVAE" / "data" / "Extreme" / raw_path.name).resolve()
    tried.append(fallback)
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        ".npy not found. Tried (in order, cwd → KoVAE → Baseline → TimeVAE/data/Extreme/<name>):\n  "
        + "\n  ".join(str(t) for t in tried)
    )


def define_args():
    parser = argparse.ArgumentParser(description="KoVAE — train on (N,T,C) .npy only")

    # general
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--pinv_solver', type=bool, default=False)

    # data: (N, T, C) array in .npy; T and C inferred from file
    parser.add_argument(
        '--npy_path',
        type=str,
        required=True,
        help='(N,T,C) or (N,T) .npy. Relative path: first resolved vs cwd, then KoVAE, then Baseline.',
    )
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--missing_value', type=float, default=0.)

    # model
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='the hidden dimension of the output decoder lstm')

    # loss params
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--w_rec', type=float, default=1.)
    parser.add_argument('--w_kl', type=float, default=.007)
    parser.add_argument('--w_pred_prior', type=float, default=0.005)

    return parser


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('cuda is available')
    else:
        device = torch.device("cpu")
    return device


parser = define_args()
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args):

    args.device = set_seed_device(args.seed)

    npy_path = resolve_npy_path(args.npy_path)

    data_name = parse_data_name_from_npy(npy_path)
    arr_norm, scaler, t_file, c_file = extreme_npy_loading(str(npy_path))
    args.seq_len = t_file
    args.inp_dim = c_file

    ori_tensor = torch.from_numpy(arr_norm)
    train_set = Data.TensorDataset(ori_tensor)

    name = 'KOVAE-{}_bs={}-rnn_size={}-z_dim={}-lr={}-n_layers={}=' \
           '-weight:kl={}-pred={}-w_decay={}-seed={}'.format(
        args.epochs, args.batch_size, args.hidden_dim, args.z_dim, args.lr, args.num_layers,
        args.w_kl, args.w_pred_prior, args.weight_decay, args.seed)
    args.log_dir = '%s/%s/%s' % ('./logs', data_name, name)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                   worker_init_fn=seed_worker, generator=g)

    logging.info('Loaded %s shape (N,T,C)=%s, data_name=%s', npy_path, arr_norm.shape, data_name)

    model = KoVAE(args).to(device=args.device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    tf.io.gfile.makedirs(os.path.dirname(args.log_dir))

    params_num = sum(param.numel() for param in model.parameters())
    logging.info(args)
    logging.info("number of model parameters: {}".format(params_num))
    print("number of model parameters: {}".format(params_num))

    logging.info("Starting training loop at step %d." % (0,))

    for epoch in range(0, args.epochs):
        logging.info("Running Epoch : {}".format(epoch + 1))

        model.train()
        losses_agg_tr = []
        for i, data in enumerate(train_loader, 1):
            X = data[0].to(args.device)

            optimizer.zero_grad()
            x_rec, Z_enc, Z_enc_prior = model(X)

            losses = model.loss(X, x_rec, Z_enc, Z_enc_prior)
            losses[0].backward()
            optimizer.step()

            losses_agg_tr = agg_losses(losses_agg_tr, losses)

        log_losses(epoch, losses_agg_tr, model.names)

    logging.info("Training is complete")

    args.device = set_seed_device(args.seed)
    model.eval()
    with torch.no_grad():
        generated_data = []
        for data in train_loader:
            n_sample = data[0].shape[0]
            generated_data.append(model.sample_data(n_sample).detach().cpu().numpy())
    generated_data = np.vstack(generated_data)

    logging.info("Data generation is complete, shape %s", generated_data.shape)

    ori_batches = []
    for data in train_loader:
        ori_batches.append(data[0].detach().cpu().numpy())
    ori_data = np.vstack(ori_batches)

    from metrics.discriminative_torch import discriminative_score_metrics
    args.device = set_seed_device(args.seed)
    disc_res = []
    for ii in range(10):
        dsc = discriminative_score_metrics(ori_data, generated_data, args)
        disc_res.append(dsc)
    disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)

    print('test/disc_mean: ', disc_mean)
    print('test/disc_std: ', disc_std)

    out_dir = PROJECT_DIR / 'datasets' / 'Extreme'
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_orig = inverse_MinMax_NTC(generated_data, scaler)
    out_path = out_dir / f'generate_{data_name}.npy'
    np.save(out_path, gen_orig.astype(np.float32))
    logging.info("Saved generated (N,T,C)=%s to %s", gen_orig.shape, out_path)


if __name__ == '__main__':
    main(args)
