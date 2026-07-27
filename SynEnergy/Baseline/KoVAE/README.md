# KoVAE
[Koopman VAE (KoVAE)](https://openreview.net/pdf?id=eY7sLb0dVF), a new generative framework that is based on a novel design for the model prior.

<div align=center><img src="fig/arch_fig.png" width="100%"></div>

## :snake: Environment
Install the environment from the yaml file given here: environment.yml
```bash
conda env create -f environment.yml
```

## Data
Stocks and Energy data are located in /datasets. Sine is generated.
utils_data.py contains functions to load the data both in regular and irregular setups. Specifically, the irregular data is pre-processed by the TimeDataset_irregular class,
and it might take a while. Once the data pre-processing is done, it is saved in the /datasets folder.



## :rocket: Training

`run_regular.py` expects a single `--npy_path` to an array of shape **(N, T, C)** or **(N, T)** (treated as C=1). Min–max normalization is applied per channel; training logs go under `logs/<data_name>/...`. Synthetic data in the **original scale** is written to `datasets/Extreme/generate_<name>.npy`, where `<name>` is parsed from filenames like `ECG_seq_250_sample_1000.npy` → `ECG`.

```bash
python3 run_regular.py --npy_path datasets/Extreme/windspeed.npy --w_kl <a> --w_pred_prior <b>
```

From the `KoVAE` folder, sibling TimeVAE data is one level up: e.g. `--npy_path ../TimeVAE/data/Extreme/ECG_seq_250_sample_1000.npy` (using `../../TimeVAE` points at the wrong directory). Absolute paths always work.

You can also run the irregular setup with a similar command, adding a missing rate of 0.3,0.5,0.7:
```bash
python3 run_irregular.py --dataset <data_name> --w_kl <a> --w_pred <b> --missing_value 0.<x> --inp_dim <num_of_channels_in_dataset>
```

## :page_facing_up: Paper
```bibtex
@inproceedings{
naiman2024generative,
title={Generative Modeling of Regular and Irregular Time Series Data via Koopman {VAE}s},
author={Ilan Naiman and N. Benjamin Erichson and Pu Ren and Michael W. Mahoney and Omri Azencot},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=eY7sLb0dVF}
}
```
