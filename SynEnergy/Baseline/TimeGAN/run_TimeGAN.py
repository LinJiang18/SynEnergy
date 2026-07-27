import argparse
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
import tsgm


def minmax_scale(x):
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    x_scaled = (x - x_min) / (x_max - x_min + 1e-8)
    return x_scaled, x_min, x_max


def minmax_inverse(x_scaled, x_min, x_max):
    return x_scaled * (x_max - x_min + 1e-8) + x_min


def _generator_output_to_numpy(out):
    """Keras (torch backend) may return a CUDA tensor; numpy needs host CPU arrays."""
    if hasattr(out, "detach") and hasattr(out, "cpu"):
        return out.detach().cpu().numpy()
    if hasattr(out, "cpu") and callable(out.cpu):
        return out.cpu().numpy()
    return np.asarray(out)


def generate_samples(model, num_samples, seq_len, n_features, batch_size):
    all_fake = []

    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    for _ in range(num_full_batches):
        z = np.random.uniform(
            0, 1, size=(batch_size, seq_len, n_features)
        ).astype("float32")
        x_fake_batch = _generator_output_to_numpy(model.generator(z))
        all_fake.append(x_fake_batch)

    if remainder > 0:
        z = np.random.uniform(
            0, 1, size=(batch_size, seq_len, n_features)
        ).astype("float32")
        x_fake_batch = _generator_output_to_numpy(model.generator(z))
        all_fake.append(x_fake_batch[:remainder])

    x_fake = np.concatenate(all_fake, axis=0)
    return x_fake


def _build_timegan_graph(model, x_scaled):
    """
    tsgm only constructs `model.generator` inside `fit()`.
    Use epochs=0 to build the graph without updating weights (for load + sample).
    """
    model.fit(x_scaled, epochs=0)


def main():
    parser = argparse.ArgumentParser(description="TimeGAN training / sampling for toy example.")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Skip training: load generator weights and only generate (needs --generator-weights).",
    )
    parser.add_argument(
        "--generator-weights",
        type=str,
        default=None,
        help="Path to FinalGenerator weights (.weights.h5). Default: Dataset/Results/timegan_generator_<dataset>.weights.h5",
    )
    args = parser.parse_args()

    dataset_name = "temperature"
    input_path = "Dataset/temperature_seq_144_sample_1520.npy"

    origin_save_path = "Dataset/Results/origin_data_unnorm_" + dataset_name + "_timeGAN.npy"
    output_path = "Dataset/Results/generate_data_unnorm_" + dataset_name + "_timeGAN.npy"
    default_gen_weights = os.path.join(
        "Dataset", "Results", f"timegan_generator_{dataset_name}.weights.h5"
    )
    generator_weights_path = args.generator_weights or default_gen_weights

    batch_size = 128
    epochs = 100
    hidden_dim = 24
    n_layers = 3
    module = "gru"
    gamma = 1.0

    x = np.load(input_path).astype("float32")
    if not args.sample_only:
        np.save(origin_save_path, x)

    print("loaded data shape:", x.shape)
    if not args.sample_only:
        print("saved original data to:", origin_save_path)

    n, t, c = x.shape
    num_generate = n

    os.makedirs("Dataset/Results", exist_ok=True)

    x_scaled, x_min, x_max = minmax_scale(x)

    model = tsgm.models.timeGAN.TimeGAN(
        seq_len=t,
        n_features=c,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        batch_size=batch_size,
        module=module,
        gamma=gamma,
    )

    model.compile(
        d_optimizer=keras.optimizers.Adam(1e-3),
        g_optimizer=keras.optimizers.Adam(1e-3),
        emb_optimizer=keras.optimizers.Adam(1e-3),
        supgan_optimizer=keras.optimizers.Adam(1e-3),
        ae_optimizer=keras.optimizers.Adam(1e-3),
        emb_loss=keras.losses.MeanSquaredError(),
        clf_loss=keras.losses.BinaryCrossentropy(),
    )

    if args.sample_only:
        if not os.path.isfile(generator_weights_path):
            raise FileNotFoundError(
                "Generator weights not found; sample-only mode requires a trained checkpoint. "
                "Run full training once (weights are saved automatically) or pass "
                f"--generator-weights. Path checked: {os.path.abspath(generator_weights_path)}"
            )
        _build_timegan_graph(model, x_scaled)
        model.generator.load_weights(generator_weights_path)
        print("loaded generator weights from:", os.path.abspath(generator_weights_path))
    else:
        model.fit(x_scaled, epochs=epochs)
        model.generator.save_weights(generator_weights_path)
        print("saved generator weights to:", os.path.abspath(generator_weights_path))

    x_fake_scaled = generate_samples(
        model=model,
        num_samples=num_generate,
        seq_len=t,
        n_features=c,
        batch_size=batch_size,
    )

    x_fake = minmax_inverse(x_fake_scaled, x_min, x_max).astype("float32")

    print("generated data shape:", x_fake.shape)
    np.save(output_path, x_fake)
    print("saved generated data to:", output_path)


if __name__ == "__main__":
    main()