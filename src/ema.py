
import torch 
import numpy as np
import copy
import os
import re

from tqdm import tqdm


def std_to_gamma(std):
    """Methods adapted from the paper: https://arxiv.org/abs/2312.02696."""

    if not isinstance(std, np.ndarray):
        std = np.array(std)

    tmp = std.astype(np.float64).flatten()

    tmp = tmp ** -2
    gamma = np.array([np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp], dtype=np.float64)
    return gamma.reshape(std.shape)


def gamma_to_std(gammas):
    """Methods adapted from the paper: https://arxiv.org/abs/2312.02696."""

    if not isinstance(gammas, np.ndarray):
        gammas = np.array(gammas)

    gamma = gammas.astype(np.float64)
    return np.sqrt((gamma + 1) / (np.square(gamma + 2) * (gamma + 3)))


def calc_beta(std, t):
    """
    Calculates the beta value for the EMA update. Methods adapted from the paper:
    https://arxiv.org/abs/2312.02696.
    """

    gamma = std_to_gamma(np.array(std))
    return (1 - 1 / t) ** (gamma + 1)


def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    """Methods adapted from the paper: https://arxiv.org/abs/2312.02696."""

    t_ratio = t_a / t_b

    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)

    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den


def solve_weights(t_i, gamma_i, t_r, gamma_r):
    """Methods adapted from the paper: https://arxiv.org/abs/2312.02696."""

    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)

    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    X = np.linalg.solve(A, B)
    return X


def calculate_posthoc_ema(out_std, ema_dir, verbose=True):
    files = [f for f in os.listdir(ema_dir)]
    assert len(files) > 0, "No EMA snapshots found in the results directory"

    regex_std = r"[0-9]*\.[0-9]+"
    regex_step = r"_(\d+)\.pt$"

    in_stds, in_ts, state_dicts_paths = [], [], []
    for file in os.listdir(ema_dir):
        match_std = re.search(regex_std, file)
        match_step = re.search(regex_step, file)

        if match_std and match_step:
            in_stds.append(float(match_std.group(0)))
            in_ts.append(int(match_step.group(1)))
            state_dicts_paths.append(file)

    # Convert to numpy arrays
    in_stds = np.array(in_stds)
    in_gammas = std_to_gamma(in_stds)
    in_ts = np.array(in_ts)
    out_ts = np.max(in_ts)
    out_gamma = std_to_gamma(out_std)

    # If the output std is in the input stds, return the corresponding state_dict
    if out_std in in_stds:
        idx = np.argmax((out_std == in_stds) & (out_ts == in_ts))
        
        # Convert the state_dict to float32
        snapshot = torch.load(os.path.join(ema_dir, state_dicts_paths[idx]), weights_only=True)
        return snapshot["state_dict"]

    # Solve linear system
    weights = solve_weights(in_ts, in_gammas, out_ts, out_gamma).flatten()

    # Create the EMA state_dict
    example = torch.load(os.path.join(ema_dir, state_dicts_paths[0]), weights_only=True)
    res = { k: torch.zeros_like(v, dtype=torch.float32) for k, v in example["state_dict"].items() }

    # Calculate the EMA state_dict
    for w, file in tqdm(zip(weights, state_dicts_paths), desc=f"computing ema state_dict (std={out_std})", total=len(weights), disable=not verbose):
        sd = torch.load(os.path.join(ema_dir, file), weights_only=True)["state_dict"]

        for key in res.keys():
            res[key] += sd[key].float() * w

    return res


class EMA():
    @torch.no_grad()
    def __init__(self, net, results_dir, stds=[0.05, 0.1]):
        # Stores a reference to the model
        self.emas = { s: copy.deepcopy(net).eval().requires_grad_(False) for s in stds }
        self.ema_dir = os.path.join(results_dir, "ema")
        os.makedirs(self.ema_dir, exist_ok=True)

    @torch.no_grad()
    def update(self, t, model):
        """Updates the EMA parameters.

        Args:
            t: The current time step.
            model: The model to update the EMA parameters with.

        """

        for std, ema in self.emas.items():
            beta = calc_beta(std, t)

            for name, param in ema.named_parameters():
                # ema_p = ema_p * (1 - beta) + model_p * beta
                param.lerp_(model.get_parameter(name), beta)

    @torch.no_grad()
    def save_snapshot(self, t):
        """Saves the EMA models to disk at the current time step t.

        Args:
            t: The current time step.

        """

        for std, ema in self.emas.items():
            torch.save(
                { "std": std, "t": t, "state_dict": copy.deepcopy(ema).cpu().half().state_dict() },
                os.path.join(self.ema_dir, f"{std:.3f}_{t:07d}.pt"),
            )
