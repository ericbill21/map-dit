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


def calc_beta(std, t, delta_t):
    """
    Calculates the beta value for the EMA update. Methods adapted from the paper:
    https://arxiv.org/abs/2312.02696.
    """

    gamma = std_to_gamma(np.array(std))
    return (1 - delta_t / (t + delta_t)) ** (gamma + 1)


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


def calculate_posthoc_ema(out_std, results_dir, verbose=True):
    files = [f for f in os.listdir(results_dir) if f.startswith("ema_")]
    assert len(files) > 0, "No EMA snapshots found in the results directory"
    
    regex_std = r'(?<=ema_)[0-9]*\.[0-9]+'
    regex_step = r'_(\d+)\.pt$'

    in_stds, in_ts, state_dicts_paths = [], [], []
    for file in os.listdir(results_dir):
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
        snapshot = torch.load(os.path.join(results_dir, state_dicts_paths[idx]), weights_only=True)
        return { k : v.float() for k, v in snapshot["state_dict"].items() }
    
    # Solve linear system
    weights = solve_weights(in_ts, in_gammas, out_ts, out_gamma).flatten()

    # Create the EMA state_dict
    example = torch.load(os.path.join(results_dir, state_dicts_paths[0]), weights_only=True)
    res = { k : torch.zeros_like(v, dtype=torch.float32) for k, v in example["state_dict"].items() }

    # Calculate the EMA state_dict
    for w, file in tqdm(zip(weights, state_dicts_paths), desc="Calculating EMA state_dict", total=len(weights), disable=not verbose):
        sd = torch.load(os.path.join(results_dir, file), weights_only=True)["state_dict"]

        for key in res.keys():
            res[key] += sd[key].float() * w

    return res


class EMA():
    @torch.no_grad()
    def __init__(self, net, results_dir, stds=[0.05, 0.1]):
        # Stores a reference to the model
        self.net = net
        self.emas = { s: copy.deepcopy(net) for s in stds }

        # Create the results directory to store the EMA models
        self.results_dir = os.path.join(results_dir, "ema")
        os.makedirs(self.results_dir, exist_ok=True)

    @torch.no_grad()
    def update(self, t, t_delta):
        """ Updates the EMA parameters

            Args:
                t: the current time step
                t_delta: the time step delta, i.e. the batch size
        """

        for std, ema in self.emas.items():
            beta = calc_beta(std, t, t_delta)

            for ema_p, net_p in zip(ema.parameters(), self.net.parameters()):
                ema_p.lerp_(net_p, 1 - beta) # ema_p = ema_p * (1 - beta) + net_p * beta

    @torch.no_grad()
    def save_snapshot(self, t):
        """ Saves the EMA models to disk at the current time step t

            Args:
                t: the current time step
        """

        for std, ema in self.emas.items():
            res = {
                "std"        : std,
                "t"          : t,
                "state_dict" : { k : v.half() for k, v in ema.state_dict().items() } # Save the model in float16
            }
            torch.save(res, os.path.join(self.results_dir, f"ema_{std:.3f}_{t:07d}.pt"))        
