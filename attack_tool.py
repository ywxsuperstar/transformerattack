from collections import defaultdict

from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F

from breaching.attacks.auxiliaries.common import optimizer_lookup

import torch
import time

from breaching.cases.data.datasets_text import meric_two_sentence_samilarity, shuffle_token
from breaching.cases.models.language_models import TransformerModel_disencod
from breaching.cases.models.losses import CausalLoss
from breaching.attacks.auxiliaries.regularizers import TotalVariation
from breaching.attacks.auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup, grad_euclidean
# from breaching.attacks.auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup

import os

from modeling import BertForSentimentClassification
from breaching.cases.data.sst2dataset import SSTDataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

device = torch.device(f'cuda:6') if torch.cuda.is_available() else torch.device('cpu')

try:
    import breaching
except ModuleNotFoundError:
    import os;

    os.chdir("..")
    import breaching
# writer = SummaryWriter('logs')
vocab_size = 30522  # gpt:50257
num_trials = 1
cfg = breaching.get_config(overrides=["case=10_causal_lang_training", "attack=tag"])
cfg.case.data.shape = [30]
data_shape_len = cfg.case.data.shape

setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

dm, ds = torch.tensor(0, device=device), torch.tensor(1, device=device)


# 用于恢复标签的初始化
def _initialize_data(data_shape):
    """Note that data is initialized "inside" the network normalization."""
    # init_type = cfg.init
    init_type = "randn-trunc"
    if init_type == "randn":
        candidate = torch.randn(data_shape, setup)
    elif init_type == "randn-trunc":
        candidate = (torch.randn(data_shape) * 0.1).clamp(-0.1, 0.1)
        candidate = candidate.to(device)
    elif init_type == "rand":
        candidate = (torch.rand(data_shape, setup) * 2) - 1.0
    elif init_type == "zeros":
        candidate = torch.zeros(data_shape, setup)
    # Initializations from Wei et al, "A Framework for Evaluating Gradient Leakage
    #                                  Attacks in Federated Learning"
    elif any(c in init_type for c in ["red", "green", "blue", "dark", "light"]):  # init_types like 'red-true'
        candidate = torch.zeros(data_shape, setup)
        if "light" in init_type:
            candidate = torch.ones(data_shape, setup)
        else:
            nonzero_channel = 0 if "red" in init_type else 1 if "green" in init_type else 2
            candidate[:, nonzero_channel, :, :] = 1
        if "-true" in init_type:
            # Shift to be truly RGB, not just normalized RGB
            candidate = (candidate - dm) / ds

    candidate.to(memory_format=torch.contiguous_format)
    candidate.requires_grad = True
    candidate.grad = torch.zeros_like(candidate)
    return candidate


def _init_optimizer(candidate):
    optimizer, scheduler = optimizer_lookup(
        candidate,
        cfg.attack.optim.optimizer,
        cfg.attack.optim.step_size,
        scheduler=cfg.attack.optim.step_size_decay,
        warmup=cfg.attack.optim.warmup,
        max_iterations=cfg.attack.optim.max_iterations,
    )
    return optimizer, scheduler


def _run_trial(model, shared_data, label_template, stats, trial, loss_fn, dryrun=False):
    """Run a single reconstruction trial."""
    objective_fn = objective_lookup.get("tag-euclidean")  # get(cfg.objective.type)
    objective = objective_fn(cfg.attack.objective)  # objective_fn(cfg.objective)
    objective.initialize(loss_fn, cfg.case.impl, shared_data[0]["metadata"]["local_hyperparams"])

    # Initialize candidate reconstruction data
    data_shape, token_embedding_dim = data_shape_len[0], 96
    data_shape = [data_shape, token_embedding_dim]
    candidate_data = _initialize_data([shared_data[0]["metadata"]["num_data_points"], *data_shape])
    candidate_labels = _initialize_data(label_template.shape)

    best_candidate = candidate_data.detach().clone()
    minimal_value_so_far = torch.as_tensor(float("inf")).to(device)

    # Initialize optimizers
    optimizer, scheduler = _init_optimizer([candidate_data, candidate_labels])
    current_wallclock = time.time()

    for iteration in range(cfg.attack.optim.max_iterations):

        best_candidate = candidate_data.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf")).to(device)

        def closure():
            optimizer.zero_grad()
            augmentations = torch.nn.Sequential()
            candidate_augmented = candidate_data
            candidate_augmented.data = augmentations(candidate_data.data)

            real_gradient = shared_data[0]["gradients"]

            model.zero_grad()
            outs = model(inputs_embeds=candidate_data,
                         unk_mask=torch.zeros(1, candidate_data.shape[1], dtype=torch.long).to(device))

            task_loss = loss_fn(outs, candidate_labels.softmax(dim=-1))  # task_loss是假梯度的candidate和lanbel之间的loss
            dummy_gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=True)
            total_objective = grad_euclidean(dummy_gradient, real_gradient)  # Obeject:真假梯度之间的欧式距离

            if total_objective.requires_grad:
                total_objective.backward(inputs=[candidate_data, candidate_labels])

            # 梯度裁剪
            with torch.no_grad():
                if cfg.attack.optim.grad_clip is not None:
                    for element in [candidate_data, candidate_labels]:
                        grad_norm = element.grad.norm()
                        if grad_norm > cfg.attack.optim.grad_clip:
                            element.grad.mul_(cfg.attack.optim.grad_clip / (grad_norm + 1e-6))

            current_task_loss = task_loss  # Side-effect this because of L-BFGS closure limitations :<

            return total_objective, current_task_loss

        objective_value, task_loss = optimizer.step(closure)
        scheduler.step()

        with torch.no_grad():
            # Project into image space
            if cfg.attack.optim.boxed:
                candidate_data.data = torch.max(
                    torch.min(candidate_data, (1 - dm) / ds), -dm / ds
                )

            if objective_value < minimal_value_so_far:
                minimal_value_so_far = objective_value.detach()
                best_candidate = candidate_data.detach().clone()
                best_labels = candidate_labels.detach().clone()

        if iteration + 1 == cfg.attack.optim.max_iterations or iteration % cfg.attack.optim.callback == 0:
            timestamp = time.time()
            p = candidate_labels.softmax(dim=-1)
            label_entropy = torch.where(p > 0, -p * torch.log(p), torch.zeros_like(p), ).sum(
                dim=-1
            ).mean() / torch.log(torch.as_tensor(p.shape[-1], dtype=torch.float))

            print(
                f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s | "
                f" Label Entropy: {label_entropy:2.4f}."
            )
            # writer.add_scalar('Rec. loss', objective_value.item(), iteration + 1)
            # writer.add_scalar('Task loss', task_loss.item(), iteration + 1)

            current_wallclock = timestamp

        stats[f"Trial_{trial}_Val"].append(objective_value.item())

        if dryrun:
            break
    return best_candidate.detach(), best_labels.detach()


def _score_trial(candidate, labels, rec_model, shared_data, loss_fn):
    """Score candidate solutions based on some criterion."""

    if cfg.attack.restarts.scoring in ["euclidean", "cosine-similarity"]:
        objective = Euclidean() if cfg.attack.restarts.scoring == "euclidean" else CosineSimilarity()
        objective.initialize(loss_fn, cfg.case.impl, shared_data[0]["metadata"]["local_hyperparams"])
        score = 0
        for model, data in zip(rec_model, shared_data):
            score += objective(model, data["gradients"], candidate, labels)[0]
    elif cfg.attack.restarts.scoring in ["TV", "total-variation"]:
        score = TotalVariation(scale=1.0)(candidate)
    else:
        raise ValueError(f"Scoring mechanism {cfg.scoring} not implemented.")
    return score if score.isfinite() else float("inf")


def _select_optimal_reconstruction(candidate_solutions, candidate_labels, scores, stats):
    """Choose one of the candidate solutions based on their scores (for now).

    More complicated combinations are possible in the future."""
    optimal_val, optimal_index = torch.min(scores, dim=0)
    optimal_solution = candidate_solutions[optimal_index]
    optimal_labels = candidate_labels[optimal_index]
    stats["opt_value"] = optimal_val.item()
    if optimal_val.isfinite():
        print(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
        return optimal_solution, optimal_labels
    else:
        print("No valid reconstruction could be found.")
        return torch.zeros_like(optimal_solution), torch.zeros_like(optimal_labels)


def _postprocess_text_data(reconstruct_user_data, true_embed_weight, models=None):
    """Post-process text data to recover tokens."""

    def _max_similarity(recovered_embeddings, true_embeddings):
        recovered_embeddings = recovered_embeddings - recovered_embeddings.mean(dim=-1, keepdim=True)
        true_embeddings = true_embeddings - true_embeddings.mean(dim=-1, keepdim=True)
        norm_rec = recovered_embeddings.pow(2).sum(dim=-1)
        norm_true = true_embeddings.pow(2).sum(dim=-1)
        cosim = recovered_embeddings.matmul(true_embeddings.T) / norm_rec[:, None] / norm_true[None, :]
        return cosim.argmax(dim=1)

    # Use extracted embeddings:
    # embedding_weight = embeddings[0]["weight"]
    embedding_weight = true_embed_weight

    if cfg.attack.token_recovery == "from-embedding":  # 这里
        # This is the DLG strategy. Look up all inputs in embedding space.
        recovered_embeddings = reconstruct_user_data["data"]  # (1, 16, 96)
        base_shape = recovered_embeddings.shape[0:2]  # (1,16)
        recovered_embeddings = recovered_embeddings.view(-1, recovered_embeddings.shape[-1])  # (16,96)
        true_embeddings = embedding_weight

        recovered_tokens = _max_similarity(recovered_embeddings, true_embeddings).view(*base_shape)

    elif cfg.attack.token_recovery == "from-labels":
        # Only works well in some causal-lm?
        recovered_tokens = reconstruct_user_data["labels"]
    elif cfg.attack.token_recovery == "from-limited-embedding":
        # Retrieve possible embeddings from gradient data
        recovered_embeddings = reconstruct_user_data["data"]
        base_shape = recovered_embeddings.shape[0:2]
        recovered_embeddings = recovered_embeddings.view(-1, recovered_embeddings.shape[-1])
        active_embedding_ids = reconstruct_user_data["labels"].unique()
        true_embeddings = embedding_weight[active_embedding_ids, :]
        matches = _max_similarity(recovered_embeddings, true_embeddings)
        recovered_tokens = active_embedding_ids[matches].view(*base_shape)

    reconstruct_user_data["data"] = recovered_tokens
    return reconstruct_user_data


def reconstruct(shared_data, model, loss_fn, true_embed_weight, dryrun=False):
    # # Initialize stats module for later usage:   # rec_models, labels, stats = prepare_attack(shared_data)
    stats = defaultdict(list)
    shared_data = shared_data.copy()  # Shallow copy is enough

    for data in shared_data:
        data["gradients"] = [g.to(dtype=torch.float32) for g in data["gradients"]]  # data["gradients"]转为list

    num_data_points = 1
    labels = _initialize_data(
        [num_data_points, data_shape_len[0], 30522])  # 1*len*30522 segmentation type in_shape->out_shape tasks

    # Main reconstruction loop starts here:
    scores = torch.zeros(num_trials)
    candidate_solutions, candidate_labels = [], []
    for trial in range(num_trials):
        data, label = _run_trial(model, shared_data, labels, stats, trial, loss_fn, dryrun)
        candidate_solutions += [data]
        candidate_labels += [labels.argmax(dim=-1)]
        scores[trial] = _score_trial(candidate_solutions[trial], candidate_labels[trial], [model], shared_data, loss_fn)
    optimal_solution, optimal_labels = _select_optimal_reconstruction(
        candidate_solutions, candidate_labels, scores, stats
    )
    reconstructed_data = dict(data=optimal_solution, labels=optimal_labels)
    reconstructed_data = _postprocess_text_data(reconstructed_data, true_embed_weight)
    reconstructed_data["raw_embeddings"] = optimal_solution

    return reconstructed_data, stats


def print_data(user_data, tokenizer):
    """Print decoded user data to output."""
    decoded_tokens = tokenizer.batch_decode(user_data, clean_up_tokenization_spaces=True)
    for line in decoded_tokens:
        print(line)


# writer.close()


# configure attack
def build_attack_config():
    import argparse

    parser = argparse.ArgumentParser(description='start reconstruct data!!!')
    parser.add_argument('-train_type', action="store", type=str, default="mjp_unk_drloc")
    parser.add_argument('-use_unk', action="store", type=bool, default=True, help='use unk mask')
    parser.add_argument('-use_drloc', action="store", type=bool, default=True, help='use unk mask')
    parser.add_argument('-bestmodel_shuffle_ratio', action="store", type=float, default=0.2)
    parser.add_argument('-percent_to_shuffle', action="store", type=float, default=1.0)
    parser.add_argument('-model_type', action="store", type=str, default="bert", help='use model construct')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('-minlen', action="store", type=int, default=10, help='min length of sentence')
    parser.add_argument('-maxlen', action="store", type=int, default=30, help='max length of sentence')
    parser.add_argument('-window_size', action="store", type=int, default=16, help='window_size')
    parser.add_argument('-vocab_size', action="store", type=int, default=30522, help='vocab_size')
    parser.add_argument('-max_position_embeddings', action="store", type=int, default=512, help='max_position_embeddings')
    parser.add_argument('-type_vocab_size', action="store", type=int, default=2, help='type_vocab_size')
    parser.add_argument('-seeds', action="store", type=int, default=3047, help='seeds')
    args = parser.parse_args()
    [print(k, ": ", v) for (k, v) in vars(args).items()]
    return args
