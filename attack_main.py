import copy
import math
from collections import defaultdict

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
import torch.nn.functional as F
from torch.utils import data

from breaching.attacks.auxiliaries.common import optimizer_lookup

import torch
import os

from breaching.cases.data.datasets_text import meric_two_sentence_samilarity
from breaching.cases.models.language_models import TransformerModel_disencod
from breaching.cases.models.losses import CausalLoss
from breaching.cases.data.sst2dataset import SSTDataset
from TernaryBERT.modeling import BertForSequenceClassification_ngram
from TernaryBERT.build_model import build_model_ngram
from TernaryBERT.transformer import TransformerSentiment
from sentiment_rpe.shuffle_process import shuffle_token
from attack_tool import print_data, reconstruct, build_attack_config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

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

device = torch.device(f'cuda:6') if torch.cuda.is_available() else torch.device('cpu')
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

tokenizer = AutoTokenizer.from_pretrained("TernaryBERT/bert-base-uncased")  # bert-base-uncased gpt2

args = build_attack_config()
model_type_dir = args.model_type
train_type_dir = args.train_type.replace("_", "")  # 去除下划线
base_path = '/opt/data/private/ywx/transformerattack/TernaryBERT/savemodel/yelp/subsetngram/'
save_res_dir = os.path.join(base_path, "bert_seed3047", train_type_dir)

if args.train_type == "shuffle":
    use_drloc,use_unk = False,False
    model_path = os.path.join(base_path, "bert_seed3047", f'shuffle/best_network_{args.seeds}_{args.bestmodel_shuffle_ratio}_180_400.pth')
elif args.train_type == "mjp_unk":
    use_drloc,use_unk = False,True
    model_path = os.path.join(base_path, "bert_seed3047", f'mjpunk/best_network_{args.seeds}_{args.bestmodel_shuffle_ratio}_180_400.pth')
elif args.train_type == "mjp_unk_drloc":
    use_drloc,use_unk = True,True
    model_path = os.path.join(base_path, "bert_seed3047", f'mjpunkdrloc/best_network_{args.seeds}_{args.bestmodel_shuffle_ratio}_180_400.pth')


model_ori = TransformerModel_disencod(
    ntokens=vocab_size,  # 词表的大小
    ninp=96,  # 输入词向量的大小
    nhead=8,  # 8头attention
    nhid=1536,  # 隐藏层个数
    nlayers=3,  # transformer的encoder是3层
    dropout=0,
    ape=True,
    use_unk=True,
)
model = TransformerModel_disencod(
    ntokens=vocab_size,  # 词表的大小
    ninp=96,  # 输入词向量的大小
    nhead=8,  # 8头attention
    nhid=1536,  # 隐藏层个数
    nlayers=3,  # transformer的encoder是3层
    dropout=0,
    ape=True,
    use_unk=True,
)

if args.model_type == "transformer":
    stm_model = TransformerSentiment(
        vocab_size=30522,
        num_hid=1536,
        emb_dim=96,
        num_heads=2,
        num_layers=2,
        num_classes=2,
        positional_embedding="learnable",
        train_type=args.train_type,
        shuffle_ratio=args.percent_to_shuffle,
        use_drloc=use_drloc,
        use_unk=use_unk
    )
elif args.model_type == "bert":
    # stm_model = BertForSequenceClassification_ngram(args)
    stm_model = model = build_model_ngram(device, args.train_type, use_unk, use_drloc, args.window_size)
checkpoint = torch.load(model_path)
stm_model.load_state_dict(checkpoint['model'])
model_ori.to(device)
model.to(device)
stm_model.to(device)
stm_model.eval()
model.train()

# train_set = SSTDataset(filename="data/SST-2/dev.tsv", minlen=10, maxlen=data_shape_len[0], tokenizer=tokenizer)
# train_loader = DataLoader(dataset=train_set, batch_size=1, num_workers=0, shuffle=True)
# train_df = pd.read_csv("/opt/data/private/ywx/transformerattack/data/SST-2/downdata/train.tsv", sep="\t")
# train_df = train_df.reset_index(drop=True)  # reset index

data_dir = '/opt/data/private/ywx/transformerattack/data/SST-2/val_pro.csv'
train_set = SSTDataset(data_dir, args.minlen, args.maxlen, tokenizer)
train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
loss_fn = CausalLoss()  # token prediction loss (CrossEntropyLoss())
num_correct, total_predictions = 0, 0
recover_rate_score, rouge1_score, rouge2_score, rougeL_score, bleu_score, bertscore_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
recover_rate_score_mask, rouge1_score_mask, rouge2_score_mask, rougeL_score_mask, bleu_score_mask, bertscore_f1_score_mask = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
dataset_len = len(train_set)
print("dataset_len: ", dataset_len)

with open('restoreorin.txt', 'w') as file_orig, open('restoremask.txt', 'w') as file_shuffle:
    stm_correct_mask = 0
    stm_correct, stm_total = 0, 0
    for i, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        clslabel = data["claslabel"].to(device)
        labels = data["labels"].to(device)

        print("[true data]: ")
        print_data(input_ids, tokenizer)
        shuffle_id, mask_matrix = shuffle_token(copy.deepcopy(input_ids), percent_to_shuffle=args.percent_to_shuffle)  # unk:1, pos:0
        shuffle_id = shuffle_id.to(device)

        print("[shuffle data]:")
        print_data(shuffle_id, tokenizer)

        # classification eval for origin
        # stm_logits, _, _ = stm_model(input_ids, (torch.zeros_like(input_ids)).to(device))  # 普通数据不打乱
        # stm_correct += torch.sum(torch.argmax(stm_logits, dim=1) == clslabel)
        # stm_total += clslabel.size(0)
        # # classification eval for mask
        # stm_logits_mask, _, _ = stm_model(shuffle_id, mask_matrix.to(device))
        # stm_correct_mask += torch.sum(torch.argmax(stm_logits_mask, dim=1) == clslabel)

        encode_embed = nn.Embedding(30522, 96).to(device)  # input is Word Embedding
        # initrange = 0.1
        encode_embed.weight.data *= math.sqrt(96)  # 每个元素乘以 96,缩放权重
        # nn.init.uniform_(encode_embed.wepight, -initrange, initrange)
        true_embed = encode_embed(shuffle_id)
        true_embed_weight = encode_embed.weight
        outs = model_ori(inputs_embeds=true_embed, unk_mask=mask_matrix.to(device))

        loss = loss_fn(outs, labels.to(device).clone())

        # get shuffle shared_data
        shared_grads = torch.autograd.grad(loss, model_ori.parameters())
        shared_grads = [grad.detach().clone() for grad in shared_grads]
        metadata = dict(num_data_points=1, labels=None, local_hyperparams=None)
        shared_data = dict(gradients=shared_grads, buffers=None, metadata=metadata)

        reconstructed_user_data, stats = reconstruct([shared_data], model, loss_fn, true_embed_weight, dryrun=cfg.dryrun)

        print("[construct data]:")
        print_data(reconstructed_user_data["data"], tokenizer)

        print("[-----original sequence]")
        recover_rate, rouge1, rouge2, rougeL, bleu, bertscore_f1 = meric_two_sentence_samilarity(
        reconstructed_user_data["data"], input_ids, tokenizer)

        print("[-----mask sequence]")
        recover_rate_mask, rouge1_mask, rouge2_mask, rougeL_mask, bleu_mask, bertscore_f1_mask = meric_two_sentence_samilarity(
        reconstructed_user_data["data"], shuffle_id, tokenizer)

        recover_rate_score += recover_rate
        rouge1_score += rouge1
        rouge2_score += rouge2
        rougeL_score += rougeL
        bleu_score += bleu
        bertscore_f1_score += bertscore_f1

        recover_rate_score_mask += recover_rate_mask
        rouge1_score_mask += rouge1_mask
        rouge2_score_mask += rouge2_mask
        rougeL_score_mask += rougeL_mask
        bleu_score_mask += bleu_mask
        bertscore_f1_score_mask += bertscore_f1_mask

    # 将循环外的结果写入文件
    file_orig.write("\n compare with original sequence (Outside Loop)\n")
    file_orig.write(f"ave recover rate: {(recover_rate_score / dataset_len):.4f}\n")
    file_orig.write(f"ave rouge1_score: {rouge1_score / dataset_len :.4f}\n")
    file_orig.write(f"ave rouge2_score: {rouge2_score / dataset_len :.4f}\n")
    file_orig.write(f"ave rougeL_score: {rougeL_score / dataset_len :.4f}\n")
    file_orig.write(f"ave bleu_score: {bleu_score / dataset_len :.4f}\n")
    file_orig.write(f"ave bertscore_f1_score: {bertscore_f1_score / dataset_len :.4f}\n")
    file_orig.write(f"ave stm_correct: {(stm_correct / stm_total) :.4f}\n")

    file_shuffle.write("\n compare with shuffle sequence (Outside Loop)\n")
    file_shuffle.write(f"mask recover rate_mask: {(recover_rate_score_mask / dataset_len):.4f}\n")
    file_shuffle.write(f"mask ave rouge1_mask: {rouge1_score_mask / dataset_len :.4f}\n")
    file_shuffle.write(f"mask ave rouge2_score: {rouge2_score_mask / dataset_len :.4f}\n")
    file_shuffle.write(f"mask ave rougeL_score: {rougeL_score_mask / dataset_len :.4f}\n")
    file_shuffle.write(f"bleu_score_mask: {bleu_score_mask / dataset_len :.4f}\n")
    file_shuffle.write(f"bertscore_f1_score_mask: {bertscore_f1_score_mask / dataset_len :.4f}\n")
    file_shuffle.write(f"stm_correct_mask: {(stm_correct_mask / stm_total) :.4f}\n")

    print("------------------recover vs. original sequence")
    # print(f"recover rate: {(recover_rate_score / dataset_len):.4f}")
    print(f"rouge1_score: {rouge1_score / dataset_len :.4f}")
    print(f"rouge2_score: {rouge2_score / dataset_len :.4f}")
    print(f"rougeL_score: {rougeL_score / dataset_len :.4f}")
    print(f"bleu_score: {bleu_score / dataset_len :.4f}")
    print(f"bertscore_f1_score: {bertscore_f1_score / dataset_len :.4f}")
    print(f"origin sentiment cls acc: {(stm_correct / stm_total):.4f}")

    print("\n")

    print("------------------recover vs. mask sequence")
    # print(f"recover rate_mask: {(recover_rate_score_mask / dataset_len):.4f}")
    print(f"rouge1_mask: {rouge1_score_mask / dataset_len :.4f}")
    print(f"rouge2_mask: {rouge2_score_mask / dataset_len :.4f}")
    print(f"rougeL_mask: {rougeL_score_mask / dataset_len :.4f}")
    print(f"bleu_score_mask: {bleu_score_mask / dataset_len :.4f}")
    print(f"bertscore_f1_score_mask: {bertscore_f1_score_mask / dataset_len :.4f}")
    print(f"mask sentiment cls acc: {(stm_correct_mask / stm_total):.4f}")

    print("\n\n")
    print("stm_total", stm_total)
    print("dataset_len", dataset_len)

# writer.close()
