import argparse
import torch
from PackDataset import packDataset_util_bert
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, LlamaForSequenceClassification, LlamaTokenizer
import transformers
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
import os
from torch.nn.utils import clip_grad_norm_
import csv
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text,attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def small_eval(loader):
    bias_model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = bias_model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def poe_with_r_drop_loss(output, output_2, out_3, labels):
    """Implements the combination of poe loss & r-drop loss."""
    pt = F.softmax(output, dim=1)
    pt_3 = F.softmax(out_3, dim=1)
    pt_2 = F.softmax(output_2/args.temperature, dim=1)
    joint_pt = F.softmax((0.5 * (torch.log(pt) + torch.log(pt_3)) + args.poe_alpha * torch.log(pt_2)), dim=1)
    joint_p = joint_pt.gather(1, labels.view(-1, 1))

    p_loss = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(out_3, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(out_3, dim=-1), F.softmax(output, dim=-1), reduction='none')

    batch_loss = -torch.log(joint_p) + args.rdrop_alpha * (p_loss + q_loss) / 2
    loss = batch_loss.mean()
    return loss


def poe_loss(output, output_2, labels):
    """Implements the product of expert loss."""
    pt = F.softmax(output, dim=1)
    pt_2 = F.softmax(output_2 / args.temperature, dim=1)
    joint_pt = F.softmax((torch.log(pt) + args.poe_alpha * torch.log(pt_2)), dim=1)
    joint_p = joint_pt.gather(1, labels.view(-1, 1))
    batch_loss = -torch.log(joint_p)
    bias_p = F.softmax(output_2, dim=1)
    bias_p = bias_p.gather(1, labels.view(-1, 1))
    bias_loss = -torch.log(bias_p)
    if args.do_reweight:
        logits_1 = F.softmax(output, dim=1)
        logits_1 = logits_1.gather(1, labels.view(-1, 1))
        logits_2 = F.softmax(output_2, dim=1)
        logits_2 = logits_2.gather(1, labels.view(-1, 1))
        weight_main = torch.where(logits_2 > args.reweight_threshold, 1.0 - logits_2, 1.0)
        weight_bias = torch.where(logits_1 < 0.5, logits_1, 1.0)
        # batch_loss = batch_loss * weight_main + bias_loss * weight_bias
        batch_loss = batch_loss * weight_main
    else:
        # batch_loss = batch_loss + bias_loss
        batch_loss = batch_loss
    loss = batch_loss.mean()
    return loss


def kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def train():
    last_train_avg_loss = 1e10
    try:
        print('start training main model')
        write_results(['start training PoE'])
        iter = 0
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in tqdm(train_loader_poison):
                iter += 1
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()

                output = model(padded_text, attention_masks)[0]
                output_2 = bias_model(padded_text, attention_masks)[0]
                output_3 = model(padded_text, attention_masks)[0]

                loss_1 = 0.5 * (poe_loss(output, output_2, labels) + poe_loss(output_3, output_2, labels))
                loss_2 = kl_loss(output, output_3)
                if args.rdrop_mode_1:
                    loss = loss_1 + args.rdrop_alpha * loss_2
                if args.rdrop_mode_2:
                    loss = poe_loss(output, output_2, labels) + args.rdrop_alpha * loss_2

                optimizer.zero_grad()
                optimizer_bias.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                clip_grad_norm_(bias_model.parameters(), max_norm=1)
                optimizer.step()
                optimizer_bias.step()
                scheduler.step()
                scheduler_bias.step()
                total_loss += loss.item()

        final_poison_success_rate_test = evaluaion(test_loader_poison)
        final_clean_acc_test = evaluaion(test_loader_clean)
        final_poison_success_rate_dev = evaluaion(dev_loader_poison)
        final_clean_acc_dev = evaluaion(dev_loader_clean)
        write_results(['*** final result ***', final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test])

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def write_results(result):
    with open(result_file, 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(result)


def write_results_small(result):
    with open(result_file_small, 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(result)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def initialize_bert_model(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return model


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--batch_size', type=int, default=32)
    # args for optimizer
    parser.add_argument('--lr', type=float, default=2e-5)  # learning rate for main model
    parser.add_argument('--small_lr', type=float, default=5e-4)  # learning rate for trigger-only model, larger than lr
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--bias_correction', default=True)
    # args for training
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--do_reweight", type=bool, default=False)
    parser.add_argument("--reweight_threshold", type=float, default=0.8)
    parser.add_argument("--do_reinit", type=bool, default=False)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    # args for poison
    parser.add_argument('--poison_rate', type=int, default=20)
    parser.add_argument('--clean_data_path', )
    parser.add_argument('--poison_data_path',)
    parser.add_argument('--save_path', default='')
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--num_bias_layers', type=int, default=3)
    parser.add_argument('--do_PoE', type=bool, default=True, help="If selected, train model with PoE")
    parser.add_argument('--poe_alpha', type=float, default=1.0)
    parser.add_argument('--do_Rdrop', type=bool, default=True)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--rdrop_alpha', type=float, default=1.0)
    parser.add_argument('--rdrop_mode_1', type=bool, default=False)
    parser.add_argument('--rdrop_mode_2', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--result_path', default='DPoE/results')
    parser.add_argument('--ensembel_layer_num', type=int, default=0)

    args = parser.parse_args()
    data_selected = args.data
    BATCH_SIZE = args.batch_size
    weight_decay = args.weight_decay
    lr = args.lr
    EPOCHS = args.epoch

    os.makedirs(os.path.join(args.result_path), exist_ok=True)
    if args.do_reweight:
        result_file = os.path.join(args.result_path,
                                   'epoch_{}_layer_{}_small_lr_{}_poe_alpha_{}_rdrop_{}_temp_{}_reweight_{}.csv'.format(
                                       args.epoch, args.num_hidden_layers, args.small_lr, args.poe_alpha,
                                       args.rdrop_alpha, args.temperature, args.reweight_threshold))
        result_file_small = os.path.join(args.result_path,
                                         'epoch_{}_layer_{}_small_lr_{}_poe_alpha_{}_rdrop_{}_temp_{}_reweight_{}_small_model.csv'.format(
                                             args.epoch, args.num_hidden_layers, args.small_lr, args.poe_alpha,
                                             args.rdrop_alpha, args.temperature, args.reweight_threshold))
    else:
        result_file = os.path.join(args.result_path,
                                   'epoch_{}_layer_{}_small_lr_{}_poe_alpha_{}_rdrop_{}_temp_{}.csv'.format(
                                       args.epoch, args.num_hidden_layers, args.small_lr, args.poe_alpha,
                                       args.rdrop_alpha, args.temperature))
        result_file_small = os.path.join(args.result_path,
                                         'epoch_{}_layer_{}_small_lr_{}_poe_alpha_{}_rdrop_{}_temp_{}_small_model.csv'.format(
                                             args.epoch, args.num_hidden_layers, args.small_lr, args.poe_alpha,
                                             args.rdrop_alpha, args.temperature))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)
    # load data
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
    poison_train_data, poison_dev_data, poison_test_data = get_all_data(args.poison_data_path)
    packDataset_util = packDataset_util_bert()
    train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)
    train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

    # load model
    config = AutoConfig.from_pretrained(args.model_name, num_labels=4 if data_selected == 'ag' else 2)
    config.ensemble_layer_num = args.ensembel_layer_num
    if "llama" in args.model_name:
        model = LlamaForSequenceClassification.from_pretrained(args.model_name, num_labels=4 if data_selected == 'ag' else 2)
        bias_model = LlamaForSequenceClassification.from_pretrained(args.model_name, num_labels=4 if data_selected == 'ag' else 2, num_hidden_layers=args.num_hidden_layers)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=4 if data_selected == 'ag' else 2)
        bias_model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=4 if data_selected == 'ag' else 2, num_hidden_layers=args.num_hidden_layers)

    if args.do_reinit:
        bias_model = initialize_bert_model(bias_model)

    model.cuda()
    bias_model.cuda()

    criterion = nn.CrossEntropyLoss()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    optimizer_grouped_parameters_bias = [
        {
            "params": [p for n, p in bias_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in bias_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer_bias = AdamW(
        optimizer_grouped_parameters_bias,
        lr=args.small_lr,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(poison_train_data) * args.epoch // args.batch_size
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    scheduler_bias = get_linear_schedule_with_warmup(optimizer_bias, warmup_steps, num_training_steps)

    train()

