import argparse
import os

import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model2 import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred, read_chemdisgene, evaluate_bio, export_test_bio
from evaluation import to_official, official_evaluate
from tqdm import tqdm
from time import time
import wandb

def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True,num_workers=0)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            print('Epoch{} has started!'.format(epoch) )
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                wandb.log({"loss": loss.item()}, step=num_steps)
                if step % args.gradient_accumulation_steps == 0:
                    # if args.max_grad_norm > 0:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output,_ = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output,_ = evaluate(args, model, test_features, tag="test")
                    
                    wandb.log(dev_output, step=num_steps)
                    wandb.log(test_output, step=num_steps)
                    print("step", num_steps, dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        # pred = report(args, model, test_features)
                        # with open("result.json", "w") as user:
                            # json.dump(pred, user)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)


        return num_steps

    new_layer = ["extractor", "bilinear"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 5e-5}, # 1e-4 5e-5 
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev", use_ILP = False):
    if args.dataset == 'chemdisgene' or args.dataset == 'biored' :
        return evaluate_bio(args, model, features, tag)
    
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False, num_workers=0)
    preds = []
    losses, losses_cls, losses_srr = [], [] , []

    begin_time = time()
    for batch in dataloader:
        model.eval()


        inputs = {'input_ids'     : batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos'    : batch[3],
                  'hts'           : batch[4]
                  }

        with torch.no_grad():
            loss, loss_dict, pred = model(**inputs)
            losses.append(loss.item() )
            losses_cls.append(loss_dict['loss_cls'])
            losses_srr.append(loss_dict['loss_cls'])

            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    end_time = time()
    print('the time used by inference:', end_time - begin_time)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features,dataset=args.dataset)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, tag, dataset=args.dataset)
        # if tag == 'test':
        #     exec(args.data_dir, tag, dataset=dataset, tmp=ans)
    else:
        best_f1 = best_f1_ign = 0.0
    output = {
        tag + "_F1"    : best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_loss"  : np.mean(losses),
        tag + "_loss_cls": np.mean(losses_cls),
        tag + "_loss_srr": np.mean(losses_srr)
    }
    print(tag, output)
    return best_f1, output, ans


def report(args, model, features,use_ILP = False):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False, num_workers=0)
    preds = []
    for batch in dataloader:
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4]
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)


    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features, dataset=args.dataset)
    return preds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="REGT", type=str)
    parser.add_argument("--dataset", default="docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="../PLM/bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="../trained_model/model.pth", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=4, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--Type_Enhance", action='store_true')
    parser.add_argument("--tau", default=0.2, type=float)
    parser.add_argument("--display_name", default="REGT", type=str)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--loss", default="AT", type=str)

    args = parser.parse_args()
    args.data_dir = "../dataset/dataset_{}/".format(args.dataset)
    args.load_ner_path = "../dataset/dataset_{}/meta/ner2id.json".format(args.dataset)
    args.load_rel2id_path = "../dataset/dataset_{}/meta/rel2id.json".format(args.dataset)

    wandb.init(project="DocRED_user", name=args.display_name)

    global dataset
    dataset = args.dataset
    print(str(args))
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    # rel2id = json.load(open(args.load_rel2id_path, 'r'))
    # rel2id = sorted(rel2id.items(), key=lambda kv: (kv[1], kv[0]))
    # rel2id = {kv[0]:kv[1] for kv in rel2id}
    read_map = {
        'docred': read_docred,
        'chemdisgene': read_chemdisgene,
        'redocred': read_redocred,
        'biored': read_biored
    }
    read = read_map[args.dataset]
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    priors = None
    if args.dataset == 'chemdisgene' or args.dataset == 'redocred' or args.dataset == 'docred' or args.dataset == 'biored' :
        if args.load_path == "":
            train_features, priors = read(train_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)
            priors += 1e-9
            priors = torch.tensor(priors).to(args.device)
        dev_features,_ = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)
        test_features,_ = read(test_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)

    else:
        if args.load_path == "":
            train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)
        dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    if args.Type_Enhance:
        model.resize_token_embeddings(len(tokenizer))

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    if  args.loss == 'SPU':
        model = DocREModel(config, model, dataset=args.dataset, priors_l=priors, num_labels=args.num_labels,
                       temperature=args.tau, device=args.device)    
    else:
        model = DocREModel(config, model, dataset=args.dataset, num_labels=args.num_labels,
                       temperature=args.tau, device=args.device)
    model.to(args.device)
    print('total parameters: {}'.format(count_parameters(model)))

    if args.save_path is not None:
        # save_path
        args.save_path = args.save_path.replace('.pth', '_{}.pth'.format(time()))
        print('save_path:', args.save_path)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
        if args.save_path is not None:
            model.load_state_dict(torch.load(args.save_path))
            dev_score, dev_output, pred1 = evaluate(args, model, dev_features, tag="dev")
            test_score, test_output, pred2 = evaluate(args, model, test_features, tag="test")
            print(dev_output)
            print(test_output)
            with open("./results_for_{}/train_result_{}_dev.json".format(dataset, args.name), "w") as user:
                json.dump(pred1, user)
            # pred = report(args, model, test_features)
            with open("./results_for_{}/train_result_{}_test.json".format(dataset, args.name), "w") as user:
                json.dump(pred2, user)

    elif args.load_path != "":  # Testing
        # export_test_bio(args, test_features)
        # model.load_state_dict(torch.load(args.load_path), strict=False)

        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output, pred1 = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output, pred2 = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)
        # pred = report(args, model, dev_features)
        with open("./results_for_{}/result_{}_dev.json".format(dataset, args.name), "w") as user:
            json.dump(pred1, user)
        # pred = report(args, model, test_features)
        with open("./results_for_{}/result_{}_test.json".format(dataset, args.name), "w") as user:
            json.dump(pred2, user)
        


if __name__ == "__main__":
    main()
