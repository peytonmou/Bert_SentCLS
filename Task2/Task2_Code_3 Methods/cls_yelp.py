import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm 

from datasets import load_dataset
from sklearn.model_selection import train_test_split 
import pandas as pd 

TQDM_DISABLE=True
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # classification head
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        
        # initialize weights for classifier
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_() 

    def forward(self, input_ids, attention_mask):   
        # the final bert contextualize embedding is the hidden state of [CLS] token (the first token)
        # get Bert output 
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # extract [CLS] token's hidden state (the first token)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        
        # apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return F.log_softmax(logits, dim=-1)

# create a custom Dataset Class to be used for the dataloader
class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
            })

        return batches


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename=None, flag='train', hf_dataset=None, dataset_type='sst', sample_percentage=1.0):
    """
    Creates data from HuggingFace dataset or local file, with an option to sample.

    Args:
        filename (str, optional): Path to local data file. Defaults to None.
        flag (str, optional): 'train', 'valid', or 'test'. Defaults to 'train'.
        hf_dataset (datasets.Dataset, optional): Hugging Face dataset split. Defaults to None.
        dataset_type (str, optional): Type of dataset ('yelp', 'sst2', 'sst5', 'imdb'). Defaults to 'sst'.
        sample_percentage (float, optional): Percentage of data to use (e.g., 0.5 for half).
                                             Only applies to HF datasets if flag is 'train'.
                                             Defaults to 1.0 (use all data).
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = {}
    data = []

    if hf_dataset is not None:
        # Apply sampling ONLY for the 'train' flag and if sample_percentage < 1.0
        if flag == 'train' and sample_percentage < 1.0:
            original_size = len(hf_dataset)
            sample_size = int(original_size * sample_percentage)
            print(f"Sampling {sample_percentage*100:.1f}% ({sample_size}/{original_size}) of {dataset_type} {flag} dataset.")
            # Randomly sample indices
            indices = list(range(original_size))
            random.shuffle(indices)
            sampled_indices = sorted(indices[:sample_size])
            hf_dataset = hf_dataset.select(sampled_indices)         # Use .select() for efficient sampling

        # Handle HuggingFace datasets (Yelp or SST-2)
        for example in hf_dataset:
            sent = ""
            label = -1

            if dataset_type == 'yelp':
                # Yelp format: {'label': 0-4, 'text': '...'}
                sent = example['text'].lower().strip()
                label = example['label'] + 1  # Convert 0-4 → 1-5 
            else: 
                # SST-2 format: {'sentence': '...', 'label': 0-1}
                sent = example['sentence'].lower().strip()
                label = example['label']
                # Fallback if dataset_type is unknown, assuming 'sentence' and 'label' or 'text' and 'label'
                if 'sentence' in example and 'label' in example:
                    sent = example['sentence'].lower().strip()
                    label = example['label']
                elif 'text' in example and 'label' in example:
                    sent = example['text'].lower().strip()
                    label = example['label']


            tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            data.append((sent, num_labels[label], tokens))    # Use the mapped label

    else:
        # Handle original file format (SST/IMDB style)
        with open(filename, 'r') as fp:
            for line in fp:
                try:
                    label_str, org_sent = line.split('|||')
                    sent = org_sent.lower().strip()
                    tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
                    label = int(label_str.strip())
                    if label not in num_labels:
                        num_labels[label] = len(num_labels)
                    data.append((sent, num_labels[label], tokens))
                except ValueError:
                    continue

    print(f"Loaded {len(data)} examples from {'HF dataset' if hf_dataset else filename}")
    if flag == 'train':
        return data, len(num_labels) 
    return data 


# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, device):
    model.eval() 
    y_true = []
    y_pred = []
    sents = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], \
                                                       batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def convert_example(example):
    return (example['sentence'], example['label'], 
            tokenizer.tokenize("[CLS] " + example['sentence'] + " [SEP]"))
    
def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    #Load Yelp data from HuggingFace
    yelp_dataset = load_dataset("Yelp/yelp_review_full")   
    train_data, num_labels = create_data(args.train, flag='train', hf_dataset=yelp_dataset['train'], dataset_type='yelp', sample_percentage=0.1) 
    dev_data = create_data(args.dev, flag='valid', hf_dataset=yelp_dataset['test'], dataset_type='yelp', sample_percentage=0.1)    

    # Load SST-2 from HuggingFace 
    # sst2_dataset = load_dataset('glue', 'sst2')
    # train_data, num_labels = create_data(args.train, flag='train', hf_dataset=sst2_dataset['train'])
    # dev_data = create_data(args.dev, flag='valid', hf_dataset=sst2_dataset['validation']) 
    
    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    #### Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    # initialize the Senetence Classification Model
    model = BertSentClassifier(config)
    model = model.to(device)

    lr = args.lr
    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_type_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False)   # set weights_only = False because pytorch changed the default value to True 
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid', dataset_type='sst', sample_percentage=1.0)  
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test', dataset_type='sst') 
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/yelp-train.txt")
    parser.add_argument("--dev", type=str, default="data/yelp-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="sst2-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.25)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
