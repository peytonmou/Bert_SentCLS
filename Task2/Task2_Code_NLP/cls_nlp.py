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

# Global variables for linguistic features (initialized by build_linguistic_vocabs)
POS_VOCAB = {'PAD': 0, 'UNK': 1}
DEP_VOCAB = {'PAD': 0, 'UNK': 1}
WN_VOCAB_SIZE = 20

TQDM_DISABLE=True

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
    
    def forward(self, input_ids, attention_mask, pos_tag_ids=None, dep_ids=None, wn_ids=None):    
        outputs = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            pos_tag_ids = pos_tag_ids,
            dep_ids = dep_ids,
            wn_ids = wn_ids)
        
        # extract [CLS] token's hidden state
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        
        # apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return F.log_softmax(logits, dim=-1)

class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.pos_pad_idx = POS_VOCAB['PAD']
        self.dep_pad_idx = DEP_VOCAB['PAD']
        self.pos_unk_idx = POS_VOCAB['UNK']
        self.dep_unk_idx = DEP_VOCAB['UNK'] 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def encode_linguistic_features(self, feats, max_len):
        pos_ids, dep_ids, synset_counts = [], [], []
        for feat in feats:
            # POS
            if self.args.use_pos:
                pos = feat.get('pos', 'PAD')
                pos_ids.append(POS_VOCAB.get(pos, self.pos_unk_idx)) 
            # DEP
            if self.args.use_dep:
                dep = feat.get('dep', 'PAD')
                dep_ids.append(DEP_VOCAB.get(dep, self.dep_unk_idx))
            # WordNet
            if self.args.use_wn:
                synset_count = float(feat.get('synset_count', 0)) 
                synset_counts.append(min(synset_count, WN_VOCAB_SIZE-1))

        # Padding
        pad_len = max_len - len(feats)
        if self.args.use_pos:
            pos_ids += [self.pos_pad_idx] * pad_len
        if self.args.use_dep:
            dep_ids += [self.dep_pad_idx] * pad_len
        if self.args.use_wn:
            synset_counts += [0.0] * pad_len
            
        return pos_ids, dep_ids, synset_counts

    def pad_data(self, data, args):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        features = [x[3] for x in data]
        
        # Tokenize text
        token_ids = []
        attention_masks = []
        max_len = 0
        
        for sent in sents:
            encoding = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=self.args.max_len if hasattr(self.args, 'max_len') else 512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            token_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])
            max_len = max(max_len, len(encoding['input_ids']))
        
        all_pos, all_dep, all_wn = [], [], []
        for feat in features:
            pos_ids, dep_ids, wn_ids = self.encode_linguistic_features(feat, max_len)
            if self.args.use_pos:
                all_pos.append(pos_ids[:max_len])
            if self.args.use_dep:
                all_dep.append(dep_ids[:max_len])
            if self.args.use_wn:
                all_wn.append(wn_ids[:max_len]) 

        # Construct output
        output = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'sents': sents
        }

        if self.args.use_pos:
            output['pos_tag_ids'] = torch.tensor(all_pos, dtype=torch.long)
        if self.args.use_dep:
            output['dep_ids'] = torch.tensor(all_dep, dtype=torch.long)
        if self.args.use_wn:
            output['wn_ids'] = torch.tensor(all_wn, dtype=torch.float)

        return output
       
    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens
        batches = []
        num_batches = int(np.ceil(len(all_data) / self.args.batch_size))

        for i in range(num_batches):
            batch_data = all_data[i*self.args.batch_size : (i+1)*self.args.batch_size]
            batches.append(self.pad_data(batch_data, args))

        return batches

def create_data(filename, flag='train'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = {}
    data = []

    with open(filename, 'r') as fp:
        for line in fp:
            label, org_sent = line.split(' ||| ')
            sent = org_sent.lower().strip()
            tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            label = int(label.strip())
            
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            
            ling_feats = tokenizer.extract_linguistic_features(sent)
            aligned_features = []
            token_pos = 0 
            for feat in ling_feats:
                word = feat['word']
                word_tokens = tokenizer.tokenize(word) 
                for _ in word_tokens:
                    aligned_features.append({
                        'pos': feat.get('pos', 'PAD'),
                        'dep': feat.get('dep', 'PAD'),
                        'synset_count': min(feat.get('synset_count', 0), WN_VOCAB_SIZE-1)
                    })
            if len(aligned_features) < len(tokens):
                # Pad with default values
                needed = len(tokens) - len(aligned_features)
                aligned_features.extend([{
                    'pos': 'PAD',
                    'dep': 'PAD',
                    'synset_count': 0
                }] * needed)
            elif len(aligned_features) > len(tokens):
                # Truncate if too long
                aligned_features = aligned_features[:len(tokens)]
            
            data.append((sent, label, tokens, aligned_features))
            
    print(f"Loaded {len(data)} examples from {filename}")
    return (data, len(num_labels)) if flag == 'train' else data

def build_linguistic_vocabs(data):
    global POS_VOCAB, DEP_VOCAB
    
    for _, _, _, feats in data:
        for feat in feats:
            pos = feat.get('pos', 'PAD')
            dep = feat.get('dep', 'PAD')
            
            if pos not in POS_VOCAB:
                POS_VOCAB[pos] = len(POS_VOCAB)
            if dep not in DEP_VOCAB:
                DEP_VOCAB[dep] = len(DEP_VOCAB)
    
    print(f"POS vocab size: {len(POS_VOCAB)}")
    print(f"DEP vocab size: {len(DEP_VOCAB)}")

def model_eval(dataloader, model, device):
    model.eval()
    y_true, y_pred, sents = [], [], []
    
    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        batch_data = batch[0]
        
        inputs = {
                'input_ids': batch_data['token_ids'].to(device),
                'attention_mask': batch_data['attention_mask'].to(device),}
        if args.use_pos:
            inputs['pos_tag_ids'] = batch_data['pos_tag_ids'].to(device)
        if args.use_dep:
            inputs['dep_ids'] = batch_data['dep_ids'].to(device)
        if args.use_wn:
            inputs['wn_ids'] = batch_data['wn_ids'].to(device)
        
        with torch.no_grad():
            logits = model(**inputs)
        
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(batch_data['labels'].numpy())
        y_pred.extend(preds)
        sents.extend(batch_data['sents'])

    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average='macro'),
        y_pred,
        y_true,
        sents
    )

def save_model(model, optimizer, args, config, filepath):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'pos_vocab': POS_VOCAB,
        'dep_vocab': DEP_VOCAB
    }, filepath)
    print(f"Model saved to {filepath}")

def train(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    
    # Load and prepare data
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')
    build_linguistic_vocabs(train_data)  # This populates POS_VOCAB and DEP_VOCAB
    
    # Initialize model
    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        option=args.option,
        use_pos=args.use_pos,
        use_dep=args.use_dep,
        use_wn=args.use_wn,
        pos_tag_vocab_size=len(POS_VOCAB),
        dep_vocab_size=len(DEP_VOCAB),
        wn_vocab_size=WN_VOCAB_SIZE
    )
    
    model = BertSentClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Create dataloaders
    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn
    )
    
    # Training loop
    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            batch_data = batch[0]
            
            inputs = {
                'input_ids': batch_data['token_ids'].to(device),
                'attention_mask': batch_data['attention_mask'].to(device),}
            if args.use_pos:
                inputs['pos_tag_ids'] = batch_data['pos_tag_ids'].to(device)
            if args.use_dep:
                inputs['dep_ids'] = batch_data['dep_ids'].to(device)
            if args.use_wn:
                inputs['wn_ids'] = batch_data['wn_ids'].to(device)
            
            labels = batch_data['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = F.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation
        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_dataloader):.3f} | "
              f"Train Acc: {train_acc:.3f} | Dev Acc: {dev_acc:.3f}")
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

def test(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    saved = torch.load(args.filepath, weights_only = False) 
    
    # Restore vocabularies
    global POS_VOCAB, DEP_VOCAB
    POS_VOCAB = saved.get('pos_vocab', POS_VOCAB)
    DEP_VOCAB = saved.get('dep_vocab', DEP_VOCAB)
    
    model = BertSentClassifier(saved['model_config']).to(device)
    model.load_state_dict(saved['model'])
    
    dev_data = create_data(args.dev, 'valid')
    test_data = create_data(args.test, 'test')
    
    dev_dataset = BertDataset(dev_data, args)
    test_dataset = BertDataset(test_data, args)
    
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    
    dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
    test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)
    
    # Save predictions
    for path, sents, true, pred in [
        (args.dev_out, dev_sents, dev_true, dev_pred),
        (args.test_out, test_sents, test_true, test_pred)
    ]:
        with open(path, 'w') as f:
            for s, t, p in zip(sents, true, pred):
                f.write(f"{s} ||| {t} ||| {p}\n")
    
    print(f"Dev Accuracy: {dev_acc:.3f} | Test Accuracy: {test_acc:.3f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str, choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--use_pos", action="store_true", help="Use POS tags", default=False)
    parser.add_argument("--use_dep", action="store_true", help="Use Dependency features", default=False)  
    parser.add_argument("--use_wn", action="store_true", help="Use WordNet", default=True)    
    
    args = parser.parse_args()
    if args.filepath is None:
        args.filepath = f"{args.option}-{args.epochs}-{args.lr}.pt"
    
    print(f"Arguments: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
    test(args)