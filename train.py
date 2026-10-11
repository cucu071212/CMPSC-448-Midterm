"""Shared training entry point for data.py, TextCNN and TextRNN.

Example: python train.py --model cnn --mode output --data-dir /content/drive/MyDrive/dataset
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import Vocab, build_loaders
from model.cnn import TextCNN, LABELS, ID2LABEL
from model.rnn import TextRNN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, truth, predicted, probabilities = 0.0, [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            truth.extend(y.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(logits.softmax(dim=1).cpu().tolist())
    if not truth:
        raise ValueError('Cannot evaluate an empty split')
    metrics = {'loss': total_loss / len(truth), 'accuracy': accuracy_score(truth, predicted),
               'macro_f1': f1_score(truth, predicted, labels=list(ID2LABEL), average='macro', zero_division=0)}
    return metrics, truth, predicted, probabilities


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_plots(history, matrix, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'val'):
        axes[0].plot(history.epoch, history[f'{split}_loss'], marker='o', label=split)
    axes[0].set(xlabel='Epoch', ylabel='Cross entropy', title='Training loss')
    axes[0].legend()
    axes[1].imshow(matrix, cmap='Blues')
    axes[1].set(xticks=range(3), yticks=range(3), xticklabels=list(LABELS),
                yticklabels=list(LABELS), xlabel='Predicted', ylabel='True', title='Test confusion matrix')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(matrix[i, j]), ha='center', va='center',
                         color='white' if matrix[i, j] > matrix.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(out / 'training_and_confusion.png', dpi=160)
    plt.close(fig)


def train(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu or auto')
    # Each run gets a separate directory, so previous results are preserved.
    out = args.output_dir / f'{args.model}_{args.mode}' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['resolved_device'] = str(device)
    save_json(out / 'config.json', config)
    print(f'Device: {device}\nResults: {out}\nBuilding data loaders...', flush=True)
    vocab = Vocab.load(args.vocab_path) if args.vocab_path else None
    loaders, vocab = build_loaders(args.data_dir, mode=args.mode, max_len=args.max_len,
                                   batch_size=args.batch_size, min_freq=args.min_freq,
                                   max_size=args.vocab_size, vocab=vocab, num_workers=args.num_workers)
    vocab.save(out / 'vocab.json')
    dataset_info = {}
    for split, loader in loaders.items():
        _, labels = loader.dataset.tensors
        if len(labels) == 0 or set(labels.tolist()) != set(LABELS.values()):
            raise ValueError(f'{split} must contain all three classes and labels 0/1/2')
        dataset_info[split] = {'rows': len(labels), 'class_counts': {
            name: int((labels == label).sum()) for name, label in LABELS.items()}}
    save_json(out / 'dataset_info.json', dataset_info)
    kwargs = {'vocab_size': len(vocab), 'emb_dim': args.emb_dim, 'dropout': args.dropout}
    if args.model == 'cnn':
        kwargs.update(kernel_sizes=tuple(args.kernel_sizes), num_filters=args.num_filters)
        model = TextCNN(**kwargs)
    else:
        kwargs.update(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        model = TextRNN(**kwargs)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for step, (x, y) in enumerate(loaders['train'], start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += y.size(0)
            if args.log_every and step % args.log_every == 0:
                print(f'Epoch {epoch} batch {step}/{len(loaders["train"])} loss={total_loss/count:.4f}', flush=True)
        val, _, _, _ = evaluate(model, loaders['validation'], criterion, device)
        row = {'epoch': epoch, 'train_loss': total_loss/count, 'train_accuracy': correct/count,
               **{'val_' + k: v for k, v in val.items()}, 'seconds': time.monotonic() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(f'Epoch {epoch}: train_loss={row["train_loss"]:.4f} val_loss={val["loss"]:.4f} '
              f'val_acc={val["accuracy"]:.4f} val_macro_f1={val["macro_f1"]:.4f} '
              f'({row["seconds"]:.1f}s)', flush=True)
        if val['macro_f1'] > best_f1:
            best_f1, stale = val['macro_f1'], 0
            torch.save({'model_state_dict': model.state_dict(), 'model': args.model,
                        'model_kwargs': kwargs, 'mode': args.mode, 'vocab_itos': vocab.itos,
                        'labels': LABELS, 'config': config, 'best_epoch': epoch,
                        'validation_metrics': val}, out / 'best_model.pt')
        else:
            stale += 1
        if stale >= args.patience:
            print(f'Early stopping after {args.patience} epochs without improvement.', flush=True)
            break
    checkpoint = torch.load(out / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Test is evaluated once, after all model selection is finished.
    test, truth, predicted, probabilities = evaluate(model, loaders['test'], criterion, device)
    report = classification_report(truth, predicted, labels=list(ID2LABEL), target_names=list(LABELS),
                                   output_dict=True, zero_division=0)
    metrics = {'model': args.model, 'mode': args.mode, 'best_epoch': checkpoint['best_epoch'],
               'validation': checkpoint['validation_metrics'], 'test': test,
               'classification_report': report}
    save_json(out / 'metrics.json', metrics)
    matrix = confusion_matrix(truth, predicted, labels=list(ID2LABEL))
    pd.DataFrame(matrix, index=list(LABELS), columns=list(LABELS)).to_csv(out / 'confusion_matrix.csv')
    predictions = pd.DataFrame({'test_row_index': range(len(truth)), 'true_label': truth,
                                'predicted_label': predicted,
                                'true_family': [ID2LABEL[y] for y in truth],
                                'predicted_family': [ID2LABEL[y] for y in predicted]})
    for i, name in enumerate(LABELS):
        predictions[f'p_{name}'] = np.asarray(probabilities)[:, i]
    predictions.to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame([{'model': args.model, 'mode': args.mode, 'best_epoch': checkpoint['best_epoch'],
                   'validation_macro_f1': best_f1, **{'test_' + k: v for k, v in test.items()}}]).to_csv(out / 'summary.csv', index=False)
    save_plots(pd.DataFrame(history), matrix, out)
    print(f'Test: {test}\nSaved: {out}', flush=True)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cnn', 'rnn'], default='cnn')
    parser.add_argument('--mode', choices=['input', 'output', 'both'], default='output')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'Dataset')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'training_results')
    parser.add_argument('--vocab-path', type=Path, help='Reuse a vocabulary from the same mode and training split')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    for name, default in [('epochs', 15), ('patience', 3), ('batch-size', 64), ('max-len', 512),
                          ('min-freq', 2), ('vocab-size', 30000), ('emb-dim', 128), ('hidden-dim', 128),
                          ('num-layers', 1), ('num-filters', 100), ('num-workers', 0), ('seed', 42),
                          ('threads', 4), ('log-every', 100)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--kernel-sizes', nargs='+', type=int, default=[3, 4, 5])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    args = parser.parse_args()
    for name in ('epochs', 'patience', 'batch_size', 'max_len', 'min_freq', 'emb_dim',
                 'hidden_dim', 'num_layers', 'num_filters', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if (args.vocab_size < 4 or args.num_workers < 0 or args.log_every < 0
            or not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0
            or args.grad_clip <= 0 or any(k < 1 for k in args.kernel_sizes)):
        parser.error('Invalid vocabulary, worker count, dropout, optimizer or kernel settings')
    return args


if __name__ == '__main__':
    train(parse_args())
