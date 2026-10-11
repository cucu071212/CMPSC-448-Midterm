"""Local checks; synthetic smoke training is not a full Colab GPU validation."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare_data import ensure_csvs, make_splits, validate_csvs


def main():
    notebook = json.loads((ROOT / 'Training.ipynb').read_text(encoding='utf-8-sig'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['execution_count'] is None and not cell['outputs']
    rows = []
    for i in range(120):
        for model in ('gpt-example', 'claude-example', 'gemini-example'):
            conversation = [{'role': 'user', 'content': f'Unique question {i}'},
                            {'role': 'assistant', 'content': f'Answer from {model} number {i}'}]
            rows.append({'language': 'en', 'model_a': model, 'model_b': model,
                         'conversation_a': conversation, 'conversation_b': conversation})
    expected = make_splits(rows)
    assert all(expected[name].equals(make_splits(rows)[name]) for name in expected)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        data_dir = temporary / 'data'
        with patch('datasets.load_dataset', return_value=iter(rows)) as download:
            summary = ensure_csvs(data_dir)
            assert download.call_args.kwargs['token'] is False
            assert len(summary) == 3
        with patch('datasets.load_dataset', side_effect=AssertionError('Must reuse CSVs')):
            assert ensure_csvs(data_dir) == summary
        train_csv = data_dir / 'train.csv'
        validation_csv = data_dir / 'validation.csv'
        original = validation_csv.read_bytes()
        validation_csv.write_bytes(train_csv.read_bytes())
        try:
            validate_csvs(data_dir)
        except ValueError as exc:
            assert 'overlap' in str(exc)
        else:
            raise AssertionError('Prompt leakage accepted')
        validation_csv.write_bytes(original)
        validation_csv.write_text('broken\n', encoding='utf-8')
        with patch('datasets.load_dataset', return_value=iter(rows)):
            assert ensure_csvs(data_dir) == summary
        for model in ('cnn', 'rnn'):
            for mode in ('input', 'output', 'both'):
                subprocess.run([sys.executable, str(ROOT / 'train.py'),
                    '--model', model, '--mode', mode, '--data-dir', str(data_dir),
                    '--output-dir', str(temporary / 'results'), '--device', 'cpu',
                    '--epochs', '1', '--max-len', '12', '--emb-dim', '8',
                    '--hidden-dim', '8', '--num-filters', '4', '--batch-size', '64',
                    '--num-workers', '0', '--threads', '1'], cwd=temporary, check=True)
        metrics = list((temporary / 'results').rglob('metrics.json'))
        assert len(metrics) == 6
        for path in metrics:
            assert (path.parent / 'best_model.pt').is_file()
            assert (path.parent / 'training_and_confusion.png').is_file()
    print('PASS: notebook syntax, deterministic preparation, reuse, regeneration, leakage rejection,')
    print('and all six CNN/RNN input modes (one CPU epoch on synthetic data, unrelated working directory).')


if __name__ == '__main__':
    main()
