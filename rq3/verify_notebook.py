"""Check notebook syntax and RQ3 split invariants on synthetic data."""
import ast
import json
from pathlib import Path

import pandas as pd


def main():
    notebook = json.loads(Path(__file__).with_name('RQ3_Training.ipynb').read_text(encoding='utf-8'))
    namespace = {'pd': pd, 'LABELS': {'GPT': 0, 'Claude': 1, 'Gemini': 2},
                 'SPLITS': ('train', 'validation', 'test')}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        assert cell['execution_count'] is None and cell['outputs'] == []
        tree = ast.parse(''.join(cell['source']))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'balanced', 'minimum_count', 'make_rq3_splits'}]
        if functions:
            exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook>', 'exec'), namespace)
    frames = {}
    for split in namespace['SPLITS']:
        rows = []
        for domain, n in [('coding', 4), ('general_qa', 7)]:
            for family, label in namespace['LABELS'].items():
                for i in range(n):
                    rows.append({'prompt_id': f'{split}-{domain}-{i}', 'LLM_name': family,
                                 'label': label, 'domain': domain, 'LLM_Input': f'{domain} {i}',
                                 'LLM_output': f'{family} response {i}'})
        frames[split] = pd.DataFrame(rows)
    build = namespace['make_rq3_splits']
    first = build(frames, 'coding', ['general_qa'], 42)
    second = build(frames, 'coding', ['general_qa'], 42)
    for condition, parts in first.items():
        for split, frame in parts.items():
            assert frame.equals(second[condition][split])
            assert len(frame) == 12
            assert frame.LLM_name.value_counts().eq(4).all()
    assert first['in_domain']['test'].equals(first['cross_domain']['test'])
    broken = {split: frame.copy() for split, frame in frames.items()}
    broken['train'] = broken['train'].loc[~(broken['train'].domain.eq('general_qa')
                                          & broken['train'].LLM_name.eq('Gemini'))]
    try:
        build(broken, 'coding', ['general_qa'], 42)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing source family accepted')
    broken = {split: frame.copy() for split, frame in frames.items()}
    broken['validation']['prompt_id'] = broken['validation'].prompt_id.str.replace('validation', 'train')
    try:
        build(broken, 'coding', ['general_qa'], 42)
    except ValueError:
        pass
    else:
        raise AssertionError('Prompt leakage accepted')
    print('PASS: notebook syntax, clean outputs, deterministic matching, equal family counts,')
    print('identical target tests, missing-family rejection, and prompt-leakage rejection.')


if __name__ == '__main__':
    main()
