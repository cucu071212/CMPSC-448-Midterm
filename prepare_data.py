"""Download public Arena data and prepare validated, prompt-disjoint CSV splits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET_ID = 'lmarena-ai/arena-human-preference-140k'
DATASET_REVISION = '6322995ab34d7c2693e3f47dd13fa5caa0789a74'
LABELS = {'GPT': 0, 'Claude': 1, 'Gemini': 2}
COLUMNS = ['prompt_id', 'LLM_Input', 'LLM_output', 'LLM_name', 'label']
SPLITS = ('train', 'validation', 'test')


def get_prompt_id(prompt):
    normalized = re.sub(r'\s+', ' ', prompt).strip().casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def get_family(model_name):
    name = str(model_name or '').lower()
    if name.startswith(('gpt-', 'chatgpt-')):
        return 'GPT'
    if name.startswith('claude-'):
        return 'Claude'
    if name.startswith('gemini-'):
        return 'Gemini'
    return None


def extract_text(message):
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    if isinstance(content, str):
        return content.strip() or None
    if not isinstance(content, list) or not content:
        return None
    parts = []
    for block in content:
        if (not isinstance(block, dict) or block.get('type') != 'text'
                or block.get('image') is not None or block.get('mimeType') is not None
                or not isinstance(block.get('text'), str)):
            return None
        parts.append(block['text'])
    return '\n'.join(parts).strip() or None


def extract_conversation(conversation):
    if not isinstance(conversation, list) or len(conversation) != 2:
        return None
    user, assistant = conversation
    if (not isinstance(user, dict) or not isinstance(assistant, dict)
            or user.get('role') != 'user' or assistant.get('role') != 'assistant'):
        return None
    prompt, response = extract_text(user), extract_text(assistant)
    return (prompt, response) if prompt and response else None


def make_splits(rows, seed=42):
    records = []
    for scanned, row in enumerate(rows, 1):
        if scanned % 10000 == 0:
            print(f'Scanned {scanned:,} battles', flush=True)
        if row.get('language') != 'en':
            continue
        a = extract_conversation(row.get('conversation_a'))
        b = extract_conversation(row.get('conversation_b'))
        if a is None or b is None or get_prompt_id(a[0]) != get_prompt_id(b[0]):
            continue
        for side, conversation in (('a', a), ('b', b)):
            family = get_family(row.get(f'model_{side}'))
            if family:
                records.append(dict(zip(COLUMNS, [get_prompt_id(conversation[0]),
                    conversation[0], conversation[1], family, LABELS[family]])))
    df = pd.DataFrame.from_records(records, columns=COLUMNS).drop_duplicates(
        subset=['prompt_id', 'LLM_name', 'LLM_output']).reset_index(drop=True)
    if df.empty:
        raise ValueError('No matching English single-turn responses found')
    # Preserve the existing Preprocessing.ipynb protocol: 70/15/15 by prompt,
    # then downsample each split independently. Response counts may differ.
    train_idx, rest_idx = next(GroupShuffleSplit(
        n_splits=1, train_size=0.70, random_state=seed).split(df, groups=df.prompt_id))
    rest = df.iloc[rest_idx]
    val_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, train_size=0.50, random_state=seed).split(rest, groups=rest.prompt_id))
    splits = dict(zip(SPLITS, [df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]]))
    for name, part in splits.items():
        count = int(part.LLM_name.value_counts().reindex(LABELS, fill_value=0).min())
        if count == 0:
            raise ValueError(f'{name} is missing a model family')
        splits[name] = pd.concat([part[part.LLM_name == family].sample(
            n=count, random_state=seed) for family in LABELS]).sample(
                frac=1, random_state=seed).reset_index(drop=True)
    return splits


def validate_csvs(data_dir):
    """Reject empty/malformed CSVs, wrong labels, duplicate rows and prompt leakage."""
    seen, summary = set(), {}
    for name in SPLITS:
        df = pd.read_csv(Path(data_dir) / f'{name}.csv', keep_default_na=False)
        if not set(COLUMNS).issubset(df.columns) or df.empty:
            raise ValueError(f'{name}: missing required columns or empty CSV')
        for column in COLUMNS[:-1]:
            if not df[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError(f'{name}: empty or non-text {column}')
        if not df.label.eq(df.LLM_name.map(LABELS)).all() or set(df.label) != set(LABELS.values()):
            raise ValueError(f'{name}: expected GPT=0, Claude=1, Gemini=2, all present')
        hashes = df.LLM_Input.map(get_prompt_id)
        if not hashes.eq(df.prompt_id).all():
            raise ValueError(f'{name}: prompt_id does not match normalized prompt hash')
        if df.duplicated(['prompt_id', 'LLM_name', 'LLM_output']).any():
            raise ValueError(f'{name}: duplicate examples')
        prompts = set(hashes)
        if seen & prompts:
            raise ValueError(f'{name}: prompts overlap another split')
        seen.update(prompts)
        summary[name] = {'rows': len(df), 'class_counts': {
            family: int((df.LLM_name == family).sum()) for family in LABELS}}
    return summary


def ensure_csvs(data_dir, seed=42):
    data_dir = Path(data_dir)
    try:
        summary = validate_csvs(data_dir)
        print('Reusing validated CSVs:', data_dir, flush=True)
        return summary
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f'CSV preparation required: {exc}', flush=True)
    from datasets import load_dataset
    print(f'Downloading {DATASET_ID} @ {DATASET_REVISION} (anonymous access)', flush=True)
    # token=False explicitly avoids depending on a personal account or cached token.
    rows = load_dataset(DATASET_ID, revision=DATASET_REVISION, split='train',
                        streaming=True, token=False)
    splits = make_splits(rows, seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Validate all new files before replacing any existing CSV.
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        staged = Path(temporary)
        for name, part in splits.items():
            part.to_csv(staged / f'{name}.csv', index=False)
        summary = validate_csvs(staged)
        for name in SPLITS:
            (staged / f'{name}.csv').replace(data_dir / f'{name}.csv')
    manifest = {'dataset': DATASET_ID, 'revision': DATASET_REVISION, 'seed': seed,
                'protocol': 'English text-only single-turn, prompt-group 70/15/15, balanced per split, raw responses',
                'splits': summary}
    (data_dir / 'preparation.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Created and validated CSVs:', summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    ensure_csvs(args.data_dir, args.seed)
