"""Generate the RQ3 Colab notebook using only the standard library."""
import ast
import json
from pathlib import Path

cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(True)})


def code(source):
    ast.parse(source)
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': source.splitlines(True)})


md('''# RQ3 — Do LLM fingerprints generalize across domains?
Reuse the existing TextCNN, BiLSTM, tokenizer and training loop. Each condition trains a new model from scratch using responses only to predict GPT / Claude / Gemini.

Select **Runtime → Change runtime type → T4 GPU**. Run the setup and review cells first. Download the prompt review CSV, annotate `reviewed_domain`, then upload it in the review cell before continuing. The notebook deliberately stops before training if reviewed labels are unavailable.

Default experiment: **general question answering → coding**. Compare this cross-domain model with a coding-trained baseline on the **same coding test set**. CNN and RNN give four runs. Domain definitions and counts must be reported with the results. RQ3 is extra credit in the assignment PDF (page 2).

Colab runtime resets erase `/content`; download results and reviewed labels before disconnecting. Full dataset preparation downloads about 1.61 GB on the first run.
''')
code('''from pathlib import Path
import sys, subprocess, json, hashlib
from datetime import datetime

REPO_URL = 'https://github.com/cucu071212/CMPSC-448-Midterm.git'
REPO_REF = 'main'  # Same pipeline branch used by Training.ipynb.
PROJECT_DIR = Path('/content/CMPSC-448-Midterm')
if not PROJECT_DIR.exists():
    subprocess.run(['git', 'clone', '--branch', REPO_REF, REPO_URL, str(PROJECT_DIR)], check=True)
PROJECT_DIR = PROJECT_DIR.resolve()
for name in ('prepare_data.py', 'train.py', 'data.py', 'src/cnn.py', 'src/rnn.py', 'requirements.txt'):
    if not (PROJECT_DIR / name).is_file():
        raise RuntimeError(f'Missing {name}; select the branch containing the shared training pipeline.')
sys.path.insert(0, str(PROJECT_DIR))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(PROJECT_DIR / 'requirements.txt')], check=True)
SOURCE_DIR = PROJECT_DIR / 'data'
RQ3_DIR = PROJECT_DIR / 'rq3'
REVIEW_DIR = RQ3_DIR / 'review'
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
print('Project:', PROJECT_DIR)
''')
md('''## 1. Shared settings and original data
Keep architecture, tokenizer, optimizer and training settings identical to the main notebook. `ensure_csvs` is used **only for the original source data**, never for RQ3 experiment folders.

Define domains by the **user prompt**, not by the response or its model family. `general_qa` means factual or explanatory question answering; exclude code, mathematical problem solving and creative writing. Blank labels are excluded. Do not automatically treat `other` as general QA.
''')
code('''import torch
import pandas as pd
import re
from IPython.display import display, Image
from prepare_data import ensure_csvs, validate_csvs, COLUMNS, LABELS, SPLITS

SEED = 42
EPOCHS = 15
PATIENCE = 3
BATCH_SIZE = 64
MAX_LEN = 512
TARGET_DOMAIN = 'coding'
SOURCE_DOMAINS = ['general_qa']  # Can use ['coding', 'math', 'general_qa'] with target 'writing'.
ALLOWED_DOMAINS = {'coding', 'math', 'writing', 'general_qa', 'other', 'ambiguous'}
UPLOAD_REVIEW = True  # False reuses REVIEW_PATH after you annotate it in the runtime.
DOWNLOAD_REVIEW = True
assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
assert TARGET_DOMAIN in ALLOWED_DOMAINS and TARGET_DOMAIN not in {'other', 'ambiguous'}
assert SOURCE_DOMAINS and len(set(SOURCE_DOMAINS)) == len(SOURCE_DOMAINS)
assert set(SOURCE_DOMAINS) <= ALLOWED_DOMAINS - {'other', 'ambiguous', TARGET_DOMAIN}
print('GPU:', torch.cuda.get_device_name(0))
ensure_csvs(SOURCE_DIR, seed=SEED)
print('Original data:', validate_csvs(SOURCE_DIR))
frames = {split: pd.read_csv(SOURCE_DIR / f'{split}.csv', keep_default_na=False)
          for split in SPLITS}
''')
md('''## 2. Export prompt-level review file
Suggestions are keyword heuristics only. Review all prompts you want to include, using `coding`, `math`, `writing`, `general_qa`, `other`, or `ambiguous`. One prompt has one domain across every model response. Existing review files are preserved.

For a feasible review workload, filter the CSV by `suggested_domain`; you may leave unreviewed rows blank. Report how prompts were selected and reviewed, and how many were excluded. Small reviewed subsets produce exploratory results rather than broad claims about generalization.
''')
code('''RULES = {
    'coding': re.compile(r'\\b(python|javascript|typescript|programming|debug|sql|compiler|html|css)\\b', re.I),
    'math': re.compile(r'\\b(equation|integral|derivative|algebra|probability|calculus|theorem)\\b', re.I),
    'writing': re.compile(r'\\b(essay|poem|rewrite|short story|creative writing)\\b', re.I),
}
def suggest(prompt):
    matched = [domain for domain, pattern in RULES.items() if pattern.search(prompt)]
    return matched[0] if len(matched) == 1 else ('ambiguous' if matched else 'other')

expected = pd.concat([frame[['prompt_id', 'LLM_Input']].assign(split=split)
                      for split, frame in frames.items()], ignore_index=True)
expected = expected.drop_duplicates('prompt_id').reset_index(drop=True)
REVIEW_PATH = REVIEW_DIR / 'domain_review.csv'
if not REVIEW_PATH.exists():
    review = expected.copy()
    review['suggested_domain'] = review.LLM_Input.map(suggest)
    review['reviewed_domain'] = ''
    review.to_csv(REVIEW_PATH, index=False)
else:
    print('Preserving existing review:', REVIEW_PATH)
review = pd.read_csv(REVIEW_PATH, keep_default_na=False)
display(review.head())
display(review.groupby(['split', 'suggested_domain']).size().unstack(fill_value=0))
if DOWNLOAD_REVIEW:
    from google.colab import files
    files.download(str(REVIEW_PATH))
''')
md('''## 3. Upload and validate reviewed domains
Upload the edited CSV when prompted. Only nonblank reviewed labels are used. Unknown labels, changed prompts, duplicate IDs and mismatched split assignments are rejected. Review data is validated before replacing the runtime copy.
''')
code('''if UPLOAD_REVIEW:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Upload exactly one reviewed CSV.')
    import io
    review_bytes = next(iter(uploaded.values()))
    review = pd.read_csv(io.BytesIO(review_bytes), keep_default_na=False)
else:
    review = pd.read_csv(REVIEW_PATH, keep_default_na=False)

required = {'prompt_id', 'LLM_Input', 'split', 'reviewed_domain'}
if not required <= set(review.columns) or review.prompt_id.duplicated().any():
    raise ValueError('Review CSV has missing columns or duplicate prompt IDs.')
if set(review.prompt_id) != set(expected.prompt_id):
    raise ValueError('Review CSV must retain all original prompt rows; leave unused labels blank.')
checked = expected.merge(review[list(required)], on='prompt_id', suffixes=('_original', '_review'),
                         validate='one_to_one')
if not (checked.LLM_Input_original.eq(checked.LLM_Input_review).all()
        and checked.split_original.eq(checked.split_review).all()):
    raise ValueError('Review prompts or split assignments differ from the source data.')
review['reviewed_domain'] = review.reviewed_domain.str.strip().str.lower()
unknown = set(review.reviewed_domain) - ALLOWED_DOMAINS - {''}
if unknown:
    raise ValueError(f'Unknown reviewed domains: {sorted(unknown)}')
if not review.reviewed_domain.ne('').any():
    raise ValueError('No reviewed labels. Annotate reviewed_domain and rerun this cell.')
review.to_csv(REVIEW_PATH, index=False)
domain_map = review.set_index('prompt_id').reviewed_domain
annotated = {split: frame.assign(domain=frame.prompt_id.map(domain_map))
             for split, frame in frames.items()}
counts = pd.concat([frame.assign(split=split) for split, frame in annotated.items()])
display(pd.crosstab([counts['split'], counts['domain']], counts.LLM_name))
print('Unreviewed prompts:', int(review.reviewed_domain.eq('').sum()))
''')
md('''## 4. Build matched experiments
Keep the original prompt split memberships. Cross-domain train and validation exclude the target domain. Within each split, balance GPT / Claude / Gemini. Match train and validation response counts per family across conditions to control data volume. Both conditions receive an identical, family-balanced target-domain test set.

Each condition builds its own vocabulary using only its training responses. CNN and RNN use the same CSVs and seed. New timestamped folders preserve previous experiments. This code validates leakage and matching before writing experiment files.
''')
code('''def balanced(frame, n, seed):
    return pd.concat([frame.loc[frame.LLM_name.eq(family)].sample(n=n, random_state=seed)
                      for family in LABELS], ignore_index=True).sample(
                          frac=1, random_state=seed).reset_index(drop=True)

def minimum_count(frame):
    return int(frame.LLM_name.value_counts().reindex(LABELS, fill_value=0).min())

def make_rq3_splits(annotated, target, sources, seed):
    if not sources or target in sources:
        raise ValueError('Source domains must be nonempty and exclude the target.')
    built = {'in_domain': {}, 'cross_domain': {}}
    for split in ('train', 'validation'):
        inside = annotated[split].loc[annotated[split].domain.eq(target)]
        outside = annotated[split].loc[annotated[split].domain.isin(sources)]
        n = min(minimum_count(inside), minimum_count(outside))
        if n < 1:
            raise ValueError(f'{split}: both conditions need reviewed examples from all three families.')
        built['in_domain'][split] = balanced(inside, n, seed)
        built['cross_domain'][split] = balanced(outside, n, seed)
    target_test = annotated['test'].loc[annotated['test'].domain.eq(target)]
    n = minimum_count(target_test)
    if n < 1:
        raise ValueError('Target test needs reviewed examples from all three families.')
    test = balanced(target_test, n, seed)
    for condition, parts in built.items():
        parts['test'] = test.copy()
        seen = set()
        for split in SPLITS:
            part = parts[split]
            ids = set(part.prompt_id)
            if seen & ids:
                raise ValueError(f'Prompt leakage in {condition}/{split}')
            if not ids <= set(annotated[split].prompt_id):
                raise ValueError('Original split membership changed.')
            seen.update(ids)
            if part.label.tolist() != part.LLM_name.map(LABELS).tolist():
                raise ValueError('Incorrect family labels.')
            if condition == 'cross_domain' and split != 'test' and part.domain.eq(target).any():
                raise ValueError('Target domain leaked into cross-domain fitting.')
    assert built['in_domain']['test'].equals(built['cross_domain']['test'])
    for split in ('train', 'validation'):
        assert built['in_domain'][split].LLM_name.value_counts().equals(
            built['cross_domain'][split].LLM_name.value_counts())
    return built

built = make_rq3_splits(annotated, TARGET_DOMAIN, SOURCE_DOMAINS, SEED)
SESSION_DIR = RQ3_DIR / 'experiments' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
SESSION_DIR.mkdir(parents=True, exist_ok=False)
DATA_DIRS = {}
manifest = {'target_domain': TARGET_DOMAIN, 'source_domains': SOURCE_DOMAINS,
            'seed': SEED, 'mode': 'output', 'review_sha256': hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest(),
            'reviewed_prompts': int(review.reviewed_domain.ne('').sum()),
            'unreviewed_prompts': int(review.reviewed_domain.eq('').sum()), 'conditions': {}}
for condition, parts in built.items():
    directory = SESSION_DIR / 'data' / condition
    directory.mkdir(parents=True)
    for split, part in parts.items():
        part[COLUMNS].to_csv(directory / f'{split}.csv', index=False)
    DATA_DIRS[condition] = directory
    manifest['conditions'][condition] = validate_csvs(directory)
assert (DATA_DIRS['in_domain'] / 'test.csv').read_bytes() == (DATA_DIRS['cross_domain'] / 'test.csv').read_bytes()
review.to_csv(SESSION_DIR / 'domain_review.csv', index=False)
counts.groupby(['split', 'domain', 'LLM_name']).size().to_csv(SESSION_DIR / 'domain_counts.csv')
(SESSION_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
OUTPUT_DIR = SESSION_DIR / 'results'
display(pd.DataFrame([dict(condition=c, split=s, **info)
                      for c, splits in manifest['conditions'].items() for s, info in splits.items()]))
print('Experiment folder:', SESSION_DIR)
''')
md('''## 5. Shared training and display helpers
Fresh weights for every run; no RQ1/RQ2 checkpoint or full-data vocabulary is loaded. Select the best epoch by validation macro-F1, then evaluate the test once. Defaults for the architecture, AdamW and gradient clipping come from the same `train.py` as `Training.ipynb`.
''')
code('''EXPERIMENTS = [('in_domain', 'cnn'), ('cross_domain', 'cnn'),
               ('in_domain', 'rnn'), ('cross_domain', 'rnn')]
RUNS = {}

def train_and_review(condition, model):
    RUNS.pop((condition, model), None)
    root = OUTPUT_DIR / condition
    before = set(root.glob(f'{model}_output/*/summary.csv'))
    command = [sys.executable, '-u', str(PROJECT_DIR / 'train.py'),
               '--model', model, '--mode', 'output', '--data-dir', str(DATA_DIRS[condition]),
               '--output-dir', str(root), '--epochs', str(EPOCHS), '--patience', str(PATIENCE),
               '--batch-size', str(BATCH_SIZE), '--max-len', str(MAX_LEN),
               '--seed', str(SEED), '--device', 'cuda', '--num-workers', '0']
    with subprocess.Popen(command, cwd=str(PROJECT_DIR), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f'Training failed: {condition}/{model}; exit={returncode}')
    created = set(root.glob(f'{model}_output/*/summary.csv')) - before
    if len(created) != 1:
        raise RuntimeError('Could not identify exactly one new training result.')
    run_dir = next(iter(created)).parent
    RUNS[(condition, model)] = run_dir
    display(pd.read_csv(run_dir / 'summary.csv'))
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    display(pd.DataFrame(metrics['classification_report']).T)
    display(Image(filename=str(run_dir / 'training_and_confusion.png')))
    display(pd.read_csv(run_dir / 'predictions.csv').head())
    print('Saved:', run_dir)
    return run_dir
''')
for condition, model in [('in_domain', 'cnn'), ('cross_domain', 'cnn'), ('in_domain', 'rnn'), ('cross_domain', 'rnn')]:
    md(f'## {model.upper()} — {condition.replace("_", " ")}')
    code(f"{model}_{condition}_run = train_and_review('{condition}', '{model}')\n")
md('''## 6. Compare generalization
Positive F1 drop means cross-domain performance decreased. The class-level table shows which **LLM families** remain identifiable. With three balanced classes, chance accuracy is about 33.3%. Report sample counts and avoid tuning settings on test results. A single seed and small test set limit the strength of conclusions; repeated seeds can be added for robustness.
''')
code('''tables, per_class, missing = [], [], []
for condition, model in EXPERIMENTS:
    run_dir = RUNS.get((condition, model))
    if run_dir is None:
        missing.append(f'{condition}/{model}')
        continue
    table = pd.read_csv(run_dir / 'summary.csv')
    table['condition'] = condition
    table['target_domain'] = TARGET_DOMAIN
    table['source_domains'] = ','.join(SOURCE_DOMAINS) if condition == 'cross_domain' else TARGET_DOMAIN
    table['run_dir'] = str(run_dir)
    tables.append(table)
    report = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))['classification_report']
    for family in LABELS:
        per_class.append(dict(model=model, condition=condition, family=family, **report[family]))
if tables:
    comparison = pd.concat(tables, ignore_index=True)
    comparison.to_csv(SESSION_DIR / 'comparison.csv', index=False)
    display(comparison)
    class_table = pd.DataFrame(per_class)
    class_table.to_csv(SESSION_DIR / 'per_family.csv', index=False)
    display(class_table)
    drops = comparison.pivot(index='model', columns='condition', values='test_macro_f1')
    if {'in_domain', 'cross_domain'} <= set(drops.columns):
        drops['f1_drop'] = drops['in_domain'] - drops['cross_domain']
        drops.to_csv(SESSION_DIR / 'generalization_gap.csv')
        display(drops)
if missing:
    print('Incomplete comparison; missing:', ', '.join(missing))
else:
    print('All four experiments are included.')
''')
md('''## 7. Download results and reproducibility data
The ZIP includes reviewed labels, counts, manifest, actual experiment CSVs, vocabularies, model checkpoints, metrics and comparison tables. It may be large. Keep this archive privately for reproduction; raw responses are excluded from the repository by the existing ignore rules.
''')
code('''import shutil
from google.colab import files
archive = Path(shutil.make_archive(str(SESSION_DIR.parent / (SESSION_DIR.name + '_rq3')),
                                   'zip', root_dir=SESSION_DIR))
print('Archive:', archive)
files.download(str(archive))
''')

# The default delivered notebook is the short, automatic RQ3 pipeline.
original_cells = cells
cells = []
md('''# RQ3: automatic cross-domain comparison
Open this notebook in Colab and choose **Runtime > Run all**.
Setup automatically clones the `main` branch, installs dependencies, and prepares
the public dataset when valid CSVs are unavailable. No path edits, uploads,
Google Drive mount, or manual domain review are required.
GPU is preferred automatically; CPU is used when no GPU is available (slower).
The first dataset preparation downloads about 1.61 GB in addition to dependencies.
Compares coding-trained and math-trained classifiers on the SAME coding test set,
using CNN and RNN (four runs). Domains are keyword proxies from prompts, not human-reviewed labels.
The quick run uses at most 500 training, 100 validation and 200 test responses per family,
3 epochs and 128 tokens. This is an exploratory coding-versus-math RQ3 result.
Runtime depends on hardware, downloads, and available examples.
Results are saved under `/content/CMPSC-448-Midterm/rq3/experiments` in Colab.
Download the results before disconnecting; runtime resets erase local files.
''')
code('''from pathlib import Path
import sys, subprocess, json, re
from datetime import datetime

REPO_URL = 'https://github.com/cucu071212/CMPSC-448-Midterm.git'
REPO_REF = 'main'
PROJECT_DIR = Path('/content/CMPSC-448-Midterm')
if not (PROJECT_DIR / 'train.py').is_file() and Path('train.py').is_file():
    PROJECT_DIR = Path.cwd()
if not PROJECT_DIR.exists():
    subprocess.run(['git', 'clone', '--branch', REPO_REF, REPO_URL,
                    str(PROJECT_DIR)], check=True)
PROJECT_DIR = PROJECT_DIR.resolve()
SOURCE_DIR = PROJECT_DIR / 'data'
RQ3_DIR = PROJECT_DIR / 'rq3'
for name in ('train.py', 'data.py', 'prepare_data.py', 'requirements.txt', 'src/cnn.py', 'src/rnn.py'):
    if not (PROJECT_DIR / name).is_file():
        raise RuntimeError(f'Code missing: {PROJECT_DIR / name}. Use the main branch containing the pipeline.')
sys.path.insert(0, str(PROJECT_DIR))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(PROJECT_DIR / 'requirements.txt')], check=True)
''')
code('''import pandas as pd
import torch
from IPython.display import display, Image
from prepare_data import ensure_csvs, validate_csvs, COLUMNS, LABELS, SPLITS

SEED, EPOCHS, PATIENCE, BATCH_SIZE, MAX_LEN = 42, 3, 2, 64, 128
TARGET_DOMAIN, SOURCE_DOMAINS = 'coding', ['math']
SPLIT_CAPS = {'train': 500, 'validation': 100, 'test': 200}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU (training will be slower)')
ensure_csvs(SOURCE_DIR, seed=SEED)
print('Original data:', validate_csvs(SOURCE_DIR))
frames = {split: pd.read_csv(SOURCE_DIR / f'{split}.csv', keep_default_na=False)
          for split in SPLITS}
''')
md('''## Automatic domain assignment
Use explicit coding and math keywords. Exclude prompts matching both or neither.
Save assigned domains and counts for reproducibility; no review upload is required.
''')
code('''RULES = {
    'coding': re.compile(r'\\b(python|javascript|typescript|programming|debug|sql|compiler|html|css)\\b', re.I),
    'math': re.compile(r'\\b(equation|integral|derivative|algebra|probability|calculus|theorem)\\b', re.I),
}
def assign_domain(prompt):
    matches = [domain for domain, rule in RULES.items() if rule.search(prompt)]
    return matches[0] if len(matches) == 1 else 'excluded'
expected = pd.concat([frame[['prompt_id', 'LLM_Input']].assign(split=split)
                      for split, frame in frames.items()], ignore_index=True)
assignments = expected.drop_duplicates('prompt_id').copy()
assignments['domain'] = assignments.LLM_Input.map(assign_domain)
domain_map = assignments.set_index('prompt_id').domain
annotated = {split: frame.assign(domain=frame.prompt_id.map(domain_map))
             for split, frame in frames.items()}
counts = pd.concat([frame.assign(split=split) for split, frame in annotated.items()])
display(pd.crosstab([counts['split'], counts['domain']], counts.LLM_name))
''')
md('''## Matched experiment data
Preserve original splits, balance families, match fitting counts and reuse identical target test data.
''')
build_source = next(''.join(c['source']) for c in original_cells
                    if c['cell_type'] == 'code' and ''.join(c['source']).startswith('def balanced'))
build_source = build_source.replace('n = min(minimum_count(inside), minimum_count(outside))',
    "n = min(minimum_count(inside), minimum_count(outside), globals().get('SPLIT_CAPS', {}).get(split, 10**9))")
build_source = build_source.replace('n = minimum_count(target_test)',
    "n = min(minimum_count(target_test), globals().get('SPLIT_CAPS', {}).get('test', 10**9))")
build_source = build_source.replace('reviewed examples', 'keyword-assigned examples').replace('reviewed examples', 'examples')
start = build_source.index("manifest = {")
end = build_source.index('for condition, parts in built.items():', start)
build_source = build_source[:start] + '''manifest = {'target_domain': TARGET_DOMAIN, 'source_domains': SOURCE_DOMAINS,
            'seed': SEED, 'mode': 'output', 'domain_method': 'prompt_keyword_proxy',
            'split_caps_per_family': SPLIT_CAPS, 'epochs': EPOCHS, 'max_len': MAX_LEN,
            'conditions': {}}
''' + build_source[end:]
build_source = build_source.replace("review.to_csv(SESSION_DIR / 'domain_review.csv', index=False)",
    "assignments.to_csv(SESSION_DIR / 'domain_assignments.csv', index=False)")
code(build_source)
helper = next(''.join(c['source']) for c in original_cells
              if c['cell_type'] == 'code' and ''.join(c['source']).startswith('EXPERIMENTS ='))
code(helper.replace("'--device', 'cuda'", "'--device', DEVICE"))
code('''for condition, model in EXPERIMENTS:
    print(f'Running {condition} / {model}', flush=True)
    train_and_review(condition, model)
''')
comparison_source = next(''.join(c['source']) for c in original_cells
                         if c['cell_type'] == 'code' and ''.join(c['source']).startswith('tables, per_class'))
md('''## RQ3 results
Positive F1 drop means fitting on math transfers worse to coding than fitting on coding.
Use the displayed sample counts and keyword-proxy domain definition in your report.
''')
code(comparison_source + "\nprint('Results saved:', SESSION_DIR)\n")

notebook = {'cells': cells, 'metadata': {'colab': {'name': 'RQ3_Training.ipynb'},
            'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'rq3-{i:02d}'
path = Path(__file__).with_name('RQ3_Training.ipynb')
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Created {path} ({len(cells)} cells); all code cells passed syntax validation.')
