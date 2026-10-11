"""Export prompt-level domain suggestions for review without changing source data."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
RULES = {
    'coding': re.compile(r'\b(python|javascript|typescript|programming|debug|sql|compiler|html|css)\b', re.I),
    'math': re.compile(r'\b(equation|integral|derivative|algebra|probability|calculus|theorem)\b', re.I),
    'writing': re.compile(r'\b(essay|poem|rewrite|short story|creative writing)\b', re.I),
}


def main():
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        parser.error('Source and output directories must differ')
    prompts, counts = {}, {}
    for split in ('train', 'validation', 'test'):
        with (args.source_dir / f'{split}.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                prompt_id = row['prompt_id']
                normalized = re.sub(r'\s+', ' ', row['LLM_Input']).strip().casefold()
                matched = [name for name, rule in RULES.items() if rule.search(normalized)]
                domain = matched[0] if len(matched) == 1 else ('ambiguous' if matched else 'other')
                if prompt_id in prompts:
                    previous = prompts[prompt_id]
                    previous_normalized = re.sub(r'\s+', ' ', previous['LLM_Input']).strip().casefold()
                    if previous['split'] != split or previous_normalized != normalized:
                        raise ValueError(f'Overlapping splits or inconsistent prompt: {prompt_id}')
                else:
                    prompts[prompt_id] = {
                        'prompt_id': prompt_id, 'split': split,
                        'LLM_Input': row['LLM_Input'], 'suggested_domain': domain,
                        'reviewed_domain': '',
                    }
                key = f"{split}/{domain}/{row['LLM_name']}"
                counts[key] = counts.get(key, 0) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / 'domain_review.csv'
    # Refuse to overwrite manual annotations on subsequent runs.
    with review_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'prompt_id', 'split', 'LLM_Input', 'suggested_domain', 'reviewed_domain'])
        writer.writeheader()
        writer.writerows(prompts.values())
    (args.output_dir / 'suggested_counts.json').write_text(
        json.dumps({'unique_prompts': len(prompts), 'response_counts': counts}, indent=2),
        encoding='utf-8')
    print(f'Exported {len(prompts):,} unique prompts to {review_path}')
    print('Suggestions are keyword heuristics. Review labels before building experiment splits.')


if __name__ == '__main__':
    main()
