# Quick look at the processed Arena data (lengths, formatting habits, examples).
# Run in Colab after mounting Drive:  !python scripts/explore_data.py

import re
import pandas as pd

DATA_DIR = "/content/drive/MyDrive/cmpsc-448-midterm/arena_processed"
LABELS = {"GPT": 0, "Claude": 1, "Gemini": 2}
TOKEN_RE = re.compile(r"\w+|[^\w\s]")   # rough token count: words + punctuation

# keep_default_na=False so a response like "NA" stays text, not NaN
splits = {s: pd.read_csv(f"{DATA_DIR}/{s}.csv", keep_default_na=False)
          for s in ["train", "validation", "test"]}

for name, df in splits.items():
    print(name, len(df), df["LLM_name"].value_counts().to_dict())
    assert (df["LLM_name"].map(LABELS) == df["label"]).all()

ids = {s: set(df["prompt_id"]) for s, df in splits.items()}
print("prompt overlap:", len(ids["train"] & ids["validation"]),
      len(ids["train"] & ids["test"]), len(ids["validation"] & ids["test"]))

train = splits["train"]
train["n_tok"] = train["LLM_output"].map(lambda s: len(TOKEN_RE.findall(s)))

print("\nresponse length (approx tokens)")
print(train.groupby("LLM_name")["n_tok"].describe(percentiles=[.5, .75, .9]).round(0))

for max_len in [256, 512, 1024]:
    print(f"fits in {max_len}: {(train['n_tok'] <= max_len).mean():.1%}")

# % of responses using each formatting feature
text = train["LLM_output"]
fmt = pd.DataFrame({
    "bold":   text.str.contains(r"\*\*"),
    "header": text.str.contains(r"(?m)^#{1,6}\s"),
    "list":   text.str.contains(r"(?m)^\s*(?:[-*•]|\d+\.)\s"),
    "code":   text.str.contains("```", regex=False),
    "emoji":  text.str.contains(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"),
    "emdash": text.str.contains("—", regex=False),
})
print("\nformatting (% of responses)")
print((fmt.groupby(train["LLM_name"]).mean() * 100).round(1))

print("\nmost common openings")
openings = text.str.split().str[:2].str.join(" ")
for fam in LABELS:
    print(fam, openings[train["LLM_name"] == fam].value_counts().head(5).to_dict())

print("\nexamples")
for fam in LABELS:
    t = train[train["LLM_name"] == fam].sample(1, random_state=0)["LLM_output"].iloc[0]
    print(f"[{fam}] {t[:300]!r}\n")
