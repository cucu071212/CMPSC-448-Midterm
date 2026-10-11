# RQ3: Generalization across task domains

RQ3 작업은 이 폴더에서 진행합니다. 루트의 기존 코드, `data/`, `results/`는 수정하지 않습니다.

## 폴더 구성

```text
rq3/
  prepare_domains.py       # 기존 CSV를 읽어 prompt별 도메인 검토 파일 생성
  data/                   # 도메인 검토 파일과 이후 RQ3 전용 split
  results/                # RQ3 전용 학습 결과
```

## 1. 도메인 후보 확인

프로젝트 루트에서 실행합니다. Python 표준 라이브러리만 사용합니다.

```powershell
py rq3/prepare_domains.py
```

`data/domain_review.csv`에 고유 prompt마다 한 행을 저장합니다. 도메인은 prompt만으로 추정하며, 같은 prompt의 모든 응답에는 같은 도메인을 적용합니다. `suggested_counts.json`은 기존 split, 후보 도메인, LLM family별 **응답 수**입니다.

키워드 분류는 초기 후보입니다. `reviewed_domain`에 검토한 라벨을 입력하고, ambiguous/other를 무조건 non-coding으로 간주하지 마세요. 검토 파일은 재실행 시 덮어쓰지 않습니다. 새 후보 파일이 필요하면 다른 `--output-dir`을 지정하세요.

## 2. 실험 설계

검토 후 도메인별 세 family의 표본 수를 확인하고 Coding vs Non-coding 또는 여러 도메인으로 범위를 결정합니다. 아직 실험용 split 생성이나 학습은 실행하지 않았습니다.

각 target domain에 대해 in-domain과 cross-domain 학습 조건을 만들고 동일한 target-domain test set에서 비교합니다. 기존 prompt별 train/validation/test 소속을 유지하고, cross-domain validation에는 target domain을 포함하지 않습니다. 학습 데이터 수와 family별 비율을 맞추고, vocabulary는 각 조건의 train에서만 만듭니다.

## 3. 기존 학습 코드 재사용

검토한 라벨로 실험용 CSV를 생성한 뒤 아래처럼 실행합니다. 아래 경로는 향후 만들 split의 예시이며 아직 생성되지 않았습니다.

```powershell
py train.py --model cnn --mode output --data-dir rq3/data/coding/cross_domain --output-dir rq3/results/coding/cross_domain
py train.py --model cnn --mode output --data-dir rq3/data/coding/in_domain --output-dir rq3/results/coding/in_domain
```

학습에는 루트 `requirements.txt`의 의존성이 필요합니다. 기존 `train.py`를 재사용하므로 CNN/RNN 구현을 복사할 필요가 없습니다. 주 비교 지표는 macro-F1과 `in-domain F1 - cross-domain F1`입니다.
