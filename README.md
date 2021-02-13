# DCTA

Source code for "DCTA".

## Requirements

tqdm
torch
transformers
keras

### ATE

[bert-sklearn](https://github.com/charles9n/bert-sklearn)
[seqeval](https://github.com/chakki-works/seqeval)

## Getting Started

ABSA_task : ACSC, ATSC, ATE

dataset : lap14, rest14, rest15, rest16

DPM_type : AS, Senti

replacement_strategy : AE, Seq2Seq

```
bash scripts/run.sh ABSA_task dataset DPM_type replacement_strategy

# e.g. bash scripts/run.sh ACSC rest14 AS AE
```

