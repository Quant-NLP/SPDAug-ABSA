# SPDAug-ABSA

## Requirements

`python >= 3.6.0`, Install all the requirements with pip.

```
$ pip install -r requirements.txt
```

We require [bert-sklearn](https://github.com/charles9n/bert-sklearn) and [seqeval](https://github.com/chakki-works/seqeval) for ATE.


## Getting Started

**ABSA_task** : ACSC, ATSC, ATE, SC

**dataset** : lap14, rest14, rest15, rest16, MR, SST

**SPM_type** : AS, Senti

**replacement_strategy** : AE, Seq2Seq

```
bash scripts/run.sh ABSA_task dataset SPM_type replacement_strategy

# e.g. bash scripts/run.sh ACSC rest14 AS AE
```

