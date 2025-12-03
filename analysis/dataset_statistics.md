# EDA: ArchitRastogi/it-retrieval-triplets-mc4

## File: train
- Path: `it_mc4_eda_fast/data/train.jsonl`
- Total lines: **8,591,070** | Bad JSON: **0**
- Missing fields: {'query': 0, 'positive': 0, 'negative': 0}

**query**
- chars: mean=96.5, p50=125, p90=129, p95=129, p99=129
- tokens: mean=15.1, p50=17, p90=22, p95=24, p99=26

**positive**
- chars: mean=314.2, p50=318, p90=473, p95=493, p99=509
- tokens: mean=47.4, p50=47, p90=73, p95=78, p99=85

**negative**
- chars: mean=285.5, p50=275, p90=464, p95=488, p99=508
- tokens: mean=43.0, p50=41, p90=71, p95=76, p99=84

## File: train_random_neg
- Path: `it_mc4_eda_fast/data/train_random_neg.jsonl`
- Total lines: **16,847,571** | Bad JSON: **0**
- Missing fields: {'query': 0, 'positive': 0, 'negative': 0}

**query**
- chars: mean=105.2, p50=128, p90=129, p95=129, p99=129
- tokens: mean=16.5, p50=18, p90=23, p95=24, p99=26

**positive**
- chars: mean=299.6, p50=297, p90=466, p95=489, p99=508
- tokens: mean=45.2, p50=44, p90=71, p95=77, p99=85

**negative**
- chars: mean=299.6, p50=297, p90=466, p95=489, p99=508
- tokens: mean=45.2, p50=44, p90=71, p95=77, p99=85


---
## Takeaway: Are queries shorter than passages?

- **train**: queries are **shorter** than positives (median tokens: query=17, positive=47).
- **train_random_neg**: queries are **shorter** than positives (median tokens: query=18, positive=44).