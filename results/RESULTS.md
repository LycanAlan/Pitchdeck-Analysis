# PitchLens — retrieval ablation

_Generated 2026-08-29 04:29_

### Retrieval ablation (n=72 questions)

| retriever     | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 | precision@1 | precision@3 | precision@5 | precision@10 | ndcg@1 | ndcg@3 | ndcg@5 | ndcg@10 |   mrr | median_latency_ms | queries | errors | skipped_unanswerable |
| :------------ | ----: | ----: | ----: | -----: | -------: | -------: | -------: | --------: | ----------: | ----------: | ----------: | -----------: | -----: | -----: | -----: | ------: | ----: | ----------------: | ------: | -----: | -------------------: |
| hybrid+rerank | 0.862 | 0.969 | 0.985 |  0.985 |    0.831 |    0.954 |    0.969 |     0.977 |       0.862 |       0.338 |       0.206 |        0.124 |  0.862 |  0.918 |  0.924 |   0.928 | 0.917 |           725.194 |      65 |      0 |                    7 |
