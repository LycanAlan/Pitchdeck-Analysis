# PitchLens — measurement suite

_Generated 2026-08-29 19:57_

### Extraction coverage

| extractor                | decks_usable | pct_decks | pages_readable |
| :----------------------- | :----------- | --------: | -------------: |
| text-only                | 4/19         |    21.100 |            153 |
| adaptive (text + vision) | 19/19        |   100.000 |            440 |

_measured with pymupdf only -- no API key, no model calls_

### Per-deck text layer

| deck              | pages | text_chars | chars_per_page | text_pages | page_coverage | class      |
| :---------------- | ----: | ---------: | -------------: | ---------: | ------------: | :--------- |
| airbnb            |    18 |          0 |          0.000 |          0 |         0.000 | image_only |
| Brex              |    19 |          0 |          0.000 |          0 |         0.000 | image_only |
| coinbase          |     8 |          0 |          0.000 |          0 |         0.000 | image_only |
| dropbox           |    22 |          0 |          0.000 |          0 |         0.000 | image_only |
| Dwolla            |    18 |          0 |          0.000 |          0 |         0.000 | image_only |
| facebook          |    28 |          0 |          0.000 |          0 |         0.000 | image_only |
| frontapp          |    21 |          0 |          0.000 |          0 |         0.000 | image_only |
| linkedin_series_b |    37 |          0 |          0.000 |          0 |         0.000 | image_only |
| mixpanel          |    12 |          0 |          0.000 |          0 |         0.000 | image_only |
| Monzo             |    16 |          0 |          0.000 |          0 |         0.000 | image_only |
| Revolut           |    12 |          0 |          0.000 |          0 |         0.000 | image_only |
| square            |    20 |          0 |          0.000 |          0 |         0.000 | image_only |
| Transferwise      |    11 |          0 |          0.000 |          0 |         0.000 | image_only |
| uber              |    13 |          0 |          0.000 |          0 |         0.000 | image_only |
| Oscar_Health      |    33 |       1505 |         45.600 |         10 |         0.303 | image_only |
| Nium_replica      |    22 |       7573 |        344.200 |         22 |         1.000 | text_layer |
| sensovision       |    27 |      12490 |        462.600 |         23 |         0.852 | text_layer |
| shopify_2025      |    53 |      28545 |        538.600 |         49 |         0.925 | text_layer |
| wework_2021       |    50 |      63178 |       1263.600 |         49 |         0.980 | text_layer |

### Retrieval ablation (n=72 questions)

| retriever     | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 | precision@1 | precision@3 | precision@5 | precision@10 | ndcg@1 | ndcg@3 | ndcg@5 | ndcg@10 |   mrr | median_latency_ms | queries | errors | skipped_unanswerable |
| :------------ | ----: | ----: | ----: | -----: | -------: | -------: | -------: | --------: | ----------: | ----------: | ----------: | -----------: | -----: | -----: | -----: | ------: | ----: | ----------------: | ------: | -----: | -------------------: |
| dense         | 0.631 | 0.800 | 0.862 |  0.923 |    0.623 |    0.777 |    0.846 |     0.908 |       0.631 |       0.272 |       0.182 |        0.115 |  0.631 |  0.719 |  0.750 |   0.771 | 0.731 |            22.148 |      65 |      0 |                    7 |
| bm25          | 0.723 | 0.908 | 0.938 |  0.969 |    0.700 |    0.869 |    0.923 |     0.954 |       0.723 |       0.303 |       0.197 |        0.112 |  0.723 |  0.805 |  0.829 |   0.840 | 0.814 |             0.901 |      65 |      0 |                    7 |
| hybrid        | 0.831 | 0.923 | 0.938 |  0.985 |    0.800 |    0.892 |    0.931 |     0.977 |       0.831 |       0.313 |       0.200 |        0.122 |  0.831 |  0.862 |  0.879 |   0.896 | 0.882 |            23.819 |      65 |      0 |                    7 |
| dense+rerank  | 0.862 | 0.923 | 0.938 |  0.938 |    0.823 |    0.908 |    0.923 |     0.931 |       0.862 |       0.323 |       0.197 |        0.119 |  0.862 |  0.886 |  0.893 |   0.896 | 0.894 |           803.861 |      65 |      0 |                    7 |
| hybrid+rerank | 0.862 | 0.969 | 0.985 |  0.985 |    0.831 |    0.954 |    0.969 |     0.977 |       0.862 |       0.338 |       0.206 |        0.125 |  0.862 |  0.918 |  0.924 |   0.928 | 0.917 |           811.998 |      65 |      0 |                    7 |

### Index persistence (cold build vs warm load)

| chunks | cold_build_s | warm_load_s | speedup |
| -----: | -----------: | ----------: | :------ |
|    286 |       13.670 |       0.032 | 426x    |

### Query latency by retrieval mode

| retriever     |  p50_ms |   p95_ms | queries |
| :------------ | ------: | -------: | ------: |
| dense         |  22.200 |   33.700 |      72 |
| bm25          |   0.900 |    1.500 |      72 |
| hybrid        |  23.900 |   45.300 |      72 |
| dense+rerank  | 814.600 | 2093.700 |      72 |
| hybrid+rerank | 901.000 | 2259.400 |      72 |
