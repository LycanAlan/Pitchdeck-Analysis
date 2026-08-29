# PitchLens — measurement suite

_Generated 2026-08-30 02:49_

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
| dense         | 0.600 | 0.785 | 0.862 |  0.954 |    0.585 |    0.762 |    0.846 |     0.938 |       0.600 |       0.272 |       0.182 |        0.117 |  0.600 |  0.698 |  0.733 |   0.762 | 0.713 |           179.472 |      65 |      0 |                    7 |
| bm25          | 0.692 | 0.831 | 0.908 |  0.908 |    0.662 |    0.800 |    0.892 |     0.892 |       0.692 |       0.282 |       0.191 |        0.103 |  0.692 |  0.752 |  0.791 |   0.791 | 0.772 |             5.826 |      65 |      0 |                    7 |
| hybrid        | 0.723 | 0.877 | 0.892 |  0.969 |    0.692 |    0.854 |    0.885 |     0.962 |       0.723 |       0.303 |       0.191 |        0.117 |  0.723 |  0.797 |  0.811 |   0.836 | 0.806 |           191.304 |      65 |      0 |                    7 |
| dense+rerank  | 0.862 | 0.938 | 0.954 |  0.954 |    0.823 |    0.923 |    0.946 |     0.946 |       0.862 |       0.328 |       0.203 |        0.120 |  0.862 |  0.896 |  0.906 |   0.906 | 0.901 |          3596.483 |      65 |      0 |                    7 |
| hybrid+rerank | 0.862 | 0.954 | 0.954 |  0.969 |    0.823 |    0.938 |    0.938 |     0.962 |       0.862 |       0.333 |       0.200 |        0.119 |  0.862 |  0.904 |  0.904 |   0.913 | 0.905 |          3507.661 |      65 |      0 |                    7 |

### Index persistence (cold build vs warm load)

| chunks | cold_build_s | warm_load_s | speedup |
| -----: | -----------: | ----------: | :------ |
|    730 |       14.090 |       0.029 | 494x    |

### Query latency by retrieval mode

| retriever     |   p50_ms |   p95_ms | queries |
| :------------ | -------: | -------: | ------: |
| dense         |   13.200 |   48.600 |      72 |
| bm25          |    7.200 |   16.200 |      72 |
| hybrid        |  199.400 |  270.500 |      72 |
| dense+rerank  | 3469.600 | 7302.700 |      72 |
| hybrid+rerank | 3611.200 | 7633.800 |      72 |
