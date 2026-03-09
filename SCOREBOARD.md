# Benchmark Scoreboard

| Model | Dataset | Epoch | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|-------|---------|-------|---------------|----------------|-------------|----------------|-----------|
| bonsai | yso-en |  | 0.655159 | 0.734786 | 0.640733 | 0.731381 | 0.454515 |
| bonsai | yso-fi |  | 0.627302 | 0.707972 | 0.633398 | 0.718595 | 0.472298 |
| bonsai_gemma3 | koko |  | 0.383369 | 0.464903 | 0.314250 | 0.390276 | 0.229376 |
| bonsai_ovis2 | koko |  | 0.419245 | 0.499740 | 0.344294 | 0.423102 | 0.253293 |
| fasttext | yso-en |  | 0.429327 | 0.601442 | 0.458165 | 0.622390 | 0.317633 |
| fasttext | yso-fi |  | 0.475827 | 0.637037 | 0.425691 | 0.601266 | 0.306065 |
| mean_weighted(bonsai,fasttext,mllm) | yso-en |  | 0.654260 | 0.761685 | 0.639565 | 0.754048 | 0.452093 |
| mean_weighted(bonsai,fasttext,mllm) | yso-fi |  | 0.709956 | 0.812725 | 0.669945 | 0.791698 | 0.508171 |
| mean_weighted(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.432602 | 0.534406 | 0.358735 | 0.459425 | 0.262923 |
| mllm | koko |  | 0.162187 | 0.159443 | 0.154528 | 0.155313 | 0.110493 |
| mllm | yso-en |  | 0.000314 | 0.001211 | 0.000195 | 0.000689 | 0.000000 |
| mllm | yso-fi |  | 0.616455 | 0.669603 | 0.578451 | 0.706146 | 0.437427 |
| nn | koko |  |  |  | 0.374594 | 0.425015 | 0.276537 |
| nn | yso-en |  |  |  | 0.496716 | 0.615022 | 0.350647 |
| nn | yso-fi |  |  |  | 0.689080 | 0.768269 | 0.522291 |
| torch_mean(bonsai,fasttext,mllm) | yso-en | 8 | 0.653498 | 0.761266 | 0.638713 | 0.752943 | 0.452598 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 2 | 0.711559 | 0.812470 | 0.679665 | 0.796320 | 0.517343 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 6 | 0.429499 | 0.532532 | 0.357888 | 0.459138 | 0.262848 |
| torch_mean_bias(bonsai,fasttext,mllm) | yso-en | 12 | 0.674246 | 0.746445 | 0.624893 | 0.694130 | 0.444747 |
| torch_mean_bias(bonsai,fasttext,mllm) | yso-fi | 18 | 0.734635 | 0.807127 | 0.681998 | 0.763381 | 0.518083 |
| torch_mean_bias(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 5 | 0.433004 | 0.497867 | 0.344974 | 0.393518 | 0.255928 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-en | 12 | 0.688122 | 0.753923 | 0.656698 | 0.735480 | 0.469012 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710736 | 0.811807 | 0.687398 | 0.799336 | 0.521631 |
| torch_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435223 | 0.548767 | 0.357736 | 0.467187 | 0.261571 |
| torch_nn(bonsai,fasttext,mllm) | yso-en | 20 | 0.850619 | 0.871051 | 0.604233 | 0.654747 | 0.434140 |
| torch_nn(bonsai,fasttext,mllm) | yso-fi | 20 | 0.836512 | 0.879323 | 0.691390 | 0.743584 | 0.531532 |
| torch_nn(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.658696 | 0.712685 | 0.382478 | 0.423285 | 0.282331 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 19 | 0.626856 | 0.741188 | 0.515139 | 0.671659 | 0.344941 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439253 | 0.555085 | 0.361643 | 0.473905 | 0.264727 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 20 | 0.619971 | 0.738825 | 0.522815 | 0.679542 | 0.355704 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|------|-------|--------------|----------------|-----------|
| 1 | torch_mean_residual | 0.567277 | 0.667334 | 0.417405 |
| 2 | torch_nn | 0.559367 | 0.607205 | 0.416001 |
| 3 | torch_mean | 0.558755 | 0.669467 | 0.410930 |
| 4 | mean_weighted | 0.556082 | 0.668390 | 0.407729 |
| 5 | torch_mean_bias | 0.550622 | 0.617010 | 0.406253 |
| 6 | torch_per_label | 0.528984 | 0.654006 | 0.384600 |
| 7 | torch_per_label_l1_delta | 0.527370 | 0.653453 | 0.385864 |
| 8 | nn | 0.520130 | 0.602769 | 0.383158 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Test NDCG@1000 | Test NDCG@10 | Test F1@5 |
|------|-------|----------------|--------------|-----------|
| 1 | torch_mean | 0.669467 | 0.558755 | 0.410930 |
| 2 | mean_weighted | 0.668390 | 0.556082 | 0.407729 |
| 3 | torch_mean_residual | 0.667334 | 0.567277 | 0.417405 |
| 4 | torch_per_label | 0.654006 | 0.528984 | 0.384600 |
| 5 | torch_per_label_l1_delta | 0.653453 | 0.527370 | 0.385864 |
| 6 | torch_mean_bias | 0.617010 | 0.550622 | 0.406253 |
| 7 | torch_nn | 0.607205 | 0.559367 | 0.416001 |
| 8 | nn | 0.602769 | 0.520130 | 0.383158 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Test F1@5 | Test NDCG@10 | Test NDCG@1000 |
|------|-------|-----------|--------------|----------------|
| 1 | torch_mean_residual | 0.417405 | 0.567277 | 0.667334 |
| 2 | torch_nn | 0.416001 | 0.559367 | 0.607205 |
| 3 | torch_mean | 0.410930 | 0.558755 | 0.669467 |
| 4 | mean_weighted | 0.407729 | 0.556082 | 0.668390 |
| 5 | torch_mean_bias | 0.406253 | 0.550622 | 0.617010 |
| 6 | torch_per_label_l1_delta | 0.385864 | 0.527370 | 0.653453 |
| 7 | torch_per_label | 0.384600 | 0.528984 | 0.654006 |
| 8 | nn | 0.383158 | 0.520130 | 0.602769 |
