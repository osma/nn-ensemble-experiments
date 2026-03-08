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
| torch_mean(bonsai,fasttext,mllm) | yso-en | 10 | 0.653554 | 0.761254 | 0.638497 | 0.752728 | 0.452598 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 3 | 0.711555 | 0.812537 | 0.679665 | 0.796472 | 0.517343 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 7 | 0.430841 | 0.533480 | 0.358801 | 0.459657 | 0.263474 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-en | 12 | 0.688122 | 0.753923 | 0.656698 | 0.735480 | 0.469012 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710736 | 0.811807 | 0.687398 | 0.799336 | 0.521631 |
| torch_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435223 | 0.548767 | 0.357736 | 0.467187 | 0.261571 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 19 | 0.626856 | 0.741188 | 0.515139 | 0.671659 | 0.344941 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439253 | 0.555085 | 0.361643 | 0.473905 | 0.264727 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 10 | 0.595045 | 0.724810 | 0.504414 | 0.669583 | 0.350857 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|------|-------|--------------|----------------|-----------|
| 1 | torch_mean_residual | 0.567277 | 0.667334 | 0.417405 |
| 2 | torch_mean | 0.558988 | 0.669619 | 0.411138 |
| 3 | mean_weighted | 0.556082 | 0.668390 | 0.407729 |
| 4 | torch_per_label | 0.528984 | 0.654006 | 0.384600 |
| 5 | torch_per_label_l1_delta | 0.521236 | 0.650133 | 0.384248 |
| 6 | nn | 0.520130 | 0.602769 | 0.383158 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Test NDCG@1000 | Test NDCG@10 | Test F1@5 |
|------|-------|----------------|--------------|-----------|
| 1 | torch_mean | 0.669619 | 0.558988 | 0.411138 |
| 2 | mean_weighted | 0.668390 | 0.556082 | 0.407729 |
| 3 | torch_mean_residual | 0.667334 | 0.567277 | 0.417405 |
| 4 | torch_per_label | 0.654006 | 0.528984 | 0.384600 |
| 5 | torch_per_label_l1_delta | 0.650133 | 0.521236 | 0.384248 |
| 6 | nn | 0.602769 | 0.520130 | 0.383158 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Test F1@5 | Test NDCG@10 | Test NDCG@1000 |
|------|-------|-----------|--------------|----------------|
| 1 | torch_mean_residual | 0.417405 | 0.567277 | 0.667334 |
| 2 | torch_mean | 0.411138 | 0.558988 | 0.669619 |
| 3 | mean_weighted | 0.407729 | 0.556082 | 0.668390 |
| 4 | torch_per_label | 0.384600 | 0.528984 | 0.654006 |
| 5 | torch_per_label_l1_delta | 0.384248 | 0.521236 | 0.650133 |
| 6 | nn | 0.383158 | 0.520130 | 0.602769 |
