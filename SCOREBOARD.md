# Benchmark Scoreboard

| Model | Dataset | Epoch | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|-------|---------|-------|---------------|----------------|-------------|----------------|-----------|
| bonsai | yso-en |  | 0.655159 | 0.734786 | 0.640733 | 0.731381 | 0.454515 |
| bonsai | yso-fi |  | 0.627302 | 0.707972 | 0.633398 | 0.718595 | 0.472298 |
| bonsai_gemma3 | koko |  | 0.383369 | 0.464903 | 0.314250 | 0.390276 | 0.229376 |
| bonsai_ovis2 | koko |  | 0.419245 | 0.499740 | 0.344294 | 0.423102 | 0.253293 |
| fasttext | yso-en |  | 0.429327 | 0.601442 | 0.458165 | 0.622390 | 0.317633 |
| fasttext | yso-fi |  | 0.475827 | 0.637037 | 0.425691 | 0.601266 | 0.306065 |
| mean(bonsai,fasttext,mllm) | yso-en |  | 0.452422 | 0.636637 | 0.372590 | 0.582518 | 0.253387 |
| mean(bonsai,fasttext,mllm) | yso-fi |  | 0.710079 | 0.811516 | 0.679141 | 0.795979 | 0.515742 |
| mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.421571 | 0.527285 | 0.351614 | 0.454947 | 0.257752 |
| mean_weighted(bonsai,fasttext,mllm) | yso-en |  | 0.654260 | 0.761685 | 0.639565 | 0.754048 | 0.452093 |
| mean_weighted(bonsai,fasttext,mllm) | yso-fi |  | 0.709956 | 0.812725 | 0.669945 | 0.791698 | 0.508171 |
| mean_weighted(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.432602 | 0.534406 | 0.358735 | 0.459425 | 0.262923 |
| mllm | koko |  | 0.162187 | 0.159443 | 0.154528 | 0.155313 | 0.110493 |
| mllm | yso-en |  | 0.000314 | 0.001211 | 0.000195 | 0.000689 | 0.000000 |
| mllm | yso-fi |  | 0.616455 | 0.669603 | 0.578451 | 0.706146 | 0.437427 |
| nn | koko |  |  |  | 0.374594 | 0.425015 | 0.276537 |
| nn | yso-en |  |  |  | 0.496716 | 0.615022 | 0.350647 |
| nn | yso-fi |  |  |  | 0.689080 | 0.768269 | 0.522291 |
| torch_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 20 | 0.873205 | 0.878189 | 0.631249 | 0.668056 | 0.459232 |
| torch_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 20 | 0.888047 | 0.908317 | 0.699446 | 0.741196 | 0.538196 |
| torch_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 10 | 0.507234 | 0.609628 | 0.358664 | 0.423520 | 0.264630 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-en | 19 | 0.717133 | 0.793257 | 0.653973 | 0.732769 | 0.465930 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756034 | 0.838082 | 0.696430 | 0.786967 | 0.529307 |
| torch_lowrank_residual_epsclamp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.454791 | 0.548610 | 0.364144 | 0.438432 | 0.267507 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-en | 15 | 0.711034 | 0.789757 | 0.652036 | 0.731976 | 0.464101 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756853 | 0.837086 | 0.694785 | 0.785086 | 0.527698 |
| torch_lowrank_residual_mix_temp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 13 | 0.465222 | 0.555876 | 0.363119 | 0.430717 | 0.268176 |
| torch_lowrank_residual_sigmoid(bonsai,fasttext,mllm) | yso-en | 2 | 0.608646 | 0.731768 | 0.552307 | 0.699277 | 0.392470 |
| torch_lowrank_residual_sigmoid(bonsai,fasttext,mllm) | yso-fi | 1 | 0.684933 | 0.795629 | 0.663998 | 0.784065 | 0.493791 |
| torch_lowrank_residual_sigmoid(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435170 | 0.548795 | 0.358693 | 0.468941 | 0.262387 |
| torch_mean(bonsai,fasttext,mllm) | yso-en | 11 | 0.653498 | 0.761288 | 0.638713 | 0.752942 | 0.452598 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 3 | 0.711586 | 0.812470 | 0.679727 | 0.796308 | 0.517343 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 6 | 0.430893 | 0.533464 | 0.358721 | 0.459573 | 0.263393 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-en | 12 | 0.688122 | 0.753923 | 0.656698 | 0.735480 | 0.469012 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710736 | 0.811807 | 0.687398 | 0.799336 | 0.521631 |
| torch_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435223 | 0.548767 | 0.357736 | 0.467187 | 0.261571 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 10 | 0.639140 | 0.746777 | 0.578776 | 0.707315 | 0.417140 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 5 | 0.724874 | 0.820597 | 0.697428 | 0.804396 | 0.536466 |
| torch_mean_residual_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439454 | 0.560167 | 0.362957 | 0.478706 | 0.267136 |
| torch_mean_residual_mlp(bonsai,fasttext,mllm) | yso-en | 2 | 0.569953 | 0.685693 | 0.502149 | 0.631101 | 0.356382 |
| torch_mean_residual_mlp(bonsai,fasttext,mllm) | yso-fi | 1 | 0.671643 | 0.764801 | 0.638591 | 0.743532 | 0.487439 |
| torch_mean_residual_mlp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.387847 | 0.496890 | 0.299059 | 0.399759 | 0.227024 |
| torch_nn(bonsai,fasttext,mllm) | yso-en | 20 | 0.850043 | 0.870760 | 0.574758 | 0.630995 | 0.403958 |
| torch_nn(bonsai,fasttext,mllm) | yso-fi | 20 | 0.835443 | 0.880353 | 0.679182 | 0.737592 | 0.520296 |
| torch_nn(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.659054 | 0.714089 | 0.373728 | 0.416170 | 0.275036 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-en | 20 | 0.620905 | 0.737203 | 0.588106 | 0.718240 | 0.424256 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-fi | 1 | 0.709384 | 0.810460 | 0.684051 | 0.796184 | 0.516068 |
| torch_nn_simple(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 15 | 0.429079 | 0.531054 | 0.358503 | 0.459624 | 0.262127 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-en | 12 | 0.715551 | 0.784838 | 0.557582 | 0.657785 | 0.397202 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-fi | 12 | 0.764292 | 0.839745 | 0.691875 | 0.768785 | 0.527237 |
| torch_nn_split(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.534741 | 0.629276 | 0.378951 | 0.444978 | 0.279118 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-en | 12 | 0.724809 | 0.790965 | 0.579959 | 0.672386 | 0.412873 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-fi | 12 | 0.764355 | 0.840039 | 0.691735 | 0.772526 | 0.527790 |
| torch_nn_split_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.538397 | 0.631761 | 0.379664 | 0.443365 | 0.278224 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 19 | 0.626856 | 0.741188 | 0.515139 | 0.671659 | 0.344941 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439253 | 0.555085 | 0.361643 | 0.473905 | 0.264727 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 20 | 0.619971 | 0.738825 | 0.522815 | 0.679542 | 0.355704 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Avg Test NDCG@10 |
|------|-------|----------------|
| 1 | torch_lowrank_residual_epsclamp | 0.571516 |
| 2 | torch_lowrank_residual_mix_temp | 0.569980 |
| 3 | torch_mean_residual | 0.567277 |
| 4 | torch_lowrank_mix | 0.563120 |
| 5 | torch_mean | 0.559054 |
| 6 | mean_weighted | 0.556082 |
| 7 | torch_nn_split_per_label | 0.550453 |
| 8 | torch_mean_residual_lowrank_mix | 0.546387 |
| 9 | torch_nn_simple | 0.543553 |
| 10 | torch_nn_split | 0.542803 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Avg Test NDCG@1000 |
|------|-------|----------------|
| 1 | torch_mean | 0.669608 |
| 2 | mean_weighted | 0.668390 |
| 3 | torch_mean_residual | 0.667334 |
| 4 | torch_mean_residual_lowrank_mix | 0.663472 |
| 5 | torch_nn_simple | 0.658016 |
| 6 | torch_per_label | 0.654006 |
| 7 | torch_per_label_l1_delta | 0.653453 |
| 8 | torch_lowrank_residual_epsclamp | 0.652723 |
| 9 | torch_lowrank_residual_sigmoid | 0.650761 |
| 10 | torch_lowrank_residual_mix_temp | 0.649260 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Avg Test F1@5 |
|------|-------|----------------|
| 1 | torch_lowrank_residual_epsclamp | 0.420915 |
| 2 | torch_lowrank_mix | 0.420686 |
| 3 | torch_lowrank_residual_mix_temp | 0.419992 |
| 4 | torch_mean_residual | 0.417405 |
| 5 | torch_mean | 0.411111 |
| 6 | mean_weighted | 0.407729 |
| 7 | torch_mean_residual_lowrank_mix | 0.406914 |
| 8 | torch_nn_split_per_label | 0.406296 |
| 9 | torch_nn_split | 0.401186 |
| 10 | torch_nn_simple | 0.400817 |

## Top 10 Models by Avg of 3 Test Metrics (across datasets)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) across datasets |
|------|-------|----------------|
| 1 | torch_mean_residual | 0.550672 |
| 2 | torch_lowrank_residual_epsclamp | 0.548384 |
| 3 | torch_mean | 0.546591 |
| 4 | torch_lowrank_residual_mix_temp | 0.546410 |
| 5 | mean_weighted | 0.544067 |
| 6 | torch_mean_residual_lowrank_mix | 0.538924 |
| 7 | torch_nn_simple | 0.534129 |
| 8 | torch_lowrank_mix | 0.531577 |
| 9 | torch_nn_split_per_label | 0.528725 |
| 10 | torch_nn_split | 0.522613 |

## Top 10 Models by Avg of 3 Test Metrics (koko)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_mean_residual_lowrank_mix | 0.369600 |
| 2 | torch_nn_split | 0.367682 |
| 3 | torch_per_label_l1_delta | 0.367199 |
| 4 | torch_nn_split_per_label | 0.367084 |
| 5 | torch_per_label | 0.366758 |
| 6 | torch_lowrank_residual_sigmoid | 0.363340 |
| 7 | torch_mean_residual | 0.362165 |
| 8 | torch_mean | 0.360562 |
| 9 | mean_weighted | 0.360361 |
| 10 | torch_nn_simple | 0.360085 |

## Top 10 Models by Avg of 3 Test Metrics (yso-en)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_mean_residual | 0.620397 |
| 2 | torch_lowrank_residual_epsclamp | 0.617557 |
| 3 | torch_lowrank_residual_mix_temp | 0.616038 |
| 4 | mean_weighted | 0.615235 |
| 5 | torch_mean | 0.614751 |
| 6 | torch_lowrank_mix | 0.586179 |
| 7 | torch_nn_simple | 0.576867 |
| 8 | torch_mean_residual_lowrank_mix | 0.567744 |
| 9 | torch_nn_split_per_label | 0.555073 |
| 10 | torch_lowrank_residual_sigmoid | 0.548018 |

## Top 10 Models by Avg of 3 Test Metrics (yso-fi)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_per_label | 0.690252 |
| 2 | torch_per_label_l1_delta | 0.680134 |
| 3 | torch_mean_residual_lowrank_mix | 0.679430 |
| 4 | torch_lowrank_residual_epsclamp | 0.670901 |
| 5 | torch_mean_residual | 0.669455 |
| 6 | torch_lowrank_residual_mix_temp | 0.669190 |
| 7 | torch_nn_simple | 0.665434 |
| 8 | torch_mean | 0.664459 |
| 9 | torch_nn_split_per_label | 0.664017 |
| 10 | mean | 0.663621 |
