# Benchmark Scoreboard

| Model | Dataset | Epoch | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|-------|---------|-------|---------------|----------------|-------------|----------------|-----------|
| bonsai | yso-en |  | 0.655159 | 0.734786 | 0.640733 | 0.731381 | 0.454515 |
| bonsai | yso-fi |  | 0.627302 | 0.707972 | 0.633398 | 0.718595 | 0.472298 |
| bonsai_gemma3 | koko |  | 0.383369 | 0.464903 | 0.314250 | 0.390276 | 0.229376 |
| bonsai_ovis2 | koko |  | 0.419245 | 0.499740 | 0.344294 | 0.423102 | 0.253293 |
| fasttext | yso-en |  | 0.429327 | 0.601442 | 0.458165 | 0.622390 | 0.317633 |
| fasttext | yso-fi |  | 0.475827 | 0.637037 | 0.425691 | 0.601266 | 0.306065 |
| mean(bonsai,fasttext,mllm) | yso-en |  | 0.684010 | 0.794765 | 0.600714 | 0.738365 | 0.422840 |
| mean(bonsai,fasttext,mllm) | yso-fi |  | 0.710079 | 0.811516 | 0.679141 | 0.795979 | 0.515742 |
| mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.421571 | 0.527285 | 0.351614 | 0.454947 | 0.257752 |
| mean_weighted(bonsai,fasttext,mllm) | yso-en |  | 0.687386 | 0.795056 | 0.614547 | 0.745857 | 0.431815 |
| mean_weighted(bonsai,fasttext,mllm) | yso-fi |  | 0.709956 | 0.812725 | 0.669945 | 0.791698 | 0.508171 |
| mean_weighted(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.432602 | 0.534406 | 0.358735 | 0.459425 | 0.262923 |
| mllm | koko |  | 0.162187 | 0.159443 | 0.154528 | 0.155313 | 0.110493 |
| mllm | yso-en |  | 0.534782 | 0.581381 | 0.312900 | 0.358993 | 0.221332 |
| mllm | yso-fi |  | 0.616455 | 0.669603 | 0.578451 | 0.706146 | 0.437427 |
| nn | koko |  |  |  | 0.374594 | 0.425015 | 0.276537 |
| nn | yso-en |  |  |  | 0.643143 | 0.712582 | 0.458153 |
| nn | yso-fi |  |  |  | 0.689080 | 0.768269 | 0.522291 |
| torch_3stage(bonsai,fasttext,mllm) | yso-en | 5 | 0.665094 | 0.776431 | 0.622553 | 0.747911 | 0.437849 |
| torch_3stage(bonsai,fasttext,mllm) | yso-fi | 3 | 0.680906 | 0.792897 | 0.671315 | 0.787147 | 0.504351 |
| torch_3stage(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.401366 | 0.506620 | 0.338846 | 0.441537 | 0.246241 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-en | 20 | 0.748086 | 0.824475 | 0.654614 | 0.736405 | 0.461863 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756034 | 0.838082 | 0.696430 | 0.786967 | 0.529307 |
| torch_lowrank_residual_epsclamp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.454791 | 0.548610 | 0.364144 | 0.438432 | 0.267507 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-en | 20 | 0.747455 | 0.823615 | 0.655750 | 0.738232 | 0.463115 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756853 | 0.837086 | 0.694785 | 0.785086 | 0.527698 |
| torch_lowrank_residual_mix_temp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 13 | 0.465222 | 0.555876 | 0.363119 | 0.430717 | 0.268176 |
| torch_mean(bonsai,fasttext,mllm) | yso-en | 1 | 0.684793 | 0.795369 | 0.601542 | 0.738840 | 0.421134 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 3 | 0.711586 | 0.812470 | 0.679727 | 0.796308 | 0.517343 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 6 | 0.430893 | 0.533464 | 0.358721 | 0.459573 | 0.263393 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-en | 3 | 0.702887 | 0.801773 | 0.634044 | 0.757152 | 0.447385 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710736 | 0.811807 | 0.687398 | 0.799336 | 0.521631 |
| torch_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435223 | 0.548767 | 0.357736 | 0.467187 | 0.261571 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 5 | 0.703847 | 0.804152 | 0.640272 | 0.762450 | 0.455739 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 5 | 0.724906 | 0.820559 | 0.697428 | 0.804458 | 0.536466 |
| torch_mean_residual_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.438590 | 0.559694 | 0.361132 | 0.474147 | 0.266729 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-en | 11 | 0.681651 | 0.792474 | 0.598897 | 0.737826 | 0.416735 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-fi | 1 | 0.709384 | 0.810460 | 0.684051 | 0.796184 | 0.516068 |
| torch_nn_simple(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 15 | 0.429079 | 0.531054 | 0.358503 | 0.459624 | 0.262127 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-en | 12 | 0.746666 | 0.824106 | 0.617811 | 0.714514 | 0.432653 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-fi | 12 | 0.764292 | 0.839745 | 0.691875 | 0.768785 | 0.527237 |
| torch_nn_split(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.534741 | 0.629276 | 0.378951 | 0.444978 | 0.279118 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-en | 12 | 0.790678 | 0.854107 | 0.661751 | 0.743532 | 0.463044 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-fi | 12 | 0.785900 | 0.860110 | 0.709825 | 0.785869 | 0.539094 |
| torch_nn_split_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.551779 | 0.644473 | 0.381411 | 0.447886 | 0.280445 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 15 | 0.724006 | 0.815469 | 0.659227 | 0.771079 | 0.473627 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439253 | 0.555085 | 0.361643 | 0.473905 | 0.264727 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 13 | 0.714631 | 0.811372 | 0.648720 | 0.765062 | 0.463588 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |
| torch_per_label_mlp(bonsai,fasttext,mllm) | yso-en | 1 | 0.722371 | 0.814921 | 0.650776 | 0.764912 | 0.467200 |
| torch_per_label_mlp(bonsai,fasttext,mllm) | yso-fi | 6 | 0.728348 | 0.824308 | 0.697123 | 0.806150 | 0.534772 |
| torch_per_label_mlp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.442098 | 0.559631 | 0.361286 | 0.474052 | 0.266288 |
| torch_per_label_residual_lowrank_mix_active(bonsai,fasttext,mllm) | yso-en | 1 | 0.722495 | 0.808272 | 0.651005 | 0.758634 | 0.466942 |
| torch_per_label_residual_lowrank_mix_active(bonsai,fasttext,mllm) | yso-fi | 1 | 0.728239 | 0.822888 | 0.697189 | 0.804671 | 0.534990 |
| torch_per_label_residual_lowrank_mix_active(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.442085 | 0.528383 | 0.361227 | 0.444598 | 0.266173 |
| torch_reg_mean_residual(bonsai,fasttext,mllm) | yso-en | 1 | 0.689306 | 0.781867 | 0.611709 | 0.708791 | 0.434239 |
| torch_reg_mean_residual(bonsai,fasttext,mllm) | yso-fi | 1 | 0.691964 | 0.772237 | 0.653525 | 0.736514 | 0.495014 |
| torch_reg_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.416510 | 0.510249 | 0.329807 | 0.409213 | 0.247815 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Avg Test NDCG@10 |
|------|-------|----------------|
| 1 | torch_nn_split_per_label | 0.584329 |
| 2 | torch_per_label | 0.577014 |
| 3 | torch_lowrank_residual_epsclamp | 0.571729 |
| 4 | torch_lowrank_residual_mix_temp | 0.571218 |
| 5 | torch_per_label_residual_lowrank_mix_active | 0.569807 |
| 6 | torch_per_label_mlp | 0.569728 |
| 7 | torch_per_label_l1_delta | 0.569338 |
| 8 | nn | 0.568939 |
| 9 | torch_mean_residual_lowrank_mix | 0.566277 |
| 10 | torch_nn_split | 0.562879 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Avg Test NDCG@1000 |
|------|-------|----------------|
| 1 | torch_per_label | 0.687146 |
| 2 | torch_per_label_l1_delta | 0.681959 |
| 3 | torch_per_label_mlp | 0.681705 |
| 4 | torch_mean_residual_lowrank_mix | 0.680352 |
| 5 | torch_mean_residual | 0.674558 |
| 6 | torch_per_label_residual_lowrank_mix_active | 0.669301 |
| 7 | mean_weighted | 0.665660 |
| 8 | torch_mean | 0.664907 |
| 9 | torch_nn_simple | 0.664545 |
| 10 | mean | 0.663097 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Avg Test F1@5 |
|------|-------|----------------|
| 1 | torch_nn_split_per_label | 0.427528 |
| 2 | torch_per_label | 0.427495 |
| 3 | torch_per_label_mlp | 0.422753 |
| 4 | torch_per_label_residual_lowrank_mix_active | 0.422702 |
| 5 | torch_per_label_l1_delta | 0.421825 |
| 6 | torch_lowrank_residual_mix_temp | 0.419663 |
| 7 | torch_mean_residual_lowrank_mix | 0.419645 |
| 8 | torch_lowrank_residual_epsclamp | 0.419559 |
| 9 | nn | 0.418994 |
| 10 | torch_nn_split | 0.413003 |

## Top 10 Models by Avg of 3 Test Metrics (across datasets)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) across datasets |
|------|-------|----------------|
| 1 | torch_per_label | 0.563885 |
| 2 | torch_per_label_mlp | 0.558062 |
| 3 | torch_per_label_l1_delta | 0.557707 |
| 4 | torch_nn_split_per_label | 0.556984 |
| 5 | torch_mean_residual_lowrank_mix | 0.555425 |
| 6 | torch_per_label_residual_lowrank_mix_active | 0.553937 |
| 7 | torch_lowrank_residual_epsclamp | 0.548408 |
| 8 | torch_mean_residual | 0.548160 |
| 9 | torch_lowrank_residual_mix_temp | 0.547409 |
| 10 | nn | 0.541074 |

## Top 10 Models by Avg of 3 Test Metrics (koko)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_nn_split_per_label | 0.369914 |
| 2 | torch_nn_split | 0.367682 |
| 3 | torch_mean_residual_lowrank_mix | 0.367336 |
| 4 | torch_per_label_mlp | 0.367209 |
| 5 | torch_per_label_l1_delta | 0.367199 |
| 6 | torch_per_label | 0.366758 |
| 7 | torch_mean_residual | 0.362165 |
| 8 | torch_mean | 0.360562 |
| 9 | mean_weighted | 0.360361 |
| 10 | torch_nn_simple | 0.360085 |

## Top 10 Models by Avg of 3 Test Metrics (yso-en)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_per_label | 0.634644 |
| 2 | torch_per_label_mlp | 0.627629 |
| 3 | torch_per_label_l1_delta | 0.625790 |
| 4 | torch_per_label_residual_lowrank_mix_active | 0.625527 |
| 5 | torch_nn_split_per_label | 0.622776 |
| 6 | torch_mean_residual_lowrank_mix | 0.619487 |
| 7 | torch_lowrank_residual_mix_temp | 0.619032 |
| 8 | torch_lowrank_residual_epsclamp | 0.617627 |
| 9 | torch_mean_residual | 0.612860 |
| 10 | nn | 0.604626 |

## Top 10 Models by Avg of 3 Test Metrics (yso-fi)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_per_label | 0.690252 |
| 2 | torch_per_label_l1_delta | 0.680134 |
| 3 | torch_mean_residual_lowrank_mix | 0.679451 |
| 4 | torch_per_label_mlp | 0.679348 |
| 5 | torch_per_label_residual_lowrank_mix_active | 0.678950 |
| 6 | torch_nn_split_per_label | 0.678263 |
| 7 | torch_lowrank_residual_epsclamp | 0.670901 |
| 8 | torch_mean_residual | 0.669455 |
| 9 | torch_lowrank_residual_mix_temp | 0.669190 |
| 10 | torch_nn_simple | 0.665434 |
