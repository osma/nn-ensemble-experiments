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
| torch_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 20 | 0.894743 | 0.905112 | 0.639037 | 0.676570 | 0.464928 |
| torch_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 20 | 0.888047 | 0.908317 | 0.699446 | 0.741196 | 0.538196 |
| torch_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 10 | 0.507234 | 0.609628 | 0.358664 | 0.423520 | 0.264630 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-en | 20 | 0.748086 | 0.824475 | 0.654614 | 0.736405 | 0.461863 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756034 | 0.838082 | 0.696430 | 0.786967 | 0.529307 |
| torch_lowrank_residual_epsclamp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.454791 | 0.548610 | 0.364144 | 0.438432 | 0.267507 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-en | 20 | 0.747455 | 0.823615 | 0.655750 | 0.738232 | 0.463115 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756853 | 0.837086 | 0.694785 | 0.785086 | 0.527698 |
| torch_lowrank_residual_mix_temp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 13 | 0.465222 | 0.555876 | 0.363119 | 0.430717 | 0.268176 |
| torch_lowrank_residual_sigmoid(bonsai,fasttext,mllm) | yso-en | 2 | 0.698533 | 0.797882 | 0.638850 | 0.758441 | 0.450171 |
| torch_lowrank_residual_sigmoid(bonsai,fasttext,mllm) | yso-fi | 1 | 0.684933 | 0.795629 | 0.663998 | 0.784065 | 0.493791 |
| torch_lowrank_residual_sigmoid(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435170 | 0.548795 | 0.358693 | 0.468941 | 0.262387 |
| torch_mean(bonsai,fasttext,mllm) | yso-en | 1 | 0.684793 | 0.795369 | 0.601542 | 0.738840 | 0.421134 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 3 | 0.711586 | 0.812470 | 0.679727 | 0.796308 | 0.517343 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 6 | 0.430893 | 0.533464 | 0.358721 | 0.459573 | 0.263393 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-en | 3 | 0.702887 | 0.801773 | 0.634044 | 0.757152 | 0.447385 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710736 | 0.811807 | 0.687398 | 0.799336 | 0.521631 |
| torch_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435223 | 0.548767 | 0.357736 | 0.467187 | 0.261571 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 5 | 0.703847 | 0.804152 | 0.640272 | 0.762450 | 0.455739 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 5 | 0.724906 | 0.820559 | 0.697428 | 0.804458 | 0.536466 |
| torch_mean_residual_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.438590 | 0.559694 | 0.361132 | 0.474147 | 0.266729 |
| torch_mean_residual_mlp(bonsai,fasttext,mllm) | yso-en | 1 | 0.686870 | 0.782217 | 0.609227 | 0.718428 | 0.431877 |
| torch_mean_residual_mlp(bonsai,fasttext,mllm) | yso-fi | 1 | 0.671643 | 0.764801 | 0.638591 | 0.743532 | 0.487439 |
| torch_mean_residual_mlp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.387847 | 0.496890 | 0.299059 | 0.399759 | 0.227024 |
| torch_nn(bonsai,fasttext,mllm) | yso-en | 20 | 0.830137 | 0.873855 | 0.619329 | 0.688294 | 0.439753 |
| torch_nn(bonsai,fasttext,mllm) | yso-fi | 20 | 0.835443 | 0.880353 | 0.679182 | 0.737592 | 0.520296 |
| torch_nn(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.659054 | 0.714089 | 0.373728 | 0.416170 | 0.275036 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-en | 11 | 0.681651 | 0.792474 | 0.598897 | 0.737826 | 0.416735 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-fi | 1 | 0.709384 | 0.810460 | 0.684051 | 0.796184 | 0.516068 |
| torch_nn_simple(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 15 | 0.429079 | 0.531054 | 0.358503 | 0.459624 | 0.262127 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-en | 12 | 0.746666 | 0.824106 | 0.617811 | 0.714514 | 0.432653 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-fi | 12 | 0.764292 | 0.839745 | 0.691875 | 0.768785 | 0.527237 |
| torch_nn_split(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.534741 | 0.629276 | 0.378951 | 0.444978 | 0.279118 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-en | 12 | 0.755700 | 0.829270 | 0.616533 | 0.710895 | 0.435301 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-fi | 12 | 0.764355 | 0.840039 | 0.691735 | 0.772526 | 0.527790 |
| torch_nn_split_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.538397 | 0.631761 | 0.379664 | 0.443365 | 0.278224 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 15 | 0.724006 | 0.815469 | 0.659227 | 0.771079 | 0.473627 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439253 | 0.555085 | 0.361643 | 0.473905 | 0.264727 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 13 | 0.714631 | 0.811372 | 0.648720 | 0.765062 | 0.463588 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Avg Test NDCG@10 |
|------|-------|----------------|
| 1 | torch_per_label | 0.577014 |
| 2 | torch_lowrank_residual_epsclamp | 0.571729 |
| 3 | torch_lowrank_residual_mix_temp | 0.571218 |
| 4 | torch_per_label_l1_delta | 0.569338 |
| 5 | nn | 0.568939 |
| 6 | torch_mean_residual_lowrank_mix | 0.566277 |
| 7 | torch_lowrank_mix | 0.565716 |
| 8 | torch_nn_split | 0.562879 |
| 9 | torch_nn_split_per_label | 0.562644 |
| 10 | torch_mean_residual | 0.559726 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Avg Test NDCG@1000 |
|------|-------|----------------|
| 1 | torch_per_label | 0.687146 |
| 2 | torch_per_label_l1_delta | 0.681959 |
| 3 | torch_mean_residual_lowrank_mix | 0.680352 |
| 4 | torch_mean_residual | 0.674558 |
| 5 | torch_lowrank_residual_sigmoid | 0.670482 |
| 6 | mean_weighted | 0.665660 |
| 7 | torch_mean | 0.664907 |
| 8 | torch_nn_simple | 0.664545 |
| 9 | mean | 0.663097 |
| 10 | torch_lowrank_residual_epsclamp | 0.653935 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Avg Test F1@5 |
|------|-------|----------------|
| 1 | torch_per_label | 0.427495 |
| 2 | torch_lowrank_mix | 0.422585 |
| 3 | torch_per_label_l1_delta | 0.421825 |
| 4 | torch_lowrank_residual_mix_temp | 0.419663 |
| 5 | torch_mean_residual_lowrank_mix | 0.419645 |
| 6 | torch_lowrank_residual_epsclamp | 0.419559 |
| 7 | nn | 0.418994 |
| 8 | torch_nn_split_per_label | 0.413772 |
| 9 | torch_nn_split | 0.413003 |
| 10 | torch_nn | 0.411695 |

## Top 10 Models by Avg of 3 Test Metrics (across datasets)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) across datasets |
|------|-------|----------------|
| 1 | torch_per_label | 0.563885 |
| 2 | torch_per_label_l1_delta | 0.557707 |
| 3 | torch_mean_residual_lowrank_mix | 0.555425 |
| 4 | torch_lowrank_residual_epsclamp | 0.548408 |
| 5 | torch_mean_residual | 0.548160 |
| 6 | torch_lowrank_residual_mix_temp | 0.547409 |
| 7 | torch_lowrank_residual_sigmoid | 0.542149 |
| 8 | nn | 0.541074 |
| 9 | torch_nn_split_per_label | 0.539559 |
| 10 | torch_nn_split | 0.539547 |

## Top 10 Models by Avg of 3 Test Metrics (koko)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_nn_split | 0.367682 |
| 2 | torch_mean_residual_lowrank_mix | 0.367336 |
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
| 1 | torch_per_label | 0.634644 |
| 2 | torch_per_label_l1_delta | 0.625790 |
| 3 | torch_mean_residual_lowrank_mix | 0.619487 |
| 4 | torch_lowrank_residual_mix_temp | 0.619032 |
| 5 | torch_lowrank_residual_epsclamp | 0.617627 |
| 6 | torch_lowrank_residual_sigmoid | 0.615821 |
| 7 | torch_mean_residual | 0.612860 |
| 8 | nn | 0.604626 |
| 9 | mean_weighted | 0.597406 |
| 10 | torch_lowrank_mix | 0.593512 |

## Top 10 Models by Avg of 3 Test Metrics (yso-fi)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_per_label | 0.690252 |
| 2 | torch_per_label_l1_delta | 0.680134 |
| 3 | torch_mean_residual_lowrank_mix | 0.679451 |
| 4 | torch_lowrank_residual_epsclamp | 0.670901 |
| 5 | torch_mean_residual | 0.669455 |
| 6 | torch_lowrank_residual_mix_temp | 0.669190 |
| 7 | torch_nn_simple | 0.665434 |
| 8 | torch_mean | 0.664459 |
| 9 | torch_nn_split_per_label | 0.664017 |
| 10 | mean | 0.663621 |
