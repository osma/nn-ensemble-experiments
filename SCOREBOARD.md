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
| torch_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 20 | 0.873205 | 0.878189 | 0.631249 | 0.668056 | 0.459232 |
| torch_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 20 | 0.888047 | 0.908317 | 0.699446 | 0.741196 | 0.538196 |
| torch_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 10 | 0.507234 | 0.609628 | 0.358664 | 0.423520 | 0.264630 |
| torch_lowrank_residual(bonsai,fasttext,mllm) | yso-en | 20 | 0.717563 | 0.793103 | 0.653607 | 0.732105 | 0.465664 |
| torch_lowrank_residual(bonsai,fasttext,mllm) | yso-fi | 20 | 0.753699 | 0.835708 | 0.695448 | 0.785801 | 0.528905 |
| torch_lowrank_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 10 | 0.464530 | 0.553372 | 0.365720 | 0.430831 | 0.270281 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-en | 19 | 0.717133 | 0.793257 | 0.653973 | 0.732769 | 0.465930 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756034 | 0.838082 | 0.696430 | 0.786967 | 0.529307 |
| torch_lowrank_residual_epsclamp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.454791 | 0.548610 | 0.364144 | 0.438432 | 0.267507 |
| torch_lowrank_residual_mix(bonsai,fasttext,mllm) | yso-en | 20 | 0.874919 | 0.880580 | 0.627940 | 0.665309 | 0.455926 |
| torch_lowrank_residual_mix(bonsai,fasttext,mllm) | yso-fi | 20 | 0.893764 | 0.911524 | 0.693493 | 0.734036 | 0.536008 |
| torch_lowrank_residual_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 10 | 0.513638 | 0.615358 | 0.357799 | 0.423026 | 0.265301 |
| torch_lowrank_residual_mix_nobias(bonsai,fasttext,mllm) | yso-en | 20 | 0.828541 | 0.853303 | 0.632876 | 0.669145 | 0.453603 |
| torch_lowrank_residual_mix_nobias(bonsai,fasttext,mllm) | yso-fi | 20 | 0.851940 | 0.887688 | 0.688544 | 0.734351 | 0.534384 |
| torch_lowrank_residual_mix_nobias(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.531422 | 0.662379 | 0.366639 | 0.445961 | 0.268024 |
| torch_lowrank_residual_mix_norms(bonsai,fasttext,mllm) | yso-en | 20 | 0.840331 | 0.864360 | 0.626935 | 0.676813 | 0.454098 |
| torch_lowrank_residual_mix_norms(bonsai,fasttext,mllm) | yso-fi | 20 | 0.826929 | 0.877662 | 0.690672 | 0.758672 | 0.529551 |
| torch_lowrank_residual_mix_norms(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 9 | 0.519358 | 0.620390 | 0.363637 | 0.427857 | 0.269193 |
| torch_lowrank_residual_mix_relu(bonsai,fasttext,mllm) | yso-en | 20 | 0.853536 | 0.864209 | 0.623285 | 0.665481 | 0.452075 |
| torch_lowrank_residual_mix_relu(bonsai,fasttext,mllm) | yso-fi | 20 | 0.837724 | 0.880177 | 0.692675 | 0.741166 | 0.528703 |
| torch_lowrank_residual_mix_relu(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 17 | 0.608507 | 0.664806 | 0.371847 | 0.408982 | 0.276031 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-en | 15 | 0.711034 | 0.789757 | 0.652036 | 0.731976 | 0.464101 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756853 | 0.837086 | 0.694785 | 0.785086 | 0.527698 |
| torch_lowrank_residual_mix_temp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 13 | 0.465222 | 0.555876 | 0.363119 | 0.430717 | 0.268176 |
| torch_lowrank_residual_sigmoid(bonsai,fasttext,mllm) | yso-en | 2 | 0.608646 | 0.731768 | 0.552307 | 0.699277 | 0.392470 |
| torch_lowrank_residual_sigmoid(bonsai,fasttext,mllm) | yso-fi | 1 | 0.684933 | 0.795630 | 0.663998 | 0.784065 | 0.493791 |
| torch_lowrank_residual_sigmoid(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435170 | 0.548795 | 0.358693 | 0.468941 | 0.262387 |
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
| torch_nn_dropout_only(bonsai,fasttext,mllm) | yso-en | 20 | 0.854554 | 0.871082 | 0.605290 | 0.655398 | 0.440549 |
| torch_nn_dropout_only(bonsai,fasttext,mllm) | yso-fi | 20 | 0.838046 | 0.879032 | 0.692188 | 0.744477 | 0.531826 |
| torch_nn_dropout_only(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.656562 | 0.710903 | 0.383384 | 0.422408 | 0.281413 |
| torch_nn_log1p(bonsai,fasttext,mllm) | yso-en | 20 | 0.822583 | 0.853946 | 0.571915 | 0.642968 | 0.407884 |
| torch_nn_log1p(bonsai,fasttext,mllm) | yso-fi | 20 | 0.813764 | 0.861914 | 0.683364 | 0.749880 | 0.521699 |
| torch_nn_log1p(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.560471 | 0.657016 | 0.369241 | 0.431439 | 0.271380 |
| torch_nn_lowrank(bonsai,fasttext,mllm) | yso-en | 20 | 0.759509 | 0.814700 | 0.599030 | 0.670688 | 0.430101 |
| torch_nn_lowrank(bonsai,fasttext,mllm) | yso-fi | 20 | 0.786500 | 0.852349 | 0.687086 | 0.752909 | 0.524924 |
| torch_nn_lowrank(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.557428 | 0.647667 | 0.378512 | 0.435131 | 0.276145 |
| torch_nn_mlp_only(bonsai,fasttext,mllm) | yso-en | 20 | 0.419654 | 0.641652 | 0.091606 | 0.177297 | 0.057808 |
| torch_nn_mlp_only(bonsai,fasttext,mllm) | yso-fi | 20 | 0.375500 | 0.553484 | 0.123865 | 0.220500 | 0.095023 |
| torch_nn_mlp_only(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.482524 | 0.608666 | 0.226808 | 0.305933 | 0.165068 |
| torch_nn_residual(bonsai,fasttext,mllm) | yso-en | 20 | 0.850619 | 0.871051 | 0.604233 | 0.654747 | 0.434140 |
| torch_nn_residual(bonsai,fasttext,mllm) | yso-fi | 20 | 0.836512 | 0.879323 | 0.691390 | 0.743584 | 0.531532 |
| torch_nn_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.658696 | 0.712685 | 0.382478 | 0.423285 | 0.282331 |
| torch_nn_sigmoid(bonsai,fasttext,mllm) | yso-en | 3 | 0.517272 | 0.677855 | 0.449975 | 0.632732 | 0.318253 |
| torch_nn_sigmoid(bonsai,fasttext,mllm) | yso-fi | 1 | 0.709647 | 0.810724 | 0.683949 | 0.796133 | 0.515905 |
| torch_nn_sigmoid(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.366502 | 0.507263 | 0.299015 | 0.430012 | 0.210142 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-en | 20 | 0.620905 | 0.737203 | 0.588106 | 0.718240 | 0.424256 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-fi | 1 | 0.709384 | 0.810460 | 0.684051 | 0.796184 | 0.516068 |
| torch_nn_simple(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 15 | 0.429079 | 0.531054 | 0.358503 | 0.459624 | 0.262127 |
| torch_nn_weightdecay(bonsai,fasttext,mllm) | yso-en | 20 | 0.820673 | 0.855153 | 0.598908 | 0.665719 | 0.434261 |
| torch_nn_weightdecay(bonsai,fasttext,mllm) | yso-fi | 20 | 0.818528 | 0.872079 | 0.691622 | 0.752863 | 0.530428 |
| torch_nn_weightdecay(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 20 | 0.693470 | 0.732928 | 0.382427 | 0.415747 | 0.282556 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 19 | 0.626856 | 0.741188 | 0.515139 | 0.671659 | 0.344941 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439253 | 0.555085 | 0.361643 | 0.473905 | 0.264727 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 20 | 0.619971 | 0.738825 | 0.522815 | 0.679542 | 0.355704 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Avg Test NDCG@10 |
|------|-------|----------------|
| 1 | torch_lowrank_residual | 0.571592 |
| 2 | torch_lowrank_residual_epsclamp | 0.571516 |
| 3 | torch_lowrank_residual_mix_temp | 0.569980 |
| 4 | torch_mean_residual | 0.567277 |
| 5 | torch_lowrank_mix | 0.563120 |
| 6 | torch_lowrank_residual_mix_nobias | 0.562686 |
| 7 | torch_lowrank_residual_mix_relu | 0.562602 |
| 8 | torch_lowrank_residual_mix_norms | 0.560415 |
| 9 | torch_nn_dropout_only | 0.560287 |
| 10 | torch_lowrank_residual_mix | 0.559744 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Avg Test NDCG@1000 |
|------|-------|----------------|
| 1 | torch_mean | 0.669467 |
| 2 | mean_weighted | 0.668390 |
| 3 | torch_mean_residual | 0.667334 |
| 4 | torch_nn_simple | 0.658016 |
| 5 | torch_per_label | 0.654006 |
| 6 | torch_per_label_l1_delta | 0.653453 |
| 7 | torch_lowrank_residual_epsclamp | 0.652723 |
| 8 | torch_lowrank_residual_sigmoid | 0.650761 |
| 9 | torch_lowrank_residual | 0.649579 |
| 10 | torch_lowrank_residual_mix_temp | 0.649260 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Avg Test F1@5 |
|------|-------|----------------|
| 1 | torch_lowrank_residual | 0.421617 |
| 2 | torch_lowrank_residual_epsclamp | 0.420915 |
| 3 | torch_lowrank_mix | 0.420686 |
| 4 | torch_lowrank_residual_mix_temp | 0.419992 |
| 5 | torch_lowrank_residual_mix | 0.419078 |
| 6 | torch_lowrank_residual_mix_relu | 0.418936 |
| 7 | torch_lowrank_residual_mix_nobias | 0.418670 |
| 8 | torch_nn_dropout_only | 0.417929 |
| 9 | torch_lowrank_residual_mix_norms | 0.417614 |
| 10 | torch_mean_residual | 0.417405 |

## Top 10 Models by Avg of 3 Test Metrics (koko)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_per_label_l1_delta | 0.367199 |
| 2 | torch_per_label | 0.366758 |
| 3 | torch_lowrank_residual_sigmoid | 0.363340 |
| 4 | torch_nn_lowrank | 0.363263 |
| 5 | torch_nn | 0.362698 |
| 6 | torch_nn_residual | 0.362698 |
| 7 | torch_nn_dropout_only | 0.362402 |
| 8 | torch_mean_residual | 0.362165 |
| 9 | mean_weighted | 0.360361 |
| 10 | torch_nn_weightdecay | 0.360243 |

## Top 10 Models by Avg of 3 Test Metrics (yso-en)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_mean_residual | 0.620397 |
| 2 | torch_lowrank_residual_epsclamp | 0.617557 |
| 3 | torch_lowrank_residual | 0.617125 |
| 4 | torch_lowrank_residual_mix_temp | 0.616038 |
| 5 | mean_weighted | 0.615235 |
| 6 | torch_mean | 0.614751 |
| 7 | torch_mean_bias | 0.587923 |
| 8 | torch_lowrank_mix | 0.586179 |
| 9 | torch_lowrank_residual_mix_norms | 0.585949 |
| 10 | torch_lowrank_residual_mix_nobias | 0.585208 |

## Top 10 Models by Avg of 3 Test Metrics (yso-fi)

| Rank | Model | Avg(Test NDCG@1000, NDCG@10, F1@5) |
|------|-------|----------------|
| 1 | torch_per_label | 0.690252 |
| 2 | torch_per_label_l1_delta | 0.680134 |
| 3 | torch_lowrank_residual_epsclamp | 0.670901 |
| 4 | torch_lowrank_residual | 0.670051 |
| 5 | torch_mean_residual | 0.669455 |
| 6 | torch_lowrank_residual_mix_temp | 0.669190 |
| 7 | torch_nn_simple | 0.665434 |
| 8 | torch_nn_sigmoid | 0.665329 |
| 9 | torch_mean | 0.664443 |
| 10 | nn | 0.659880 |
