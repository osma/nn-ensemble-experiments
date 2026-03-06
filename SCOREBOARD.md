# Benchmark Scoreboard

| Model | Dataset | Epoch | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|-------|---------|-------|---------------|----------------|-------------|----------------|-----------|
| bonsai | yso-en |  | 0.655159 | 0.734786 | 0.640733 | 0.731381 | 0.454515 |
| bonsai | yso-fi |  | 0.627302 | 0.707972 | 0.633398 | 0.718595 | 0.472298 |
| bonsai_gemma3 | koko |  | 0.383339 | 0.455443 | 0.314190 | 0.381060 | 0.229376 |
| bonsai_ovis2 | koko |  | 0.321541 | 0.394300 | 0.273783 | 0.343067 | 0.197926 |
| fasttext | yso-en |  | 0.429327 | 0.601442 | 0.458165 | 0.622390 | 0.317633 |
| fasttext | yso-fi |  | 0.475827 | 0.637037 | 0.425691 | 0.601266 | 0.306065 |
| mean(bonsai,fasttext,mllm) | yso-en |  | 0.452422 | 0.636637 | 0.372590 | 0.582518 | 0.253387 |
| mean(bonsai,fasttext,mllm) | yso-fi |  | 0.710079 | 0.811516 | 0.679141 | 0.795979 | 0.515742 |
| mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.393507 | 0.494590 | 0.333554 | 0.429657 | 0.243392 |
| mean_weighted(bonsai,fasttext,mllm) | yso-en |  | 0.654260 | 0.761685 | 0.639565 | 0.754048 | 0.452093 |
| mean_weighted(bonsai,fasttext,mllm) | yso-fi |  | 0.709956 | 0.812725 | 0.669945 | 0.791698 | 0.508171 |
| mean_weighted(bonsai_gemma3,bonsai_ovis2,mllm) | koko |  | 0.405949 | 0.503265 | 0.339526 | 0.433621 | 0.247509 |
| mllm | koko |  | 0.162187 | 0.159443 | 0.154528 | 0.155313 | 0.110493 |
| mllm | yso-en |  | 0.000314 | 0.001211 | 0.000195 | 0.000689 | 0.000000 |
| mllm | yso-fi |  | 0.616455 | 0.669603 | 0.578451 | 0.706146 | 0.437427 |
| nn | koko |  |  |  | 0.374594 | 0.423777 | 0.276537 |
| nn | yso-en |  |  |  | 0.496716 | 0.615022 | 0.350647 |
| nn | yso-fi |  |  |  | 0.689080 | 0.768269 | 0.522291 |
| torch_mean(bonsai,fasttext,mllm) | yso-en | 6 | 0.653498 | 0.761261 | 0.638523 | 0.752934 | 0.452598 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 2 | 0.711518 | 0.812464 | 0.679695 | 0.796348 | 0.517125 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 8 | 0.400313 | 0.499489 | 0.338326 | 0.433083 | 0.246876 |
| torch_mean_bias(bonsai,fasttext,mllm) | yso-en | 10 | 0.674754 | 0.744848 | 0.626509 | 0.693832 | 0.445743 |
| torch_mean_bias(bonsai,fasttext,mllm) | yso-fi | 10 | 0.731111 | 0.805586 | 0.677608 | 0.767045 | 0.518328 |
| torch_mean_bias(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 7 | 0.416765 | 0.481197 | 0.336639 | 0.384107 | 0.247349 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 10 | 0.653694 | 0.768761 | 0.559324 | 0.714516 | 0.385272 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.725988 | 0.823065 | 0.710171 | 0.816454 | 0.544132 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.409285 | 0.533807 | 0.343566 | 0.458088 | 0.253179 |
| torch_per_label_conv(bonsai,fasttext,mllm) | yso-en | 10 | 0.504217 | 0.672545 | 0.406139 | 0.604129 | 0.275098 |
| torch_per_label_conv(bonsai,fasttext,mllm) | yso-fi | 10 | 0.729305 | 0.821383 | 0.701943 | 0.806468 | 0.533019 |
| torch_per_label_conv(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.404417 | 0.532019 | 0.340194 | 0.455562 | 0.249534 |
| torch_per_label_freq_gate(bonsai,fasttext,mllm) | yso-en | 1 | 0.437708 | 0.623800 | 0.365644 | 0.578752 | 0.247562 |
| torch_per_label_freq_gate(bonsai,fasttext,mllm) | yso-fi | 1 | 0.716475 | 0.814877 | 0.696598 | 0.806859 | 0.528884 |
| torch_per_label_freq_gate(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 2 | 0.403321 | 0.524100 | 0.334008 | 0.445630 | 0.244112 |
| torch_per_label_freq_reg(bonsai,fasttext,mllm) | yso-en | 10 | 0.563844 | 0.705697 | 0.427240 | 0.617491 | 0.279591 |
| torch_per_label_freq_reg(bonsai,fasttext,mllm) | yso-fi | 5 | 0.733841 | 0.826144 | 0.707222 | 0.813457 | 0.538189 |
| torch_per_label_freq_reg(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 2 | 0.407270 | 0.528842 | 0.338014 | 0.448229 | 0.248230 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 10 | 0.639525 | 0.759576 | 0.558206 | 0.713001 | 0.391758 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.414738 | 0.539033 | 0.343949 | 0.458412 | 0.253256 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |
|------|-------|--------------|----------------|-----------|
| 1 | torch_mean | 0.552181 | 0.660788 | 0.405533 |
| 2 | mean_weighted | 0.549679 | 0.659789 | 0.402591 |
| 3 | torch_mean_bias | 0.546919 | 0.614995 | 0.403807 |
| 4 | torch_per_label | 0.537687 | 0.663019 | 0.394194 |
| 5 | torch_per_label_l1_delta | 0.533470 | 0.659303 | 0.393555 |
| 6 | nn | 0.520130 | 0.602356 | 0.383158 |
| 7 | torch_per_label_freq_reg | 0.490825 | 0.626392 | 0.355337 |
| 8 | torch_per_label_conv | 0.482759 | 0.622053 | 0.352550 |
| 9 | torch_per_label_freq_gate | 0.465417 | 0.610414 | 0.340186 |
| 10 | mean | 0.461762 | 0.602718 | 0.337507 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Test NDCG@1000 | Test NDCG@10 | Test F1@5 |
|------|-------|----------------|--------------|-----------|
| 1 | torch_per_label | 0.663019 | 0.537687 | 0.394194 |
| 2 | torch_mean | 0.660788 | 0.552181 | 0.405533 |
| 3 | mean_weighted | 0.659789 | 0.549679 | 0.402591 |
| 4 | torch_per_label_l1_delta | 0.659303 | 0.533470 | 0.393555 |
| 5 | torch_per_label_freq_reg | 0.626392 | 0.490825 | 0.355337 |
| 6 | torch_per_label_conv | 0.622053 | 0.482759 | 0.352550 |
| 7 | torch_mean_bias | 0.614995 | 0.546919 | 0.403807 |
| 8 | torch_per_label_freq_gate | 0.610414 | 0.465417 | 0.340186 |
| 9 | mean | 0.602718 | 0.461762 | 0.337507 |
| 10 | nn | 0.602356 | 0.520130 | 0.383158 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Test F1@5 | Test NDCG@10 | Test NDCG@1000 |
|------|-------|-----------|--------------|----------------|
| 1 | torch_mean | 0.405533 | 0.552181 | 0.660788 |
| 2 | torch_mean_bias | 0.403807 | 0.546919 | 0.614995 |
| 3 | mean_weighted | 0.402591 | 0.549679 | 0.659789 |
| 4 | torch_per_label | 0.394194 | 0.537687 | 0.663019 |
| 5 | torch_per_label_l1_delta | 0.393555 | 0.533470 | 0.659303 |
| 6 | nn | 0.383158 | 0.520130 | 0.602356 |
| 7 | torch_per_label_freq_reg | 0.355337 | 0.490825 | 0.626392 |
| 8 | torch_per_label_conv | 0.352550 | 0.482759 | 0.622053 |
| 9 | torch_per_label_freq_gate | 0.340186 | 0.465417 | 0.610414 |
| 10 | mean | 0.337507 | 0.461762 | 0.602718 |
