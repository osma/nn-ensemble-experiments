# Benchmark Scoreboard

| Model | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 |
|-------|---------------|----------------|-------------|----------------|
| bonsai | 0.657092 | 0.741322 | 0.633398 | 0.718595 |
| fasttext | 0.454610 | 0.621074 | 0.425691 | 0.601266 |
| mean | 0.716997 | 0.822220 | 0.679141 | 0.795979 |
| mean_weighted | 0.716469 | 0.822129 | 0.678707 | 0.796209 |
| mllm | 0.630515 | 0.746998 | 0.578451 | 0.706146 |
| nn | 0.730767 | 0.802585 | 0.689080 | 0.768269 |
| torch_mean_bias_epoch01 | 0.732147 | 0.822386 | 0.691070 | 0.783230 |
| torch_mean_bias_epoch02 | 0.738371 | 0.825648 | 0.694869 | 0.781851 |
| torch_mean_bias_epoch03 | 0.741993 | 0.827945 | 0.695354 | 0.780471 |
| torch_mean_bias_epoch04 | 0.744400 | 0.828848 | 0.696578 | 0.779068 |
| torch_mean_bias_epoch05 | 0.746452 | 0.830574 | 0.696882 | 0.778904 |
| torch_mean_bias_epoch06 | 0.747344 | 0.830534 | 0.696407 | 0.777719 |
| torch_mean_bias_epoch07 | 0.749158 | 0.831588 | 0.696723 | 0.776562 |
| torch_mean_bias_epoch08 | 0.750164 | 0.832355 | 0.696755 | 0.776086 |
| torch_mean_bias_epoch09 | 0.751271 | 0.832595 | 0.695506 | 0.774993 |
| torch_mean_bias_epoch10 | 0.751250 | 0.832654 | 0.695324 | 0.774319 |
| torch_mean_epoch01 | 0.716999 | 0.822104 | 0.679112 | 0.795787 |
| torch_mean_epoch02 | 0.717260 | 0.821987 | 0.679046 | 0.795396 |
| torch_mean_epoch03 | 0.717763 | 0.822026 | 0.679445 | 0.795770 |
| torch_mean_epoch04 | 0.717890 | 0.822200 | 0.680066 | 0.795824 |
| torch_mean_epoch05 | 0.717960 | 0.822255 | 0.680588 | 0.796080 |
| torch_mean_epoch06 | 0.718037 | 0.822161 | 0.680860 | 0.795942 |
| torch_mean_epoch07 | 0.717910 | 0.822099 | 0.681228 | 0.796105 |
| torch_mean_epoch08 | 0.717924 | 0.821980 | 0.681514 | 0.796219 |
| torch_mean_epoch09 | 0.717926 | 0.822016 | 0.681525 | 0.796220 |
| torch_mean_epoch10 | 0.717918 | 0.821978 | 0.681803 | 0.796406 |
| torch_per_label_epoch01 | 0.732326 | 0.832904 | 0.693471 | 0.805278 |
| torch_per_label_epoch02 | 0.744878 | 0.840904 | 0.702059 | 0.810632 |
| torch_per_label_epoch03 | 0.752365 | 0.845690 | 0.704220 | 0.812596 |
| torch_per_label_epoch04 | 0.753959 | 0.845927 | 0.703586 | 0.809834 |
| torch_per_label_epoch05 | 0.752426 | 0.841230 | 0.699834 | 0.802418 |
| torch_per_label_epoch06 | 0.749958 | 0.834720 | 0.694206 | 0.795031 |
| torch_per_label_epoch07 | 0.747225 | 0.828635 | 0.688216 | 0.785975 |
| torch_per_label_epoch08 | 0.744619 | 0.823076 | 0.682301 | 0.777972 |
| torch_per_label_epoch09 | 0.742248 | 0.818652 | 0.676396 | 0.770349 |
| torch_per_label_epoch10 | 0.739971 | 0.815035 | 0.672104 | 0.763907 |

## Top 10 Models by Test NDCG@10

| Rank | Model | Test NDCG@10 | Test NDCG@1000 |
|------|-------|--------------|----------------|
| 1 | torch_per_label_epoch03 | 0.704220 | 0.812596 |
| 2 | torch_per_label_epoch04 | 0.703586 | 0.809834 |
| 3 | torch_per_label_epoch02 | 0.702059 | 0.810632 |
| 4 | torch_per_label_epoch05 | 0.699834 | 0.802418 |
| 5 | torch_mean_bias_epoch05 | 0.696882 | 0.778904 |
| 6 | torch_mean_bias_epoch08 | 0.696755 | 0.776086 |
| 7 | torch_mean_bias_epoch07 | 0.696723 | 0.776562 |
| 8 | torch_mean_bias_epoch04 | 0.696578 | 0.779068 |
| 9 | torch_mean_bias_epoch06 | 0.696407 | 0.777719 |
| 10 | torch_mean_bias_epoch09 | 0.695506 | 0.774993 |

## Top 10 Models by Test NDCG@1000

| Rank | Model | Test NDCG@1000 | Test NDCG@10 |
|------|-------|----------------|--------------|
| 1 | torch_per_label_epoch03 | 0.812596 | 0.704220 |
| 2 | torch_per_label_epoch02 | 0.810632 | 0.702059 |
| 3 | torch_per_label_epoch04 | 0.809834 | 0.703586 |
| 4 | torch_per_label_epoch01 | 0.805278 | 0.693471 |
| 5 | torch_per_label_epoch05 | 0.802418 | 0.699834 |
| 6 | torch_mean_epoch10 | 0.796406 | 0.681803 |
| 7 | torch_mean_epoch09 | 0.796220 | 0.681525 |
| 8 | torch_mean_epoch08 | 0.796219 | 0.681514 |
| 9 | mean_weighted | 0.796209 | 0.678707 |
| 10 | torch_mean_epoch07 | 0.796105 | 0.681228 |
