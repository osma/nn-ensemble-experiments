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
| torch_3stage(bonsai,fasttext,mllm) | yso-en | 2 | 0.649859 | 0.765633 | 0.528679 | 0.687371 | 0.377071 |
| torch_3stage(bonsai,fasttext,mllm) | yso-fi | 1 | 0.664377 | 0.775213 | 0.626816 | 0.753115 | 0.480077 |
| torch_3stage(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.436830 | 0.549279 | 0.358635 | 0.468554 | 0.262808 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-en | 20 | 0.748086 | 0.824475 | 0.654614 | 0.736405 | 0.461863 |
| torch_lowrank_residual_epsclamp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756034 | 0.838082 | 0.696430 | 0.786967 | 0.529307 |
| torch_lowrank_residual_epsclamp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.454791 | 0.548610 | 0.364144 | 0.438432 | 0.267507 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-en | 20 | 0.747455 | 0.823615 | 0.655750 | 0.738232 | 0.463115 |
| torch_lowrank_residual_mix_temp(bonsai,fasttext,mllm) | yso-fi | 20 | 0.756740 | 0.837466 | 0.695460 | 0.785514 | 0.528406 |
| torch_lowrank_residual_mix_temp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 13 | 0.465222 | 0.555876 | 0.363119 | 0.430717 | 0.268176 |
| torch_mean(bonsai,fasttext,mllm) | yso-en | 1 | 0.684793 | 0.795369 | 0.601542 | 0.738840 | 0.421134 |
| torch_mean(bonsai,fasttext,mllm) | yso-fi | 3 | 0.711586 | 0.812470 | 0.679727 | 0.796308 | 0.517343 |
| torch_mean(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 6 | 0.430893 | 0.533464 | 0.358721 | 0.459573 | 0.263393 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-en | 3 | 0.702887 | 0.801773 | 0.634044 | 0.757152 | 0.447385 |
| torch_mean_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710736 | 0.811807 | 0.687398 | 0.799336 | 0.521631 |
| torch_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435223 | 0.548767 | 0.357736 | 0.467187 | 0.261571 |
| torch_mean_residual_bias_per_model(bonsai,fasttext,mllm) | yso-en | 1 | 0.692195 | 0.795264 | 0.623849 | 0.747570 | 0.439277 |
| torch_mean_residual_bias_per_model(bonsai,fasttext,mllm) | yso-fi | 1 | 0.708425 | 0.808106 | 0.683045 | 0.795515 | 0.513818 |
| torch_mean_residual_bias_per_model(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.425044 | 0.547631 | 0.351774 | 0.466812 | 0.258286 |
| torch_mean_residual_bias_residual(bonsai,fasttext,mllm) | yso-en | 3 | 0.702781 | 0.801785 | 0.634035 | 0.757160 | 0.447385 |
| torch_mean_residual_bias_residual(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710725 | 0.811879 | 0.687098 | 0.799073 | 0.521394 |
| torch_mean_residual_bias_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435378 | 0.549635 | 0.357903 | 0.467733 | 0.261421 |
| torch_mean_residual_delta_tanh_clamp(bonsai,fasttext,mllm) | yso-en | 1 | 0.691627 | 0.796332 | 0.623692 | 0.750391 | 0.441138 |
| torch_mean_residual_delta_tanh_clamp(bonsai,fasttext,mllm) | yso-fi | 1 | 0.699828 | 0.803911 | 0.676865 | 0.793029 | 0.509067 |
| torch_mean_residual_delta_tanh_clamp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.432176 | 0.548682 | 0.356272 | 0.468751 | 0.260805 |
| torch_mean_residual_freq_weighted_delta(bonsai,fasttext,mllm) | yso-en | 3 | 0.702859 | 0.801762 | 0.634661 | 0.757517 | 0.447683 |
| torch_mean_residual_freq_weighted_delta(bonsai,fasttext,mllm) | yso-fi | 2 | 0.710826 | 0.811934 | 0.687567 | 0.799662 | 0.522855 |
| torch_mean_residual_freq_weighted_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.435295 | 0.548706 | 0.357814 | 0.466835 | 0.261462 |
| torch_mean_residual_globalxdelta(bonsai,fasttext,mllm) | yso-en | 1 | 0.692306 | 0.796779 | 0.624606 | 0.751390 | 0.441895 |
| torch_mean_residual_globalxdelta(bonsai,fasttext,mllm) | yso-fi | 1 | 0.699648 | 0.803570 | 0.676377 | 0.792537 | 0.508068 |
| torch_mean_residual_globalxdelta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.432541 | 0.549126 | 0.356553 | 0.469326 | 0.260765 |
| torch_mean_residual_l2_anchor_global(bonsai,fasttext,mllm) | yso-en | 16 | 0.713445 | 0.810939 | 0.645636 | 0.763440 | 0.461992 |
| torch_mean_residual_l2_anchor_global(bonsai,fasttext,mllm) | yso-fi | 6 | 0.726354 | 0.821728 | 0.698331 | 0.806099 | 0.536818 |
| torch_mean_residual_l2_anchor_global(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 5 | 0.438967 | 0.559425 | 0.361355 | 0.476810 | 0.265731 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-en | 5 | 0.703847 | 0.804152 | 0.640272 | 0.762450 | 0.455739 |
| torch_mean_residual_lowrank_mix(bonsai,fasttext,mllm) | yso-fi | 5 | 0.724906 | 0.820559 | 0.697428 | 0.804458 | 0.536466 |
| torch_mean_residual_lowrank_mix(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.438590 | 0.559694 | 0.361132 | 0.474147 | 0.266729 |
| torch_mean_residual_softmax_global(bonsai,fasttext,mllm) | yso-en | 8 | 0.706926 | 0.805781 | 0.648991 | 0.766754 | 0.461473 |
| torch_mean_residual_softmax_global(bonsai,fasttext,mllm) | yso-fi | 5 | 0.724702 | 0.820463 | 0.696986 | 0.804392 | 0.536324 |
| torch_mean_residual_softmax_global(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.438342 | 0.559278 | 0.361896 | 0.477831 | 0.266607 |
| torch_mean_residual_softmax_global_l2_anchor(bonsai,fasttext,mllm) | yso-en | 16 | 0.713340 | 0.810969 | 0.645476 | 0.763398 | 0.461776 |
| torch_mean_residual_softmax_global_l2_anchor(bonsai,fasttext,mllm) | yso-fi | 6 | 0.726384 | 0.821850 | 0.698110 | 0.806149 | 0.536644 |
| torch_mean_residual_softmax_global_l2_anchor(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 5 | 0.439002 | 0.559453 | 0.361353 | 0.476813 | 0.265713 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-en | 11 | 0.681651 | 0.792474 | 0.598897 | 0.737826 | 0.416735 |
| torch_nn_simple(bonsai,fasttext,mllm) | yso-fi | 1 | 0.709384 | 0.810460 | 0.684051 | 0.796184 | 0.516068 |
| torch_nn_simple(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 15 | 0.429079 | 0.531054 | 0.358503 | 0.459624 | 0.262127 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-en | 12 | 0.746666 | 0.824106 | 0.617811 | 0.714514 | 0.432653 |
| torch_nn_split(bonsai,fasttext,mllm) | yso-fi | 12 | 0.764292 | 0.839745 | 0.691875 | 0.768785 | 0.527237 |
| torch_nn_split(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.534741 | 0.629276 | 0.378951 | 0.444978 | 0.279118 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-en | 12 | 0.784181 | 0.848789 | 0.648981 | 0.732718 | 0.458613 |
| torch_nn_split_per_label(bonsai,fasttext,mllm) | yso-fi | 12 | 0.779001 | 0.853775 | 0.696663 | 0.775173 | 0.531756 |
| torch_nn_split_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 12 | 0.544597 | 0.640002 | 0.376592 | 0.446252 | 0.276799 |
| torch_per_label(bonsai,fasttext,mllm) | yso-en | 15 | 0.722371 | 0.814886 | 0.650776 | 0.764991 | 0.467200 |
| torch_per_label(bonsai,fasttext,mllm) | yso-fi | 5 | 0.728346 | 0.824324 | 0.697123 | 0.806156 | 0.534772 |
| torch_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.442077 | 0.558998 | 0.361336 | 0.473884 | 0.266223 |
| torch_per_label_apex(bonsai,fasttext,mllm) | yso-en | 7 | 0.705346 | 0.804996 | 0.646978 | 0.765773 | 0.459256 |
| torch_per_label_apex(bonsai,fasttext,mllm) | yso-fi | 5 | 0.724470 | 0.820458 | 0.697312 | 0.804652 | 0.536479 |
| torch_per_label_apex(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 5 | 0.434240 | 0.557843 | 0.359000 | 0.476713 | 0.263887 |
| torch_per_label_bias_global_plus_delta(bonsai,fasttext,mllm) | yso-en | 16 | 0.724260 | 0.816058 | 0.656047 | 0.768774 | 0.472266 |
| torch_per_label_bias_global_plus_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.728169 | 0.824068 | 0.695609 | 0.804951 | 0.535320 |
| torch_per_label_bias_global_plus_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440609 | 0.558307 | 0.361324 | 0.473755 | 0.265462 |
| torch_per_label_elastic_anchor(bonsai,fasttext,mllm) | yso-en | 9 | 0.711313 | 0.807990 | 0.653161 | 0.768733 | 0.466575 |
| torch_per_label_elastic_anchor(bonsai,fasttext,mllm) | yso-fi | 4 | 0.724863 | 0.822136 | 0.694199 | 0.803687 | 0.532063 |
| torch_per_label_elastic_anchor(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.442434 | 0.561829 | 0.363811 | 0.478812 | 0.268185 |
| torch_per_label_global_plus_delta(bonsai,fasttext,mllm) | yso-en | 3 | 0.702425 | 0.801179 | 0.635216 | 0.758246 | 0.449534 |
| torch_per_label_global_plus_delta(bonsai,fasttext,mllm) | yso-fi | 2 | 0.707021 | 0.809255 | 0.692057 | 0.803521 | 0.524475 |
| torch_per_label_global_plus_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.433219 | 0.546388 | 0.356868 | 0.465507 | 0.260260 |
| torch_per_label_global_times_scale(bonsai,fasttext,mllm) | yso-en | 20 | 0.709628 | 0.809799 | 0.643669 | 0.765229 | 0.455654 |
| torch_per_label_global_times_scale(bonsai,fasttext,mllm) | yso-fi | 14 | 0.709633 | 0.811381 | 0.682260 | 0.796095 | 0.517386 |
| torch_per_label_global_times_scale(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.432622 | 0.555527 | 0.359675 | 0.476947 | 0.264296 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-en | 13 | 0.714631 | 0.811372 | 0.648720 | 0.765062 | 0.463588 |
| torch_per_label_l1_delta(bonsai,fasttext,mllm) | yso-fi | 5 | 0.723691 | 0.820476 | 0.698256 | 0.806495 | 0.535650 |
| torch_per_label_l1_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.440279 | 0.558067 | 0.361038 | 0.474321 | 0.266237 |
| torch_per_label_mlp(bonsai,fasttext,mllm) | yso-en | 1 | 0.722371 | 0.814921 | 0.650776 | 0.764912 | 0.467200 |
| torch_per_label_mlp(bonsai,fasttext,mllm) | yso-fi | 3 | 0.737340 | 0.831555 | 0.698582 | 0.806576 | 0.534560 |
| torch_per_label_mlp(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.442098 | 0.559631 | 0.361286 | 0.474052 | 0.266288 |
| torch_per_label_mlp_additive_delta(bonsai,fasttext,mllm) | yso-en | 2 | 0.722825 | 0.815386 | 0.650038 | 0.762893 | 0.466016 |
| torch_per_label_mlp_additive_delta(bonsai,fasttext,mllm) | yso-fi | 1 | 0.728276 | 0.824303 | 0.697128 | 0.805955 | 0.534772 |
| torch_per_label_mlp_additive_delta(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.442570 | 0.553137 | 0.358676 | 0.460274 | 0.264883 |
| torch_per_label_mlp_gate_per_label(bonsai,fasttext,mllm) | yso-en | 1 | 0.722371 | 0.814921 | 0.650776 | 0.764912 | 0.467200 |
| torch_per_label_mlp_gate_per_label(bonsai,fasttext,mllm) | yso-fi | 6 | 0.728348 | 0.824308 | 0.697123 | 0.806150 | 0.534772 |
| torch_per_label_mlp_gate_per_label(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.444023 | 0.560678 | 0.361536 | 0.465728 | 0.266017 |
| torch_per_label_mlp_gate_per_sample(bonsai,fasttext,mllm) | yso-en | 1 | 0.722371 | 0.814918 | 0.650776 | 0.764916 | 0.467200 |
| torch_per_label_mlp_gate_per_sample(bonsai,fasttext,mllm) | yso-fi | 5 | 0.728348 | 0.824314 | 0.697123 | 0.806152 | 0.534772 |
| torch_per_label_mlp_gate_per_sample(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.442091 | 0.559679 | 0.361286 | 0.474100 | 0.266266 |
| torch_per_label_mlp_layernorm_feats(bonsai,fasttext,mllm) | yso-en | 1 | 0.722713 | 0.815400 | 0.650211 | 0.763108 | 0.466447 |
| torch_per_label_mlp_layernorm_feats(bonsai,fasttext,mllm) | yso-fi | 1 | 0.728210 | 0.824211 | 0.697001 | 0.805845 | 0.534772 |
| torch_per_label_mlp_layernorm_feats(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.444570 | 0.557160 | 0.361464 | 0.461302 | 0.266139 |
| torch_per_label_mlp_rank_bottleneck(bonsai,fasttext,mllm) | yso-en | 4 | 0.722368 | 0.814968 | 0.650776 | 0.764786 | 0.467200 |
| torch_per_label_mlp_rank_bottleneck(bonsai,fasttext,mllm) | yso-fi | 4 | 0.728348 | 0.824327 | 0.697123 | 0.806153 | 0.534772 |
| torch_per_label_mlp_rank_bottleneck(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 4 | 0.443047 | 0.560156 | 0.360948 | 0.470921 | 0.266199 |
| torch_per_label_mlp_remove_centering(bonsai,fasttext,mllm) | yso-en | 1 | 0.722371 | 0.793573 | 0.650776 | 0.737115 | 0.467200 |
| torch_per_label_mlp_remove_centering(bonsai,fasttext,mllm) | yso-fi | 1 | 0.728346 | 0.809969 | 0.697123 | 0.793181 | 0.534772 |
| torch_per_label_mlp_remove_centering(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.441929 | 0.495094 | 0.361142 | 0.410901 | 0.266119 |
| torch_per_label_residual_lowrank_mix_active(bonsai,fasttext,mllm) | yso-en | 1 | 0.722495 | 0.808272 | 0.651005 | 0.758634 | 0.466942 |
| torch_per_label_residual_lowrank_mix_active(bonsai,fasttext,mllm) | yso-fi | 1 | 0.728239 | 0.822888 | 0.697189 | 0.804671 | 0.534990 |
| torch_per_label_residual_lowrank_mix_active(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.442085 | 0.528383 | 0.361227 | 0.444598 | 0.266173 |
| torch_per_label_softmax_global(bonsai,fasttext,mllm) | yso-en | 10 | 0.712864 | 0.808071 | 0.655809 | 0.770418 | 0.467638 |
| torch_per_label_softmax_global(bonsai,fasttext,mllm) | yso-fi | 5 | 0.726390 | 0.821145 | 0.696299 | 0.803387 | 0.533387 |
| torch_per_label_softmax_global(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.441699 | 0.559549 | 0.364113 | 0.478021 | 0.268925 |
| torch_per_label_softmax_global_l2_anchor(bonsai,fasttext,mllm) | yso-en | 12 | 0.719316 | 0.814469 | 0.652481 | 0.769640 | 0.465023 |
| torch_per_label_softmax_global_l2_anchor(bonsai,fasttext,mllm) | yso-fi | 5 | 0.727944 | 0.824020 | 0.696975 | 0.806097 | 0.534801 |
| torch_per_label_softmax_global_l2_anchor(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 3 | 0.439339 | 0.555319 | 0.361812 | 0.474325 | 0.264873 |
| torch_per_label_softmax_global_scale(bonsai,fasttext,mllm) | yso-en | 8 | 0.713549 | 0.809017 | 0.653860 | 0.769861 | 0.469596 |
| torch_per_label_softmax_global_scale(bonsai,fasttext,mllm) | yso-fi | 4 | 0.725911 | 0.822380 | 0.695032 | 0.803800 | 0.533575 |
| torch_per_label_softmax_global_scale(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 2 | 0.441323 | 0.557100 | 0.363759 | 0.476322 | 0.267681 |
| torch_reg_mean_residual(bonsai,fasttext,mllm) | yso-en | 1 | 0.689306 | 0.781867 | 0.611709 | 0.708791 | 0.434239 |
| torch_reg_mean_residual(bonsai,fasttext,mllm) | yso-fi | 1 | 0.691964 | 0.772237 | 0.653525 | 0.736514 | 0.495014 |
| torch_reg_mean_residual(bonsai_gemma3,bonsai_ovis2,mllm) | koko | 1 | 0.416510 | 0.510249 | 0.329807 | 0.409213 | 0.247815 |

## Top 10 Models by Avg Test NDCG@10 (across datasets)

| Rank | Model | Avg Test NDCG@10 |
|------|-------|----------------|
| 1 | torch_nn_split_per_label | 0.574079 |
| 2 | torch_per_label_softmax_global | 0.572074 |
| 3 | torch_lowrank_residual_epsclamp | 0.571729 |
| 4 | torch_lowrank_residual_mix_temp | 0.571443 |
| 5 | torch_per_label_bias_global_plus_delta | 0.570993 |
| 6 | torch_per_label_softmax_global_scale | 0.570884 |
| 7 | torch_per_label_softmax_global_l2_anchor | 0.570423 |
| 8 | torch_per_label_elastic_anchor | 0.570390 |
| 9 | torch_per_label_mlp | 0.570215 |
| 10 | torch_per_label_mlp_gate_per_label | 0.569812 |

## Top 10 Models by Avg Test NDCG@1000 (across datasets)

| Rank | Model | Avg Test NDCG@1000 |
|------|-------|----------------|
| 1 | torch_per_label_softmax_global | 0.683942 |
| 2 | torch_per_label_elastic_anchor | 0.683744 |
| 3 | torch_per_label_softmax_global_l2_anchor | 0.683354 |
| 4 | torch_per_label_softmax_global_scale | 0.683328 |
| 5 | torch_mean_residual_softmax_global | 0.682992 |
| 6 | torch_per_label_bias_global_plus_delta | 0.682493 |
| 7 | torch_per_label_apex | 0.682379 |
| 8 | torch_mean_residual_softmax_global_l2_anchor | 0.682120 |
| 9 | torch_mean_residual_l2_anchor_global | 0.682116 |
| 10 | torch_per_label_l1_delta | 0.681959 |

## Top 10 Models by Avg Test F1@5 (across datasets)

| Rank | Model | Avg Test F1@5 |
|------|-------|----------------|
| 1 | torch_per_label_bias_global_plus_delta | 0.424349 |
| 2 | torch_per_label_softmax_global_scale | 0.423617 |
| 3 | torch_per_label_softmax_global | 0.423317 |
| 4 | torch_per_label_mlp_gate_per_sample | 0.422746 |
| 5 | torch_per_label | 0.422732 |
| 6 | torch_per_label_mlp_rank_bottleneck | 0.422724 |
| 7 | torch_per_label_residual_lowrank_mix_active | 0.422702 |
| 8 | torch_per_label_mlp_remove_centering | 0.422697 |
| 9 | torch_per_label_mlp | 0.422683 |
| 10 | torch_per_label_mlp_gate_per_label | 0.422663 |

## Top 10 Models by Weighted Avg (across datasets)

| Rank | Model | Weighted Avg (0.4 NDCG@10, 0.4 F1@5, 0.2 NDCG@1000) across datasets |
|------|-------|----------------|
| 1 | torch_per_label_softmax_global | 0.534945 |
| 2 | torch_per_label_bias_global_plus_delta | 0.534636 |
| 3 | torch_per_label_softmax_global_scale | 0.534466 |
| 4 | torch_per_label_elastic_anchor | 0.533815 |
| 5 | torch_per_label_mlp | 0.533528 |
| 6 | torch_per_label_softmax_global_l2_anchor | 0.533466 |
| 7 | torch_per_label_mlp_gate_per_sample | 0.533334 |
| 8 | torch_per_label | 0.533326 |
| 9 | torch_per_label_mlp_rank_bottleneck | 0.533060 |
| 10 | torch_mean_residual_softmax_global | 0.532902 |

## Top 10 Models by Weighted Avg (koko)

| Rank | Model | Weighted Avg (0.4 NDCG@10, 0.4 F1@5, 0.2 NDCG@1000) |
|------|-------|----------------|
| 1 | torch_nn_split | 0.352223 |
| 2 | torch_nn_split_per_label | 0.350607 |
| 3 | torch_per_label_softmax_global | 0.348819 |
| 4 | torch_per_label_elastic_anchor | 0.348561 |
| 5 | torch_per_label_softmax_global_scale | 0.347840 |
| 6 | torch_mean_residual_softmax_global | 0.346967 |
| 7 | torch_mean_residual_l2_anchor_global | 0.346196 |
| 8 | torch_mean_residual_softmax_global_l2_anchor | 0.346189 |
| 9 | torch_mean_residual_lowrank_mix | 0.345974 |
| 10 | torch_per_label_mlp_gate_per_sample | 0.345841 |

## Top 10 Models by Weighted Avg (yso-en)

| Rank | Model | Weighted Avg (0.4 NDCG@10, 0.4 F1@5, 0.2 NDCG@1000) |
|------|-------|----------------|
| 1 | torch_per_label_bias_global_plus_delta | 0.605080 |
| 2 | torch_per_label_softmax_global | 0.603462 |
| 3 | torch_per_label_softmax_global_scale | 0.603355 |
| 4 | torch_per_label_elastic_anchor | 0.601641 |
| 5 | torch_per_label_softmax_global_l2_anchor | 0.600930 |
| 6 | torch_per_label | 0.600189 |
| 7 | torch_per_label_mlp_gate_per_sample | 0.600174 |
| 8 | torch_per_label_mlp | 0.600173 |
| 9 | torch_per_label_mlp_gate_per_label | 0.600173 |
| 10 | torch_per_label_mlp_rank_bottleneck | 0.600148 |

## Top 10 Models by Weighted Avg (yso-fi)

| Rank | Model | Weighted Avg (0.4 NDCG@10, 0.4 F1@5, 0.2 NDCG@1000) |
|------|-------|----------------|
| 1 | torch_mean_residual_l2_anchor_global | 0.655279 |
| 2 | torch_mean_residual_softmax_global_l2_anchor | 0.655131 |
| 3 | torch_per_label_l1_delta | 0.654861 |
| 4 | torch_per_label_mlp | 0.654572 |
| 5 | torch_mean_residual_lowrank_mix | 0.654449 |
| 6 | torch_per_label_apex | 0.654447 |
| 7 | torch_mean_residual_softmax_global | 0.654202 |
| 8 | torch_per_label | 0.653989 |
| 9 | torch_per_label_mlp_rank_bottleneck | 0.653989 |
| 10 | torch_per_label_mlp_gate_per_label | 0.653988 |
