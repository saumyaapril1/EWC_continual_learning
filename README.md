# EWC_continual_learning

The implementation is to understand how EWC (Elastic WeightConsolidation) helps overcome catastrophic forgetting in continual learning, hence comparing the two cases:
1) Precision when training without consolidation
2) Preciasion when training with consolidation

Following are the arguments considered during the implementation.
-hidden_size
-hidden_layer_num
-hidden_dropout_prob
-input_dropout_prob
-task_number
-epochs_per_task
-lamda
-lr
-batch_size
-weight_decay
-test_size
-random_seed
-no_gpu
-fisher_estm_sample_size
-eval_log_interval
-loss_log_interval
-consolidate_






References:  
Kirkpatrick,Pascanu et al(2017)

Copyright (c) 2017 Ha Junsoo
