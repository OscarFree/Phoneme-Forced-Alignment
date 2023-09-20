---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: wav2vec2-large-english-phoneme_v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-base960-english-phoneme_v2

This model is a fine-tuned version of [facebook/wav2vec2-large](https://huggingface.co/facebook/wav2vec2-large) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4069
- Cer: 0.0900

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 50
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Cer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 2.18          | 6.94  | 500  | 0.3118          | 0.0923 |
| 0.2622        | 13.88 | 1000 | 0.4387          | 0.1218 |
| 0.2145        | 20.83 | 1500 | 0.4441          | 0.1121 |
| 0.1429        | 27.77 | 2000 | 0.4001          | 0.1045 |
| 0.0927        | 34.72 | 2500 | 0.4692          | 0.1062 |
| 0.0598        | 41.66 | 3000 | 0.3960          | 0.0971 |
| 0.0356        | 48.61 | 3500 | 0.4069          | 0.0900 |


### Framework versions

- Transformers 4.23.0.dev0
- Pytorch 1.12.1.post201
- Datasets 2.5.2.dev0
- Tokenizers 0.12.1
