# LaX official implementation.

## ViT Pre-training
Prepare your ImageNet-1k at {data_dir}.
`cd vits_pretraining`
Run the pre-training via:
```
  bash ./ddp_pretrain_vits.sh 8 \
  --data-dir {data_dir} \
  --model {model_name} \
  --batch-size 128 \
  --epochs 300 \
  --opt adamw \
  --weight-decay 0.3 \
  --warmup-epochs 10 \
  --sched cosine \
  --lr {learning_rate} \
  --pin-mem \
  --workers 8 \
  --fast-norm \
  --aa original \
  --mixup 0.2
```
For dense models, replace {learning_rate} by 3e-3, and {model_name} by:
`vit_base_patch16_224`
or
`vit_large_patch16_224`

For low-rank models, the {learning_rate} is 1e-3, use the follows to train:
For example, train LaX-CoLA ViT-B, run:
```
bash ./ddp_pretrain_vits.sh 4 \
  --data-dir {data_dir} \
  --model Lax_CoLA_vit_base_16_224 \
  --batch-size 128 \
  --epochs 300 \
  --opt adamw \
  --weight-decay 0.3 \
  --warmup-epochs 10 \
  --sched cosine \
  --lr 1e-3 \
  --torchcompile \
  --pin-mem \
  --workers 8 \
  --fast-norm \
  --aa original \
  --mixup 0.2
```
For SVD and LaX-SVD, replace {model_name} by one of the follows:
`[Plain_CoLA_base_16_224_Ablation_SVD, Plain_CoLA_base_16_224_Ablation_LaxSVD，Plain_CoLA_large_16_224_Ablation_SVD, Plain_CoLA_large_16_224_Ablation_LaxSVD]`

For TT and LaX-TT, replace {model_name} by one of the follows:
`[Plain_TT_4cores_vit_base, Lax_TT_4cores_vit_base, Plain_TT_4cores_vit_large, Lax_TT_4cores_vit_large]`

For CoLA and LaX-CoLA, replace {model_name} by one of the follows:
`[Plain_CoLA_vit_base_16_224, Lax_CoLA_vit_base_16_224, Plain_CoLA_vit_large_16_224, Lax_CoLA_vit_large_16_224]`


