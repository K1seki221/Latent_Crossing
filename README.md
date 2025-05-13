# LaX: Official Implementation

## ViTs Pre-training

Prepare your ImageNet-1K dataset under `{data_dir}`.

Navigate to the ViT pre-training directory by `cd vits_pretraining`

### Launch Pre-training

Run the following command to start pre-training:

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

### Dense Model Settings

For dense baselines:

* Set `{learning_rate}` to `3e-3`
* Choose `{model_name}` from:

  * `vit_base_patch16_224`
  * `vit_large_patch16_224`

### Low-Rank Model Settings

For low-rank models, use `{learning_rate} = 1e-3`. For example, to pre-train LaX-CoLA ViT-B:

```bash
bash ./ddp_pretrain_lax_vit.sh 8 \
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

### Supported Model Variants

* **SVD / LaX-SVD:**

  ```
  Plain_CoLA_base_16_224_Ablation_SVD
  Plain_CoLA_base_16_224_Ablation_LaxSVD
  Plain_CoLA_large_16_224_Ablation_SVD
  Plain_CoLA_large_16_224_Ablation_LaxSVD
  ```

* **Tensor Train (TT) / LaX-TT:**

  ```
  Plain_TT_4cores_vit_base
  Lax_TT_4cores_vit_base
  Plain_TT_4cores_vit_large
  Lax_TT_4cores_vit_large
  ```

* **CoLA / LaX-CoLA:**

  ```
  Plain_CoLA_vit_base_16_224
  Lax_CoLA_vit_base_16_224
  Plain_CoLA_vit_large_16_224
  Lax_CoLA_vit_large_16_224
  ```
## LLMs Pre-training
Coming Soon.

## LaX-LoRA Fine-Tuning
Coming Soon.


