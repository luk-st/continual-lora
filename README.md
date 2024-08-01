# ZipLora Analysis

## LoRA models

Training LoRA subject model:

```python
sh scripts/train_lora_subject.sh
```

Training LoRA style model:

```python
sh scripts/train_lora_style.sh
```

Sampling LoRA model:

```python
python lora/sample_lora.py --pretrained_model_name_or_path lukasz-staniszewski/dog_subject  --use_cuda --use_refiner --prompt "a sbu man playing on a piano" --save_path /net/tscratch/people/plglukaszst/projects/ziplora-analysis/outputs/ls_subject/playing_piano.png
```

## Downlad SCD

```sh
pip install gdown
gdown 1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46
mv checkpoint.pth res/csd_checkpoint.pth
```