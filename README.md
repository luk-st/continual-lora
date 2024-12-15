# ZipLora Analysis

## Datasets
1. Download the [Unlearn Dataset](https://drive.google.com/drive/folders/1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM) and place it in the /data/style_unlearn directory.

2. To generate style datasets, run:
```python
python preprocess/generate_style_dataset.py
```

3. To generate object datasets, run:
Generate dataset for objects:
```python
python preprocess/generate_object_dataset.py
```


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

## Credits
The repository contains code from [task_vectors](https://github.com/mlfoundations/task_vectors) and [magmax](https://github.com/danielm1405/magmax).


## IMPORTANT

In the file: `python3.11/site-packages/diffusers/pipelines/pipeline_utils.py`, line ~278 (`DiffusionPipeline.save_pretrained()`):
```
save_kwargs = {"max_shard_size": "15GB"}
```
