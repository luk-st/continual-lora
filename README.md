# <h1 align="center">Low-Rank Continual Personalization of Diffusion Models</h1>



## Virtual Environment
Download the repository:
```
git clone git@github.com:lukasz-staniszewski/lora-cl.git
```

Create conda enviroment and activate it:
```
conda create -p <ENV_PATH> python=3.11
conda activate <ENV_PATH>
```

Install all dependencies:
```
pip install -r requirements.txt
```

**Note ⚠️** Change settings in the file: `python3.11/site-packages/diffusers/pipelines/pipeline_utils.py`, line ~278 (`DiffusionPipeline.save_pretrained()`):
```
save_kwargs = {"max_shard_size": "15GB"}
```


## Downlad CSD Model
Download model used for style metric:
```sh
pip install gdown
gdown 1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46
mv checkpoint.pth res/csd_checkpoint.pth
```

## Prepare Datasets
Download the [Unlearn Dataset](https://drive.google.com/drive/folders/1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM) and place it in the `/data/style_unlearn directory`.

To generate style datasets, run:
```python
python preprocess/generate_style_dataset.py
```

To generate object datasets, run:
```python
python preprocess/generate_object_dataset.py
```


## Models Training & Sampling

### Training LoRA Models

To train LoRA models for subjects and styles, run all experiments with different orders and object/style seeds:

```shell
sh slurm/run_all_objects.sh
sh slurm/run_all_styles.sh
```
To train a specific model for either objects or styles, use the following commands:
```shell
sh slurm/train_obj.sh
sh slurm/train_style.sh
```

### Sampling LoRA Models

Run sampling for all trained object or style models:
```shell
sh slurm/run_sampling_all_objects.sh
sh slurm/run_sampling_all_styles.sh
```

To sample a specific trained model:
```shell
sh slurm/run_sampling.sh
```

### Evaluating LoRA Models
Evaluate all models (by default all style models):
```shell
sh slurm/run_eval.sh
```

### Sign Conflicts Experiment

To execute the sign conflicts experiment across all models:
```shell
sh slurm/run_sign_conflicts.sh
```


## Credits
The repository contains code from [task_vectors](https://github.com/mlfoundations/task_vectors) and [magmax](https://github.com/danielm1405/magmax).
