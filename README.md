# Code for paper Conditional Generation of 3D Brain Tumor ROIs via VQGAN and Temporal-Agnostic Masked Transformer

[Paper link](https://openreview.net/forum?id=LLoSHPorlM), accepted at MIDL 2024

## Usage

To train the 3D-VQGAN model locally, run:
```python
python3 ./train_ae_128_cond_is.py
```
You can also change the model configurations/parameters in the *cond_model_config* file in the configs folder.

To run the script in the SLURM, change the python script name and run:
```shell
sbatch ./train.sh
```
To train the temporal-agnostic masked transformer model locally, run:
```python
python3 ./train_transformer_cond.py
```
You can also run this in the server, simply change the python file name in train.sh accordingly.
You can change the hyperparameters of the transformer in the *transformer_cond_config* file in the configs folder, and also in the vqgan_transformer_cond.py file in the model folder, for some parameters with default values

To sample images from the trained transformer, run: 
```python
python3 ./sample_transformer_cond.py
```
You can change the *temperature, topp, topk* parameters in the sampling method in the *sample_cond_configs* in the configs folder, which controls the diversity and quality. The default is topp = None, topk = None, temperature = 1.0

===========================================================================

For MS-SSIM and MMD calculation, please refer to [this repo](https://github.com/cyclomon/3dbraingen), and for the FID score, the implementation is available at [here](https://github.com/mseitzer/pytorch-fid)
