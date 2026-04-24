# PeLA: Perception-enhanced Linear Attention for Lightweight Image Super-Resolution

### Installation
```
# Install dependent packages
cd PeLASR
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```
You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation

Put PELASR_arch.py to the path "basicsr/archs".

### Training
- Put yml to the path "options/train/".
- Run the following commands for training:
```python
python basicsr/train.py -opt options/train/train_PELASR_DF2K_d56n10_x4.yml
```
- X2, X3 are the same.

### Testing
- Download the pretrained models.
- Put yml to the path "options/test/".
- Run the following commands:
```python
python basicsr/test.py -opt options/test/test_PELASR_DF2K_d56n10_x4.yml
```
- X2, X3 are the same.
- The test results will be in './results'.


### Results
## Citation
If you find this repository helpful, you may cite:

```tex

```

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox.
