# Pytorch版UNet，包含知识蒸馏

## Requirements
* `torch==0.4.0`
* `torchsummary==1.5`
* `tqdm`

## Training the model
First, we run the model on teacher mode and student mode：
```
usage: python train.py --mode MODE 

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Choose training mode, student or teacher
``` 
For example, to run the model on teacher mode, you would type:
```
python train.py --mode=teacher
```
After training the student model and the teacher model, we trained the distillation model.
```
python train_dis.py 
```
## Test the model
```
usage: python predict.py --mode MODE

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Which model will be predicted?
```
For example, to predict the model on teacher mode, you would type:
```
python predict.py --mode=teacher
```
## Get snr and lsd matrics
```
python get_snr_lsd_score.py
```