# Dnn_watermarking_project
This is the official repository for our project titled **"Securing Machine Learning Models against Tampering Attacks through Watermarking"**.

**Authors:** Amogh Manoj Joshi & Satyam Raj</br>
**Course:** 598 ADIP, Fall 2023

### Requirements

- Python 3.x
- torchvision
- vit_pytorch

### Usage Instructions
(Model_name is the only mandatory argument), all other arguments are optional

For watermarking:
```python
!python codes/watermark.py --model_name='ResNet18' 
```

For attacking the watermarked model (the code automatically fetches for the saved weight file in the models folder, alternatively you can also pass the model path):
```python
!python codes/attack.py --model_name='ResNet18'
```
### Optional Arguments
```python
- task: 'train_and_embed': Trains a model on the original dataset, and then finetunes it on the trigger set
        'embed': Directly finetunes the model on the trigger set (assumes the model has been trained on the original dataset)
         'verify': Evaluates the performance of a model on the test set and trigger set
- dataset_name: ['CIFAR-10', 'EuroSAT'] (Can be extended for more datasets also)
- trigger_path: Path to the signature image
- trigger_pos: Spacial position of the trigger in the image (of the format (x,y))
- trigger_size: Size of the trigger in the image (of the format (s,s))
- trigger_type: One among ['Embed', 'Blend', 'Noise']
- source_label: Class to choose the images from for watermarking (takes the class index as the input)
- target_label: Class to relabel the images to after watermarking (takes the class index as the input)
- model_path: Path to the trained or watermarked model (.pth or.pt) file
- train_epochs: Number of epochs to train the model on the original dataset
- wm_epochs: Number of epochs to watermark the model using the trigger set
```
