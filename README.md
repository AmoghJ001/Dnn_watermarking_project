# Dnn_watermarking_project
This is the official repository for our project titled **"Towards more robust Intellectual Property (IP) Ownership Verification of Deep Neural Networks (DNNs) through Watermarking"**.

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
