## Introduction
UNET is a convolutional neural network commonly used for segmentation related. Here I have attempted to implement UNET in pytorch

## Link to the paper
```
https://arxiv.org/pdf/1505.04597.pdf
```

## Deoendencies
Install the necessary depandencies by running
```
pip3 install -r requirements.txt
```

## Files Structure
- model.py : Contains the UNET model
- summary.py : Contains the code to show the model architecture
- test.py : Test to see whether the UNET model is working properly

## Run the tests
```
python test.py
```

## Get the model summary
```
python summary.py
```

## Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1         [-1, 64, 570, 570]           1,792
              ReLU-2         [-1, 64, 570, 570]               0
            Conv2d-3         [-1, 64, 568, 568]          36,928
              ReLU-4         [-1, 64, 568, 568]               0
        DoubleConv-5         [-1, 64, 568, 568]               0
         MaxPool2d-6         [-1, 64, 284, 284]               0
            Conv2d-7        [-1, 128, 282, 282]          73,856
              ReLU-8        [-1, 128, 282, 282]               0
            Conv2d-9        [-1, 128, 280, 280]         147,584
             ReLU-10        [-1, 128, 280, 280]               0
       DoubleConv-11        [-1, 128, 280, 280]               0
        MaxPool2d-12        [-1, 128, 140, 140]               0
           Conv2d-13        [-1, 256, 138, 138]         295,168
             ReLU-14        [-1, 256, 138, 138]               0
           Conv2d-15        [-1, 256, 136, 136]         590,080
             ReLU-16        [-1, 256, 136, 136]               0
       DoubleConv-17        [-1, 256, 136, 136]               0
        MaxPool2d-18          [-1, 256, 68, 68]               0
           Conv2d-19          [-1, 512, 66, 66]       1,180,160
             ReLU-20          [-1, 512, 66, 66]               0
           Conv2d-21          [-1, 512, 64, 64]       2,359,808
             ReLU-22          [-1, 512, 64, 64]               0
       DoubleConv-23          [-1, 512, 64, 64]               0
        MaxPool2d-24          [-1, 512, 32, 32]               0
           Conv2d-25         [-1, 1024, 30, 30]       4,719,616
             ReLU-26         [-1, 1024, 30, 30]               0
           Conv2d-27         [-1, 1024, 28, 28]       9,438,208
             ReLU-28         [-1, 1024, 28, 28]               0
       DoubleConv-29         [-1, 1024, 28, 28]               0
  ConvTranspose2d-30          [-1, 512, 56, 56]       2,097,664
           Conv2d-31          [-1, 512, 54, 54]       4,719,104
             ReLU-32          [-1, 512, 54, 54]               0
           Conv2d-33          [-1, 512, 52, 52]       2,359,808
             ReLU-34          [-1, 512, 52, 52]               0
       DoubleConv-35          [-1, 512, 52, 52]               0
  ConvTranspose2d-36        [-1, 256, 104, 104]         524,544
           Conv2d-37        [-1, 256, 102, 102]       1,179,904
             ReLU-38        [-1, 256, 102, 102]               0
           Conv2d-39        [-1, 256, 100, 100]         590,080
             ReLU-40        [-1, 256, 100, 100]               0
       DoubleConv-41        [-1, 256, 100, 100]               0
  ConvTranspose2d-42        [-1, 128, 200, 200]         131,200
           Conv2d-43        [-1, 128, 198, 198]         295,040
             ReLU-44        [-1, 128, 198, 198]               0
           Conv2d-45        [-1, 128, 196, 196]         147,584
             ReLU-46        [-1, 128, 196, 196]               0
       DoubleConv-47        [-1, 128, 196, 196]               0
  ConvTranspose2d-48         [-1, 64, 392, 392]          32,832
           Conv2d-49         [-1, 64, 390, 390]          73,792
             ReLU-50         [-1, 64, 390, 390]               0
           Conv2d-51         [-1, 64, 388, 388]          36,928
             ReLU-52         [-1, 64, 388, 388]               0
       DoubleConv-53         [-1, 64, 388, 388]               0
           Conv2d-54          [-1, 8, 388, 388]             520
----------------------------------------------------------------
Total params: 31,032,200
Trainable params: 31,032,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.74
Forward/backward pass size (MB): 2412.21
Params size (MB): 118.38
Estimated Total Size (MB): 2534.33
----------------------------------------------------------------

## Reference
- https://www.youtube.com/watch?v=IHq1t7NxS8k