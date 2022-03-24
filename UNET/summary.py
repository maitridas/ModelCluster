from model import UNET
from torchsummary import summary

if __name__=='__main__':
    print("Displaying the UNET architecture ------>")
    test_model = UNET()
    summary(test_model, (3, 572, 572))