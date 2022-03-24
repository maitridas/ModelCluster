import torch
from model import UNET

def test():
    # batchsize, inputchannels, size, size
    print("Running the tests -------->")
    x = torch.randn((3, 3, 572, 572))
    model = UNET(in_c=3, out_c=8)
    preds = model(x)
    print("Size of the input:", x.shape)
    print("Size of the output:", preds.shape)

if __name__=="__main__":
    test()