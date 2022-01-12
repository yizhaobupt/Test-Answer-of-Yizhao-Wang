# Question 1

### environment
```
  torch = 1.8.0
  python = 3.9.7
  numpy = 1.22.0
  torchvision = 0.9.0
```
### code

`dataloader.py`: Generates training and test set by choosing images from '0' and '7' in MNIST images to bags. According to the number of '0' in the bag, a bag is given a purity label which represents the percent of '0' in the whole bag.


`main.py`: Trains the model mentioned in the paper with the Adam optimization algorithm. In order to save the time, the training only takes 2 epochs. And the train loss of each epoch and test loss will be recorded in the `output.txt`.

`model.py`: The model is based on the 'SRTPMs'. It imports the `resnet_no_bn.py` and `distribution_pooling_filter.py`

### Note

I write the code based on the AttentionDeepMIL and SRTPMs(the paper in question 1).


* Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712. [link](https://arxiv.org/pdf/1802.04712.pdf).
