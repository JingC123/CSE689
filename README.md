## Archery-Cloth Simulation
  <div align=center><img width="650" src="./imgs/bird.jpg"/></div>

### Video
<video src="./video.mov" width="320" height="200" controls preload></video>
### Problem description
We were doing image recognition of vehicles and using the data from Kaggle and Cifar. This recognition includes convolutional neural networks in pytorch. However, once the professor announced the bird classification challenge, we understood that we can use the same skill for classifying different types of birds. We can use a neural network to train the bird dataset with up to 38562 images in it and see how accurate our model is. In addition to our accuracy, we also can compete with others training accuracy in Kaggle. This can help us to think about how to improve our model, including changing the number of epochs or the size of learning rate.  
  
Here is the link of our Github Repo: [Github website](https://github.com/JingC123/CSE455_Project)

### Related work

**ResNet**: PyTorch's ResNet model was used to be our pretrained model. Resnet models were proposed in “Deep Residual Learning for Image Recognition”. There are several versions of resnet models which contain different layers respectively. In the kaggle competition, “resnet18”(vision:v0.6.0) which contains 18 layers, was used as an example of the pretrained model. Detailed model architectures and accuracy can be found online. We tested different versions of resnet pretrained model to get the best accuracy result for the competition problem. 
  <div align=center><img width="650" src="./imgs/resnet.png"/></div>

**Dataset**: We are using the kaggle bird identification dataset provided by the instructor
There are 555 different images categories (birds) given by integers [0-554].
Here is the link: [https://www.kaggle.com/c/birds21wi](https://www.kaggle.com/c/birds21wi)

### Results
The training plan we used in the end is:  
Resnet101 pretrained model with total 20 epochs.  
1. In the first 15 epochs, we set the lr = 0.01 and decay = 0.0005.
2. The next 5 epochs, we set the lr=0.001 and decay remains the same.
  <div align=center><img width="650" src="./imgs/new_losses_2.png"/></div>
  <div align=center><img width="650" src="./imgs/new_kaggle.png"/></div>
In the end, the score we got is 0.85, and we are 2nd in the competition :)


### Expectation
While we can receive a relatively accurate model, there are still places to improve. First, the data converting process does cannot be applied to the final model due to hardware limitations. If the images are resized to 256\*256, the size of the dataset will be too big to store in Google Drive or to load into RAM directly. Second, the size of images can be larger if we have more VRAM. Finally, since we now have a more complicated data input, we assume that using a more complicated model might improve the result more than what we tested in small images.   
