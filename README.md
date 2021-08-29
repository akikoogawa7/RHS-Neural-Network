# RHS Plant Classification Model
Building a multi-classification model using Convolutional Neural Networks (CNNs)

<img src="/imgs/0.jpg" alt="plant" width="400"/>


### Description
For this deep learning project, I decided to make use of the images which I had collected from an earlier web scraping project, from the [Royal Horticultural Society database](https://www.rhs.org.uk/Plants/Search-Results?form-mode=true&context=l%3Den%26q%3D%2523all%26sl%3DplantForm). 
<br><br>
This CNN model and MVP (Minimum Viable Product) hopes to accurately classify the 3147 different images of plants. However, because of the large amount of possible plant images to be potentially classified, the default number of classes has been set to `n_classes = 50`.<br>
The number of images have been batched to a length of 64 per batch and are shuffled.<br>
The learning rate and number of epochs have been adjusted to attain the highest accuracy score. Gradient descent optimiser Adam (Adaptive Moment Estimation) has been used throughout training.

### Process
- Preprocessed image folders: Firstly, the images downloaded were preprocessed in order to label them based on their actual labels.
- Converted images to tensor: Images were then transformed into tensors using PIL.
- Created dataset class: Transformed the dataset to fit the requirements of the model.<br>
``default_transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.CenterCrop(4),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])``
- CNN class built using Conv2d, BatchNorm2d, ReLU and Dropout. Fully connected networks built using Linear, ReLU and Softmax. 
`in_channels = 3, kernal_size = 5`
<br>
Batch runs were logged via tensorboard, to interactively observe which loss curve had the most steepness. 

### Outcome
So far the highest accuracy score is 53%, that is with the learning rate - 0.001 run for 1000 epochs.