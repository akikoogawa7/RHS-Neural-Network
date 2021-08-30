# RHS Plant Classification Model
Building a multi-classification model using Convolutional Neural Networks (CNNs)

<img src="/imgs/1.jpg" alt="plant" width="400"/>

### Description
For this deep learning project, I decided to make use of the images which I had collected from an earlier web scraping project, from the [Royal Horticultural Society database](https://www.rhs.org.uk/Plants/Search-Results?form-mode=true&context=l%3Den%26q%3D%2523all%26sl%3DplantForm). 
<br><br>
This CNN model and MVP (Minimum Viable Product) hopes to accurately classify the 3147 different types of plants. However, because of the large amount of possible plant images to be potentially classified, the default number of classes has been set to `n_classes = 50`.<br>
The number of images have been batched to a length of 64 per batch and are shuffled.<br>
The learning rate and number of epochs have been adjusted to attain the highest accuracy score. Gradient descent optimiser Adam (Adaptive Moment Estimation) has been used throughout training.

### Process
#### Data Preprocessing
- Preprocessed image folders: Firstly, the images downloaded were preprocessed in order to label them based on their actual labels.
- Converted images to tensor: Images were then transformed into tensors using PIL.
- Created dataset class: Augmented images<br>
`transforms.RandomRotation(180),
transforms.CenterCrop(4),
transforms.Resize([64, 64]),
transforms.ToTensor()`

<img src="/imgs/before_transform.jpg" alt="before transform" width="300"><img src="/imgs/after_transform.jpg" alt="after transform" width="250">

- Split data into training/validation set with ratio of 80/20.

#### Feature Extraction
- CNN class built using Conv2d, BatchNorm2d, LeakyReLU and Dropout. Fully connected networks built using Linear, LeakyReLU and Softmax. 
- The negative slope for LeakyReLU was set to 0.01.
#### Regression Analysis
- Batch runs were logged via `tensorboard`.
- Accuracy score was used from `torchmetrics`.
### Outcome
Highest accuracy score so far for the training set is 62.5% with `lr = 0.001`, `epochs = 1000`, `kernal_size = 5`. However validation set accuracy is 21.9% which means the model has overfit the training data too well. This might be due to the small training set which only contains 168 images.
### Next Steps
- Apply more image augmentation
- Decrease learning rate
- Train with more images / larger dataset
- Adjust dropout probability from 0.5