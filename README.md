# RHS Plant Classification Model
Building a multi-classification model using Convolutional Neural Networks (CNNs)

### Description
For this deep learning project, I decided to make use of the images which I had collected from an earlier web scraping project, from the [Royal Horticultural Society database](https://www.rhs.org.uk/Plants/Search-Results?form-mode=true&context=l%3Den%26q%3D%2523all%26sl%3DplantForm). 
<br><br>
This CNN model and MVP (Minimum Viable Product) hopes to accurately classify the 3147 different images of plants. However, because of the large amount of possible plant images to be potentially classified, the default number of classes has been set to 50, although <code>n_classes</code> is mutable as keyword argument.<br>
The number of images have been batched to a length of 64 per batch and are shuffled.<br>
The learning rate and number of epochs have been adjusted to attain the highest accuracy score. Different gradient descent optimisers such as Adam (Adaptive Moment Estimation) and SGD (Stochastic Gradient Descent) have been compared to achieve the best model.

### Process
Batch runs were logged via tensorboard, to interactively observe which loss curve had the most steepness. 

### Outcome
So far the highest accuracy score is 53%, that is with the learning rate - 0.001 run for 1000 epochs. 

### Evaluation

