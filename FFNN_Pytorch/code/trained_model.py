import scipy.io as sio
import torch



# load the MNIST dataset
data = sio.loadmat('MNIST.mat')

loaded_net = torch.load('best_network.dat')
# True labels.
numbers = data['output_labels']
numbers = torch.LongTensor(numbers).squeeze()

# Handwritten digits.
images = data['input_images'] 
images  = torch.Tensor(images)



accuracy = loaded_net.accuracy(x_test=images,y_test=numbers)

print("ACCURACY BEST HP ON MNIST DATASET :", round(accuracy,2), "%")