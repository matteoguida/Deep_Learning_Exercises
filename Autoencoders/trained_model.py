"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Matteo Guida

Assigment 4: Images reconstruction with an autoencoder (AE), denoising autoencoder (DAE) and generative capabilities exploration sampling from latent space.

    - The script loaded the best found hypothesis returns the mean reconstruction error MNIST dataset.

 
"""
import torch 
from torchvision import transforms
import scipy.io as sio
from torch.utils.data import DataLoader#,Dataset




class mat_to_dataloader():
    def __init__(self, filepath, transform=None):
        data=sio.loadmat(filepath)
        self.images = data['input_images']
        self.numbers = data['output_labels'].astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx].reshape((28,28))
        number = self.numbers[idx][0]
        sample = [image, number]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample

if __name__ == "__main__":

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = mat_to_dataloader(filepath='MNIST.mat', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)


    loaded_net = torch.load('./models/best_network.dat')
    mean_loss = loaded_net.test_epoch(test_dataloader=test_dataloader)
    print("\nMEAN LOSS ON THE IMAGES PRESENT IN MNIST.mat FILE :", round(float(mean_loss.numpy().flatten()),3))
    print("\n For the other requested results please check EX4_Matteo_Guida.ipynb")

