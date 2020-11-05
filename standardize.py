import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def find_std_and_mean():
    dataset = datasets.ImageFolder('Flowers', transform = transforms.ToTensor())

    print(dataset)
    loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False
    )

    print("hallo")
    mean = 0.
    std = 0.
    nb_samples = 0.
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(mean)
    print(std)

def standardize_images():
    transform = transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.4557, 0.4188, 0.2996], std=[0.2510, 0.2236, 0.2287]))
    dataset = datasets.ImageFolder('Flowers', transform = transform)

    print(dataset)
    loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False
    )
    for images, _ in loader:
        images.normalize
def run():
    torch.multiprocessing.freeze_support()
    find_std_and_mean()
    print('loop')

if __name__ == '__main__':
    run()


