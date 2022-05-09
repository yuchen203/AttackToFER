import argparse
import torch
import torchvision.transforms as transforms

# parser
parser = argparse.ArgumentParser(description='attack to FER on CK+')
parser.add_argument('--model', type=str, default='VGG19', choices=['VGG19', 'Resnet18'], help='CNN architecture')
parser.add_argument('--fold', type=int, default=1, help='k fold number')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

# data loader
cut_size = 44
transformTrain = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transformTest = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# data set
trainSet = CK(split = 'Training', fold = opt.fold, transform=transformTrain)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=opt.bs, shuffle=True)
testSet = CK(split = 'Testing', fold = opt.fold, transform=transformTest)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=5, shuffle=False)

if __name__ == '__main__':
    print(opt.model)
