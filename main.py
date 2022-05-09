import argparse
import torch
import torchvision.transforms as transforms
from fer import FER2013

# parser
parser = argparse.ArgumentParser(description='attack to FER on CK+')
parser.add_argument('--model', type=str, default='VGG19', choices=['VGG19', 'Resnet18'], help='CNN architecture')
parser.add_argument('--fold', type=int, default=1, help='k fold number')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

# Data
cut_size = 44
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
PublicTestset = FER2013(split='PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=0)
PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=0)

if __name__ == '__main__':
    print(opt.model)
