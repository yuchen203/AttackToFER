import argparse

# parser
parser = argparse.ArgumentParser(description='attack to FER on CK+')
parser.add_argument('--model', type=str, default='VGG19', choices=['VGG19', 'Resnet18'], help='CNN architecture')
parser.add_argument('--fold', type=int, default=1, help='k fold number')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

if __name__ == '__main__':
    print(opt.model)
