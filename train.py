import argparse

parser = argparse.ArgumentParser(description='Train MoVNect with teacher network')

parser.add_argument('--dataset', dest='dataset', default='dummy', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--lr', dest='lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--batch-size', dest='batch-size', type=int, default=4, help='batch size')

args = parser.parse_args()

# # TODO
#    - implement distillation network part
#    - pretrain network 2D pose estimation part with 2D pose dataset
#    - train pretrained network 3D pose estimation part with 3D pose dataset
