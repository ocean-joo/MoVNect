import argparse

parser = argparse.ArgumentParser(description='Train MoVNect with teacher network')

parser.add_argument('--dataset', dest='dataset', default='dummy', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--lr', dest='lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--batch-size', dest='batch-size', type=int, default=4, help='batch size')

args = parser.parse_args()

# # TODO
# To train student network, teacher network should be pretrained.
# Teacher network(VNect) is pretrained for 2D pose estimation on MPII dataset.
# And for 3D pose, Human3.6m dataset is used.

# The same goes for student network. 
# For generalizability, student network should be pretrained with 2d pose data.
# (Because most 3d data consist of indoor images)