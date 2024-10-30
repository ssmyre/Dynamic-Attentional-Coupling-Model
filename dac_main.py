import torch
import argparse
from MMT.src.utils import *
from torch.utils.data import DataLoader
import time
# from MMT.src.model import MultiTransModelContext_Reduced
# from MMT.src.modelv2 import MultiTransModelContext_Reduced
# from MMT.src.modelv3 import MultiTransModelContext_Reduced
from MMT.src.model_color_image import MultiTransModelContext_Reduced


from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Multimodal Transformer Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MMT',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--root', type=str, default='/home/ssmyre/Multimodal_Transformer_Scott',
                    help='name of root directory')
parser.add_argument('--save_dir', type=str, default='mmt_trans_model_bird_sliding_mask',
                    help='name of folder to save model checkpoints')
# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--dataset', type=str, default='slide_bird',
                    help='dataset to use (default: mouse)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=3,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=2,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--z_dim', type=int, default = 300,
                    help='size of embedding dimension (default: 32)')

# Tuning
parser.add_argument('--batch_size', type=int, default=61, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=151,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult_pred',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())

valid_partial_mode = args.vonly + args.aonly

if valid_partial_mode == 0:
    args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = True


torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

# train_data = get_data(args, dataset, 'train')
# valid_data = get_data(args, dataset, 'valid')
# test_data = get_data(args, dataset, 'test')

train_data = get_data_color(args, dataset, 'train')
valid_data = get_data_color(args, dataset, 'valid')
test_data = get_data_color(args, dataset, 'test')


train_data = get_dataloader(train_data, args.batch_size)
valid_data = get_dataloader(valid_data, args.batch_size)
test_data = get_dataloader(test_data, args.batch_size)

train_loader = DataLoader(train_data, batch_size=1, shuffle = False, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle = False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle = False, drop_last=True)

print('Finish loading the data....')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())


####################################################################
#
# Run Model
#
####################################################################

logdirectory = hyp_params.save_dir
writer = SummaryWriter(logdirectory)

start = time.time()
model = MultiTransModelContext_Reduced(hyp_params, writer)
# model.load_state(hyp_params.save_dir+'/checkpoint_050.tar',save_dir=hyp_params.save_dir)
model.train_loop(train_loader, test_loader, hyp_params, epochs=501, test_freq=None, vis_freq=None, save_freq=50, training_prob=[0.1, 0.1, 0.8])
elapsed = time.time() - start
print('Elapsed time (mins): {:5.2f}'.format(elapsed/60))
# model.reconstruction_plot(train_loader)

if __name__ == '__main__':
    # test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
    pass

#        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
