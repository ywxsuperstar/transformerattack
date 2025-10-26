import argparse

parser = argparse.ArgumentParser(description='sentiment class with sst2')

parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,6,7')  # 1,2,3,4,5,6,7
parser.add_argument('-learning_rate', action="store", type=float, default=5e-5)
parser.add_argument('-num_epochs', action="store", type=int, default=30)
parser.add_argument('-train_type', action="store", type=str, default='mjp_unk_drloc')
parser.add_argument('-bestmodel_shuffle_ratio', action="store", type=float, default=0.5)
parser.add_argument('-shuffle_ratio', action="store", type=float, default=0.5)
parser.add_argument('-batch_size', action="store", type=int, default=2)
parser.add_argument('-minlen', action="store", type=int, default=180, help='min length of sentence')
parser.add_argument('-maxlen', action="store", type=int, default=512, help='max length of sentence')  # sentence是400
parser.add_argument('-use_unk', action="store", type=int, default=1, help='use unk mask')
parser.add_argument('-use_drloc', action="store", type=int, default=1, help='use drloc')
parser.add_argument('-model_type', action="store", type=str, default="bert", help='use model construct')
parser.add_argument('-data_type', action="store", type=str, default="yelp", help='data type')
parser.add_argument('-num_classes', action="store", type=int, default=5, help='num classes')
parser.add_argument('-pretrained_bert', action="store", type=str, default=None, help='pretrained_bert') 
parser.add_argument('-seeds', action="store", type=int, default=3047, help='seeds') 
parser.add_argument('-lambda_dlocr', action="store", type=float, default=1e-2, help='lambda_dlocr')
parser.add_argument('-window_size', action="store", type=int, default=16, help='window_size')
parser.add_argument('-warmup_steps', action="store", type=float, default=0.06, help='warmup_steps')

args = parser.parse_args()


