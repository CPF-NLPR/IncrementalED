import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train a neural network for incremental learning event detection')
    parser.add_argument('--data_path', type=str, default='../data/train_data', help='path to the training file')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0, help='coefficient of weight decay')
    parser.add_argument('--lr', type=float, default=2e-05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='size of the training batches')
    parser.add_argument('--dropout', type=float, default=0.1, help='the size of dropout')
    parser.add_argument('--gpu', dest="gpu", action="store_const", const=True, default=True, required=False, help='optional flag to use GPU if available')
    parser.add_argument('--bert_file', type=str, default='/bert_base/')
    parser.add_argument('--gradient_accumulation_steps', type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--warmup_proportion', default=0.1,type=float,help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--adam_epsilon", default=1e-7, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--total_cls", default=11, type=int, help="The number of labels.")
    parser.add_argument("--task_num", default=10, type=int, help="The number of tasks.")
    parser.add_argument("--max_size", default=100, type=int, help="The size of memory.")

    args = parser.parse_args()
    for arg in vars(args):
        print('{}={}'.format(arg.upper(), getattr(args, arg)))
    print('')
    return args

FLAGS = parse_args()
