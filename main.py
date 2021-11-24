import dgl
import argparse
from src import gcn, rgcn

def main(args):
    graphs, labels = dgl.load_graphs("data/vaccination.bin")
    labels = labels['glabel']
    if args.graph == 'homogeneous':
        gcn.train(graphs[0], labels, args)
    elif args.graph == 'heterogeneous':
        rgcn.train(graphs[1], labels, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--graph", type=str, default='heterogeneous',
                       help="which type of graph (homogeneous/heterogeneous) to use for user node prediction")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    
    main(args)