import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv

from .util import evaluate

class GCN(nn.Module):
    """Graph Convolutional Network
    - Paper: https://arxiv.org/abs/1609.02907
    - Code: https://github.com/tkipf/gcn
    """
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

def train(g, labels, args):

    g = dgl.add_self_loop(g)

    features = g.ndata['features']
    in_feats = features.shape[1]

    n_classes = labels.unique().size(dim=0)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    
    n_train = int(n_nodes * 0.6)
    n_val = int(n_nodes * 0.2)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    print("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_nodes,
           n_edges,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    g.ndata['norm'] = norm.unsqueeze(1)

    # create model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
        
    # use cross entropy loss
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.n_epochs):
        model.train()

        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(model, logits, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}". format(epoch, loss.item(), acc))

    print()
    logits = model(features)
    loss = loss_fcn(logits[test_mask], labels[test_mask])
    acc = evaluate(model, logits, labels, test_mask)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(acc, loss.item()))