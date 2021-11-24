import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, WeightBasis

from .util import evaluate

class RGCN(nn.Module):
    """Relational Graph Convolutional Network
    - Paper: https://arxiv.org/abs/1609.02907
    - Code: https://github.com/tkipf/gcn
    """
    def __init__(self,
                 g,
                 h_dim, 
                 out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 self_loop=False):
        super(RGCN, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_bases = len(self.rel_names)
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.self_loop = self_loop

        self.embed_layer = RelGraphEmbed(g, self.h_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.self_loop,
            dropout=self.dropout, weight=False))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.self_loop))

    def forward(self, h=None, blocks=None):
        if h is None:
            # full graph training
            h = self.embed_layer()
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h


class RelGraphConvLayer(nn.Module):
    """Relational Graph Convolution Layer."""
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = HeteroGraphConv({
                rel : GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RelGraphEmbed(nn.Module):
    """Embedding Layer"""
    def __init__(self,
                 g,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self):
        return self.embeds

def train(g, labels, args):
    category = 'user'
    
    n_classes = labels.unique().size(dim=0)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    n_users = g.num_nodes(category)
    
    n_train = int(n_users * 0.6)
    n_val = int(n_users * 0.2)
    train_mask = torch.zeros(n_users, dtype=torch.bool)
    val_mask = torch.zeros(n_users, dtype=torch.bool)
    test_mask = torch.zeros(n_users, dtype=torch.bool)
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

    # create model
    model = RGCN(g,
                 args.n_hidden,
                 n_classes,
                 num_hidden_layers=args.n_layers - 2,
                 dropout=args.dropout,
                 self_loop=args.self_loop)

    # use cross entropy loss
    loss_fcn = torch.nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
       
        logits = model()[category]
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        acc = evaluate(model, logits, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}". format(epoch, loss.item(), acc))
        
    print()
    logits = model()[category]
    loss = F.cross_entropy(logits[test_mask], labels[test_mask])
    acc = evaluate(model, logits, labels, test_mask)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(acc, loss.item()))
