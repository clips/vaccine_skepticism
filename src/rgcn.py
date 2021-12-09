import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from .util import evaluate

class RGCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes):
        super(RGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_feats, n_classes) for name in g.etypes
            })

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype])
            g.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        g.multi_update_all(funcs, 'sum')
        return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}

class RGCN(nn.Module):
    def __init__(self, 
                 g, 
                 in_feats, 
                 n_hidden, 
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 category):
        super(RGCN, self).__init__()

        embed_dict = {ntype : nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), in_feats))
                      for ntype in g.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        self.activation = activation
        self.category = category
        
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(RGCNLayer(g, in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(RGCNLayer(g, n_hidden, n_hidden))
        # output layer
        self.layers.append(RGCNLayer(g, n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h_dict = layer(g, h_dict)
                h_dict = {k: self.dropout(h) for k, h in h_dict.items()}
            else:
                h_dict = layer(g, self.embed)
                h_dict = {k : self.activation(h) for k, h in h_dict.items()}
        return h_dict[self.category]
    
def train(g, labels, args):
    category = 'user'
    
    n_classes = labels.unique().size(dim=0)
    n_nodes = g.num_nodes(category)
    n_edges = g.number_of_edges()
    
    n_train = int(n_nodes * args.train_size)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:] = True
    
    print("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Train samples %d
      #Val samples %d""" %
          (n_nodes,
           n_edges,
           train_mask.int().sum().item(),
           val_mask.int().sum().item()))

    # create model
    model = RGCN(g,
                 args.n_hidden,
                 args.n_hidden,
                 2,
                 0,
                 F.relu,
                 args.dropout,
                 category)

    # use cross entropy loss
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.n_epochs):
        
        # forward
        logits = model(g)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = evaluate(model, logits, labels, val_mask)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}". format(epoch, loss.item(), acc))