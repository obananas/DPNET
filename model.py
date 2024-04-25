""" Componets of the model
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

# import torch_geometric.nn import Sequential, GATConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        return self.clf(x)


class GraphAttentionConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphAttentionConv, self).__init__()
        self.out_dim = out_features
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_normal_(self.weights.data)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        self.attention = AttentionInfLevel(out_features, 0.25)

    def forward(self, input, adj):
        h = torch.spmm(input, self.weights)
        h_prime = self.attention(h, adj) + self.bias
        return h_prime


class AttentionInfLevel(nn.Module):
    def __init__(self, dim_features, dropout):
        super(AttentionInfLevel, self).__init__()
        self.dropout = dropout
        self.dim_features = dim_features
        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data)
        nn.init.xavier_normal_(self.a2.data)
        self.leaky_relu = nn.LeakyReLU(0.25)

    def forward(self, h, adj):
        N = h.size()[0]
        e1 = torch.matmul(h, self.a1).repeat(1, N)
        e2 = torch.matmul(h, self.a2).repeat(1, N).t()
        e = e1 + e2
        e = self.leaky_relu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        del zero_vec
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, h)  # h' = alpha * h(hw)
        return h_prime


class StructureGuidedFeatureEnhancement(nn.Module):
    def __init__(self, hid_dim, dropout=0.5, n_heads=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.query = nn.Linear(hid_dim, hid_dim)
        self.key = nn.Linear(hid_dim, hid_dim)
        self.value = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        Q, K, V = self.query(x), self.key(x), self.value(x)
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        A = torch.matmul(Q, K_T) / self.scale
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(A, dim=-1))
        agg_feature = torch.matmul(attention, V)
        agg_feature = agg_feature.permute(0, 2, 1, 3).contiguous()
        agg_feature = agg_feature.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))
        agg_feature = self.do(self.fc(agg_feature))
        return agg_feature


class DPNET(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        # self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.GraphAttentionConv = nn.ModuleList(
            [GraphAttentionConv(in_dim[view], hidden_dim[0]).to(device) for view in range(self.views)])
        self.QMappLayer = nn.ModuleList(
            [LinearLayer(in_dim[view] + hidden_dim[0], hidden_dim[0]) for view in range(self.views)])
        self.QMappLayer = nn.ModuleList(
            [LinearLayer(in_dim[view] + hidden_dim[0], hidden_dim[0]) for view in range(self.views)])
        self.FeatureEnhancement = nn.ModuleList(
            [StructureGuidedFeatureEnhancement(hidden_dim[0]) for _ in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.mlp = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, adj_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            # FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            feature[view] = F.leaky_relu(self.GraphAttentionConv[view](data_list[view], adj_list[view]), 0.25)
            feature[view] = self.mlp[view](data_list[view]) + feature[view]
            feature[view] = self.FeatureEnhancement[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        for view in range(self.views):
            MMLoss = MMLoss  # +torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss
        return MMLoss, MMlogit

    def infer(self, data_list, adj_list):
        MMlogit = self.forward(data_list, adj_list, infer=True)
        return MMlogit
