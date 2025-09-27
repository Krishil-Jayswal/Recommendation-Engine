from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from torchinfo import summary

class NodeEmbedding(nn.Module):
    """
    Projects the node features into embedding space.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj_feats = nn.Linear(in_features, out_features)
    def forward(self, node_feats):
        return self.proj_feats(node_feats)

class ConvLayer(nn.Module):
    """
    A single layer of message passing, aggregation, updation and activation.
    """
    def __init__(self, hidden_dim: int, dropout):
        super().__init__()
        conv = SAGEConv(hidden_dim, hidden_dim)
        self.conv = HeteroConv({
            ("user", "rates", "book"): conv,
            ("book", "rev_rates", "user"): conv
        }, aggr='sum')
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.conv(x_dict, edge_index_dict)
        out_dict = {k: self.act(v) for k, v in out_dict.items()}
        out_dict = {k: self.dropout(v) for k, v in out_dict.items()}

        return out_dict

class Recommender(nn.Module):
    """
    Simple GCN-based encoder.
    """
    def __init__(self, dim_dict, num_layers=3, dropout=0.2):
        super().__init__()
        self.user_proj = NodeEmbedding(dim_dict["user"], dim_dict["hidden"])
        self.book_proj = NodeEmbedding(dim_dict["book"], dim_dict["hidden"])
    
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ConvLayer( dim_dict["hidden"], dropout=dropout))

        self.user_norm = nn.LayerNorm(dim_dict["hidden"])
        self.book_norm = nn.LayerNorm(dim_dict["hidden"])

    def forward(self, data: HeteroData):
        x_dict = {ntype: data[ntype].x for ntype in data.node_types}
        x_dict["user"] = self.user_proj(x_dict["user"])
        x_dict["book"] = self.book_proj(x_dict["book"])

        edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        x_dict["user"] = self.user_norm(x_dict["user"])
        x_dict["book"] = self.book_norm(x_dict["book"])

        return x_dict
    
if __name__ == "__main__":
    dim_dict = {"user": 384, "book": 387, "hidden": 512}
    
    model = Recommender(dim_dict)
    summary(model)