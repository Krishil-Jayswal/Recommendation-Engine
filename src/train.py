import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import Recommender
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ==============================
# Config
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = os.path.join(os.getcwd(), "data", "books", "processed", "graph.pt")
models_dir = os.path.join(os.getcwd(), "models")
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

epochs = 20
batch_size = 4096
lr = 1e-3
weight_decay = 1e-5
save_epoch = 2

# ==============================
# Data Loading
# ==============================

data = torch.load(data_dir, map_location="cpu", weights_only=False)

split = RandomLinkSplit(
    num_val=0.2,
    num_test=0.1,
    is_undirected=False,
    add_negative_train_samples=False,
    edge_types=[("user", "rates", "book")],
    rev_edge_types=[("book", "rev_rates", "user")]
)

train, val, test = split(data)

train_loader = LinkNeighborLoader(
    train,
    edge_label_index=("user", "rates", "book"),
    num_neighbors={
        ("user", "rates", "book"): [10, 10, 10],
        ("book", "rev_rates", "user"): [10, 10, 10]
    },
    batch_size=4096,
    shuffle=True
)

# ==============================
# Model and Optimizer
# ==============================

dim_dict = {
    "user": data["user"].x.size(1),
    "book": data["book"].x.size(1),
    "hidden": 512
    }

model = Recommender(dim_dict).to(device)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
writer = SummaryWriter(logs_dir)

# ==============================
# Training Loop
# ==============================

global_step = 0
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        h = model(batch)
        edge_label_index = batch[("user", "rates", "book")].edge_label_index
        
        pos_user_idx, pos_book_idx = edge_label_index
        pos_score = (h["user"][pos_user_idx] * h["book"][pos_book_idx]).sum(dim=-1)
        
        neg_user_idx, neg_book_idx = negative_sampling(
            edge_index=batch[("user", "rates", "book")].edge_index,
            num_nodes=(batch["user"].num_nodes, batch["book"].num_nodes),
            num_neg_samples=edge_label_index.size(1)
        )
        neg_score = (h["user"][neg_user_idx] * h["book"][neg_book_idx]).sum(dim=-1)
        
        loss = -F.logsigmoid(pos_score-neg_score).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        pbar.set_postfix({"batch_loss": loss.item()})
        writer.add_scalar("Loss/Batch", loss.item(), global_step)

    avg_loss = total_loss/len(train_loader)
    print(f"Epoch: {epoch} | Loss: {avg_loss:.4f}")

    writer.add_scalar("Loss/Epoch", avg_loss, epoch)

    if epoch%save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(models_dir, f"recommender_{epoch:02}.pt"))
        print(f"Checkpoint saved for epoch: {epoch}.")