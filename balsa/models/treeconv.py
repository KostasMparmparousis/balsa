# Copyright 2022 The Balsa Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn

from balsa.util import plans_lib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TreeConvolution(nn.Module):
    """Balsa's tree convolution neural net: (query, plan) -> value.

    Value is either cost or latency.
    """

    def __init__(self, feature_size, plan_size, label_size, version=None):
        super(TreeConvolution, self).__init__()
        # None: default
        assert version is None, version
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )
        self.conv = nn.Sequential(
            TreeConv1d(32 + plan_size, 512),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(512, 256),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(256, 128),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeMaxPool(),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )
        self.reset_weights()
        self._register_embedding_hook()
        self.final_plan_embedding = None

    def _register_embedding_hook(self):
        def hook(module, inputs):
            # inputs[0] shape: [batch, 128]
            self.final_embedding = inputs[0].detach().cpu().numpy()
            
        # Register on the LayerNorm in out_mlp
        self.out_mlp[1].register_forward_pre_hook(hook)

    def reset_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Weights/embeddings.
                nn.init.kaiming_normal_(p, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                # nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                # assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)

    def forward(self, query_feats, trees, indexes):
        """Forward pass.

        Args:
          query_feats: Query encoding vectors.  Shaped as
            [batch size, query dims].
          trees: The input plan features.  Shaped as
            [batch size, plan dims, max tree nodes].
          indexes: For Tree convolution.

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        self.final_embedding = None
        query_embs = self.query_mlp(query_feats.unsqueeze(1)) # Check out unsqu

        query_embs = query_embs.transpose(1, 2)
        max_subtrees = trees.shape[-1]
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
        concat = torch.cat((query_embs, trees), axis=1)

        out = self.conv((concat, indexes))
        out = self.out_mlp(out)
        return out

class TreeConv1d(nn.Module):
    """Conv1d adapted to tree data."""

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.weights = nn.Conv1d(in_dims, out_dims, kernel_size=3, stride=3)

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        data, indexes = trees
        feats = self.weights(
            torch.gather(data, 2,
                         indexes.expand(-1, -1, self._in_dims).transpose(1, 2)))
        zeros = torch.zeros((data.shape[0], self._out_dims),
                            dtype=data.dtype,
                            device=data.device).unsqueeze(2)
        feats = torch.cat((zeros, feats), dim=2)
        return feats, indexes


class TreeMaxPool(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].max(dim=2).values

class TreeAct(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return self.activation(trees[0]), trees[1]


class TreeStandardize(nn.Module):
    def forward(self, trees):
        data, indexes = trees
        if torch.isnan(data).any():
            print("  [TreeStandardize-DEBUG] !!! ERROR: Input `data` to TreeStandardize contains NaN!")        
        mu = torch.mean(data, dim=(1, 2), keepdim=True)
        if torch.isnan(mu).any():
            print("  [TreeStandardize-DEBUG] !!! ERROR: Mean is NaN!")
        variance = torch.var(data, dim=(1, 2), keepdim=True, unbiased=False)
        var = torch.clamp(variance, min=1e-6)
        if torch.isnan(var).any() or (var < 1e-6).any():
            print(f"  [TreeStandardize-DEBUG] !!! WARNING: Variance is NaN or near zero. var = {var.flatten()[:5]}...")
        standardized = (data - mu) / torch.sqrt(var + 1e-5)
        if torch.isnan(standardized).any() and not torch.isnan(data).any():
            print("  [TreeStandardize-DEBUG] !!! ERROR: NaN was CREATED during standardization!")
        return standardized, indexes

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb

# @profile
def _batch(data):
    lens = [vec.shape[0] for vec in data]
    if len(set(lens)) == 1:
        # Common path.
        return np.asarray(data)
    xs = np.zeros((len(data), np.max(lens), data[0].shape[1]), dtype=np.float32)
    for i, vec in enumerate(data):
        xs[i, :vec.shape[0], :] = vec
    return xs


# @profile
def _make_preorder_ids_tree(curr, root_index=1):
    """Returns a tuple containing a tree of preorder positional IDs.

    Returns (tree structure, largest id under me).  The tree structure itself
    (the first slot) is a 3-tuple:

    If curr is a leaf:
      tree structure is (my id, 0, 0) (note that valid IDs start with 1)
    Else:
      tree structure is
        (my id, tree structure for LHS, tree structure for RHS).

    This function traverses each node exactly once (i.e., O(n) time complexity).
    """
    if not curr.children:
        return (root_index, 0, 0), root_index
    
    # Under every Bitmap Heap Scan is a Bitmap Index Scan, these do not need to be
    # considered seperately -> directly act as if the Bitmap Heap Scan was the leaf node
    #
    if curr.node_type == 'Bitmap Heap Scan':
        return (root_index, 0, 0), root_index
        
    lhs, lhs_max_id = _make_preorder_ids_tree(curr.children[0],
                                              root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr.children[1],
                                              root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id


# @profile
def _walk(curr, vecs):
    if curr[1] == 0:
        # curr is a leaf.
        vecs.append(curr)
    else:
        vecs.append((curr[0], curr[1][0], curr[2][0]))
        _walk(curr[1], vecs)
        _walk(curr[2], vecs)


# @profile
def _make_indexes(root):
    # Join(A, B) --> preorder_ids = (1, (2, 0, 0), (3, 0, 0))
    # Join(Join(A, B), C) --> preorder_ids = (1, (2, 3, 4), (5, 0, 0))
    preorder_ids, _ = _make_preorder_ids_tree(root)
    vecs = []
    _walk(preorder_ids, vecs)
    # Continuing with the Join(A,B) example:
    # Preorder traversal _walk() produces
    #   [1, 2, 3]
    #   [2, 0, 0]
    #   [3, 0, 0]
    # which would be reshaped into
    #   array([[1],
    #          [2],
    #          [3],
    #          [2],
    #          [0],
    #          [0],
    #    ...,
    #          [0]])
    vecs = np.asarray(vecs).reshape(-1, 1)
    return vecs


# @profile
def _featurize_tree(curr_node, node_featurizer):

    def _bottom_up(curr):
        """Calls node_featurizer on each node exactly once, bottom-up."""
        if hasattr(curr, '__node_feature_vec'):
            return curr.__node_feature_vec
        if not curr.children:
            vec = node_featurizer.FeaturizeLeaf(curr)
            curr.__node_feature_vec = vec
            return vec
        left_vec = _bottom_up(curr.children[0])
        right_vec = _bottom_up(curr.children[1])
        vec = node_featurizer.Merge(curr, left_vec, right_vec)
        curr.__node_feature_vec = vec
        return vec

    _bottom_up(curr_node)
    vecs = []
    plans_lib.MapNode(curr_node,
                      lambda node: vecs.append(node.__node_feature_vec))
    # Add a zero-vector at index 0.
    ret = np.zeros((len(vecs) + 1, vecs[0].shape[0]), dtype=np.float32)
    ret[1:] = vecs
    return ret


# @profile
def make_and_featurize_trees(trees, node_featurizer):
    indexes = torch.from_numpy(_batch([_make_indexes(x) for x in trees])).long()
    trees = torch.from_numpy(
        _batch([_featurize_tree(x, node_featurizer) for x in trees
               ])).transpose(1, 2)
    return trees, indexes

# import matplotlib.pyplot as plt

# def main():
#     from torch.autograd import gradcheck
#     global DEVICE
#     DEVICE = torch.device("cpu")  # Run on CPU for safe gradcheck
#     torch.manual_seed(42)

#     # Feature dims from your featurizer
#     query_dims = 68
#     plan_dims = 128
#     label_size = 1
#     batch_size = 2
    
#     max_num_nodes = 8 
#     feature_tensor_len = max_num_nodes + 1
#     index_tensor_len = max_num_nodes * 3

#     # Build model
#     model = TreeConvolution(query_dims, plan_dims, label_size).to(DEVICE)
#     ReportModel(model)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#     criterion = nn.MSELoss()

#     # Dummy batch creation
#     # query_feats = torch.randn(batch_size, query_dims, device=DEVICE, requires_grad=True)
#     # trees = torch.randn(batch_size, plan_dims, feature_tensor_len, device=DEVICE, requires_grad=True)
#     # indexes = torch.randint(0, feature_tensor_len, (batch_size, index_tensor_len, 1), dtype=torch.long, device=DEVICE)

#     def create_structured_batch(batch_size, query_dims, plan_dims, num_nodes):
#         query_feats = torch.randn(batch_size, query_dims, device=DEVICE)
#         trees = torch.randn(batch_size, plan_dims, num_nodes + 1, device=DEVICE)
#         indexes = torch.randint(0, num_nodes + 1, (batch_size, num_nodes * 3, 1), 
#                                 dtype=torch.long, device=DEVICE)
        
#         # Create a simple, learnable rule
#         labels_raw = trees[:, 0, 1:].sum(dim=1).unsqueeze(1)
        
#         # --- START OF FIX ---
#         # Normalize the labels
#         label_mean = labels_raw.mean(dim=0, keepdim=True)
#         label_std = labels_raw.std(dim=0, keepdim=True)
#         labels = (labels_raw - label_mean) / (label_std + 1e-6) # Add epsilon for stability
#         # --- END OF FIX ---
        
#         return query_feats, trees, indexes, labels

#     epochs = 50
#     loss_history = []    
#     for epoch in range (epochs):
#         opt.zero_grad()
#         query_feats, trees, indexes, labels = create_structured_batch(
#             batch_size, query_dims, plan_dims, max_num_nodes
#         )
#         output = model(query_feats, trees, indexes)
#         loss = criterion(output, labels)
#         loss.backward()
#         opt.step()
#         loss_history.append(loss.item())
                
#         if epoch % 10 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
#         # Gradient check every 10 epochs
#         if epoch % 10 == 0:
#             print("\n=== Gradient check per parameter ===")
#             for name, p in model.named_parameters():
#                 if p.grad is None:
#                     print(f"No grad for {name}")
#                 else:
#                     print(f"{name}: grad mean={p.grad.abs().mean().item():.6f}, "
#                           f"norm={p.grad.data.norm(2).item():.6f}")
#             print("")

#     # Plot after training
#     plt.figure(figsize=(8,5))
#     plt.plot(loss_history, label="Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss Curve")
#     plt.legend()
#     plt.grid(True)

#     # Save as PNG
#     plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
#     print("Loss curve saved as loss_curve.png")

# if __name__ == "__main__":
#     main()