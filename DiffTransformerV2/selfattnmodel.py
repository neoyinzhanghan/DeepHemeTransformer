import torch
import pytorch_lightning as pl
from torch.nn import (
    Sequential,
    Linear,
    Tanh,
    Softmax,
    Sigmoid,
    ELU,
    BatchNorm1d,
    Dropout,
    Flatten,
    MultiheadAttention,
)


# AggregateConcatenate Network. This will create a aggregator network for all cells and an
# adj_input network for all cells. Output from aggregator newtork will be aggregated
# concatenated with the output from the input network.
class AggregateConcatenate(pl.LightningModule):
    def __init__(
        self,
        init_embed_size,
        hidden_size,
        agg_out_size,
        agg_method="normal",
        dropout=0.2,
    ):
        super().__init__()

        # This dict will have the num of heads to use for each aggregation method.
        agg_dict = {
            "normal": (4, self.normal_aggregation),
        }

        self.num_heads, self.aggregation = agg_dict[agg_method]

        self.wbc_aggregators = Sequential(
            Flatten(start_dim=0, end_dim=1),
            Linear(init_embed_size, hidden_size),
            ELU(),
            BatchNorm1d(hidden_size),
            Dropout(dropout),
            Linear(hidden_size, agg_out_size),
            Tanh(),
        )

        self.wbc_adj_input_network = Sequential(
            Flatten(start_dim=0, end_dim=1),
            Linear(init_embed_size, hidden_size),
            ELU(),
            BatchNorm1d(hidden_size),
            Dropout(dropout),
            Linear(hidden_size, agg_out_size),
            Tanh(),
        )

    def normal_aggregation(self, agg_embeddings):
        # agg_embeddings is of shape B x T x agg_out_size
        # We will use (n=4) aggregation functions, mean, max and min and standard deviation which will be applied on all the queries in a bag.
        # Every bag will be represented by a tensor of shape n x agg_out_size, where every row is the aggregation of the queries in the bag.
        # The batch will be represented by a tensor of shape B x n x agg_out_size, which will consist of the bag representations of every bag in the batch.

        batch_representations = []

        batch_representations.append(torch.mean(agg_embeddings, dim=1))
        batch_representations.append(torch.max(agg_embeddings, dim=1).values)
        batch_representations.append(torch.min(agg_embeddings, dim=1).values)
        batch_representations.append(torch.std(agg_embeddings, dim=1))

        return torch.stack(batch_representations, dim=1)  # B x 4 x agg_out_size

    def generalized_mean(self, inp_tensor, power=1.0):
        # inp_tensor is of shape T x agg_out_size
        # We will take the generalized mean of the tensor along the zeroth dimension.
        # For p=0, it is the geometric mean, for p=1, it is the arithmetic mean and for p=infinity, it is the max function.
        # Output shape: agg_out_size

        n = inp_tensor.shape[0]
        generalized_mean = (1 / n * torch.sum(inp_tensor**power, dim=0)) ** (1 / power)
        return generalized_mean

    def log_sum_exponentiation(self, inp_tensor, power=1.0):
        # inp_tensor is of shape T x agg_out_size
        # We will take the log sum exponentiation of the tensor along the zeroth dimension.
        # The output shape is agg_out_size

        n = inp_tensor.shape[0]
        sum_exp = (1 / n) * torch.sum(torch.exp(power * inp_tensor), dim=0)
        log_sum_exp = (1 / power) * torch.log(sum_exp)
        return log_sum_exp

    def forward(self, wbc):
        """ """
        B = wbc.shape[0]

        wbc_agg_embeddings = self.wbc_aggregators(wbc)
        wbc_agg_embeddings = wbc_agg_embeddings.view(B, wbc.shape[1], -1)
        wbc_aggregations = self.aggregation(wbc_agg_embeddings)

        wbc_adj_inputs = self.wbc_adj_input_network(wbc)
        wbc_adj_inputs = wbc_adj_inputs.view(B, wbc.shape[1], -1)

        # Concatenate the aggregations and the adjacent inputs which can then be passed to the
        # self attention network.
        concatenated_all = torch.cat((wbc_aggregations, wbc_adj_inputs), dim=1)

        return concatenated_all


# Multi head Self Attention Model that will create a key, query, value and apply attention mechanism.
class MyMultiheadAttention(pl.LightningModule):
    def __init__(
        self,
        inp_embedding_size,
        attn_head_size,
        num_attn_heads,
        num_agg_heads,
        dropout=0.3,
    ):
        super().__init__()

        # number of aggregation heads in the concatenated length of the sequence.
        self.num_agg_heads = num_agg_heads

        self.key = Sequential(
            Linear(inp_embedding_size, 128),
            ELU(),
            Dropout(dropout),
            Linear(128, attn_head_size),
            Tanh(),
        )

        self.query = Sequential(
            Linear(inp_embedding_size, 128),
            ELU(),
            Dropout(dropout),
            Linear(128, attn_head_size),
            Sigmoid(),
        )

        self.value = Sequential(
            Linear(inp_embedding_size, 128),
            ELU(),
            Dropout(dropout),
            Linear(128, attn_head_size),
        )

        self.multihead_attention = MultiheadAttention(
            embed_dim=attn_head_size,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, input_tensor):
        # input_tensor is of shape B x L x E    where L is the length of the sequence and E is the embedding size.

        key = self.key(input_tensor)  # B x L x attn_head_size
        query = self.query(input_tensor)  # B x L x attn_head_size
        value = self.value(input_tensor)  # B x L x attn_head_size

        # Use the same input for key, query, and value
        full_output, attn = self.multihead_attention(query, key, value)

        # Extract the attn weights from the heads to the T_max cells.
        # We extract the attn values of the first 2n heads to the cell and patch embeddings.
        head_attn = attn[:, : 2 * self.num_agg_heads, 2 * self.num_agg_heads :]
        # Extract the head embeddings from the full output.
        head_output = full_output[:, : 2 * self.num_agg_heads, :]  # B x 2n x outsize

        return head_output, full_output, attn, head_attn


# MIL with self attention model
class MILSelfAttention(pl.LightningModule):
    def __init__(self, init_mil_embed, mil_head, attn_head_size, agg_method):
        super().__init__()

        agg_dict = {"normal": 4, "gm": 3, "lse": 3}

        self.num_agg_heads = agg_dict[agg_method]  # number of aggregation heads

        self.aggregation = AggregateConcatenate(
            init_mil_embed, 256, mil_head, agg_method, dropout=0.2
        )

        self.attention1 = MyMultiheadAttention(
            inp_embedding_size=mil_head,
            attn_head_size=attn_head_size,
            num_attn_heads=1,
            num_agg_heads=self.num_agg_heads,
            dropout=0.2,
        )

        self.attention2 = MyMultiheadAttention(
            inp_embedding_size=attn_head_size,
            attn_head_size=attn_head_size,
            num_attn_heads=1,
            num_agg_heads=self.num_agg_heads,
            dropout=0.2,
        )

        self.classifier = Sequential(
            Linear(2 * self.num_agg_heads * attn_head_size, 128),
            ELU(),
            BatchNorm1d(128),
            Dropout(0.3),
            Linear(128, 64),
            ELU(),
            BatchNorm1d(64),
            Dropout(0.3),
            Linear(64, 9),
            Softmax(dim=1),
        )

    def forward(self, cells):
        wbc = cells

        concatenated_all = self.aggregation(wbc)

        head_out1, full_out1, attn1, head_attn1 = self.attention1(concatenated_all)

        head_out2, full_out2, attn2, head_attn2 = self.attention2(full_out1)

        B = head_out2.shape[0]
        head_out2 = torch.flatten(head_out2, start_dim=1)

        pred_probs = self.classifier(head_out2)

        return pred_probs
