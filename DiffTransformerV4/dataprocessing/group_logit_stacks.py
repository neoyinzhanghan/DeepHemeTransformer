import os
import torch
from BMAassumptions import index_map
from BMAassumptions import removed_indices
from BMAassumptions import BMA_final_classes
from tqdm import tqdm

ungrouped_logit_stack_dir = (
    "/media/hdd3/neo/DiffTransformerV1DataMini/ungrouped_logit_stacks"
)
save_dir = "/media/hdd3/neo/DiffTransformerV1DataMini/logit_stacks"

# get a list of all the pt files in the ungrouped_logit_stack_dir
ungrouped_logit_stack_files = [
    f for f in os.listdir(ungrouped_logit_stack_dir) if f.endswith(".pt")
]


for ungrouped_logit_stack_file in tqdm(ungrouped_logit_stack_files):
    # get the logits tensor
    ungrouped_logit_stack = torch.load(
        os.path.join(ungrouped_logit_stack_dir, ungrouped_logit_stack_file)
    )

    # the ungrouped_logit_stack should have shape [num_cells, 23]
    # set the removed indices amongs the 23 classes to 0
    for removed_index in removed_indices:
        ungrouped_logit_stack[:, removed_index] = 0

    # renormalize the logits so that they sum to 1
    ungrouped_logit_stack = ungrouped_logit_stack / ungrouped_logit_stack.sum(
        dim=1, keepdim=True
    )

    # use the index map to group the logits into 9 classes

    # create a zero tensor to store the grouped logits that have the shape [num_cells, len(BMA_final_classes)]
    grouped_logit_stack = torch.zeros(
        ungrouped_logit_stack.shape[0], len(BMA_final_classes)
    )

    for key in index_map:
        for index in index_map[key]:
            grouped_logit_stack[:, key] += ungrouped_logit_stack[:, index]

    # print the shape of the grouped_logit_stack
    print(f"Shape of grouped_logit_stack: {grouped_logit_stack.shape}")

    # save the grouped_logit_stack to the save_dir
    torch.save(
        grouped_logit_stack,
        os.path.join(save_dir, ungrouped_logit_stack_file),
    )
