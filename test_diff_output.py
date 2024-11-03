from DeepHemeTransformer import DeepHemeModule, load_model
from cell_dataloader import ImagePathDataset, custom_collate_fn

model_checkpoint_path = ""
model = load_model(model_checkpoint_path)
model.eval()
