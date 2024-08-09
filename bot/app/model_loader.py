import os
import torch
from ultimate_accompaniment_transformer.x_transformer_1_23_2 import *
from pathlib import Path
from huggingface_hub import hf_hub_download

def load_model():
    # Hardcoded parameters
    dim = 2048
    depth = 4
    heads = 16
    model_precision = "bfloat16"
    models_dir_relative = "../repos/Ultimate-Accompaniment-Transformer/Models"
    full_path_to_models_dir = str(Path(models_dir_relative).resolve())
    model_checkpoint_file_name = 'Ultimate_Accompaniment_Transformer_Small_Improved_Trained_Model_13649_steps_0.3229_loss_0.898_acc.pth'
    model_path = f"{full_path_to_models_dir}/Small_Improved/{model_checkpoint_file_name}"

    # Ensure the model is downloaded
    if not os.path.isfile(model_path):
        hf_hub_download(
            repo_id='asigalov61/Ultimate-Accompaniment-Transformer',
            filename=model_checkpoint_file_name,
            local_dir=f"{full_path_to_models_dir}/Small_Improved",
            local_dir_use_symlinks=False
        )

    # Model instantiation
    device_type = 'cuda'
    dtype = 'bfloat16' if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    SEQ_LEN = 8192  # Model's seq len
    PAD_IDX = 767  # Model's pad index

    # Instantiate the model
    model = TransformerWrapper(
        num_tokens=PAD_IDX + 1,
        max_seq_len=SEQ_LEN,
        attn_layers=Decoder(dim=dim, depth=depth, heads=heads, attn_flash=True)
    )

    model = AutoregressiveWrapper(model, ignore_index=PAD_IDX)
    model.cuda()

    # Load model checkpoint
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model
