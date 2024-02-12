from pathlib import Path

def get_config():
    return {
        "batch_size": 1000,  # Number of samples processed before the model is updated
        "num_epochs": 1,  # Total number of complete passes through the training dataset
        "lr": 10**-4,  # Learning rate for the optimizer
        "seq_len": 850,  # Fixed sequence length for model inputs
        "d_model": 256,  # Dimensionality of the model's embeddings (e.g., size of transformer model embeddings)
        "datasource": 'df',  # Name of the dataset or source of the data
        "src_1": "titles",  # Source 1
        "src_2": "abstracts",  # Source 2
        "tgt": "terms",  # Target 
        "model_folder": "weights",  # Directory to save model weights
        "model_basename": "tmodel_",  # Base name for saved model files
        "preload": "latest",  # Specifies preloading the latest model weights before training
        "tokenizer_file": "tokenizer_{0}.json",  # File path format for saving/loading the tokenizer
        "experiment_name": "runs/tmodel"  # Name for the experiment, possibly used for logging or saving outputs
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])