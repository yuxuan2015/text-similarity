from utils.config import DEVICE

DEFAULT_CONFIG = {
    'lr': 0.01,
    'epoch': 30,
    'lr_decay': 0.05,
    'batch_size': 128,
    'dropout': 0.5,
    'static': False,
    'non_static': False,
    'embedding_dim': 300,
    'num_layers': 2,
    'pad_index': 1,
    'vector_path': '',
    'tag_num': 0,
    'fix_length': 20,
    'vocabulary_size': 0,
    'word_vocab': None,
    'tag_vocab': None,
    'save_path': './saves'
}
