class Env(object):
    thread = 8
    word_dim = 300  # Dimension of word vector
    sgm_epochs = 1000  # Fasttext skipgram epochs
    sgm_result = 'wordvec'  # Direction of result
    sgm_ws = 5  # Size of the context window
    sgm_lr_update_rate = 100  # Change the rate of updates for the learning rate