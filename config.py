class Env(object):
    thread = 8  # Number of thread 
    word_dim = 300  # Dimension of word vector
    sgm_epochs = 1000  # Fasttext skipgram epochs
    sgm_result = 'wordvec'  # Direction of result
    sgm_ws = 5  # Size of the context window
    sgm_lr_update_rate = 100  # Change the rate of updates for the learning rate
    
    sest_dim = 300
    sest_epochs = 5000
    sest_result = 'sentvec'
    sest_ws = 5
    sest_lr_update_rate = 100
    
    cuda = True