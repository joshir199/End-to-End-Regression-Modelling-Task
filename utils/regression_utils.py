from torch.autograd import Variable
from data.CustomDataset import CustomData2D


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    # Build batched samples from a CustomDataset
    x = []
    y = []
    for xi, yi in CustomData2D(batch_size):
        x.append(xi)
        y.append(yi)
    return Variable(x), Variable(y)
