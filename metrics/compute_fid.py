from metrics.fid import *
from datasets import SplitCIFAR10, CIFAR10, SplitMNIST, MNIST

device = torch.device('cuda')
num_workers = 8
batch_size = 100
dims = 2048
image_size = 32
path1 = '/home/yy2694/continual-ddpm/images/mnist_uncond_joint/'
# path2 = ''
# dataset1 = None
dataset2 = MNIST(image_size)
# dataset2 = SplitCIFAR10(image_size, target=0)

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

def load_stats_from_file(path):
    assert path.endswith('.npz')
    with np.load(path) as f:
        m, s = f['mu'][:], f['sigma'][:]
    return m, s

def compute_stats_from_path(path):
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
    dataset = ImagePathDataset(files, transforms=transforms.ToTensor())
    m, s = calculate_activation_statistics(dataset, model, batch_size, dims, device, num_workers)
    return m, s

def compute_stats_from_dataset(dataset):
    return calculate_activation_statistics(dataset, model, batch_size, dims, device, num_workers)

m1, s1 = compute_stats_from_path(path1)
m2, s2 = compute_stats_from_dataset(dataset2)

fid_value = calculate_frechet_distance(m1, s1, m2, s2)

print('FID: ', fid_value)

# path = pathlib.Path(path)
# files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
# dataset = ImagePathDataset(files, transforms=transforms.ToTensor())