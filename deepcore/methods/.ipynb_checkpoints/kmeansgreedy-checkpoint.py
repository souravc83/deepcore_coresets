from .earlytrain import EarlyTrain
import torch
import numpy as np
import os   
import faiss
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel

def get_selected_index(data, num_clusters, d_intermediate=512, niter=20):
    verbose = True
    d = data.shape[1]
    data_np = data.cpu().numpy()
    
    # Apply PCA, is d_intermediate is smaller than d
    if d_intermediate == d:
        data_np_small = data_np
    else:
        mat = faiss.PCAMatrix (d, d_intermediate)
        mat.train(data_np)
        data_np_small = mat.apply(data_np)
      
    
    kmeans = faiss.Kmeans(d_intermediate, num_clusters, niter=niter, verbose=verbose, gpu=True)
    kmeans.train(data_np_small)
    index = faiss.IndexFlatL2 (d_intermediate)
    index.add (data_np_small)
    D, I = index.search (kmeans.centroids, 1)
    del D 
    return I.squeeze()      


def k_means_greedy(matrix, budget: int, device, d_intermediate, random_seed=None, index=None, print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")

    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)


    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num
    selected_index = get_selected_index(data=matrix, num_clusters=budget, d_intermediate=d_intermediate)
    
    return index[selected_index]


class kMeansGreedy(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=0,
                 specific_model="ResNet18", balance: bool = False, d_intermediate = 12, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = True, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)

        self.min_distances = None

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda : self.finish_run()
            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index),
                    num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)
            self.construct_matrix = _construct_matrix

        self.balance = balance
        self.d_intermediate = d_intermediate

        if 'save_path' in kwargs:
            self.save_path = kwargs['save_path']
        else:
            self.save_path = None

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size, batch_indices):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def old_construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                        torch.utils.data.Subset(self.dst_train, index),
                                                batch_size=self.args.selection_batch,
                                                num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch,
                                                             sample_num)] = self.model.embedding_recorder.embedding

        self.model.no_grad = False
        return matrix

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = []

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                    torch.utils.data.Subset(self.dst_train, index),
                                    batch_size=self.args.selection_batch,
                                    num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)

        self.model.no_grad = False

        #filename = 'embedding_matrix.pt'
        #full_fname = os.path.join(self.save_path, filename)
        #torch.save(torch.cat(matrix, dim=0), full_fname)

        return torch.cat(matrix, dim=0)

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def select(self, **kwargs):
        self.run()

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]

                matrix = self.construct_matrix(class_index)
                selection_result = np.append(
                    selection_result,
                    k_means_greedy(matrix, 
                                   budget=round(self.fraction * len(class_index)),
                                   device=self.args.device,
                                   d_intermediate=self.d_intermediate,
                                   random_seed=self.random_seed,
                                   index=class_index,
                    )
                )

        else:
            matrix = self.construct_matrix()
            del self.model_optimizer
            del self.model
            selection_result = k_means_greedy(matrix, budget=self.coreset_size,
                                               device=self.args.device,
                                               d_intermediate=self.d_intermediate,
                                               random_seed=self.random_seed,
                                               )
        return {"indices": selection_result}
