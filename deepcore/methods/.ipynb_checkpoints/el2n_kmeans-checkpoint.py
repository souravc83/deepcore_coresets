from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import torch.nn.functional as F

import faiss
from .methods_utils import euclidean_dist
from collections import defaultdict 
from datetime import datetime
import os


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


def get_selected_index_el2n(data, num_clusters, el2n_score, index, d_intermediate=512, niter=20):
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
    D, I = kmeans.index.search(data_np_small, 1)
    del D

    cluster_dict = defaultdict(list)

    for counter, val in enumerate(I):
        cluster_dict[val[0]].append(counter)

    selection_result = np.array([], dtype=np.int32)

    for cluster_num, member_list in cluster_dict.items():
        if len(member_list) == 0:
            continue 
        max_val = - np.inf 
        max_index = 0
        for val in member_list:
            if el2n_score[index[val]] > max_val:
                max_val = el2n_score[index[val]]
                max_index = val   
        selection_result = np.append(selection_result, max_index)

    return selection_result 


def k_means_greedy_el2n(matrix, budget: int, el2n_score, device, d_intermediate=512, random_seed=None, index=None, scoring_method="max_score", print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)


    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num
    
    if scoring_method == "max_score":
    	selected_index = get_selected_index_el2n(data=matrix, 
                                                 num_clusters=budget, 
                                                 el2n_score=el2n_score, 
                                                 index=index,
                                                 d_intermediate=d_intermediate)
    elif scoring_method == "weighted":
    	selected_index = get_selected_index(data=matrix, 
                                            num_clusters=budget, 
                                            d_intermediate=d_intermediate)
    
    else:
      raise ValueError(f"Scoring method is {scoring_method}")
    
    return index[selected_index]




class EL2NkMeans(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=10, el2n_repeat=2,
                 specific_model=None, balance=False, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = True, el2n_epochs = 0, d_intermediate=24,
                 scoring_method="max_score", **kwargs):

        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        self.epochs = epochs
        self.kmeans_epochs = epochs 
        self.el2n_epochs = el2n_epochs 

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = el2n_repeat

        self.balance = balance
        self.d_intermediate = d_intermediate
        
        # scoring method must be one of "weighted", "max_score"
        self.scoring_method = scoring_method
        if scoring_method not in ["weighted", "max_score"]: 
          raise ValueError(f"scoring method is {scoring_method}")

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
        
        if 'save_path' in kwargs:
            self.save_path = kwargs['save_path']
        else:
            self.save_path = None




    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

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
        
        return torch.cat(matrix, dim=0)



    def finish_run(self):

        self.model.eval()

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers,
            shuffle=False)

        sample_num = self.n_train
        num_classes = 10 # this is for CIFAR-10, need to change later

        with torch.no_grad():
            for i, (input, targets) in enumerate(batch_loader):
            
                outputs = self.model(input.to(self.args.device))
                batch_num = targets.shape[0]
                targets_onehot = F.one_hot(targets.to(self.args.device), num_classes=num_classes)
                el2n_score = torch.linalg.vector_norm(
                    x=(outputs - targets_onehot),
                    ord=2,
                    dim=1
                )

                self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num),
                self.cur_repeat] = el2n_score

        self.model.train()

        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def _get_el2n_scores(self):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.args.device)
        self.epochs = self.el2n_epochs
        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = int(time.time() * 1000) % 100000
            print(f"Run {self.cur_repeat} times for EL2N score calculation")
        
        print("El2N score calculations done")

        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        
        
        # save the El2N scores
        if self.save_path:
            time_now = datetime.now()
            filename = os.path.join(self.save_path, f'el2n_scores_{time_now}.csv')
        
            np.savetxt(filename, self.norm_mean, delimiter=',')
        
        return self.norm_mean




    def select(self, **kwargs):
        all_el2n_scores = self._get_el2n_scores()
        self.epochs = self.kmeans_epochs
        self.run()

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]

                matrix = self.construct_matrix(class_index)
                selection_result = np.append(
                    selection_result,
                    k_means_greedy_el2n(matrix, 
                                   budget=round(self.fraction * len(class_index)),
                                   el2n_score=all_el2n_scores,
                                   device=self.args.device,
                                   random_seed=self.random_seed,
                                   index=class_index,
                                   scoring_method=self.scoring_method,
                                   d_intermediate=self.d_intermediate
                    )
                )

        else:
            matrix = self.construct_matrix()
            del self.model_optimizer
            del self.model
            selection_result = k_means_greedy_el2n(matrix, budget=self.coreset_size,
                                               el2n_score=all_el2n_scores,
                                               device=self.args.device,
                                               random_seed=self.random_seed,
                                               scoring_method=self.scoring_method,
                                               d_intermediate=self.d_intermediate
                                               )
        if self.scoring_method == "max_score":
        	return {"indices": selection_result}
        elif self.scoring_method == "weighted":
            weights = all_el2n_scores[selection_result]
            return {"indices": selection_result, "weights": weights}
        else:
          raise ValueError(f"Scoring method is {self.scoring_method}")

