from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import torch.nn.functional as F
from datetime import datetime
import os 


class GrandSecond(EarlyTrain):
    """
    This is an implementation of the grand score, in line with the implementation in the original paper.
    https://arxiv.org/pdf/2107.07075.pdf
    Also see corresponding code: https://github.com/mansheej/data_diet
    """
    
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=1, grand_repeat=2,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = grand_repeat

        self.balance = balance
        self.grand_norm_matrix = torch.zeros(self.n_train, self.repeat)
        
        # to calculate grand scores for each sample, we will need to make the batch size = 1
        self.args.selection_batch = 1
        
        if 'save_path' in kwargs:
            self.save_path = kwargs['save_path']
        else:
            self.save_path = None


    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size, batch_indices):
        if batch_idx % 5000 == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    
    def calc_grand_scores(self, batch_indices):
        # print(batch_indices)
        sample_num = batch_indices[0]
        sample_grand_vec = torch.cat(
            [param.grad.flatten() for param in self.model.parameters()] 
        )
        
        self.grand_norm_matrix[sample_num, self.cur_repeat] = torch.linalg.vector_norm(
            sample_grand_vec,
            ord=2
        )
        


    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module



    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.args.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = int(time.time() * 1000) % 100000
            print(f"Run {self.cur_repeat} times for Grand Score calculation")

        self.norm_mean = torch.mean(self.grand_norm_matrix, dim=1).cpu().detach().numpy()
        
        # save the Grand scores
        if self.save_path:
            time_now = datetime.now()
            filename = os.path.join(self.save_path, f'grand_scores_{time_now}.csv')
        
            np.savetxt(filename, self.norm_mean, delimiter=',')
        
        
        
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": self.norm_mean}
