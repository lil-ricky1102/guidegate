import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
import dgl
import numpy as np
import pandas as pd
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.nn.utils.rnn import pad_sequence


class DTIDataset(data.Dataset):

    def __init__(self, df_samples,list_IDs, df,compounds,proteins, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.df_samples = df_samples
        self.compounds = compounds
        self.proteins = proteins
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        drugs_len = len(self.df_samples)
        return drugs_len

    def __getitem__(self, index):
        
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']   

        one_sample= self.df_samples[index]
        v_smiles = self.compounds[int(one_sample[1])]
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer,edge_featurizer=self.bond_featurizer)
        #Graph
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        if num_actual_nodes < self.max_drug_nodes:
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                           torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})

        v_d = v_d.add_self_loop()
        #Protein seq
        v_p = self.proteins[int(one_sample[2])]
        y = [self.df_samples[index,3]]

        return v_smiles, v_d, v_p, torch.tensor(y)


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches

def load_tensor(file_name, dtype):
    data = np.load(file_name + '.npy', allow_pickle=True)
    if data.dtype == np.object_:
        data = [np.array(d, dtype=np.float32) if isinstance(d, (list, np.ndarray)) else np.zeros_like(d, dtype=np.float32) for d in data]
    return [dtype(d) for d in data]

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



def preparedata(type,dataset):
    dir_input = ('dataset/'+dataset+'/')
    compounds = load_tensor(dir_input + 'smilesembeddings', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteinsembeddings', torch.FloatTensor) 
    trainfiles = pd.read_csv(dir_input + type + '/train/' + 'train.csv')
    validfiles = pd.read_csv(dir_input + type + '/valid/' + 'valid.csv')
    testfiles = pd.read_csv(dir_input + type + '/test/' + 'test.csv')
    trainfiles_samples = pd.read_csv(dir_input + type + '/train/' + 'samples.csv')
    validfiles_samples = pd.read_csv(dir_input + type + '/valid/' + 'samples.csv')
    testfiles_samples= pd.read_csv(dir_input + type + '/test/' + 'samples.csv')
    

    return trainfiles_samples.values,validfiles_samples.values,testfiles_samples.values,trainfiles,validfiles,testfiles,compounds,proteins


def collatef(x):
    d_smiles, d_graph, p, y = zip(*x)
    d_graph = dgl.batch(d_graph)
    reshaped_d_smiles = torch.cat(d_smiles, dim=0).reshape(-1, 24, 32)  # reshape to (N, 24, 32)
    max_prot_len = 1200
    p_list = [p1[:max_prot_len] for p1 in p]
    p_padded = pad_sequence(p_list, batch_first=True)
    return reshaped_d_smiles, d_graph, p_padded, torch.tensor(y)

def preparedataset(batch_size,type,dataset):
    trainfiles_samples,validfiles_samples,testfiles_samples,trainsamples,validsamples,testsamples,compounds,proteins=preparedata(type,dataset)
    trainloader = DataLoader(DTIDataset(trainfiles_samples,trainsamples.index.values,trainsamples,compounds,proteins),shuffle=True,batch_size=batch_size,collate_fn=collatef, drop_last=True)
    validloader = DataLoader(DTIDataset(validfiles_samples,validsamples.index.values,validsamples,compounds, proteins), shuffle=False, batch_size=batch_size,
                            collate_fn=collatef, drop_last=False)
    testloader = DataLoader(DTIDataset(testfiles_samples,testsamples.index.values,testsamples,compounds, proteins), shuffle=False, batch_size=batch_size,
                             collate_fn=collatef, drop_last=False)
    return trainloader,validloader,testloader,compounds,proteins
