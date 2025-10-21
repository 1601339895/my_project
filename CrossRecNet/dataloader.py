import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Tuple
from collections import defaultdict
from torchvision.transforms import Resize

class MicroExpressionDataset:
    def __init__(self, root_dir: str, csv_path: str, au_path: str):

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_path)
        self.au = pd.read_csv(au_path)

        self.subject_data = defaultdict(list)
        for _, row in self.df.iterrows():
            flow_path = os.path.join(root_dir, row['subject'], row['expression'], 'flow.npy')
            onset_path = os.path.join(root_dir, row['subject'], row['expression'], 'onset.png')
            apex_path = os.path.join(root_dir, row['subject'], row['expression'], 'apex.png')
            AU_row = self.au[self.au['expression'] == row['expression']]
            AU_row = AU_row[AU_row['person'] == row['subject']]
            AU_row = AU_row.values.flatten()
            AUs = [ [AU_row[2],AU_row[3]],
                    [AU_row[4],AU_row[5]],
                    [AU_row[6],AU_row[7]],
                    [AU_row[8],AU_row[9]],
                    [AU_row[10],AU_row[11]],
                    [AU_row[12],AU_row[13]],
                    [AU_row[14],AU_row[15]],
                    [AU_row[16],AU_row[17]]]
            self.subject_data[row['subject']].append((onset_path, apex_path, flow_path, row['label'], AUs))

        self.all_subjects = list(self.subject_data.keys())

class FlowTestDataset(Dataset):
    def __init__(self, flow_samples: List[Tuple[str, int]], img_size: Tuple[int, int] = (224, 224)):
        self.samples = flow_samples
        self.resize = Resize(img_size)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        onset, _, flow_path, label = self.samples[idx]
        flow = torch.from_numpy(np.load(flow_path)).float().permute(2,0,1)  # [2,H,W]
        img = Image.open(onset).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0
        onset = self.resize(img_tensor)
        return onset, self.resize(flow), torch.tensor(label)

class PairTrainDataset(Dataset):
    def __init__(self, subject_data: Dict[str, List[Tuple[str, str, str, int]]], excluded_subjects: List[str], seed: int = 42, img_size: Tuple[int, int] = (224, 224)):

        self.available_data = {k:v for k,v in subject_data.items() if k not in excluded_subjects}

        self.class_data = defaultdict(lambda: defaultdict(list))
        for subject, samples in self.available_data.items():
            for onset_path, apex_path, flow_path, label, AUs in samples:
                self.class_data[label][subject].append((onset_path, apex_path, flow_path, AUs))

        self.pairs = self._generate_pairs()
        random.seed(seed)
        random.shuffle(self.pairs)
        self.resize = Resize(img_size)

    # def _generate_pairs(self) -> List:
    #     pairs = []
    #     for class_label in self.class_data:
    #         # 收集当前类别的所有样本（跨subject）
    #         class_samples = []
    #         for subject in self.class_data[class_label]:
    #             for sample in self.class_data[class_label][subject]:
    #                 class_samples.append((subject, *sample))  # (subject, onset, apex, flow)
            
    #         # 随机打乱样本顺序
    #         random.shuffle(class_samples)
            
    #         # 如果样本数量为奇数，舍弃最后一个
    #         if len(class_samples) % 2 != 0:
    #             class_samples.pop()
            
    #         # 两两配对（保证每个样本只使用一次）
    #         for i in range(0, len(class_samples), 2):
    #             subj1, onset1, apex1, flow1 = class_samples[i]
    #             subj2, onset2, apex2, flow2 = class_samples[i+1]
    #             pairs.append((onset1, apex1, flow1, onset2, apex2, flow2, class_label))
        
    #     return pairs

    def _generate_pairs(self, pairs_per_sample: int = 10) -> List:
        pairs = []
        for class_label in self.class_data:
            class_samples = []
            for subject in self.class_data[class_label]:
                for sample in self.class_data[class_label][subject]:
                    class_samples.append((subject, *sample))  # (subject, onset, apex, flow)

            if len(class_samples) < 2:
                continue

            used_indices = set()
            for i in range(len(class_samples)):
                subj1, onset1, apex1, flow1, AU1 = class_samples[i]
                
                possible_matches = [j for j in range(len(class_samples)) 
                                if j != i and j not in used_indices]
                
                random.shuffle(possible_matches)
                selected_matches = possible_matches[:pairs_per_sample]
                
                for j in selected_matches:
                    subj2, onset2, apex2, flow2, AU2 = class_samples[j]
                    pairs.append((onset1, apex1, flow1, AU1, onset2, apex2, flow2, AU2, class_label))
                    
                if len(selected_matches) == pairs_per_sample:
                    used_indices.add(i)
        
        return pairs

    # def _generate_pairs(self) -> List:
    #     pairs = []
    #     for class_label in self.class_data:
    #         subjects = list(self.class_data[class_label].keys())

    #         for i in range(len(subjects)):
    #             for j in range(i+1, len(subjects)):
    #                 subj1 = subjects[i]
    #                 subj2 = subjects[j]

    #                 for (onset1, apex1, flow1) in self.class_data[class_label][subj1]:
    #                     for (onset2, apex2, flow2) in self.class_data[class_label][subj2]:
    #                         pairs.append((onset1, apex1, flow1, onset2, apex2, flow2, class_label))

    #     return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        onset1, apex1, flow1, AU1, onset2, apex2, flow2, AU2, label = self.pairs[idx]
        
        def load_image(path):
            img = Image.open(path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0
            return self.resize(img_tensor)

        def load_flow(path):
            flow = torch.from_numpy(np.load(path)).float().permute(2,0,1)
            return self.resize(flow) 
        
        def merge_lists_to_tensor(list1, list2):
            tensor1 = torch.tensor(list1, dtype=torch.float32)  
            tensor2 = torch.tensor(list2, dtype=torch.float32)
            merged_tensor = torch.stack([tensor1, tensor2], dim=0)
            
            return merged_tensor

        return (
            torch.stack([load_image(onset1), load_image(onset2)]),  # [2,3,H,W]
            torch.stack([load_image(apex1), load_image(apex2)]),    # [2,3,H,W]
            torch.stack([load_flow(flow1), load_flow(flow2)]),      # [2,3,H,W]
            torch.tensor(label),
            merge_lists_to_tensor(AU1, AU2)                         # [2,8,2]
        )

def get_train_loader(
    dataset: MicroExpressionDataset,
    excluded_subjects: List[str],
    batch_size: int = 16,
    seed: int = 42,
    img_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
) -> DataLoader:
    train_dataset = PairTrainDataset(dataset.subject_data, excluded_subjects, seed=seed, img_size=img_size)

    def collate_fn(batch):
        onset = torch.stack([x[0] for x in batch])      # [B,2,3,H,W]
        apex = torch.stack([x[1] for x in batch])       # [B,2,3,H,W]
        flow = torch.stack([x[2] for x in batch])       # [B,2,3,H,W]
        labels = torch.tensor([x[3] for x in batch])    # [B]
        AUs = torch.stack([x[4] for x in batch])        # [B,2,8,2]
        return onset, apex, flow, labels, AUs

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

def get_test_loader(
    dataset: MicroExpressionDataset,
    test_subjects: List[str],
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
) -> DataLoader:
    test_samples = []
    for subject in test_subjects:
        if subject in dataset.subject_data:
            test_samples.extend(dataset.subject_data[subject])

    test_dataset = FlowTestDataset(test_samples, img_size=img_size)

    def collate_fn(batch): 
        onset = torch.stack([x[0] for x in batch])    # [B,3,H,W]
        flows = torch.stack([x[1] for x in batch])    # [B,3,H,W]
        labels = torch.tensor([x[2] for x in batch])  # [B]
        return onset, flows, labels

    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )