import torch
from typing import Tuple


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset class for text classification."""
    
    def __init__(self,text,target):
        self.text = text
        self.target = target
        
    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.text)
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        """Return item at given index."""
       
        text = self.text[index]
        target = self.target[index]
        
        return text, target
       