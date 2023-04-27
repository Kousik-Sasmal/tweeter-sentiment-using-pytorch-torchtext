import pandas as pd
import torch

from typing import Tuple

class CustomData(torch.utils.data.Dataset):
    """Custom dataset class for text classification."""
    
    def __init__(self, csv_dir: str):
        """Initialize CustomData dataset.
        
        Args:
            csv_dir: Directory path for CSV file.
        """
        self.csv_dir = csv_dir
        self.df = pd.read_csv(self.csv_dir)
        
        self.text = self.df['tweet']
        self.label = self.df['sentiment']
        self.label_idx = self.df['label']
        
    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.df)
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        """Return item at given index.
        
        Args:
            index: Index of item to retrieve.
        
        Returns:
            text: Text string of tweet.
            label_idx: Integer index of label.
        """
        text = self.text[index]
        label_idx = self.label_idx[index]
        return text, label_idx
    