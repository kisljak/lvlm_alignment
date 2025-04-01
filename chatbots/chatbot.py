from abc import abstractmethod

import torch


class ChatBot:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    @abstractmethod
    def chat(self, image: str, message: str):
        pass

    @abstractmethod
    def bulk_chat(self, image: [str], message: [str]):
        assert len(image) == len(message)