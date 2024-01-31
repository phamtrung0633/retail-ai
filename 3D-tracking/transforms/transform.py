from abc import ABCMeta, abstractmethod
import torch

class Transform(metaclass = ABCMeta):

    @property
    @abstractmethod
    def _pool(self):
        pass
    
    @property
    @abstractmethod
    def _count(self):
        pass

    def cls(self):
        return self.__class__

    def get_pool(self):
        return self.cls()._pool

    def set_pool(self, new):
        self.cls()._pool = new

    def __init__(self):
        self.cls()._count += 1
        if not self.is_initialised():
            self.initialise()

    def __del__(self):
        self.cls()._count -= 1

        if not (self.cls()._count):
            self.set_pool(None)

        # print(f"{self.cls()} destroyed, _pool: {self.cls()._pool}, _count: {self.cls()._count}")

    @abstractmethod
    def is_initialised(self):
        pass

    @abstractmethod
    def initialise(self):
        pass

    @abstractmethod
    def forward(self):
        pass

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')