import os
import yaml
import torch
import warnings
from types import SimpleNamespace
from typing import Union

class ConfigManager:
    def __init__(self, config_path: str = './configs/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self._process_device_config()
        self._validate_config()
        # print(f"DEBUG - lr type: {type(self.config.lr)}")
        
    def _load_config(self) -> SimpleNamespace:
        # load and parse yaml file
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return SimpleNamespace(**config_dict)
    
    def _process_device_config(self):
        # device config
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                self.config.device = 'cuda:0'
            else:
                self.config.device = 'cpu'
                warnings.warn(
                    "NO CUDA DEVICE AVAILABLE! WILL USE CPU INSTEAD! IT'S HIGHLY RECOMMEND TO USE GPU DEVICE!",
                    RuntimeWarning,
                    stacklevel=2
                )
            
        # cuDNN optimization
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
    
    def _validate_config(self):
        # validate config
        if 'cuda' in self.config.device:
            device_id = int(self.config.device.split(':')[-1]) if ':' in self.config.device else 0
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"Device {self.config.device} not exists!")
            else:
                print("GPU available!")

        if self.config.batch_size <= 0:
            raise ValueError("batch_size should be greater or equal to 0!")
    
    def get_config(self) -> SimpleNamespace:
        # get unltimate config
        return self.config

    def print_config(self):
        # print current config
        print("="*40 + " Config INFO " + "="*40)
        for key, value in vars(self.config).items():
            print(f"{key:20}: {value}")
        print("="*90)

#### RUN THIS SCRIPT AS MAIN WOULD LOAD AND PRINT CONFIG INFO.
#### DO NOT RUN THIS SCRIPT IN THIS SUBDIR. RUN THIS SCRIPT IN ROOT DIR.
if __name__ == '__main__':
    config_manager = ConfigManager()
    config = config_manager.get_config()
    config_manager.print_config()