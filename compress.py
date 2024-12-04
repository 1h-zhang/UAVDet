import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'F:\deeplearningnet\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml',
        'data':r'F:\deeplearningnet\ultralytics-main\VisDrone.yaml',
        'imgsz': 640,
        'epochs': 250,
        'batch': 4,
        'workers': 2,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov8-N-lamp-exp',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 1.5,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': r'F:\deeplearningnet\ultralytics-main\ultralytics\cfg\hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)