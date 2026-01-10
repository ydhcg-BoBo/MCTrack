from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .datasets.mot_rgbt import MOT_RGBT

dataset_factory = {
  'mot_rgbt': MOT_RGBT,
}

def get_dataset(dataset):
  return dataset_factory[dataset]
