# Copyright (c) OpenMMLab. All rights reserved.
from .dota_metric import DOTAMetric
from .rotated_coco_metric import RotatedCocoMetric
from .EV_bench import EVBench

__all__ = ['DOTAMetric', 'RotatedCocoMetric', 'EVBench']
