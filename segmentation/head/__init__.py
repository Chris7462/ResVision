"""
Segmentation head module
Provides FCN decoder/head architectures for semantic segmentation
"""

from .decoder import FCNDecoder


__all__ = [
    'FCNDecoder'
]