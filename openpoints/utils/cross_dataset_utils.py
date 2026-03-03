# ============================================================
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import logging

# ============================================================
# ============================================================

SCANNET_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
]  # 20 classes

S3DIS_CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
]  # 13 classes


# ============================================================
# ============================================================
#   ScanNet ID -> S3DIS ID

SCANNET_TO_S3DIS_MAP = {
    0: 2,    # wall → wall
    1: 1,    # floor → floor
    2: 12,   # cabinet → clutter
    3: 12,   # bed → clutter
    4: 8,    # chair → chair
    5: 9,    # sofa → sofa
    6: 7,    # table → table
    7: 6,    # door → door
    8: 5,    # window → window
    9: 10,   # bookshelf → bookcase
    10: 12,  # picture → clutter
    11: 12,  # counter → clutter (could be table)
    12: 7,   # desk → table
    13: 12,  # curtain → clutter
    14: 12,  # refrigerator → clutter
    15: 12,  # shower curtain → clutter
    16: 12,  # toilet → clutter
    17: 12,  # sink → clutter
    18: 12,  # bathtub → clutter
    19: 12,  # otherfurniture → clutter
}



def get_scannet_to_s3dis_mapping_tensor(device='cuda'):
    """scannet_pred[i]  s3dis_class[mapping[i]]"""
    mapping = torch.zeros(20, dtype=torch.long, device=device)
    for scannet_id, s3dis_id in SCANNET_TO_S3DIS_MAP.items():
        mapping[scannet_id] = s3dis_id
    return mapping


# ============================================================
# ============================================================

class ZeroShotClassMapper(nn.Module):
    """

    """

    def __init__(self, source_classes=20, target_classes=13, aggregation='max'):
        """
        Args:
        """
        super().__init__()
        self.source_classes = source_classes
        self.target_classes = target_classes
        self.aggregation = aggregation

        mapping_matrix = torch.zeros(source_classes, target_classes)
        for src_id, tgt_id in SCANNET_TO_S3DIS_MAP.items():
            mapping_matrix[src_id, tgt_id] = 1.0

        self.register_buffer('mapping_matrix', mapping_matrix)

        logging.info(
            f"[ZeroShotClassMapper] : {source_classes} → {target_classes} ")
        logging.info(f"[ZeroShotClassMapper] : {aggregation}")

    def forward(self, logits):
        """

        Args:

        Returns:
        """
        original_shape = logits.shape

        if len(original_shape) == 2:
            # [N, C] -> [1, N, C]
            logits = logits.unsqueeze(0)

        B, N, C = logits.shape
        assert C == self.source_classes, f"Expected {self.source_classes} classes, got {C}"

        target_logits = torch.full(
            (B, N, self.target_classes),
            float('-inf'),
            device=logits.device,
            dtype=logits.dtype
        )

        for tgt_id in range(self.target_classes):
            src_mask = self.mapping_matrix[:, tgt_id] > 0
            if src_mask.sum() > 0:
                src_logits = logits[:, :, src_mask]  # [B, N, num_sources]

                if self.aggregation == 'max':
                    target_logits[:, :, tgt_id] = src_logits.max(dim=-1)[0]
                elif self.aggregation == 'sum':
                    target_logits[:, :, tgt_id] = src_logits.sum(dim=-1)
                elif self.aggregation == 'mean':
                    target_logits[:, :, tgt_id] = src_logits.mean(dim=-1)

        if len(original_shape) == 2:
            target_logits = target_logits.squeeze(0)

        return target_logits

    def map_predictions(self, predictions):
        """

        Args:

        Returns:
        """
        device = predictions.device
        mapping = get_scannet_to_s3dis_mapping_tensor(device)
        return mapping[predictions]


# ============================================================
# ============================================================

class ZeroShotModelWrapper(nn.Module):
    """
    """

    def __init__(self, model, aggregation='max'):
        """
        Args:
        """
        super().__init__()
        self.model = model
        self.class_mapper = ZeroShotClassMapper(
            source_classes=20,
            target_classes=13,
            aggregation=aggregation
        )

    def forward(self, data):
        """
        """
        output = self.model(data)

        if isinstance(output, dict) and 'logits' in output:
            output['logits'] = self.class_mapper(output['logits'])
        elif isinstance(output, torch.Tensor):
            output = self.class_mapper(output)

        return output


# ============================================================
# ============================================================

def print_class_mapping():
    """"""
    print("\n" + "="*60)
    print("ScanNet → S3DIS ")
    print("="*60)
    print(f"{'ScanNet ID':<12} {'ScanNet Class':<18} {'→':<3} {'S3DIS ID':<10} {'S3DIS Class':<15}")
    print("-"*60)

    for scannet_id, s3dis_id in SCANNET_TO_S3DIS_MAP.items():
        scannet_name = SCANNET_CLASSES[scannet_id]
        s3dis_name = S3DIS_CLASSES[s3dis_id]
        print(
            f"{scannet_id:<12} {scannet_name:<18} {'→':<3} {s3dis_id:<10} {s3dis_name:<15}")

    print("="*60 + "\n")


def get_mappable_classes():
    """ clutter"""
    mappable = {}
    for scannet_id, s3dis_id in SCANNET_TO_S3DIS_MAP.items():
        if s3dis_id != 12:
            mappable[SCANNET_CLASSES[scannet_id]] = S3DIS_CLASSES[s3dis_id]
    return mappable


if __name__ == "__main__":
    print_class_mapping()

    mapper = ZeroShotClassMapper()

    fake_logits = torch.randn(2, 1000, 20)  # [B=2, N=1000, C=20]
    mapped = mapper(fake_logits)
    print(f": {fake_logits.shape}")
    print(f": {mapped.shape}")

    fake_preds = torch.randint(0, 20, (1000,))
    mapped_preds = mapper.map_predictions(fake_preds)
    print(f": {fake_preds.shape} → {mapped_preds.shape}")
    print(f": [{mapped_preds.min()}, {mapped_preds.max()}]")
