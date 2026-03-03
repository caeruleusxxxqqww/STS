import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from .build import LOSS
import numpy as np

# try:
#     from .build import LOSS
# except ImportError:
#     print("[Warning] Standalone mode detected. Using Mock LOSS registry.")
#     class MockRegistry:
#         def register_module(self, name=None):
#             return lambda cls: cls
#     LOSS = MockRegistry()
# # ======================================================================


def simple_knn_graph(pos, k, batch, chunk_size=512):
    """

    """
    edge_index_list = []

    if pos.shape[0] == 0:
        return torch.zeros((2, 0), device=pos.device, dtype=torch.long)

    if batch is None:
        batch_size = 1
        batch = torch.zeros(pos.shape[0], device=pos.device, dtype=torch.long)
    else:
        try:
            batch_size = int(batch.cpu().max().item()) + 1
        except:
            batch_size = 1
            batch = torch.zeros(
                pos.shape[0], device=pos.device, dtype=torch.long)

    for b in range(batch_size):
        mask = batch == b
        p = pos[mask]
        M = p.shape[0]

        current_k = k
        if M < k + 1:
            current_k = M - 1
            if current_k <= 0:
                continue

        max_chunk = max(128, min(chunk_size, 50_000_000 // (M * 4 + 1)))

        local_edges_list = []

        for i in range(0, M, max_chunk):
            end = min(i + max_chunk, M)
            p_src = p[i:end]

            with torch.no_grad():
                dist = torch.cdist(p_src, p)

                _, indices = dist.topk(current_k + 1, dim=1, largest=False)
                neighbors = indices[:, 1:]

                row_local = torch.arange(
                    i, end, device=pos.device).view(-1, 1).repeat(1, current_k).flatten()
                col_local = neighbors.flatten()

                local_edges_list.append(torch.stack([row_local, col_local]))

                del dist, indices, neighbors, p_src

        if len(local_edges_list) > 0:
            local_edge_index = torch.cat(local_edges_list, dim=1)
            del local_edges_list

            global_indices = torch.nonzero(mask).squeeze(-1)
            global_row = global_indices[local_edge_index[0]]
            global_col = global_indices[local_edge_index[1]]

            edge_index_list.append(torch.stack([global_row, global_col]))

            del local_edge_index, global_indices, global_row, global_col

        del p, mask

    if len(edge_index_list) == 0:
        return torch.zeros((2, 0), device=pos.device, dtype=torch.long)

    result = torch.cat(edge_index_list, dim=1)
    del edge_index_list
    return result


# =========================================================================
# =========================================================================
def apply_semantic_texture_swapping(data, return_info=True):
    """




    Args:

    Returns:
    """
    if not ('x' in data and 'y' in data and 'pos' in data):
        if return_info:
            data['sts_applied'] = False
        return data

    if torch.rand(1) > 0.5:
        if return_info:
            data['sts_applied'] = False
        return data

    try:
        rgb_slice = slice(0, 3)
        pos = data['pos']
        x_rgb = data['x'][:, rgb_slice]
        labels = data['y']
        N = labels.shape[0]

        if return_info:
            original_texture = x_rgb.clone()

        patch_size = 50

        unique_labels = torch.unique(labels)
        max_id = unique_labels.max()
        if max_id > 20:
            valid_candidates = unique_labels[unique_labels != max_id]
        else:
            valid_candidates = unique_labels

        if valid_candidates.numel() < 2:
            if return_info:
                data['sts_applied'] = False
            return data

        target_label = valid_candidates[torch.randint(
            0, valid_candidates.numel(), (1,)).item()]

        target_indices_all = torch.where(labels == target_label)[0]
        if target_indices_all.numel() < patch_size:
            if return_info:
                data['sts_applied'] = False
            return data

        anchor_idx = target_indices_all[torch.randint(
            0, target_indices_all.numel(), (1,)).item()]
        anchor_pos = pos[anchor_idx].unsqueeze(0)  # (1, 3)

        target_pos_all = pos[target_indices_all]  # (N_tgt, 3)
        dists = torch.cdist(anchor_pos, target_pos_all).squeeze(0)  # (N_tgt,)

        k = min(patch_size, target_indices_all.shape[0])
        _, topk_indices_local = dists.topk(k, largest=False)

        patch_indices = target_indices_all[topk_indices_local]

        source_candidates = valid_candidates[valid_candidates != target_label]
        if source_candidates.numel() == 0:
            if return_info:
                data['sts_applied'] = False
            return data

        source_label = source_candidates[torch.randint(
            0, source_candidates.numel(), (1,)).item()]
        source_indices_all = torch.where(labels == source_label)[0]

        if source_indices_all.numel() == 0:
            if return_info:
                data['sts_applied'] = False
            return data

        src_sample_idx = torch.randint(
            0, source_indices_all.numel(), (k,), device=pos.device)
        global_src_idx = source_indices_all[src_sample_idx]

        source_texture = x_rgb[global_src_idx]  # (k, 3)

        data.x[patch_indices, rgb_slice] = source_texture

        if return_info:
            sts_mask = torch.zeros(N, dtype=torch.float32, device=pos.device)
            sts_mask[patch_indices] = 1.0

            data['sts_mask'] = sts_mask
            data['sts_original_texture'] = original_texture
            data['sts_applied'] = True

        return data

    except Exception as e:
        # print(f"[STS Error] {e}")
        if return_info:
            data['sts_applied'] = False
        return data


# =========================================================================
# =========================================================================
@LOSS.register_module(name="TextureConsistencyLoss")
class TextureConsistencyLoss(nn.Module):
    """

    Ψ(G, T_sts) ≈ M ⊙ Ψ(G, T_src) + (1-M) ⊙ Ψ(G, T_tgt)


    L_cons = (1/N) * sum_i || F^spliced_i - (m_i * F^src_i + (1-m_i) * F^tgt_i) ||^2


    """

    def __init__(self, weight=0.1, mode='simplified'):
        """
        Args:
        """
        super().__init__()
        self.weight = weight
        self.mode = mode
        print(f"[L_cons] Ready. W={self.weight}, Mode={self.mode}")

    def forward(self, features_spliced, features_tgt, mask=None, features_src=None):
        """
        Args:

        Returns:
        """
        if self.mode == 'full' and features_src is not None and mask is not None:
            # L_cons = (1/N) * sum_i || F^spliced_i - (m_i * F^src_i + (1-m_i) * F^tgt_i) ||^2

            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)  # (N, 1)

            target_features = mask * features_src + (1 - mask) * features_tgt

            # MSE Loss
            loss = (features_spliced - target_features).pow(2).mean()

        else:

            if mask is not None:
                if mask.dim() == 1:
                    mask_bool = mask > 0.5
                else:
                    mask_bool = mask.squeeze(-1) > 0.5

                if mask_bool.sum() > 0:
                    loss = (features_spliced[mask_bool] -
                            features_tgt[mask_bool]).pow(2).mean()
                else:
                    loss = torch.tensor(
                        0.0, device=features_spliced.device, requires_grad=True)
            else:
                loss = (features_spliced - features_tgt).pow(2).mean()

        return self.weight * loss

    def forward_from_dict(self, output_dict_spliced, output_dict_clean=None):
        """

        Args:
                - 'features': F^spliced
                - 'features': F^tgt

        Returns:
        """
        if output_dict_clean is None:
            device = output_dict_spliced['features'].device
            del output_dict_spliced
            return torch.tensor(0.0, device=device, requires_grad=True)

        features_spliced = output_dict_spliced.get('features')
        mask = output_dict_spliced.get('sts_mask', None)
        fallback_device = output_dict_spliced.get(
            'logits', torch.empty(1)).device
        del output_dict_spliced

        features_tgt = output_dict_clean.get('features')
        del output_dict_clean

        if features_spliced is None or features_tgt is None:
            return torch.tensor(0.0, device=fallback_device, requires_grad=True)

        return self.forward(features_spliced, features_tgt, mask)


def estimate_point_normals(pos, k=16, batch=None):
    """

    Args:

    Returns:
    """
    N = pos.shape[0]
    device = pos.device

    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            edge_index = simple_knn_graph(pos, k=k, batch=batch)

            if edge_index.shape[1] == 0:
                normals = torch.zeros(N, 3, device=device)
                normals[:, 2] = 1.0
                return normals

            row, col = edge_index[0], edge_index[1]

            neighbor_pos = pos[col]  # (E, 3)
            neighbor_center = scatter(neighbor_pos, row, dim=0,
                                      dim_size=N, reduce='mean')  # (N, 3)

            centered_neighbors = neighbor_pos - neighbor_center[row]  # (E, 3)

            del neighbor_pos, neighbor_center

            cn0 = centered_neighbors[:, 0]
            cn1 = centered_neighbors[:, 1]
            cn2 = centered_neighbors[:, 2]

            xx = scatter(cn0 * cn0, row, dim=0, dim_size=N, reduce='mean')
            xy = scatter(cn0 * cn1, row, dim=0, dim_size=N, reduce='mean')
            xz = scatter(cn0 * cn2, row, dim=0, dim_size=N, reduce='mean')
            yy = scatter(cn1 * cn1, row, dim=0, dim_size=N, reduce='mean')
            yz = scatter(cn1 * cn2, row, dim=0, dim_size=N, reduce='mean')
            zz = scatter(cn2 * cn2, row, dim=0, dim_size=N, reduce='mean')

            del centered_neighbors, cn0, cn1, cn2, edge_index, row, col

            cov = torch.stack([
                torch.stack([xx, xy, xz], dim=1),
                torch.stack([xy, yy, yz], dim=1),
                torch.stack([xz, yz, zz], dim=1)
            ], dim=1)  # (N, 3, 3)

            del xx, xy, xz, yy, yz, zz

            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            normals = eigenvectors[:, :, 0].clone()

            del eigenvalues, eigenvectors, cov

            normals = F.normalize(normals, p=2, dim=-1)

        except Exception as e:
            normals = torch.zeros(N, 3, device=device)
            normals[:, 2] = 1.0

    return normals


@LOSS.register_module(name="TSILoss")
class TextureSmoothnessInvarianceLoss(nn.Module):
    def __init__(self, k_neighbors=16, weight=1.0, sigma_g=0.1, lambda_tex=5.0, lambda_geo=None, lambda_s=None, debug=False, debug_interval=500):
        """
        Semantically-Aware TSI Loss (SA-TSI)

        L_TSI = (1/N) * sum_i sum_{j in N(i)} I(y_i=y_j) * W_geo(i,j) * W_inv(i,j) * ||z_i - z_j||^2

        Args:
        """
        super().__init__()
        self.k = k_neighbors
        self.weight = weight
        self.sigma_g = sigma_g
        self.lambda_tex = lambda_tex

        self.debug = debug
        self.debug_interval = debug_interval
        self.call_count = 0

        if lambda_s is not None or lambda_geo is not None:
            print(
                f"[Warning] 'lambda_s/lambda_geo' is deprecated. Use 'sigma_g' instead.")

        print(
            f"[SA-TSI Loss] Ready. K={self.k}, W={self.weight}, sigma_g={self.sigma_g}, gamma={self.lambda_tex}, debug={self.debug}")

    def forward(self, output_dict, target=None):
        """
        Semantically-Aware TSI Loss

        L_TSI = (1/N) * sum_i sum_{j in N(i)} I(y_i=y_j) * W_geo(i,j) * W_inv(i,j) * ||z_i - z_j||^2

        """
        if 'features' not in output_dict:
            return torch.tensor(0.0, device=output_dict.get('logits', torch.empty(1)).device, requires_grad=True)

        features = F.normalize(output_dict['features'], p=2, dim=-1)
        pos = output_dict['pos'].detach()
        rgb = output_dict['rgb'].detach()
        batch = output_dict.get('batch', None)
        if batch is not None:
            batch = batch.detach()

        if target is None:
            target = output_dict.get('target', None)

        del output_dict

        N = pos.shape[0]
        if target is not None and target.dim() == 2:
            target = target.reshape(-1)
            if target.shape[0] != N:
                target = None

        if N == 0 or features.shape[0] == 0:
            return torch.tensor(0.0, device=pos.device, requires_grad=True)

        device = pos.device
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        else:
            batch = batch.to(device)

        if batch.shape[0] != N:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        MAX_POINTS = 20000
        if N > MAX_POINTS:
            sample_idx = torch.randperm(N, device=device)[:MAX_POINTS]
            features = features[sample_idx]
            pos = pos[sample_idx]
            rgb = rgb[sample_idx]
            batch = batch[sample_idx]
            if target is not None:
                target = target[sample_idx]
            N = MAX_POINTS

        try:
            if self.debug and self.call_count % self.debug_interval == 0:
                print(f"\n[TSI-Loss Debug] Input shapes:")
                print(
                    f"  pos: {pos.shape}, features: {features.shape}, rgb: {rgb.shape}")
                print(
                    f"  batch: {batch.shape}, target: {target.shape if target is not None else 'None'}")
                print(f"  N (after sample): {N}")

            if features.shape[0] != N or rgb.shape[0] != N or batch.shape[0] != N:
                print(
                    f"[TSI-Loss] Shape mismatch: N={N}, features={features.shape[0]}, rgb={rgb.shape[0]}, batch={batch.shape[0]}")
                return torch.tensor(0.0, device=device, requires_grad=True)

            edge_index = simple_knn_graph(pos, k=self.k, batch=batch)

            if edge_index.shape[1] == 0:
                print(f"[TSI-Loss] Empty edge index, skipping")
                return torch.tensor(0.0, device=device, requires_grad=True)

            row, col = edge_index[0], edge_index[1]

            row = torch.clamp(row, 0, N - 1)
            col = torch.clamp(col, 0, N - 1)

            if target is not None and target.shape[0] == N:
                semantic_gate = (target[row] == target[col]).float()
            else:
                semantic_gate = torch.ones(
                    row.shape[0], dtype=torch.float, device=device)

            try:
                normals = estimate_point_normals(pos, k=self.k, batch=batch)
                if normals.shape[0] == N:
                    normal_diff_sq = (
                        normals[row] - normals[col]).pow(2).sum(dim=-1)
                    geo_weight = torch.exp(-normal_diff_sq /
                                           (2 * self.sigma_g ** 2))
                else:
                    normal_diff_sq = torch.zeros(row.shape[0], device=device)
                    geo_weight = torch.ones(
                        row.shape[0], dtype=torch.float, device=device)
            except Exception as e:
                if self.debug:
                    print(f"[TSI-Loss] Normal estimation failed: {e}")
                normal_diff_sq = torch.zeros(row.shape[0], device=device)
                geo_weight = torch.ones(
                    row.shape[0], dtype=torch.float, device=device)

            rgb_diff_sq = (rgb[row] - rgb[col]).pow(2).sum(dim=-1)
            inv_weight = torch.tanh(self.lambda_tex * rgb_diff_sq)

            feat_diff_sq = (features[row] - features[col]).pow(2).sum(dim=-1)

            edge_loss = semantic_gate * geo_weight * inv_weight * feat_diff_sq

            final_loss = edge_loss.mean() * self.weight

            self.call_count += 1
            if self.debug and self.call_count % self.debug_interval == 0:
                print(
                    f"[TSI-Loss] Step {self.call_count}, Loss: {final_loss.item():.6f}")

            return final_loss

        except Exception as e:
            print(f"[TSI-Loss Error] {e}")
            try:
                return features.sum() * 0.0
            except:
                try:
                    return torch.tensor(0.0, device=device, requires_grad=True)
                except:
                    return torch.tensor(0.0, requires_grad=True)

# =========================================================================
# =========================================================================


def apply_scannet_c_corruption(data, corruption_type, severity=1, feature_keys='x'):
    """
    Apply Robustness Corruptions (ScanNet-C style)
    Args:
        data: Input data dictionary (contains 'x' [B, C, N] or [C, N])
        corruption_type: str ('clean', 'jitter', 'noise', 'dropout', 'geometry_only')
        severity: int (1-5), controls intensity
    """
    if corruption_type == 'clean' or corruption_type is None:
        return data

    if 'x' not in data:
        return data

    feat = data['x'].clone()

    is_batch = feat.dim() == 3
    if not is_batch:
        feat = feat.unsqueeze(0)  # (1, C, N)

    B, C, N = feat.shape

    key_list = feature_keys.split(',')
    rgb_start = 0
    found_rgb = False
    for key in key_list:
        if key == 'pos':
            rgb_start += 3
        elif key == 'x':
            found_rgb = True
            break
        elif key == 'heights':
            rgb_start += 1
        elif key == 'normals':
            rgb_start += 3

    rgb_end = rgb_start + 3
    if not found_rgb or rgb_end > C:
        return data

    rgb = feat[:, rgb_start:rgb_end, :]


    if corruption_type == 'noise':
        scale = 0.05 * severity  # 0.035 0.05
        noise = torch.randn_like(rgb) * scale
        rgb = rgb + noise

    elif corruption_type == 'jitter':
        scale = 0.1 * severity  # 0.05 0.1 0.15
        noise = (torch.rand(B, 3, 1, device=rgb.device) - 0.5) * 2 * scale
        rgb = rgb + noise

        # noise = (torch.rand(B, 3, N, device=rgb.device) - 0.5) * 2 * scale
        # rgb = rgb + noise

    elif corruption_type == 'dropout':
        prob = 0.1 * severity  # 0.07 0.1
        mask = torch.rand(B, 1, N, device=rgb.device) < prob
        rgb[mask.expand(-1, 3, -1)] = 0.0

        # random_colors = torch.rand(B, 3, N, device=rgb.device)
        # rgb = torch.where(mask.expand(-1, 3, -1), random_colors, rgb)

    # elif corruption_type == 'geometry_only':

    # elif corruption_type == 'high_freq':
    #     noise = torch.randn_like(rgb) * 0.1 * severity
    #     rgb = rgb + noise

    rgb = torch.clamp(rgb, 0, 1)

    feat[:, rgb_start:rgb_end, :] = rgb

    if not is_batch:
        feat = feat.squeeze(0)

    data['x'] = feat
    return data

# =========================================================================
# =========================================================================


class SemanticConflictGenerator:
    """




    """

    S3DIS_CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                     'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']

    S3DIS_TARGET_CLASSES = {
        7: 'chair',
        8: 'table',
        10: 'sofa',
        5: 'window',
        6: 'door',
    }

    S3DIS_SOURCE_CLASSES = {
        0: 'ceiling',
        1: 'floor',
        2: 'wall',
    }

    SCANNET_CLASSES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                       'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                       'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                       'bathtub', 'otherfurniture']

    SCANNET_TARGET_CLASSES = {
        4: 'chair',
        5: 'sofa',
        6: 'table',
        16: 'toilet',
        18: 'bathtub',
    }

    SCANNET_SOURCE_CLASSES = {
        0: 'wall',
        1: 'floor',
        13: 'curtain',
    }

    SCANNET_TO_S3DIS_MAP = {
        0: 2,   # wall → wall
        1: 1,   # floor → floor
        3: 10,  # bed → sofa (large horizontal furniture)
        4: 7,   # chair → chair
        5: 10,  # sofa → sofa
        6: 8,   # table → table
        7: 6,   # door → door
        8: 5,   # window → window
        11: 8,  # counter → table (flat horizontal surface)
        12: 8,  # desk → table (desk is a subtype of table)
    }

    S3DIS_TO_SCANNET_MAP = {
        2: 0,   # wall → wall
        1: 1,   # floor → floor
        7: 4,   # chair → chair
        10: 5,  # sofa → sofa
        8: 6,   # table → table
        6: 7,   # door → door
        5: 8,   # window → window
    }

    CROSS_DOMAIN_SCANNET_TO_S3DIS_TARGET = {
        7: 'chair',
        8: 'table',
        10: 'sofa',
        6: 'door',
    }

    CROSS_DOMAIN_SCANNET_TO_S3DIS_SOURCE = {
        2: 'wall',
        1: 'floor',
        0: 'ceiling',
    }

    CROSS_DOMAIN_S3DIS_TO_SCANNET_TARGET = {
        4: 'chair',
        5: 'sofa',
        6: 'table',
        7: 'door',
    }

    CROSS_DOMAIN_S3DIS_TO_SCANNET_SOURCE = {
        0: 'wall',
        1: 'floor',
    }

    TARGET_CLASSES = S3DIS_TARGET_CLASSES
    SOURCE_CLASSES = S3DIS_SOURCE_CLASSES

    def __init__(self,
                 dataset='s3dis',
                 target_classes=None,
                 source_classes=None,
                 attack_ratio=1.0,
                 preserve_boundary=True,
                 seed=42):
        """
        Args:
        """
        self.dataset = dataset.lower()

        if self.dataset == 'scannet':
            default_target = self.SCANNET_TARGET_CLASSES
            default_source = self.SCANNET_SOURCE_CLASSES
            self.class_names = self.SCANNET_CLASSES
            self.num_classes = 20
        else:
            default_target = self.S3DIS_TARGET_CLASSES
            default_source = self.S3DIS_SOURCE_CLASSES
            self.class_names = self.S3DIS_CLASSES
            self.num_classes = 13

        self.target_classes = target_classes or default_target
        self.source_classes = source_classes or default_source
        self.attack_ratio = attack_ratio
        self.preserve_boundary = preserve_boundary
        self.rng = np.random.RandomState(seed)

        print(f"[SemanticConflict] Initialized for {self.dataset.upper()}:")
        print(
            f"  Target classes (Geometry): {list(self.target_classes.values())}")
        print(
            f"  Source classes (Texture): {list(self.source_classes.values())}")
        print(f"  Attack ratio: {attack_ratio:.0%}")

    def generate_conflict_sample(self, data, return_attack_info=False):
        """



        Args:
            data: dict with keys 'pos', 'x' (features with RGB), 'y' (labels)

        Returns:
        """
        if isinstance(data, dict):
            pos = data.get('pos')
            x = data.get('x')  # Features, RGB is typically first 3 channels
            y = data.get('y')  # Labels
        else:
            pos = getattr(data, 'pos', None)
            x = getattr(data, 'x', None)
            y = getattr(data, 'y', None)

        if pos is None or x is None or y is None:
            if return_attack_info:
                return data, {'success': False, 'reason': 'missing_data'}
            return data

        is_tensor = torch.is_tensor(pos)
        device = pos.device if is_tensor else None

        pos_np = pos.cpu().numpy() if is_tensor else pos
        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y

        original_x_shape = x_np.shape
        original_x_ndim = x_np.ndim
        x_transposed = False

        if pos_np.ndim == 3:  # (B, N, 3)
            pos_np = pos_np.reshape(-1, pos_np.shape[-1])
        if x_np.ndim == 3:
            if x_np.shape[1] < x_np.shape[2]:
                x_np = x_np.transpose(0, 2, 1).reshape(-1, x_np.shape[1])
                x_transposed = True
            else:
                x_np = x_np.reshape(-1, x_np.shape[-1])
        if y_np.ndim == 2:  # (B, N)
            y_np = y_np.reshape(-1)

        N = len(y_np)
        attack_info = {
            'success': False,
            'target_class': None,
            'source_class': None,
            'num_attacked_points': 0,
            'attack_mask': np.zeros(N, dtype=bool)
        }

        target_ids = list(self.target_classes.keys())
        source_ids = list(self.source_classes.keys())

        available_targets = []
        for tid in target_ids:
            mask = (y_np == tid)
            if mask.sum() > 100:
                available_targets.append((tid, mask))

        available_sources = []
        for sid in source_ids:
            mask = (y_np == sid)
            if mask.sum() > 50:
                available_sources.append((sid, mask))

        if len(available_targets) == 0 or len(available_sources) == 0:
            if return_attack_info:
                attack_info['reason'] = 'no_valid_classes'
                return data, attack_info
            return data

        target_id, target_mask = available_targets[self.rng.randint(
            len(available_targets))]
        source_id, source_mask = available_sources[self.rng.randint(
            len(available_sources))]

        target_indices = np.where(target_mask)[0]
        source_indices = np.where(source_mask)[0]

        if self.preserve_boundary and len(target_indices) > 50:
            target_pos = pos_np[target_indices]
            centroid = target_pos.mean(axis=0)
            dists = np.linalg.norm(target_pos - centroid, axis=1)

            boundary_threshold = np.percentile(dists, 90)
            boundary_mask = dists >= boundary_threshold
            interior_indices = target_indices[~boundary_mask]
        else:
            interior_indices = target_indices

        if self.attack_ratio < 1.0:
            num_to_attack = int(len(interior_indices) * self.attack_ratio)
            self.rng.shuffle(interior_indices)
            interior_indices = interior_indices[:num_to_attack]

        if len(interior_indices) == 0:
            if return_attack_info:
                attack_info['reason'] = 'no_interior_points'
                return data, attack_info
            return data

        source_sample_idx = self.rng.choice(
            source_indices, size=len(interior_indices), replace=True)
        source_texture = x_np[source_sample_idx, :3]

        x_np_attacked = x_np.copy()
        x_np_attacked[interior_indices, :3] = source_texture

        if original_x_ndim == 3:
            if x_transposed:
                B = original_x_shape[0]
                C = original_x_shape[1]
                N_per_batch = original_x_shape[2]
                x_np_attacked = x_np_attacked.reshape(
                    B, N_per_batch, C).transpose(0, 2, 1)
            else:
                x_np_attacked = x_np_attacked.reshape(original_x_shape)

        if isinstance(data, dict):
            data = data.copy()
            if is_tensor:
                data['x'] = torch.from_numpy(x_np_attacked).to(device)
            else:
                data['x'] = x_np_attacked
        else:
            if is_tensor:
                data.x = torch.from_numpy(x_np_attacked).to(device)
            else:
                data.x = x_np_attacked

        attack_info['success'] = True
        attack_info['target_class'] = (
            target_id, self.target_classes.get(target_id, str(target_id)))
        attack_info['source_class'] = (
            source_id, self.source_classes.get(source_id, str(source_id)))
        attack_info['num_attacked_points'] = len(interior_indices)
        attack_info['attack_mask'] = np.isin(np.arange(N), interior_indices)
        attack_info['original_texture'] = x_np[interior_indices, :3]

        if return_attack_info:
            return data, attack_info
        return data

    def generate_all_conflict_samples(self, data):
        """


        Args:
            data: dict with keys 'pos', 'x', 'y'

        Returns:
        """
        if isinstance(data, dict):
            pos = data.get('pos')
            x = data.get('x')
            y = data.get('y')
        else:
            pos = getattr(data, 'pos', None)
            x = getattr(data, 'x', None)
            y = getattr(data, 'y', None)

        if pos is None or x is None or y is None:
            return data, []

        is_tensor = torch.is_tensor(pos)
        device = pos.device if is_tensor else None

        pos_np = pos.cpu().numpy() if is_tensor else pos
        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y

        original_x_shape = x_np.shape
        original_x_ndim = x_np.ndim
        x_transposed = False

        if pos_np.ndim == 3:
            pos_np = pos_np.reshape(-1, pos_np.shape[-1])
        if x_np.ndim == 3:
            if x_np.shape[1] < x_np.shape[2]:
                x_np = x_np.transpose(0, 2, 1).reshape(-1, x_np.shape[1])
                x_transposed = True
            else:
                x_np = x_np.reshape(-1, x_np.shape[-1])
        if y_np.ndim == 2:
            y_np = y_np.reshape(-1)

        N = len(y_np)

        available_targets = []
        for tid in self.target_classes:
            mask = (y_np == tid)
            if mask.sum() > 100:
                available_targets.append((tid, mask))

        available_sources = []
        for sid in self.source_classes:
            mask = (y_np == sid)
            if mask.sum() > 50:
                available_sources.append((sid, mask))

        if len(available_targets) == 0 or len(available_sources) == 0:
            return data, []

        x_np_attacked = x_np.copy()
        attack_infos = []

        for target_id, target_mask in available_targets:
            target_indices = np.where(target_mask)[0]

            if self.preserve_boundary and len(target_indices) > 50:
                target_pos = pos_np[target_indices]
                centroid = target_pos.mean(axis=0)
                dists = np.linalg.norm(target_pos - centroid, axis=1)
                boundary_threshold = np.percentile(dists, 90)
                boundary_mask = dists >= boundary_threshold
                interior_indices = target_indices[~boundary_mask]
            else:
                interior_indices = target_indices

            if self.attack_ratio < 1.0:
                num_to_attack = int(len(interior_indices) * self.attack_ratio)
                self.rng.shuffle(interior_indices)
                interior_indices = interior_indices[:num_to_attack]

            if len(interior_indices) == 0:
                continue

            valid_sources = [(sid, smask) for sid, smask in available_sources
                             if sid != target_id]
            if len(valid_sources) == 0:
                continue
            source_id, source_mask = valid_sources[
                self.rng.randint(len(valid_sources))]
            source_indices = np.where(source_mask)[0]

            original_texture = x_np[interior_indices, :3].copy()

            source_sample_idx = self.rng.choice(
                source_indices, size=len(interior_indices), replace=True)
            x_np_attacked[interior_indices, :3] = x_np[source_sample_idx, :3]

            attack_infos.append({
                'success': True,
                'target_class': (target_id,
                                 self.target_classes.get(target_id, str(target_id))),
                'source_class': (source_id,
                                 self.source_classes.get(source_id, str(source_id))),
                'num_attacked_points': len(interior_indices),
                'attack_mask': np.isin(np.arange(N), interior_indices),
                'original_texture': original_texture,
            })

        if len(attack_infos) == 0:
            return data, []

        if original_x_ndim == 3:
            if x_transposed:
                B = original_x_shape[0]
                C = original_x_shape[1]
                N_per_batch = original_x_shape[2]
                x_np_attacked = x_np_attacked.reshape(
                    B, N_per_batch, C).transpose(0, 2, 1)
            else:
                x_np_attacked = x_np_attacked.reshape(original_x_shape)

        if isinstance(data, dict):
            data = data.copy()
            if is_tensor:
                data['x'] = torch.from_numpy(x_np_attacked).to(device)
            else:
                data['x'] = x_np_attacked
        else:
            if is_tensor:
                data.x = torch.from_numpy(x_np_attacked).to(device)
            else:
                data.x = x_np_attacked

        return data, attack_infos

    def create_conflict_dataset(self, base_dataloader, max_samples=None):
        """

        Args:

        Returns:
            List of (attacked_data, attack_info, original_data)
        """
        conflict_samples = []

        for i, data in enumerate(base_dataloader):
            if max_samples and i >= max_samples:
                break

            original_data = {k: v.clone() if torch.is_tensor(v) else v
                             for k, v in data.items()}

            attacked_data, attack_info = self.generate_conflict_sample(
                data, return_attack_info=True)

            if attack_info['success']:
                conflict_samples.append(
                    (attacked_data, attack_info, original_data))

        print(
            f"[SemanticConflict] Created {len(conflict_samples)} conflict samples")
        return conflict_samples


class SemanticConflictEvaluator:
    """

       TR = P(pred = y_geo | T = T_conflict)

       TDR = P(pred = y_tex | T = T_conflict)

       GF = mIoU(conflict) / mIoU(clean) for target classes

    """

    def __init__(self, dataset='s3dis', num_classes=None, class_names=None,
                 ignore_index=-1, prediction_remap=None):
        """
        Args:
        """
        self.dataset = dataset.lower()

        if self.dataset == 'scannet':
            self.num_classes = num_classes or 20
            self.class_names = class_names or SemanticConflictGenerator.SCANNET_CLASSES
        else:  # s3dis
            self.num_classes = num_classes or 13
            self.class_names = class_names or SemanticConflictGenerator.S3DIS_CLASSES

        self.ignore_index = ignore_index
        self.prediction_remap = prediction_remap

        if prediction_remap:
            print(f"[Evaluator] Cross-domain prediction remapping enabled:")
            for src_id, tgt_id in sorted(prediction_remap.items()):
                print(f"  Model pred {src_id} → Data label {tgt_id}")

        self.reset()

    def reset(self):
        """"""
        self.total_attacked_points = 0
        self.correct_geo_predictions = 0
        self.deceptive_tex_predictions = 0
        self.other_predictions = 0

        self.per_class_stats = {
            'attacked': np.zeros(self.num_classes),
            'correct': np.zeros(self.num_classes),
            'deceived': np.zeros(self.num_classes),
        }

        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions, labels, attack_info):
        """

        Args:
            attack_info: dict from SemanticConflictGenerator
        """
        if not attack_info['success']:
            return

        pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        label_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        if self.prediction_remap is not None:
            remapped = np.full_like(pred_np, -1)
            for src_id, tgt_id in self.prediction_remap.items():
                remapped[pred_np == src_id] = tgt_id
            pred_np = remapped

        attack_mask = attack_info['attack_mask']
        target_class_id = attack_info['target_class'][0]
        source_class_id = attack_info['source_class'][0]

        attacked_pred = pred_np[attack_mask]
        attacked_label = label_np[attack_mask]

        n_attacked = len(attacked_pred)
        self.total_attacked_points += n_attacked

        correct = (attacked_pred == target_class_id).sum()
        deceived = (attacked_pred == source_class_id).sum()
        other = n_attacked - correct - deceived

        self.correct_geo_predictions += correct
        self.deceptive_tex_predictions += deceived
        self.other_predictions += other

        self.per_class_stats['attacked'][target_class_id] += n_attacked
        self.per_class_stats['correct'][target_class_id] += correct
        self.per_class_stats['deceived'][target_class_id] += deceived

        valid_mask = (attacked_label >= 0) & (
            attacked_label < self.num_classes)
        valid_pred = attacked_pred[valid_mask]
        valid_label = attacked_label[valid_mask]
        for p, l in zip(valid_pred, valid_label):
            if 0 <= p < self.num_classes:
                self.confusion_matrix[l, p] += 1

    def compute_metrics(self):
        """

        Returns:
            dict: {
                'texture_robustness': float,
                'texture_deception_rate': float,
                'miou_attacked': float,
                'per_class_robustness': dict,
            }
        """
        if self.total_attacked_points == 0:
            return {'error': 'no_attacked_points'}

        texture_robustness = self.correct_geo_predictions / self.total_attacked_points
        texture_deception_rate = self.deceptive_tex_predictions / self.total_attacked_points
        other_rate = self.other_predictions / self.total_attacked_points

        per_class_robustness = {}
        for cls_id, cls_name in enumerate(self.class_names):
            attacked = self.per_class_stats['attacked'][cls_id]
            if attacked > 0:
                per_class_robustness[cls_name] = {
                    'attacked_points': int(attacked),
                    'robustness': self.per_class_stats['correct'][cls_id] / attacked,
                    'deception_rate': self.per_class_stats['deceived'][cls_id] / attacked,
                }

        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(
            axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(
            intersection, dtype=float), where=union > 0)

        attacked_classes = self.per_class_stats['attacked'] > 0
        if attacked_classes.sum() > 0:
            miou_attacked = iou[attacked_classes].mean()
        else:
            miou_attacked = 0.0

        return {
            'texture_robustness': texture_robustness,
            'texture_deception_rate': texture_deception_rate,
            'other_prediction_rate': other_rate,
            'miou_attacked_classes': miou_attacked,
            'total_attacked_points': self.total_attacked_points,
            'per_class_robustness': per_class_robustness,
            'class_ious': {self.class_names[i]: iou[i] for i in range(len(iou)) if iou[i] > 0},
        }

    def print_report(self, model_name="Model"):
        """"""
        metrics = self.compute_metrics()

        if 'error' in metrics:
            print(f"[{model_name}] Error: {metrics['error']}")
            return metrics

        print("\n" + "=" * 70)
        print(f"🔬 Semantic Conflict Test Report: {model_name}")
        print("=" * 70)

        print(
            f"\n📊 Global Metrics (on {metrics['total_attacked_points']:,} attacked points):")
        print(
            f"   ✅ Texture Robustness (TR):     {metrics['texture_robustness']:.2%}")
        print(
            f"   ❌ Texture Deception Rate (TDR): {metrics['texture_deception_rate']:.2%}")
        print(
            f"   ❓ Other Predictions:            {metrics['other_prediction_rate']:.2%}")
        print(
            f"   📈 mIoU on Attacked Classes:     {metrics['miou_attacked_classes']:.2%}")

        print(f"\n📋 Per-Class Breakdown:")
        print(
            f"   {'Class':<20} {'Attacked':<12} {'Robustness':<12} {'Deception':<12}")
        print(f"   {'-'*56}")
        for cls_name, stats in metrics['per_class_robustness'].items():
            print(f"   {cls_name:<20} {stats['attacked_points']:<12} "
                  f"{stats['robustness']:.2%}       {stats['deception_rate']:.2%}")

        print("=" * 70 + "\n")

        return metrics


def run_semantic_conflict_test(model, dataloader, device='cuda', model_name="Model", dataset='s3dis'):
    """


    Args:

    Returns:

    Usage:
        metrics_baseline = run_semantic_conflict_test(
            model_baseline, val_loader, device='cuda', model_name='Baseline', dataset='s3dis'
        )

        metrics_ours = run_semantic_conflict_test(
            model_ours, val_loader, device='cuda', model_name='Ours (STS)', dataset='s3dis'
        )

        print(f"TR Improvement: {metrics_ours['texture_robustness'] - metrics_baseline['texture_robustness']:.2%}")
    """
    generator = SemanticConflictGenerator(dataset=dataset)
    evaluator = SemanticConflictEvaluator(dataset=dataset)

    model.eval()
    model.to(device)

    print(f"\n[Semantic Conflict Test] Running on {model_name}...")

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data_attacked, attack_info = generator.generate_conflict_sample(
                data, return_attack_info=True)

            if not attack_info['success']:
                continue

            for key in data_attacked:
                if torch.is_tensor(data_attacked[key]):
                    data_attacked[key] = data_attacked[key].to(device)

            try:
                output = model(data_attacked)
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('seg_logits'))
                else:
                    logits = output

                if logits.dim() == 3:
                    predictions = logits.argmax(dim=1).squeeze(0)
                else:
                    predictions = logits.argmax(dim=-1)

            except Exception as e:
                print(f"  [Warning] Forward failed for batch {batch_idx}: {e}")
                continue

            labels = data_attacked.get('y', data_attacked.get('label'))
            if labels is not None:
                if labels.dim() > 1:
                    labels = labels.squeeze()
                evaluator.update(predictions.cpu(), labels.cpu(), attack_info)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    metrics = evaluator.print_report(model_name)

    return metrics


def compare_models_on_conflict_test(model_baseline, model_ours, dataloader, device='cuda', dataset='s3dis'):
    """


    Args:

    Returns:
    """
    print("\n" + "🔬" * 35)
    print(
        f"SEMANTIC CONFLICT TEST ({dataset.upper()}): BASELINE vs OURS (STS)")
    print("🔬" * 35)

    metrics_baseline = run_semantic_conflict_test(
        model_baseline, dataloader, device, model_name="Baseline", dataset=dataset
    )

    metrics_ours = run_semantic_conflict_test(
        model_ours, dataloader, device, model_name="Ours (STS)", dataset=dataset
    )

    if 'error' not in metrics_baseline and 'error' not in metrics_ours:
        tr_improvement = metrics_ours['texture_robustness'] - \
            metrics_baseline['texture_robustness']
        tdr_reduction = metrics_baseline['texture_deception_rate'] - \
            metrics_ours['texture_deception_rate']
        miou_improvement = metrics_ours['miou_attacked_classes'] - \
            metrics_baseline['miou_attacked_classes']

        print("\n" + "=" * 70)
        print("📊 COMPARISON SUMMARY")
        print("=" * 70)
        print(
            f"   Texture Robustness Improvement:    +{tr_improvement:.2%} ⬆️")
        print(f"   Texture Deception Rate Reduction:  -{tdr_reduction:.2%} ⬇️")
        print(
            f"   mIoU Improvement on Attacked:      +{miou_improvement:.2%} ⬆️")
        print("=" * 70)

        if tr_improvement > 0.05 and tdr_reduction > 0.05:
            print("✅ SUCCESS: STS significantly improves geometry reliance!")
        elif tr_improvement > 0:
            print("⚠️ PARTIAL: STS shows improvement but may need stronger training.")
        else:
            print("❌ FAILED: STS does not improve robustness. Check implementation.")

    return {
        'baseline': metrics_baseline,
        'ours': metrics_ours,
        'improvement': {
            'texture_robustness': tr_improvement if 'error' not in metrics_ours else None,
            'texture_deception_reduction': tdr_reduction if 'error' not in metrics_ours else None,
            'miou': miou_improvement if 'error' not in metrics_ours else None,
        }
    }


# =========================================================================
# =========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 Semantic Conflict Test - Standalone Demo")
    print("=" * 70)

    print("\n[Step 1] Creating synthetic test data...")

    num_points = 5000

    pos = torch.rand(num_points, 3) * 10

    labels = torch.zeros(num_points, dtype=torch.long)
    labels[pos[:, 2] < 0.5] = 1
    labels[pos[:, 2] > 9.0] = 0
    labels[(pos[:, 0] > 3) & (pos[:, 0] < 5) & (pos[:, 1] > 3)
           & (pos[:, 1] < 5)] = 4
    labels[(pos[:, 0] > 6) & (pos[:, 0] < 8) & (pos[:, 1] > 6)
           & (pos[:, 1] < 8)] = 6

    colors = torch.zeros(num_points, 3)
    colors[labels == 0] = torch.tensor([0.8, 0.8, 0.8])
    colors[labels == 1] = torch.tensor([0.6, 0.4, 0.2])
    colors[labels == 4] = torch.tensor([0.8, 0.2, 0.2])
    colors[labels == 6] = torch.tensor([0.2, 0.6, 0.2])

    colors += torch.randn_like(colors) * 0.05
    colors = colors.clamp(0, 1)

    x = torch.cat([colors, torch.rand(num_points, 3)], dim=1)

    data = {
        'pos': pos,
        'x': x,
        'y': labels
    }

    print(f"  Created scene with {num_points} points")
    print(f"  Class distribution: wall={int((labels==0).sum())}, floor={int((labels==1).sum())}, "
          f"chair={int((labels==4).sum())}, table={int((labels==6).sum())}")

    print("\n[Step 2] Generating conflict sample...")

    generator = SemanticConflictGenerator(seed=42)
    data_attacked, attack_info = generator.generate_conflict_sample(
        data, return_attack_info=True)

    if attack_info['success']:
        print(f"  ✅ Attack successful!")
        print(
            f"  Target: {attack_info['target_class'][1]} (id={attack_info['target_class'][0]})")
        print(
            f"  Source: {attack_info['source_class'][1]} (id={attack_info['source_class'][0]})")
        print(f"  Attacked points: {attack_info['num_attacked_points']}")

        original_colors = data['x'][:, :3].numpy()
        attacked_colors = data_attacked['x'][:, :3].numpy()
        diff = np.abs(original_colors - attacked_colors).sum(axis=1)
        changed_points = (diff > 0.01).sum()
        print(f"  Color changed points: {changed_points}")
    else:
        print(f"  ⚠️ Attack failed: {attack_info.get('reason', 'unknown')}")

    print("\n[Step 3] Simulating model predictions...")

    evaluator = SemanticConflictEvaluator()

    print("\n  [Baseline Model Simulation]")
    baseline_pred = labels.clone()
    attack_mask = attack_info['attack_mask']
    source_id = attack_info['source_class'][0]

    deceive_mask = torch.rand(attack_mask.sum()) < 0.7
    baseline_pred[torch.from_numpy(attack_mask)] = torch.where(
        deceive_mask,
        torch.full((deceive_mask.sum(),), source_id, dtype=torch.long),
        baseline_pred[torch.from_numpy(attack_mask)]
    )

    evaluator.update(baseline_pred, labels, attack_info)
    baseline_metrics = evaluator.print_report("Baseline (Simulated)")

    print("\n  [Ours (STS) Model Simulation]")
    evaluator.reset()

    ours_pred = labels.clone()
    deceive_mask = torch.rand(attack_mask.sum()) < 0.15
    ours_pred[torch.from_numpy(attack_mask)] = torch.where(
        deceive_mask,
        torch.full((deceive_mask.sum(),), source_id, dtype=torch.long),
        ours_pred[torch.from_numpy(attack_mask)]
    )

    evaluator.update(ours_pred, labels, attack_info)
    ours_metrics = evaluator.print_report("Ours/STS (Simulated)")

    print("\n" + "=" * 70)
    print("📊 SIMULATION COMPARISON")
    print("=" * 70)
    tr_improvement = ours_metrics['texture_robustness'] - \
        baseline_metrics['texture_robustness']
    tdr_reduction = baseline_metrics['texture_deception_rate'] - \
        ours_metrics['texture_deception_rate']
    print(f"   Texture Robustness Improvement:    +{tr_improvement:.2%} ⬆️")
    print(f"   Texture Deception Rate Reduction:  -{tdr_reduction:.2%} ⬇️")
    print("=" * 70)

    print("\n✅ Semantic Conflict Test Demo Complete!")
    print("\nTo run on real models, use:")
    print("  from openpoints.loss.custom_innovations import run_semantic_conflict_test")
    print("  metrics = run_semantic_conflict_test(model, val_loader, device='cuda')")


# =========================================================================
# =========================================================================

def visualize_semantic_conflict_test(
    data_original,
    data_conflict,
    pred_baseline,
    pred_ours,
    attack_info,
    output_path,
    dataset='s3dis',
    model_name_baseline="Baseline",
    model_name_ours="Ours (STS)"
):
    """


    Args:

    Returns:
    """
    import sys
    import os
    import numpy as np

    script_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    visualize_script = os.path.join(
        script_dir, 'scripts', 'visualize_qualitative_comparison.py')

    if not os.path.exists(visualize_script):
        print(f"[Warning] Visualization script not found: {visualize_script}")
        print("Skipping visualization.")
        return None

    sys.path.insert(0, os.path.dirname(visualize_script))
    try:
        from visualize_qualitative_comparison import (
            visualize_qualitative_comparison,
            S3DIS_COLORMAP,
            SCANNET_COLORMAP
        )
    except ImportError as e:
        print(f"[Warning] Failed to import visualization functions: {e}")
        print("Skipping visualization.")
        return None

    pos = data_conflict.get('pos')
    if pos is None:
        pos = data_original.get('pos')
    if torch.is_tensor(pos):
        pos = pos.cpu().numpy()
    if pos.ndim == 3:  # (B, N, 3)
        pos = pos.reshape(-1, 3)
    elif pos.ndim == 2 and pos.shape[0] == 3:  # (3, N)
        pos = pos.T
    elif pos.ndim == 2 and pos.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected pos shape: {pos.shape}")

    N = pos.shape[0]

    x_original = data_original.get('x')
    if torch.is_tensor(x_original):
        x_original = x_original.cpu().numpy()

    if x_original.ndim == 3:
        if x_original.shape[1] < x_original.shape[2]:  # (B, C, N)
            x_original = x_original.transpose(
                0, 2, 1).reshape(-1, x_original.shape[1])
        else:  # (B, N, C)
            x_original = x_original.reshape(-1, x_original.shape[-1])
    elif x_original.ndim == 2 and x_original.shape[0] < x_original.shape[1] and x_original.shape[0] <= 10:
        # (C, N) -> (N, C)
        x_original = x_original.T

    if len(x_original) != N:
        print(
            f"[Warning] Original x length ({len(x_original)}) != pos length ({N}), truncating/padding...")
        if len(x_original) > N:
            x_original = x_original[:N]
        else:
            padding = np.tile(x_original[-1:], (N - len(x_original), 1))
            x_original = np.vstack([x_original, padding])

    colors_clean = x_original[:, :3]
    colors_clean = np.clip(colors_clean, 0, 1)

    x_conflict = data_conflict.get('x')
    if torch.is_tensor(x_conflict):
        x_conflict = x_conflict.cpu().numpy()

    if x_conflict.ndim == 3:
        if x_conflict.shape[1] < x_conflict.shape[2]:  # (B, C, N)
            x_conflict = x_conflict.transpose(
                0, 2, 1).reshape(-1, x_conflict.shape[1])
        else:  # (B, N, C)
            x_conflict = x_conflict.reshape(-1, x_conflict.shape[-1])
    elif x_conflict.ndim == 2 and x_conflict.shape[0] < x_conflict.shape[1] and x_conflict.shape[0] <= 10:
        x_conflict = x_conflict.T

    if len(x_conflict) != N:
        print(
            f"[Warning] Conflict x length ({len(x_conflict)}) != pos length ({N}), truncating/padding...")
        if len(x_conflict) > N:
            x_conflict = x_conflict[:N]
        else:
            padding = np.tile(x_conflict[-1:], (N - len(x_conflict), 1))
            x_conflict = np.vstack([x_conflict, padding])

    colors_conflict = x_conflict[:, :3]
    colors_conflict = np.clip(colors_conflict, 0, 1)

    gt_labels = data_conflict.get('y')
    if gt_labels is None:
        gt_labels = data_original.get('y')
    if torch.is_tensor(gt_labels):
        gt_labels = gt_labels.cpu().numpy()
    if gt_labels.ndim > 1:
        gt_labels = gt_labels.reshape(-1)

    if len(gt_labels) != N:
        print(
            f"[Warning] Labels length ({len(gt_labels)}) != pos length ({N}), truncating/padding...")
        if len(gt_labels) > N:
            gt_labels = gt_labels[:N]
        else:
            padding = np.full(N - len(gt_labels),
                              gt_labels[-1] if len(gt_labels) > 0 else 0)
            gt_labels = np.concatenate([gt_labels, padding])

    if torch.is_tensor(pred_baseline):
        pred_baseline = pred_baseline.cpu().numpy()
    if pred_baseline.ndim > 1:
        pred_baseline = pred_baseline.reshape(-1)

    if len(pred_baseline) != N:
        print(
            f"[Warning] Baseline pred length ({len(pred_baseline)}) != pos length ({N}), truncating/padding...")
        if len(pred_baseline) > N:
            pred_baseline = pred_baseline[:N]
        else:
            padding = np.full(N - len(pred_baseline),
                              pred_baseline[-1] if len(pred_baseline) > 0 else 0)
            pred_baseline = np.concatenate([pred_baseline, padding])

    if torch.is_tensor(pred_ours):
        pred_ours = pred_ours.cpu().numpy()
    if pred_ours.ndim > 1:
        pred_ours = pred_ours.reshape(-1)

    if len(pred_ours) != N:
        print(
            f"[Warning] Ours pred length ({len(pred_ours)}) != pos length ({N}), truncating/padding...")
        if len(pred_ours) > N:
            pred_ours = pred_ours[:N]
        else:
            padding = np.full(N - len(pred_ours),
                              pred_ours[-1] if len(pred_ours) > 0 else 0)
            pred_ours = np.concatenate([pred_ours, padding])

    assert len(
        colors_clean) == N, f"Color clean length mismatch: {len(colors_clean)} vs {N}"
    assert len(
        colors_conflict) == N, f"Color conflict length mismatch: {len(colors_conflict)} vs {N}"
    assert len(
        gt_labels) == N, f"Label length mismatch: {len(gt_labels)} vs {N}"
    assert len(
        pred_baseline) == N, f"Baseline pred length mismatch: {len(pred_baseline)} vs {N}"
    assert len(
        pred_ours) == N, f"Ours pred length mismatch: {len(pred_ours)} vs {N}"

    if dataset.lower() == 's3dis':
        colormap = S3DIS_COLORMAP
    elif dataset.lower() == 'scannet':
        colormap = SCANNET_COLORMAP
    else:
        colormap = S3DIS_COLORMAP

    visualize_qualitative_comparison(
        points=pos,
        colors_input=colors_conflict,
        pred_baseline=pred_baseline,
        pred_ours=pred_ours,
        gt_labels=gt_labels,
        colormap=colormap,
        output_path=output_path,
        dataset_name=dataset.lower(),
        show_zoom_in=False,
        zoom_regions=None,
        camera_view='default',
        colors_clean=colors_clean
    )

    if attack_info.get('success', False):
        print(f"\n[Visualization] Semantic Conflict Test:")
        print(
            f"  Target Class: {attack_info['target_class'][1]} (id={attack_info['target_class'][0]})")
        print(
            f"  Source Class: {attack_info['source_class'][1]} (id={attack_info['source_class'][0]})")
        print(f"  Attacked Points: {attack_info['num_attacked_points']}")
        print(f"  Visualization saved to: {output_path}")

    return output_path


def run_semantic_conflict_test_with_visualization(
    model_baseline,
    model_ours,
    dataloader,
    device='cuda',
    model_name_baseline="Baseline",
    model_name_ours="Ours (STS)",
    dataset='s3dis',
    output_dir='visualizations/conflict_test',
    max_samples=10,
    cfg_baseline=None,
    cfg_ours=None
):
    """


    Args:

    Returns:
    """
    import os
    from openpoints.dataset import get_features_by_keys

    generator = SemanticConflictGenerator(
        dataset=dataset,
        attack_ratio=1.0,
        preserve_boundary=False,
        seed=42
    )
    evaluator_baseline = SemanticConflictEvaluator(dataset=dataset)
    evaluator_ours = SemanticConflictEvaluator(dataset=dataset)

    model_baseline.eval()
    model_ours.eval()
    model_baseline.to(device)
    model_ours.to(device)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Semantic Conflict Test with Visualization] Running...")
    print(f"  Output directory: {output_dir}")
    print(f"  Max visualization samples: {max_samples}")

    feature_keys_baseline = None
    feature_keys_ours = None
    if cfg_baseline is not None:
        feature_keys_baseline = getattr(cfg_baseline, 'feature_keys', None)
        if feature_keys_baseline is None:
            feature_keys_baseline = getattr(cfg_baseline, 'model', {}).get(
                'feature_keys', None) if hasattr(cfg_baseline, 'model') else None
    if cfg_ours is not None:
        feature_keys_ours = getattr(cfg_ours, 'feature_keys', None)
        if feature_keys_ours is None:
            feature_keys_ours = getattr(cfg_ours, 'model', {}).get(
                'feature_keys', None) if hasattr(cfg_ours, 'model') else None

    print(f"[Debug] feature_keys_baseline: {feature_keys_baseline}")
    print(f"[Debug] feature_keys_ours: {feature_keys_ours}")

    if feature_keys_baseline is None or feature_keys_ours is None:
        print("[Info] Inferring feature_keys from first batch...")
        for batch in dataloader:
            if 'x' in batch and batch['x'] is not None:
                if feature_keys_baseline is None:
                    feature_keys_baseline = 'x'
                if feature_keys_ours is None:
                    feature_keys_ours = 'x'
            break

    visualization_paths = []
    sample_count = 0

    #
    #
    #
    # ============================================================================

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)


            data_original = {k: v.clone() if torch.is_tensor(
                v) else v for k, v in data.items()}

            if batch_idx == 0 and sample_count == 0:
                print(f"\n[Debug] Data format before attack:")
                print(f"  data['x'].shape: {data.get('x', 'N/A')}")
                if torch.is_tensor(data.get('x')):
                    print(f"  data['x'].dtype: {data['x'].dtype}")
                    print(
                        f"  data['x'].min/max: {data['x'].min():.3f}/{data['x'].max():.3f}")

            data_conflict, attack_info = generator.generate_conflict_sample(
                data, return_attack_info=True)

            if not attack_info['success']:
                if batch_idx == 0:
                    print(
                        f"  [Warning] Attack failed for batch {batch_idx}: {attack_info.get('reason', 'unknown')}")
                continue

            if batch_idx == 0 and sample_count == 0:
                print(f"\n[Debug] Data format after attack:")
                print(
                    f"  data_conflict['x'].shape: {data_conflict.get('x', 'N/A')}")
                if torch.is_tensor(data_conflict.get('x')):
                    print(
                        f"  data_conflict['x'].dtype: {data_conflict['x'].dtype}")
                    print(
                        f"  data_conflict['x'].min/max: {data_conflict['x'].min():.3f}/{data_conflict['x'].max():.3f}")
                print(
                    f"  Attack info: target={attack_info.get('target_class')}, source={attack_info.get('source_class')}, points={attack_info.get('num_attacked_points')}")

            for key in data_conflict:
                if torch.is_tensor(data_conflict[key]):
                    data_conflict[key] = data_conflict[key].to(device)


            feature_keys_baseline_actual = feature_keys_baseline if feature_keys_baseline else 'x'
            try:
                data_conflict_baseline = {
                    k: v.clone() if torch.is_tensor(v) else v
                    for k, v in data_conflict.items()}
                data_conflict_baseline['x'] = get_features_by_keys(
                    data_conflict_baseline, feature_keys_baseline_actual)

                if batch_idx == 0 and sample_count == 0:
                    print(
                        f"  [Verified] Baseline model will use attacked features (shape: {data_conflict_baseline['x'].shape})")
            except Exception as e:
                print(
                    f"  [ERROR] Failed to build features for baseline: {e}")
                print(
                    f"  [ERROR] This will cause incorrect results! Skipping batch {batch_idx}...")
                continue

            feature_keys_ours_actual = feature_keys_ours if feature_keys_ours else 'x'
            try:
                data_conflict_ours = {
                    k: v.clone() if torch.is_tensor(v) else v
                    for k, v in data_conflict.items()}
                data_conflict_ours['x'] = get_features_by_keys(
                    data_conflict_ours, feature_keys_ours_actual)

                if batch_idx == 0 and sample_count == 0:
                    print(
                        f"  [Verified] Ours model will use attacked features (shape: {data_conflict_ours['x'].shape})")
            except Exception as e:
                print(
                    f"  [ERROR] Failed to build features for ours: {e}")
                print(
                    f"  [ERROR] This will cause incorrect results! Skipping batch {batch_idx}...")
                continue

            try:
                output_baseline = model_baseline(data_conflict_baseline)
                if isinstance(output_baseline, dict):
                    logits_baseline = output_baseline.get(
                        'logits', output_baseline.get('seg_logits'))
                else:
                    logits_baseline = output_baseline

                if logits_baseline.dim() == 3:  # (B, C, N)
                    pred_baseline = logits_baseline.argmax(dim=1)  # (B, N)
                    pred_baseline = pred_baseline.reshape(-1)
                else:
                    pred_baseline = logits_baseline.argmax(dim=-1)
                    if pred_baseline.dim() > 1:
                        pred_baseline = pred_baseline.reshape(-1)
            except Exception as e:
                print(
                    f"  [Warning] Baseline forward failed for batch {batch_idx}: {e}")
                continue

            try:
                output_ours = model_ours(data_conflict_ours)
                if isinstance(output_ours, dict):
                    logits_ours = output_ours.get(
                        'logits', output_ours.get('seg_logits'))
                else:
                    logits_ours = output_ours

                if logits_ours.dim() == 3:  # (B, C, N)
                    pred_ours = logits_ours.argmax(dim=1)  # (B, N)
                    pred_ours = pred_ours.reshape(-1)
                else:
                    pred_ours = logits_ours.argmax(dim=-1)
                    if pred_ours.dim() > 1:
                        pred_ours = pred_ours.reshape(-1)
            except Exception as e:
                print(
                    f"  [Warning] Ours forward failed for batch {batch_idx}: {e}")
                continue

            labels = data_conflict.get('y', data_conflict.get('label'))
            if labels is not None:
                if labels.dim() > 1:
                    labels = labels.reshape(-1)

                evaluator_baseline.update(pred_baseline.cpu(),
                                          labels.cpu(), attack_info)
                evaluator_ours.update(pred_ours.cpu(),
                                      labels.cpu(), attack_info)

            if sample_count < max_samples:
                output_path = os.path.join(
                    output_dir,
                    f"conflict_test_sample_{batch_idx:04d}.png"
                )

                try:
                    result = visualize_semantic_conflict_test(
                        data_original=data_original,
                        data_conflict=data_conflict,
                        pred_baseline=pred_baseline.cpu(),
                        pred_ours=pred_ours.cpu(),
                        attack_info=attack_info,
                        output_path=output_path,
                        dataset=dataset,
                        model_name_baseline=model_name_baseline,
                        model_name_ours=model_name_ours
                    )
                    if result is not None:
                        visualization_paths.append(output_path)
                        sample_count += 1
                        print(
                            f"  ✅ Visualized sample {sample_count}/{max_samples}: {output_path}")
                    else:
                        print(
                            f"  [Warning] Visualization returned None for batch {batch_idx} (check import or data format)")
                except Exception as e:
                    import traceback
                    print(
                        f"  [Warning] Visualization failed for batch {batch_idx}: {e}")
                    print(f"  [Debug] Traceback: {traceback.format_exc()}")

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    print("\n" + "=" * 70)
    print(f"📊 Semantic Conflict Test Results")
    print("=" * 70)

    metrics_baseline = evaluator_baseline.print_report(model_name_baseline)

    metrics_ours = evaluator_ours.print_report(model_name_ours)

    tr_improvement = None
    tdr_reduction = None
    miou_improvement = None

    if 'error' not in metrics_baseline and 'error' not in metrics_ours:
        print("\n" + "=" * 70)
        print("📊 COMPARISON SUMMARY")
        print("=" * 70)
        tr_improvement = metrics_ours.get('texture_robustness', 0) - \
            metrics_baseline.get('texture_robustness', 0)
        tdr_reduction = metrics_baseline.get('texture_deception_rate', 0) - \
            metrics_ours.get('texture_deception_rate', 0)
        miou_improvement = metrics_ours.get('miou_attacked_classes', 0) - \
            metrics_baseline.get('miou_attacked_classes', 0)

        print(
            f"   Texture Robustness Improvement:    +{tr_improvement:.2%} ⬆️")
        print(f"   Texture Deception Rate Reduction:  -{tdr_reduction:.2%} ⬇️")
        print(
            f"   mIoU Improvement on Attacked:      +{miou_improvement:.2%} ⬆️")
        print("=" * 70)

        if tr_improvement > 0.05 and tdr_reduction > 0.05:
            print("✅ SUCCESS: STS significantly improves geometry reliance!")
        elif tr_improvement > 0:
            print("⚠️ PARTIAL: STS shows improvement but may need stronger training.")
        else:
            print("❌ FAILED: STS does not improve robustness. Check implementation.")

    print(
        f"\n[Visualization] Generated {len(visualization_paths)} visualization images:")
    for path in visualization_paths:
        print(f"  - {path}")

    comparison_dict = {
        'tr_improvement': tr_improvement,
        'tdr_reduction': tdr_reduction,
        'miou_improvement': miou_improvement,
    }

    return {
        'baseline': metrics_baseline,
        'ours': metrics_ours,
        'comparison': comparison_dict
    }, visualization_paths
