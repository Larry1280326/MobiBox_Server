"""Model loading utilities for TSFM inference in MobiBox.

Adapted from TSFM val_scripts/human_activity_recognition/model_loading.py
for use within the MobiBox service package.
"""

import json
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn as nn

from .encoder import IMUActivityRecognitionEncoder
from .semantic_alignment import SemanticAlignmentHead
from .token_text_encoder import LearnableLabelBank, TokenTextEncoder, ChannelTextFusion


class SemanticAlignmentModel(nn.Module):
    """
    Semantic alignment model for inference.

    Combines encoder, semantic head, and channel text fusion for
    generating semantic embeddings from IMU data.
    """

    def __init__(
        self,
        encoder: IMUActivityRecognitionEncoder,
        semantic_head: SemanticAlignmentHead,
        num_heads: int = 4,
        dropout: float = 0.1,
        text_encoder: Optional[TokenTextEncoder] = None,
        text_dim: int = None
    ):
        super().__init__()
        self.encoder = encoder
        self.semantic_head = semantic_head
        self.semantic_dim = semantic_head.output_dim

        # Token-level text encoder (frozen backbone)
        self.text_encoder = text_encoder if text_encoder is not None else TokenTextEncoder()

        # Channel text fusion
        self.channel_fusion = ChannelTextFusion(
            d_model=encoder.d_model,
            num_heads=num_heads,
            dropout=dropout,
            text_dim=text_dim
        )

    def forward(
        self,
        patches: torch.Tensor,
        channel_descriptions: List[List[str]],
        channel_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-patched data.

        Args:
            patches: (batch, max_patches, target_patch_size, max_channels) padded patches
            channel_descriptions: List of channel description lists per sample
            channel_mask: (batch, max_channels) boolean mask for valid channels
            patch_mask: (batch, max_patches) boolean mask for valid patches

        Returns:
            Semantic embeddings of shape (batch, semantic_dim)
        """
        batch_size = patches.shape[0]
        max_channels = patches.shape[3]
        device = patches.device

        # Pad channel descriptions to max_channels
        batched_channel_descs = [
            descs + ["[PAD]"] * (max_channels - len(descs))
            for descs in channel_descriptions
        ]

        # Encode patches
        encoded_batch = self.encoder(
            patches,
            batched_channel_descs,
            channel_mask=channel_mask,
            patch_attention_mask=patch_mask
        )

        # Batch-encode channel descriptions
        all_descs = [desc for descs in batched_channel_descs for desc in descs]
        all_tokens, all_masks = self.text_encoder.encode(all_descs, device)

        # Reshape to (B, C, seq_len, D) for batched fusion
        seq_len = all_tokens.shape[1]
        text_tokens = all_tokens.reshape(batch_size, max_channels, seq_len, -1)
        text_masks = all_masks.reshape(batch_size, max_channels, seq_len)

        # Apply channel text fusion
        encoded_batch = self.channel_fusion(encoded_batch, text_tokens, text_masks)

        # Get semantic embedding
        return self.semantic_head(
            encoded_batch,
            channel_mask=channel_mask,
            patch_mask=patch_mask,
            normalize=True
        )


def _load_hyperparams(hyperparams_path: Path) -> dict:
    """Load and normalize hyperparameters from JSON, handling both formats."""
    with open(hyperparams_path) as f:
        hp = json.load(f)

    # New format: 'config' contains the full architecture dict
    if 'config' in hp:
        cfg = hp['config']
        return {
            'encoder': {
                'd_model': cfg['d_model'],
                'num_heads': cfg['num_heads'],
                'num_temporal_layers': cfg['num_temporal_layers'],
                'dim_feedforward': cfg['dim_feedforward'],
                'dropout': cfg.get('dropout', 0.1),
                'use_cross_channel': cfg.get('use_cross_channel', True),
                'cnn_channels': cfg.get('cnn_channels', [32, 64]),
                'cnn_kernel_sizes': cfg.get('cnn_kernel_sizes', [5]),
                'target_patch_size': cfg.get('target_patch_size', 64),
                'use_channel_encoding': False,
                'feature_extractor_type': cfg.get('feature_extractor_type', 'cnn'),
                'spectral_ratio': cfg.get('spectral_ratio', 0.25),
            },
            'semantic_head': {
                'd_model_fused': cfg.get('d_model_fused', cfg['d_model']),
                'semantic_dim': cfg.get('semantic_dim', cfg['d_model']),
                'num_temporal_layers': cfg.get('num_semantic_temporal_layers', 2),
                'num_fusion_queries': cfg.get('num_fusion_queries', 4),
                'use_fusion_self_attention': cfg.get('use_fusion_self_attention', True),
                'num_pool_queries': cfg.get('num_pool_queries', 4),
                'use_pool_self_attention': cfg.get('use_pool_self_attention', True),
                'per_patch_prediction': cfg.get('per_patch_prediction', False),
            },
            'channel_text_fusion': {
                'num_heads': cfg.get('channel_text_num_heads', 4),
            },
            'label_bank': {
                'sentence_bert_model': cfg.get('contrastive_text_model', cfg.get('sentence_bert_model', 'all-MiniLM-L6-v2')),
                'd_model': cfg.get('semantic_dim', cfg['d_model']),
                'num_heads': cfg.get('label_bank_num_heads', 4),
                'num_queries': cfg.get('label_bank_num_queries', 4),
                'num_prototypes': cfg.get('label_bank_num_prototypes', 1),
            },
        }

    # Legacy format: separate sections
    enc = hp.get('encoder', {})
    sem = hp.get('semantic', {})
    head = hp.get('semantic_head', {})
    tok = hp.get('token_level_text', {})
    d_model = enc.get('d_model', 384)
    return {
        'encoder': {
            'd_model': d_model,
            'num_heads': enc.get('num_heads', 8),
            'num_temporal_layers': enc.get('num_temporal_layers', 4),
            'dim_feedforward': enc.get('dim_feedforward', 1536),
            'dropout': enc.get('dropout', 0.1),
            'use_cross_channel': enc.get('use_cross_channel', True),
            'cnn_channels': enc.get('cnn_channels', [32, 64]),
            'cnn_kernel_sizes': enc.get('cnn_kernel_sizes', [5]),
            'target_patch_size': enc.get('target_patch_size', 64),
            'use_channel_encoding': enc.get('use_channel_encoding', False),
            'feature_extractor_type': enc.get('feature_extractor_type', 'cnn'),
            'spectral_ratio': enc.get('spectral_ratio', 0.25),
        },
        'semantic_head': {
            'd_model_fused': sem.get('d_model_fused', d_model),
            'semantic_dim': sem.get('semantic_dim', d_model),
            'num_temporal_layers': head.get('num_temporal_layers', 2),
            'num_fusion_queries': head.get('num_fusion_queries', 4),
            'use_fusion_self_attention': head.get('use_fusion_self_attention', True),
            'num_pool_queries': head.get('num_pool_queries', 4),
            'use_pool_self_attention': head.get('use_pool_self_attention', True),
        },
        'channel_text_fusion': {
            'num_heads': tok.get('num_heads', 4),
        },
        'label_bank': {
            'sentence_bert_model': sem.get('contrastive_text_model', sem.get('sentence_bert_model', 'all-MiniLM-L6-v2')),
            'd_model': sem.get('semantic_dim', d_model),
            'num_heads': tok.get('num_heads', 4),
            'num_queries': tok.get('num_queries', 4),
            'num_prototypes': tok.get('num_prototypes', 1),
        },
    }


def load_model(
    checkpoint_path: str,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[SemanticAlignmentModel, dict, Path]:
    """
    Load a SemanticAlignmentModel from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.
        device: Device to load the model onto.
        verbose: Whether to print loading information.

    Returns:
        model: The loaded SemanticAlignmentModel in inference mode.
        checkpoint: The full checkpoint dict.
        hyperparams_path: Path to hyperparameters.json alongside the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint to CPU first to avoid MPS issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')

    # Load hyperparameters from checkpoint directory
    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    if not hyperparams_path.exists():
        raise FileNotFoundError(
            f"hyperparameters.json not found at {hyperparams_path}. "
            "This checkpoint may be from an older incompatible version."
        )

    hp = _load_hyperparams(hyperparams_path)
    enc_cfg = hp['encoder']
    head_cfg = hp['semantic_head']
    fusion_cfg = hp['channel_text_fusion']
    lb_cfg = hp['label_bank']

    d_model = enc_cfg['d_model']

    # Create encoder
    encoder = IMUActivityRecognitionEncoder(
        d_model=d_model,
        num_heads=enc_cfg['num_heads'],
        num_temporal_layers=enc_cfg['num_temporal_layers'],
        dim_feedforward=enc_cfg['dim_feedforward'],
        dropout=enc_cfg['dropout'],
        use_cross_channel=enc_cfg['use_cross_channel'],
        cnn_channels=enc_cfg['cnn_channels'],
        cnn_kernel_sizes=enc_cfg['cnn_kernel_sizes'],
        target_patch_size=enc_cfg['target_patch_size'],
        use_channel_encoding=enc_cfg['use_channel_encoding'],
        feature_extractor_type=enc_cfg.get('feature_extractor_type', 'cnn'),
        spectral_ratio=enc_cfg.get('spectral_ratio', 0.25),
    )

    # Create semantic head
    semantic_head = SemanticAlignmentHead(
        d_model=d_model,
        d_model_fused=head_cfg['d_model_fused'],
        output_dim=head_cfg['semantic_dim'],
        num_temporal_layers=head_cfg['num_temporal_layers'],
        num_heads=enc_cfg['num_heads'],
        dim_feedforward=head_cfg['d_model_fused'] * 4,
        dropout=enc_cfg['dropout'],
        num_fusion_queries=head_cfg['num_fusion_queries'],
        use_fusion_self_attention=head_cfg['use_fusion_self_attention'],
        num_pool_queries=head_cfg['num_pool_queries'],
        use_pool_self_attention=head_cfg['use_pool_self_attention'],
        per_patch_prediction=head_cfg.get('per_patch_prediction', False),
    )

    # Create shared text encoder
    shared_text_encoder = TokenTextEncoder(model_name=lb_cfg['sentence_bert_model'])
    contrastive_text_dim = lb_cfg['d_model']

    # Create full model
    model = SemanticAlignmentModel(
        encoder,
        semantic_head,
        num_heads=fusion_cfg['num_heads'],
        dropout=enc_cfg['dropout'],
        text_encoder=shared_text_encoder,
        text_dim=contrastive_text_dim,
    )

    # Convert legacy combined gate weights to split gate format if needed
    state_dict = checkpoint['model_state_dict']
    old_gate_w = 'channel_fusion.gate.0.weight'
    old_gate_b = 'channel_fusion.gate.0.bias'
    if old_gate_w in state_dict and old_gate_b in state_dict:
        W = state_dict.pop(old_gate_w)
        b = state_dict.pop(old_gate_b)
        d = W.shape[0]
        state_dict['channel_fusion.gate_sensor.weight'] = W[:, :d]
        state_dict['channel_fusion.gate_channel.weight'] = W[:, d:]
        state_dict['channel_fusion.gate_channel.bias'] = b

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    benign_patterns = ('channel_encoding',)
    benign_unexpected = [k for k in unexpected_keys if any(p in k for p in benign_patterns)]
    critical_unexpected = [k for k in unexpected_keys if not any(p in k for p in benign_patterns)]
    if critical_unexpected:
        raise RuntimeError(
            f"Checkpoint has {len(critical_unexpected)} unexpected keys: {critical_unexpected[:10]}"
        )
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing_keys)} keys: {missing_keys[:10]}"
        )

    model.eval()
    model = model.to(device)

    if verbose:
        print(f"  Loaded checkpoint from epoch {epoch}")
        print(f"  Encoder: d_model={d_model}, layers={enc_cfg['num_temporal_layers']}, heads={enc_cfg['num_heads']}")
        print(f"  Semantic head: d_fused={head_cfg['d_model_fused']}, layers={head_cfg['num_temporal_layers']}")

    return model, checkpoint, hyperparams_path


def load_label_bank(
    checkpoint: dict,
    device: torch.device,
    hyperparams_path: Path,
    verbose: bool = True,
    text_encoder: TokenTextEncoder = None,
) -> LearnableLabelBank:
    """
    Load a LearnableLabelBank with trained state from a checkpoint.

    Args:
        checkpoint: The checkpoint dict (from torch.load or load_model).
        device: Device to load the label bank onto.
        hyperparams_path: Path to hyperparameters.json.
        verbose: Whether to print loading information.
        text_encoder: Optional shared TokenTextEncoder.

    Returns:
        label_bank: The loaded LearnableLabelBank in inference mode.
    """
    hp = _load_hyperparams(hyperparams_path)
    lb_cfg = hp['label_bank']

    label_bank = LearnableLabelBank(
        model_name=lb_cfg['sentence_bert_model'],
        device=device,
        d_model=lb_cfg['d_model'],
        num_heads=lb_cfg['num_heads'],
        num_queries=lb_cfg['num_queries'],
        num_prototypes=lb_cfg['num_prototypes'],
        dropout=0.0,
        text_encoder=text_encoder,
    )

    if 'label_bank_state_dict' in checkpoint:
        label_bank.load_state_dict(checkpoint['label_bank_state_dict'])
        if verbose:
            print(f"  Loaded LearnableLabelBank state (d={lb_cfg['d_model']}, heads={lb_cfg['num_heads']}, queries={lb_cfg['num_queries']})")
    else:
        if verbose:
            print("  Warning: No label_bank_state_dict in checkpoint, using untrained weights")

    label_bank.eval()
    return label_bank