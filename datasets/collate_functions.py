#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: collate_functions.py

import torch
from typing import List


def tagger_collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, attention_mask, wordpiece_label_idx_lst
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(3):
        # 0 -> tokens
        # 1 -> token_type_ids
        # 2 -> attention_mask
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # 3 -> sequence_label
    # -100 is ignore_index in the cross-entropy loss function.
    pad_output = torch.full([batch_size, max_length], -100, dtype=batch[0][3].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    # 4 -> is word_piece_label
    pad_output = torch.full([batch_size, max_length], -100, dtype=batch[0][4].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][4]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    return output


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))

    return output

def mul_collate_fun(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    # [inputs, start_labels, end_labels, start_labels_mask, end_labels_mask, match_labels]

    all_input_ids = torch.cat([b[0]['input_ids'] for b in batch], dim=0)
    all_token_type_ids = torch.cat([b[0]['token_type_ids'] for b in batch], dim=0)
    all_attention_mask = torch.cat([b[0]['attention_mask'] for b in batch], dim=0)
    all_start_labels = torch.cat([b[1].unsqueeze(0) for b in batch],dim=0)
    all_end_labels = torch.cat([b[2].unsqueeze(0) for b in batch], dim=0)
    all_start_labels_mask = torch.cat([b[3].unsqueeze(0) for b in batch], dim=0)
    all_end_labels_mask = torch.cat([b[4].unsqueeze(0) for b in batch], dim=0)
    all_match_labels = torch.cat([b[5].unsqueeze(0) for b in batch], dim=0)
    return all_input_ids, all_token_type_ids, all_attention_mask, all_start_labels,all_end_labels,all_start_labels_mask,all_end_labels_mask,all_match_labels

