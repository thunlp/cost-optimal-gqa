from pathlib import Path
from typing import Union, Optional, List, Dict
import time

import numpy as np
from torch import Tensor
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def build_dataset(
    tokenizer: PreTrainedTokenizerBase,
    data_dir: Union[str, Path] = "/home/test/test07/data/slimpj-chunked",
    streaming: bool = True,
    n_workers: int = 8,
    overwrite_cache: bool = False,
    token_ids_only: bool = True,
    max_len: int = 512,
    eos_token_id: Optional[int] = None,
    shift_labels: bool = False,
    **kwargs,
):
    """
    Returns an iterable of batches of token IDs.

    This will use `load_dataset` from the HuggingFace Datasets library to load the
    data from `data_dir`, tokenize each example, concatenate the input IDs, add an
    EOS token ID at the end of each sequence, then split into chunks of `max_len`
    tokens, and return a tensor of (batch_size, max_len).
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    assert eos_token_id is not None

    # Get all data_files
    data_dir = Path(data_dir).absolute()
    all_data_files = []
    for chunk_dir in sorted(data_dir.glob('chunk*')):
        all_data_files += sorted(chunk_dir.glob("*.jsonl"))
    all_data_files = [str(x) for x in all_data_files]

    data_files = {
        "train": all_data_files,
    }
    raw_dataset = load_dataset(
        str(data_dir),
        data_files=data_files,
        streaming=streaming,
        split='train',
    )

    text_column_name = 'text'
    col_names = ['text', 'meta']
    # text_column_name = "input"
    # col_names = ["input", "output"]

    # Tokenize in streaming mode
    def process_fn(examples: dict) -> Dict[str, Tensor]:
        '''
        A process function to use with `Dataset.map`. It tokenizes
        texts in the batch, concatenate them, and split into chunks
        with `max_len` tokens (discarding the last chunk if
        incomplete).
        '''
        texts: List[str] = examples[text_column_name]
        # print("##### PROCESS_FN #####")
        # print("texts:", len(texts), texts[0][:50])
        # t0 = time.time()
        encodings = tokenizer(texts, max_length=10 ** 6, truncation=True, return_tensors='np')
        # print('tokenize time', time.time() - t0)

        # Append EOS token
        orig_input_ids: np.ndarray = encodings['input_ids']
        batch_ids = [np.append(ids, eos_token_id) for ids in orig_input_ids]
        # t0 = time.time()
        concat_ids = np.concatenate(batch_ids)
        # print('concate time', time.time() - t0)
        total_len = concat_ids.shape[0]
        if shift_labels:
            chunk_len = max_len + 1  # The input IDs with be chunk_len - 1
        else:
            chunk_len = max_len

        # Rounded down to multiple of chunk_len.
        # So the last remainder chunk is discarded.
        total_len = total_len // chunk_len * chunk_len
        n_chunks = total_len // chunk_len
        # print(f"{total_len = :,}, {chunk_len = }, {n_chunks = }")

        if shift_labels:
            input_ids = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
            labels = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
        else:
            input_ids = np.zeros((n_chunks, chunk_len), dtype=np.int64)
            labels = np.zeros((n_chunks, chunk_len), dtype=np.int64)

        for i in range(n_chunks):
            this_chunk = concat_ids[i * chunk_len : (i + 1) * chunk_len]
            if shift_labels:
                # Next token prediction with teacher forcing
                input_ids[i] = this_chunk[:-1]
                labels[i] = this_chunk[1:]
            else:
                input_ids[i] = this_chunk
                labels[i] = this_chunk

        input_ids: Tensor = torch.from_numpy(input_ids)
        labels: Tensor = torch.from_numpy(labels)

        # print(f"{input_ids.shape = }")

        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }
        return batch

    if streaming:
        # Streaming dataset does not support multi-processing yet.
        tokenized_dataset = raw_dataset.map(
            process_fn,
            batched=True,
            remove_columns=col_names if token_ids_only else [],
        )
    else:
        tokenized_dataset = raw_dataset.map(
            process_fn,
            batched=True,
            num_proc=n_workers,
            remove_columns=col_names if token_ids_only else [],
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return tokenized_dataset
