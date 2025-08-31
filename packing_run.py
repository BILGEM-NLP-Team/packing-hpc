from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any, Optional, Literal


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    
from dataclasses import asdict, dataclass, field
import torch
import torch.nn.functional as F
IGNORE_INDEX = -100
    
@dataclass
class DataArguments:
    cutoff_len: int
    neat_packing: bool
    
@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...

import bisect
def search_for_fit(numbers: list[int], capacity: int) -> int:

    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)

def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:

    numbers.sort() 
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


@dataclass
class PackedPretrainDatasetProcessor(DatasetProcessor):
    r"""
    neatpcking true
    """

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        eos_token = self.tokenizer.eos_token
        docs = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

        tokenized = self.tokenizer(docs, add_special_tokens=True)["input_ids"]

        block_size = self.data_args.cutoff_len
        from collections import defaultdict

        chunks = []
        lengths = []
        length2indexes = defaultdict(list)

        for idx, toks in enumerate(tokenized):
            if len(toks) <= block_size:
                chunks.append(toks)
            else:
                parts = [toks[i:i+block_size] for i in range(0, len(toks), block_size)] # stride eklenebilir contextler arası overlap olması açısından.
                chunks.extend(parts)

        for i, c in enumerate(chunks):
            l = len(c)
            lengths.append(l)
            length2indexes[l].append(i)  

        knapsacks  = greedy_knapsack(lengths, block_size) 
       
        model_inputs = defaultdict(list)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        MAX_SEG_ID = 127
        
        for sack in knapsacks:
            packed_ids, packed_attn, packed_pos = [], [], []
            for seg_idx, seg_len in enumerate(sack):
                
                # --- INT8 SINIRINI AŞAN SEGMENTİ ATLAMA --- onlar zaten çok çok kısa verilerdir önceden filtrelenmiş olması gerekirdi.
                if seg_idx >= MAX_SEG_ID:
                    # logger.warning_rank0(
                    #     f"Segment {seg_idx} (>{MAX_SEG_ID-1}) atlandı; "
                    #     "attention_mask int8 aralığını koru."
                    # )
                    continue
                # ------------------------------------------
                
                ch_idx = length2indexes[seg_len].pop()
                seq = chunks[ch_idx]

                packed_ids.extend(seq)
                # packed_pos.extend(range(len(seq)))
                if self.data_args.neat_packing:
                    packed_attn.extend([seg_idx + 1] * len(seq))
                else:
                    packed_attn.extend([1] * len(seq))

            # fa2 korumasi?
            pad_len = block_size + 1 - len(packed_ids)
            if pad_len < 0:
                packed_ids = packed_ids[: block_size + 1]         
                packed_attn = packed_attn[: block_size + 1]
                # packed_pos  = packed_pos[:  block_size + 1]
                pad_len = 0

            if pad_len:
                packed_ids.extend([pad_id] * pad_len)
                packed_attn.extend([0] * pad_len) 
                # packed_pos.extend([0] * pad_len)

            model_inputs["input_ids"].append(packed_ids)
            model_inputs["attention_mask"].append(packed_attn)
            # model_inputs["position_ids"].append(packed_pos) # gerek yok
           
        return model_inputs

    # 
    # def print_data_example(self, ex):
    #     print(self.tokenizer.decode(ex["input_ids"], skip_special_tokens=False))
    
    def print_data_example(self, ex):
        print("input_ids:\n{}".format(ex["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(ex["input_ids"], skip_special_tokens=False)))


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


@dataclass 
class PretrainDataCollatorWith4DAttentionMask:
 
    tokenizer: Any                                
    pad_to_multiple_of: int | None = 8            
    label_pad_token_id: int = IGNORE_INDEX        
    block_diag_attn: bool = True
    attn_implementation: Literal[
        "eager", "sdpa", "flash_attention_2", "flash_attention_3"
    ] = "flash_attention_3"
    compute_dtype: torch.dtype = torch.bfloat16

    def _pad_field(self, features, key, pad_id):
        longest = max(len(f[key]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            longest = ( (longest + m - 1) // m ) * m
        for f in features:
            f[key].extend([pad_id] * (longest - len(f[key])))

    def __call__(self, features):
        # input_ids, attention_mask aynı uzunlukta olsun
        self._pad_field(features, "input_ids",      self.tokenizer.pad_token_id)
        self._pad_field(features, "attention_mask", 0)
        
        batch = {
            k: torch.tensor([f[k] for f in features])
            for k in ("input_ids", "attention_mask")
        }


        #labels = input, fakat pad token’ları IGNORE_INDEX (veya verilen id)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        batch["labels"] = labels

        if self.block_diag_attn and (self.attn_implementation != "flash_attention_2" and self.attn_implementation != "flash_attention_3"):
            batch["attention_mask"] = prepare_4d_attention_mask(
                batch["attention_mask"], self.compute_dtype
            )

        for k, v in batch.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                batch[k] = v.to(self.compute_dtype)

        return batch
    
    
def get_seqlens_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:

    bsz      = attention_mask.size(0)
    device   = attention_mask.device
    max_num  = int(attention_mask.max().item())       

    counts = torch.zeros((bsz, max_num), dtype=torch.int32, device=device)

    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1, dtype=torch.int32)

    seqlens = counts.flatten()
    seqlens = seqlens[seqlens.nonzero(as_tuple=True)].contiguous()  # int32 kalıyor, her modelde çalışsın
    return seqlens

def get_unpad_data(attention_mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", int]:

    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch




def configure_packing(block_diag_attn: bool, is_trainable: bool) -> None:
    if not is_trainable or not block_diag_attn:
        return

    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
    