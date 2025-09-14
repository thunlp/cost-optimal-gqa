from typing import Optional

from datasets import IterableDataset

from . import slimpj


def get_data(
    tokenizer,
    data_name: str = "slimpj",
    data_path: Optional[str] = None,
    max_len: int = 512,
    shift_labels: bool = False,
    is_seq2seq: bool = False,
    **kwargs,
) -> IterableDataset:
    '''
    Will return an IterableDataset
    '''
    if data_name == "slimpj":
        assert data_path is not None
        return slimpj.build_dataset(
            tokenizer,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            **kwargs,
        )  # type: ignore
    else:
        raise ValueError(f"Unknown data name: {data_name}")
