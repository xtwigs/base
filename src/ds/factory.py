from src.ds.base import BaseDataset
from src.ds.iwslt17 import IWSLT17
from src.ds.wmt23_10concat import WMT23_10concat


def build_dataset(
    name: str,
    source: str,
    target: str,
    is_encoder_decoder: bool,
    train_batch_size: int,
    val_batch_size: int,
) -> BaseDataset:
    ds_dict = dict(
        (cls.name, cls)
        for cls in (
            IWSLT17,
            WMT23_10concat,
        )
    )

    assert name in ds_dict.keys(), f"Dataset {name} not supported."
    return ds_dict[name](
        source, target, is_encoder_decoder, train_batch_size, val_batch_size
    )
