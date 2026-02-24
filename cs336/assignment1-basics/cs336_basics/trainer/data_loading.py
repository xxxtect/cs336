import torch
import numpy.typing as npt


def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_len = len(dataset)
    inputs = torch.empty(batch_size, context_length, dtype=torch.long)
    targets = torch.empty(batch_size, context_length, dtype=torch.long)
    for i in range(batch_size):
        # randomly select
        start_idx = torch.randint(0, dataset_len - context_length, (1,)).item()
        input_seq = dataset[start_idx: start_idx + context_length]
        input_target = dataset[start_idx + 1: start_idx + context_length + 1]
        inputs[i] = torch.tensor(input_seq, dtype=torch.long)
        targets[i] = torch.tensor(input_target, dtype=torch.long)

    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    return (inputs, targets)