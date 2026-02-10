def unpack_batch(batch, device: str):
    """Unpack (x, y, meta) batch and move tensors to device."""
    if len(batch) == 2:
        x, y = batch
        meta = None
    else:
        x, y, meta = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta
