import logging


def print_model_parameters(model):
    logging.info(f"{'Name':40} {'RequiresGrad':12} {'Dtype':15} {'Shape':20} {'#Params':10}")
    logging.info("-" * 100)

    total_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        bytes_ = param.element_size() * numel

        total_params += numel
        total_bytes += bytes_

        logging.info(
            f"{name:40} "
            f"{str(param.requires_grad):12} "
            f"{str(param.dtype):15} "
            f"{str(tuple(param.shape)):20} "
            f"{numel:10}"
        )

    logging.info("-" * 100)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Total size: {total_bytes / (1024**2):.2f} MB")
