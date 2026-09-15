# EVOLVE-BLOCK-START

GPU_MEM_SIZE = 80  # GB


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # gready algorithm to place models to the GPUs with smallest gpu_id first

    placement = dict()
    for gpu_id in range(gpu_num):
        placement[gpu_id] = []

    for model in models:
        for gpu_id in range(gpu_num):
            if model.model_size <= GPU_MEM_SIZE - sum(
                model.model_size for model in placement[gpu_id]
            ):
                placement[gpu_id].append(model)
                break
    return placement


# EVOLVE-BLOCK-END
