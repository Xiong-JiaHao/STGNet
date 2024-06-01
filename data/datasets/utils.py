import sys


def get_dataset(split, arg_obj):
    """
    Load the specified dataset based on the argument object.

    Args:
        split (str): The data split to load, e.g., 'train', 'test', or 'valid'.
        arg_obj: An object containing the arguments passed to the program.

    Returns:
        DataSet: An instance of the dataset class corresponding to the specified dataset and split.

    Raises:
        SystemExit: If the specified dataset is not found, the function exits with an error code.
    """
    dataset = arg_obj.dataset.lower()

    if dataset == 'pure_supervised':
        from data.datasets.PURE_supervised import PURESupervised as DataSet
        print('Using PURE supervised dataset.')
        print('Using UBFC unsupervised dataset.')
    elif dataset == 'ubfc_supervised':
        from data.datasets.UBFC_supervised import UBFCSupervised as DataSet
        print('Using UBFC supervised dataset.')
    else:
        print('Dataset not found. Exiting.')
        sys.exit(-1)

    return DataSet(split, arg_obj)
