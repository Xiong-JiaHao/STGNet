import sys

def get_dataset(split, arg_obj):
    dataset = arg_obj.dataset.lower()

    if dataset == 'pure_unsupervised':
        from data.datasets.PURE_unsupervised import PUREUnsupervised as DataSet
        print('Using PURE unsupervised dataset.')
    elif dataset == 'pure_supervised':
        from data.datasets.PURE_supervised import PURESupervised as DataSet
        print('Using PURE supervised dataset.')
    elif dataset == 'ubfc_unsupervised':
        from data.datasets.UBFC_unsupervised import UBFCUnsupervised as DataSet
        print('Using UBFC unsupervised dataset.')
    elif dataset == 'ubfc_supervised':
        from data.datasets.UBFC_supervised import UBFCSupervised as DataSet
        print('Using UBFC supervised dataset.')
    elif dataset == 'vipl_unsupervised':
        from data.datasets.VIPL_unsupervised import VIPLUnsupervised as DataSet
        print('Using VIPL unsupervised dataset.')
    elif dataset == 'vipl_supervised':
        from data.datasets.VIPL_supervised import VIPLSupervised as DataSet
        print('Using VIPL supervised dataset.')
    else:
        print('Dataset not found. Exiting.')
        sys.exit(-1)

    return DataSet(split, arg_obj)
