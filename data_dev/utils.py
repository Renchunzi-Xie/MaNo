from data.cifar10 import *
from data.cifar100 import *
from data.tinyimagenet import *
from data.imagenet import *
from data.pacs import *
from data.office_home import *
import torch
from data.wilds_rr1 import *
from data.breeds import *

def build_dataloader(dataname, args):
    random_seeds = torch.randint(0, 10000, (2,))
    if args['severity'] == 0:
        seed = 1
        datatype = 'train'
        corruption_type = 'clean'
    else:
        corruption_type = args['corruption']
        seed = random_seeds[1]
        datatype = 'test'
    if dataname == 'cifar10':
        valset = load_cifar10_image(corruption_type,
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype=datatype,
                                    seed=seed,
                                    num_samples=args['num_samples'])
    elif dataname == 'cifar100':
        valset = load_cifar100_image(corruption_type,
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype=datatype,
                                    seed=seed,
                                    num_samples=args['num_samples'])
    elif dataname == 'tinyimagenet':
        valset = load_tinyimagenet(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'imagenet':
        valset = load_Imagenet(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               num_classes= args['num_classes'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif (dataname == 'pacs') or (dataname == 'domainnet'):
        if (dataname == 'pacs') and (corruption_type == 'sketch_pacs'):
            corruption_type = 'sketch'
        valset = get_pacs_loader(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'office_home':
        valset = get_offce_home_loader(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'wilds_rr1':
        valset = load_wilds_rxrx1(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif (dataname == 'entity13') or (dataname == 'entity30') or (dataname == 'living17') or (dataname == 'nonliving26'):
        if (datatype == 'train') and (args['alg']!='frechet'):
            name = args['train_data_name']
        else:
            name = args['dataname']

        valset = get_imagenet_breeds(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               name=name,
                               corruption_severity=args['severity'],
                               datatype=datatype)
    else:
        raise Exception("Not Implemented Error")

    valset_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=args['batch_size'],
                                                num_workers = 4,
                                                shuffle=True)
    return valset_loader
