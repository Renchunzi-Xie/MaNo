from algs.mano import MaNo


def create_alg(alg_name, val_loader, device, args):
    alg_dict = {

        'mano': MaNo,

    }
    model = alg_dict[alg_name](val_loader, device, args)
    return model
