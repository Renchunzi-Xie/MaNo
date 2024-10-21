import torch
from models.utils import get_model, get_imagenet_model


class Base_alg:
    def __init__(self, val_loader, device, args):
        super(Base_alg, self).__init__()
        # load the pre-trained model.
        self.get_path(args)
        # self.save_dir_path = './checkpoints/{}'.format(args['dataname'] + '_' + args['arch'])
        if ('imagenet' in args['dataname']) & (args['num_classes'] == 1000):
            self.base_model = get_imagenet_model(args['arch'], args['num_classes'], args['seed']).to(device)
        else:
            self.base_model = get_model(args['arch'], args['num_classes'], args['seed']).to(device)
            self.base_model.load_state_dict(torch.load('{}/base_model.pt'.format(self.save_dir_path), map_location=device))

        self.val_loader = val_loader
        self.device = device
        self.args = args

    def get_path(self, args):
        dataname = args['dataname']
        if args['dataname'] in ['pacs', 'office_home', 'domainnet']:
            self.save_dir_path = './checkpoints/{}'.format(args['source'] + '_' + args['arch'])
        else:
            self.save_dir_path = './checkpoints/{}'.format(dataname + '_' + args['arch'])

    def evaluation(self):
        score = 0
        return score

    def test(self):
        self.base_model.eval()
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                inputs, targets = batch_data[0], batch_data[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.base_model(inputs)
                _, predicted = outputs.max(1)
                total += len(predicted)
                correct += predicted.eq(targets).sum().item()
        return 100 * correct/total
