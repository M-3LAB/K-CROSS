from configuration.architecture.config import parse_arguments_federated
from architecture.federated.cyclegan import FedCycleGAN
from architecture.federated.munit import FedMunit
from architecture.federated.unit import FedUnit



def federated_learning():
    args = parse_arguments_federated()

    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} Must Be Refered!'.format(key_arg))

    if args.model == 'cyclegan':
        work = FedCycleGAN(args=args)
    elif args.model == 'munit':
        work = FedMunit(args=args)
    elif args.model == 'unit':
        work = FedUnit(args=args)
    else:
        raise ValueError('Model Is Invalid!')   

    work.run_work_flow()





if __name__ == '__main__':
    federated_learning()