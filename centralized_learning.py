from configuration.architecture.config import parse_arguments_centralized
from architecture.centralized.train import CentralizedTrain


def centralized_learning():
    args = parse_arguments_centralized()
    
    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} Rust Be Refered!'.format(key_arg))

    work = CentralizedTrain(args=args)
    work.run_work_flow()



if __name__ == '__main__':
    centralized_learning()