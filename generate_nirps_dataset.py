from configuration.nirps.config import parse_arguments_nirps
from architecture.nirps.train import NIRPSTrain

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = parse_arguments_nirps()

    for key_arg in ['dataset', 'model', 'source_domain', 'target_domain']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} Must Be Refered!'.format(key_arg))

    work = NIRPSTrain(args=args)
    work.run_work_flow()