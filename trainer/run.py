from .work import Worker
import yaml
import os

def main():
    # Parse the arguments and load configuration
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    current_computer = config['current_computer']
    file_paths = config['computers'][current_computer]['file_paths']
    type = config['type']
    wsis = file_paths[f'{type}_wsis']

    worker = Worker(config)
    worker.train()
    # worker.train_multi_model()
    # worker.train_generation()
    # worker.test()
    # worker.contour_analysis_multi(0, )
    # for wsi in wsis:
        # worker.train_one_WSI(wsi)
        # worker.test_one_WSI(wsi)
        # worker.test_TATI(wsi, 0, save_path = None)
        # worker.plot_TI_Result(wsi, 0, save_path = None)
        # worker.contour_analysis(wsi, 0, save_path = None)


if __name__ == "__main__":
    main()