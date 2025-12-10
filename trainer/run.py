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
    test_type = file_paths[f'test_type']

    # wsis = file_paths[f'HCC_old_wsis']
    # wsis = file_paths[f'HCC_wsis']
    wsis = file_paths[f'CC_wsis']

    worker = Worker(config)
    # worker.train()
    # worker.train_multi_model()
    worker.train_generation(mode = 'selected', labeled = False,  replay = True)
    # worker.test()
    # worker.contour_analysis_multi()
    # for wsi in wsis:
        # worker.train_one_WSI(wsi)
        # worker.test_one_WSI(wsi)
        # worker.test_TATI(wsi, 0)
        # worker.plot_TI_Result(wsi, 0)
        # worker.train_generation_one_WSI(wsi, mode = 'selected', labeled=False)
        # worker.test_all(wsi, config['generation'], mode = 'selected')
        # for gen in range(config['generation']+1):
        #     worker.plot_all_result(wsi, gen, mode = 'selected', plot_type = 'pred', plot_heatmap = False, plot_boundary = True)
        #     if gen != 0:
        #         worker.test_flip(wsi, gen, mode = 'selected')
        #         worker.plot_all_result(wsi, gen, mode = 'selected', plot_type = 'flip', plot_boundary = True)
        #     worker.test_TATI(wsi, gen, mode = 'selected')
        #     worker.plot_TI_Result(wsi, gen, mode = 'selected')

        # worker.test_all(wsi, config['generation'], mode = 'selected', model_wsi = 'multi')
        # for gen in range(4, config['generation']+1):
        #     worker.plot_all_result(wsi, gen, mode = 'selected', plot_type = 'pred', model_wsi = 'multi', plot_heatmap = False, plot_boundary = True)
        #     if gen != 0:
        #         worker.test_flip(wsi, gen, mode = 'selected', model_wsi = 'multi')
        #         worker.plot_all_result(wsi, gen, mode = 'selected', plot_type = 'flip', model_wsi = 'multi', plot_boundary = True)
        #     worker.test_TATI(wsi, gen, mode = 'selected', model_wsi = 'multi')
        #     worker.plot_TI_Result(wsi, gen, mode = 'selected', model_wsi = 'multi')


if __name__ == "__main__":
    main()