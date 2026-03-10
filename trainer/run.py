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
    wsis = file_paths[f'HCC_wsis']
    # wsis = file_paths[f'CC_wsis']

    worker = Worker(config)
    # worker.train()
    # worker.train_multi_model()
    # worker.train_generation(labeled = False,  replay = True)

    for wsi in wsis:
        # One WSI Train / Test
        # worker.train_one_WSI(wsi)
        # worker.test_TATI(wsi, 0)
        # worker.plot_TI_Result(wsi, 0)

        # Classification Test
        # worker.test_TATI(wsi, 0, model_wsi= 'multi')
        # worker.plot_TI_Result(wsi, 0, model_wsi= 'multi')

        # Two Stage Test
        # worker.test_TATI_two_stage(wsi, 0, model_wsi= 'multi')
        # worker.test_all(wsi, 0)
        # worker.plot_all_result(wsi, 0, plot_type = 'pred', plot_heatmap = False, plot_boundary = True)

        # Error Rate Train / Test
        # worker.train_on_error_rate(wsi, labeled=False, replay=False)
        # worker.test_TATI(wsi, 0)

        # Generation Training - 1WSI
        worker.train_generation_one_WSI(wsi, labeled=False, replay=False)
        worker.test_all(wsi, config['generation'])
        for gen in range(config['generation']+1):
            if gen != 0:
                worker.test_flip(wsi, gen)
                worker.plot_all_result(wsi, gen, plot_type = 'flip', plot_boundary = True)
                worker.plot_all_result(wsi, gen, plot_type = 'pred', plot_heatmap = False, plot_boundary = True)
            worker.test_TATI(wsi, gen)
            worker.plot_TI_Result(wsi, gen)

        # Generation Training - Multi WSI
        # worker.test_all(wsi, config['generation'], model_wsi='multi')
        # for gen in range(config['generation']+1):
        #     if gen != 0:
        #         worker.test_flip(wsi, gen, model_wsi='multi')
        #         worker.plot_all_result(wsi, gen, plot_type = 'flip', plot_boundary = True)
        #         worker.plot_all_result(wsi, gen, plot_type = 'pred', plot_heatmap = False, plot_boundary = True)
        #     worker.test_TATI(wsi, gen, model_wsi='multi')
        #     worker.plot_TI_Result(wsi, gen, model_wsi='multi')


if __name__ == "__main__":
    main()