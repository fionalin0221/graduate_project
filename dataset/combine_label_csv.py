import pandas as pd
import os
import yaml

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
current_computer = config['current_computer']
wsi_type = config['type']
file_paths = config['computers'][current_computer]['file_paths']
state = config['state']
print(wsi_type, state)

csv_dir = file_paths['HCC_csv_dir'] if wsi_type == "HCC" else file_paths['CC_csv_dir']
wsis = file_paths[f'{wsi_type}_wsis']

def process():
    for wsi in wsis:
        if wsi_type == 'HCC':
            _wsi = wsi if state == 'old' else wsi + 91
            csv_path = f'{csv_dir}/{_wsi}'
        elif wsi_type == 'CC':
            _wsi = f'1{wsi:04d}'
            csv_path = f'{csv_dir}/{wsi}'
            
        file2_path = f'{csv_path}/{_wsi}_patch_in_region_filter_2_v2.csv'
        file3_path = f'{csv_path}/{_wsi}_patch_in_region_filter_3_v2.csv'
        file3_new_name = f'{csv_path}/{_wsi}_patch_in_region_filter_3_v2_fibrosis.csv'

        if os.path.exists(file3_path):
            os.rename(file3_path, file3_new_name)
        
        df2 = pd.read_csv(file2_path)
        df3_fibrosis = pd.read_csv(file3_new_name)

        df3_f_only = df3_fibrosis[df3_fibrosis['label'] == 'F']

        df_combined = pd.concat([df2, df3_f_only], ignore_index=True)
        df_result = df_combined.drop_duplicates(subset='file_name', keep='last')

        df_result.to_csv(file3_path, index=False)
        print(f'already save {file3_path}')

def check():
    for wsi in wsis:
        correct = True
        if wsi_type == 'HCC':
            _wsi = wsi if state == 'old' else wsi + 91
            csv_path = f'{csv_dir}/{_wsi}'
        elif wsi_type == 'CC':
            _wsi = f'1{wsi:04d}'
            csv_path = f'{csv_dir}/{wsi}'
        print(_wsi)

        file2_path = f'{csv_path}/{_wsi}_patch_in_region_filter_2_v2.csv'
        file3_path = f'{csv_path}/{_wsi}_patch_in_region_filter_3_v2_fibrosis.csv'
        final_path = f'{csv_path}/{_wsi}_patch_in_region_filter_3_v2.csv'
        
        df2 = pd.read_csv(file2_path)
        df3_fib = pd.read_csv(file3_path)
        df_final = pd.read_csv(final_path)

        has_duplicates = df_final['file_name'].duplicated().any()
        if has_duplicates:
            print('error: duplicate!')
            correct = False

        df3_f_only = df3_fib[df3_fib['label'] == 'F']

        set_final = set(df_final['file_name'])
        set_f_only = set(df3_f_only['file_name'])
        is_df3_fully_in = set_f_only.issubset(set_final)
        if not is_df3_fully_in:
            print('error: not all fibrosis in final csv.')
            correct = False
        
        set_df2 = set(df2['file_name'])
        expected_count = len(set_df2.union(set_f_only))
        if len(df_final) == expected_count:
            print(f'the merge count is correct: {len(df_final)}=={expected_count}')
        else:
            print(f'error: expected count {expected_count} is not the same as real count {len(df_final)}.')
            correct = False

        overlap = set_df2.intersection(set_f_only)
        if overlap:
            sample_fn = list(overlap)[0]
            final_label = df_final[df_final['file_name'] == sample_fn]['label'].values[0]
            if final_label != 'F':
                print('error: merge label is wrong.')
                correct = False
        
        if correct:
            print('all check has pass.')

if __name__ == '__main__':
    process()