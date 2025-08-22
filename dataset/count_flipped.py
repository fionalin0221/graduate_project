import pandas as pd
from collections import defaultdict, deque

patch_size = 448
wsi = 2

for num_trial in range(1, 2):
    for gen in range(1, 5):
        df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/1{wsi:04d}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_with_gt.csv")
        # 先篩選 flipped_label 和 pred_label 不同的 patch
        df = df[df['flipped_label'] != df['pred_label']].copy()

        # 從 file_name 提取座標
        df[['x', 'y']] = df['file_name'].str.extract(r'(\d+)_(\d+)\.tif').astype(int)

        results = []

        # 按類別分組
        for cls, group in df.groupby('flipped_label'):
            coords_set = set(zip(group['x'], group['y']))
            visited = set()
            cluster_sizes = []

            # BFS 找群集
            for coord in coords_set:
                if coord not in visited:
                    queue = deque([coord])
                    visited.add(coord)
                    size = 0

                    while queue:
                        cx, cy = queue.popleft()
                        size += 1

                        for dx, dy in [(patch_size, 0), (-patch_size, 0), (0, patch_size), (0, -patch_size)]:
                            neighbor = (cx + dx, cy + dy)
                            if neighbor in coords_set and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                    cluster_sizes.append(size)

            # 統計群大小 vs 個數
            size_counts = defaultdict(int)
            for s in cluster_sizes:
                size_counts[s] += 1

            for size, count in size_counts.items():
                results.append({
                    "class": cls,
                    "cluster_size": size,
                    "count": count
                })

        # 轉成 DataFrame
        stats_df = pd.DataFrame(results).sort_values(by=["class", "cluster_size"])

        # 存成 csv
        stats_df.to_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/1{wsi:04d}_Gen{gen}_selected_patches_flipped_size_distribution.csv", index=False)

        print(stats_df)
