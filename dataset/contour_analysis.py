def contour_analysis(self, wsi, gen, save_path):
    if save_path == None:
        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
        save_path = f'{self.save_dir}/100WTC_Result/{_wsi}/trial_{self.num_trial}'

    if gen == 0:
        condition = f'{self.class_num}_class'
    else:
        condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"

    df = pd.read_csv(f"{save_path}/Metric/{_wsi}_{condition}_labels_predictions.csv")
    # df = pd.read_csv(f"{save_path}/TI/{_wsi}_{condition}_patch_in_region_filter_2_v2_TI.csv")
    
    all_patches = df['file_name'].to_list() #patches_in_hcc_hulls

    ### Get (x, y, pseudo-label) of every patch ###
    all_pts = []
    for idx, img_name in enumerate(all_patches):
        if self.state == "old":
            match = re.search(r'-(\d+)-(\d+)-\d{5}x\d{5}', img_name)
            if match:
                x = match.group(1)
                y = match.group(2)
            else:
                print("Style Error")
        else:
            x, y = img_name[:-4].split('_')

        label = self.classes.index(df['pred_label'][idx])  # N=0, H=1
        # label = 1 if df['H_pred'][idx] > df["N_pred"][idx] else 0

        x = (int(x)) // self.patch_size
        y = (int(y)) // self.patch_size
        
        all_pts.append([x, y, label])  #label 0,1

    all_pts = np.array(all_pts)
    
    ### First sorted pts on x, then on y ###
    sorted_index = np.lexsort((all_pts[:, 1], all_pts[:, 0]))
    sorted_all_pts = all_pts[sorted_index]

    x_max, y_max = np.max(sorted_all_pts[:, 0]), np.max(sorted_all_pts[:, 1])

    label_map = np.full((y_max + 1, x_max + 1), -1, dtype=np.int32)

    # Fill the label map with the labels from sorted_all_pts
    for x, y, label in sorted_all_pts:
        label_map[int(y), int(x)] = label

    # img = cv2.imread(f'{save_path}/test_img.png')
    # # resized_img = cv2.resize(img, (1500, 1500))
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary_img = np.where(gray_img > 127, 1, 0).astype(np.uint8)    
    # label_map = np.array(binary_img)

    for cl in range(len(self.classes)):
        # Create a binary mask for the target label
        binary_mask = (label_map == cl).astype(np.uint8)

        # Find contours using OpenCV
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 2: Analyze connected regions with opposite labels inside each contour
        results = []
        
        # Collect results for this contour
        contours_data = []

        # Define the opposite label (e.g., if target_label is 1, opposite_label is 0)
        opposite_label = 1 - cl

        # Plot the binary mask and contours
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Contours for Class {cl}")
        plt.axis('off')
        plt.imsave(f"{save_path}/Metric/BinaryContour_Class{cl}.png", binary_mask, cmap='gray', format='png')
        plt.close()

        contour_idx = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_thresh:
                
                # Create a mask with the same size as the label_map
                mask = np.zeros_like(label_map, dtype=np.uint8)
                
                # Fill the contour area in the mask and set it to 1
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

                # Extract the region within the contour that has the opposite label
                opposite_region = np.where((mask == 1) & (label_map == opposite_label), 1, 0)
                
                # Perform connected components analysis on the opposite region
                num_connected_components, labels, stats, centroids = cv2.connectedComponentsWithStats(opposite_region.astype(np.uint8), connectivity=8)

                # Collect results for the contour
                for i in range(1, num_connected_components):  # Skip the background (label 0)
                    contours_data.append({
                        "contour_index": contour_idx,
                        "connected_region_index": i,
                        "connected_patch_size": stats[i, cv2.CC_STAT_AREA],
                        "connected_region_centroid_x": centroids[i][0],
                        "connected_region_centroid_y": centroids[i][1]
                    })
                contour_idx += 1
        
        connected_patch_sizes = [data["connected_patch_size"] for data in contours_data]

        plt.figure(figsize=(10, 6))
        if connected_patch_sizes:
            max_patch_size = max(connected_patch_sizes)
            plt.hist(connected_patch_sizes, bins=max_patch_size, color='blue', alpha=0.7, edgecolor='black')
        else:
            plt.hist(connected_patch_sizes, bins=10, color='blue', alpha=0.7, edgecolor='black')

        plt.title(f"Distribution of Connected Patch Sizes for Class {cl}", fontsize=14)
        plt.xlabel("Patch Size", fontsize=12)
        plt.ylabel("Number", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{save_path}/Metric/ConnectedPatchSize_Distribution_Class{cl}.png")

        # Save results for this class to a CSV
        pd.DataFrame(contours_data).to_csv(f"{save_path}/Metric/ContourAnalysis_Class{cl}.csv", index=False)
        print(f"Contour analysis results for class {cl} saved")

def contour_analysis_multi(self, gen):
    file_list_normal, file_list_hcc = [], []
    for wsi in self.hcc_wsis:
        _wsi = wsi+91 if (self.state == "new" and self.type == "HCC") else wsi
        save_path = f'{self.save_dir}/100WTC_Result/{_wsi}/trial_{self.num_trial}'

        if gen == 0:
            condition = f'{self.class_num}_class'
        else:
            condition = f"Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}"
        
        file_list_normal.append(f"{save_path}/Metric/ContourAnalysis_Class0.csv")
        file_list_hcc.append(f"{save_path}/Metric/ContourAnalysis_Class1.csv")

    
    def safe_read_csv(file_name):
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            try:
                return pd.read_csv(file_name)
            except pd.errors.EmptyDataError:
                print(f"File is empty: {file_name}")
                return pd.DataFrame()
        else:
            print(f"File not found or is empty: {file_name}")
            return pd.DataFrame()
    
    save_path = f'{self.save_dir}/100WTC_Result'
    all_sizes_normal, all_sizes_hcc = [], []
    # Read Normal category files
    for file_name in file_list_normal:
        data = safe_read_csv(file_name)
        if not data.empty and 'connected_patch_size' in data.columns:
            all_sizes_normal.extend(data['connected_patch_size'].tolist())

    # Read HCC category files
    for file_name in file_list_hcc:
        data = safe_read_csv(file_name)
        if not data.empty and 'connected_patch_size' in data.columns:
            all_sizes_hcc.extend(data['connected_patch_size'].tolist())
        
    if all_sizes_normal:
        all_sizes_normal = np.array(all_sizes_normal)

        sorted_sizes = np.sort(all_sizes_normal)
        cumulative_distribution = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_sizes, cumulative_distribution, marker='o', linestyle='-', color='b')
        plt.title("Cumulative Distribution of HCC Connected Component Sizes", fontsize=14)
        plt.xlabel("Connected Component Size", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{save_path}/cumulative_distribution_plot_normal.png")
        # plt.show()
    else:
        print("No data of normal available for plotting.")

    if all_sizes_hcc:
        all_sizes_hcc = np.array(all_sizes_hcc)

        sorted_sizes = np.sort(all_sizes_hcc)
        cumulative_distribution = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_sizes, cumulative_distribution, marker='o', linestyle='-', color='b')
        plt.title("Cumulative Distribution of Normal Connected Component Sizes", fontsize=14)
        plt.xlabel("Connected Component Size", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{save_path}/cumulative_distribution_plot_hcc.png")
        # plt.show()
    else:
        print("No data of hcc available for plotting.")