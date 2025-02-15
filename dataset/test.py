import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

print("debug")
print(torch.cuda.is_available())

train_loss_list = [0.194108502,0.12833176,0.109597589,0.09775072,0.085420897,0.077891686,0.069493554,0.062969841,0.05655]
valid_loss_list = [0.454657939,0.173584941,0.113100196,0.091761335,0.088030437,0.108318072,0.065371412,0.806971064,0.44929]
train_acc_list = [0.928885771,0.954655011,0.961660911,0.965761925,0.970105763,0.972920714,0.975609756,0.977849126,0.98038]
valid_acc_list = [0.807609321,0.935270426,0.960802647,0.969073648,0.970943613,0.964470656,0.977056962,0.735327963,0.84717]

# 轉成 DataFrame
df = pd.DataFrame({
    "train_loss": train_loss_list,
    "valid_loss": valid_loss_list,
    "train_acc": train_acc_list,
    "valid_acc": valid_acc_list
})

# 定義輸出檔案路徑
csv_path = "/workspace/graduate_project/dataset/epoch_log.csv"

# 存成 CSV，不寫入 index
# df.to_csv(csv_path, index=False)

# plt.figure(figsize=(8, 6))
epochs = range(1, len(train_loss_list) + 1)

# plt.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
# plt.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="-", color="red")

# # 圖表設定
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Train vs Valid Loss")
# plt.legend()
# plt.grid(True)

# 創建圖表
fig, ax1 = plt.subplots(figsize=(8, 6))

# 繪製 loss（左側 Y 軸）
l1 = ax1.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
l2 = ax1.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="--", color="red")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.tick_params(axis="y", labelcolor="blue")
# ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.5)

# 創建第二個 Y 軸（右側）來顯示 accuracy
ax2 = ax1.twinx()
l3 = ax2.plot(epochs, train_acc_list, label="Train Accuracy", marker="^", linestyle="-", color="green")
l4 = ax2.plot(epochs, valid_acc_list, label="Valid Accuracy", marker="D", linestyle="--", color="orange")
ax2.set_ylabel("Accuracy")
ax2.tick_params(axis="y", labelcolor="green")
# ax2.legend(loc="lower right")

lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# 設定標題
plt.title("Train vs Valid Loss & Accuracy")


# 顯示圖表
plt.savefig("loss_and_accuracy_curve.png", dpi=300, bbox_inches="tight")


            # train_loss_arr = np.array(train_loss_list).reshape(1,len(train_loss_list))
            # valid_loss_arr = np.array(valid_loss_list).reshape(1,len(train_loss_list))
            # train_acc_arr = np.array(train_acc_list).reshape(1,len(train_loss_list))
            # valid_acc_arr = np.array(valid_acc_list).reshape(1,len(train_loss_list))

            # training_log = np.append(train_loss_arr,valid_loss_arr,axis=0)
            # training_log = np.append(training_log,train_acc_arr,axis=0)
            # training_log = np.append(training_log,valid_acc_arr,axis=0)
            
            # np.savetxt(f"{loss_save_path}/{condition}_epoch_log.csv", header = "train_loss,valid_loss,train_acc,valid_acc", delimiter=",",comments="")