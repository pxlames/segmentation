文件路径
gt_masks_path = '/home/xkw/pxlames/gt_masks.npy'
scores_path = '/home/xkw/pxlames/scores.npy'
# 读取保存的 numpy 数组文件
gt_masks = np.load(gt_masks_path)  # 加载真实标签数据
scores = np.load(scores_path)      # 加载预测结果数据

# 修改 gt_masks 和 scores 的形状，将最后一个维度转到第三个维度
gt_masks = np.transpose(gt_masks, (1, 2, 0))
scores = np.transpose(scores, (1, 2, 0))
# print("gt_masks 和 scores 文件已成功加载。")
# print(f"gt_masks 的形状: {gt_masks.shape}")
# print(f"scores 的形状: {scores.shape}")

cal_idx = np.arange(0, 15)
val_idx = np.setdiff1d(np.arange(1, scores.shape[2]), cal_idx) # 最后剩下的用于校准集

# 提取校准集的 score 数据
cal_scores = scores[:, :, cal_idx]  # 索引从 0 开始，cal_idx 减去 1
cal_gt_masks = gt_masks[:, :, cal_idx]  # 索引从 0 开始，cal_idx 减去 1



val_scores = scores[:, :, val_idx]  # 索引从 0 开始，cal_idx 减去 1
val_gt_masks = gt_masks[:, :, val_idx]  # 索引从 0 开始，cal_idx 减去 1



# 假设 val_scores 是模型的输出，scores 为二维数组
scores = val_scores[:, :, 2]
masks = val_gt_masks[:, :, 2]





# 绘制 ground_truth
plt.figure(figsize=(8, 8))
plt.imshow(ground_truth, cmap='gray')
plt.title('血管分割标注图')
plt.axis('off')
plt.show()