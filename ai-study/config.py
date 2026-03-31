# ==================== 全局配置 ====================

# 训练序列长度
TIME_STEP_LIST = [24, 48, 60, 96, 128]
# 训练神经元数
HIDDEN_SIZE_LIST = [32, 64, 128]
# 
BATCH_SIZE_LIST = [64, 128, 256]
# 学习率
LR_LIST = [1e-3, 5e-4, 1e-4]
# 防过拟合
DROPOUT_LIST = [0.2, 0.3, 0.4]
# 训练轮次（过小：欠拟合，过大：过拟合）
EPOCHS_LIST = [20, 30, 50, 60, 80]

# 交易成本
# 点差
SPREAD = 0.02
# 滑点 0.01 ~ 0.03
SLIPPAGE = 0.02

# 特征列
FEATURES = [
    "open", "high", "low", "close", "volume",
    "ma20", "rsi", "macd", "atr", "vol_ma20", "vol_ratio"
]

# 预测目标列
TARGETS = ["next_o", "next_h", "next_l", "next_c"]