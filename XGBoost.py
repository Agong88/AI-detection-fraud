import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             precision_recall_curve)
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取 & 清理資料
df = pd.read_excel('jar.xlsx')

drop_cols = ['Transaction_ID', 'User_ID', 'Timestamp',
             'Risk_Score', 'Fraud_Probability']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df.drop(columns=['Fraud_Label'])
y = df['Fraud_Label']

# 2. 前處理與XGBoost模型
cat_cols = X.select_dtypes(include='object').columns.tolist()
preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

# 防止過擬合的 XGBoost 參數設定
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=1,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ('prep', preprocess),
    ('clf',  xgb_clf)
])

# 3. 切分資料 → 訓練 / 測試
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
model.fit(X_train, y_train)

# 4. 測試集預測 & 門檻掃描
y_prob = model.predict_proba(X_test)[:, 1]

print('\n=== 不同決策門檻下 Precision / Recall ===')
thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]
for thr in thresholds:
    y_pred = (y_prob >= thr).astype(int)
    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print(f'閾值 {thr:.2f} → Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}')

# 5. 選定最終門檻並評估
THRESHOLD = 0.15
y_pred_final = (y_prob >= THRESHOLD).astype(int)

print(f'\n=== 使用閾值 {THRESHOLD:.2f} 的最終效能 ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_final):.4f}')
print(f'Precision: {precision_score(y_test, y_pred_final):.4f}')
print(f'Recall   : {recall_score(y_test, y_pred_final):.4f}')
print(f'F1-score : {f1_score(y_test, y_pred_final):.4f}')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_final))


# 6. 互動式單筆預測
input_cols = [
    ('Transaction_Type', '交易類型'),
    ('Device_Type', '使用裝置類型'),
    ('Location', '交易地點'),
    ('IP_Address_Flag', 'IP 位址是否可疑 (0/1)'),
    ('Authentication_Method', '認證方式')
]

device_type_map = {1: 'Laptop', 2: 'Mobile', 3: 'Tablet'}
cat_options = {col: sorted(df[col].unique()) for col in cat_cols}
dtypes = X.dtypes  # 保留欄位型別

print('\n👉 請依序輸入欄位進行單筆預測：')
manual = {}
for col, cname in input_cols:
    if col == 'Device_Type':
        for k, v in device_type_map.items():
            print(f'  {k}: {v}')
        manual[col] = device_type_map[int(input('Device_Type (輸入 1-3): '))]
    elif col in cat_cols:
        opts = cat_options[col]
        for i, v in enumerate(opts):
            print(f'  {i}: {v}')
        idx = int(input(f'{cname} (輸入 0-{len(opts)-1}): '))
        manual[col] = opts[idx]
    else:
        manual[col] = int(input(f'{cname}: '))

# 填補未輸入欄位 (眾數 / 平均值)
for col in X.columns:
    if col not in manual:
        manual[col] = df[col].mode()[0] if col in cat_cols else df[col].mean()

manual_df = pd.DataFrame([manual]).astype(dtypes)

prob = model.predict_proba(manual_df)[0, 1]
pred = int(prob >= THRESHOLD)
print('\n=== 單筆交易預測結果 ===')
print(f'詐欺機率 = {prob:.3f} (閾值 {THRESHOLD})')
print('🚨 判定為「詐欺交易」 🚨' if pred else '✅ 判定為「正常交易」 ✅')