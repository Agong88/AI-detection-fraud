import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             precision_recall_curve)

#1. 讀取&清理
df = pd.read_excel('jar.xlsx')
drop_cols = ['Transaction_ID', 'User_ID', 'Timestamp',
             'Risk_Score', 'Fraud_Probability']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df.drop(columns=['Fraud_Label'])
y = df['Fraud_Label']

#2. Pipeline
cat_cols = X.select_dtypes(include='object').columns.tolist()
preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

# 設定弱學習器為淺層決策樹以避免過擬合
base_tree = DecisionTreeClassifier(
    max_depth=2,
    min_samples_leaf=7,
    random_state=42
)

# 建立 AdaBoost 分類器，使用 base_tree 作為基學習器
ada_clf = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=400,
    learning_rate=0.05,
    random_state=42
)

# 將前處理和分類器組合成管線
model = Pipeline([
    ('prep', preprocess),
    ('clf',  ada_clf)
])

#3. 切分訓練/測試集
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

#4. 掃描不同門檻值下的 Precision/Recall
print('\n=== 門檻掃描 (調整分類門檻) ===')
for thr in [0.50, 0.45, 0.40, 0.35, 0.30]:
    y_hat = (y_prob >= thr).astype(int)
    prec  = precision_score(y_test, y_hat)
    rec   = recall_score(y_test, y_hat)
    f1    = f1_score(y_test, y_hat)
    print(f"thr={thr:.2f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

#5. 設定最終決策門檻值
THRESHOLD = 0.3125 

y_pred_fixed = (y_prob >= THRESHOLD).astype(int)
print(f'\n=== 門檻 {THRESHOLD:.2f} 的模型效能 ===')
print(f'準確率 (Accuracy) : {accuracy_score(y_test, y_pred_fixed):.4f}')
print(f'精確率 (Precision): {precision_score(y_test, y_pred_fixed):.4f}')
print(f'召回率 (Recall)   : {recall_score(y_test, y_pred_fixed):.4f}')
print(f'F1-分數 (F1-score) : {f1_score(y_test, y_pred_fixed):.4f}')
print('混淆矩陣 (Confusion Matrix):\n', confusion_matrix(y_test, y_pred_fixed))

# 6. 互動式單筆預測
input_cols = [
    ('Transaction_Type', '交易類型'),
    ('Device_Type', '使用裝置類型'),
    ('Location', '交易地點'),
    ('IP_Address_Flag', 'IP 位址是否可疑'),
    ('Authentication_Method', '認證方式')
]
device_type_map = {1:'Laptop', 2:'Mobile', 3:'Tablet'}
cat_options = {col: sorted(df[col].unique()) for col in cat_cols}
dtypes = X.dtypes

print('\n👉 請依序輸入以下欄位資料進行單筆預測：')
manual = {}
for col, cname in input_cols:
    if col == 'Device_Type':
        for k,v in device_type_map.items():
            print(f'  {k}: {v}')
        manual[col] = int(input('Device_Type (1-3): '))
    elif col in cat_cols:
        opts = cat_options[col]
        for i, v in enumerate(opts):
            print(f'  {i}: {v}')
        manual[col] = opts[int(input(f'{cname} (輸入編號 0-{len(opts)-1}): '))]
    else:
        manual[col] = int(input(f'{cname} (0 或 1): '))

# 對於其餘模型需要的欄位，用訓練集中最常見值或平均值填補
for col in X.columns:
    if col not in manual:
        manual[col] = df[col].mode()[0] if col in cat_cols else df[col].mean()
manual_df = pd.DataFrame([manual]).astype(dtypes)

# 產生預測
prob = model.predict_proba(manual_df)[0, 1]
pred = int(prob >= THRESHOLD)
print('\n你的輸入資料之預測結果：')
print(f'詐欺機率 = {prob:.3f}  (決策門檻 = {THRESHOLD})')
print('🚨 判斷為「詐欺交易」 🚨' if pred else '✅ 判斷為「正常交易」 ✅')