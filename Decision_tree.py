import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
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
    [('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

tree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=9,
    ccp_alpha=0.0,
    class_weight={0:1, 1:1},
    random_state=42
)

model = Pipeline([('prep', preprocess),
                  ('clf',  tree)])

# 3. 訓練 / 測試
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

#5. 選定門檻
THRESHOLD = 0.16

# 性能摘要
y_pred_fixed = (y_prob >= THRESHOLD).astype(int)
print('\n=== 門檻 %.2f 的效能 ===' % THRESHOLD)
print(f'準確率 (Accuracy) : {accuracy_score(y_test, y_pred_fixed):.4f}')
print(f'精確率 (Precision): {precision_score(y_test, y_pred_fixed):.4f}')
print(f'召回率 (Recall)   : {recall_score(y_test, y_pred_fixed):.4f}')
print(f'F1-分數 (F1-score) : {f1_score(y_test, y_pred_fixed):.4f}')
print('混淆矩陣 (Confusion Matrix):\n', confusion_matrix(y_test, y_pred_fixed))

#6. 互動預測
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

print('\n👉 請開始輸入指定欄位資料：')
manual = {}

for col, cname in input_cols:
    if col == 'Device_Type':
        for k,v in device_type_map.items(): print(f'  {k}: {v}')
        manual[col] = int(input('Device_Type (1-3): '))
    elif col in cat_cols:
        opts = cat_options[col]
        for i,v in enumerate(opts): print(f'  {i}: {v}')
        manual[col] = opts[int(input(f'{col} (0-{len(opts)-1}): '))]
    else:
        manual[col] = int(input(f'{col} (0/1): '))

for col in X.columns:
    if col not in manual:
        manual[col] = df[col].mode()[0] if col in cat_cols else df[col].mean()
manual_df = pd.DataFrame([manual]).astype(dtypes)

prob = model.predict_proba(manual_df)[0,1]
pred = int(prob >= THRESHOLD)

print('\n你的輸入預測結果：')
print(f'機率 = {prob:.3f}  (門檻 {THRESHOLD})')
print('🚨 詐騙交易 🚨' if pred else '✅ 正常交易 ✅')
