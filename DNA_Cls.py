import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.svm import NuSVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
import joblib
warnings.filterwarnings('ignore')

# ============================
# 1. Đọc dữ liệu từ FASTA
# ============================
def read_fasta(file_path, label):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        # Loại bỏ sequence có ký tự lạ (chỉ giữ A, T, C, G)
        if all(base in 'ATCG' for base in seq) and len(seq) > 0:
        # if len(seq) > 0:
            sequences.append(seq)
    df = pd.DataFrame({'sequence': sequences, 'label': label})
    return df

# Thay bằng đường dẫn file của bạn
diabetic_file = 'dataset/DMT2_1296.fasta'      # 1296 sequences
non_diabetic_file = 'dataset/NONDM.fasta'

df_diabetic = read_fasta(diabetic_file, 1)
df_non = read_fasta(non_diabetic_file, 0)

df = pd.concat([df_diabetic, df_non], ignore_index=True)
print(f"Tổng số mẫu: {len(df)} | Diabetic: {df['label'].sum()} | Non-diabetic: {len(df)-df['label'].sum()}")

# ============================
# 2. Feature Extraction: TF-IDF trên k-mers (3-6)
# ============================
vectorizer = TfidfVectorizer(
    analyzer='char',      # character level
    ngram_range=(3, 6),   # k-mers từ 3 đến 6
    min_df=2,
    max_df=0.9
)

X = vectorizer.fit_transform(df['sequence'])
y = df['label'].values

print(f"Kích thước feature matrix: {X.shape}")

# ============================
# 3. Train/Test split + SMOTE (chỉ trên train)
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

print(f"Sau SMOTE - Train: {X_train_res.shape[0]} mẫu")

# ============================
# 4. Model 1: NuSVC
# ============================
nusvc = NuSVC(
    nu=0.5,               # có thể tune bằng GridSearchCV
    kernel='rbf',
    gamma='scale',
    probability=True,     # cần cho log_loss và ROC-AUC
    random_state=42
)

nusvc.fit(X_train_res, y_train_res)
y_pred_nu = nusvc.predict(X_test)
y_prob_nu = nusvc.predict_proba(X_test)

# ============================
# 5. Model 2: XGBoost
# ============================
xgb = XGBClassifier(
    n_estimators=300,     # boosting rounds như bài báo
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)

# ============================
# 6. Đánh giá
# ============================
def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n=== {model_name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob[:, 1]):.4f}")
    print(f"Log Loss : {log_loss(y_true, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model(y_test, y_pred_nu, y_prob_nu, "NuSVC")
evaluate_model(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")


print("\n=== Đang lưu các model ===")

# Lưu NuSVC model
joblib.dump(nusvc, 'model/nusvc_diabetes_model.joblib')
print("✓ Đã lưu NuSVC model → nusvc_diabetes_model.joblib")

# Lưu XGBoost model (cách khuyến nghị)
xgb.save_model('model/xgb_diabetes_model.json')
print("✓ Đã lưu XGBoost model → xgb_diabetes_model.json")

# Lưu thêm TF-IDF Vectorizer (RẤT QUAN TRỌNG)
# Vì sau này dự đoán cần dùng cùng vectorizer để transform dữ liệu mới
joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')
print("✓ Đã lưu TF-IDF Vectorizer → tfidf_vectorizer.joblib")

print("\nĐã lưu xong 3 file:")
print("1. nusvc_diabetes_model.joblib")
print("2. xgb_diabetes_model.json")
print("3. tfidf_vectorizer.joblib")