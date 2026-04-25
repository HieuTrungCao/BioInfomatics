import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.svm import NuSVC
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================
# 1. DNABERT Config & Encoder
# ============================
class DNABERTEncoder:
    def __init__(self, model_name='zhihan1996/DNA_bert_6', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _seq_to_kmers(self, seq, k=6):
        """DNABERT yêu cầu input là các k-mer cách nhau bởi dấu cách"""
        return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

    def encode(self, sequences, batch_size=16):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i : i + batch_size]
                # Format sequences sang k-mers
                kmers_list = [self._seq_to_kmers(s) for s in batch_seqs]
                
                inputs = self.tokenizer(kmers_list, return_tensors='pt', 
                                         padding=True, truncation=True, 
                                         max_length=512).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Lấy vector [CLS] (đại diện cho toàn bộ chuỗi)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
                
                if i % 100 == 0:
                    print(f"Đã encode {i}/{len(sequences)} chuỗi...")
                    
        return np.vstack(embeddings)

# ============================
# 2. Đọc dữ liệu (Giữ nguyên)
# ============================
def read_fasta(file_path, label):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        if all(base in 'ATCG' for base in seq) and len(seq) > 0:
            sequences.append(seq)
    return pd.DataFrame({'sequence': sequences, 'label': label})

# Load data
df_diabetic = read_fasta('dataset/DMT2_1296.fasta', 1)
df_non = read_fasta('dataset/NONDM.fasta', 0)
df = pd.concat([df_diabetic, df_non], ignore_index=True)

# ============================
# 3. Trích xuất đặc trưng bằng DNABERT
# ============================
print("--- Bắt đầu trích xuất đặc trưng bằng DNABERT ---")
encoder = DNABERTEncoder()
# Lưu ý: DNABERT tốn tài nguyên, nếu RAM ít bạn nên test với df.head(100) trước
X = encoder.encode(df['sequence'].tolist())
y = df['label'].values
print(f"Kích thước feature matrix (DNABERT): {X.shape}")

# ============================
# 4. Train/Test split + ADASYN
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

# ============================
# 5. Huấn luyện (Giữ nguyên tham số tối ưu của bạn)
# ============================
# NuSVC
nusvc = NuSVC(nu=0.5, kernel='rbf', probability=True, random_state=42)
nusvc.fit(X_train_res, y_train_res)

# XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, 
                    random_state=42, eval_metric='logloss')
xgb.fit(X_train_res, y_train_res)

# ============================
# 6. Đánh giá & Lưu trữ
# ============================
def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n=== {model_name} (DNABERT Features) ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob[:, 1]):.4f}")

evaluate_model(y_test, nusvc.predict(X_test), nusvc.predict_proba(X_test), "NuSVC")
evaluate_model(y_test, xgb.predict(X_test), xgb.predict_proba(X_test), "XGBoost")

# Lưu trữ
joblib.dump(nusvc, 'model/nusvc_dnabert.joblib')
xgb.save_model('model/xgb_dnabert.json')
# Lưu ý: Không cần lưu vectorizer như TF-IDF, vì encoder dùng model pre-trained