import pandas as pd
from Bio import SeqIO

def read_fasta(file_path, label):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        # Loại bỏ sequence có ký tự lạ (chỉ giữ A, T, C, G)
        if all(base in 'ATCG' for base in seq) and len(seq) > 0:
            sequences.append(seq)
        else:
            print(seq)
            
    df = pd.DataFrame({'sequence': sequences, 'label': label})
    return df


diabetic_file = 'dataset/DMT2_1296.fasta'      # 1296 sequences
non_diabetic_file = 'dataset/NONDM.fasta'

df_diabetic = read_fasta(diabetic_file, 1)