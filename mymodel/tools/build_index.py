import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ================= 配置 =================
JSON_PATH = 'mini_wlasl_dictionary.json'  # 上一步生成的字典文件
MODEL_NAME = 'all-mpnet-base-v2'            # SBERT 模型名称
OUTPUT_KEYS_PATH = 'index_keys.npy'        # 保存 Key 的列表
OUTPUT_VECS_PATH = 'index_vectors.npy'     # 保存 向量 的矩阵
# ========================================

def build_vector_index():
    print(f"1. Loading dictionary from {JSON_PATH}...")
    if not os.path.exists(JSON_PATH):
        print("Error: Dictionary file not found!")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        super_dict = json.load(f)
    
    # 提取所有的 Keys (即标准化的英文单词)
    # 比如 ['book', 'father', 'let me see', ...]
    keys_list = list(super_dict.keys())
    print(f"   Found {len(keys_list)} unique keys.")

    print(f"2. Loading SBERT model: {MODEL_NAME}...")
    # 第一次运行会自动下载模型到本地缓存
    model = SentenceTransformer(MODEL_NAME)

    print("3. Encoding keys into vectors (This might take a minute on CPU)...")
    # model.encode 会把文本列表转换成 numpy 矩阵
    # 结果是一个 [N, 384] 的矩阵，N是单词数，384是向量维度
    embeddings = model.encode(keys_list, convert_to_tensor=False, show_progress_bar=True)

    print("4. Saving index to disk...")
    # 我们保存两个文件：
    # 1. 文本列表 (和矩阵行号一一对应)
    np.save(OUTPUT_KEYS_PATH, np.array(keys_list))
    # 2. 向量矩阵
    np.save(OUTPUT_VECS_PATH, embeddings)

    print("\nDone! Index built successfully.")
    print(f"Keys saved to: {OUTPUT_KEYS_PATH}")
    print(f"Vectors saved to: {OUTPUT_VECS_PATH} (Shape: {embeddings.shape})")

if __name__ == "__main__":
    build_vector_index()