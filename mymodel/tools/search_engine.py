import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

class MotionRetriever:
    def __init__(self, keys_path='index_keys.npy', vecs_path='index_vectors.npy', model_name='all-mpnet-base-v2'):
        print("Initializing Motion Retriever...")
        
        # 1. 加载模型 (用于把用户输入的 Query 转成向量)
        self.model = SentenceTransformer(model_name)
        
        # 2. 加载预先计算好的索引
        if not os.path.exists(keys_path) or not os.path.exists(vecs_path):
            raise FileNotFoundError("Index files not found! Run build_index.py first.")
            
        self.keys = np.load(keys_path, allow_pickle=True)
        self.vectors = np.load(vecs_path)
        print(f"Index loaded. Vocabulary size: {len(self.keys)}")

    def search(self, query_text, top_k=3, threshold=0.6):
        """
        核心检索函数
        :param query_text: 用户输入的单词 (如 "dad")
        :param top_k: 返回前 K 个候选项
        :param threshold: 相似度阈值 (0~1)，低于此值认为没找到
        :return: 结果列表 [{'key': 'father', 'score': 0.85}, ...]
        """
        # 1. 把查询词转向量
        query_vec = self.model.encode(query_text, convert_to_tensor=True)
        
        # 2. 计算余弦相似度 (Cosine Similarity)
        # util.cos_sim 极其高效，能瞬间计算 query 和 5000 个 keys 的距离
        # self.vectors 需要先转成 tensor 才能和 query_vec 计算
        scores = util.cos_sim(query_vec, self.vectors)[0]

        # 3. 找到分数最高的 top_k 个索引
        # torch.topk 返回 (values, indices)
        top_results = torch_topk(scores, k=top_k) # 也可以用 numpy argpartition，但 SBERT 返回的是 tensor

        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            score = float(score)
            if score < threshold:
                continue # 低于阈值直接跳过
            
            match_key = self.keys[idx.item()]
            results.append({
                'query': query_text,
                'match_key': match_key,
                'score': round(score, 4)
            })
            
        return results

# 辅助函数：为了不引入 torch 依赖给主逻辑增加负担，这里用 util.cos_sim 自带的逻辑
# 但为了演示方便，这里简单写一个 numpy 版本的 search，不需要 torch
class SimpleMotionRetriever:
    def __init__(self, keys_path='index_keys.npy', vecs_path='index_vectors.npy', model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.keys = np.load(keys_path, allow_pickle=True)
        self.vectors = np.load(vecs_path)
        # 归一化向量，这样 点积(Dot Product) 就等于 余弦相似度
        # axis=1 表示按行归一化
        norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norm + 1e-10)

    def search(self, query_text, top_k=3, threshold=0.45): # 阈值通常 0.5-0.6 比较安全
        query_vec = self.model.encode(query_text)
        # 归一化 query
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # 矩阵乘法计算相似度 (1 x 384) dot (384 x N) -> (1 x N)
        scores = np.dot(query_vec, self.vectors.T)
        
        # 获取 top_k 索引
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            match_key = self.keys[idx]
            
            # Debug 打印：看看前几名到底是谁，多少分
            # print(f"[DEBUG] Query: {query_text} | Candidate: {match_key} | Score: {score:.4f}")

            if score < threshold:
                continue 
            
            results.append({
                'match_key': match_key,
                'score': round(score, 4)
            })
            
        if not results:
             # 如果失败了，返回 UNK，但也把最高分的那个“失败者”带上，方便分析
             best_guess_idx = top_indices[0]
             return [{
                 "match_key": "[UNK]", 
                 "score": 0.0, 
                 "debug_best_guess": self.keys[best_guess_idx],
                 "debug_best_score": float(scores[best_guess_idx])
             }]

        return results

# ================= 测试代码 =================
if __name__ == "__main__":
    retriever = SimpleMotionRetriever()
    
    test_queries = [
        "dad",          # 应该匹配 "father"
        "papa",         # 应该匹配 "father"
        "automobile",   # 应该匹配 "car" (如果字典里有)
        "let me see",   # 精确匹配
        "let's see",    # 模糊匹配
        "Kalin",        # 生僻词 (应该找不到或分很低)
        "angry",        # 应该匹配 "angry"
        "furious",       # 应该匹配 "angry" 或 "mad"
        "sing",
        "Sing",
        "a"
    ]
    
    print("-" * 30)
    for q in test_queries:
        res = retriever.search(q, top_k=1, threshold=0.6)
        print(f"Query: '{q}' \t-> Found: {res}")
    print("-" * 30)