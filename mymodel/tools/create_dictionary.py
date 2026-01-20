import json
import csv
import re
import os
from collections import defaultdict

# =================配置路径=================
# 请将这里改为你实际的文件路径
WLASL_PATH = 'WLASL_v0.3.json'  # WLASL 主文件
ASL_LEX_PATH = 'ASL-LEX_View_Data.csv'    # ASL-Lex 词典定义文件
MSASL_SYN_PATH = 'MSASL_synonym.json' # 同义词表

# ASLcitizen 的三个分割文件
ASL_CITIZEN_FILES = [
    'train.csv', 
    'val.csv', 
    'test.csv'
]


def normalize_text(text):
    if not text: return ""
    text = text.lower().strip()
    text = re.sub(r'\s*\(\d+\)', '', text) # 去掉 (1)
    text = re.sub(r'_\d+$', '', text)      # 去掉 _1
    
    # === 新增：把连字符替换为空格 ===
    text = text.replace('-', ' ') 
    text = text.replace('_', ' ') 
    # ============================
    
    # 把多余的空格缩减为一个 "let    me" -> "let me"
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def build_synonym_map(json_path):
    syn_map = {}
    if not os.path.exists(json_path):
        return syn_map
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for group in data:
        # 选取第一个作为标准词，并清洗
        canonical = normalize_text(group[0])
        for word in group:
            clean_word = normalize_text(word)
            if clean_word and clean_word != canonical:
                syn_map[clean_word] = canonical
    return syn_map

def build_asl_lex_map(csv_path):
    code_to_word = {}
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get('Code (survey)')
            raw_word = row.get('Entry ID')
            if code and raw_word:
                code_to_word[code] = normalize_text(raw_word)
    return code_to_word

def main():
    synonym_map = build_synonym_map(MSASL_SYN_PATH)
    lex_code_map = build_asl_lex_map(ASL_LEX_PATH)
    super_dict = defaultdict(list)
    
    # --- 处理 ASLcitizen ---
    print("Processing ASLcitizen...")
    for filename in ASL_CITIZEN_FILES:
        if not os.path.exists(filename): continue
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get('ASL-LEX Code')
                video_file = row.get('Video file')
                if code in lex_code_map:
                    word = lex_code_map[code]
                    if word in synonym_map: word = synonym_map[word]
                    
                    entry = {
                        "dataset": "ASLcitizen",
                        "video_file": video_file,
                        "code": code
                    }
                    super_dict[word].append(entry)

    # --- 处理 WLASL ---
    print("Processing WLASL...")
    if os.path.exists(WLASL_PATH):
        with open(WLASL_PATH, 'r', encoding='utf-8') as f:
            wlasl_data = json.load(f)
        for item in wlasl_data:
            word = normalize_text(item.get('gloss'))
            if word in synonym_map: word = synonym_map[word]
            
            for inst in item.get('instances', []):
                entry = {
                    "dataset": "WLASL",
                    "video_id": inst.get('video_id'),
                    "frame_start": inst.get('frame_start'),
                    "frame_end": inst.get('frame_end')
                }
                super_dict[word].append(entry)

    # --- 输出统计与文件 ---
    sorted_keys = sorted(super_dict.keys())
    print(f"\nFinal Statistics: {len(sorted_keys)} unique keys.")

    # 1. 保存完整索引 (JSON)
    with open('super_dictionary_index.json', 'w', encoding='utf-8') as f:
        json.dump(super_dict, f, indent=2)
    print("Saved: super_dictionary_index.json (Full Data)")

    # 2. 【新功能】保存纯单词列表 (TXT) - 供肉眼检查
    with open('vocabulary_list.txt', 'w', encoding='utf-8') as f:
        for key in sorted_keys:
            f.write(f"{key}\n")
    print("Saved: vocabulary_list.txt (Check this file for formatting!)")

    # 3. 打印几个样例检查格式
    print("\n--- Sample Keys Check ---")
    check_list = ['let-me-see', '1-dollar', 'book', 'father', 'dad']
    for k in check_list:
        if k in super_dict:
            print(f"'{k}': Found {len(super_dict[k])} clips")
        else:
            print(f"'{k}': Not found (Correct if it was merged)")

if __name__ == "__main__":
    main()

# =========================================

# def normalize_text(text):
#     """
#     清洗逻辑 V2:
#     1. 转小写
#     2. 去除括号变体
#     3. 去除结尾的 _1, _2 数字后缀
#     4. 【修改】将原有的下划线 _ 和空格 替换为连字符 -
#     """
#     if not text:
#         return ""
    
#     text = text.lower().strip()
    
#     # 去除括号及内容 "book (1)" -> "book"
#     text = re.sub(r'\s*\(\d+\)', '', text)
    
#     # 去除结尾的数字后缀 "about_1" -> "about"
#     text = re.sub(r'_\d+$', '', text)
    
#     # 【核心修改】将所有空格和剩余的下划线替换为连字符 -
#     # 先把下划线转空格，处理掉多余空格，再统一转连字符
#     text = text.replace('_', ' ')
#     # Split 会自动处理多个连续空格，join 保证只有一个连字符
#     text = "-".join(text.split())
    
#     return text