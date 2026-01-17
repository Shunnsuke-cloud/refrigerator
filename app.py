import streamlit as st
import google.generativeai as genai
import os
import pickle
import csv
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# ========================
# 設定ファイル経路
# ========================
TRAINING_DATA_FILE = "recipe_training_data.csv"
MODEL_FILE = "cooking_time_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# ========================
# 1. APIの設定
# ========================
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ========================
# 2. 機械学習ユーティリティ関数
# ========================

def load_training_data():
    """CSVから学習データを読み込む"""
    if not Path(TRAINING_DATA_FILE).exists():
        return [], []
    
    recipe_names = []
    cooking_times = []
    try:
        with open(TRAINING_DATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and 'recipe_name' in row and 'cooking_time' in row:
                    recipe_names.append(row['recipe_name'])
                    cooking_times.append(float(row['cooking_time']))
    except Exception as e:
        st.warning(f"学習データの読み込みエラー: {e}")
    
    return recipe_names, cooking_times

def save_training_data(recipe_name, cooking_time):
    """新しいデータをCSVに追加"""
    try:
        file_exists = Path(TRAINING_DATA_FILE).exists()
        with open(TRAINING_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['recipe_name', 'cooking_time'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'recipe_name': recipe_name, 'cooking_time': cooking_time})
    except Exception as e:
        st.error(f"データ保存エラー: {e}")

def train_model():
    """モデルを訓練して保存"""
    recipe_names, cooking_times = load_training_data()
    
    # データが不足している場合はスキップ
    if len(recipe_names) < 2:
        return None, None
    
    try:
        # TF-IDFベクトライザーの作成と訓練
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=100)
        X = vectorizer.fit_transform(recipe_names)
        y = np.array(cooking_times)
        
        # 線形回帰モデルの訓練
        ml_model = LinearRegression()
        ml_model.fit(X, y)
        
        # モデルとベクトライザーを保存
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(ml_model, f)
        with open(VECTORIZER_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return ml_model, vectorizer
    except Exception as e:
        st.error(f"モデル訓練エラー: {e}")
        return None, None

def load_model():
    """保存されたモデルとベクトライザーを読み込む"""
    model_exists = Path(MODEL_FILE).exists()
    vectorizer_exists = Path(VECTORIZER_FILE).exists()
    
    if not (model_exists and vectorizer_exists):
        return None, None
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            ml_model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        return ml_model, vectorizer
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None, None

def predict_cooking_time(recipe_name):
    """料理名から調理時間を予測"""
    ml_model, vectorizer = load_model()
    
    if ml_model is None or vectorizer is None:
        return None
    
    try:
        X = vectorizer.transform([recipe_name])
        predicted_time = ml_model.predict(X)[0]
        # 予測時間を整数で返す（最小1分）
        return max(int(round(predicted_time)), 1)
    except Exception as e:
        st.error(f"予測エラー: {e}")
        return None

def extract_recipe_name(recipe_text):
    """レシピテキストから料理名を抽出"""
    lines = recipe_text.strip().split('\n')
    for line in lines:
        # 「- 料理名」形式を探す
        if '料理名' in line:
            # コロンまたはその他の区切り文字の後の内容を抽出
            match = re.search(r'料理名[：:]\s*(.+?)(?:\n|$)', line)
            if match:
                name = match.group(1).strip()
                return name
        # 最初の行が料理名の場合
        if line.strip() and not line.startswith('- ') and not line.startswith('【'):
            return line.strip()
    
    return "不明な料理"

# ========================
# 3. UI の設計
# ========================
st.set_page_config(page_title="AI料理レシピ生成", page_icon="🍳", layout="wide")
st.title("🍳 AI残り物レシピメーカー + 調理時間予測")
st.caption("冷蔵庫にあるものを入力して、今日のご飯を決めよう！調理時間も予測します。")

# ========================
# 4. サイドバー - 設定
# ========================
with st.sidebar:
    st.header("⚙️ 設定")
    mode = st.selectbox("料理のジャンル", ["和食", "洋食", "中華", "エスニック", "スイーツ"])
    diet = st.checkbox("ヘルシー志向（低カロリー）")
    
    st.divider()
    st.header("📊 学習データ状況")
    recipe_names, cooking_times = load_training_data()
    st.metric("学習済みレシピ数", len(recipe_names))
    
    if len(recipe_names) > 0:
        avg_time = np.mean(cooking_times)
        st.metric("平均調理時間", f"{avg_time:.0f}分")

# ========================
# 5. メイン画面 - 入力フォーム
# ========================
st.header("📝 食材を入力")
ingredients = st.text_area(
    "食材を入力してください（例：鶏肉、なす、ポン酢）",
    placeholder="カンマ区切りで入力..."
)

# ========================
# 6. レシピ生成ロジック
# ========================
if st.button("🎯 レシピを提案してもらう", use_container_width=True):
    if not ingredients:
        st.warning("⚠️ 食材を入力してください。")
    else:
        with st.spinner("✨ AIが美味しいレシピを考えています..."):
            # AIへの詳細な指示
            prompt = f"""
            以下の条件で料理のレシピを1つ提案してください。
            食材: {ingredients}
            ジャンル: {mode}
            ヘルシー優先: {"はい" if diet else "いいえ"}
            
            出力形式：
            - 料理名
            - 調理時間
            - 材料
            - 手順（箇条書き）
            - AIのおすすめポイント
            """
            
            try:
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
                recipe_text = response.text
                
                # レシピを表示
                st.markdown("### 👨‍🍳 AIのおすすめレシピ")
                st.write(recipe_text)
                
                # 料理名を抽出
                extracted_recipe_name = extract_recipe_name(recipe_text)
                
                # レイアウト：2列に分割
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("---")
                    st.markdown("### ⏱️ 調理時間予測")
                    
                    # 予測時間を計算
                    predicted_time = predict_cooking_time(extracted_recipe_name)
                    
                    if predicted_time is not None:
                        st.success(f"📌 予測調理時間: **{predicted_time}分**")
                        st.info(f"料理名: {extracted_recipe_name}")
                    else:
                        st.info("🤔 学習データが不足しています。実際の調理時間を記録してください。")
                        st.info(f"料理名: {extracted_recipe_name}")
                
                with col2:
                    st.markdown("---")
                    st.markdown("### 📋 調理時間を記録")
                    
                    # 実際の調理時間を入力
                    actual_time = st.number_input(
                        "実際にかかった時間（分）",
                        min_value=1,
                        max_value=240,
                        value=predicted_time if predicted_time is not None else 30,
                        step=1
                    )
                    
                    # 記録ボタン
                    if st.button("💾 調理時間を記録する", use_container_width=True):
                        save_training_data(extracted_recipe_name, actual_time)
                        st.success(f"✅ '{extracted_recipe_name}' の調理時間 {actual_time}分を記録しました！")
                        
                        # モデルを再訓練
                        with st.spinner("🔄 モデルを再訓練中..."):
                            train_model()
                            st.success("🎓 モデルの再訓練が完了しました！")
                        
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")

# ========================
# 7. フッター
# ========================
st.markdown("---")
st.caption("Powered by Google Gemini API + scikit-learn & Streamlit")