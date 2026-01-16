import streamlit as st
import google.generativeai as genai
import os

# APIの設定
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# もしこれでも404が出るなら、この1行を試してください（通常は不要ですが念のため）
# os.environ["GOOGLE_API_VERSION"] = "v1" 

model = genai.GenerativeModel('gemini-3-flash-preview')

# 2. 画面のデザイン
st.set_page_config(page_title="AI料理レシピ生成", page_icon="🍳")
st.title("🍳 AI残り物レシピメーカー")
st.caption("冷蔵庫にあるものを入力して、今日のご飯を決めよう！")

# 3. 入力フォーム
with st.sidebar:
    st.header("設定")
    mode = st.selectbox("料理のジャンル", ["和食", "洋食", "中華", "エスニック", "スイーツ"])
    diet = st.checkbox("ヘルシー志向（低カロリー）")

ingredients = st.text_area("食材を入力してください（例：鶏肉、なす、ポン酢）", placeholder="カンマ区切りで入力...")

# 4. 生成ロジック
if st.button("レシピを提案してもらう"):
    if not ingredients:
        st.warning("食材を入力してください。")
    else:
        with st.spinner("AIが美味しいレシピを考えています..."):
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
                response = model.generate_content(prompt)
                st.markdown("### 👨‍🍳 AIのおすすめレシピ")
                st.write(response.text)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")


st.markdown("---")
st.caption("Powered by Google Gemini API & Streamlit")