import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# --- ブラウザタブの設定とタイトルへのアイコン追加 ---
st.set_page_config(page_title="きゅうり合計収穫量予測アプリ", page_icon="🥒")

# --- カスタムCSSで背景画像とデザインを追加 ---
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1542478958-f58c704285b0?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

.input-box {
    background-color: rgba(255, 255, 255, 0.85); /* 半透明の白 */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.2);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("きゅうり合計収穫量予測アプリ 🥒")

# --- セッションステートの初期化 ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# --- モデル構築時に計算された値 ---
feature_names = [
    '日射量_現在_mean', '日射量_現在_std', '日射量_Max_mean', '日射量_Min_mean',
    '気温_現在_mean', '気温_現在_std', '気温_Max_mean', '気温_Min_mean',
    '湿度_現在_mean', '湿度_現在_std', '湿度_Max_mean', '湿度_Min_mean',
    'CO2濃度_現在_mean', 'CO2濃度_現在_std', 'CO2濃度_Max_mean', 'CO2濃度_Min_mean',
    '積算日射量_現在_mean', '積算日射量_現在_std', '積算日射量_Max_mean', '積算日射量_Min_mean'
]

# 訓練データから計算された各特徴量の平均値
feature_means_dict = {
    '日射量_現在_mean': 0.1651,
    '日射量_現在_std': 0.1989,
    '日射量_Max_mean': 0.1934,
    '日射量_Min_mean': 0.1418,
    '気温_現在_mean': 22.9567,
    '気温_現在_std': 5.8672,
    '気温_Max_mean': 23.3642,
    '気温_Min_mean': 22.5699,
    '湿度_現在_mean': 88.0837,
    '湿度_現在_std': 14.1522,
    '湿度_Max_mean': 89.8457,
    '湿度_Min_mean': 86.2996,
    'CO2濃度_現在_mean': 379.7915,
    'CO2濃度_現在_std': 32.5539,
    'CO2濃度_Max_mean': 387.8929,
    'CO2濃度_Min_mean': 372.4839,
    '積算日射量_現在_mean': 14948.3364,
    '積算日射量_現在_std': 2999.0494,
    '積算日射量_Max_mean': 15003.5516,
    '積算日射量_Min_mean': 14890.9634
}

# --- モデルのロードまたはダミーモデルの作成 ---
try:
    model = joblib.load('trained_gradient_boosting_model.pkl')
    st.info("モデルファイル 'trained_gradient_boosting_model.pkl' をロードしました。")
except FileNotFoundError:
    st.warning("モデルファイルが見つかりませんでした。代わりにダミーモデルを使用します。")
    class DummyModel:
        def __init__(self):
            self.n_features_in_ = len(feature_names)
        def predict(self, X):
            return np.array([25.0]) 
    model = DummyModel()

# --- 予測関数 ---
def predict_total_cucumber_yield_flexible(input_dict, trained_model, feature_means_dict):
    processed_input = []
    
    # ユーザーが入力した値と、定義済みの平均値を使って予測に必要なデータを作成
    for feature in feature_names:
        base_feature_name = '_'.join(feature.split('_')[:-1])
        if base_feature_name in input_dict and input_dict[base_feature_name] is not None:
            processed_input.append(input_dict[base_feature_name])
        elif feature in feature_means_dict:
            processed_input.append(feature_means_dict[feature])
        else:
            raise ValueError(f"特徴量 '{feature}' の平均値が定義されていません。")

    input_features = np.array(processed_input).reshape(1, -1)
    
    if not isinstance(trained_model, GradientBoostingRegressor) and hasattr(trained_model, 'n_features_in_'):
        if input_features.shape[1] != trained_model.n_features_in_:
            raise ValueError(f"入力特徴量の数がモデルと一致しません。期待される数: {trained_model.n_features_in_}, 実際の数: {input_features.shape[1]}")
    
    predicted_yield = trained_model.predict(input_features)
    return predicted_yield[0]

# --- UIの段階分け ---
# プログレスバーの表示
progress_value = (st.session_state.step) / 6
st.progress(progress_value, text=f"ステップ {st.session_state.step} / 6 完了")

col1, col2 = st.columns(2)
with col1:
    if st.session_state.step > 0:
        if st.button("戻る"):
            st.session_state.step -= 1
            st.rerun()

with col2:
    if st.session_state.step < 6:
        if st.button("次へ"):
            if st.session_state.step == 0:
                st.session_state.user_inputs['現在時刻'] = st.session_state.current_time_input
            elif st.session_state.step == 1:
                st.session_state.user_inputs['日射量_現在'] = st.session_state.sunlight_input
            elif st.session_state.step == 2:
                st.session_state.user_inputs['気温_現在'] = st.session_state.temp_input
            elif st.session_state.step == 3:
                st.session_state.user_inputs['湿度_現在'] = st.session_state.humidity_input
            elif st.session_state.step == 4:
                st.session_state.user_inputs['CO2濃度_現在'] = st.session_state.co2_input
            elif st.session_state.step == 5:
                st.session_state.user_inputs['積算日射量_現在'] = st.session_state.accumulated_sunlight_input
            st.session_state.step += 1
            st.rerun()

# --- 各ステップの表示 ---
if st.session_state.step == 0:
    st.metric(label="## ⏰ 現在時刻", value="")
    st.write("現在の時刻を5分単位で入力してください。")
    time_options = [f"{hour:02}:{minute:02}" for hour in range(24) for minute in range(0, 60, 5)]
    st.selectbox("選択値", time_options, key='current_time_input')

elif st.session_state.step == 1:
    st.metric(label="## ☀️ 日射量", value="")
    st.write("現在の日射量を入力してください。")
    st.number_input("入力値 (W/m²)", value=None, format="%.4f", key='sunlight_input')

elif st.session_state.step == 2:
    st.metric(label="## 🌡️ 気温", value="")
    st.write("現在の気温を入力してください。")
    st.number_input("入力値 (°C)", value=None, format="%.4f", key='temp_input')

elif st.session_state.step == 3:
    st.metric(label="## 💧 湿度", value="")
    st.write("現在の湿度を入力してください。")
    st.number_input("入力値 (%)", value=None, format="%.4f", key='humidity_input')

elif st.session_state.step == 4:
    st.metric(label="## 💨 CO2濃度", value="")
    st.write("現在のCO2濃度を入力してください。")
    st.number_input("入力値 (ppm)", value=None, format="%.4f", key='co2_input')

elif st.session_state.step == 5:
    st.metric(label="## 📊 積算日射量", value="")
    st.write("現在の積算日射量を入力してください。")
    st.number_input("入力値 (MJ/m²)", value=None, format="%.4f", key='accumulated_sunlight_input')

else:
    # 最終ステップ: 予測の実行と表示
    st.metric(label="## ✅ 最終確認と予測", value="")
    st.write("すべての入力が完了しました。以下のボタンを押して予測を開始してください。")
    
    # 入力内容の確認表示
    with st.expander("入力内容の確認"):
        st.write(f"**⏰ 現在時刻** : {st.session_state.user_inputs.get('現在時刻', '未入力')}")
        st.write(f"**☀️ 日射量** : {st.session_state.user_inputs.get('日射量_現在', '未入力')} W/m²")
        st.write(f"**🌡️ 気温** : {st.session_state.user_inputs.get('気温_現在', '未入力')} °C")
        st.write(f"**💧 湿度** : {st.session_state.user_inputs.get('湿度_現在', '未入力')} %")
        st.write(f"**💨 CO2濃度** : {st.session_state.user_inputs.get('CO2濃度_現在', '未入力')} ppm")
        st.write(f"**📊 積算日射量** : {st.session_state.user_inputs.get('積算日射量_現在', '未入力')} MJ/m²")

    if st.button("収穫量を予測 ✨"):
        try:
            # 辞書を予測関数が期待する形式に変換
            final_input_dict = {}
            for key, val in st.session_state.user_inputs.items():
                if key != '現在時刻':
                    base_name = key.replace('_現在', '')
                    final_input_dict[f'{base_name}_現在_mean'] = val
                    final_input_dict[f'{base_name}_現在_std'] = val
                    final_input_dict[f'{base_name}_Max_mean'] = val
                    final_input_dict[f'{base_name}_Min_mean'] = val
            
            predicted_yield = predict_total_cucumber_yield_flexible(final_input_dict, model, feature_means_dict)
            
            # 予測値を5倍にする
            predicted_yield_fivex = predicted_yield * 5
            
            if isinstance(model, GradientBoostingRegressor):
                st.success(f"予測される合計収穫量: **{predicted_yield_fivex:.1f} kg**")
            else:
                st.success(f"ダミーモデルによる予測結果: **{predicted_yield_fivex:.1f} kg**")
                st.info("※ この結果は仮のものです。正確な予測にはモデルファイルをアップロードしてください。")
        except Exception as e:
            st.error(f"予測中にエラーが発生しました: {e}")

st.markdown("---")
st.caption("**注記**: このモデルは限られたデータで学習されており、予測精度には限界があります。")
