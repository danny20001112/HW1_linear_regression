
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_crisp_dm_linear_regression():
    """
    Main function to run the Streamlit App.
    """
    st.set_page_config(page_title="CRISP-DM Linear Regression", layout="wide")

    # --- Sidebar for User Input ---
    st.sidebar.title("數據生成參數")
    st.sidebar.markdown("請在此調整用於生成模擬數據的參數。")
    a_true = st.sidebar.slider("真實斜率 (a)", 1.0, 10.0, 2.5, 0.1)
    noise_level = st.sidebar.slider("噪聲大小", 0.0, 20.0, 5.0, 0.5)
    num_points = st.sidebar.slider("數據點數量", 50, 500, 100, 10)
    b_true = 5.0  # 固定 b 以簡化問題

    # --- Main App Body ---
    st.title("使用 CRISP-DM 流程的簡單線性迴歸分析")
    st.markdown("這是一個互動式 Web 應用，用於演示機器學習的 **CRISP-DM** 流程。您可以透過左側的滑桿來即時改變數據特性，並觀察模型的變化。")

    # --- 1. Business Understanding (業務理解) ---
    with st.expander("第一步: 業務理解 (Business Understanding)", expanded=True):
        st.markdown("""
        - **目標:** 從一組帶有噪聲的數據點中，找出其背後的線性關係。
        - **任務:** 建立一個簡單線性迴歸模型 `y = ax + b`。
        - **成功標準:** 模型預測的線條能很好地擬合數據，且找出的 `a` (斜率) 和 `b` (截距) 與真實值接近。
        """)

    # --- Data Generation (Shared across steps) ---
    X = np.linspace(0, 50, num_points)
    noise = np.random.normal(0, noise_level, num_points)
    y = a_true * X + b_true + noise
    X_reshaped = X.reshape(-1, 1)

    # --- 2. Data Understanding (數據理解) ---
    with st.expander("第二步: 數據理解 (Data Understanding)"):
        st.markdown("**過程:** 根據 `y = ax + b + noise` 的公式生成數據，並透過散點圖進行可視化，以直觀地觀察其分佈和線性趨勢。")
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(X, y, alpha=0.7, label='Generated Data Points')
        ax1.set_title('Data Understanding: Generated Raw Data')
        ax1.set_xlabel('X (Feature)')
        ax1.set_ylabel('y (Target)')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

    # --- 3. Data Preparation (數據準備) ---
    with st.expander("第三步: 數據準備 (Data Preparation)"):
        st.markdown("**過程:** 為了訓練和評估模型，我們需要將數據分為訓練集 (80%) 和測試集 (20%)。同時，特徵 `X` 需要被轉換為 scikit-learn 需要的二維格式。")
        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
        st.write(f"- 數據已分割: `{len(X_train)}` 筆訓練數據, `{len(X_test)}` 筆測試數據。")
        st.write(f"- `X` 的原始 shape: `{X.shape}` -> 轉換後 shape: `{X_reshaped.shape}`")

    # --- 4. Modeling (模型建立) ---
    with st.expander("第四步: 模型建立 (Modeling)"):
        st.markdown("**過程:** 我們選擇 `scikit-learn` 的 `LinearRegression` 作為模型，並使用準備好的訓練集進行訓練。")
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.success("模型訓練完成！")
        st.write(f"- **模型找出的斜率 (a):** `{model.coef_[0]:.4f}`")
        st.write(f"- **模型找出的截距 (b):** `{model.intercept_:.4f}`")

    # --- 5. Evaluation (模型評估) ---
    with st.expander("第五步: 模型評估 (Evaluation)", expanded=True):
        st.markdown("**過程:** 使用 **測試集** 來評估模型的性能，並將模型的預測結果與真實數據進行視覺化比較。")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("性能指標")
            st.metric(label="均方誤差 (MSE)", value=f"{mse:.4f}", help="越小越好")
            st.metric(label="R-squared (R²)", value=f"{r2:.4f}", help="越接近 1 越好")
        with col2:
            st.subheader("參數比較")
            st.metric(label="真實斜率 vs 模型斜率", value=f"{a_true:.2f} vs {model.coef_[0]:.2f}")
            st.metric(label="真實截距 vs 模型截距", value=f"{b_true:.2f} vs {model.intercept_:.2f}")

        st.markdown("--- ")
        st.subheader("視覺化評估")
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.scatter(X, y, alpha=0.3, label='All Data Points')
        ax2.plot(X, a_true * X + b_true, color='red', linewidth=3, linestyle='--', label=f'True Line (a={a_true:.2f})')
        ax2.plot(X, model.predict(X_reshaped), color='green', linewidth=3, label=f'Model's Prediction (a={model.coef_[0]:.2f})')
        ax2.set_title('Evaluation: Model Prediction vs. True Line')
        ax2.set_xlabel('X (Feature)')
        ax2.set_ylabel('y (Target)')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    # --- 6. Deployment (部署) ---
    with st.expander("第六步: 部署 (Deployment)"):
        st.markdown("**過程:** 在這個範例中，部署階段意味著我們可以使用這個訓練好的模型來預測新的、未見過的數據點。")
        new_x = st.number_input("請輸入一個新的 x 值來進行預測:", value=25.0, format="%.2f")
        if new_x is not None:
            new_x_reshaped = np.array([[new_x]])
            predicted_y = model.predict(new_x_reshaped)
            st.success(f"對於新的 x = `{new_x}`，模型預測的 y 值為: **`{predicted_y[0]:.4f}`**")

if __name__ == '__main__':
    run_crisp_dm_linear_regression()
