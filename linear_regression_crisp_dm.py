import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_crisp_dm_linear_regression():
    """
    A function that performs a simple linear regression task following the CRISP-DM methodology.
    It allows user to define the parameters for data generation.
    """
    
    # --- 1. Business Understanding (業務理解) ---
    print("--- 1. Business Understanding (業務理解) ---")
    print("目標: 我們的目標是從一組帶有噪聲的數據點中，找出其背後的線性關係。")
    print("任務: 建立一個簡單線性迴歸模型 y = ax + b。")
    print("成功標準: 模型預測的線條能很好地擬合數據，且找出的 a (斜率) 和 b (截距) 與真實值接近。")
    print("-" * 50)
    
    # --- User Input ---
    print("請設定數據生成參數:")
    try:
        a_true = float(input("請輸入真實的斜率 a (例如: 2.5): "))
    except ValueError:
        print("輸入無效，使用預設值 a = 2.5")
        a_true = 2.5
        
    try:
        noise_level = float(input("請輸入噪聲大小 (例如: 5.0): "))
    except ValueError:
        print("輸入無效，使用預設值 noise = 5.0")
        noise_level = 5.0
        
    try:
        num_points = int(input("請輸入數據點數量 (例如: 100): "))
    except ValueError:
        print("輸入無效，使用預設值 num_points = 100")
        num_points = 100
    
    b_true = 5.0 # 我們固定 b 以簡化問題
    print(f"參數設定: a={a_true}, b={b_true}, 噪聲={noise_level}, 數據點數量={num_points}\n")
    
    # --- 2. Data Understanding (數據理解) ---
    print("--- 2. Data Understanding (數據理解) ---")
    print("過程: 根據 y = ax + b + noise 的公式生成數據。")
    # Generate x values
    X = np.linspace(0, 50, num_points)
    # Generate y values with noise
    noise = np.random.normal(0, noise_level, num_points)
    y = a_true * X + b_true + noise
    
    print(f"已生成 {len(X)} 個數據點。")
    print("下一步: 我們將數據可視化，以直觀地觀察其分佈和線性趨勢。")
    
    # Plotting the generated data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7, label='Generated Data Points')
    plt.title('Data Understanding: Generated Raw Data')
    plt.xlabel('X (Feature)')
    plt.ylabel('y (Target)')
    plt.legend()
    plt.grid(True)
    print("圖表已生成，請查看。")
    plt.show()
    print("-" * 50)

    # --- 3. Data Preparation (數據準備) ---
    print("--- 3. Data Preparation (數據準備) ---")
    print("過程: 為了訓練和評估模型，我們需要將數據分為訓練集和測試集。")
    # scikit-learn's LinearRegression model expects X to be a 2D array.
    X_reshaped = X.reshape(-1, 1)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
    print(f"數據已分割: {len(X_train)} 筆訓練數據, {len(X_test)} 筆測試數據。")
    print("X 已被轉換為 scikit-learn 需要的二維格式。")
    print("-" * 50)

    # --- 4. Modeling (模型建立) ---
    print("--- 4. Modeling (模型建立) ---")
    print("過程: 我們選擇 scikit-learn 的 LinearRegression 作為我們的模型。")
    # Create a linear regression model instance
    model = LinearRegression()
    
    print("模型正在使用訓練數據進行訓練...")
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    print("模型訓練完成！")
    print(f"模型找出的斜率 (a): {model.coef_[0]:.4f}")
    print(f"模型找出的截距 (b): {model.intercept_:.4f}")
    print("-" * 50)

    # --- 5. Evaluation (模型評估) ---
    print("--- 5. Evaluation (模型評估) ---")
    print("過程: 使用測試集來評估模型的性能，並將模型的預測結果與真實數據進行比較。")
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"模型在測試集上的表現:")
    print(f"  - 均方誤差 (Mean Squared Error): {mse:.4f}")
    print(f"  - R-squared (決定係數): {r2:.4f} (越接近1越好)")
    print("\n比較模型找出的參數與真實參數:")
    print(f"  - 真實斜率 a: {a_true}")
    print(f"  - 模型斜率 a: {model.coef_[0]:.4f}")
    print(f"  - 真實截距 b: {b_true}")
    print(f"  - 模型截距 b: {model.intercept_:.4f}")
    
    print("\n下一步: 我們將模型的預測結果可視化。")
    # Plotting the results
    plt.figure(figsize=(12, 7))
    # All data points
    plt.scatter(X, y, alpha=0.5, label='All Data Points')
    # The true line
    plt.plot(X, a_true * X + b_true, color='red', linewidth=3, label=f'True Line (a={a_true}, b={b_true})')
    # The model's learned line
    plt.plot(X, model.predict(X_reshaped), color='green', linewidth=3, label=f'Model's Prediction (a={model.coef_[0]:.2f}, b={model.intercept_:.2f})')
    
    plt.title('Evaluation: Model Prediction vs. True Line')
    plt.xlabel('X (Feature)')
    plt.ylabel('y (Target)')
    plt.legend()
    plt.grid(True)
    print("最終結果圖表已生成，請查看。")
    plt.show()
    print("-" * 50)

    # --- 6. Deployment (部署) ---
    print("--- 6. Deployment (部署) ---")
    print("過程: 在這個範例中，部署階段意味著我們可以使用這個訓練好的模型來預測新的、未見過的數據點。")
    
    try:
        new_x_str = input("請輸入一個新的 x 值來進行預測 (或直接按 Enter 跳過): ")
        if new_x_str:
            new_x = float(new_x_str)
            # Reshape the new_x for prediction
            new_x_reshaped = np.array([[new_x]])
            predicted_y = model.predict(new_x_reshaped)
            print(f"對於新的 x = {new_x}, 模型預測的 y 值為: {predicted_y[0]:.4f}")
        else:
            print("未輸入新值，跳過預測。")
    except ValueError:
        print("輸入無效，跳過預測。")
        
    print("\nCRISP-DM 流程完成！")

if __name__ == '__main__':
    run_crisp_dm_linear_regression()