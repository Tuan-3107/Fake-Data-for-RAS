#!/usr/bin/env python3
"""
Huấn luyện mô hình Random Forest và Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import pickle

print("="*70)
print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING")
print("="*70)

# Đọc dữ liệu
print("\n📂 Đọc dữ liệu từ file CSV...")
try:
    df = pd.read_csv('greenhouse_data.csv')
    print(f"✅ Đã đọc {len(df)} mẫu dữ liệu")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file greenhouse_data.csv")
    print("💡 Hãy chạy: python3 generate_data.py trước")
    exit()

# Chuẩn bị features (X) và labels (y)
print("\n🔧 Chuẩn bị dữ liệu...")
X = df[['temp', 'humidity_air', 'humidity_soil', 'light', 'forecast_temp', 'rain_prob']].values

# Labels cho Random Forest (phân loại ON/OFF)
y_fan_on = df['fan_on'].values
y_pump_on = df['pump_on'].values
y_light_on = df['light_on'].values

# Labels cho Linear Regression (dự đoán PWM)
y_fan_pwm = df['fan_pwm'].values
y_pump_pwm = df['pump_pwm'].values
y_light_pwm = df['light_pwm'].values

# Chia train/test (80% train, 20% test)
X_train, X_test, y_fan_on_train, y_fan_on_test = train_test_split(
    X, y_fan_on, test_size=0.2, random_state=42
)

# (Dùng cùng X_train, X_test cho tất cả models)
_, _, y_pump_on_train, y_pump_on_test = train_test_split(X, y_pump_on, test_size=0.2, random_state=42)
_, _, y_light_on_train, y_light_on_test = train_test_split(X, y_light_on, test_size=0.2, random_state=42)

_, _, y_fan_pwm_train, y_fan_pwm_test = train_test_split(X, y_fan_pwm, test_size=0.2, random_state=42)
_, _, y_pump_pwm_train, y_pump_pwm_test = train_test_split(X, y_pump_pwm, test_size=0.2, random_state=42)
_, _, y_light_pwm_train, y_light_pwm_test = train_test_split(X, y_light_pwm, test_size=0.2, random_state=42)

print(f"  • Train set: {len(X_train)} mẫu")
print(f"  • Test set: {len(X_test)} mẫu")

# ===== RANDOM FOREST =====
print("\n" + "="*70)
print("🌳 HUẤN LUYỆN RANDOM FOREST (Quyết định BẬT/TẮT)")
print("="*70)

print("  Đang train model Quạt...", end=" ")
rf_fan = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fan.fit(X_train, y_fan_on_train)
print("✅")

print("  Đang train model Bơm...", end=" ")
rf_pump = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_pump.fit(X_train, y_pump_on_train)
print("✅")

print("  Đang train model Đèn...", end=" ")
rf_light = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_light.fit(X_train, y_light_on_train)
print("✅")

# Đánh giá Random Forest
fan_pred = rf_fan.predict(X_test)
pump_pred = rf_pump.predict(X_test)
light_pred = rf_light.predict(X_test)

print("\n📊 KẾT QUẢ RANDOM FOREST:")
print(f"  🌪️  Quạt  - Accuracy: {accuracy_score(y_fan_on_test, fan_pred)*100:.2f}%")
print(f"  💦 Bơm  - Accuracy: {accuracy_score(y_pump_on_test, pump_pred)*100:.2f}%")
print(f"  💡 Đèn  - Accuracy: {accuracy_score(y_light_on_test, light_pred)*100:.2f}%")

# ===== LINEAR REGRESSION =====
print("\n" + "="*70)
print("📏 HUẤN LUYỆN LINEAR REGRESSION (Dự đoán PWM)")
print("="*70)

print("  Đang train model Quạt PWM...", end=" ")
lr_fan = LinearRegression()
lr_fan.fit(X_train, y_fan_pwm_train)
print("✅")

print("  Đang train model Bơm PWM...", end=" ")
lr_pump = LinearRegression()
lr_pump.fit(X_train, y_pump_pwm_train)
print("✅")

print("  Đang train model Đèn PWM...", end=" ")
lr_light = LinearRegression()
lr_light.fit(X_train, y_light_pwm_train)
print("✅")

# Đánh giá Linear Regression
fan_pwm_pred = lr_fan.predict(X_test)
pump_pwm_pred = lr_pump.predict(X_test)
light_pwm_pred = lr_light.predict(X_test)

print("\n📊 KẾT QUẢ LINEAR REGRESSION:")
print(f"  🌪️  Quạt PWM  - R²: {r2_score(y_fan_pwm_test, fan_pwm_pred):.4f}, MSE: {mean_squared_error(y_fan_pwm_test, fan_pwm_pred):.2f}")
print(f"  💦 Bơm PWM  - R²: {r2_score(y_pump_pwm_test, pump_pwm_pred):.4f}, MSE: {mean_squared_error(y_pump_pwm_test, pump_pwm_pred):.2f}")
print(f"  💡 Đèn PWM  - R²: {r2_score(y_light_pwm_test, light_pwm_pred):.4f}, MSE: {mean_squared_error(y_light_pwm_test, light_pwm_pred):.2f}")

# ===== LƯU MÔ HÌNH =====
print("\n" + "="*70)
print("💾 LƯU MÔ HÌNH")
print("="*70)

models = {
    'rf_fan.pkl': rf_fan,
    'rf_pump.pkl': rf_pump,
    'rf_light.pkl': rf_light,
    'lr_fan_pwm.pkl': lr_fan,
    'lr_pump_pwm.pkl': lr_pump,
    'lr_light_pwm.pkl': lr_light
}

for filename, model in models.items():
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ {filename}")

print("\n" + "="*70)
print("🎉 HOÀN TẤT! Tất cả mô hình đã được huấn luyện và lưu.")
print("="*70)
print("\n💡 Giải thích kết quả:")
print("  • Accuracy > 90%: Rất tốt! ✅")
print("  • R² Score > 0.8: Dự đoán chính xác! ✅")
print("  • MSE càng thấp càng tốt")
print("\n📌 Bước tiếp theo:")
print("  Chạy: python3 test_model.py để xem mô hình hoạt động")
