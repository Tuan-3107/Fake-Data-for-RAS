#!/usr/bin/env python3
"""
Huấn luyện mô hình với dữ liệu THỰC TẾ
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
import pickle

print("="*70)
print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH - DỮ LIỆU THỰC TẾ")
print("="*70)

# Đọc dữ liệu
print("\n📂 Đọc dữ liệu từ file CSV...")
try:
    df = pd.read_csv('greenhouse_data_realistic.csv')
    print(f"✅ Đã đọc {len(df)} mẫu dữ liệu")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file greenhouse_data_realistic.csv")
    print("💡 Hãy chạy: python3 generate_data_realistic.py trước")
    exit()

# Chuẩn bị dữ liệu
print("\n🔧 Chuẩn bị dữ liệu...")
X = df[['temp', 'humidity_air', 'humidity_soil', 'light', 'forecast_temp', 'rain_prob']].values

y_fan_on = df['fan_on'].values
y_pump_on = df['pump_on'].values
y_light_on = df['light_on'].values

y_fan_pwm = df['fan_pwm'].values
y_pump_pwm = df['pump_pwm'].values
y_light_pwm = df['light_pwm'].values

# Chia train/test
X_train, X_test, y_fan_on_train, y_fan_on_test = train_test_split(
    X, y_fan_on, test_size=0.2, random_state=42
)

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

# Đánh giá chi tiết
fan_pred = rf_fan.predict(X_test)
pump_pred = rf_pump.predict(X_test)
light_pred = rf_light.predict(X_test)

print("\n📊 KẾT QUẢ RANDOM FOREST:")
print(f"  🌪️  Quạt  - Accuracy: {accuracy_score(y_fan_on_test, fan_pred)*100:.2f}%")
print(f"  💦 Bơm  - Accuracy: {accuracy_score(y_pump_on_test, pump_pred)*100:.2f}%")
print(f"  💡 Đèn  - Accuracy: {accuracy_score(y_light_on_test, light_pred)*100:.2f}%")

# Cross-validation để kiểm tra overfitting
print("\n🔍 KIỂM TRA ĐỘ ỔN ĐỊNH (Cross-Validation):")
cv_fan = cross_val_score(rf_fan, X, y_fan_on, cv=5, scoring='accuracy')
cv_pump = cross_val_score(rf_pump, X, y_pump_on, cv=5, scoring='accuracy')
cv_light = cross_val_score(rf_light, X, y_light_on, cv=5, scoring='accuracy')

print(f"  🌪️  Quạt  - CV Mean: {cv_fan.mean()*100:.2f}% (±{cv_fan.std()*100:.2f}%)")
print(f"  💦 Bơm  - CV Mean: {cv_pump.mean()*100:.2f}% (±{cv_pump.std()*100:.2f}%)")
print(f"  💡 Đèn  - CV Mean: {cv_light.mean()*100:.2f}% (±{cv_light.std()*100:.2f}%)")

# Confusion Matrix
print("\n📋 CONFUSION MATRIX (Ma trận nhầm lẫn):")
print("  Quạt:")
cm_fan = confusion_matrix(y_fan_on_test, fan_pred)
print(f"    True Neg: {cm_fan[0,0]}, False Pos: {cm_fan[0,1]}")
print(f"    False Neg: {cm_fan[1,0]}, True Pos: {cm_fan[1,1]}")

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

# Đánh giá
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
    'rf_fan_realistic.pkl': rf_fan,
    'rf_pump_realistic.pkl': rf_pump,
    'rf_light_realistic.pkl': rf_light,
    'lr_fan_pwm_realistic.pkl': lr_fan,
    'lr_pump_pwm_realistic.pkl': lr_pump,
    'lr_light_pwm_realistic.pkl': lr_light
}

for filename, model in models.items():
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ {filename}")

print("\n" + "="*70)
print("🎉 HOÀN TẤT!")
print("="*70)
print("\n💡 GIẢI THÍCH KẾT QUẢ:")
print("  • Accuracy 85-95%: THỰC TẾ và TỐT ✅")
print("  • Accuracy 100%: QUÁ HOÀN HẢO, có thể overfitting ⚠️")
print("  • R² Score 0.8-0.95: Dự đoán tốt ✅")
print("  • Cross-Validation ổn định: Không bị overfitting ✅")
print("\n📌 Bước tiếp theo:")
print("  Chạy: python3 test_model_realistic.py")
