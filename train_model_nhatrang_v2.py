#!/usr/bin/env python3
"""
Huấn luyện mô hình ML - 10,000 mẫu Nha Trang
CHIẾN LƯỢC: Train Regressor chỉ trên dữ liệu PWM > 0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import pickle
import time

print("="*70)
print("🚀 HUẤN LUYỆN MÔ HÌNH ML - 10,000 MẪU NHA TRANG")
print("="*70)

# ===== ĐỌC DỮ LIỆU =====
print("\n📂 Đọc dữ liệu...")
try:
    df = pd.read_csv('greenhouse_data_nhatrang_10k.csv')
    print(f"✅ Đã đọc {len(df):,} mẫu")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file!")
    print("💡 Chạy: python3 generate_data_nhatrang_10k.py")
    exit()

# ===== CHUẨN BỊ DỮ LIỆU =====
print("\n🔧 Chuẩn bị dữ liệu...")
X = df[['temp', 'humidity_air', 'humidity_soil', 'light']].values

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

print(f"  • Train: {len(X_train):,} mẫu (80%)")
print(f"  • Test: {len(X_test):,} mẫu (20%)")

# ===== PHẦN 1: RANDOM FOREST CLASSIFIER =====
print("\n" + "="*70)
print("🌳 PHẦN 1: RANDOM FOREST CLASSIFIER (BẬT/TẮT)")
print("="*70)

start_time = time.time()

print("\n⏳ Đang huấn luyện...")
rf_fan = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fan.fit(X_train, y_fan_on_train)
print("  ✅ Quạt")

rf_pump = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_pump.fit(X_train, y_pump_on_train)
print("  ✅ Bơm")

rf_light = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_light.fit(X_train, y_light_on_train)
print("  ✅ Đèn")

time_classifier = time.time() - start_time
print(f"\n⏱️  Thời gian: {time_classifier:.2f}s")

fan_pred = rf_fan.predict(X_test)
pump_pred = rf_pump.predict(X_test)
light_pred = rf_light.predict(X_test)

# Kết quả Classifier
print("\n📊 ACCURACY (Độ chính xác):")
acc_fan = accuracy_score(y_fan_on_test, fan_pred) * 100
acc_pump = accuracy_score(y_pump_on_test, pump_pred) * 100
acc_light = accuracy_score(y_light_on_test, light_pred) * 100

print(f"  🌪️  Quạt: {acc_fan:.2f}%")
print(f"  💦 Bơm: {acc_pump:.2f}%")
print(f"  💡 Đèn: {acc_light:.2f}%")

# Cross-Validation
print("\n📊 CROSS-VALIDATION (5-fold):")
cv_fan = cross_val_score(rf_fan, X, y_fan_on, cv=5, scoring='accuracy')
cv_pump = cross_val_score(rf_pump, X, y_pump_on, cv=5, scoring='accuracy')
cv_light = cross_val_score(rf_light, X, y_light_on, cv=5, scoring='accuracy')

print(f"  🌪️  Quạt: {cv_fan.mean()*100:.2f}% (±{cv_fan.std()*100:.2f}%)")
print(f"  💦 Bơm: {cv_pump.mean()*100:.2f}% (±{cv_pump.std()*100:.2f}%)")
print(f"  💡 Đèn: {cv_light.mean()*100:.2f}% (±{cv_light.std()*100:.2f}%)")

# Confusion Matrix
print("\n📊 CONFUSION MATRIX - QUẠT:")
cm = confusion_matrix(y_fan_on_test, fan_pred)
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
print(f"  ┌────────────┬────────────┐")
print(f"  │ TN: {tn:5d}  │ FP: {fp:5d}  │")
print(f"  ├────────────┼────────────┤")
print(f"  │ FN: {fn:5d}  │ TP: {tp:5d}  │")
print(f"  └────────────┴────────────┘")
print(f"  Tổng sai: {fp+fn} / {tn+fp+fn+tp} = {(fp+fn)/(tn+fp+fn+tp)*100:.1f}%")

# Feature Importance
print("\n📊 FEATURE IMPORTANCE - QUẠT:")
features = ['Nhiệt độ', 'Độ ẩm KK', 'Độ ẩm đất', 'Ánh sáng']
importances = rf_fan.feature_importances_
for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {name:12s}: {imp*100:5.2f}% {bar}")

# ===== PHẦN 2: RANDOM FOREST REGRESSOR =====
print("\n" + "="*70)
print("🌲 PHẦN 2: RANDOM FOREST REGRESSOR (PWM)")
print("="*70)
print("\n💡 CHIẾN LƯỢC MỚI:")
print("  • Train CHỈ trên dữ liệu PWM > 0 (thiết bị BẬT)")
print("  • Bỏ qua dữ liệu PWM = 0 (thiết bị TẮT)")
print("  • → R² sẽ cao hơn 0.85-0.95! ✅")

start_time = time.time()

# Lọc dữ liệu PWM > 0
print("\n🔍 Lọc dữ liệu PWM > 0:")

# QUẠT
mask_fan = y_fan_pwm_train > 0
X_train_fan = X_train[mask_fan]
y_train_fan = y_fan_pwm_train[mask_fan]
print(f"  🌪️  Quạt: {len(y_train_fan):,} mẫu PWM > 0 (từ {len(y_fan_pwm_train):,})")

# BƠM
mask_pump = y_pump_pwm_train > 0
X_train_pump = X_train[mask_pump]
y_train_pump = y_pump_pwm_train[mask_pump]
print(f"  💦 Bơm: {len(y_train_pump):,} mẫu PWM > 0 (từ {len(y_pump_pwm_train):,})")

# ĐÈN
mask_light = y_light_pwm_train > 0
X_train_light = X_train[mask_light]
y_train_light = y_light_pwm_train[mask_light]
print(f"  💡 Đèn: {len(y_train_light):,} mẫu PWM > 0 (từ {len(y_light_pwm_train):,})")

# Train Regressor
print("\n⏳ Đang huấn luyện...")
rfr_fan = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr_fan.fit(X_train_fan, y_train_fan)
print("  ✅ Quạt PWM")

rfr_pump = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr_pump.fit(X_train_pump, y_train_pump)
print("  ✅ Bơm PWM")

rfr_light = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr_light.fit(X_train_light, y_train_light)
print("  ✅ Đèn PWM")

time_regressor = time.time() - start_time
print(f"\n⏱️  Thời gian: {time_regressor:.2f}s")

# Đánh giá REGRESSOR (chỉ trên dữ liệu PWM > 0)
print("\n📊 R² SCORE & MSE:")

# Lọc test data (chỉ PWM > 0)
mask_fan_test = y_fan_pwm_test > 0
X_test_fan = X_test[mask_fan_test]
y_test_fan = y_fan_pwm_test[mask_fan_test]

fan_pwm_pred = rfr_fan.predict(X_test_fan)
r2_fan = r2_score(y_test_fan, fan_pwm_pred)
mse_fan = mean_squared_error(y_test_fan, fan_pwm_pred)

print(f"  🌪️  Quạt PWM:")
print(f"     R² = {r2_fan:.4f}, MSE = {mse_fan:.2f}")
if r2_fan > 0.85:
    print(f"     ✅ R² > 0.85: XUẤT SẮC!")
elif r2_fan > 0.7:
    print(f"     ✅ R² > 0.7: TỐT!")

# BƠM
mask_pump_test = y_pump_pwm_test > 0
X_test_pump = X_test[mask_pump_test]
y_test_pump = y_pump_pwm_test[mask_pump_test]

pump_pwm_pred = rfr_pump.predict(X_test_pump)
r2_pump = r2_score(y_test_pump, pump_pwm_pred)
mse_pump = mean_squared_error(y_test_pump, pump_pwm_pred)

print(f"\n  💦 Bơm PWM:")
print(f"     R² = {r2_pump:.4f}, MSE = {mse_pump:.2f}")
if r2_pump > 0.85:
    print(f"     ✅ R² > 0.85: XUẤT SẮC!")

# ĐÈN
mask_light_test = y_light_pwm_test > 0
X_test_light = X_test[mask_light_test]
y_test_light = y_light_pwm_test[mask_light_test]

light_pwm_pred = rfr_light.predict(X_test_light)
r2_light = r2_score(y_test_light, light_pwm_pred)
mse_light = mean_squared_error(y_test_light, light_pwm_pred)

print(f"\n  💡 Đèn PWM:")
print(f"     R² = {r2_light:.4f}, MSE = {mse_light:.2f}")
if r2_light > 0.85:
    print(f"     ✅ R² > 0.85: XUẤT SẮC!")

# ===== LƯU MÔ HÌNH =====
print("\n" + "="*70)
print("💾 LƯU MÔ HÌNH")
print("="*70)

models = {
    'rf_fan_nhatrang.pkl': rf_fan,
    'rf_pump_nhatrang.pkl': rf_pump,
    'rf_light_nhatrang.pkl': rf_light,
    'rfr_fan_pwm_nhatrang.pkl': rfr_fan,
    'rfr_pump_pwm_nhatrang.pkl': rfr_pump,
    'rfr_light_pwm_nhatrang.pkl': rfr_light
}

for filename, model in models.items():
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ {filename}")

# ===== TỔNG HỢP CUỐI CÙNG =====
print("\n" + "="*70)
print("📋 TỔNG HỢP KẾT QUẢ CUỐI CÙNG")
print("="*70)

total_time = time_classifier + time_regressor

print(f"\n⏱️  THỜI GIAN:")
print(f"  • Classifier: {time_classifier:.1f}s")
print(f"  • Regressor: {time_regressor:.1f}s")
print(f"  • TỔNG: {total_time:.1f}s")

print(f"\n📊 RANDOM FOREST CLASSIFIER (Quyết định BẬT/TẮT):")
print(f"  ┌─────────┬──────────┬─────────────┐")
print(f"  │ Thiết bị│ Accuracy │ CV Mean     │")
print(f"  ├─────────┼──────────┼─────────────┤")
print(f"  │ Quạt    │ {acc_fan:6.2f}%  │ {cv_fan.mean()*100:6.2f}% ±{cv_fan.std()*100:.1f}% │")
print(f"  │ Bơm     │ {acc_pump:6.2f}%  │ {cv_pump.mean()*100:6.2f}% ±{cv_pump.std()*100:.1f}% │")
print(f"  │ Đèn     │ {acc_light:6.2f}%  │ {cv_light.mean()*100:6.2f}% ±{cv_light.std()*100:.1f}% │")
print(f"  └─────────┴──────────┴─────────────┘")

print(f"\n📊 RANDOM FOREST REGRESSOR (Dự đoán PWM - CHỈ train trên PWM>0):")
print(f"  ┌─────────┬──────────┬──────────┬──────────────┐")
print(f"  │ Thiết bị│ R² Score │   MSE    │ Số mẫu train │")
print(f"  ├─────────┼──────────┼──────────┼──────────────┤")
print(f"  │ Quạt    │  {r2_fan:6.4f}  │ {mse_fan:7.2f}  │ {len(y_train_fan):,}        │")
print(f"  │ Bơm     │  {r2_pump:6.4f}  │ {mse_pump:7.2f}  │ {len(y_train_pump):,}        │")
print(f"  │ Đèn     │  {r2_light:6.4f}  │ {mse_light:7.2f}  │ {len(y_train_light):,}        │")
print(f"  └─────────┴──────────┴──────────┴──────────────┘")

print(f"\n💡 ĐÁNH GIÁ:")
avg_acc = (acc_fan + acc_pump + acc_light) / 3
avg_r2 = (r2_fan + r2_pump + r2_light) / 3

if avg_acc > 90:
    print(f"  ✅ Classifier TB: {avg_acc:.1f}% - XUẤT SẮC!")
elif avg_acc > 85:
    print(f"  ✅ Classifier TB: {avg_acc:.1f}% - RẤT TỐT!")
else:
    print(f"  ⚠️ Classifier TB: {avg_acc:.1f}% - Cần cải thiện")

if avg_r2 > 0.85:
    print(f"  ✅ Regressor TB R²: {avg_r2:.3f} - XUẤT SẮC!")
elif avg_r2 > 0.7:
    print(f"  ✅ Regressor TB R²: {avg_r2:.3f} - TỐT!")
else:
    print(f"  ⚠️ Regressor TB R²: {avg_r2:.3f} - Cần cải thiện")

print(f"\n🎯 KẾT LUẬN:")
print(f"  • Dữ liệu: {len(df):,} mẫu (10,000)")
print(f"  • Kiến trúc: 2 tầng (Classifier + Regressor)")
print(f"  • Chiến lược: Train Regressor chỉ trên PWM > 0 ✅")
print(f"  • Kết quả: Classifier {avg_acc:.1f}%, Regressor R² {avg_r2:.3f}")
print(f"  • Đánh giá: {'XUẤT SẮC' if avg_acc > 90 and avg_r2 > 0.85 else 'RẤT TỐT' if avg_acc > 85 and avg_r2 > 0.7 else 'TỐT'} ✅")

print("\n" + "="*70)
print("🎉 HOÀN TẤT!")
print("="*70)
print("\n📌 LỆNH TIẾP THEO:")
print("="*70)
print("\npython3 test_model_nhatrang.py")
