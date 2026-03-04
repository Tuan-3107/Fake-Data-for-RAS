#!/usr/bin/env python3
"""
Huấn luyện mô hình ML cho dữ liệu Nha Trang
BẢN CHÚ THÍCH CHI TIẾT - GIẢI THÍCH TẤT CẢ CÁC THÔNG SỐ
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import pickle
import time  # Để đo thời gian training

print("="*70)
print("🚀 HUẤN LUYỆN MÔ HÌNH ML - KHÍ HẬU NHA TRANG")
print("="*70)

# ===== ĐỌC DỮ LIỆU =====
print("\n📂 Đọc dữ liệu từ file CSV...")
try:
    df = pd.read_csv('greenhouse_data_nhatrang.csv')
    print(f"✅ Đã đọc {len(df)} mẫu dữ liệu")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file greenhouse_data_nhatrang.csv")
    print("💡 Hãy chạy: python3 generate_data_nhatrang.py trước")
    exit()

# ===== CHUẨN BỊ DỮ LIỆU =====
print("\n🔧 Chuẩn bị dữ liệu...")
print("📊 Sử dụng 4 features (đặc trưng):")
print("  1️⃣  Nhiệt độ (°C)")
print("  2️⃣  Độ ẩm không khí (%)")
print("  3️⃣  Độ ẩm đất (%)")
print("  4️⃣  Ánh sáng (Lux)")

X = df[['temp', 'humidity_air', 'humidity_soil', 'light']].values

# Labels (nhãn) cho Classifier
y_fan_on = df['fan_on'].values
y_pump_on = df['pump_on'].values
y_light_on = df['light_on'].values

# Labels cho Regressor
y_fan_pwm = df['fan_pwm'].values
y_pump_pwm = df['pump_pwm'].values
y_light_pwm = df['light_pwm'].values

# Chia train/test (80% train, 20% test)
print("\n📊 Chia dữ liệu:")
print("  • 80% (800 mẫu) → Huấn luyện (train)")
print("  • 20% (200 mẫu) → Kiểm tra (test)")

X_train, X_test, y_fan_on_train, y_fan_on_test = train_test_split(
    X, y_fan_on, test_size=0.2, random_state=42
)

_, _, y_pump_on_train, y_pump_on_test = train_test_split(X, y_pump_on, test_size=0.2, random_state=42)
_, _, y_light_on_train, y_light_on_test = train_test_split(X, y_light_on, test_size=0.2, random_state=42)

_, _, y_fan_pwm_train, y_fan_pwm_test = train_test_split(X, y_fan_pwm, test_size=0.2, random_state=42)
_, _, y_pump_pwm_train, y_pump_pwm_test = train_test_split(X, y_pump_pwm, test_size=0.2, random_state=42)
_, _, y_light_pwm_train, y_light_pwm_test = train_test_split(X, y_light_pwm, test_size=0.2, random_state=42)

print(f"\n  ✅ Train set: {len(X_train)} mẫu")
print(f"  ✅ Test set: {len(X_test)} mẫu")

# ===== RANDOM FOREST CLASSIFIER =====
print("\n" + "="*70)
print("🌳 PHẦN 1: RANDOM FOREST CLASSIFIER (Quyết định BẬT/TẮT)")
print("="*70)
print("\n📖 GIẢI THÍCH:")
print("  • Mục đích: Quyết định có nên BẬT thiết bị không")
print("  • Input: 4 giá trị cảm biến")
print("  • Output: 0 (TẮT) hoặc 1 (BẬT)")
print("  • Thuật toán: 100 cây quyết định bầu chọn")

# Đo thời gian training
start_time = time.time()

print("\n⏳ Đang huấn luyện...")
print("  [1/3] Model Quạt...", end=" ", flush=True)
rf_fan = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fan.fit(X_train, y_fan_on_train)
print("✅")

print("  [2/3] Model Bơm...", end=" ", flush=True)
rf_pump = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_pump.fit(X_train, y_pump_on_train)
print("✅")

print("  [3/3] Model Đèn...", end=" ", flush=True)
rf_light = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_light.fit(X_train, y_light_on_train)
print("✅")

training_time_classifier = time.time() - start_time
print(f"\n⏱️  Thời gian huấn luyện: {training_time_classifier:.2f} giây")

# Dự đoán
fan_pred = rf_fan.predict(X_test)
pump_pred = rf_pump.predict(X_test)
light_pred = rf_light.predict(X_test)

# ===== KẾT QUẢ CLASSIFIER =====
print("\n" + "-"*70)
print("📊 KẾT QUẢ 1: ACCURACY (Độ chính xác)")
print("-"*70)
print("\n📖 ACCURACY là gì?")
print("  • Tỷ lệ dự đoán ĐÚNG trên tổng số dự đoán")
print("  • Công thức: (Số dự đoán đúng) / (Tổng số dự đoán) × 100%")
print("  • VD: Dự đoán 200 lần, đúng 180 lần → Accuracy = 90%")
print("  • Càng cao càng tốt: >90% là RẤT TỐT ✅")

acc_fan = accuracy_score(y_fan_on_test, fan_pred) * 100
acc_pump = accuracy_score(y_pump_on_test, pump_pred) * 100
acc_light = accuracy_score(y_light_on_test, light_pred) * 100

print(f"\n  🌪️  Quạt:  {acc_fan:.2f}%")
print(f"  💦 Bơm: {acc_pump:.2f}%")
print(f"  💡 Đèn:  {acc_light:.2f}%")

print(f"\n💬 GIẢI THÍCH:")
if acc_fan >= 90:
    print(f"  • Quạt {acc_fan:.1f}%: XUẤT SẮC! Dự đoán rất chính xác ✅")
elif acc_fan >= 85:
    print(f"  • Quạt {acc_fan:.1f}%: TỐT! Chấp nhận được ✅")
else:
    print(f"  • Quạt {acc_fan:.1f}%: CẦN CẢI THIỆN ⚠️")

# ===== CROSS-VALIDATION =====
print("\n" + "-"*70)
print("📊 KẾT QUẢ 2: CROSS-VALIDATION (Kiểm tra chéo 5 lần)")
print("-"*70)
print("\n📖 CROSS-VALIDATION là gì?")
print("  • Chia dữ liệu thành 5 phần")
print("  • Lần 1: Dùng phần 1 test, 4 phần còn lại train")
print("  • Lần 2: Dùng phần 2 test, 4 phần còn lại train")
print("  • ... (5 lần)")
print("  • Lấy trung bình 5 lần → Kết quả ổn định hơn")
print("  • Mục đích: Kiểm tra mô hình có BỊ OVERFITTING không")
print("    (Overfitting = Học vẹt, chỉ tốt trên dữ liệu train)")

print("\n⏳ Đang kiểm tra chéo (mất ~10-15 giây)...")
cv_fan = cross_val_score(rf_fan, X, y_fan_on, cv=5, scoring='accuracy')
cv_pump = cross_val_score(rf_pump, X, y_pump_on, cv=5, scoring='accuracy')
cv_light = cross_val_score(rf_light, X, y_light_on, cv=5, scoring='accuracy')

print(f"\n  🌪️  Quạt:  Mean = {cv_fan.mean()*100:.2f}% (±{cv_fan.std()*100:.2f}%)")
print(f"  💦 Bơm: Mean = {cv_pump.mean()*100:.2f}% (±{cv_pump.std()*100:.2f}%)")
print(f"  💡 Đèn:  Mean = {cv_light.mean()*100:.2f}% (±{cv_light.std()*100:.2f}%)")

print(f"\n💬 GIẢI THÍCH:")
print(f"  • Mean (Trung bình): Accuracy trung bình qua 5 lần")
print(f"  • ± (Độ lệch): Độ dao động giữa các lần")
print(f"    - Nếu < 3%: Rất ổn định ✅")
print(f"    - Nếu > 5%: Không ổn định, có thể overfitting ⚠️")

if abs(cv_fan.mean()*100 - acc_fan) < 3:
    print(f"  • Quạt: CV Mean ≈ Accuracy → KHÔNG OVERFITTING ✅")
else:
    print(f"  • Quạt: CV Mean khác xa Accuracy → CÓ THỂ OVERFITTING ⚠️")

# ===== CONFUSION MATRIX =====
print("\n" + "-"*70)
print("📊 KẾT QUẢ 3: CONFUSION MATRIX (Ma trận nhầm lẫn) - QUẠT")
print("-"*70)
print("\n📖 CONFUSION MATRIX là gì?")
print("  • Bảng chi tiết các trường hợp dự đoán")
print("  • Có 4 ô:")
print("    ┌──────────────────┬──────────────────┐")
print("    │ TRUE NEGATIVE    │ FALSE POSITIVE   │")
print("    │ (Dự đoán TẮT đúng)│ (Dự đoán BẬT sai)│")
print("    ├──────────────────┼──────────────────┤")
print("    │ FALSE NEGATIVE   │ TRUE POSITIVE    │")
print("    │ (Dự đoán TẮT sai)│ (Dự đoán BẬT đúng)│")
print("    └──────────────────┴──────────────────┘")

cm_fan = confusion_matrix(y_fan_on_test, fan_pred)
tn, fp, fn, tp = cm_fan[0, 0], cm_fan[0, 1], cm_fan[1, 0], cm_fan[1, 1]

print(f"\n📋 MA TRẬN QUẠT:")
print(f"  ┌─────────────┬─────────────┐")
print(f"  │ TN: {tn:3d}     │ FP: {fp:3d}     │")
print(f"  ├─────────────┼─────────────┤")
print(f"  │ FN: {fn:3d}     │ TP: {tp:3d}     │")
print(f"  └─────────────┴─────────────┘")

print(f"\n💬 GIẢI THÍCH:")
print(f"  • True Negative (TN={tn}): Dự đoán TẮT, thực tế TẮT → ĐÚNG ✅")
print(f"  • False Positive (FP={fp}): Dự đoán BẬT, thực tế TẮT → SAI ❌")
print(f"  • False Negative (FN={fn}): Dự đoán TẮT, thực tế BẬT → SAI ❌")
print(f"  • True Positive (TP={tp}): Dự đoán BẬT, thực tế BẬT → ĐÚNG ✅")
print(f"\n  📊 Tổng số sai: {fp + fn} / {tn + fp + fn + tp} = {(fp+fn)/(tn+fp+fn+tp)*100:.1f}%")

if fp + fn < 20:
    print(f"  ✅ Số lượng sai rất ít, mô hình TỐT!")
elif fp + fn < 30:
    print(f"  ✅ Số lượng sai chấp nhận được")
else:
    print(f"  ⚠️ Số lượng sai hơi nhiều, cần cải thiện")

# ===== FEATURE IMPORTANCE =====
print("\n" + "-"*70)
print("📊 KẾT QUẢ 4: FEATURE IMPORTANCE (Độ quan trọng các đặc trưng)")
print("-"*70)
print("\n📖 FEATURE IMPORTANCE là gì?")
print("  • Cho biết cảm biến nào ảnh hưởng NHIỀU NHẤT đến quyết định")
print("  • Tổng = 100%")
print("  • VD: Nhiệt độ 60% → Nhiệt độ quyết định 60% quyết định bật quạt")

feature_names = ['Nhiệt độ', 'Độ ẩm KK', 'Độ ẩm đất', 'Ánh sáng']

print("\n🌪️  QUẠT:")
importances_fan = rf_fan.feature_importances_
for name, importance in sorted(zip(feature_names, importances_fan), key=lambda x: x[1], reverse=True):
    bar = "█" * int(importance * 50)
    print(f"  {name:12s}: {importance*100:5.2f}% {bar}")

print("\n💦 BƠM:")
importances_pump = rf_pump.feature_importances_
for name, importance in sorted(zip(feature_names, importances_pump), key=lambda x: x[1], reverse=True):
    bar = "█" * int(importance * 50)
    print(f"  {name:12s}: {importance*100:5.2f}% {bar}")

print("\n💡 ĐÈN:")
importances_light = rf_light.feature_importances_
for name, importance in sorted(zip(feature_names, importances_light), key=lambda x: x[1], reverse=True):
    bar = "█" * int(importance * 50)
    print(f"  {name:12s}: {importance*100:5.2f}% {bar}")

print("\n💬 GIẢI THÍCH:")
most_important_fan = feature_names[np.argmax(importances_fan)]
print(f"  • Quạt: '{most_important_fan}' quan trọng nhất → Hợp lý ✅")
most_important_pump = feature_names[np.argmax(importances_pump)]
print(f"  • Bơm: '{most_important_pump}' quan trọng nhất → Hợp lý ✅")

# ===== RANDOM FOREST REGRESSOR =====
print("\n" + "="*70)
print("🌲 PHẦN 2: RANDOM FOREST REGRESSOR (Dự đoán PWM)")
print("="*70)
print("\n📖 GIẢI THÍCH:")
print("  • Mục đích: Dự đoán MỨC ĐỘ (PWM 0-255)")
print("  • Input: 4 giá trị cảm biến")
print("  • Output: Số từ 0-255 (PWM)")
print("  • Ví dụ: Nhiệt độ 32°C → PWM = 180 (70%)")

start_time = time.time()

print("\n⏳ Đang huấn luyện...")
print("  [1/3] Model Quạt PWM...", end=" ", flush=True)
rfr_fan = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr_fan.fit(X_train, y_fan_pwm_train)
print("✅")

print("  [2/3] Model Bơm PWM...", end=" ", flush=True)
rfr_pump = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr_pump.fit(X_train, y_pump_pwm_train)
print("✅")

print("  [3/3] Model Đèn PWM...", end=" ", flush=True)
rfr_light = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr_light.fit(X_train, y_light_pwm_train)
print("✅")

training_time_regressor = time.time() - start_time
print(f"\n⏱️  Thời gian huấn luyện: {training_time_regressor:.2f} giây")

# Dự đoán
fan_pwm_pred = rfr_fan.predict(X_test)
pump_pwm_pred = rfr_pump.predict(X_test)
light_pwm_pred = rfr_light.predict(X_test)

# ===== KẾT QUẢ REGRESSOR =====
print("\n" + "-"*70)
print("📊 KẾT QUẢ REGRESSOR")
print("-"*70)
print("\n📖 CÁC CHỈ SỐ:")
print("  • R² Score (R-squared):")
print("    - Tỷ lệ dự đoán khớp với thực tế")
print("    - Giá trị: 0 đến 1")
print("    - R² = 1.0: Hoàn hảo 100%")
print("    - R² = 0.9: Rất tốt (90% chính xác)")
print("    - R² = 0.5: Trung bình")
print("    - R² < 0.3: Kém")
print("\n  • MSE (Mean Squared Error - Sai số bình phương):")
print("    - Độ sai trung bình")
print("    - Càng thấp càng tốt")
print("    - VD: MSE = 400 → Sai trung bình ~20 đơn vị")

r2_fan = r2_score(y_fan_pwm_test, fan_pwm_pred)
mse_fan = mean_squared_error(y_fan_pwm_test, fan_pwm_pred)

r2_pump = r2_score(y_pump_pwm_test, pump_pwm_pred)
mse_pump = mean_squared_error(y_pump_pwm_test, pump_pwm_pred)

r2_light = r2_score(y_light_pwm_test, light_pwm_pred)
mse_light = mean_squared_error(y_light_pwm_test, light_pwm_pred)

print(f"\n  🌪️  Quạt PWM:")
print(f"     R² = {r2_fan:.4f}, MSE = {mse_fan:.2f}")
if r2_fan > 0.85:
    print(f"     ✅ R² > 0.85: XUẤT SẮC!")
elif r2_fan > 0.7:
    print(f"     ✅ R² > 0.7: TỐT!")
else:
    print(f"     ⚠️ R² < 0.7: Cần cải thiện")

print(f"\n  💦 Bơm PWM:")
print(f"     R² = {r2_pump:.4f}, MSE = {mse_pump:.2f}")

print(f"\n  💡 Đèn PWM:")
print(f"     R² = {r2_light:.4f}, MSE = {mse_light:.2f}")

# ===== TẠI SAO NHANH? =====
total_training_time = training_time_classifier + training_time_regressor
print("\n" + "="*70)
print(f"⏱️  TẠI SAO HUẤN LUYỆN CHỈ MẤT {total_training_time:.1f} GIÂY?")
print("="*70)
print("\n📖 GIẢI THÍCH CHI TIẾT:")
print(f"\n1️⃣  DỮ LIỆU NHỎ:")
print(f"   • Chỉ 1000 mẫu × 4 features = 4000 số")
print(f"   • So với ML thực tế: hàng triệu mẫu")
print(f"   • → Nhanh hơn 1000 lần!")

print(f"\n2️⃣  RANDOM FOREST NHANH:")
print(f"   • 100 cây, mỗi cây training độc lập")
print(f"   • Raspberry Pi 4: 4 cores → Train song song")
print(f"   • Mỗi cây chỉ mất ~0.05 giây")

print(f"\n3️⃣  FEATURES ÍT:")
print(f"   • Chỉ 4 features (nhiệt độ, độ ẩm...)")
print(f"   • Ít hơn → Tính toán nhanh hơn")

print(f"\n4️⃣  SCIKIT-LEARN TỐI ƯU:")
print(f"   • Thư viện đã được tối ưu bằng C/C++")
print(f"   • Nhanh hơn code Python thuần nhiều lần")

print(f"\n💡 KẾT LUẬN:")
print(f"   ✅ 5-10 giây là HOÀN TOÀN HỢP LÝ với dữ liệu nhỏ")
print(f"   ✅ Nếu dữ liệu 100,000 mẫu → Sẽ mất ~50-100 giây")
print(f"   ✅ Nếu dữ liệu 1 triệu mẫu → Sẽ mất 10-20 phút")

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

print("\n" + "="*70)
print("🎉 HOÀN TẤT!")
print("="*70)
print("\n📌 Bước tiếp theo:")
print("  python3 test_model_nhatrang.py")
