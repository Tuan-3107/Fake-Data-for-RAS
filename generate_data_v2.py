#!/usr/bin/env python3
"""
Tạo dữ liệu THỰC TẾ - KHÔNG CẦN API THỜI TIẾT
Chỉ dùng 4 cảm biến: Nhiệt độ, Độ ẩm KK, Độ ẩm đất, Ánh sáng
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🔄 Bắt đầu tạo dữ liệu (KHÔNG dùng API thời tiết)...")

np.random.seed(42)
num_samples = 1000
data = []

for i in range(num_samples):
    timestamp = datetime.now() - timedelta(minutes=45*i)
    hour = timestamp.hour
    is_day = 6 <= hour <= 18
    
    # === 4 CẢM BIẾN CHÍNH ===
    
    # 1. NHIỆT ĐỘ (18-40°C)
    if is_day:
        temp = np.random.normal(32, 5)
    else:
        temp = np.random.normal(24, 4)
    temp = np.clip(temp, 18, 40)
    
    # 2. ĐỘ ẨM KHÔNG KHÍ (25-95%)
    # Tỷ lệ nghịch với nhiệt độ
    humidity_air = 100 - temp + np.random.normal(0, 10)
    humidity_air = np.clip(humidity_air, 25, 95)
    
    # 3. ĐỘ ẨM ĐẤT (15-85%)
    # Giảm dần theo thời gian, tăng khi tưới
    if i == 0:
        humidity_soil = 60
    else:
        humidity_soil = data[-1]['humidity_soil'] - np.random.uniform(0.3, 2.5)
        if data[-1]['pump_on'] == 1:
            humidity_soil += np.random.uniform(8, 18)
    humidity_soil = np.clip(humidity_soil, 15, 85)
    
    # 4. ÁNH SÁNG (0-1000 Lux)
    if is_day:
        cloud_factor = np.random.uniform(0.3, 1.0)
        light = np.random.normal(750 * cloud_factor, 200)
    else:
        light = np.random.normal(30, 25)
    light = np.clip(light, 0, 1000)
    
    # === LOGIC QUYẾT ĐỊNH - DỰA VÀO 4 CẢM BIẾN ===
    
    # QUẠT: Nhiệt độ + Độ ẩm KK
    fan_score = 0
    if temp > 28: fan_score += 3
    if temp > 32: fan_score += 2
    if humidity_air < 50: fan_score += 1
    if temp > 35: fan_score += 1
    
    # Nhiễu 8%
    fan_noise = np.random.choice([0, 1], p=[0.92, 0.08])
    
    if fan_score >= 4:
        fan_on = 1
    elif fan_score == 3:
        fan_on = np.random.choice([0, 1], p=[0.3, 0.7])
    else:
        fan_on = 0
    
    if fan_noise:
        fan_on = 1 - fan_on
    
    if fan_on:
        fan_pwm = int(np.clip((temp - 20) * 12 + np.random.uniform(-20, 20), 50, 255))
    else:
        fan_pwm = 0
    
    # BƠM: Độ ẩm đất + Nhiệt độ + Độ ẩm KK
    pump_score = 0
    if humidity_soil < 35: pump_score += 4
    if humidity_soil < 45: pump_score += 2
    if temp > 30: pump_score += 1
    if humidity_air < 60: pump_score += 1
    
    # Nhiễu 10%
    pump_noise = np.random.choice([0, 1], p=[0.90, 0.10])
    
    if pump_score >= 5:
        pump_on = 1
    elif pump_score >= 3:
        pump_on = np.random.choice([0, 1], p=[0.4, 0.6])
    else:
        pump_on = 0
    
    if pump_noise:
        pump_on = 1 - pump_on
    
    if pump_on:
        pump_pwm = int(np.clip((55 - humidity_soil) * 8 + np.random.uniform(-15, 15), 30, 255))
    else:
        pump_pwm = 0
    
    # ĐÈN: Ánh sáng + Giờ trong ngày
    light_score = 0
    if light < 300 and is_day: light_score += 3
    if light < 200 and is_day: light_score += 2
    if light < 100 and is_day: light_score += 1
    if 6 <= hour <= 8 or 16 <= hour <= 18: light_score += 1
    
    # Nhiễu 5%
    light_noise = np.random.choice([0, 1], p=[0.95, 0.05])
    
    if light_score >= 4:
        light_on = 1
    elif light_score == 3:
        light_on = np.random.choice([0, 1], p=[0.5, 0.5])
    else:
        light_on = 0
    
    if light_noise:
        light_on = 1 - light_on
    
    if light_on:
        light_pwm = int(np.clip((350 - light) * 0.9 + np.random.uniform(-10, 10), 20, 255))
    else:
        light_pwm = 0
    
    # Lưu dữ liệu (CHỈ 4 CẢM BIẾN)
    data.append({
        'temp': round(temp, 1),
        'humidity_air': round(humidity_air, 1),
        'humidity_soil': round(humidity_soil, 1),
        'light': round(light, 1),
        'fan_on': fan_on,
        'fan_pwm': fan_pwm,
        'pump_on': pump_on,
        'pump_pwm': pump_pwm,
        'light_on': light_on,
        'light_pwm': light_pwm
    })
    
    if (i + 1) % 100 == 0:
        print(f"  ✓ Đã tạo {i+1}/{num_samples} mẫu...")

df = pd.DataFrame(data)
df.to_csv('greenhouse_data_v2.csv', index=False)

print(f"\n✅ Hoàn thành! Đã tạo {len(df)} mẫu")
print(f"📁 File: greenhouse_data_v2.csv")
print(f"\n📊 Thống kê:")
print(f"  • Nhiệt độ: {df['temp'].min():.1f}°C - {df['temp'].max():.1f}°C")
print(f"  • Độ ẩm KK: {df['humidity_air'].min():.1f}% - {df['humidity_air'].max():.1f}%")
print(f"  • Độ ẩm đất: {df['humidity_soil'].min():.1f}% - {df['humidity_soil'].max():.1f}%")
print(f"  • Ánh sáng: {df['light'].min():.1f} - {df['light'].max():.1f} Lux")
print(f"\n🎯 Quyết định:")
print(f"  • Quạt bật: {df['fan_on'].sum()} lần ({df['fan_on'].mean()*100:.1f}%)")
print(f"  • Bơm bật: {df['pump_on'].sum()} lần ({df['pump_on'].mean()*100:.1f}%)")
print(f"  • Đèn bật: {df['light_on'].sum()} lần ({df['light_on'].mean()*100:.1f}%)")

print("\n🎯 Xem 5 dòng đầu:")
print(df.head())

print("\n💡 ĐẶC ĐIỂM:")
print("  ✅ Chỉ dùng 4 cảm biến (không cần API thời tiết)")
print("  ✅ Nhiễu 5-10%")
print("  ✅ Logic phức tạp")
print("  ✅ Accuracy dự kiến: 85-92%")
