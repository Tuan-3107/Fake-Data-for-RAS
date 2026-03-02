#!/usr/bin/env python3
"""
Tạo dữ liệu THỰC TẾ hơn với nhiễu và logic phức tạp
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🔄 Bắt đầu tạo dữ liệu THỰC TẾ...")

np.random.seed(42)
num_samples = 1000
data = []

for i in range(num_samples):
    timestamp = datetime.now() - timedelta(minutes=45*i)
    hour = timestamp.hour
    is_day = 6 <= hour <= 18
    
    # NHIỆT ĐỘ (thêm biến động lớn hơn)
    if is_day:
        temp = np.random.normal(32, 5)  # Tăng độ lệch chuẩn
    else:
        temp = np.random.normal(24, 4)
    temp = np.clip(temp, 18, 40)  # Mở rộng phạm vi
    
    # ĐỘ ẨM KHÔNG KHÍ (có nhiễu)
    humidity_air = 95 - temp + np.random.normal(0, 10)  # Thêm nhiễu
    humidity_air = np.clip(humidity_air, 25, 95)
    
    # ĐỘ ẨM ĐẤT (biến động theo thời gian)
    if i == 0:
        humidity_soil = 60
    else:
        # Giảm dần + nhiễu
        humidity_soil = data[-1]['humidity_soil'] - np.random.uniform(0.3, 2.5)
        # Tăng lại nếu đã tưới
        if data[-1]['pump_on'] == 1:
            humidity_soil += np.random.uniform(8, 18)
    humidity_soil = np.clip(humidity_soil, 15, 85)
    
    # ÁNH SÁNG (có mây che)
    if is_day:
        # Có lúc có mây → ánh sáng giảm
        cloud_factor = np.random.uniform(0.3, 1.0)
        light = np.random.normal(750 * cloud_factor, 200)
    else:
        light = np.random.normal(30, 25)
    light = np.clip(light, 0, 1000)
    
    # DỰ BÁO
    forecast_temp = temp + np.random.uniform(-4, 4)
    rain_prob = np.random.uniform(0, 100)
    
    # ===== LOGIC THỰC TẾ - PHỨC TẠP HƠN =====
    
    # QUẠT: Nhiều yếu tố ảnh hưởng
    fan_score = 0
    if temp > 28: fan_score += 3
    if temp > 32: fan_score += 2
    if humidity_air < 50: fan_score += 1
    if forecast_temp > 33: fan_score += 1
    
    # Thêm NHIỄU: Đôi khi quyết định sai
    fan_noise = np.random.choice([0, 1], p=[0.92, 0.08])  # 8% sai quyết định
    
    if fan_score >= 3:
        fan_on = 1
    elif fan_score == 2:
        fan_on = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% bật
    else:
        fan_on = 0
    
    # Áp dụng nhiễu
    if fan_noise:
        fan_on = 1 - fan_on
    
    # PWM không hoàn toàn tuyến tính
    if fan_on:
        fan_pwm = int(np.clip((temp - 20) * 12 + np.random.uniform(-20, 20), 50, 255))
    else:
        fan_pwm = 0
    
    # BƠM: Logic phức tạp
    pump_score = 0
    if humidity_soil < 40: pump_score += 3
    if humidity_soil < 50: pump_score += 1
    if rain_prob < 50: pump_score += 1
    if temp > 30: pump_score += 1
    
    # Nhiễu
    pump_noise = np.random.choice([0, 1], p=[0.90, 0.10])  # 10% sai
    
    if pump_score >= 4:
        pump_on = 1
    elif pump_score == 3:
        pump_on = np.random.choice([0, 1], p=[0.4, 0.6])
    else:
        pump_on = 0
    
    if pump_noise:
        pump_on = 1 - pump_on
    
    if pump_on:
        pump_pwm = int(np.clip((55 - humidity_soil) * 8 + np.random.uniform(-15, 15), 30, 255))
    else:
        pump_pwm = 0
    
    # ĐÈN: Phụ thuộc vào mây + giờ trong ngày
    light_score = 0
    if light < 300 and is_day: light_score += 3
    if light < 200 and is_day: light_score += 2
    if 6 <= hour <= 8 or 16 <= hour <= 18: light_score += 1  # Sáng sớm/chiều tối
    
    # Nhiễu
    light_noise = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% sai
    
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
    
    # Lưu dữ liệu
    data.append({
        'temp': round(temp, 1),
        'humidity_air': round(humidity_air, 1),
        'humidity_soil': round(humidity_soil, 1),
        'light': round(light, 1),
        'forecast_temp': round(forecast_temp, 1),
        'rain_prob': round(rain_prob, 1),
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
df.to_csv('greenhouse_data_realistic.csv', index=False)

print(f"\n✅ Hoàn thành! Đã tạo {len(df)} mẫu THỰC TẾ")
print(f"📁 File: greenhouse_data_realistic.csv")
print(f"\n📊 Thống kê:")
print(f"  • Nhiệt độ: {df['temp'].min():.1f}°C - {df['temp'].max():.1f}°C")
print(f"  • Độ ẩm đất: {df['humidity_soil'].min():.1f}% - {df['humidity_soil'].max():.1f}%")
print(f"  • Quạt bật: {df['fan_on'].sum()} lần ({df['fan_on'].mean()*100:.1f}%)")
print(f"  • Bơm bật: {df['pump_on'].sum()} lần ({df['pump_on'].mean()*100:.1f}%)")
print(f"  • Đèn bật: {df['light_on'].sum()} lần ({df['light_on'].mean()*100:.1f}%)")

print("\n🎯 Xem 5 dòng đầu:")
print(df.head())

print("\n💡 LƯU Ý:")
print("  Dữ liệu này có thêm:")
print("  - Nhiễu (noise) 5-10%")
print("  - Logic phức tạp (nhiều điều kiện)")
print("  - Trường hợp mơ hồ")
print("  → Accuracy sẽ thực tế hơn: 85-95%")
