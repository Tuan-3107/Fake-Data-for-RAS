#!/usr/bin/env python3
"""
Tạo dữ liệu giả mô phỏng cảm biến nhà kính
Không cần ESP32 hay cảm biến thật
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("🔄 Bắt đầu tạo dữ liệu giả...")

# Seed để kết quả giống nhau
np.random.seed(42)

# Tạo 1000 mẫu dữ liệu (giả lập 1 tháng, mỗi 45 phút)
num_samples = 1000
data = []

for i in range(num_samples):
    # Thời gian
    timestamp = datetime.now() - timedelta(minutes=45*i)
    hour = timestamp.hour
    
    # Chu kỳ ngày/đêm
    is_day = 6 <= hour <= 18
    
    # NHIỆT ĐỘ (20-38°C)
    if is_day:
        temp = np.random.normal(32, 4)  # Ban ngày nóng hơn
    else:
        temp = np.random.normal(24, 3)  # Ban đêm mát hơn
    temp = np.clip(temp, 20, 38)
    
    # ĐỘ ẨM KHÔNG KHÍ (30-90%)
    humidity_air = 95 - temp + np.random.normal(0, 8)
    humidity_air = np.clip(humidity_air, 30, 90)
    
    # ĐỘ ẨM ĐẤT (20-80%)
    humidity_soil = np.random.uniform(25, 75)
    
    # ÁNH SÁNG (0-1000 Lux)
    if is_day:
        light = np.random.normal(750, 200)
    else:
        light = np.random.normal(30, 20)
    light = np.clip(light, 0, 1000)
    
    # DỰ BÁO THỜI TIẾT (giả lập)
    forecast_temp = temp + np.random.uniform(-3, 3)
    rain_prob = np.random.uniform(0, 100)
    
    # === LOGIC QUYẾT ĐỊNH (Ground Truth) ===
    
    # QUẠT: Bật khi nhiệt độ > 29°C
    fan_on = 1 if temp > 29 else 0
    fan_pwm = int(np.clip((temp - 20) * 13, 0, 255)) if fan_on else 0
    
    # BƠM: Bật khi độ ẩm đất < 45% VÀ không có mưa
    pump_on = 1 if (humidity_soil < 45 and rain_prob < 70) else 0
    pump_pwm = int(np.clip((50 - humidity_soil) * 10, 0, 255)) if pump_on else 0
    
    # ĐÈN: Bật khi ánh sáng < 250 Lux VÀ là giờ ban ngày
    light_on = 1 if (light < 250 and is_day) else 0
    light_pwm = int(np.clip((300 - light) * 1.0, 0, 255)) if light_on else 0
    
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
    
    # Hiển thị tiến trình
    if (i + 1) % 100 == 0:
        print(f"  ✓ Đã tạo {i+1}/{num_samples} mẫu...")

# Chuyển thành DataFrame
df = pd.DataFrame(data)

# Lưu file
df.to_csv('greenhouse_data.csv', index=False)

print(f"\n✅ Hoàn thành! Đã tạo {len(df)} mẫu dữ liệu")
print(f"📁 File: greenhouse_data.csv")
print(f"\n📊 Thống kê:")
print(f"  • Nhiệt độ: {df['temp'].min():.1f}°C - {df['temp'].max():.1f}°C")
print(f"  • Độ ẩm đất: {df['humidity_soil'].min():.1f}% - {df['humidity_soil'].max():.1f}%")
print(f"  • Quạt bật: {df['fan_on'].sum()} lần ({df['fan_on'].mean()*100:.1f}%)")
print(f"  • Bơm bật: {df['pump_on'].sum()} lần ({df['pump_on'].mean()*100:.1f}%)")
print(f"  • Đèn bật: {df['light_on'].sum()} lần ({df['light_on'].mean()*100:.1f}%)")

print("\n🎯 Xem 5 dòng đầu:")
print(df.head())
