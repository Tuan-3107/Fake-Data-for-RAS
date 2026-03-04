#!/usr/bin/env python3
"""
Tạo 10,000 mẫu dữ liệu cho khí hậu NHA TRANG
Dữ liệu thực tế, phù hợp triển khai ML
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("="*70)
print("🌴 TẠO 10,000 MẪU DỮ LIỆU - KHÍ HẬU NHA TRANG")
print("="*70)

print("\n📍 Đặc điểm khí hậu Nha Trang:")
print("  • Nhiệt độ: 20-36°C (trung bình 27°C)")
print("  • Độ ẩm: 70-95% (khí hậu ẩm)")
print("  • Mùa khô: Tháng 1-8")
print("  • Mùa mưa: Tháng 9-12")

np.random.seed(42)
num_samples = 10000  # 10,000 mẫu
data = []

print(f"\n🔄 Đang tạo {num_samples:,} mẫu dữ liệu...")
print("   (Có thể mất ~30-60 giây)")

for i in range(num_samples):
    timestamp = datetime.now() - timedelta(minutes=45*i)
    hour = timestamp.hour
    month = timestamp.month
    is_day = 6 <= hour <= 18
    
    # ===== NHIỆT ĐỘ =====
    if month in [5, 6, 7, 8]:
        base_temp_day = 32
        base_temp_night = 26
    elif month in [12, 1, 2]:
        base_temp_day = 26
        base_temp_night = 22
    else:
        base_temp_day = 29
        base_temp_night = 24
    
    if is_day:
        if 11 <= hour <= 14:
            temp = np.random.normal(base_temp_day + 1, 1.5)
        elif 6 <= hour <= 9:
            temp = np.random.normal(base_temp_day - 3, 1.2)
        else:
            temp = np.random.normal(base_temp_day - 1, 1.3)
    else:
        if 0 <= hour <= 4:
            temp = np.random.normal(base_temp_night - 1, 1.0)
        else:
            temp = np.random.normal(base_temp_night, 1.2)
    
    temp = np.clip(temp, 20, 36)
    
    # ===== ĐỘ ẨM KHÔNG KHÍ =====
    if month in [9, 10, 11, 12]:
        base_humidity = 85
    else:
        base_humidity = 78
    
    humidity_air = base_humidity + (28 - temp) * 0.8 + np.random.normal(0, 3)
    if not is_day:
        humidity_air += 5
    humidity_air = np.clip(humidity_air, 70, 95)
    
    # ===== ĐỘ ẨM ĐẤT =====
    if i == 0:
        humidity_soil = 55
    else:
        evaporation_rate = (temp - 20) * 0.08 + (100 - humidity_air) * 0.02
        humidity_soil = data[-1]['humidity_soil'] - evaporation_rate
        
        if data[-1]['pump_on'] == 1:
            water_added = data[-1]['pump_pwm'] / 255 * 15
            humidity_soil += water_added
        
        if month in [9, 10, 11, 12]:
            humidity_soil += np.random.uniform(0, 1.5)
    
    humidity_soil = np.clip(humidity_soil, 20, 85)
    
    # ===== ÁNH SÁNG =====
    if is_day:
        if 11 <= hour <= 14:
            cloud_factor = np.random.uniform(0.7, 1.0)
            light = np.random.normal(900 * cloud_factor, 100)
        elif 6 <= hour <= 9 or 15 <= hour <= 18:
            cloud_factor = np.random.uniform(0.5, 0.9)
            light = np.random.normal(600 * cloud_factor, 150)
        else:
            light = np.random.normal(750, 150)
    else:
        light = np.random.normal(10, 8)
    
    light = np.clip(light, 0, 1000)
    
    # ===== LOGIC QUYẾT ĐỊNH =====
    
    # QUẠT
    fan_score = 0
    if temp > 27: fan_score += 2
    if temp > 30: fan_score += 3
    if temp > 33: fan_score += 2
    if humidity_air < 75: fan_score += 1
    
    fan_noise = np.random.choice([0, 1], p=[0.92, 0.08])
    
    if fan_score >= 5:
        fan_on = 1
    elif fan_score >= 3:
        fan_on = np.random.choice([0, 1], p=[0.3, 0.7])
    else:
        fan_on = 0
    
    if fan_noise:
        fan_on = 1 - fan_on
    
    if fan_on:
        fan_pwm = int(np.clip((temp - 22) * 15 + np.random.uniform(-20, 20), 50, 255))
    else:
        fan_pwm = 0
    
    # BƠM
    pump_score = 0
    if humidity_soil < 40: pump_score += 4
    if humidity_soil < 50: pump_score += 2
    if temp > 30: pump_score += 1
    if humidity_air < 75: pump_score += 1
    
    if month in [9, 10, 11, 12]:
        pump_score -= 1
    
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
        pump_pwm = int(np.clip((60 - humidity_soil) * 7 + np.random.uniform(-15, 15), 30, 255))
    else:
        pump_pwm = 0
    
    # ĐÈN
    light_score = 0
    if light < 400 and is_day: light_score += 3
    if light < 250 and is_day: light_score += 2
    if light < 150 and is_day: light_score += 1
    if 6 <= hour <= 8 or 16 <= hour <= 18: light_score += 1
    
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
        light_pwm = int(np.clip((450 - light) * 0.7 + np.random.uniform(-10, 10), 20, 255))
    else:
        light_pwm = 0
    
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
    
    if (i + 1) % 1000 == 0:
        print(f"  ✓ Đã tạo {i+1:,}/{num_samples:,} mẫu ({(i+1)/num_samples*100:.0f}%)...")

df = pd.DataFrame(data)
df.to_csv('greenhouse_data_nhatrang_10k.csv', index=False)

print(f"\n" + "="*70)
print(f"✅ HOÀN THÀNH!")
print(f"="*70)
print(f"\n📁 File: greenhouse_data_nhatrang_10k.csv")
print(f"📊 Tổng số mẫu: {len(df):,}")

print(f"\n📈 THỐNG KÊ:")
print(f"\n🌡️  Nhiệt độ:")
print(f"  • TB: {df['temp'].mean():.1f}°C | Min: {df['temp'].min():.1f}°C | Max: {df['temp'].max():.1f}°C")

print(f"\n💧 Độ ẩm KK:")
print(f"  • TB: {df['humidity_air'].mean():.1f}% | Min: {df['humidity_air'].min():.1f}% | Max: {df['humidity_air'].max():.1f}%")

print(f"\n🌱 Độ ẩm đất:")
print(f"  • TB: {df['humidity_soil'].mean():.1f}% | Min: {df['humidity_soil'].min():.1f}% | Max: {df['humidity_soil'].max():.1f}%")

print(f"\n☀️  Ánh sáng:")
print(f"  • TB: {df['light'].mean():.1f} Lux | Max: {df['light'].max():.1f} Lux")

print(f"\n🎯 Quyết định:")
print(f"  🌪️  Quạt: {df['fan_on'].sum():,} lần bật ({df['fan_on'].mean()*100:.1f}%)")
print(f"  💦 Bơm: {df['pump_on'].sum():,} lần bật ({df['pump_on'].mean()*100:.1f}%)")
print(f"  💡 Đèn: {df['light_on'].sum():,} lần bật ({df['light_on'].mean()*100:.1f}%)")

print(f"\n💡 ĐẶC ĐIỂM:")
print(f"  ✅ 10,000 mẫu - đủ lớn cho ML thực tế")
print(f"  ✅ Dữ liệu phù hợp khí hậu Nha Trang")
print(f"  ✅ Có nhiễu 5-10% (thực tế)")

print(f"\n" + "="*70)
print(f"📌 LỆNH TIẾP THEO:")
print(f"="*70)
print(f"\npython3 train_model_nhatrang.py")
