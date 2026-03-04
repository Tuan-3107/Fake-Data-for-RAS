#!/usr/bin/env python3
"""
Tạo dữ liệu THỰC TẾ cho khí hậu NHA TRANG
Dựa trên số liệu khí hậu thực tế của Nha Trang, Khánh Hòa
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("="*70)
print("🌴 TẠO DỮ LIỆU CHO KHÍ HẬU NHA TRANG")
print("="*70)

# ===== THÔNG SỐ KHÍ HẬU NHA TRANG =====
print("\n📍 Đặc điểm khí hậu Nha Trang:")
print("  • Nhiệt độ trung bình: 23-28°C")
print("  • Nhiệt độ cao nhất: 32-35°C (tháng 5-8)")
print("  • Nhiệt độ thấp nhất: 20-22°C (tháng 12-2)")
print("  • Độ ẩm trung bình: 79-82%")
print("  • Mùa khô: Tháng 1-8")
print("  • Mùa mưa: Tháng 9-12")

np.random.seed(42)
num_samples = 1000
data = []

print(f"\n🔄 Đang tạo {num_samples} mẫu dữ liệu...")

for i in range(num_samples):
    timestamp = datetime.now() - timedelta(minutes=45*i)
    hour = timestamp.hour
    month = timestamp.month
    
    # Chu kỳ ngày/đêm
    is_day = 6 <= hour <= 18
    
    # ===== 1. NHIỆT ĐỘ (Theo khí hậu Nha Trang) =====
    
    # Nhiệt độ cơ sở theo tháng
    if month in [5, 6, 7, 8]:  # Mùa nóng
        base_temp_day = 32
        base_temp_night = 26
    elif month in [12, 1, 2]:  # Mùa mát
        base_temp_day = 26
        base_temp_night = 22
    else:  # Tháng chuyển tiếp
        base_temp_day = 29
        base_temp_night = 24
    
    # Nhiệt độ thay đổi theo giờ trong ngày
    if is_day:
        if 11 <= hour <= 14:  # Giữa trưa - nóng nhất
            temp = np.random.normal(base_temp_day + 1, 1.5)
        elif 6 <= hour <= 9:  # Sáng sớm - mát
            temp = np.random.normal(base_temp_day - 3, 1.2)
        else:  # Chiều
            temp = np.random.normal(base_temp_day - 1, 1.3)
    else:  # Ban đêm
        if 0 <= hour <= 4:  # Nửa đêm - lạnh nhất
            temp = np.random.normal(base_temp_night - 1, 1.0)
        else:  # Đêm sớm/khuya
            temp = np.random.normal(base_temp_night, 1.2)
    
    # Giới hạn nhiệt độ hợp lý cho Nha Trang
    temp = np.clip(temp, 20, 36)
    
    # ===== 2. ĐỘ ẨM KHÔNG KHÍ (Nha Trang: 75-95%) =====
    
    # Độ ẩm cao vào mùa mưa, thấp hơn vào mùa khô
    if month in [9, 10, 11, 12]:  # Mùa mưa - ẩm hơn
        base_humidity = 85
    else:  # Mùa khô
        base_humidity = 78
    
    # Độ ẩm thay đổi theo nhiệt độ (nghịch đảo)
    humidity_air = base_humidity + (28 - temp) * 0.8 + np.random.normal(0, 3)
    
    # Ban đêm ẩm hơn ban ngày
    if not is_day:
        humidity_air += 5
    
    humidity_air = np.clip(humidity_air, 70, 95)
    
    # ===== 3. ĐỘ ẨM ĐẤT (20-80%) =====
    
    # Độ ẩm đất giảm dần theo thời gian, tăng khi tưới
    if i == 0:
        humidity_soil = 55
    else:
        # Tốc độ bay hơi phụ thuộc nhiệt độ và độ ẩm không khí
        evaporation_rate = (temp - 20) * 0.08 + (100 - humidity_air) * 0.02
        humidity_soil = data[-1]['humidity_soil'] - evaporation_rate
        
        # Nếu đã tưới thì độ ẩm tăng
        if data[-1]['pump_on'] == 1:
            water_added = data[-1]['pump_pwm'] / 255 * 15  # Tối đa +15%
            humidity_soil += water_added
        
        # Mùa mưa - độ ẩm đất cao hơn
        if month in [9, 10, 11, 12]:
            humidity_soil += np.random.uniform(0, 1.5)
    
    humidity_soil = np.clip(humidity_soil, 20, 85)
    
    # ===== 4. ÁNH SÁNG (0-1000 Lux) =====
    
    if is_day:
        if 11 <= hour <= 14:  # Giữa trưa - sáng nhất
            # Nha Trang nhiều nắng, ít mây
            cloud_factor = np.random.uniform(0.7, 1.0)
            light = np.random.normal(900 * cloud_factor, 100)
        elif 6 <= hour <= 9 or 15 <= hour <= 18:  # Sáng/chiều
            cloud_factor = np.random.uniform(0.5, 0.9)
            light = np.random.normal(600 * cloud_factor, 150)
        else:
            light = np.random.normal(750, 150)
    else:  # Ban đêm
        light = np.random.normal(10, 8)
    
    light = np.clip(light, 0, 1000)
    
    # ===== LOGIC QUYẾT ĐỊNH - PHÙ HỢP NHA TRANG =====
    
    # QUẠT: Nha Trang nóng ẩm, cần quạt nhiều
    fan_score = 0
    if temp > 27: fan_score += 2  # Ngưỡng thấp hơn vì Nha Trang nóng
    if temp > 30: fan_score += 3
    if temp > 33: fan_score += 2
    if humidity_air < 75: fan_score += 1  # Khô thì cần thông gió
    
    # Nhiễu 8%
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
    
    # BƠM: Nha Trang nóng, cần tưới nhiều
    pump_score = 0
    if humidity_soil < 40: pump_score += 4
    if humidity_soil < 50: pump_score += 2
    if temp > 30: pump_score += 1  # Nóng thì cần tưới thêm
    if humidity_air < 75: pump_score += 1  # Khô thì tưới
    
    # Mùa mưa giảm nhu cầu tưới
    if month in [9, 10, 11, 12]:
        pump_score -= 1
    
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
        pump_pwm = int(np.clip((60 - humidity_soil) * 7 + np.random.uniform(-15, 15), 30, 255))
    else:
        pump_pwm = 0
    
    # ĐÈN: Nha Trang nhiều nắng, ít cần đèn bổ sung
    light_score = 0
    if light < 400 and is_day: light_score += 3
    if light < 250 and is_day: light_score += 2
    if light < 150 and is_day: light_score += 1
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
        light_pwm = int(np.clip((450 - light) * 0.7 + np.random.uniform(-10, 10), 20, 255))
    else:
        light_pwm = 0
    
    # Lưu dữ liệu
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
df.to_csv('greenhouse_data_nhatrang.csv', index=False)

print(f"\n" + "="*70)
print(f"✅ HOÀN THÀNH!")
print(f"="*70)
print(f"\n📁 File: greenhouse_data_nhatrang.csv")
print(f"📊 Số mẫu: {len(df)}")

print(f"\n📈 THỐNG KÊ CHI TIẾT:")
print(f"\n🌡️  NHIỆT ĐỘ:")
print(f"  • Trung bình: {df['temp'].mean():.1f}°C")
print(f"  • Thấp nhất: {df['temp'].min():.1f}°C")
print(f"  • Cao nhất: {df['temp'].max():.1f}°C")
print(f"  • Độ lệch chuẩn: {df['temp'].std():.1f}°C")

print(f"\n💧 ĐỘ ẨM KHÔNG KHÍ:")
print(f"  • Trung bình: {df['humidity_air'].mean():.1f}%")
print(f"  • Thấp nhất: {df['humidity_air'].min():.1f}%")
print(f"  • Cao nhất: {df['humidity_air'].max():.1f}%")

print(f"\n🌱 ĐỘ ẨM ĐẤT:")
print(f"  • Trung bình: {df['humidity_soil'].mean():.1f}%")
print(f"  • Thấp nhất: {df['humidity_soil'].min():.1f}%")
print(f"  • Cao nhất: {df['humidity_soil'].max():.1f}%")

print(f"\n☀️  ÁNH SÁNG:")
print(f"  • Trung bình: {df['light'].mean():.1f} Lux")
print(f"  • Thấp nhất: {df['light'].min():.1f} Lux")
print(f"  • Cao nhất: {df['light'].max():.1f} Lux")

print(f"\n🎯 QUYẾT ĐỊNH:")
print(f"  🌪️  Quạt bật: {df['fan_on'].sum()} lần ({df['fan_on'].mean()*100:.1f}%)")
print(f"  💦 Bơm bật: {df['pump_on'].sum()} lần ({df['pump_on'].mean()*100:.1f}%)")
print(f"  💡 Đèn bật: {df['light_on'].sum()} lần ({df['light_on'].mean()*100:.1f}%)")

print(f"\n🎯 Xem 10 dòng đầu:")
print(df.head(10))

print(f"\n💡 ĐẶC ĐIỂM DỮ LIỆU:")
print(f"  ✅ Nhiệt độ: 20-36°C (phù hợp Nha Trang)")
print(f"  ✅ Độ ẩm KK: 70-95% (khí hậu ẩm)")
print(f"  ✅ Độ ẩm đất: Giảm theo nhiệt độ và bay hơi")
print(f"  ✅ Ánh sáng: Nha Trang nhiều nắng")
print(f"  ✅ Phân biệt mùa khô/mưa")
print(f"  ✅ Chu kỳ ngày/đêm rõ ràng")
