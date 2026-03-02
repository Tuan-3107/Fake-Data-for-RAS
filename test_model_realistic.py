#!/usr/bin/env python3
"""
Test mô hình THỰC TẾ với dữ liệu có nhiễu
"""

import numpy as np
import pickle
import time
from datetime import datetime

# Load models
print("📂 Đang load mô hình...")
try:
    rf_fan = pickle.load(open('rf_fan_realistic.pkl', 'rb'))
    rf_pump = pickle.load(open('rf_pump_realistic.pkl', 'rb'))
    rf_light = pickle.load(open('rf_light_realistic.pkl', 'rb'))
    
    lr_fan = pickle.load(open('lr_fan_pwm_realistic.pkl', 'rb'))
    lr_pump = pickle.load(open('lr_pump_pwm_realistic.pkl', 'rb'))
    lr_light = pickle.load(open('lr_light_pwm_realistic.pkl', 'rb'))
    
    print("✅ Đã load thành công!\n")
except FileNotFoundError:
    print("❌ Lỗi: Chưa có file mô hình!")
    print("💡 Hãy chạy: python3 train_model_realistic.py trước")
    exit()

def fake_sensor_realistic():
    """Tạo dữ liệu cảm biến giả - THỰC TẾ hơn"""
    hour = datetime.now().hour
    is_day = 6 <= hour <= 18
    
    # Nhiệt độ với biến động lớn
    temp = np.random.normal(32 if is_day else 24, 5)
    temp = np.clip(temp, 18, 40)
    
    # Độ ẩm không khí với nhiễu
    humidity_air = 95 - temp + np.random.normal(0, 10)
    humidity_air = np.clip(humidity_air, 25, 95)
    
    # Độ ẩm đất biến động
    humidity_soil = np.random.uniform(20, 80)
    
    # Ánh sáng (có lúc có mây)
    if is_day:
        cloud_factor = np.random.uniform(0.3, 1.0)
        light = np.random.normal(750 * cloud_factor, 200)
    else:
        light = np.random.normal(30, 25)
    light = np.clip(light, 0, 1000)
    
    # Dự báo
    forecast_temp = temp + np.random.uniform(-4, 4)
    rain_prob = np.random.uniform(0, 100)
    
    return {
        'temp': round(temp, 1),
        'humidity_air': round(humidity_air, 1),
        'humidity_soil': round(humidity_soil, 1),
        'light': round(light, 1),
        'forecast_temp': round(forecast_temp, 1),
        'rain_prob': round(rain_prob, 1)
    }

def predict(sensors):
    """Dự đoán hành động"""
    X = np.array([[
        sensors['temp'],
        sensors['humidity_air'],
        sensors['humidity_soil'],
        sensors['light'],
        sensors['forecast_temp'],
        sensors['rain_prob']
    ]])
    
    # Random Forest
    fan_on = rf_fan.predict(X)[0]
    pump_on = rf_pump.predict(X)[0]
    light_on = rf_light.predict(X)[0]
    
    # Linear Regression
    fan_pwm = int(np.clip(lr_fan.predict(X)[0], 0, 255))
    pump_pwm = int(np.clip(lr_pump.predict(X)[0], 0, 255))
    light_pwm = int(np.clip(lr_light.predict(X)[0], 0, 255))
    
    # Kết hợp
    return {
        'fan_on': fan_on,
        'fan_pwm': fan_pwm if fan_on else 0,
        'pump_on': pump_on,
        'pump_pwm': pump_pwm if pump_on else 0,
        'light_on': light_on,
        'light_pwm': light_pwm if light_on else 0
    }

# ===== CHẠY LIÊN TỤC =====
print("="*70)
print("🚀 BẮT ĐẦU MÔ PHỎNG - DỮ LIỆU THỰC TẾ (Nhấn Ctrl+C để dừng)")
print("="*70)
print("💡 Lưu ý: Có thể thấy quyết định không hoàn hảo 100% - đây là THỰC TẾ!")
print()

try:
    count = 0
    while True:
        count += 1
        
        # Đọc cảm biến giả
        sensors = fake_sensor_realistic()
        
        # Dự đoán
        result = predict(sensors)
        
        # Hiển thị
        print("="*70)
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Lần #{count}")
        print("="*70)
        
        print("\n📊 CẢM BIẾN:")
        print(f"  🌡️  Nhiệt độ: {sensors['temp']}°C")
        print(f"  💧 Độ ẩm KK: {sensors['humidity_air']}%")
        print(f"  🌱 Độ ẩm đất: {sensors['humidity_soil']}%")
        print(f"  ☀️  Ánh sáng: {sensors['light']} Lux")
        print(f"  🌦️  Dự báo: {sensors['forecast_temp']}°C, mưa {sensors['rain_prob']:.0f}%")
        
        print("\n🤖 QUYẾT ĐỊNH AI:")
        
        status_fan = "✅ BẬT" if result['fan_on'] else "❌ TẮT"
        print(f"  🌪️  Quạt: {status_fan} | PWM: {result['fan_pwm']}/255 ({result['fan_pwm']/2.55:.0f}%)")
        
        status_pump = "✅ BẬT" if result['pump_on'] else "❌ TẮT"
        print(f"  💦 Bơm: {status_pump} | PWM: {result['pump_pwm']}/255 ({result['pump_pwm']/2.55:.0f}%)")
        
        status_light = "✅ BẬT" if result['light_on'] else "❌ TẮT"
        print(f"  💡 Đèn: {status_light} | PWM: {result['light_pwm']}/255 ({result['light_pwm']/2.55:.0f}%)")
        
        # Giải thích quyết định
        print("\n💬 PHÂN TÍCH:")
        if sensors['temp'] > 30:
            print("  • Nhiệt độ cao → Nên bật quạt")
        if sensors['humidity_soil'] < 40:
            print("  • Đất khô → Nên bật bơm")
        if sensors['light'] < 300 and 6 <= datetime.now().hour <= 18:
            print("  • Thiếu sáng ban ngày → Nên bật đèn")
        
        print("\n" + "="*70 + "\n")
        
        time.sleep(3)
        
except KeyboardInterrupt:
    print("\n\n⛔ Đã dừng chương trình!")
    print(f"✅ Tổng cộng chạy {count} lần")
