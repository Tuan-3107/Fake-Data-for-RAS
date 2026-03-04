#!/usr/bin/env python3
"""
Test mô hình ML - 10,000 mẫu Nha Trang
"""

import numpy as np
import pickle
import time
from datetime import datetime

# Load models
print("📂 Đang load mô hình...")
try:
    rf_fan = pickle.load(open('rf_fan_nhatrang.pkl', 'rb'))
    rf_pump = pickle.load(open('rf_pump_nhatrang.pkl', 'rb'))
    rf_light = pickle.load(open('rf_light_nhatrang.pkl', 'rb'))
    
    rfr_fan = pickle.load(open('rfr_fan_pwm_nhatrang.pkl', 'rb'))
    rfr_pump = pickle.load(open('rfr_pump_pwm_nhatrang.pkl', 'rb'))
    rfr_light = pickle.load(open('rfr_light_pwm_nhatrang.pkl', 'rb'))
    
    print("✅ Đã load 6 mô hình thành công!")
    print("  • 3 Classifier (BẬT/TẮT)")
    print("  • 3 Regressor (PWM)\n")
except FileNotFoundError:
    print("❌ Lỗi: Chưa có mô hình!")
    print("💡 Chạy: python3 train_model_nhatrang_v2.py")
    exit()

def fake_sensor_nhatrang():
    """Mô phỏng cảm biến Nha Trang"""
    hour = datetime.now().hour
    month = datetime.now().month
    is_day = 6 <= hour <= 18
    
    # Nhiệt độ
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
        else:
            temp = np.random.normal(base_temp_day - 1, 1.3)
    else:
        temp = np.random.normal(base_temp_night, 1.2)
    temp = np.clip(temp, 20, 36)
    
    # Độ ẩm KK
    if month in [9, 10, 11, 12]:
        base_humidity = 85
    else:
        base_humidity = 78
    humidity_air = base_humidity + (28 - temp) * 0.8 + np.random.normal(0, 3)
    if not is_day:
        humidity_air += 5
    humidity_air = np.clip(humidity_air, 70, 95)
    
    # Độ ẩm đất
    humidity_soil = np.random.uniform(25, 75)
    
    # Ánh sáng
    if is_day:
        if 11 <= hour <= 14:
            light = np.random.normal(850, 100)
        else:
            light = np.random.normal(500, 150)
    else:
        light = np.random.normal(10, 8)
    light = np.clip(light, 0, 1000)
    
    return {
        'temp': round(temp, 1),
        'humidity_air': round(humidity_air, 1),
        'humidity_soil': round(humidity_soil, 1),
        'light': round(light, 1)
    }

def predict(sensors):
    """Dự đoán hành động"""
    X = np.array([[
        sensors['temp'],
        sensors['humidity_air'],
        sensors['humidity_soil'],
        sensors['light']
    ]])
    
    # Classifier
    fan_on = rf_fan.predict(X)[0]
    pump_on = rf_pump.predict(X)[0]
    light_on = rf_light.predict(X)[0]
    
    # Regressor
    fan_pwm = int(np.clip(rfr_fan.predict(X)[0], 0, 255))
    pump_pwm = int(np.clip(rfr_pump.predict(X)[0], 0, 255))
    light_pwm = int(np.clip(rfr_light.predict(X)[0], 0, 255))
    
    return {
        'fan_on': fan_on,
        'fan_pwm': fan_pwm if fan_on else 0,
        'pump_on': pump_on,
        'pump_pwm': pump_pwm if pump_on else 0,
        'light_on': light_on,
        'light_pwm': light_pwm if light_on else 0
    }

def analyze(sensors, result):
    """Phân tích quyết định"""
    reasons = []
    
    if sensors['temp'] > 29:
        reasons.append(f"🌡️  Nhiệt độ cao ({sensors['temp']}°C) → Nên bật quạt")
    
    if sensors['humidity_soil'] < 45:
        reasons.append(f"🌱 Đất khô ({sensors['humidity_soil']}%) → Nên tưới")
    
    if sensors['light'] < 400 and 6 <= datetime.now().hour <= 18:
        reasons.append(f"💡 Thiếu sáng ({sensors['light']} Lux) → Bật đèn")
    
    return reasons

# ===== CHẠY MÔ PHỎNG =====
print("="*70)
print("🌴 MÔ PHỎNG HỆ THỐNG - NHA TRANG")
print("="*70)
print("💡 Nhấn Ctrl+C để dừng\n")

try:
    count = 0
    while True:
        count += 1
        
        sensors = fake_sensor_nhatrang()
        result = predict(sensors)
        analysis = analyze(sensors, result)
        
        print("="*70)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Lần #{count}")
        print("="*70)
        
        print("\n📊 CẢM BIẾN:")
        print(f"  🌡️  Nhiệt độ: {sensors['temp']}°C")
        print(f"  💧 Độ ẩm KK: {sensors['humidity_air']}%")
        print(f"  🌱 Độ ẩm đất: {sensors['humidity_soil']}%")
        print(f"  ☀️  Ánh sáng: {sensors['light']} Lux")
        
        print("\n🤖 QUYẾT ĐỊNH AI:")
        
        status_fan = "✅ BẬT" if result['fan_on'] else "❌ TẮT"
        print(f"  🌪️  Quạt: {status_fan} | PWM: {result['fan_pwm']}/255 ({result['fan_pwm']/2.55:.0f}%)")
        
        status_pump = "✅ BẬT" if result['pump_on'] else "❌ TẮT"
        print(f"  💦 Bơm: {status_pump} | PWM: {result['pump_pwm']}/255 ({result['pump_pwm']/2.55:.0f}%)")
        
        status_light = "✅ BẬT" if result['light_on'] else "❌ TẮT"
        print(f"  💡 Đèn: {status_light} | PWM: {result['light_pwm']}/255 ({result['light_pwm']/2.55:.0f}%)")
        
        if analysis:
            print("\n💬 PHÂN TÍCH:")
            for reason in analysis:
                print(f"  {reason}")
        
        print("\n📡 MQTT → ESP32:")
        print(f"  Topic: greenhouse/actuators")
        print(f"  Data: {result['fan_pwm']},{result['pump_pwm']},{result['light_pwm']}")
        
        print("\n" + "="*70 + "\n")
        
        time.sleep(3)
        
except KeyboardInterrupt:
    print("\n\n⛔ Dừng!")
    print(f"✅ Đã chạy {count} lần")
    print("\n🎉 Hệ thống hoạt động tốt!")
