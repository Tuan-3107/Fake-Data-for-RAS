#!/usr/bin/env python3
"""
Test mô hình - CHỈ 4 CẢM BIẾN
Không cần API thời tiết
"""

import numpy as np
import pickle
import time
from datetime import datetime

# Load models
print("📂 Đang load mô hình...")
try:
    rf_fan = pickle.load(open('rf_fan_v2.pkl', 'rb'))
    rf_pump = pickle.load(open('rf_pump_v2.pkl', 'rb'))
    rf_light = pickle.load(open('rf_light_v2.pkl', 'rb'))
    
    rfr_fan = pickle.load(open('rfr_fan_pwm_v2.pkl', 'rb'))
    rfr_pump = pickle.load(open('rfr_pump_pwm_v2.pkl', 'rb'))
    rfr_light = pickle.load(open('rfr_light_pwm_v2.pkl', 'rb'))
    
    print("✅ Đã load 6 mô hình thành công!")
    print("  • 3 Random Forest Classifier (BẬT/TẮT)")
    print("  • 3 Random Forest Regressor (PWM)\n")
except FileNotFoundError:
    print("❌ Lỗi: Chưa có file mô hình!")
    print("💡 Hãy chạy: python3 train_model_v2.py trước")
    exit()

def fake_sensor_v2():
    """
    Tạo dữ liệu cảm biến giả - CHỈ 4 CẢM BIẾN
    Mô phỏng dữ liệu từ ESP32 thực tế
    """
    hour = datetime.now().hour
    is_day = 6 <= hour <= 18
    
    # 1. NHIỆT ĐỘ
    temp = np.random.normal(32 if is_day else 24, 5)
    temp = np.clip(temp, 18, 40)
    
    # 2. ĐỘ ẨM KHÔNG KHÍ
    humidity_air = 100 - temp + np.random.normal(0, 10)
    humidity_air = np.clip(humidity_air, 25, 95)
    
    # 3. ĐỘ ẨM ĐẤT
    humidity_soil = np.random.uniform(20, 80)
    
    # 4. ÁNH SÁNG
    if is_day:
        cloud_factor = np.random.uniform(0.3, 1.0)
        light = np.random.normal(750 * cloud_factor, 200)
    else:
        light = np.random.normal(30, 25)
    light = np.clip(light, 0, 1000)
    
    return {
        'temp': round(temp, 1),
        'humidity_air': round(humidity_air, 1),
        'humidity_soil': round(humidity_soil, 1),
        'light': round(light, 1)
    }

def predict(sensors):
    """Dự đoán hành động dựa trên 4 cảm biến"""
    X = np.array([[
        sensors['temp'],
        sensors['humidity_air'],
        sensors['humidity_soil'],
        sensors['light']
    ]])
    
    # Random Forest Classifier - Quyết định BẬT/TẮT
    fan_on = rf_fan.predict(X)[0]
    pump_on = rf_pump.predict(X)[0]
    light_on = rf_light.predict(X)[0]
    
    # Random Forest Regressor - Dự đoán PWM
    fan_pwm = int(np.clip(rfr_fan.predict(X)[0], 0, 255))
    pump_pwm = int(np.clip(rfr_pump.predict(X)[0], 0, 255))
    light_pwm = int(np.clip(rfr_light.predict(X)[0], 0, 255))
    
    # Kết hợp quyết định
    return {
        'fan_on': fan_on,
        'fan_pwm': fan_pwm if fan_on else 0,
        'pump_on': pump_on,
        'pump_pwm': pump_pwm if pump_on else 0,
        'light_on': light_on,
        'light_pwm': light_pwm if light_on else 0
    }

def analyze_decision(sensors, result):
    """Phân tích và giải thích quyết định"""
    reasons = []
    
    # Quạt
    if sensors['temp'] > 30:
        reasons.append(f"🌡️  Nhiệt độ cao ({sensors['temp']}°C) → Nên bật quạt")
    if sensors['temp'] < 25:
        reasons.append(f"❄️  Nhiệt độ thấp ({sensors['temp']}°C) → Không cần quạt")
    
    # Bơm
    if sensors['humidity_soil'] < 40:
        reasons.append(f"🌱 Đất khô ({sensors['humidity_soil']}%) → Nên bật bơm")
    if sensors['humidity_soil'] > 60:
        reasons.append(f"💧 Đất ẩm ({sensors['humidity_soil']}%) → Không cần tưới")
    
    # Đèn
    if sensors['light'] < 300 and 6 <= datetime.now().hour <= 18:
        reasons.append(f"💡 Thiếu sáng ({sensors['light']} Lux) → Nên bật đèn")
    if sensors['light'] > 500:
        reasons.append(f"☀️  Đủ sáng ({sensors['light']} Lux) → Không cần đèn")
    
    return reasons

# ===== CHẠY LIÊN TỤC =====
print("="*70)
print("🚀 BẮT ĐẦU MÔ PHỎNG - CHỈ 4 CẢM BIẾN")
print("="*70)
print("💡 Mô phỏng hệ thống thực tế (không cần API thời tiết)")
print("   Nhấn Ctrl+C để dừng\n")

try:
    count = 0
    while True:
        count += 1
        
        # Đọc cảm biến giả (mô phỏng ESP32)
        sensors = fake_sensor_v2()
        
        # Dự đoán bằng AI
        result = predict(sensors)
        
        # Phân tích quyết định
        analysis = analyze_decision(sensors, result)
        
        # Hiển thị kết quả
        print("="*70)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Lần #{count}")
        print("="*70)
        
        print("\n📊 DỮ LIỆU CẢM BIẾN (từ ESP32):")
        print(f"  🌡️  Nhiệt độ: {sensors['temp']}°C")
        print(f"  💧 Độ ẩm không khí: {sensors['humidity_air']}%")
        print(f"  🌱 Độ ẩm đất: {sensors['humidity_soil']}%")
        print(f"  ☀️  Ánh sáng: {sensors['light']} Lux")
        
        print("\n🤖 QUYẾT ĐỊNH CỦA AI:")
        
        # Quạt
        status_fan = "✅ BẬT" if result['fan_on'] else "❌ TẮT"
        print(f"  🌪️  Quạt: {status_fan} | PWM: {result['fan_pwm']}/255 ({result['fan_pwm']/2.55:.0f}%)")
        
        # Bơm
        status_pump = "✅ BẬT" if result['pump_on'] else "❌ TẮT"
        print(f"  💦 Bơm: {status_pump} | PWM: {result['pump_pwm']}/255 ({result['pump_pwm']/2.55:.0f}%)")
        
        # Đèn
        status_light = "✅ BẬT" if result['light_on'] else "❌ TẮT"
        print(f"  💡 Đèn: {status_light} | PWM: {result['light_pwm']}/255 ({result['light_pwm']/2.55:.0f}%)")
        
        # Phân tích
        if analysis:
            print("\n💬 PHÂN TÍCH:")
            for reason in analysis:
                print(f"  {reason}")
        
        # Lệnh gửi cho ESP32
        print("\n📡 LỆNH GỬI CHO ESP32 (qua MQTT):")
        print(f"  Topic: greenhouse/actuators")
        print(f"  Message: {result['fan_pwm']},{result['pump_pwm']},{result['light_pwm']}")
        
        print("\n" + "="*70 + "\n")
        
        time.sleep(3)  # Đợi 3 giây (mô phỏng đọc cảm biến)
        
except KeyboardInterrupt:
    print("\n\n⛔ Đã dừng chương trình!")
    print(f"✅ Tổng cộng chạy {count} lần")
    print("\n📊 THỐNG KÊ:")
    print("  • Mô hình: Random Forest (Classifier + Regressor)")
    print("  • Input: 4 cảm biến")
    print("  • Output: 3 actuators (PWM 0-255)")
    print("  • Không cần: API thời tiết")
    print("\n🎉 Hệ thống sẵn sàng cho triển khai thực tế!")
