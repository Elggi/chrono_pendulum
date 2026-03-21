#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

const int PIN_START = 7;
const int PIN_DIR   = 8;
const int PIN_PWM   = 9;
const int ENC_A     = 3;
const int ENC_B     = 2;

volatile long enc_count = 0;
volatile unsigned long last_enc_us = 0;

int last_pwm = 0;

unsigned long last_status_ms = 0;
const unsigned long STATUS_PERIOD_MS = 20;   // 50 Hz

// 너무 짧은 펄스는 노이즈로 보고 무시
// 값은 엔코더 상태 보면서 20~200us 정도에서 조정
const unsigned long ENC_GLITCH_US = 50;

// A상 상승엣지에서만 샘플링하는 1x 디코딩
void isr_enc_a_rising() {
  unsigned long now_us = micros();
  if ((now_us - last_enc_us) < ENC_GLITCH_US) {
    return;
  }
  last_enc_us = now_us;

  bool a = digitalRead(ENC_A);
  bool b = digitalRead(ENC_B);

  // A rising 시점의 B 상태로 방향 판별
  // 방향이 반대로 나오면 ++ / -- 바꾸면 됨
  if (a) {
    if (b) enc_count--;
    else   enc_count++;
  }
}

long readEnc() {
  noInterrupts();
  long c = enc_count;
  interrupts();
  return c;
}

void zeroEnc() {
  noInterrupts();
  enc_count = 0;
  interrupts();
}

void applyPWM(int pwm) {
  pwm = constrain(pwm, -255, 255);
  last_pwm = pwm;

  if (pwm >= 0) {
    digitalWrite(PIN_DIR, HIGH);
    analogWrite(PIN_PWM, pwm);
  } else {
    digitalWrite(PIN_DIR, LOW);
    analogWrite(PIN_PWM, -pwm);
  }
}

void sendStatus() {
  long enc = readEnc();
  unsigned long ms = millis();

  float shunt_mV   = ina219.getShuntVoltage_mV();
  float bus_V      = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW   = ina219.getPower_mW();

  // S,enc,pwm,ms,bus_v,current_mA,power_mW
  Serial.print("S,");
  Serial.print(enc);
  Serial.print(",");
  Serial.print(last_pwm);
  Serial.print(",");
  Serial.print(ms);
  Serial.print(",");
  Serial.print(bus_V, 4);
  Serial.print(",");
  Serial.print(current_mA, 4);
  Serial.print(",");
  Serial.println(power_mW, 4);
}

void setup() {
  pinMode(PIN_START, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_PWM, OUTPUT);

  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);

  digitalWrite(PIN_START, HIGH);
  digitalWrite(PIN_DIR, HIGH);
  analogWrite(PIN_PWM, 0);

  Serial.begin(115200);
  delay(1000);

  Wire.begin();
  if (!ina219.begin()) {
    Serial.println("INA219,ERR");
    while (1) delay(10);
  }

  Serial.println("BOOT,OK");
  Serial.println("INA219,OK");

  // 기존 CHANGE x 2개 대신 A상 rising만 사용
  attachInterrupt(digitalPinToInterrupt(ENC_A), isr_enc_a_rising, RISING);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("U,")) {
      int pwm = line.substring(2).toInt();
      applyPWM(pwm);
      sendStatus();
    }
    else if (line == "ZERO") {
      zeroEnc();
      Serial.println("ZERO_OK");
    }
    else if (line == "PING") {
      Serial.println("PONG");
    }
  }

  unsigned long now = millis();
  if (now - last_status_ms >= STATUS_PERIOD_MS) {
    last_status_ms = now;
    sendStatus();
  }
}
