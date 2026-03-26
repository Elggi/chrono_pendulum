#include <Wire.h>
#include <Adafruit_INA219.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

Adafruit_INA219 ina219;

const int PIN_START = 7;
const int PIN_DIR   = 8;
const int PIN_PWM   = 9;
const int ENC_A     = 3;
const int ENC_B     = 2;

volatile long enc_count = 0;
volatile unsigned long last_enc_us = 0;

int last_pwm = 0;
bool ina219_ok = false;

unsigned long last_status_ms = 0;
const unsigned long STATUS_PERIOD_MS = 20;   // 50 Hz

// 너무 짧은 펄스는 노이즈로 보고 무시
// 값은 엔코더 상태 보면서 20~200us 정도에서 조정
const unsigned long ENC_GLITCH_US = 50;

const size_t CMD_BUF_LEN = 32;
char cmd_buf[CMD_BUF_LEN];
size_t cmd_len = 0;

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

void readPowerMetrics(float &bus_V, float &current_mA, float &power_mW) {
  bus_V = NAN;
  current_mA = NAN;
  power_mW = NAN;

  if (!ina219_ok) {
    return;
  }

  bus_V = ina219.getBusVoltage_V();
  current_mA = ina219.getCurrent_mA();
  power_mW = ina219.getPower_mW();

  if (!isfinite(bus_V) || !isfinite(current_mA) || !isfinite(power_mW)) {
    ina219_ok = false;
    Serial.println("INA219,RUNTIME_ERR");
    bus_V = NAN;
    current_mA = NAN;
    power_mW = NAN;
  }
}

void sendStatus() {
  long enc = readEnc();
  unsigned long ms = millis();
  float bus_V = NAN;
  float current_mA = NAN;
  float power_mW = NAN;
  readPowerMetrics(bus_V, current_mA, power_mW);

  // S,enc,pwm,ms,bus_v,current_mA,power_mW
  Serial.print("S,");
  Serial.print(enc);
  Serial.print(",");
  Serial.print(last_pwm);
  Serial.print(",");
  Serial.print(ms);
  Serial.print(",");
  if (isfinite(bus_V)) Serial.print(bus_V, 4);
  else Serial.print("nan");
  Serial.print(",");
  if (isfinite(current_mA)) Serial.print(current_mA, 4);
  else Serial.print("nan");
  Serial.print(",");
  if (isfinite(power_mW)) Serial.println(power_mW, 4);
  else Serial.println("nan");
}

void handleCommand(const char *line) {
  if (strncmp(line, "U,", 2) == 0) {
    int pwm = atoi(line + 2);
    applyPWM(pwm);
    sendStatus();
  }
  else if (strcmp(line, "ZERO") == 0) {
    zeroEnc();
    Serial.println("ZERO_OK");
  }
  else if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  }
}

void pollSerial() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();

    if (ch == '\r') {
      continue;
    }

    if (ch == '\n') {
      cmd_buf[cmd_len] = '\0';
      if (cmd_len > 0) {
        handleCommand(cmd_buf);
      }
      cmd_len = 0;
      continue;
    }

    if (cmd_len < (CMD_BUF_LEN - 1)) {
      cmd_buf[cmd_len++] = ch;
    } else {
      cmd_len = 0;
      Serial.println("CMD,OVERRUN");
    }
  }
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
  ina219_ok = ina219.begin();
  if (!ina219_ok) {
    Serial.println("INA219,ERR");
  } else {
    Serial.println("INA219,OK");
  }

  Serial.println("BOOT,OK");

  // 기존 CHANGE x 2개 대신 A상 rising만 사용
  attachInterrupt(digitalPinToInterrupt(ENC_A), isr_enc_a_rising, RISING);
}

void loop() {
  pollSerial();

  unsigned long now = millis();
  if (now - last_status_ms >= STATUS_PERIOD_MS) {
    last_status_ms = now;
    sendStatus();
  }
}
