// Host build driver for the firmware sketch (compile check only).
#include "Arduino.h"
SerialStub Serial;
#include "../bb8_mcu/bb8_mcu.ino"
int main() { setup(); for (int i = 0; i < 100; i++) loop(); return 0; }
