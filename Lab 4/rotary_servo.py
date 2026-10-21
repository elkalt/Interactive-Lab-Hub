"""Use the I2C rotary encoder to position a servo on the Pi Servo HAT."""

import time
import board
from adafruit_seesaw import seesaw, rotaryio
import pi_servo_hat


SERVO_CHANNEL = 0
SERVO_MIN = 0
SERVO_MAX = 120
SERVO_CENTER = 60
DEGREES_PER_STEP = 5
SERVO_RANGE_STEPS = (SERVO_MAX - SERVO_MIN) // DEGREES_PER_STEP
SERVO_CYCLE_STEPS = 2 * SERVO_RANGE_STEPS
CENTER_STEP = (SERVO_CENTER - SERVO_MIN) // DEGREES_PER_STEP


encoder_board = seesaw.Seesaw(board.I2C(), addr=0x36)
encoder = rotaryio.IncrementalEncoder(encoder_board)
servo_hat = pi_servo_hat.PiServoHat()
servo_hat.restart()

last_angle = None
print("Turn the encoder to move the servo; press Ctrl+C to stop.")

try:
    while True:
        # Negate the count so clockwise turns initially increase the angle.
        steps = -encoder.position
        # Reflect at each endpoint so the positional servo never jumps from
        # one end of its range to the other.
        phase = (CENTER_STEP + steps) % SERVO_CYCLE_STEPS
        angle_step = (phase if phase <= SERVO_RANGE_STEPS
                      else SERVO_CYCLE_STEPS - phase)
        angle = SERVO_MIN + angle_step * DEGREES_PER_STEP

        if angle != last_angle:
            servo_hat.move_servo_position(SERVO_CHANNEL, angle)
            print(f"Encoder: {steps}  Servo: {angle} degrees")
            last_angle = angle
        time.sleep(0.02)
finally:
    servo_hat.move_servo_position(SERVO_CHANNEL, SERVO_CENTER)
    print("Servo returned to center.")
