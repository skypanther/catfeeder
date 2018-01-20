"""
Requires https://github.com/swkim01/RPi.GPIO-PineA64
Follow install instructions there
"""

import RPi.GPIO as GPIO
import datetime
from time import sleep

PAUSE = 2  # seconds


class FeederController:

    def __init__(self, open_pin, close_pin, pir_pin):
        self.delay_before_closing = 5000  # 5 seconds

        self.motor_open_pin = open_pin
        self.motor_close_pin = close_pin
        self.pir_pin = pir_pin
        # set up GPIO and pins
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.motor_open_pin, GPIO.OUT)
        GPIO.setup(self.motor_close_pin, GPIO.OUT)
        GPIO.setup(pir_pin, GPIO.IN)
        # reset output pins to start
        GPIO.output(self.motor_open_pin, False)
        GPIO.output(self.motor_close_pin, False)

    def open_food_bowl(self):
        # open food bowl cover
        GPIO.output(self.motor_close_pin, False)
        GPIO.output(self.motor_open_pin, True)

    def close_food_bowl(self):
        # close food bowl cover
        GPIO.output(self.motor_open_pin, False)
        GPIO.output(self.motor_close_pin, True)

    def watch_cat_presence(self, wait_time_seconds=30):
        """
        PIR looks for motion. But, while cat eats at the bowl
        there will be little / no motion until they leave.
        So, we'll look for motion as they approach the bowl,
        then a period of no motion, then motion as they leave.

        @param wait_time - time in seconds to assume cat is present/eating
        """
        start_time = datetime.now()
        keep_looking = True
        while keep_looking:
            now = datetime.now()
            num_seconds = (start_time - now).total_seconds()
            pir_value = GPIO.input(self.pir_pin)
            if num_seconds > wait_time_seconds and pir_value == 0:
                # no cat or cat left
                keep_looking = False
                self.close_food_bowl()
            sleep(PAUSE)
