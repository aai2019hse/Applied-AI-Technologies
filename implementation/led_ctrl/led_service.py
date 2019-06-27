#!/usr/bin/env python3
# Author: HO4X / love2code 

import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

import fcntl, os
import errno

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
fcntl.fcntl(sock, fcntl.F_SETFL, os.O_NONBLOCK)

import time
from neopixel import *
import argparse

# LED strip configuration:
LED_COUNT      = 19      # Number of LED pixels.
LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
#LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

START_WITH_ZERO = True

def start_up():
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()
    for i in range(strip.numPixels() / 2):
        for x in range(0, 255):
            strip.setPixelColor(i + 9, Color(x, x, 0))
            strip.setPixelColor(-i + 8, Color(x, x, 0))        
            strip.show()
            time.sleep(0.000000001);
    for i in range(strip.numPixels() / 2):
        for x in range(0, 255):
            strip.setPixelColor(i + 9, Color(255 - x, 255 - x, 0))
            strip.setPixelColor(-i + 8, Color(255 - x, 255 - x, 0))        
            strip.show()
            time.sleep(0.000000001);
    pass
    
def fade_down(): 
    strip.fill(255, 255, 255)
    strip.show()

# Main program logic follows:
if __name__ == '__main__':
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
    args = parser.parse_args()

    # Create NeoPixel object with appropriate configuration.
    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    # Intialize the library (must be called once before other functions).
    strip.begin()
    print ('Press Ctrl-C to quit.')
    if not args.clear:
        print('Use "-c" argument to clear LEDs on exit')
    
    LAST_MESSAGE = b""
    
    time_last = time.time();
    direction = 1; 
    x = 100
    try:
        #Turn off all LEDs
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(0, 0, 0))
            strip.show()
        while True:
            try: 
                data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
                
            except socket.error, e:
                if(LAST_MESSAGE == b"OFF"):
                    for i in range(strip.numPixels()):
                        strip.setPixelColor(i, Color(0, 0, 0))
                    strip.show()
                if(LAST_MESSAGE == b"BOOTING"):
                    start_up()
                    #if(time.time() - time_last > 0.01):
                        #time_last = time.time()
                        #if(direction == 1): 
                            #x = x + 1
                            #if(x >= 175): 
                                #direction = 0; 
                        #if(direction == 0): 
                            #x = x - 1
                            #if(x <= 25): 
                                #direction = 1; 
                        #for i in range(strip.numPixels()):
                            #strip.setPixelColor(i, Color(x, x, 0)) # g r b
                        #strip.show()
                            
                if(LAST_MESSAGE == b"CALC"): 
                    if(time.time() - time_last > 0.001):
                        time_last = time.time()
                        if(direction == 1): 
                            x = x + 1
                            if(x >= 255): 
                                direction = 0; 
                        if(direction == 0): 
                            x = x - 1
                            if(x <= 205): 
                                direction = 1; 
                        for i in range(strip.numPixels()):
                            strip.setPixelColor(i, Color(x, x, x))
                        strip.show()
                        
                if(LAST_MESSAGE == b"RUNNING"): 
                    if(time.time() - time_last > 0.01):
                        time_last = time.time()
                        if(START_WITH_ZERO):
                            x = 0
                            direction = 1 
                            START_WITH_ZERO = 0
                        if(direction == 1): 
                            x = x + 1
                            if(x >= 255): 
                                direction = 0; 
                        if(direction == 0): 
                            x = x - 1
                            if(x <= 125): 
                                direction = 1; 
                        for i in range(strip.numPixels()):
                            strip.setPixelColor(i, Color(0, 0, x))
                        strip.show()
                        
                if(LAST_MESSAGE == b"DENNIED"): 
                    if(time.time() - time_last > 0.0001):
                        time_last = time.time()
                        if(direction == 1): 
                            x = x + 1
                            if(x >= 255): 
                                direction = 0; 
                        if(direction == 0): 
                            x = x - 1
                            if(x <= 25): 
                                direction = 1; 
                        for i in range(strip.numPixels()):
                            strip.setPixelColor(i, Color(0, x, 0))
                        strip.show()
                        
                if(LAST_MESSAGE == b"OK"): 
                    if(time.time() - time_last > 0.008):
                        time_last = time.time()
                        if(direction == 1): 
                            x = x + 1
                            if(x >= 255): 
                                direction = 0; 
                        if(direction == 0): 
                            x = x - 1
                            if(x <= 25): 
                                direction = 1; 
                        for i in range(strip.numPixels()):
                            strip.setPixelColor(i, Color(x, 0, 0))
                        strip.show()
            else:
                if(data != LAST_MESSAGE): 
                    if (LAST_MESSAGE == b"DENNIED"):
                        for xx in range(0, x):
                            for i in range(strip.numPixels()):
                                strip.setPixelColor(i, Color(0, x - xx, 0))
                            strip.show()
                            time.sleep(0.0003)
                    if (LAST_MESSAGE == b"OK"):
                        for xx in range(0, x):
                            for i in range(strip.numPixels()):
                                strip.setPixelColor(i, Color(x - xx, 0, 0))
                            strip.show()
                            time.sleep(0.0003)
                        
                    START_WITH_ZERO = True
                LAST_MESSAGE = data
                #print "received message:", data
    except KeyboardInterrupt:
        if args.clear:
            colorWipe(strip, Color(0,0,0), 10)
