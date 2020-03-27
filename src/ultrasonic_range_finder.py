#!/usr/bin/python
# -*- Coding: utf-8 -*-
import serial
import re


def vapor_on():
    pass


def vapor_off():
    pass


def main():
    with serial.Serial('COM3', 9600, timeout=1) as ser:

        while True:
            print("test please input")
            flag = input()
            print(flag)
            if flag == 's':
                # flag = bytes(in_data, 'utf-8')
                ser.write(1)

            c = ser.readline()
            print('recv data')
            print(c)

            if flag == bytes('a', 'utf-8'):
                break
        ser.close()


if __name__=="__main__":
    main()
