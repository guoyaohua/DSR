# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:35:04 2018

@author: John Kwok
"""
import msvcrt
while True:
    if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
        print("结束触发")