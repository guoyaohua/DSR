#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket, msvcrt, time, os

address = ('10.10.100.254', 59225)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
s.settimeout(1)
s.connect(address)
data=b''
start = time.time()
print('Recording Data...\nPress Any Key to Stop...')

while True:
    data_seg = s.recv(1024)
    data += data_seg
    print(data)
    '''
    if msvcrt.kbhit():
        end = time.time()
        log_str = '%6.1f sec'%(end-start)
        log_str += time.strftime('  |  %Y-%m-%d %H:%M:%S\n')
        print( log_str )
        with open('../data/log.txt', 'a') as fd:
            fd.write( log_str )
        break
'''
'''
file_name = '../data/raw/' + str(int(time.time()))
with open(file_name, 'wb') as fd:
    fd.write(data)

# Check data cosistency
os.system("python DataCheck.py "+file_name)
'''