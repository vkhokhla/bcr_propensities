#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:12:17 2017

@author: sergej
"""

f = open('C_CLIENTS_DATA_TABLE.dsv', 'r')

i = 0
for l in f.readlines():
    i += 1
    if i == 1 or (i > 4875540 and i < 4875550):
        print(l)
    
    if i > 4875550:
        break

f.close()