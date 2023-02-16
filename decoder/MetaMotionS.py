#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 08:58:47 2022

@author: Jackson Cagle
"""

import numpy as np
import struct

def decodeBRAVOWearableStructure(filename):
    with open(filename, "rb") as file:
        rawBytes = file.read()
        
    currentIndex = 80
    Headers = rawBytes[:currentIndex].decode("utf-8").rstrip("\x00")
    
    Data = {"DeviceID": Headers, "Accelerometer": {"Time": [], "Data": []}, "RSSAccelerometer": {"Time": [], "Data": []}, "Gyroscope": {"Time": [], "Data": []}, "AmbientLight": {"Time": [], "Data": []}}
    while currentIndex < len(rawBytes)-1:
        DataType = rawBytes[currentIndex]
        if DataType == 25:
            DataValues = np.frombuffer(rawBytes[currentIndex+4:currentIndex+16], np.float32, count=3)
            Timestamp = np.frombuffer(rawBytes[currentIndex+16:currentIndex+24], np.float64, count=1)[0]
            Data["Accelerometer"]["Time"].append(Timestamp)
            Data["Accelerometer"]["Data"].append(DataValues)
            currentIndex += 24
        elif DataType == 26:
            DataValues = np.frombuffer(rawBytes[currentIndex+4:currentIndex+8], np.float32, count=1)[0]
            Timestamp = np.frombuffer(rawBytes[currentIndex+8:currentIndex+16], np.float64, count=1)[0]
            Data["RSSAccelerometer"]["Time"].append(Timestamp)
            Data["RSSAccelerometer"]["Data"].append(DataValues)
            currentIndex += 16
        elif DataType == 35:
            DataValues = np.frombuffer(rawBytes[currentIndex+4:currentIndex+16], np.float32, count=3)
            Timestamp = np.frombuffer(rawBytes[currentIndex+16:currentIndex+24], np.float64, count=1)[0]
            Data["Gyroscope"]["Time"].append(Timestamp)
            Data["Gyroscope"]["Data"].append(DataValues)
            currentIndex += 24
    
    Data["Accelerometer"]["Time"] = np.array(Data["Accelerometer"]["Time"])
    Data["Accelerometer"]["Data"] = np.array(Data["Accelerometer"]["Data"])
    Data["Gyroscope"]["Time"] = np.array(Data["Gyroscope"]["Time"])
    Data["Gyroscope"]["Data"] = np.array(Data["Gyroscope"]["Data"])
    Data["AmbientLight"]["Time"] = np.array(Data["AmbientLight"]["Time"])
    Data["AmbientLight"]["Data"] = np.array(Data["AmbientLight"]["Data"])
    
    return Data