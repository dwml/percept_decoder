#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:00:53 2021

@author: Jackson Cagle
"""

import numpy as np
import pandas as pd

def decodeHPFCSV(filename, skiprows=0):
    CSV = pd.read_csv(filename, skiprows=skiprows)
    
    Delsys = dict()
    Delsys["EMG"] = [np.zeros((0,1))] * 16
    Delsys["Acc"] = [np.zeros((0,3))] * 16
    Delsys["Gyro"] = [np.zeros((0,3))] * 16
    Delsys["Mag"] = [np.zeros((0,3))] * 16
    
    lastKey = ""
    DelsysRate = 1111.111
    IMURate = 148.148
    
    for key in CSV.keys():
        if key.find("EMG") > 0:
            sensorID = int(key.split(":")[0].replace("Trigno IM sensor ","").replace("Trigno sensor ",""))
            Delsys["EMG"][sensorID-1] = np.array(CSV[key])
            EMGTime = np.array(CSV[lastKey])
            DelsysRate = 1 / np.median(np.diff(EMGTime))
        elif key.find("Acc") > 0:
            sensorID = int(key.split(":")[0].replace("Trigno IM sensor ","").replace("Trigno sensor ",""))
            if Delsys["Acc"][sensorID-1].shape[0] == 0:
                Delsys["Acc"][sensorID-1] = np.zeros((len(np.array(CSV[key])), 3))
            if key.find(f"{sensorID}.X") > 0:
                Delsys["Acc"][sensorID-1][:,0] = np.array(CSV[key])
            elif key.find(f"{sensorID}.Y") > 0:
                Delsys["Acc"][sensorID-1][:,1] = np.array(CSV[key])
            elif key.find(f"{sensorID}.Z") > 0:
                Delsys["Acc"][sensorID-1][:,2] = np.array(CSV[key])
            IMUTime = np.array(CSV[lastKey])
            IMUTime = IMUTime[IMUTime != 0]
            IMURate = 1 / np.median(np.diff(IMUTime))
            
        elif key.find("Gyro") > 0:
            sensorID = int(key.split(":")[0].replace("Trigno IM sensor ","").replace("Trigno sensor ",""))
            if Delsys["Gyro"][sensorID-1].shape[0] == 0:
                Delsys["Gyro"][sensorID-1] = np.zeros((len(np.array(CSV[key])), 3))
            if key.find(f"{sensorID}.X") > 0:
                Delsys["Gyro"][sensorID-1][:,0] = np.array(CSV[key])
            elif key.find(f"{sensorID}.Y") > 0:
                Delsys["Gyro"][sensorID-1][:,1] = np.array(CSV[key])
            elif key.find(f"{sensorID}.Z") > 0:
                Delsys["Gyro"][sensorID-1][:,2] = np.array(CSV[key])
        elif key.find("Mag") > 0:
            sensorID = int(key.split(":")[0].replace("Trigno IM sensor ","").replace("Trigno sensor ",""))
            if Delsys["Mag"][sensorID-1].shape[0] == 0:
                Delsys["Mag"][sensorID-1] = np.zeros((len(np.array(CSV[key])), 3))
            if key.find(f"{sensorID}.X") > 0:
                Delsys["Mag"][sensorID-1][:,0] = np.array(CSV[key])
            elif key.find(f"{sensorID}.Y") > 0:
                Delsys["Mag"][sensorID-1][:,1] = np.array(CSV[key])
            elif key.find(f"{sensorID}.Z") > 0:
                Delsys["Mag"][sensorID-1][:,2] = np.array(CSV[key])
        elif key.find("Analog") > 0:
            sensorID = int(key.split(":")[0].replace("Trigno Analog Input Adapter ",""))
            if Delsys["EMG"][sensorID-1].shape[0] == 0:
                Delsys["EMG"][sensorID-1] = np.zeros((len(np.array(CSV[key])), 4))
            if key.find(f"{sensorID}.A") > 0:
                Delsys["EMG"][sensorID-1][:,0] = np.array(CSV[key])
            elif key.find(f"{sensorID}.B") > 0:
                Delsys["EMG"][sensorID-1][:,1] = np.array(CSV[key])
            elif key.find(f"{sensorID}.C") > 0:
                Delsys["EMG"][sensorID-1][:,2] = np.array(CSV[key])
            elif key.find(f"{sensorID}.D") > 0:
                Delsys["EMG"][sensorID-1][:,3] = np.array(CSV[key])
                
        lastKey = key
    
    Delsys["EMGTime"] = np.array(range(np.max([Delsys["EMG"][i].shape[0] for i in range(16)]))) / DelsysRate
    Delsys["IMUTime"] = np.array(range(np.max([Delsys["Acc"][i].shape[0] for i in range(16)]))) / IMURate
    Delsys["EMGRate"] = DelsysRate
    Delsys["IMURate"] = IMURate
    
    for sensorID in range(16):
        for t in range(1,Delsys["Acc"][sensorID].shape[0]):
            if Delsys["Acc"][sensorID][t,0] == 0:
                Delsys["Acc"][sensorID][t,0] = Delsys["Acc"][sensorID][t-1,0]
            if Delsys["Acc"][sensorID][t,1] == 0:
                Delsys["Acc"][sensorID][t,1] = Delsys["Acc"][sensorID][t-1,1]
            if Delsys["Acc"][sensorID][t,2] == 0:
                Delsys["Acc"][sensorID][t,2] = Delsys["Acc"][sensorID][t-1,2]
                
        for t in range(1,Delsys["Gyro"][sensorID].shape[0]):
            if Delsys["Gyro"][sensorID][t,0] == 0:
                Delsys["Gyro"][sensorID][t,0] = Delsys["Gyro"][sensorID][t-1,0]
            if Delsys["Gyro"][sensorID][t,1] == 0:
                Delsys["Gyro"][sensorID][t,1] = Delsys["Gyro"][sensorID][t-1,1]
            if Delsys["Gyro"][sensorID][t,2] == 0:
                Delsys["Gyro"][sensorID][t,2] = Delsys["Gyro"][sensorID][t-1,2]
                
        for t in range(1,Delsys["Mag"][sensorID].shape[0]):
            if Delsys["Mag"][sensorID][t,0] == 0:
                Delsys["Mag"][sensorID][t,0] = Delsys["Mag"][sensorID][t-1,0]
            if Delsys["Mag"][sensorID][t,1] == 0:
                Delsys["Mag"][sensorID][t,1] = Delsys["Mag"][sensorID][t-1,1]
            if Delsys["Mag"][sensorID][t,2] == 0:
                Delsys["Mag"][sensorID][t,2] = Delsys["Mag"][sensorID][t-1,2]

    return Delsys

def decodeBMLDelsysFormat(filename):
    with open(filename, "rb") as file:
        rawData = file.read()
    
    PackageOnset = np.where([rawData[i:i+3] == b"BML" for i in range(len(rawData)-2)])[0]
    PackageType = [[rawData[i+4:i+8]][0] for i in PackageOnset]
    
    TriggerSequence = np.where([PackageType[i] == b'Trg ' for i in range(len(PackageType))])[0]
    if len(TriggerSequence) > 0:
        
        Triggers = dict()
        Triggers["Time"] = np.zeros((len(TriggerSequence)))
        Triggers["String"] = list()
        Triggers["Unique"] = list()
        Triggers["ByteOnset"] = np.zeros((len(TriggerSequence)),dtype=int)
        Triggers["DataOnset"] = np.zeros((len(TriggerSequence)),dtype=int)
        
        for i in range(len(TriggerSequence)):
            onsetByte = int(PackageOnset[TriggerSequence[i]])
            Triggers["Time"][i] = int.from_bytes(rawData[onsetByte + 20 : onsetByte + 28], "little")
            Triggers["String"].append(rawData[onsetByte + 8 : onsetByte + 20].decode("utf-8").replace("\x00",""))
            if Triggers["String"][-1] not in Triggers["Unique"]:
                Triggers["Unique"].append(Triggers["String"][-1])
            Triggers["ByteOnset"][i] = onsetByte
            Triggers["DataOnset"][i] = onsetByte + 32
        
        Delsys = dict()
        Delsys["Triggers"] = Triggers
        Delsys["EMG"] = [np.zeros((0,1))] * 16
        Delsys["Acc"] = [np.zeros((0,3))] * 16
        Delsys["Gyro"] = [np.zeros((0,3))] * 16
        Delsys["Mag"] = [np.zeros((0,3))] * 16
        
        for i in range(len(Triggers["String"])):
            for sensorID in range(16):
                if Triggers["String"][i].find(f"imuEMG{sensorID}") >= 0:
                    if i == len(Triggers["String"])-1:
                        payload = rawData[Triggers["DataOnset"][i]:]
                    else:
                        payload = rawData[Triggers["DataOnset"][i]:Triggers["ByteOnset"][i+1]]
                    Delsys["EMG"][sensorID] = np.concatenate((Delsys["EMG"][sensorID], np.frombuffer(payload, dtype="<f4", count=int(len(payload)/4), offset=0).reshape((int(len(payload)/4),1))), axis=0)
                
                elif Triggers["String"][i].find(f"imuAUX{sensorID}") >= 0:
                    if i == len(Triggers["String"])-1:
                        payload = rawData[Triggers["DataOnset"][i]:]
                    else:
                        payload = rawData[Triggers["DataOnset"][i]:Triggers["ByteOnset"][i+1]]
                    reconstructedData = np.frombuffer(payload, dtype="<f4", count=int(len(payload)/4), offset=0)
                    dataLength = int(len(reconstructedData) / 3)
                    
                    Delsys["Acc"][sensorID] = np.concatenate((Delsys["Acc"][sensorID], reconstructedData[dataLength*0:dataLength*1].reshape(3, int(dataLength/3)).T), axis=0)
                    Delsys["Gyro"][sensorID] = np.concatenate((Delsys["Gyro"][sensorID], reconstructedData[dataLength*1:dataLength*2].reshape(3, int(dataLength/3)).T), axis=0)
                    Delsys["Mag"][sensorID] = np.concatenate((Delsys["Mag"][sensorID], reconstructedData[dataLength*2:dataLength*3].reshape(3, int(dataLength/3)).T), axis=0)
        
        
        
        Delsys["EMGTime"] = np.array(range(np.max([Delsys["EMG"][i].shape[0] for i in range(16)]))) / 1111.111
        Delsys["IMUTime"] = np.array(range(np.max([Delsys["Acc"][i].shape[0] for i in range(16)]))) / 148.148
        
        for sensorID in range(16):
            for t in range(1,Delsys["Acc"][sensorID].shape[0]):
                if Delsys["Acc"][sensorID][t,0] == 0:
                    Delsys["Acc"][sensorID][t,0] = Delsys["Acc"][sensorID][t-1,0]
                if Delsys["Acc"][sensorID][t,1] == 0:
                    Delsys["Acc"][sensorID][t,1] = Delsys["Acc"][sensorID][t-1,1]
                if Delsys["Acc"][sensorID][t,2] == 0:
                    Delsys["Acc"][sensorID][t,2] = Delsys["Acc"][sensorID][t-1,2]
                    
            for t in range(1,Delsys["Gyro"][sensorID].shape[0]):
                if Delsys["Gyro"][sensorID][t,0] == 0:
                    Delsys["Gyro"][sensorID][t,0] = Delsys["Gyro"][sensorID][t-1,0]
                if Delsys["Gyro"][sensorID][t,1] == 0:
                    Delsys["Gyro"][sensorID][t,1] = Delsys["Gyro"][sensorID][t-1,1]
                if Delsys["Gyro"][sensorID][t,2] == 0:
                    Delsys["Gyro"][sensorID][t,2] = Delsys["Gyro"][sensorID][t-1,2]
                    
            for t in range(1,Delsys["Mag"][sensorID].shape[0]):
                if Delsys["Mag"][sensorID][t,0] == 0:
                    Delsys["Mag"][sensorID][t,0] = Delsys["Mag"][sensorID][t-1,0]
                if Delsys["Mag"][sensorID][t,1] == 0:
                    Delsys["Mag"][sensorID][t,1] = Delsys["Mag"][sensorID][t-1,1]
                if Delsys["Mag"][sensorID][t,2] == 0:
                    Delsys["Mag"][sensorID][t,2] = Delsys["Mag"][sensorID][t-1,2]

        
        return Delsys
    
    return []

def decodeSummitPackage(filename):
    with open(filename, "rb") as file:
        rawData = file.read()
    
    PackageOnset = np.where([rawData[i:i+3] == b"BML" for i in range(len(rawData)-2)])[0]
    PackageType = [[rawData[i+4:i+8]][0] for i in PackageOnset]
    
    TriggerSequence = np.where([PackageType[i] == b'Trg ' for i in range(len(PackageType))])[0]
    
    if len(TriggerSequence) > 0:
        Triggers = dict()
        Triggers["Time"] = np.zeros((len(TriggerSequence)))
        Triggers["String"] = list()
        Triggers["Unique"] = list()
        
        for i in range(len(TriggerSequence)):
            onsetByte = int(PackageOnset[TriggerSequence[i]])
            Triggers["Time"][i] = int.from_bytes(rawData[onsetByte + 20 : onsetByte + 28], "little")
            Triggers["String"].append(rawData[onsetByte + 8 : onsetByte + 20].decode("utf-8").replace("\x00",""))
            if Triggers["String"][-1] not in Triggers["Unique"]:
                Triggers["Unique"].append(Triggers["String"][-1])

        return Triggers
    
    return []