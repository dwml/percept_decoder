# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:05:53 2023

@author: Jackson
"""

import os, sys
import json
from pydicom.fileset import FileSet
import numpy as np
import nibabel as nib
#from nibabel.nicom import csareader

def convertDICOM2NifTi(series):
    ds = series[0].load()
    seriesName = ds.SeriesDescription
    try:
        zSlicePixelDistance = np.float32(ds.SliceThickness)
        orientation = np.array(ds.ImageOrientationPatient).reshape((2,3))
        sliceOrigins = np.zeros((len(series), 3))
        img = np.zeros((len(series), int(ds.Rows), int(ds.Columns)))
    except:
        print(f"Skipping {seriesName}, not MRIs")
        return
    
    if len(series) == 1:
        print("Skipping MOSAIC Image for now")
        return
    
    for instance in series:
        ds = instance.load()
        img[int(ds.InstanceNumber)-1,:,:] = ds.pixel_array
        sliceOrigins[int(ds.InstanceNumber)-1,:] = ds.ImagePositionPatient
    
    ColLength = orientation[0,:] * int(ds.Columns) * ds.PixelSpacing[1]
    RowLength = orientation[1,:] * int(ds.Rows) * ds.PixelSpacing[0]
    ZLength = sliceOrigins[-1,:] - sliceOrigins[0,:]
    SliceOrigin = sliceOrigins[0,:]
    
    pixelSpacing = np.zeros(3)
    pixelSpacing[np.argmax(np.abs(ColLength))] = ds.PixelSpacing[1]
    pixelSpacing[np.argmax(np.abs(RowLength))] = ds.PixelSpacing[0]
    pixelSpacing[np.argmax(np.abs(ZLength))] = zSlicePixelDistance
    
    currentOrientation = [np.argmax(np.abs(ZLength)), np.argmax(np.abs(RowLength)), np.argmax(np.abs(ColLength))]
    img = img.transpose(np.argsort(currentOrientation))
    
    currentDirection = np.array([ZLength, RowLength, ColLength])
    imgDirection = currentDirection[np.argsort(currentOrientation),:]
    
    AnatomicalOrientation = "BIPED"
    try:
        AnatomicalOrientation = ds[0x0010,0x2210]
    except:
        pass
    
    if AnatomicalOrientation != "BIPED":
        raise Exception(f"The Anatomical Orientation [{AnatomicalOrientation}] is present. Nonstandard BIPED Orientation")
    
    # BIPED is standard orientation. Increment X => Left and Increment Y => Posterior and Increment Z => Top
    if AnatomicalOrientation == "BIPED":
        img = np.flip(img,0)
        img = np.flip(img,1)
        SliceOrigin[0] = -(SliceOrigin[0] + imgDirection[0,0])
        SliceOrigin[1] = -(SliceOrigin[1] + imgDirection[1,1])
    
    if imgDirection[2,2] < 0:
        raise Exception("Z-Index Decrement")
    
    tform = np.eye(4)
    tform[:3,3] = SliceOrigin
    
    header = nib.Nifti1Header()
    header.set_sform(tform)
    header.set_qform(tform)
    header.set_data_shape(img.shape)
    header.set_zooms(pixelSpacing)
    header.set_xyzt_units(2)
    header.set_data_dtype(np.float32)
    nii = nib.Nifti1Image(img, header.get_best_affine(), header)
    return nii, AnatomicalOrientation

def convertAtlas2NifTi(ds):
    orientation = np.array(ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient).reshape((2,3))
    dsPixelSpacing = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    img = ds.pixel_array
    
    sliceOrigins = np.zeros((len(ds.PerFrameFunctionalGroupsSequence), 3))
    for index, segment in enumerate(ds.PerFrameFunctionalGroupsSequence):
        sliceOrigins[index,:] = segment.PlanePositionSequence[0].ImagePositionPatient
    zSlicePixelDistance = np.mean(np.diff(sliceOrigins[:,2]))
    
    ColLength = orientation[0,:] * int(ds.Columns) * dsPixelSpacing[1]
    RowLength = orientation[1,:] * int(ds.Rows) * dsPixelSpacing[0]
    ZLength = sliceOrigins[-1,:] - sliceOrigins[0,:]
    SliceOrigin = sliceOrigins[0,:]
    
    pixelSpacing = np.zeros(3)
    pixelSpacing[np.argmax(np.abs(ColLength))] = dsPixelSpacing[1]
    pixelSpacing[np.argmax(np.abs(RowLength))] = dsPixelSpacing[0]
    pixelSpacing[np.argmax(np.abs(ZLength))] = np.abs(zSlicePixelDistance)
    
    currentOrientation = [np.argmax(np.abs(ZLength)), np.argmax(np.abs(RowLength)), np.argmax(np.abs(ColLength))]
    img = img.transpose(np.argsort(currentOrientation))
    
    currentDirection = np.array([ZLength, RowLength, ColLength])
    imgDirection = currentDirection[np.argsort(currentOrientation),:]
    
    AnatomicalOrientation = "BIPED"
    try:
        AnatomicalOrientation = ds[0x0010,0x2210]
    except:
        pass
    
    if AnatomicalOrientation != "BIPED":
        raise Exception(f"The Anatomical Orientation [{AnatomicalOrientation}] is present. Nonstandard BIPED Orientation")
    
    # BIPED is standard orientation. Increment X => Left and Increment Y => Posterior and Increment Z => Top
    if AnatomicalOrientation == "BIPED":
        img = np.flip(img,0)
        img = np.flip(img,1)
        SliceOrigin[0] = -(SliceOrigin[0] + imgDirection[0,0])
        SliceOrigin[1] = -(SliceOrigin[1] + imgDirection[1,1])
    
    if imgDirection[2,2] < 0:
        img = np.flip(img,2)
        SliceOrigin[2] = SliceOrigin[2] + ZLength[2]
    
    tform = np.eye(4)
    tform[:3,3] = SliceOrigin
    
    header = nib.Nifti1Header()
    header.set_sform(tform)
    header.set_qform(tform)
    header.set_data_shape(img.shape)
    header.set_zooms(pixelSpacing)
    header.set_xyzt_units(2)
    header.set_data_dtype(np.float32)
    nii = nib.Nifti1Image(img, header.get_best_affine(), header)
    return nii, AnatomicalOrientation

def decodeBrainLabDCM(DICOMDIR_File):
    Data = dict()
    DICOMDIR = FileSet(DICOMDIR_File)
    SourceFolder = os.path.dirname(os.path.abspath(DICOMDIR_File))
    
    Data["Atlas"] = dict()
    Data["NifTi"] = dict()
    Data["Trajectories"] = dict()
    
    Trajectories = DICOMDIR.find(SeriesDescription="Points and Trajectories")
    for instance in Trajectories:
        ds = instance.load()
        referenceDICOMs = DICOMDIR.find(SeriesInstanceUID=ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
        seriesName = referenceDICOMs[0].load().SeriesDescription
        if not seriesName in Data["NifTi"].keys():
            nii, orientation = convertDICOM2NifTi(referenceDICOMs)
            nib.save(nii, SourceFolder + os.path.sep + seriesName.replace(">","_").replace("<","_") + ".nii.gz")
            Data["NifTi"][seriesName] = SourceFolder + os.path.sep + seriesName + ".nii.gz"
        
        coordinates = ds.SurfaceSequence[0].SurfacePointsSequence[0].PointCoordinatesData
        coordinates = np.frombuffer(bytearray(coordinates), dtype=np.float32).reshape([2,3])
        if orientation == "BIPED":
            coordinates[:,:2] *= -1
            
        coordinateName = ds.SegmentSequence[0].SegmentLabel
        Data["Trajectories"][coordinateName] = {
            "referenceImage": seriesName,
            "positions": coordinates.tolist()
        }
        
    Objects = DICOMDIR.find(SeriesDescription="Objects")
    for instance in Objects:
        ds = instance.load()
        referenceDICOMs = DICOMDIR.find(SeriesInstanceUID=ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
        seriesName = referenceDICOMs[0].load().SeriesDescription
        if not seriesName in Data["NifTi"].keys():
            nii, orientation = convertDICOM2NifTi(referenceDICOMs)
            nib.save(nii, SourceFolder + os.path.sep + seriesName.replace(">","_").replace("<","_") + ".nii.gz")
            Data["NifTi"][seriesName] = SourceFolder + os.path.sep + seriesName + ".nii.gz"
        
        segmentName = ds.SegmentSequence[0].SegmentLabel
        nii, orientation = convertAtlas2NifTi(ds)
        nib.save(nii, SourceFolder + os.path.sep + segmentName + ".nii.gz")
        
        Data["Atlas"][segmentName] = {
            "referenceImage": seriesName,
            "object": SourceFolder + os.path.sep + segmentName + ".nii.gz"
        }
    
    with open(SourceFolder + os.path.sep + "output.json", "w+") as fid:
        json.dump(Data, fid)

if __name__ == "__main__":
    DICOMDIR_File = sys.argv[1]
    decodeBrainLabDCM(DICOMDIR_File)
