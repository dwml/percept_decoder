import sys
import numpy as np
import nibabel as nib
import time
import struct
from scipy import io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
import subprocess
from stl import mesh
from skimage import measure
import trimesh

from utility.SignalProcessingUtility import rssq

def decodeUFVTK(filename):
    """Decode UFVTK file into header and image field

    :param filename: Full/Relative Path to the UFVTK file you want to decode.
    :type filename: string, require
    :return: header dictionary describing the image data
    :rtype: dict
    :return: 3D numpy array (signed 16-bit) for raw image data.
    :rtype: numpy.ndarray
    """

    """
    Decode UFVTK file into header and image field

    Parameters:
    filename (string): Full/Relative Path to the UFVTK file you want to decode.

    Returns:
    info (dict): header dictionary describing the image data
    img (ndarray): typically a 3D numpy array (signed 16-bit) for raw image data.
    """

    with open(filename, "rb") as rawFile:
        rawData = rawFile.read()
    
    info = dict()
    info["magic"] = np.frombuffer(rawData, dtype="S1", count=4, offset=0)
    info["size"] = np.frombuffer(rawData, dtype=">u4", count=3, offset=4)
    info["dimension"] = np.frombuffer(rawData, dtype=">f4", count=3, offset=16)
    info["pixelOrigin"] = np.frombuffer(rawData, dtype=">f4", count=3, offset=28)
    info["brw2pix"] = np.frombuffer(rawData, dtype=">f4", count=9, offset=40).reshape((3,3))
    info["pix2brw"] = np.frombuffer(rawData, dtype=">f4", count=9, offset=76).reshape((3,3))
    info["suggestions"] = np.frombuffer(rawData, dtype=">i4", count=2, offset=112)
    info["xfrm"] = np.frombuffer(rawData, dtype=">f4", count=15, offset=120).reshape((5,3))
    
    img_raw = np.frombuffer(rawData, dtype=">i2", count=np.prod(info["size"]), offset=180).reshape(info["size"], order="F")
    img = img_raw - np.min(img_raw)
    return info, img

def encodeUFVTK(nii, filename):
    """
    Encode NifTi file as a standard UFVTK file. 

    Parameters:
    nii (nibabel.nifti1.NifTi1Image): Nibabel NifTi1 object.
    filename (string): Full/Relative Path to the UFVTK file you want to save.

    Returns:
    None
    """
    def swap32buffer(value):
        return np.array(struct.unpack("<i", struct.pack(">i", value)), dtype=np.int32).tobytes()
    def swap16buffer(value):
        return np.array(struct.unpack(f"<{len(value)}h", struct.pack(f">{len(value)}h", *value)), dtype=np.int16).tobytes()
    def swapSinglebuffer(value):
        return np.array(struct.unpack("<f", struct.pack(">f", value)), dtype=np.float32).tobytes()

    rawData = nii.get_fdata()

    xdim = nii.header["dim"][1]
    ydim = nii.header["dim"][2]
    zdim = nii.header["dim"][3]
    xoffset = int((512-xdim)/2)
    yoffset = int((512-ydim)/2)
    vtkImage = np.zeros((512,512,zdim), dtype=np.int16)
    for k in range(zdim):
        vtkImage[slice(xoffset,xoffset+xdim),slice(yoffset,yoffset+ydim),k] = rawData[:,:,k]
    vtkImage = np.reshape(vtkImage,[512*512*zdim], order="F")

    with open(filename,"wb") as file:
        # Header
        file.write(np.array([83,86,84,75],dtype=np.int8).tobytes())
        # Dimensions
        file.write(swap32buffer(512))
        file.write(swap32buffer(512))
        file.write(swap32buffer(zdim))
        # pixel dimension
        for value in nii.header["pixdim"][1:4]:
            file.write(swapSinglebuffer(value))
        # pixel origin
        affine = nii.header.get_sform()
        origin = affine[:3,-1]
        file.write(swapSinglebuffer(-xoffset*nii.header["pixdim"][1]+origin[0]))
        file.write(swapSinglebuffer(-yoffset*nii.header["pixdim"][2]+origin[1]))
        file.write(swapSinglebuffer(origin[2]))
        # brw transform
        file.write(np.array([0,0,0,0,63,128,0,0,0,0,0,0,63,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0],dtype=np.int8).tobytes())
        file.write(np.array([0,0,0,0,63,128,0,0,0,0,0,0,63,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0],dtype=np.int8).tobytes())
        # suggest mean & window
        file.write(np.array([0,0,6,33],dtype=np.int8).tobytes())
        file.write(np.array([0,0,3,77],dtype=np.int8).tobytes())
        # xfrm
        file.write(np.array([0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0,63,128,0,0,63,128,0,0,63,128,0,0],dtype=np.int8).tobytes())
        # Raw Image Matrx, big endian 16-bit int with MATLAB's Column-wise Ordering
        file.write(swap16buffer(vtkImage))

def computeTransformMatrix(xfrm):
    """
    Convert UFVTK xfrm field to standard Transformation Matrix (3D)

    Parameters:
    xfrm (4x4 matrix): xfrm extracted from UFVTK xfrm file (or img header) that describe image references.

    Returns:
    tform (4x4 matrix): Affine-3D Transformation Matrix
    """
    rotateMat = np.array([xfrm[2,[1,0,2]],xfrm[1,[1,0,2]],xfrm[3,[1,0,2]]])
    rotationVec = Rotation.from_matrix(rotateMat).as_euler("xyz",degrees=True)
    translationVec = xfrm[0,:] @ xfrm[1:4,:].T
    scaleVec = xfrm[-1,[1,0,2]]
    tform = computeTransformationMatrix(scaleVec, translationVec[[1,0,2]]*[1,1,1], rotationVec*[-1,-1,1])
    return tform.T

def loadCRW(filename):
    """
    Extract surgical planning parameters from CRW file

    Parameters:
    filename (string): Full/Relative Path to the CRW file you want to read.

    Returns:
    crw (dict): Key-value pair of parameters read from CRW file.
    """
    with open(filename) as file:
        lines = file.read().splitlines()

    crw = {}
    n = 0
    while n < len(lines):
        if lines[n].startswith("AC Point") > 0:
            valid = lines[n+1]
            AP = float(lines[n+2].split(" = ")[1])
            LT = float(lines[n+3].split(" = ")[1])
            AX = float(lines[n+4].split(" = ")[1])
            crw["AC"] = np.array([LT, AP, AX])
            n += 5
        elif lines[n].startswith("PC Point") > 0:
            valid = lines[n+1]
            AP = float(lines[n+2].split(" = ")[1])
            LT = float(lines[n+3].split(" = ")[1])
            AX = float(lines[n+4].split(" = ")[1])
            crw["PC"] = np.array([LT, AP, AX])
            n += 5
        elif lines[n].startswith("Ctrln Point") > 0:
            valid = lines[n+1]
            AP = float(lines[n+2].split(" = ")[1])
            LT = float(lines[n+3].split(" = ")[1])
            AX = float(lines[n+4].split(" = ")[1])
            crw["MC"] = np.array([LT, AP, AX])
            n += 5
        elif lines[n].startswith("Target Point") > 0:
            valid = lines[n+1]
            AP = float(lines[n+2].split(" = ")[1])
            LT = float(lines[n+3].split(" = ")[1])
            AX = float(lines[n+4].split(" = ")[1])
            crw["TargetPt"] = np.array([LT, AP, AX])
            n += 5
        elif lines[n].startswith("Entry Point") > 0:
            valid = lines[n+1]
            AP = float(lines[n+2].split(" = ")[1])
            LT = float(lines[n+3].split(" = ")[1])
            AX = float(lines[n+4].split(" = ")[1])
            crw["EntryPt"] = np.array([LT, AP, AX])
            n += 5
        elif lines[n].startswith("Func Targ Point") > 0:
            valid = lines[n+1]
            AP = float(lines[n+2].split(" = ")[1])
            LT = float(lines[n+3].split(" = ")[1])
            AX = float(lines[n+4].split(" = ")[1])
            crw["FuncTargetPt"] = np.array([LT, AP, AX])
            n += 5
        elif lines[n].startswith("Orientation") > 0:
            crw["Orientation"] = float(lines[n].split(" ")[-1])
            n += 1
        elif lines[n].startswith("ACPC Angle") > 0:
            crw["ACPCAngle"] = float(lines[n].split(" ")[-1])
            n += 1
        elif lines[n].startswith("Cline Angle") > 0:
            crw["ClineAngle"] = float(lines[n].split(" ")[-1])
            n += 1
        else:
            n += 1

    return crw

def getACPCTransform(crw):
    """
    Extract surgical planning parameters from CRW file

    Parameters:
    crw (dict): Key-value pair of parameters read from CRW file.

    Returns:
    tform (4x4 matrix): Affine-3D Transformation Matrix
    """
    
    Origin = (crw["AC"] + crw["PC"]) / 2
    temp = (crw["MC"] - Origin) / rssq(crw["MC"] - Origin, axis=0)
    J = (crw["AC"] - crw["PC"]) / rssq(crw["AC"] - crw["PC"], axis=0)
    I = np.cross(J, temp) / rssq(np.cross(J, temp), axis=0)
    K = np.cross(I, J) / rssq(np.cross(I, J), axis=0)
    
    Old = np.array([np.concatenate((Origin+I, [1])), np.concatenate((Origin+J, [1])), np.concatenate((Origin+K, [1])), np.concatenate((Origin, [1]))])
    New = np.eye(4)
    New[:3, -1] = 1
    tform = np.linalg.lstsq(Old, New, rcond=None)[0]
    return tform

def makeNifTi(info, img):
    """
    Convert UFVTK header and raw data into Nibabel NifTi1 object

    Parameters:
    info (dict): header dictionary describing the image data
    img (ndarray): typically a 3D numpy array (signed 16-bit) for raw image data.

    Returns:
    nii (nibabel.nifti1.NifTi1Image): Nibabel NifTi1 object.
    """
    
    tform = computeTransformMatrix(info["xfrm"])
    tform[:3,-1] += info["pixelOrigin"]

    header = nib.Nifti1Header()
    header.set_qform(tform)
    header.set_sform(tform)
    header.set_data_shape(img.shape)
    header.set_zooms(info["dimension"])
    header.set_xyzt_units(2)
    nii = nib.Nifti1Image(img, header.get_best_affine(), header)
    return nii

def transformNifTi(nii, tform, refDimension=None, dtype=np.int16):
    """
    Python implementation of apply_transformation given a standard affine 3D matrix.
    It is only being used as backup if ANTs is not available, as ANTs is significantly faster and consume less memory.

    Parameters:
    nii (nibabel.nifti1.NifTi1Image): Nibabel NifTi1 object.
    tform (4x4 matrix): Affine-3D Transformation Matrix

    Returns:
    nii (nibabel.nifti1.NifTi1Image): Transformed Nibabel NifTi1 object.
    """
    img = nii.get_fdata().astype(dtype)
    affine = nii.header.get_sform()
    origin = affine[:3,-1]
    originalDimension = [np.arange(nii.header["dim"][n]) * nii.header["pixdim"][n] + origin[n-1] for n in range(1,4)]

    if refDimension is None:
        refDimension = originalDimension
    elif type(refDimension) == nib.nifti1.Nifti1Image:
        affine = refDimension.header.get_sform()
        origin = affine[:3,-1]
        refDimension = [np.arange(refDimension.header["dim"][n]) * refDimension.header["pixdim"][n] + origin[n-1] for n in range(1,4)]
    else:
        affine = np.eye(4)
        for i in range(3):
            affine[i,i] = np.mean(np.diff(refDimension[i]))
        affine[:3,-1] = [refDimension[i][0] for i in range(3)]
    
    meshGrids = np.array([(x,y,z,1) for z in refDimension[2] for y in refDimension[1] for x in refDimension[0]], dtype=dtype)
    warpInterp = RegularGridInterpolator(originalDimension, img, bounds_error=False, fill_value=np.min(img))
    queryPoints = np.matmul(meshGrids , np.linalg.inv(tform))[:,0:3]
    del(meshGrids)
    warpedImage = warpInterp(queryPoints).reshape((len(refDimension[0]),len(refDimension[1]),len(refDimension[2])), order="F").astype(dtype)
    del(queryPoints)
    warpedNifTi = nib.nifti1.Nifti1Image(warpedImage, affine)
    return warpedNifTi

def trimNifTi(nii):
    header = nii.header
    dimension = header["pixdim"][1:4]
    qform = header.get_qform()
    img = nii.get_fdata()
    
    xDataAvailable = np.array([not np.all(img[i,:,:] == 0) for i in range(img.shape[0])])
    firstIndex = np.where(xDataAvailable)[0][0]
    qform[0,3] = qform[0,3] + dimension[0] * firstIndex
    img = img[xDataAvailable,:,:]
    
    xDataAvailable = np.array([not np.all(img[:,i,:] == 0) for i in range(img.shape[1])])
    firstIndex = np.where(xDataAvailable)[0][0]
    qform[1,3] = qform[1,3] + dimension[1] * firstIndex
    img = img[:,xDataAvailable,:]
    
    xDataAvailable = np.array([not np.all(img[:,:,i] == 0) for i in range(img.shape[2])])
    firstIndex = np.where(xDataAvailable)[0][0]
    qform[2,3] = qform[2,3] + dimension[2] * firstIndex
    img = img[:,:,xDataAvailable]
    
    header.set_qform(qform)
    nii = nib.Nifti1Image(img, header.get_qform(), header)
    return nii

def antsApplyTransform(INPUT_IMAGE, OUTPUT_IMAGE, TRANSFORMATION_MAT, referenceImage=None, format="short", transformType=0):
    """
    antsRegistration -v 1 -d 3 -o [TRANSFORMATION_PREFIX, TRANSFORMED_FILE  ] \ 
        -n Linear \ 
        -u \ 
        -f 1 \ 
        -w [0.005,0.095] \ 
        -a 1 \ 
        -r [FIXED_IMAGE,MOVE_IMAGE,1] \ 
        -t Affine[0.25] \ 
        -c [1000x500x250x0,1e-8,10] \ 
        -s 4x3x2x1vox \ 
        -f 12x8x4x1 \ 
        -m MI[FIXED_IMAGE,MOVE_IMAGE,1,32,Random,0.25]

    antsApplyTransforms -i INPUT_IMAGE \ 
        -o OUTPUT_IMAGE \ 
        -r REFERENCE_IMAGE \ 
        -u short \ 
        -t [TRANSFORMATION_MAT,0]
    """
    if not referenceImage:
        subprocess.call(f"antsApplyTransforms -i {INPUT_IMAGE} -o {OUTPUT_IMAGE} \
                        -r {INPUT_IMAGE} -u {format} -t [{TRANSFORMATION_MAT},{transformType}]",
                    shell=True)
    else:
        subprocess.call(f"antsApplyTransforms -i {INPUT_IMAGE} -o {OUTPUT_IMAGE} \
                        -r {referenceImage} -u {format} -t [{TRANSFORMATION_MAT},{transformType}]",
                    shell=True)

def generateANTsTransformation(tform, filename):
    AffineTransform_double_3_3 = np.linalg.inv(tform)[:,:3].reshape((12,1))
    AffineTransform_double_3_3[9] *= -1
    AffineTransform_double_3_3[10] *= -1
    sio.savemat(filename, {"AffineTransform_double_3_3": AffineTransform_double_3_3, 
                           "fixed": np.zeros((3,1))}, format="4")

def loadAtlasTransform(filename):
    fmrisavedata = sio.loadmat(filename, simplify_cells=True)
    savestruct = fmrisavedata["savestruct"]
    transform = dict()
    if "mvmtleft" in savestruct.keys():
        scale = savestruct["scaleleft"][[1,0,2]].astype(float)
        translation = savestruct["mvmtleft"][[1,0,2]].astype(float)
        rotation = savestruct["rotationleft"][[1,0,2]].astype(float)
        transform["Left"] = computeTransformationMatrix(scale, translation, rotation)

    if "mvmtright" in savestruct.keys():
        scale = savestruct["scaleright"][[1,0,2]].astype(float)
        translation = savestruct["mvmtright"][[1,0,2]].astype(float)
        rotation = savestruct["rotationright"][[1,0,2]].astype(float)
        # UF Software actually invert the following parameters on display.
        translation[0] *= -1
        rotation[1] *= -1
        rotation[2] *= -1
        transform["Right"] = computeTransformationMatrix(scale, translation, rotation)
    
    return transform
    
def computeTransformationMatrix(scale, translation, rotation):
    Rotation = np.deg2rad(rotation)
    xRotation = np.array([[1,0,0,0], 
                [0,np.cos(Rotation[0]),-np.sin(Rotation[0]),0], 
                [0,np.sin(Rotation[0]),np.cos(Rotation[0]),0], 
                [0,0,0,1]])
    yRotation = np.array([[np.cos(Rotation[1]),0,np.sin(Rotation[1]),0], 
                [0,1,0,0],
                [-np.sin(Rotation[1]),0,np.cos(Rotation[1]),0],
                [0,0,0,1]])
    zRotation = np.array([[np.cos(Rotation[2]),-np.sin(Rotation[2]),0,0],
                [np.sin(Rotation[2]),np.cos(Rotation[2]),0,0],
                [0,0,1,0],
                [0,0,0,1]])

    Scale = np.eye(4)
    for i in range(3):
        Scale[i,i] = scale[i]

    Translation = np.eye(4)
    for i in range(3):
        Translation[i,-1] = translation[i]

    tform = xRotation @ yRotation @ zRotation @ Scale @ Translation
    return tform

def volume2stl(atlas, outFile, threshold=0.4):
    voxelData = atlas.get_fdata()
    verts, faces, normals, values = measure.marching_cubes(voxelData, threshold)
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces);
    trimesh.smoothing.filter_taubin(mesh);
    mesh.apply_transform(atlas.affine);
    mesh.apply_transform([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]);
    mesh.export(outFile, file_type="stl");