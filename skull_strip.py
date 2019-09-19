# !pip install pydrive
#!pip install pydicom

#from google.colab import drive
#drive.mount('/content/drive')


import numpy as np
# from preprocess import skullstrip
import pydicom
import cv2
import os
from operator import itemgetter
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from skimage import img_as_ubyte
import scipy as sp
import scipy.ndimage
from scipy.ndimage import zoom
from PIL import Image
from scipy.misc import imsave
import time



def skullstrip(img,mask,flag,expand):
    se5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    for m in range(0,expand):
      mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, se5)
    img_org = img
    imgray = np.multiply(img_org,mask)
    ret, imgf = cv2.threshold(imgray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    se4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    imgf = cv2.morphologyEx(imgf, cv2.MORPH_ERODE, se3)
    imgf = cv2.morphologyEx(imgf, cv2.MORPH_OPEN, se)
    imgf = cv2.morphologyEx(imgf, cv2.MORPH_DILATE, se2)
    
    if flag == 1:
      imgf = cv2.morphologyEx(imgf, cv2.MORPH_DILATE, se)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imgf, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    if len(sizes)<2:
      return np.zeros((img.shape[0],img.shape[1]), dtype = 'uint8'), np.zeros((img.shape[0],img.shape[1]), dtype = 'uint8')
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    des = 255 - img2
    des = cv2.convertScaleAbs(img2)
    _,contour,hier = cv2.findContours(des,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)
    img2 = cv2.bitwise_not(des)
    out_mask = des
  
    des = des/255
    des = np.array(des,dtype = 'uint8')
    img_msk = np.multiply(des,img_org)
#     np.save('/content/drive/My Drive/Parkinson/testnp',out_mask)
  

 
    
    #-----------obtaining mask through convex hull ------------------

    _,contours_msk,hierarchy = cv2.findContours(out_mask,2,1)
    cnt = contours_msk[0]
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    for i in range(0,defects.shape[0]-1):
        defects[i+1][0][0] = defects[i][0][1]
    start_p = np.zeros((1,1,4), dtype = 'int32')
    start_p[0,0,:] = defects[0,0,:]
    defects = np.concatenate((defects,start_p), axis = 0)
    defects[defects.shape[0]-2][0][1] = defects[0][0][0]

    c_mask = np.zeros((out_mask.shape[0],out_mask.shape[1],3), dtype = 'uint8')
 
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(c_mask,start,end,[255,255,255],2)
    msk_thresh = cv2.cvtColor(c_mask,cv2.COLOR_RGB2GRAY)
    _, imthresh = cv2.threshold(msk_thresh, 0, 255, 80)

    for m in range(0,imthresh.shape[0]):
      df = np.diff(imthresh[m,:])
      p = np.where(df == 255)[0]
      if len(p) == 2:
        imthresh[m,p[0]+1:p[1]+1] = 255
    imthresh = imthresh/255
    imthresh = np.array(imthresh, dtype = 'uint8')
    return img_msk, imthresh

def thru_plane_position(dcm):
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos

def dicom_read(path):
    PathDicom = path
    lstFilesDCM = []  # create an empty list
    cnt = 0
    for dirName, subdirList, fileList in os.walk(PathDicom):
        lstFilesDCM = [None for _ in range(len(fileList))]
        for filename in fileList:
#             print(filename)
            if ".dcm" in filename.lower():  # check whether the file's DICOM
               #lstFilesDCM.append(os.path.join(dirName,filename))
                lstFilesDCM[cnt] = os.path.join(dirName,filename)
                cnt = cnt + 1
    try:
      ds = [pydicom.dcmread(fname) for fname in lstFilesDCM]
      ds = [(dcm, thru_plane_position(dcm)) for dcm in ds]
      ds = sorted(ds, key=itemgetter(1))
    except:
      ds = [0 for fname in lstFilesDCM]
    return ds
  
def img_normalize(img):
    img = np.array(img, dtype = 'float64')
    img = img/np.max(img)
    img = np.round(img * 255)
    out = np.uint8(img)
    return out
  
def strip_volume(ds):
    slice_msk = int(len(ds)*0.4)
    slice_h = int(len(ds) * 0.6)

    img = ds[slice_msk][0].pixel_array
    img = img_normalize(img)
    mask = np.ones((img.shape[0],img.shape[1]), dtype = 'uint8')

    out = np.zeros((img.shape[0], img.shape[1], len(ds)), dtype = 'uint8')
    out[:,:,slice_msk], mask = skullstrip(img,mask,0,0)
    mask_org = mask

    for x in reversed(range(0,slice_msk)):
        img = ds[x][0].pixel_array
        img = img_normalize(img)
        out[:,:,x], out_msk = skullstrip(img,mask,0,0)
        mask = out_msk
    
    mask = mask_org
    cntmask = 0
    mask = mask_org
    for x in range(slice_msk, slice_h):
        img = ds[x][0].pixel_array
        img = img_normalize(img)

        if x>78:
            cntmask = cntmask + 1
        out[:,:,x], out_msk = skullstrip(img,mask,x<int(0.54 * len(ds)) and x>int(0.46 * len(ds)),cntmask)
        
    img = ds[slice_h][0].pixel_array
    img = img_normalize(img)
    out[:,:,slice_h], mask = skullstrip(img,out_msk,0,0)
    
    for x in range(slice_h,len(ds)):
        img = ds[x][0].pixel_array
        img = img_normalize(img)
        out[:,:,x], out_msk = skullstrip(img,mask,0,0)
        mask = out_msk

    return out

def volume_crop(img_in, direction):
    if direction == 'sagittal':
      x_start = np.zeros(img_in.shape[2], dtype = 'int32')
      x_end = np.zeros(img_in.shape[2], dtype = 'int32')
      y_start = np.zeros(img_in.shape[2], dtype = 'int32')
      y_end = np.zeros(img_in.shape[2], dtype = 'int32')
      for m in range(0,img_in.shape[2]):
        img = img_in[:,:,m]
        _, imthresh = cv2.threshold(img, 0, 255, 80)
        _,contour,hier = cv2.findContours(imthresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        idx =0 
        if (len(contour) > 0):
          for cnt in contour:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
          x_start[m] = x
          x_end[m] = x + w
          y_start[m] = y
          y_end[m] = y+h
        else:
          x_start[m] = 1000
          x_end[m] = 0
          y_start[m] = 1000
          y_end[m] = 0
      px_start = np.min(x_start)
      px_end = np.max(x_end)
      py_start = np.min(y_start)
      py_end = np.max(y_end)
      roi = np.zeros((py_end-py_start,px_end - px_start,len(ds)), dtype = 'uint8')
      for m in range(0,img_in.shape[2]):
        img = img_in[:,:,m]
        roi[:,:,m] = img[py_start:py_end, px_start:px_end]

        
    if direction == 'axial':
      x_start = np.zeros(img_in.shape[1], dtype = 'int32')
      x_end = np.zeros(img_in.shape[1], dtype = 'int32')
      y_start = np.zeros(img_in.shape[1], dtype = 'int32')
      y_end = np.zeros(img_in.shape[1], dtype = 'int32')
      for m in range(0,img_in.shape[1]):
        img = img_in[:,m,:]
        _, imthresh = cv2.threshold(img, 0, 255, 80)
        _,contour,hier = cv2.findContours(imthresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        idx =0 
        if (len(contour) > 0):
          for cnt in contour:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
          x_start[m] = x
          x_end[m] = x + w
          y_start[m] = y
          y_end[m] = y+h
        else:
          x_start[m] = 1000
          x_end[m] = 0
          y_start[m] = 1000
          y_end[m] = 0
      px_start = np.min(x_start)
      px_end = np.max(x_end)
      py_start = np.min(y_start)
      py_end = np.max(y_end)
      roi = np.zeros((py_end-py_start,img_in.shape[1],px_end - px_start), dtype = 'uint8')
      for m in range(0,img_in.shape[1]):
        img = img_in[:,m,:]
        roi[:,m,:] = img[py_start:py_end, px_start:px_end]
    return roi

  
def img_resize(img,x,y,z):
  out_resize = np.zeros((x,y,img.shape[2]), dtype = 'uint8')
  out_resizef = np.zeros((x,y,z), dtype = 'uint8')
  print('the size of out_resize', out_resizef.shape)
  print('size:', out_resizef[:,1,:].shape)
  for m in range(0,img.shape[2]):
    out_resize[:,:,m] = cv2.resize(img[:,:,m],(y,x))
  for m in range(0,out_resize.shape[1]):
    out_resizef[:,m,:] = cv2.resize(out_resize[:,m,:],(z,x))
  return out_resizef


data_path = '/content/drive/My Drive/Control Dataset/PPMI'
save_path = '/content/drive/My Drive/Control Dataset/'
print('Reading data path ... \n')
a = os.listdir(data_path)
path_list = [None for _ in range(len(a))]
for x in range(0,len(a)):
    path = data_path + '/' + a[x]
    c = os.listdir(path)
    while os.path.isdir(path + '/' + c[0]) == True:
        path = path + '/' + c[0]
        c = os.listdir(path)
        path_list[x] = path
print('Reading DICOM image ...')

index = 0
for m in range(0,len(path_list)):
  start = time.time()
  print('*************Processing subject number :' + str(m)+'/'+str(len(path_list)) + '-->' +str(a[m])+'<--'+'******************')
  ds = dicom_read(path_list[m])
  print('Skull stripping...')
  try:
    out = strip_volume(ds)
    print('Crop and Resizing...')
    out_roi = volume_crop(out,'sagittal')
    out_roif = volume_crop(out_roi,'axial')

    out_resize = img_resize(out_roif,80,100,108)
    np.save(save_path +'/' + str(m)+ '.npy',out_resize)
    end = time.time()
    print('Elapsed time:' + str(end - start) + 's')
    print('\n')
  except:
    print('Warning: The Metadata for DCM file is corrupted and could not be read!!!!!!')
    print('Skipping this item....')