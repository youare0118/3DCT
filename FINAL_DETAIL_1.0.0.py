#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #
import pandas as pd #
import dicom, pydicom #
import os #
import scipy.ndimage 
from scipy import ndimage
import matplotlib.pyplot as plt


# In[2]:


import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.morphology import binary_dilation, binary_opening, disk, remove_small_objects
from skimage.measure import label, regionprops, perimeter
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage import measure, feature
from skimage.segmentation import clear_border
from matplotlib import colors
from scipy import ndimage
from scipy.ndimage import rotate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


# DICOM 파일 업로드
# 지정된 경로에서 DICOM 파일들을 로드하고, 해당 파일들을 스캔 이미지로 변환하는 함수
# 최종적으로 변환된 이미지 볼륨('image'), 스캔 이미지의 픽셀 간격, 슬레이스 두께를 반환
def load_scan(path):
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = slices[0].SliceThickness

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    return image, pixel_spacing, slice_thickness


# In[4]:


# INPUT DATA 경로 설정
# DICOM 파일(INPUT DATA) 경로 설정 및 이미지와 관련된 정보를 변수에 저장
path = "input/sample_images/patients_one"
image, pixel_spacing, slice_thickness = load_scan(path)


# In[5]:


# 이미지 시각화 및 출력
# 경로에 있는 DICOM 이미지를 표시하고 화면에 출력
def visualize_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


# In[6]:


# 이미지 전처리 및 시각화
# 이미지 스케일링 이후 확인을 위한 첫 번째 슬레이스 이미지 시각화
first_slice = image[0]  # 첫 번째 슬라이스 이미지 선택
visualize_image(first_slice) 

# 이미지 스케일링
scaled_image = image.astype(np.float32)
scaled_image = (scaled_image - np.min(scaled_image)) / (np.max(scaled_image) - np.min(scaled_image))


# In[7]:


# 이미지 3D 볼륨 데이터 시각화
# 데이터를 시각화 하여 10X10 격자 형태로 출력
def visualize_volume(volume):
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 30))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(volume[i], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
# 전체 3D 볼륨 시각화
visualize_volume(scaled_image)


# In[8]:


# 이미지 데이터를 HU로 변환
# DCIOM 파일을 불러와 이미지 데이터를 HU값으로 변환하는 함수와 뼈 영역을 추출하는 함수
def convert_to_hu(image, slope, intercept):
    hu_image = image * slope + intercept
    return hu_image

# 뼈 영역 추출
def extract_bone(image, threshold):
    bone_image = image > threshold
    return bone_image

slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]


# In[11]:


# 자동 임계값 선택 알고리즘
# 이미지 데이터에 대한 임계값을 계산하고 DICOM 이미지의 정보를 가져옴
def auto_threshold(image):
    # 히스토그램 계산
    hist, bins = np.histogram(image.flatten(), bins=256, range=[np.min(image), np.max(image)])

    # 히스토그램 누적합 계산
    cumsum = hist.cumsum()

    # 누적합 정규화
    cumsum_normalized = cumsum / float(cumsum[-1])

    # 픽셀 임계값 선택
    threshold = bins[np.where(cumsum_normalized >= 0.99)[0][0]]-250
    
    # 픽셀값을 사용한 hu값 변환 과정 HU = (Pixel Value - Pixel Value of Water) × Scale Factor
    # HU = (픽셀값 - 흑백물질의 픽셀값[물의 픽셀값]) X 스케일 팩터
    hu_threshold = (threshold-0) * 1

    return hu_threshold

# DICOM 이미지 정보 가져오기
dicom_info = slices[0]  # 첫 번째 슬라이스


# In[13]:


# DICOM 이미지 정보 가져오기
dicom_info = slices[0]  # 첫 번째 슬라이스 정보 사용
slope = dicom_info.RescaleSlope
intercept = dicom_info.RescaleIntercept

# HU 변환 적용
hu_image = convert_to_hu(image, slope, intercept)

# 자동 임계값 선택
threshold = auto_threshold(hu_image)

# 뼈 영역 추출
bone_image = extract_bone(hu_image, threshold)
expanded_bone_image = np.repeat(bone_image, 5, axis=0)

# 뼈 영역 시각화
visualize_volume(expanded_bone_image)


# In[14]:


# 이진화된 CT이미지들을 3DNUMPY 배열화
bone_volume = expanded_bone_image.transpose(2, 1, 0).astype(np.uint8)

# 변환된 3D numpy 배열의 형태와 데이터 타입 확인
print(bone_volume.shape)
print(bone_volume.dtype)


# In[22]:


def plot_3d_model(volume):
    # 뼈 영역의 좌표와 면을 계산합니다.
    verts, faces, _, _ = measure.marching_cubes(volume, level=0)

    # 3D 모델 생성
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 면을 이용하여 삼각형 모양의 3D 모델을 생성합니다.
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.8, 0.8, 0.8]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    # 축 범위 설정 (역으로 설정)
    ax.set_xlim(volume.shape[0], 0)
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(volume.shape[2], 0)
    
    plt.show()


plot_3d_model(bone_volume)
# bone_volume을 3D 플롯으로 생성


# In[23]:


#노이즈 제거
def clean_3d_model(input_image):
    min_size = 8000  # 제거할 객체의 최소 크기
    cleaned_image = remove_small_objects(input_image, min_size=min_size)
    return cleaned_image
    
cleaned_image = clean_3d_model(expanded_bone_image)

plot_3d_model(cleaned_image)


# In[25]:


from stl import mesh
from skimage import measure

def save_as_stl(volume, output_path):
    # STL 파일로 변환할 3D 모델 생성
    vertices, faces = extract_surface(volume)
    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[face[j]]

    # STL 파일로 저장
    mesh_data.save(output_path)

# 3D 모델의 표면 추출
def extract_surface(volume):
    # Marching Cubes 알고리즘을 사용하여 등치 표면 추출
    threshold = 0.5
    verts, faces, _, _ = measure.marching_cubes(volume, level=threshold)

    return verts, faces

# 3D 모델을 STL 파일로 저장
output_path = "output/cleanmodel6.stl"
save_as_stl(cleaned_image, output_path)


# In[ ]:




