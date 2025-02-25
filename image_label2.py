import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import copy, deepcopy

class Image_Label(QLabel):
    mpsignal = pyqtSignal(str)

    def __init__(self, parent):
        super(QLabel, self).__init__(parent)
        self.setMinimumSize(1, 1)
        self.setMouseTracking(False)
        self.image = None
        self.processedSlice = None
        self.processedImage = None
        self.voxel_size = None
        self.imgr, self.imgc = None, None
        self.imgpos_x, self.imgpos_y = None, None
        self.pos_x = 20
        self.pos_y = 20
        self.imgr, self.imgc = None, None
        # 遇到list就停，圖上的顯示白色只是幌子
        self.pos_xy = []
        # 十字的中心點！每個QLabel指定不同中心點，這樣可以用一樣的paintevent function
        self.crosscenter = [0, 0]
        self.mouseclicked = None
        self.sliceclick = False
        # 決定用哪種paintEvent的type, general代表一般的
        self.type = 'general'
        self.slice_loc = [0, 0, 0]
        self.slice_loc_restore = [0, 0, 0]
        self.mousein = False
        self.color = False

        self.axial_index = 0
        self.sagittal_index = 0
        self.coronal_index = 0

    def mousePressEvent(self, event: QMouseEvent):
        self.crosscenter[0] = event.x()
        self.crosscenter[1] = event.y()

        self.mpsignal.emit(self.type)

        self.slice_loc_restore = self.slice_loc.copy()
        self.update()

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        self.display_image()

    def display_image(self, window=1):
        if self.processedImage is None:
            return

        # Determine the slice based on the specified type
        if self.type == "axial":
            self.processedSlice = self.processedImage[:, :, self.axial_index]
            x_voxel, y_voxel = (self.voxel_size[2], self.voxel_size[1])
            slice_number = "{}/{}".format(self.axial_index + 1, self.processedImage.shape[2])
        elif self.type == "sagittal":
            self.processedSlice = self.processedImage[:, self.sagittal_index, :]
            x_voxel, y_voxel = (self.voxel_size[2], self.voxel_size[0])
            slice_number = "{}/{}".format(self.sagittal_index + 1, self.processedImage.shape[1])
        elif self.type == "coronal":
            self.processedSlice = self.processedImage[self.coronal_index, :, :]
            x_voxel, y_voxel = (self.voxel_size[0], self.voxel_size[1])
            slice_number = "{}/{}".format(self.coronal_index + 1, self.processedImage.shape[0])

        # Calculate display dimensions
        height_temp = int(
            self.width() / (self.processedSlice.shape[1] * y_voxel) * (self.processedSlice.shape[0] * x_voxel))
        weight_temp = int(
            self.height() / (self.processedSlice.shape[0] * x_voxel) * (self.processedSlice.shape[1] * y_voxel))

        # Resize and pad the image for display
        if height_temp <= self.height():
            display = cv2.resize(self.processedSlice, [self.width(), height_temp], interpolation=cv2.INTER_CUBIC)
            display = cv2.copyMakeBorder(display, int(np.abs(self.height() - height_temp) / 2),
                                         int(np.abs(self.height() - height_temp) / 2), 0, 0, cv2.BORDER_CONSTANT)
        elif weight_temp <= self.width():
            display = cv2.resize(self.processedSlice, [weight_temp, self.height()], interpolation=cv2.INTER_CUBIC)
            display = cv2.copyMakeBorder(display, 0, 0, int(np.abs(self.width() - weight_temp) / 2),
                                         int(np.abs(self.width() - weight_temp) / 2), cv2.BORDER_CONSTANT)

        # Convert to RGB and add text
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        display = cv2.putText(display, slice_number, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Create QImage and display
        img = QImage(display, display.shape[1], display.shape[0], display.strides[0], QImage.Format.Format_RGB888)
        if window == 1:
            self.setPixmap(
                QPixmap.fromImage(img).scaled(self.width(), self.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.update()


    def display_image_fusion(self, window=1):
        if self.processedImage is None:
            return
        if self.type == "axial":
            self.processedSlice = self.processedImage_dis[:, :, self.axial_index]
            try:
                self.seg_dis = self.seg[:, :, self.axial_index]
            except IndexError:
                self.seg_dis = np.zeros((self.seg.shape[0], self.seg.shape[1]))
            x_voxel, y_voxel = (self.voxel_size[2], self.voxel_size[1])
            slice_number = "{}/{}".format(self.axial_index + 1, self.processedImage.shape[2])

        if self.type == "sagittal":
            self.processedSlice = self.processedImage_dis[:, self.sagittal_index, :]
            self.seg_dis = self.seg[:, self.sagittal_index, :]
            try:
                self.seg_dis = self.seg[:, self.sagittal_index, :]
            except IndexError:
                self.seg_dis = np.zeros((self.seg.shape[0], self.seg.shape[2]))

            x_voxel, y_voxel = (self.voxel_size[2], self.voxel_size[0])
            slice_number = "{}/{}".format(self.sagittal_index + 1, self.processedImage.shape[1])
            # title = "sagittal"

        if self.type == "coronal":
            self.processedSlice = self.processedImage_dis[self.coronal_index, :, :]
            try:
                self.seg_dis = self.seg[self.coronal_index, :, :]
            except IndexError:
                self.seg_dis = np.zeros((self.seg.shape[1], self.seg.shape[2]))
            x_voxel, y_voxel = (self.voxel_size[0], self.voxel_size[1])
            slice_number = "{}/{}".format(self.coronal_index + 1, self.processedImage.shape[0])
            # title = "coronal"

        height_temp = int(
            self.width() / (self.processedSlice.shape[1] * y_voxel) * (self.processedSlice.shape[0] * x_voxel))
        weight_temp = int(
            self.height() / (self.processedSlice.shape[0] * x_voxel) * (self.processedSlice.shape[1] * y_voxel))
        if height_temp <= self.height():
            display = cv2.resize(self.processedSlice, [self.width(), height_temp], interpolation=cv2.INTER_CUBIC)
            spect = cv2.resize(self.seg_dis, [self.width(), height_temp], interpolation=cv2.INTER_NEAREST)
            display = cv2.copyMakeBorder(display, int(np.abs(self.height() - height_temp) / 2),
                                         int(np.abs(self.height() - height_temp) / 2), 0, 0, cv2.BORDER_CONSTANT)
            spect = cv2.copyMakeBorder(spect, int(np.abs(self.height() - height_temp) / 2),
                                       int(np.abs(self.height() - height_temp) / 2), 0, 0, cv2.BORDER_CONSTANT)

        elif weight_temp <= self.width():
            display = cv2.resize(self.processedSlice, [weight_temp, self.height()], interpolation=cv2.INTER_CUBIC)
            spect = cv2.resize(self.seg_dis, [weight_temp, self.height()], interpolation=cv2.INTER_NEAREST)
            display = cv2.copyMakeBorder(display, 0, 0, int(np.abs(self.width() - weight_temp) / 2),
                                         int(np.abs(self.width() - weight_temp) / 2), cv2.BORDER_CONSTANT)
            spect = cv2.copyMakeBorder(spect, 0, 0, int(np.abs(self.width() - weight_temp) / 2),
                                       int(np.abs(self.width() - weight_temp) / 2), cv2.BORDER_CONSTANT)

        display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        spect = np.uint8(spect)
        spect = cv2.applyColorMap(spect, cv2.COLORMAP_HOT)
        b = spect[:, :, 0]
        g = spect[:, :, 1]
        r = spect[:, :, 2]
        spect[:, :, 0] = r
        spect[:, :, 1] = g
        spect[:, :, 2] = np.zeros_like(b)
        display = display.astype(float) / 2 + spect.astype(float) / 2
        display = display.astype(np.uint8)
        display = cv2.putText(display, slice_number, (0, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img = QImage(display, display.shape[1], display.shape[0], display.strides[0],
                     QImage.Format.Format_RGB888)

        if window == 1:
            self.setPixmap(QPixmap.fromImage(img).scaled(self.width(), self.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignHCenter)
            self.update()


    def apply_colormap(self, image_data):
        norm = plt.Normalize(vmin=image_data.min(), vmax=image_data.max())
        cmap = cm.jet  # Use the jet colormap
        colored_image = cmap(norm(image_data))
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        return colored_image

    def scale_display(self, img, relation=None):
        if relation == None:
            if img.min()<-500:
                img[img > 300] = 300
                img[img < -500] = -500
            else:
                pass
        else:
            if relation[1] <= relation[0]:
                relation[1] = relation[0] + 1
            img[img > relation[1]] = relation[1]
            img[img < relation[0]] = relation[0]
        return (img - img.min()) * 255 / (img.max() - img.min())

    def update_image(self, img, contrast, brightness, voxel_size=None):
        self.processedImage = img.astype(float)
        if voxel_size is not None:
            self.voxel_size = voxel_size

        # Apply brightness and contrast adjustments to the processed image
        self.processedImage += brightness * 10  # Adjust brightness
        self.processedImage *= contrast  # Adjust contrast

        # Clip values to ensure they remain within [0, 255]
        self.processedImage = np.clip(self.processedImage, 0, 255).astype(np.uint8)

        self.processedImage_dis = self.scale_display(self.processedImage).astype(np.uint8)
        self.img_z, self.img_y, self.img_x = self.processedImage.shape
        self.axial_index, self.sagittal_index, self.coronal_index = [int(self.img_x / 2), int(self.img_y / 2),
                                                                     int(self.img_z / 2)]

    def update_image_fusion(self, img, seg1, contrast, brightness, voxel_size=None):
        self.processedImage = img.astype(float)
        if voxel_size is None:
            pass
        else:
            self.voxel_size = voxel_size
        self.processedImage_dis = self.scale_display(self.processedImage).astype(np.uint8)
        seg = deepcopy(seg1)

        img_shape = np.array(self.processedImage.shape)
        seg_shape = np.array(seg.shape)

        target_shape = np.maximum(img_shape, seg_shape)

        self.processedImage.shape = target_shape

        img_padded = np.zeros(target_shape, dtype=self.processedImage.dtype)
        seg_padded = np.zeros(target_shape, dtype=seg.dtype)

        img_start = (target_shape - img_shape) // 2
        seg_start = (target_shape - seg_shape) // 2

        img_end = img_start + img_shape
        seg_end = seg_start + seg_shape

        img_padded[img_start[0]:img_end[0], img_start[1]:img_end[1], img_start[2]:img_end[2]] = self.processedImage
        seg_padded[seg_start[0]:seg_end[0], seg_start[1]:seg_end[1], seg_start[2]:seg_end[2]] = seg
        seg = seg_padded
        self.processedImage = img_padded
        seg[seg > 1] = seg[seg > 1] + brightness * 10
        seg = (seg - seg.min()) / (seg.max() - seg.min()) * 255
        seg[seg > 0] = seg[seg > 0] * contrast
        seg[seg > 255] = 255
        self.seg = seg.astype(np.uint8)
        self.img_z, self.img_y, self.img_x = self.processedImage.shape
        self.axial_index, self.sagittal_index, self.coronal_index = [int(self.img_x / 2), int(self.img_y / 2),
                                                                     int(self.img_z / 2)]
    def enterEvent(self, event):
        self.flag = 1

    def leaveEvent(self, event):
        self.flag = 0

    def wheelEvent(self, event):
        delta = event.angleDelta()
        orientation = int(delta.y() / 60)
        if self.flag == 1:
            if self.type == 'axial':
                self.axial_index += orientation
                if self.axial_index < 0:
                    self.axial_index = 0
                elif self.axial_index >= self.img_x:
                    self.axial_index = self.img_x - 1
            elif self.type == 'sagittal':
                self.sagittal_index += orientation
                if self.sagittal_index < 0:
                    self.sagittal_index = 0
                elif self.sagittal_index >= self.img_y:
                    self.sagittal_index = self.img_y - 1
            elif self.type == 'coronal':
                self.coronal_index += orientation
                if self.coronal_index < 0:
                    self.coronal_index = 0
                elif self.coronal_index >= self.img_z:
                    self.coronal_index = self.img_z - 1
        if self.fusion == False:
            self.display_image(1)
        else:
            self.display_image_fusion(1)

