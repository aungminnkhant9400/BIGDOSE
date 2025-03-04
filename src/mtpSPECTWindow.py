import os
import sys

import nibabel as nib

import numpy as np
import traceback2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)

from PyQt6 import QtCore, QtWidgets, QtGui
from image_class_lu177 import Image
from src.ui.BaseLayout import Ui_MainWindow
from fusion_display import fusion_display
import SimpleITK as sitk
from PyQt6.QtCore import pyqtSignal,QThread,QTimer
import itk
import time
from scipy.signal import fftconvolve
from totalsegmentator.python_api import totalsegmentator



class RegistrationWorker(QThread):
    # 定义信号，用于线程完成后通知主线程
    finished = pyqtSignal(str)  # 参数为输出文件路径和固定图像对象

    def __init__(self, moved_img, fix_img, moved_spect, time_point):
        super().__init__()
        self.moved_img = moved_img
        self.fix_img = fix_img
        self.moved_spect = moved_spect
        self.time_point = time_point

    def run(self):
        """
        在线程中执行图像配准任务。
        """
        try:
            print(self.fix_img)

            file_path_list = str(self.fix_img).split('/')

            file_path_list.pop(-1)

            result_file_path = '/'.join(file_path_list)

            print(result_file_path)
            fixed_image = itk.imread(self.fix_img, itk.F)
            moving_image = itk.imread(self.moved_img, itk.F)
            moving_image_spect = itk.imread(self.moved_spect, itk.F)
            parameter_object = itk.ParameterObject.New()
            resolutions = 3
            parameter_map_rigid = parameter_object.GetDefaultParameterMap('affine', 3)
            parameter_object.AddParameterMap(parameter_map_rigid)
            parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions, 20.0)
            parameter_map_bspline['MaximumNumberOfIterations'] = ['300']
            parameter_map_bspline['NumberOfSpatialSamples'] = ['200']
            parameter_map_bspline['Optimizer'] = ['AdaptiveStochasticGradientDescent']
            parameter_map_bspline['NumberOfSamplesForExactGradient'] = ['5000']
            parameter_map_bspline['GridSpacingSchedule'] = ['4', '2', '1']
            parameter_map_bspline['MaximumStepLength'] = ['1']

            parameter_object.AddParameterMap(parameter_map_bspline)

            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=True)

            moving_image_spect_transformix = itk.transformix_filter(
                moving_image_spect,
                result_transform_parameters)

            moving_image_spect_transformix_np = itk.array_view_from_image(moving_image_spect_transformix)

            # Clip negative values
            moving_image_spect_transformix_np[moving_image_spect_transformix_np < 0] = 0

            # Convert the NumPy array back to an ITK image
            moving_image_spect_transformix_clipped = itk.image_view_from_array(
                moving_image_spect_transformix_np
            )

            # Set the metadata (spacing, origin, etc.) of the clipped ITK image to match the original
            moving_image_spect_transformix_clipped.CopyInformation(moving_image_spect_transformix)

            output_ct_name = 'scan_' + str(self.time_point)  + '_registered_CT.nii.gz'

            pid = '0000915556'
            result_file_path = rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}"
            itk.imwrite(result_image, result_file_path + '/' + output_ct_name)

            output_spect_name = 'scan_' + str(self.time_point) + '_registered_ECT.nii.gz'

            itk.imwrite(moving_image_spect_transformix_clipped, result_file_path + '/' + output_spect_name)
            # time.sleep(200)
            # 通知主线程任务完成
            self.finished.emit(result_file_path)
        except Exception as e:
            print(f"Registration failed: {e}")



class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #self.ui.progressBar.setMinimum(0)
        #self.ui.progressBar.setMaximum(8)
        #self.ui.progressBar.setValue(0)
        #self.ui.progressBar.setHidden(True)

        self.ui.label_23.setVisible(False)

        self.data_form = None
        self.images = []

        pid = '0000915556'
        self.images_path = {
            "time1_SPECT": rf"D:\000_177Lu_PET_CT_DIR\preprocessed_data\p3\p_3_scan_1_pet.nii.gz",
            "time1_CT": rf"D:\000_177Lu_PET_CT_DIR\preprocessed_data\p3\p_3_scan_1_ct.nii.gz",
            "time2_SPECT": rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}\time2_PT_QSPECT.nii.gz",
            "time2_CT": rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}\time2_CT2ECT.nii.gz",
            "time3_SPECT": rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}\time3_PT_QSPECT.nii.gz",
            "time3_CT": rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}\time3_CT2ECT.nii.gz",
            "time4_SPECT": rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}\time4_PT_QSPECT.nii.gz",
            "time4_CT": rf"E:\OneDrive - University of Macau\RESEARCH\Project\Ga68_predict\Data\{pid}\time4_CT2ECT.nii.gz",
            "time5_SPECT": "",
            "time5_CT": "",
        }
        self.ect_contrast = 50
        self.ct_contrast = 50

        self.ect_brightness = 50
        self.ct_brightness = 50

        self.regi_view = 'sagittal'

        # Connect Open file fucntion to the same slot function
        # 将点击事件连接到自定义函数，使用 lambda 推迟调用，并传递参数
        self.ui.tp1_ect.clicked.connect(lambda: self.data_import(1, 'ect'))
        self.ui.tp1_ct.clicked.connect(lambda: self.data_import(1, 'ct'))
        self.ui.tp2_ect.clicked.connect(lambda: self.data_import(2, 'ect'))
        self.ui.tp2_ct.clicked.connect(lambda: self.data_import(2, 'ct'))
        self.ui.tp3_ect.clicked.connect(lambda: self.data_import(3, 'ect'))
        self.ui.tp3_ct.clicked.connect(lambda: self.data_import(3, 'ct'))
        self.ui.tp4_ect.clicked.connect(lambda: self.data_import(4, 'ect'))
        self.ui.tp4_ct.clicked.connect(lambda: self.data_import(4, 'ct'))
        self.ui.tp5_ect.clicked.connect(lambda: self.data_import(5, 'ect'))
        self.ui.tp5_ct.clicked.connect(lambda: self.data_import(5, 'ct'))

        # 显示图像的点击事件同样使用 lambda 表达式
        self.ui.tp1_disp.clicked.connect(
            lambda: self.multi_image_display(self.images_path["time1_SPECT"], self.images_path["time1_CT"]))
        self.ui.tp2_disp.clicked.connect(
            lambda: self.multi_image_display(self.images_path["time2_SPECT"], self.images_path["time2_CT"]))
        self.ui.tp3_disp.clicked.connect(
            lambda: self.multi_image_display(self.images_path["time3_SPECT"], self.images_path["time3_CT"]))
        self.ui.tp4_disp.clicked.connect(
            lambda: self.multi_image_display(self.images_path["time4_SPECT"], self.images_path["time4_CT"]))
        self.ui.tp5_disp.clicked.connect(
            lambda: self.multi_image_display(self.images_path["time5_SPECT"], self.images_path["time5_CT"]))

        self.ui.display_multi.clicked.connect(
            lambda: self.regi_multi_display())

        self.ui.regi_display_view.currentTextChanged.connect(
            lambda: self.regi_view_changed())

        self.ui.display_multi_2.clicked.connect(self.display_segmentation_images)
        self.ui.display_multi_3.clicked.connect(self.display_segmentation_images_ECT)

        self.ui.pushButton_2.clicked.connect(self.segment_organs_toalsegmentor)

        self.ui.pushButton.clicked.connect(lambda: self.auto_regi())

        self.ui.checkBox_18.clicked.connect(self.display_segmentation_images_after_totalsegmentor_brain)
        self.ui.checkBox_11.clicked.connect(self.display_segmentation_images_after_totalsegmentor_liver)
        self.ui.checkBox.clicked.connect(self.display_segmentation_images_after_totalsegmentor_adrenal_gland_left)
        self.ui.checkBox_2.clicked.connect(self.display_segmentation_images_after_totalsegmentor_aorta)
        self.ui.checkBox_3.clicked.connect(self.display_segmentation_images_after_totalsegmentor_colon)
        self.ui.checkBox_4.clicked.connect(self.display_segmentation_images_after_totalsegmentor_duodenum)
        self.ui.checkBox_6.clicked.connect(self.display_segmentation_images_after_totalsegmentor_esophagus)
        self.ui.checkBox_7.clicked.connect(self.display_segmentation_images_after_totalsegmentor_gallbladder)
        self.ui.checkBox_5.clicked.connect(self.display_segmentation_images_after_totalsegmentor_heart)
        self.ui.checkBox_14.clicked.connect(self.display_segmentation_images_after_totalsegmentor_adrenal_gland_right)
        self.ui.checkBox_12.clicked.connect(self.display_segmentation_images_after_totalsegmentor_kidney_left)
        self.ui.checkBox_13.clicked.connect(self.display_segmentation_images_after_totalsegmentor_kidney_right)
        self.ui.checkBox_10.clicked.connect(self.display_segmentation_images_after_totalsegmentor_spleen)
        self.ui.checkBox_9.clicked.connect(self.display_segmentation_images_after_totalsegmentor_lung_lobe_left)
        self.ui.checkBox_8.clicked.connect(self.display_segmentation_images_after_totalsegmentor_lung_lobe_right)
        self.ui.checkBox_21.clicked.connect(self.display_segmentation_images_after_totalsegmentor_stomach)
        self.ui.checkBox_19.clicked.connect(self.display_segmentation_images_after_totalsegmentor_trachea)
        self.ui.checkBox_20.clicked.connect(self.display_segmentation_images_after_totalsegmentor_urinary_bladder)
        self.ui.checkBox_17.clicked.connect(self.display_segmentation_images_after_totalsegmentor_prostate)
        self.ui.checkBox_16.clicked.connect(self.display_segmentation_images_after_totalsegmentor_pancreas)
        self.ui.checkBox_15.clicked.connect(self.display_segmentation_images_after_totalsegmentor_small_bowel)
        self.ui.checkBox_27.clicked.connect(self.display_segmentation_images_after_totalsegmentor_parotid_gland_left)
        self.ui.checkBox_28.clicked.connect(self.display_segmentation_images_after_totalsegmentor_parotid_gland_right)
        self.ui.checkBox_29.clicked.connect(self.display_segmentation_images_after_totalsegmentor_bone_marrow)

        self.multi_regis = False
        self.worker1 = None
        self.worker2 = None
        self.worker3 = None
        self.worker4 = None

        # Connect checkboxes to the method that ensures only one is checked
        self.ui.checkBox_18.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_18))
        self.ui.checkBox_11.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_11))
        self.ui.checkBox.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox))
        self.ui.checkBox_2.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_2))
        self.ui.checkBox_3.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_3))
        self.ui.checkBox_4.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_4))
        self.ui.checkBox_6.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_6))
        self.ui.checkBox_7.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_7))
        self.ui.checkBox_5.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_5))
        self.ui.checkBox_14.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_14))
        self.ui.checkBox_12.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_12))
        self.ui.checkBox_13.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_13))
        self.ui.checkBox_10.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_10))
        self.ui.checkBox_9.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_9))
        self.ui.checkBox_8.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_8))
        self.ui.checkBox_21.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_21))
        self.ui.checkBox_19.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_19))
        self.ui.checkBox_20.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_20))
        self.ui.checkBox_17.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_17))
        self.ui.checkBox_16.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_16))
        self.ui.checkBox_15.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_15))
        self.ui.checkBox_27.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_27))
        self.ui.checkBox_28.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_28))
        self.ui.checkBox_29.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_29))
        self.ui.checkBox_22.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_22))
        self.ui.checkBox_23.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_23))
        self.ui.checkBox_24.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_24))
        self.ui.checkBox_25.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_25))
        self.ui.checkBox_26.clicked.connect(lambda: self.handle_checkbox_click(self.ui.checkBox_26))

    def handle_checkbox_click(self, clicked_checkbox):
        checkboxes = [
            self.ui.checkBox_18, self.ui.checkBox_11, self.ui.checkBox, self.ui.checkBox_2,
            self.ui.checkBox_3, self.ui.checkBox_4, self.ui.checkBox_6, self.ui.checkBox_7,
            self.ui.checkBox_5, self.ui.checkBox_14, self.ui.checkBox_12, self.ui.checkBox_13,
            self.ui.checkBox_10, self.ui.checkBox_9, self.ui.checkBox_8, self.ui.checkBox_21,
            self.ui.checkBox_19, self.ui.checkBox_20, self.ui.checkBox_17, self.ui.checkBox_16,
            self.ui.checkBox_15, self.ui.checkBox_27, self.ui.checkBox_28, self.ui.checkBox_29,
            self.ui.checkBox_22, self.ui.checkBox_23, self.ui.checkBox_24, self.ui.checkBox_25,
            self.ui.checkBox_26
        ]
        for checkbox in checkboxes:
            if checkbox != clicked_checkbox:
                checkbox.setChecked(False)

    def display_segmentation_images(self):
        print(self.images_path["time1_SPECT"])

        #self.ui.progressBar.setMinimum(0)
        #self.ui.progressBar.setMaximum(8)
        #self.ui.progressBar.setValue(0)
        #self.ui.progressBar.setHidden(True)

        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        time1_ect = self.images_path["time1_SPECT"]
        time1_ct = self.images_path["time1_CT"]

        print("**454**")
        print(os.getcwd())
        # ITK 会改变cwd位置
        if time1_ect and time1_ct:
            ect_image = Image(time1_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time1_ct, "ct")
            ct_image.read_image()

            self.ui.time1_before_regi_5.type = 'coronal'
            self.ui.time1_before_regi_5.fusion = False
            self.ui.time1_before_regi_5.update_image(ct_image.get_image_data(), self.ct_contrast, self.ct_brightness, 1,
                                              ct_image.voxel_size)
            self.ui.time1_before_regi_5.display_image(1)

            self.ui.time1_before_regi_6.type = 'sagittal'
            self.ui.time1_before_regi_6.fusion = False
            self.ui.time1_before_regi_6.update_image(ct_image.get_image_data(), self.ct_contrast, self.ct_brightness, 1,
                                              ct_image.voxel_size)
            self.ui.time1_before_regi_6.display_image(1)

            self.ui.time1_before_regi_7.type = 'axial'
            self.ui.time1_before_regi_7.fusion = False
            self.ui.time1_before_regi_7.update_image(ct_image.get_image_data(), self.ct_contrast, self.ct_brightness, 1,
                                              ct_image.voxel_size)
            self.ui.time1_before_regi_7.display_image(1)

            ##############

    def display_segmentation_images_ECT(self):
        print(self.images_path["time1_SPECT"])

        # self.ui.progressBar.setMinimum(0)
        # self.ui.progressBar.setMaximum(8)
        # self.ui.progressBar.setValue(0)
        # self.ui.progressBar.setHidden(True)

        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        time1_ect = self.images_path["time1_SPECT"]
        time1_ct = self.images_path["time1_CT"]

        print("**454**")
        print(os.getcwd())
        # ITK 会改变cwd位置
        if time1_ect and time1_ct:
            ect_image = Image(time1_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time1_ct, "ct")
            ct_image.read_image()

            self.ect_contrast = 10

            self.ui.time1_before_regi_5.color = True
            self.ui.time1_before_regi_6.color = True
            self.ui.time1_before_regi_7.color = True

            self.ui.time1_before_regi_5.type = 'coronal'
            self.ui.time1_before_regi_5.fusion = False
            self.ui.time1_before_regi_5.update_image(ect_image.get_image_data(), self.ect_contrast, self.ect_brightness,
                                                     0,
                                                     ect_image.voxel_size)
            self.ui.time1_before_regi_5.display_image(1)

            self.ui.time1_before_regi_6.type = 'sagittal'
            self.ui.time1_before_regi_6.fusion = False
            self.ui.time1_before_regi_6.update_image(ect_image.get_image_data(), self.ect_contrast, self.ect_brightness,
                                                     0,
                                                     ect_image.voxel_size)
            self.ui.time1_before_regi_6.display_image(1)

            self.ui.time1_before_regi_7.type = 'axial'
            self.ui.time1_before_regi_7.fusion = False
            self.ui.time1_before_regi_7.update_image(ect_image.get_image_data(), self.ect_contrast, self.ect_brightness,
                                                     0,
                                                     ect_image.voxel_size)
            self.ui.time1_before_regi_7.display_image(1)

            ##############

    def display_segmentation_images_after_totalsegmentor(self, organ_name):
        print(self.images_path["time1_SPECT"])

        print(f'display_segmentation_images_after_totalsegmentor_{organ_name}')

        time1_ect = self.images_path["time1_SPECT"]
        time1_ct = self.images_path["time1_CT"]

        full_list = time1_ct.split('/')
        length = len(full_list)
        num = 0
        folder = ""

        while num < length - 1:
            if num == 0:
                folder = full_list[num]
            else:
                folder += '/'
                folder += full_list[num]
            num += 1



        mask_path = folder + f'/organs_in_ui/{organ_name}.nii.gz'
        print(mask_path)



        self.ect_contrast = 0
        print(os.getcwd())

        if time1_ect and time1_ct:
            ect_image = Image(time1_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time1_ct, "ct")
            ct_image.read_image()
            seg_image = Image(mask_path, "ect")
            seg_image.read_image()

            self.ui.time1_before_regi_5.type = 'coronal'
            self.ui.time1_before_regi_5.fusion = True
            self.ui.time1_before_regi_5.update_image_fusion(ct_image.get_image_data(), seg_image.get_image_data(),
                                                            self.ect_brightness, self.ct_brightness, self.ect_contrast,
                                                            self.ct_contrast, ct_image.voxel_size)
            self.ui.time1_before_regi_5.display_image_fusion(1)

            self.ui.time1_before_regi_6.type = 'sagittal'
            self.ui.time1_before_regi_6.fusion = True
            self.ui.time1_before_regi_6.update_image_fusion(ct_image.get_image_data(), seg_image.get_image_data(),
                                                            self.ect_brightness, self.ct_brightness, self.ect_contrast,
                                                            self.ct_contrast, ct_image.voxel_size)
            self.ui.time1_before_regi_6.display_image_fusion(1)

            self.ui.time1_before_regi_7.type = 'axial'
            self.ui.time1_before_regi_7.fusion = True
            self.ui.time1_before_regi_7.update_image_fusion(ct_image.get_image_data(), seg_image.get_image_data(),
                                                            self.ect_brightness, self.ct_brightness, self.ect_contrast,
                                                            self.ct_contrast, ct_image.voxel_size)
            self.ui.time1_before_regi_7.display_image_fusion(1)

    def display_segmentation_images_after_totalsegmentor_brain(self):
        self.display_segmentation_images_after_totalsegmentor('brain')

    def display_segmentation_images_after_totalsegmentor_liver(self):
        self.display_segmentation_images_after_totalsegmentor('liver')

    def display_segmentation_images_after_totalsegmentor_adrenal_gland_left(self):
        self.display_segmentation_images_after_totalsegmentor('adrenal_gland_left')

    def display_segmentation_images_after_totalsegmentor_aorta(self):
        self.display_segmentation_images_after_totalsegmentor('aorta')

    def display_segmentation_images_after_totalsegmentor_colon(self):
        self.display_segmentation_images_after_totalsegmentor('colon')

    def display_segmentation_images_after_totalsegmentor_duodenum(self):
        self.display_segmentation_images_after_totalsegmentor('duodenum')

    def display_segmentation_images_after_totalsegmentor_esophagus(self):
        self.display_segmentation_images_after_totalsegmentor('esophagus')

    def display_segmentation_images_after_totalsegmentor_gallbladder(self):
        self.display_segmentation_images_after_totalsegmentor('gallbladder')

    def display_segmentation_images_after_totalsegmentor_heart(self):
        self.display_segmentation_images_after_totalsegmentor('heart')

    def display_segmentation_images_after_totalsegmentor_adrenal_gland_right(self):
        self.display_segmentation_images_after_totalsegmentor('adrenal_gland_right')

    def display_segmentation_images_after_totalsegmentor_kidney_left(self):
        self.display_segmentation_images_after_totalsegmentor('kidney_left')

    def display_segmentation_images_after_totalsegmentor_kidney_right(self):
        self.display_segmentation_images_after_totalsegmentor('kidney_right')

    def display_segmentation_images_after_totalsegmentor_spleen(self):
        self.display_segmentation_images_after_totalsegmentor('spleen')

    def display_segmentation_images_after_totalsegmentor_lung_lobe_left(self):
        self.display_segmentation_images_after_totalsegmentor('lung_lobe_left')

    #def display_segmentation_images_after_totalsegmentor_lung_lower_lobe_left(self):
        #self.display_segmentation_images_after_totalsegmentor('lung_lobe_left')

    def display_segmentation_images_after_totalsegmentor_lung_lobe_right(self):
        self.display_segmentation_images_after_totalsegmentor('lung_lobe_right')


    def display_segmentation_images_after_totalsegmentor_stomach(self):
        self.display_segmentation_images_after_totalsegmentor('stomach')

    def display_segmentation_images_after_totalsegmentor_trachea(self):
        self.display_segmentation_images_after_totalsegmentor('trachea')

    def display_segmentation_images_after_totalsegmentor_urinary_bladder(self):
        self.display_segmentation_images_after_totalsegmentor('urinary_bladder')

    def display_segmentation_images_after_totalsegmentor_prostate(self):
        self.display_segmentation_images_after_totalsegmentor('prostate')

    def display_segmentation_images_after_totalsegmentor_pancreas(self):
        self.display_segmentation_images_after_totalsegmentor('pancreas')

    def display_segmentation_images_after_totalsegmentor_small_bowel(self):
        self.display_segmentation_images_after_totalsegmentor('small_bowel')

    def display_segmentation_images_after_totalsegmentor_bone_marrow(self):
        self.display_segmentation_images_after_totalsegmentor('L1_to_L5_bone_marrow')

    def display_segmentation_images_after_totalsegmentor_parotid_gland_left(self):
        self.display_segmentation_images_after_totalsegmentor('parotid_gland_left')

    def display_segmentation_images_after_totalsegmentor_parotid_gland_right(self):
        self.display_segmentation_images_after_totalsegmentor('/parotid_gland_right')



    def segment_organs_toalsegmentor(self):

        self.ui.label_23.setVisible(True)
        self.ui.label_23.setText("Segmenting...")
        # Force UI update immediately
        QApplication.processEvents()

        print('using totalsegmentor')

        time1_ct = self.images_path["time1_CT"]

        full_list = time1_ct.split('/')

        length = len(full_list)

        num = 0

        while num < length-1:

            if num == 0:

                folder = full_list[num]

            else:

                folder += '/'

                folder += full_list[num]

            num += 1

        self.ui.label_23.setVisible(True)

        if not os.path.exists(folder + '/' + 'head_glands_cavities'):

            os.mkdir(folder + '/' + 'head_glands_cavities')

        if not os.path.exists(folder + '/' + 'organs_in_ui'):

            os.mkdir(folder + '/' + 'organs_in_ui')

        print('folder')

        print(time1_ct)

        totalsegmentator(time1_ct, folder + '/' + 'organs_in_ui',
                         roi_subset=['liver', 'lung_lower_lobe_left', 'kidney_right', 'kidney_left',
                                     'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_lower_lobe_right',
                                     'lung_upper_lobe_right', 'adrenal_gland_left', 'adrenal_gland_right', 'aorta',
                                     'colon', 'duodenum', 'esophagus', 'gallbladder', 'heart', 'pancreas',
                                     'small_bowel', 'spleen',
                                     'stomach', 'trachea', 'urinary_bladder', 'prostate', 'brain'], fast=True, fastest=True)  ## 1.8 minuts

        print('***************')

        #folder = r'C:\Users\aungm\PycharmProjects\BIGDOSE_Lu177\data'

        lung_lower_lobe_left = folder + '\\' + 'organs_in_ui\\' + 'lung_lower_lobe_left.nii.gz'

        lung_upper_lobe_left = folder + '\\' + 'organs_in_ui\\' + 'lung_upper_lobe_left.nii.gz'

        lung_lower_lobe_left_head = nib.load(lung_lower_lobe_left)

        lung_lower_lobe_left_array = lung_lower_lobe_left_head.get_fdata()

        print(000)

        lung_upper_lobe_left_head = nib.load(lung_upper_lobe_left)

        lung_upper_lobe_left_array = lung_upper_lobe_left_head.get_fdata()

        print(111)

        lung_left_whole_array = lung_upper_lobe_left_array + lung_lower_lobe_left_array

        nimg = nib.Nifti1Image(lung_left_whole_array, lung_lower_lobe_left_head.affine,
                               lung_lower_lobe_left_head.header)
        nimg.to_filename(folder + '\\' + 'organs_in_ui\\lung_lobe_left.nii.gz')

        print(222)

        lung_lower_lobe_right = folder + '\\' + 'organs_in_ui\\' + 'lung_lower_lobe_right.nii.gz'

        lung_middle_lobe_right = folder + '\\' + 'organs_in_ui\\' + 'lung_middle_lobe_right.nii.gz'

        lung_upper_lobe_right = folder + '\\' + 'organs_in_ui\\' + 'lung_upper_lobe_right.nii.gz'

        lung_lower_lobe_right_head = nib.load(lung_lower_lobe_right)
        lung_middle_lobe_right_head = nib.load(lung_middle_lobe_right)

        lung_lower_lobe_right_array = lung_lower_lobe_right_head.get_fdata() + lung_middle_lobe_right_head.get_fdata()

        print(000)

        lung_upper_lobe_right_head = nib.load(lung_upper_lobe_right)

        lung_upper_lobe_right_array = lung_upper_lobe_right_head.get_fdata()

        print(111)

        lung_middle_lobe_right_head = nib.load(lung_middle_lobe_right)

        lung_middle_lobe_right_array = lung_middle_lobe_right_head.get_fdata()

        lung_right_whole_array = lung_upper_lobe_right_array + lung_lower_lobe_right_array + lung_middle_lobe_right_array

        nimg = nib.Nifti1Image(lung_right_whole_array, lung_lower_lobe_right_head.affine,
                               lung_lower_lobe_right_head.header)
        nimg.to_filename(folder + '\\' + 'organs_in_ui\\lung_lobe_right.nii.gz')

        print(222)



        print('%%%%%%%%%%%%')
        self.ui.label_23.setVisible(True)
        self.ui.label_23.setText("Segmenting...")

        totalsegmentator(time1_ct, folder + '/' + 'head_glands_cavities', task='head_glands_cavities', fast=True, fastest=True, roi_subset=['parotid_gland_left', 'parotid_gland_right'])  # fastest

        totalsegmentator(time1_ct, folder + '/' + 'organs_in_ui', task='total',
                         roi_subset=['vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5'])

        vertebrae_L1 = folder + '/' + 'organs_in_ui/' + 'vertebrae_L1.nii.gz'

        vertebrae_L2 = folder + '/' + 'organs_in_ui/' + 'vertebrae_L2.nii.gz'

        vertebrae_L3 = folder + '/' + 'organs_in_ui/' + 'vertebrae_L3.nii.gz'

        vertebrae_L4 = folder + '/' + 'organs_in_ui/' + 'vertebrae_L4.nii.gz'

        vertebrae_L5 = folder + '/' + 'organs_in_ui/' + 'vertebrae_L5.nii.gz'

        vertebrae_L1_head = nib.load(vertebrae_L1)

        vertebrae_L1_array = vertebrae_L1_head.get_fdata()

        print(000)

        vertebrae_L2_head = nib.load(vertebrae_L2)

        vertebrae_L2_array = vertebrae_L2_head.get_fdata()

        print(111)

        vertebrae_L3_head = nib.load(vertebrae_L3)

        vertebrae_L3_array = vertebrae_L3_head.get_fdata()

        print(111)

        vertebrae_L4_head = nib.load(vertebrae_L4)

        vertebrae_L4_array = vertebrae_L4_head.get_fdata()

        print(111)

        vertebrae_L5_head = nib.load(vertebrae_L5)

        vertebrae_L5_array = vertebrae_L5_head.get_fdata()

        vertebrae_L1_L5_array = vertebrae_L5_array + vertebrae_L4_array + vertebrae_L3_array + vertebrae_L2_array + vertebrae_L1_array

        nimg = nib.Nifti1Image(vertebrae_L1_L5_array, vertebrae_L5_head.affine, vertebrae_L5_head.header)
        nimg.to_filename(folder + '/' + 'organs_in_ui/spine_L1_to_L5.nii.gz')

        print(222)

        ct_head = nib.load(time1_ct)

        ct_arr = ct_head.get_fdata()

        spine_L1_to_L5_head = nib.load(folder + '/' + 'organs_in_ui/spine_L1_to_L5.nii.gz')

        spine_L1_to_L5_arr = spine_L1_to_L5_head.get_fdata()

        intensity_bone = ct_arr * spine_L1_to_L5_arr

        # index = np.where((intensity_bone > 0) & (intensity_bone <= 200))

        index = np.where((intensity_bone >= 200))

        temp = np.zeros((ct_arr.shape))

        i = 0

        while i < len(index[0]):
            x = index[0][i]
            y = index[1][i]
            z = index[2][i]

            temp[x, y, z] = 1

            i += 1

        bone_marrow = spine_L1_to_L5_arr - temp

        nimg = nib.Nifti1Image(bone_marrow, vertebrae_L5_head.affine, vertebrae_L5_head.header)

        nimg.to_filename(folder + '/' + 'organs_in_ui/L1_to_L5_bone_marrow.nii.gz')
        #QMessageBox.information(self, "Segmentation Complete", "Auto segmentation complete.")

        #QMessageBox.critical(self, "Error",f"An error occurred during segmentation: {str(e)}\n{traceback2.format_exc()}")

        self.ui.label_23.setVisible(False)  # Hide the label when segmentation is complete

    def auto_regi(self):

        print(self.images_path["time1_SPECT"])

        print('***************************************************************')

        time1_ect = self.images_path["time1_SPECT"]
        time1_ct = self.images_path["time1_CT"]

        time2_ect = self.images_path["time2_SPECT"]
        time2_ct = self.images_path["time2_CT"]

        time3_ect = self.images_path["time3_SPECT"]
        time3_ct = self.images_path["time3_CT"]

        time4_ect = self.images_path["time4_SPECT"]
        time4_ct = self.images_path["time4_CT"]

        time5_ect = self.images_path["time5_SPECT"]
        time5_ct = self.images_path["time5_CT"]

        if time2_ect and time2_ct:
            self.worker1 = RegistrationWorker(time1_ct, time1_ct, time1_ect, time_point=1)
            self.worker1.finished.connect(self.regi_after_multi_display)
            self.worker1.start()

        if time2_ect and time2_ct:
            self.worker1 = RegistrationWorker(time2_ct, time1_ct, time2_ect, time_point=2)
            self.worker1.finished.connect(self.regi_after_multi_display)
            self.worker1.start()


        if time3_ect and time3_ct:
            self.worker2 = RegistrationWorker(time3_ct, time1_ct, time3_ect, time_point=3)
            self.worker2.finished.connect(self.regi_after_multi_display)
            self.worker2.start()

        if time4_ect and time4_ct:
            self.worker3 = RegistrationWorker(time4_ct, time1_ct, time4_ect, time_point=4)
            self.worker3.finished.connect(self.regi_after_multi_display)
            self.worker3.start()

        if time5_ect and time5_ct:
            self.worker4 = RegistrationWorker(time5_ct, time1_ct, time5_ect, time_point=5)
            self.worker4.finished.connect(self.regi_after_multi_display)
            self.worker4.start()


    def data_import(self, time_point, modality):
        try:
            filename, _ = QFileDialog.getOpenFileName(
                filter="Image files (*.nii *.nii.gz *.mhd *.nrrd *.dcm)")

            # 检查是否选择了文件
            if filename:
                print(f"Selected file: {filename}")
                if modality == 'ect':
                    self.images_path[f'time{time_point}_SPECT'] = filename
                    self.display_info(filename, time_point, modality)
                else:
                    self.images_path[f'time{time_point}_CT'] = filename
                    self.display_info(filename, time_point, modality)

        except Exception as e:
            QMessageBox.critical(None, "Error", f"An error occurred: {str(e)}")

    def regi_view_changed(self):
        view = self.ui.regi_display_view.currentText()
        if view == 'Coronal':
            self.regi_view = 'sagittal'
        elif view == 'Axial':
            self.regi_view = 'coronal'
        elif view == 'Sagittal':
            self.regi_view = 'axial'
        self.regi_multi_display()
        if self.multi_regis:
            self.regi_after_multi_display()


    def regi_after_multi_display(self):
        self.multi_regis = True
        print(self.images_path["time1_SPECT"])

        print('###############################################################')

        time1_ect = self.images_path["time1_SPECT"]
        time1_ct = self.images_path["time1_CT"]

        time2_ect = self.images_path["time2_SPECT"]
        time2_ct = self.images_path["time2_CT"]

        time3_ect = self.images_path["time3_SPECT"]
        time3_ct = self.images_path["time3_CT"]

        time4_ect = self.images_path["time4_SPECT"]
        time4_ct = self.images_path["time4_CT"]

        time5_ect = self.images_path["time5_SPECT"]
        time5_ct = self.images_path["time5_CT"]

        if time1_ect and time1_ct:
            ect_image = Image(time1_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time1_ct, "ct")
            ct_image.read_image()

            self.ui.time1_after_regi.type = self.regi_view
            self.ui.time1_after_regi.fusion = True
            self.ui.time1_after_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(), self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                             ct_image.voxel_size)
            self.ui.time1_after_regi.display_image_fusion(1)
            self.multi_regis = True

        if time2_ect and time2_ct:

            print(time2_ect)

            file_path_list = str(time1_ect).split('/')

            file_path_list.pop(-1)

            result_file_path = '/'.join(file_path_list)

            print(result_file_path)

            time2_ct = result_file_path + '/' + 'scan_' + str(2) + '_registered_CT.nii.gz'

            time2_ect = result_file_path + '/' + 'scan_' + str(2) + '_registered_ECT.nii.gz'

            ect_image = Image(time2_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time2_ct, "ct")
            ct_image.read_image()

            self.ui.time2_after_regi.type = self.regi_view
            self.ui.time2_after_regi.fusion = True
            self.ui.time2_after_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(),
                                                          self.ect_brightness, self.ct_brightness,
                                                          self.ect_contrast, self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time2_after_regi.display_image_fusion(1)

        if time3_ect and time3_ct:

            print(time3_ect)

            file_path_list = str(time1_ect).split('/')

            file_path_list.pop(-1)

            result_file_path = '/'.join(file_path_list)

            print(result_file_path)

            time3_ct = result_file_path + '/' + 'scan_' + str(3) + '_registered_CT.nii.gz'

            time3_ect = result_file_path + '/' + 'scan_' + str(3) + '_registered_ECT.nii.gz'

            ect_image = Image(time3_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time3_ct, "ct")
            ct_image.read_image()

            self.ui.time3_after_regi.type = self.regi_view
            self.ui.time3_after_regi.fusion = True
            self.ui.time3_after_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(),
                                                          self.ect_brightness, self.ct_brightness,
                                                          self.ect_contrast, self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time3_after_regi.display_image_fusion(1)

        if time4_ect and time4_ct:

            print(time4_ect)

            file_path_list = str(time1_ect).split('/')

            file_path_list.pop(-1)

            result_file_path = '/'.join(file_path_list)

            print(result_file_path)

            time4_ct = result_file_path + '/' + 'scan_' + str(4) + '_registered_CT.nii.gz'

            time4_ect = result_file_path + '/' + 'scan_' + str(4) + '_registered_ECT.nii.gz'

            ect_image = Image(time4_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time4_ct, "ct")
            ct_image.read_image()

            self.ui.time4_after_regi.type = self.regi_view
            self.ui.time4_after_regi.fusion = True
            self.ui.time4_after_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(),
                                                          self.ect_brightness, self.ct_brightness,
                                                          self.ect_contrast, self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time4_after_regi.display_image_fusion(1)

        if time5_ect and time5_ct:

            print(time5_ect)

            file_path_list = str(time1_ect).split('/')

            file_path_list.pop(-1)

            result_file_path = '/'.join(file_path_list)

            print(result_file_path)

            time5_ct = result_file_path + '/' + 'scan_' + str(5) + '_registered_CT.nii.gz'

            time5_ect = result_file_path + '/' + 'scan_' + str(5) + '_registered_ECT.nii.gz'

            ect_image = Image(time5_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time5_ct, "ct")
            ct_image.read_image()

            self.ui.time5_after_regi.type = self.regi_view
            self.ui.time5_after_regi.fusion = True
            self.ui.time5_after_regi.update_image_fusion(ct_image.get_image_data(),
                                                          ect_image.get_image_data(), self.ect_brightness,
                                                          self.ct_brightness, self.ect_contrast,
                                                          self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time5_after_regi.display_image_fusion(1)



    def regi_multi_display(self):

        print(self.images_path["time1_SPECT"])

        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        time1_ect = self.images_path["time1_SPECT"]
        time1_ct = self.images_path["time1_CT"]

        time2_ect = self.images_path["time2_SPECT"]
        time2_ct = self.images_path["time2_CT"]

        time3_ect = self.images_path["time3_SPECT"]
        time3_ct = self.images_path["time3_CT"]

        time4_ect = self.images_path["time4_SPECT"]
        time4_ct = self.images_path["time4_CT"]

        time5_ect = self.images_path["time5_SPECT"]
        time5_ct = self.images_path["time5_CT"]

        print("**454**")
        print(os.getcwd())
        # ITK 会改变cwd位置


        if time1_ect and time1_ct:
            ect_image = Image(time1_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time1_ct, "ct")
            ct_image.read_image()

            self.ui.time1_before_regi.type = self.regi_view
            self.ui.time1_before_regi.fusion = True
            self.ui.time1_before_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(), self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                             ct_image.voxel_size)
            self.ui.time1_before_regi.display_image_fusion(1)

        if time2_ect and time2_ct:
            ect_image = Image(time2_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time2_ct, "ct")
            ct_image.read_image()

            self.ui.time2_before_regi.type = self.regi_view
            self.ui.time2_before_regi.fusion = True
            self.ui.time2_before_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(),
                                                          self.ect_brightness, self.ct_brightness,
                                                          self.ect_contrast, self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time2_before_regi.display_image_fusion(1)

        if time3_ect and time3_ct:
            ect_image = Image(time3_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time3_ct, "ct")
            ct_image.read_image()

            self.ui.time3_before_regi.type = self.regi_view
            self.ui.time3_before_regi.fusion = True
            self.ui.time3_before_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(),
                                                          self.ect_brightness, self.ct_brightness,
                                                          self.ect_contrast, self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time3_before_regi.display_image_fusion(1)

        if time4_ect and time4_ct:
            ect_image = Image(time4_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time4_ct, "ct")
            ct_image.read_image()

            self.ui.time4_before_regi.type = self.regi_view
            self.ui.time4_before_regi.fusion = True
            self.ui.time4_before_regi.update_image_fusion(ct_image.get_image_data(), ect_image.get_image_data(),
                                                          self.ect_brightness, self.ct_brightness,
                                                          self.ect_contrast, self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time4_before_regi.display_image_fusion(1)

        if time5_ect and time5_ct:
            ect_image = Image(time5_ect, "ect")
            ect_image.read_image()
            ct_image = Image(time5_ct, "ct")
            ct_image.read_image()

            self.ui.time5_before_regi.type = self.regi_view
            self.ui.time5_before_regi.fusion = True
            self.ui.time5_before_regi.update_image_fusion(ct_image.get_image_data(),
                                                          ect_image.get_image_data(), self.ect_brightness,
                                                          self.ct_brightness, self.ect_contrast,
                                                          self.ct_contrast,
                                                          ct_image.voxel_size)
            self.ui.time5_before_regi.display_image_fusion(1)

    def multi_image_display(self, ect_path, ct_path):


        fusion_window = fusion_display()
        fusion_window.setupdata(ect_path, ct_path)
        fusion_window.exec()

    def dose_conversion_curve_fitting(self):
        kernel = r'E:\OneDrive - University of Macau\RESEARCH\Project\BIGDOSE\BIGDOSE_Lu177\src\regis_para\Lu1774.8mmsoft.nii.gz'
        scan_1_dose_rate = None
        scan_2_dose_rate = None
        scan_3_dose_rate = None
        scan_4_dose_rate = None
        scan_5_dose_rate = None

        if not self.images_path["time1_SPECT"] == None:
            scan_1_ECT_data = sitk.GetArrayFromImage(sitk.ReadImage(self.images_path["time1_SPECT"]))
            scan_1_dose_rate = fftconvolve(scan_1_ECT_data, kernel, mode='same')
        if not self.images_path["time2_SPECT"] == None:
            scan_2_ECT_data = sitk.GetArrayFromImage(sitk.ReadImage(self.images_path["time2_SPECT"]))
            scan_2_dose_rate = fftconvolve(scan_2_ECT_data, kernel, mode='same')
        if not self.images_path["time3_SPECT"] == None:
            scan_3_ECT_data = sitk.GetArrayFromImage(sitk.ReadImage(self.images_path["time3_SPECT"]))
            scan_3_dose_rate = fftconvolve(scan_3_ECT_data, kernel, mode='same')
        if not self.images_path["time4_SPECT"] == None:
            scan_4_ECT_data = sitk.GetArrayFromImage(sitk.ReadImage(self.images_path["time4_SPECT"]))
            scan_4_dose_rate = fftconvolve(scan_4_ECT_data, kernel, mode='same')

        curve_fitting_data = np.zeros_like((scan_1_dose_rate,scan_2_dose_rate,scan_3_dose_rate,scan_4_dose_rate))
        pass
    def display_info(self, path, timepoint, modality):
        image = Image(path,modality)
        image.read_image()

        if timepoint == 1:
            if modality == "ect":
                self.ui.voxel_size_1.setText(str(image.voxel_size))
                self.ui.matrix_size_1.setText(str(image.matrix_size))
            elif modality == "ct":
                self.ui.voxel_size_2.setText(str(image.voxel_size))
                self.ui.matrix_size_2.setText(str(image.matrix_size))

        elif timepoint == 2:
            if modality == "ect":
                self.ui.voxel_size_3.setText(str(image.voxel_size))
                self.ui.matrix_size_3.setText(str(image.matrix_size))
            elif modality == "ct":
                self.ui.voxel_size_4.setText(str(image.voxel_size))
                self.ui.matrix_size_4.setText(str(image.matrix_size))

        elif timepoint == 3:
            if modality == "ect":
                self.ui.voxel_size_5.setText(str(image.voxel_size))
                self.ui.matrix_size_5.setText(str(image.matrix_size))
            elif modality == "ct":
                self.ui.voxel_size_6.setText(str(image.voxel_size))
                self.ui.matrix_size_6.setText(str(image.matrix_size))

        elif timepoint == 4:
            if modality == "ect":
                self.ui.voxel_size_7.setText(str(image.voxel_size))
                self.ui.matrix_size_7.setText(str(image.matrix_size))
            elif modality == "ct":
                self.ui.voxel_size_8.setText(str(image.voxel_size))
                self.ui.matrix_size_8.setText(str(image.matrix_size))

        elif timepoint == 5:
            if modality == "ect":
                self.ui.voxel_size_9.setText(str(image.voxel_size))
                self.ui.matrix_size_9.setText(str(image.matrix_size))
            elif modality == "ct":
                self.ui.voxel_size_10.setText(str(image.voxel_size))
                self.ui.matrix_size_10.setText(str(image.matrix_size))
    def resizeEvent(self, event):
        super().resizeEvent(event)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())
