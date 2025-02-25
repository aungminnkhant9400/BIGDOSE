import sys
from PyQt6.QtWidgets import QApplication, QDialog
from image_class_lu177 import Image
from src.ui.DataInput_2 import Ui_Form
import SimpleITK as sitk
from PyQt6.QtCore import QThread, pyqtSignal
class RegistrationWorker(QThread):
    # 定义信号，用于线程完成后通知主线程
    finished = pyqtSignal(str)  # 参数为输出文件路径

    def __init__(self, moved_img, fix_img):
        super().__init__()
        self.moved_img = moved_img
        self.fix_img = fix_img

    def run(self):
        """
        在线程中执行图像配准任务。
        """
        try:
            elastixImageFilter = sitk.ElastixImageFilter()
            # 设置配准参数
            elastixImageFilter.SetFixedImage(self.fix_img)
            elastixImageFilter.SetMovingImage(self.moved_img)

            parameter_map = sitk.GetDefaultParameterMap("rigid")
            elastixImageFilter.SetParameterMap(parameter_map)

            elastixImageFilter.SetParameter("MaximumNumberOfIterations", '2000')
            elastixImageFilter.SetParameter("NumberOfSpatialSamples", '800')
            elastixImageFilter.SetParameter("Metric", 'AdvancedMattesMutualInformation')
            elastixImageFilter.SetParameter("Optimizer", 'AdaptiveStochasticGradientDescent')
            elastixImageFilter.SetParameter("NumberOfSamplesForExactGradient", '5000')
            elastixImageFilter.SetParameter("GridSpacingSchedule", ['4', '2', '1'])
            elastixImageFilter.SetParameter("NumberOfResolutions", '3')
            # 执行配准
            elastixImageFilter.Execute()
            # 保存结果图像
            regi_img = elastixImageFilter.GetResultImage()
            clipped_img = sitk.Clamp(regi_img, lowerBound=0)
            output_path = 'registered_ECT.nii.gz'
            sitk.WriteImage(clipped_img, output_path)

            # 通知主线程任务完成
            self.finished.emit(output_path)
        except Exception as e:
            print(f"Registration failed: {e}")

class fusion_display(QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # default setting
        self.ect_path = ''
        self.ct_path = ''

        self.ect_image = None  # 确保在这里初始化
        self.ct_image = None
        self.regi_ect_path = None

        # 别动
        self.ui.before_cornoal.type = 'sagittal'
        self.ui.before_axial.type = 'coronal'
        self.ui.before_sagittal.type = 'axial'

        self.ui.after_cornoal.type = 'sagittal'
        self.ui.after_axial.type = 'coronal'
        self.ui.after_sagittal.type = 'axial'

        self.ect_contrast = 50
        self.ct_contrast = 50

        self.ect_brightness = 50
        self.ct_brightness = 50

        self.ui.ect_contrast.setValue(self.ect_contrast)
        self.ui.ct_contrast.setValue(self.ct_contrast)
        self.ui.ect_brightness.setValue(self.ect_brightness)
        self.ui.ct_brightness.setValue(self.ct_brightness)

        # 启用滑块并连接信号
        self.ui.ect_contrast.setEnabled(True)
        self.ui.ct_contrast.setEnabled(True)
        self.ui.ect_brightness.setEnabled(True)
        self.ui.ct_brightness.setEnabled(True)

        # 连接滑块的信号到槽
        self.ui.ect_contrast.valueChanged.connect(lambda: self.contrast_change('ect'))
        self.ui.ct_contrast.valueChanged.connect(lambda: self.contrast_change('ct'))
        self.ui.ect_brightness.valueChanged.connect(lambda: self.brightness_change('ect'))
        self.ui.ct_brightness.valueChanged.connect(lambda: self.brightness_change('ct'))

        # 开启实时更新
        self.ui.ect_contrast.tracking = True
        self.ui.ct_contrast.tracking = True
        self.ui.ect_brightness.tracking = True
        self.ui.ct_brightness.tracking = True

        self.checkbox_ect_ct()

        self.ui.ect_button.stateChanged.connect(lambda: self.on_checkboxes_toggled())
        self.ui.ct_button.stateChanged.connect(lambda: self.on_checkboxes_toggled())

        self.ui.AutoRegister.clicked.connect(lambda: self.auto_regi(self.ect_image.image, self.ct_image.image))

    def update_display(self):
        self.on_checkboxes_toggled()
        if self.regi_ect_path != None:
            self.fusion_display_after(self.regi_ect_path)

    def default_display(self):
        if self.ect_path != '':
            self.ui.ect_button.setChecked(True)
        # if self.ct_path != '':
        #     self.ui.ct_button.setChecked(True)
        self.on_checkboxes_toggled()

    def checkbox_ect_ct(self):
        if self.ct_path == '':
            self.ui.ct_button.setEnabled(False)  # 禁用 CT 复选框
        else:
            self.ui.ct_button.setEnabled(True)  # 启用 CT 复选框

        if self.ect_path == '':
            self.ui.ect_button.setEnabled(False)  # 禁用 SPECT 复选框
        else:
            self.ui.ect_button.setEnabled(True)  # 启用 SPECT 复选框


    def setupdata(self,ect_path,ct_path):
        self.ect_path = ect_path
        self.ct_path = ct_path
        self.checkbox_ect_ct()
        if self.ect_path:
            self.ect_image = Image(self.ect_path, "ect")
            self.ect_image.read_image()
        if self.ct_path:
            self.ct_image = Image(self.ct_path, "ct")
            self.ct_image.read_image()
        self.default_display()

    def brightness_change(self,modality):
        # Access brightness from the appropriate checkbox (SPECT or CT)
        if modality == 'ect':
            self.ect_brightness = self.ui.ect_brightness.value()
        else:
            self.ct_brightness = self.ui.ct_brightness.value()

        print(self.ect_brightness)

        self.update_display()

    def contrast_change(self,modality):
        # Access contrast from the appropriate checkbox (SPECT or CT)
        if modality == 'ect':
            self.ect_contrast = self.ui.ect_contrast.value()
        else:
            self.ct_contrast = self.ui.ct_contrast.value()
        self.update_display()

    def on_checkboxes_toggled(self):
        ect_button = self.ui.ect_button.isChecked()
        ct_button = self.ui.ct_button.isChecked()

        if ect_button and ct_button:
            print("Both checkboxes are checked.")
            self.ui.ect_contrast.setEnabled(True)
            self.ui.ct_contrast.setEnabled(False)
            self.ui.ect_brightness.setEnabled(True)
            self.ui.ct_brightness.setEnabled(False)
            self.ect_brightness = self.ui.ect_brightness.value()
            self.ct_contrast = self.ui.ct_contrast.value()
            self.ect_contrast = self.ui.ect_contrast.value()
            self.ct_brightness = self.ui.ct_brightness.value()
            print(self.ect_brightness)
            self.fusion_display()
            # self.fusion_display()
        # 0 for ect, 1 for ct
        elif ect_button:
            self.ui.ect_contrast.setEnabled(True)
            self.ui.ect_brightness.setEnabled(True)
            self.single_display(0)
        elif ct_button:
            self.ui.ct_contrast.setEnabled(True)
            self.ui.ct_brightness.setEnabled(True)
            self.single_display(1)

    def fusion_display(self):

        print("start fusion display")
        self.ui.before_axial.fusion = True
        self.ui.before_cornoal.fusion = True
        self.ui.before_sagittal.fusion = True

        self.ui.before_axial.update_image_fusion(self.ct_image.get_image_data(), self.ect_image.get_image_data(), self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                         self.ct_image.voxel_size)
        self.ui.before_cornoal.update_image_fusion(self.ct_image.get_image_data(), self.ect_image.get_image_data(),  self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                           self.ct_image.voxel_size)
        self.ui.before_sagittal.update_image_fusion(self.ct_image.get_image_data(), self.ect_image.get_image_data(), self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                            self.ct_image.voxel_size)
        print("fail update")
        self.ui.before_axial.display_image_fusion(1)
        self.ui.before_cornoal.display_image_fusion(1)
        self.ui.before_sagittal.display_image_fusion(1)

    def single_display(self, modality):
        if modality == 0:
            self.ui.before_axial.fusion = False
            self.ui.before_cornoal.fusion = False
            self.ui.before_sagittal.fusion = False

            self.ui.before_axial.color = True
            self.ui.before_cornoal.color = True
            self.ui.before_sagittal.color = True

            self.ui.before_axial.update_image(self.ect_image.get_image_data(), self.ect_contrast, self.ect_brightness, 0,
                                              self.ect_image.voxel_size)
            self.ui.before_cornoal.update_image(self.ect_image.get_image_data(), self.ect_contrast, self.ect_brightness, 0,
                                                self.ect_image.voxel_size)
            self.ui.before_sagittal.update_image(self.ect_image.get_image_data(), self.ect_contrast,
                                                 self.ect_brightness, 0, self.ect_image.voxel_size)

            self.ui.before_axial.display_image(1)
            self.ui.before_cornoal.display_image(1)
            self.ui.before_sagittal.display_image(1)

        elif modality == 1:
            self.ui.before_axial.fusion = False
            self.ui.before_cornoal.fusion = False
            self.ui.before_sagittal.fusion = False

            self.ui.before_axial.color = False
            self.ui.before_cornoal.color = False
            self.ui.before_sagittal.color = False

            self.ui.before_axial.update_image(self.ct_image.get_image_data(), self.ct_contrast[0], self.ct_brightness[0], 1,
                                              self.ct_image.voxel_size)
            self.ui.before_cornoal.update_image(self.ct_image.get_image_data(), self.ct_contrast[0], self.ct_brightness[0],1,
                                                self.ct_image.voxel_size)
            self.ui.before_sagittal.update_image(self.ct_image.get_image_data(), self.ct_contrast[0],self.ct_brightness[0], 1,
                                                 self.ct_image.voxel_size)

            self.ui.before_axial.display_image(1)
            self.ui.before_cornoal.display_image(1)
            self.ui.before_sagittal.display_image(1)

    def manual_regi(self):
        pass

    def auto_regi(self, moved_img, fix_img):
        self.worker = RegistrationWorker(moved_img, fix_img)
        self.worker.finished.connect(self.on_registration_finished)
        self.worker.start()

    def on_registration_finished(self, output_path):
        # 处理注册结果
        print(f"Registration completed. Output saved at: {output_path}")
        # 调用融合显示
        self.fusion_display_after(output_path)
    def fusion_display_after(self, ect_path):

        self.regi_ect_path = ect_path
        self.reg_ect_image = Image(ect_path, "ect")
        self.reg_ect_image.read_image()

        self.ui.after_axial.fusion = True
        self.ui.after_cornoal.fusion = True
        self.ui.after_sagittal.fusion = True

        self.ui.after_axial.update_image_fusion(self.ct_image.get_image_data(), self.reg_ect_image.get_image_data(), self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                        self.ct_image.voxel_size)
        self.ui.after_cornoal.update_image_fusion(self.ct_image.get_image_data(), self.reg_ect_image.get_image_data(),self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                          self.ct_image.voxel_size)
        self.ui.after_sagittal.update_image_fusion(self.ct_image.get_image_data(), self.reg_ect_image.get_image_data(), self.ect_brightness, self.ct_brightness, self.ect_contrast, self.ct_contrast,
                                                           self.ct_image.voxel_size)

        self.ui.after_axial.display_image_fusion(1)
        self.ui.after_cornoal.display_image_fusion(1)
        self.ui.after_sagittal.display_image_fusion(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = fusion_display()
    ect_path = r'E:\OneDrive - University of Macau\RESEARCH\Project\BIGDOSE\data\0000915556\time1_PT.nii.gz'
    ct_path = r'E:\OneDrive - University of Macau\RESEARCH\Project\BIGDOSE\data\0000915556\time1_CT2ECT.nii.gz'
    window.setupdata(ect_path, ct_path)
    window.show()
    sys.exit(app.exec())
