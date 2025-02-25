import os
import SimpleITK as sitk

class Image:
    def __init__(self, path="", type=""):
        self.path = path
        self.type = type
        self.image = None

    def read_image(self):
        if self.path.endswith(".dcm"):
            # To be overwrite
            self.path = os.path.dirname(self.path)
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.path)
            reader.SetFileNames(dicom_names)
            self.image = reader.Execute()
        else:
            self.image = sitk.ReadImage(self.path)
        current_direction = self.image.GetDirection()
        if current_direction == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            print("Direction matches (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0). Correcting direction...")

            # Create a new image with the desired direction
            self.image = sitk.Flip(self.image, [False, False, True])
        self.image_show = None
        self.voxel_size = self.image.GetSpacing()
        self.voxel_size = tuple(round(v, 2) for v in self.voxel_size)
        self.origin = self.image.GetOrigin()
        self.matrix_size = self.image.GetSize()
    def get_image_data(self):
        # 在需要时，按需将图像转换为 NumPy 数组
        if self.image_show is None:  # 懒加载，避免不必要的内存占用
            self.image_show = sitk.GetArrayFromImage(self.image)
        return self.image_show

