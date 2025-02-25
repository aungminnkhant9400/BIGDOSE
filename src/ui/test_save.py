
import nibabel as nib

folder = r'C:\Users\User\Desktop\BIGDOSE_Lu177\data'

lung_lower_lobe_left = folder + '\\' + 'organs_in_ui\\' + 'lung_lower_lobe_left.nii.gz'

lung_upper_lobe_left = folder + '\\' + 'organs_in_ui\\' + 'lung_upper_lobe_left.nii.gz'

lung_lower_lobe_left_head = nib.load(lung_lower_lobe_left)

lung_lower_lobe_left_array = lung_lower_lobe_left_head.get_fdata()

print(000)


lung_upper_lobe_left_head = nib.load(lung_upper_lobe_left)

lung_upper_lobe_left_array = lung_upper_lobe_left_head.get_fdata()

print(111)

lung_left_whole_array = lung_upper_lobe_left_array + lung_lower_lobe_left_array

nimg = nib.Nifti1Image(lung_left_whole_array, lung_lower_lobe_left_head.affine, lung_lower_lobe_left_head.header)
nimg.to_filename(folder + '\\' + 'organs_in_ui\lung_lobe_left.nii.gz')

print(222)