import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from datetime import datetime
import torch

if __name__ == "__main__":
    # option 1: provide input and output as file paths
    input_path = r'C:\Users\aungm\Desktop\BigDoseData\CTimg.nii.gz'

    output_path = r'C:\Users\aungm\Desktop\BigDoseData\head_glands_cavities'

    start_time = datetime.now()

    #totalsegmentator(input_path, output_path, roi_subset=['femur_left'])

    #totalsegmentator(input_path, output_path, task='head_glands_cavities')



    # totalsegmentator(input_path, output_path)

    # option 2: provide input and output as nifti image objects
    # input_img = nib.load(input_path)
    # output_img = totalsegmentator(input_img)
    # nib.save(output_img, output_path)




    #totalsegmentator(input_path, output_path, task='head_glands_cavities',  fast=True)  # fastest

    totalsegmentator(input_path, output_path, task='head_glands_cavities',fast=True, fastest=True,  roi_subset=['parotid_gland_left', 'parotid_gland_right'])  #  4.2 分钟

    #totalsegmentator(input_path, output_path, task='head_glands_cavities', fast=True, fastest=True, roi_subset=['parotid_gland_left'])  # fastest

    ## 加注释



    
    totalsegmentator(input_path, output_path, roi_subset=['liver', 'lung_lower_lobe_left', 'kidney_right', 'kidney_left',
                                                                        'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_lower_lobe_right',
                                                                        'lung_upper_lobe_right', 'adrenal_gland_left', 'adrenal_gland_right', 'aorta',
                                                          'colon', 'duodenum', 'esophagus', 'gallbladder', 'heart', 'pancreas', 'small_bowel', 'spleen',
                                                          'stomach', 'trachea', 'urinary_bladder', 'prostate', 'brain'], fast=True, fastest=True)  ## 1.8 minuts
                                                          

                                                          


    end_time = datetime.now()

    time_for_each = end_time - start_time

    print('time_for_each： seconds', time_for_each.total_seconds())