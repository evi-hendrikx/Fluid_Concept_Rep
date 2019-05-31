
from select_voxels.selection_methods import SelectStable, SelectROI, SelectSearchlight


user_dir = "/home/10978089/"


roi_selection = SelectROI(user_dir)
searchlight_selection = SelectSearchlight(user_dir)
stable_selection_classification = SelectStable(user_dir, "classification")
stable_selection_encoding = SelectStable(user_dir, "encoding")
stable_selection_RSA = SelectStable(user_dir, "RSA")
stable_selection_clustering = SelectStable(user_dir, "clustering")

for selection in (stable_selection_clustering, stable_selection_RSA, roi_selection, searchlight_selection, stable_selection_classification, stable_selection_encoding):
    voxel_selection = selection.select_voxels()
    print(voxel_selection.keys())
    

  
    
