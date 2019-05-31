# Fluidity of Concept Representations

# Reader
Here you will find a general reader of the data (the files used are in read_pereira and it can be initiated separately by running test_pereira_reader.py). Note: the reader only selects concepts with an abstract and concrete rating from the dataset of pereira, since that is what this study focussed on. 

# Select voxels
You'll find the files used to select voxels. We selected specific regions of interest by using the regions specified in the matlab files of the participants (based on previous literature); furthermore we selected stable voxels based on presentations of a concept across paradigms; lastly we did a searchlight selection in which we, for all voxels, selected a voxel and its neighbours (this slightly differs from the searchlight selection that you could run with the file provided by pereira, in matlab since they used a different approach). The files used can be found in select_voxels and it can be initiated separately by running get_all_voxel_selections.py.

# Run analyses
With these voxel selections we ran various analyses. In this github you'll find the classification (= decoding), clustering, encoding, and RSA analyses. The files for this you can find in the "analyses" folder and you can initiate them by running pereira_runner.py. The relational analyses (encoding and RSA) employ different kinds of representations (textual, visual (based on the presented pictures), and a randomly initiated vector).

# Presenting the results
Lastly, you will find that by running final_results.py, the results will be plotted in the figures as we used them in the paper. The files for this can be found in the "result_analyses" folder
