# CoNLL_script

Here you will find a general reader of the data (the files used are in read_pereira and it can be initiated separately by running test_pereira_reader.py). Note: the reader only selects concepts with an abstract and concrete rating, since that is what I am looking into. If you want all the concepts to be selected, you'll need to change some if-statements in read_pereira.py. Furthermore, I added the concreteness rating of the concepts as an extra feature in the block, but I don't think that should lead to any problems.

Furthermore, you'll find the files used to select voxels. We selected specific regions of interest by using the regions specified in the matlab files of the participants; furthermore we selected stable voxels based on presentations of a concept across paradigms; lastly we did a searchlight selection in which we, for all voxels, selected a voxel and its neighbours (this slightly differs from the searchlight selection that you could run with the file provided by pereira, since they used a different approach). The files used can be found in select_voxels and it can be initiated separately by running get_all_voxel_selections.py.

With these selections I ran various analyses. In this github you'll find the classification (= decoding), clustering, encoding, and RSA analyses. The files for this you can find in the "analyses" folder and you can initiate them by running pereira_runner.py. The encoding employs different kinds of representations (textual, visual (based on the presented pictures), and a randomly initiated vector).

Lastly, you will find that by running final_results.py, the results will be plotted in the figures as we used them in the paper. The files for this can be found in the "result_analyses" folder.

Kind regards,
