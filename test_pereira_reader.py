from read_dataset.read_pereira import PereiraReader
from nilearn import plotting

# SET THE DATA DIR: 
data_dir = "DATA_DIR"

print("\n\Pereira Data")
pereira_reader = PereiraReader(data_dir, paradigm =1)

pereira_data = pereira_reader.read_all_events()
blocks, xyz_voxels = pereira_data


for subject_id in blocks.keys():
    print(subject_id)
    
    for block in blocks[subject_id]:
        sentences = block.sentences
        scans = [event.scan for event in block.scan_events]
        stimuli = [event.stimulus_pointers for event in block.scan_events]
        timestamps = [event.timestamp for event in block.scan_events]
        concreteness_id = [block.concreteness_id]

        print("\n\nBLOCK: " + str(block.block_id))
        print("Number of scans: " + str(len(scans)))
        print("Number of stimuli: " + str(len(stimuli)))
        print("Number of timestamps: " + str(len(timestamps)))
        print("Stimuli: \n" + str(stimuli))
        print("Sentence: \n" + str(sentences))
    
    print(len(blocks[subject_id]))
        
