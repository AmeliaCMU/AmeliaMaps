import os
import random
import math
from typing import Tuple, List
import numpy as np
import pickle as pkl
import pandas as pd

def impute(seq: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """ Imputes missing data via linear interpolation. 
    
    Inputs
    ------
        seq[pd.DataFrame]: trajectory sequence to be imputed.
        seq_len[int]: length of the trajectory sequence.
    
    Output
    ------
        seq[pd.DataFrame]: trajectory sequence after imputation.
    """
    # Create a list from starting frame to ending frame in agent sequence
    conseq_frames = set(range(int(seq[0, 0]), int(seq[-1, 0])+1))
    # Create a list of the actual frames in the agent sequence. There may be missing 
    # data from which we need to interpolate.
    actual_frames = set(seq[:, 0])
    # Compute the difference between the lists. The difference represents the missing
    # data points
    missing_frames = list(sorted(conseq_frames - actual_frames))
    # print(missing_frames)

    # Insert nan rows on where the missing data is. Then, interpolate. 
    if len(missing_frames) > 0:
        seq = pd.DataFrame(seq)
        for missing_frame in missing_frames:
            df1 = seq[:missing_frame]
            df2 = seq[missing_frame:]
            df1.loc[missing_frame] = np.nan
            seq = pd.concat([df1, df2])

        seq = seq.interpolate(method='linear').to_numpy()[:seq_len]
    return seq

class SequenceLoader:
    def __init__(
        self, 
        hist_len: int = 10, 
        pred_len: int = 20, 
        step: int = 1, 
        skip: int = 10, 
        min_agents: int = 2,
        max_agents: int = 100,
        to_process: float = 1.0,
        use_ego_agent: bool = True,
        parallel: bool = False,
    ) -> None:
        """ Dataset loader for the SWIM datasets. 
        
        Inputs:
        -------
            - in_data_dir[str]: directory containing dataset files in the format:
                Frame, ID, Altitude, Speed, Heading, Lat, Lon, Range, Bearing, x, y 
            - out_data_dir[str]: output directory where context information will be saved.
            - map_dir[str]: directory to the input map.
            - map_filepath[str]: path to map file. 
            - limits_filepath[str]: path to reference file containing the limits of the input map.
            - hist_len[int]: number of time-steps in input trajectories.
            - pred_len[int]: number of time-steps in output trajectories.
            - step[int]: subsampling step for the trajectory. 
            - skip[int]: number of consecutive frames to skip while making the dataset.
            - min_agents[int]: minimum number of agents that should be in a sequence.
            - max_agents[int]: maximum number of agents that should be in a sequence.
            - map_dims[list[h, w, c]]: the dimensions of the context patch for each trajectory.
            - map_scale[float]: scale to which the map will be re-scaled for pre-processing.
            - to_process[float]: percentage of the data to process
            - use_ego_agent: bool = True,
            - parallel: bool = False,
        """

        # Altitude, Speed, Heading, Lat, Lon, Range, Bearing, x, y
        self.dim = 9
        self.ll_idx = [False, False, False, True, True, False, False, True, True]
        self.step, self.skip = step, skip
        self.hist_final_len = int(math.ceil(hist_len / step))
        self.pred_final_len = int(math.ceil(pred_len / step))
        
        self.seq_len = hist_len + pred_len
        self.seq_final_len = self.hist_final_len + self.pred_final_len
        self.min_agents, self.max_agents = min_agents, max_agents
        self.to_process = to_process

        # Altitude, Speed, Heading, Lat, Lon, Range, Bearing, Type, x, y      
        self.raw_idx = {
            'Frame': 0, 'ID': 1, 'Altitude': 2, 'Speed': 3, 'Heading': 4, 'Lat': 5, 'Lon': 6, 
            'Range': 7, 'Bearing': 8, 'Type': 9, 'x': 10, 'y': 11
        }
        self.idxs = {
            'Altitude': 0, 'Speed': 1, 'Heading': 2, 'Lat': 3, 'Lon': 4, 'Range': 5, 'Bearing': 6, 
            'x': 7, 'y': 8
        }

        self.use_ego_agent = use_ego_agent 
        self.parallel = parallel
    
    def process_seq(self, frame_data: pd.DataFrame, frames: list, seq_idx: int) -> np.array:
        """ Processes all valid agent sequences. 

        Inputs
        ------
            frame_data[pd.Dataframe]: unprocessed scenario sequence data.
            frames[list]: list of frame indeces.
            seq_idx[int]: current sequence being processed.

        Outputs
        -------
            seq[np.array]: all processed sequences if 'num_agents_considered' >= self.min_agents.
                Otherwise, returns None
        """
        seq_mask = [False, False, True, True, True, True, True, True, True, False, True, True]
        # All data for the current sequence: from the curr index i to i + sequence length
        seq_data = np.concatenate(frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        # IDs of agents in the current sequence
        unique_agents = np.unique(seq_data[:, 1])
        num_agents = len(unique_agents)
        if num_agents < self.min_agents or num_agents > self.max_agents:
            return None, None, None

        num_agents_considered = 0
        seq = np.zeros((num_agents, self.seq_final_len, self.dim))
        agent_id_list = []
        agent_type_list = []

        for _, agent_id in enumerate(unique_agents):
            # Current sequence of agent with agent_id
            agent_seq = seq_data[seq_data[:, 1] == agent_id]

            # Start frame for the current sequence of the current agent reported to 0
            pad_front = frames.index(agent_seq[0, 0]) - seq_idx
            
            # End frame for the current sequence of the current agent: end of current agent 
            # path in the current sequence. It can be sequence length if the pedestrian 
            # appears in all frame of the sequence or less if it disappears earlier.
            pad_end = frames.index(agent_seq[-1, 0]) - seq_idx + 1
            
            # Exclude trajectories less then seq_len
            if pad_end - pad_front != self.seq_len:
                continue
            
            # Impute missing data using interpolation 
            agent_id_list.append(int(agent_id))
            agent_type_list.append(int(agent_seq[0, self.raw_idx['Type']]))
            agent_seq = impute(agent_seq, self.seq_len)[:, seq_mask]
            seq[num_agents_considered, pad_front:pad_end] = agent_seq[::self.step]
            num_agents_considered += 1
        
        if num_agents_considered < self.min_agents:
            return None, None, None
        
        return seq[:num_agents_considered], agent_id_list, agent_type_list
  
    def process_file(self, f) -> Tuple[List, List, List, List, List, List]:
        """ Processes a single data file. 
        
        Inputs
        ------
            f[str]: file to be processed. 

        Outputs
        -------
            num_agents_scenario[list]: list of number of agents in each scenario.
            ego_agent_id[list]: list of chosen ego-agents in each scenario. 
            sequences[list]: list of trajectory sequences in absolute frame.
            rel_sequences[list]: list of trajectory sequences in relative frame.
            semantic_maps[list]: list of map context corresponding to a scenario.
            bboxes[list]: list of bounding boxes in each map of each scenario.

            NOTE: returns None if the file is empty, or if there are no sequences to process. 
        """
        num_agents_scenario = []
        ego_agent_id_list = []
        agent_id_list = []
        agent_type_list = []
        sequences = []
        rel_sequences = []
        semantic_maps = []

        # check if file is empty
        if os.stat(f).st_size == 0:
            return

        # load trajectory data
        data = pd.read_csv(f)
        data = data[:][data['Type'] == 0] 
        # get the number of unique frames
        frames = data.Frame.unique().tolist()
        frame_data = []
        for frame_num in frames:
            frame = data[:][data.Frame == frame_num] 
            frame_data.append(frame)

        num_sequences = int(math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))
        if num_sequences < 1:
            return
        
        for i in range(0, num_sequences * self.skip + 1, self.skip):
        
            seq, agent_id, agent_type = self.process_seq(frame_data=frame_data, frames=frames, seq_idx=i)
            if seq is None:
                continue
            
            # randomly select the ego-agent 
            num_agents, _, _ = seq.shape
            
            ego_agent_id = random.randint(a=0, b=num_agents-1)

            # process map 
            #semantic_map = self.process_semantic_map_context(seq=seq, agent_id=ego_agent_id)

            # scenario 
            num_agents_scenario.append(num_agents)
            ego_agent_id_list.append(ego_agent_id)
            agent_id_list.append(agent_id)
            agent_type_list.append(agent_type)
            sequences.append(seq)
            #semantic_maps.append(semantic_map)
        
        return sequences
            
