import numpy as np
import torch

# Dictionary
action_dict = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 1, 0],
    3: [0, 1, 1],
    4: [1, 0, 0],
    5: [1, 0, 1],
    6: [1, 1, 0],
    7: [1, 1, 1]
}

action_map = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])


# To convert the action value (0-5) to the action vector
def convert_to_vector(action_val):
    return action_dict[action_val]

# To convert a list of action values into a list of action vectors
def convert_list_to_vectors(action_vals):
    return action_map[action_vals]

# To convert the action vector to the action value (0-5)
def convert_to_value(action_vector):
    return action_vector[0]*4 + action_vector[1]*2 + action_vector[2]


# This SumTree code was modified and adapted from 
# https://github.com/XinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch/tree/main
class SumTree(object):
    """
    Story data with its priority in the tree.
    Tree structure and array storage:

    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  
        self.tree_capacity = 2 * buffer_capacity - 1 
        self.tree = np.zeros(self.tree_capacity)

    def update_priority(self, data_index, priority):
        ''' Update the priority for one transition according to its index in buffer '''
        tree_index = data_index + self.buffer_capacity - 1 
        change = priority - self.tree[tree_index] 
        self.tree[tree_index] = priority  
        # then propagate the change through the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def prioritized_sample(self, N, batch_size, beta):
        ''' sample a batch of index and normlized IS weight according to priorites '''
        batch_index = np.zeros(batch_size, dtype=np.uint32)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            buffer_index, priority = self._get_index(v)
            batch_index[i] = buffer_index
            prob = priority / self.priority_sum  
            IS_weight[i] = (N * prob) ** (-beta)
        Normed_IS_weight = IS_weight / IS_weight.max()  

        return batch_index, Normed_IS_weight

    def _get_index(self, v):
        ''' sample a index '''
        parent_idx = 0  
        while True:
            child_left_idx = 2 * parent_idx + 1  
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  
                tree_index = parent_idx  
                break
            else:  
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  
        return data_index, self.tree[tree_index] 

    @property
    def priority_sum(self):
        return self.tree[0] 

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()

# This code was also modified and adapted from 
# https://github.com/XinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch/tree/main
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, initial_p, final_p):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

