import numpy as np

class GridWorld:
    """Making a GridWorld object to make different grid size custom environments to mimic Fronze Lake"""

    def __init__(self, size=4):
        """Setting up main parts of the Grid environment

        Args:
            size (4, optional): square grid side length. Defaults to 4.
        """
        self.size = size  # - square grid size
        self.nS = size * size  # - number of states
        self.nA = 4  # - number of actions

        self.UP, self.DOWN, self.LEFT, self.RIGHT = range(4)

        self.goal = self.nS - 1
        self.holes = {5, 7, 11}

        self.P = self.build_transition_table()  # - the MDP

    def build_transition_table(self):
        """Creating the transition function using a function and loops rather than writing a large dictionary by hand"""
        P = {
            s: {a: [] for a in range(self.nA)} for s in range(self.nS)
        }  # ^ P[state][action] = []

        for s in range(self.nS):  # * setting transitions for every state
            row, col = divmod(
                s, self.size
            )  # ? divmod(a, b) -> (a // b, a % b), this maps index to grid position
            #           state 6 in 4x4 grid
            #           6 // 4 = 1  (row)
            #           6 % 4  = 2  (col)

            for a in range(self.nA):

                if (
                    s == self.goal or s in self.holes
                ):  # * if state is already at goal or hole, loop back to itself with 0 reward

                    P[s][a].append(
                        (1.0, s, 0, True)
                    )  # ? (1.0, s, 0, True) -> (probability, next_state, reward, done)
                    
                    continue
                
                new_row, new_col = row, col # * assume we don't move
                
                # ! movement logic along with boundary checks
                if a == self.UP: # ^ up one row, but max() prevents us from going out of bounds (negative row index)
                    new_row = max(row-1, 0)
                elif a == self.DOWN: # ^ down one row, but min() prevents us from going out of bounds (exceeding row index)
                    new_row = min(row+1, self.size-1)
                elif a == self.LEFT: # ^ left one column, but max() prevents us from going out of bounds (negative column index)
                    new_col = max(col-1, 0)
                elif a == self.RIGHT: # ^ right one column, but min() prevents us from going out of bounds (exceeding column index)
                    new_col = min(col+1, self.size-1)
                    
                ns = new_row * self.size + new_col # - ns -> new state index after movement
                # ? (1,2) → 1*4 + 2 = 6 -> math for ns
                
                reward = -0.01 # - time cost
                done = False # - default done is False, is True when we hit a hole or the goal
                
                if ns in self.holes: # * hole is bad (reward = -1) and ends episode (done = True)
                    reward = -1
                    done = True
                    
                if ns in self.goal: # * goal is good (reward = 1) and ends episode (done = True)
                    reward = 1
                    done = True
                    
                P[s][a].append((1.0, ns, reward, done)) # ? P(s′∣s,a) = 1
                
                
            return P # * returns the full transition table for the MDP
                # ^ this transition function is deterministic
                # ^ each action has 1.0 probability of leading to the next state
                # ^ not slippery, yet
            
            





