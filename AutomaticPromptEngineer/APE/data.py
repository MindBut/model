import random
import numpy as np
import pandas as pd

def subsample_instruction(instructions, subsample_size):
    """Subsample instruction

    Args:
        instructions (list)
        subsample_size (int): number of subsamples to get from 
    """
    indices = random.sample(range(len(instructions)), subsample_size)
    instruction_samples = [instructions[i] for i in indices]
    return instruction_samples

def subsample_data(data, goals, depressed, anxious, subsample_size):
    """Generate subsamples from data

    Args:
        data (dataframe): 4 columns of categorical labels, 1 column input, 1 column output
        goals (list): list of counseling goals
        depressed (list): list of the extent of depression
        anxious (list): list of the extent of anxiety
        subsample_size (int): number of subsamples to generate

    Returns:
        goal, depressed, anxious, category, input (list): list of subsamples
    """
    n = data.shape[0]-1   # number of datas
    indices_goal = random.sample(range(len(goals)), subsample_size)
    indices_depressed = random.sample(range(len(depressed)), subsample_size)
    indices_anxious = random.sample(range(len(anxious)), subsample_size)
    indices_data = random.sample(n, subsample_size)
    
    goal = [goals[i] for i in indices_goal]
    depressed = [depressed[i] for i in indices_depressed]
    anxious = [anxious[i] for i in indices_anxious]
        
    category = []
    input = []
    for i in indices_data:
        value = ""
        if pd.notnull(data[i+1][0]):
            value += data[i+1][0]
        if pd.notnull(data[i+1][1]):
            if value:
                value += '/'
            value += data[i+1][1]
        if pd.notnull(data[i+1][2]):
            if value:
                value += '/'
            value += data[i+1][2]
        if pd.notnull(data[i+1][3]):
            if value:
                value += '/'
            value += data[i+1][3]
        category.append(value)
        
        assert pd.notnull(data[i+1][4])
        input.append(data[i+1][4])

    return goal, depressed, anxious, category, input
    