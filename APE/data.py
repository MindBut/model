import random


def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    categories, inputs, outputs = data
    assert len(inputs) == len(outputs)
    assert len(categories) == len(input)
    indices = random.sample(range(len(inputs)), subsample_size)
    categories = [categories[i] for i in indices]
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return categories, inputs, outputs


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    categories, inputs, outputs = data
    assert len(inputs) == len(outputs)
    assert len(categories) == len(input)
    indices = random.sample(range(len(inputs)), split_size)
    categories1 = [categories[i] for i in indices]
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    categories2 = [categories[i] for i in indices if i not in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (categories1, inputs1, outputs1), (categories2, inputs2, outputs2)
