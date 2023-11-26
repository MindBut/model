from faulthandler import disable
from APE import data, llm
from configs import config, template
import random
import numpy as np 
import pandas as pd


def generate_prompt(instructions, config):
    """Generates prompt using prompt generator

    Args:
        instructions (list): list of instructions that are to be sampled
        config (dictionary): config['generation']

    Returns:
        prompts (list): 
    """
    queries = []
    demos = data.subsample_instruction(instructions, config['num_subsamples'])
    for demo in demos: 
        temp = config.Config.GENERATION_TEMPLATE 
        filled_prompt = template.GenerationTemplate(temp).fill(demo)
        queries.append(filled_prompt)

    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    prompts = model.complete(
        queries, n=config['num_prompts_per_subsample'])
    return prompts


def eval_prompt(prompt, goals, depressed, anxious, df, config):
    """Generates evaluation prompt

    Args:
        prompt_eval_template (string): prompt template not filled 
            [PROMPT] [GOAL] [DEPRESSED] [ANXIOUS] [INPUT]
            prompt (list) : All generated instructions
            df (dataframe)
        config (dictionary)
    """
    # Random sample prompt from generated prompt list 
    indices = random.sample(len(prompt), config['num_subsamples'])
    instructions = [prompt[i] for i in indices]

    goal, depressed, anxious, category, input = data.subsample_data(df, goals, depressed, anxious, config['num_subsamples'])
    
    expanded_instructions = np.repeat(instructions, 3)
    expanded_goals = np.repeat(goal, 3)
    expanded_depressed = np.repeat(depressed, 3)
    expanded_anxious = np.repeat(anxious, 3)
    expanded_input_data = np.repeat(input, 3)
    
    queries = []
    for inst in instructions:
        temp = config.Config.EVAL_TEMPLATE 
        for i in range(config['num_inputs']):
            filled_prompt = template.EvalTemplate(temp).fill(inst, goals[i], depressed[i], anxious[i], input[i])
            queries.append(filled_prompt)

    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    outputs = model.complete(
        queries, n=config['num_outputs_per_subsample'])

    # Generate DataFrame 
    df = pd.DataFrame({
        "instruction": expanded_instructions,
        "goals": expanded_goals,
        "depressed": expanded_depressed,
        "anxious": expanded_anxious,
        "input": expanded_input_data,
        "output": outputs
    })

    # Get rid of header
    df = df.rename_axis(None, axis=1)
    
    return df


def score_prompt(df, config):
    
    queries = []
    for i in df:
        temp = config.Config.SCORE_TEMPLATE 
        filled_prompt = template.ScoreTemplate(temp).fill(df[i])
        queries.append(filled_prompt) 
    
    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    scores = model.complete(
        queries, n=config['num_scores'])
    
    df = df.insert(6, 'NewColumn', scores)
    
    return df
    