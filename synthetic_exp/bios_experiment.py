
import numpy as np
import pickle
from info import *
from prompts import *
import os
from tqdm import tqdm


def run_experiment(bedrock, trajectory, model='claude', starting_prompt = "", style_prompt = ""):

    if model == 'claude':
        call_model = anthropic_complete_prompt
    elif model == 'llama':
        call_model = llama_complete_prompt
    else:
        raise NotImplementedError("This method is not implemented yet.")

    all_bios, preferences, q_list, correct_answers, feedbacks = trajectory
    name = all_bios[0].split('was born')[0].strip()

   
    next_prompt = starting_prompt.format(bios = '\n\n'.join(all_bios))

    results, bios = [], []

    feedback_indexes = [a[0] for a in feedbacks]
    feedbacks = [a[1] for a in feedbacks]
    for i in range(len(q_list)): # number of questions

        if i in feedback_indexes: # time to give feedback
            feedback = feedbacks[feedback_indexes.index(i)] # get the feedback for this index
            feedback = feedback + "Use this information for your answer." + style_prompt

            next_prompt = add_prompt(next_prompt, feedback)

        q = questions[q_list[i]].format(name=name)
        next_prompt  = add_prompt(next_prompt, q)

        completed_prompt = call_model(next_prompt, bedrock, temperature=0)
        next_prompt = add_answer(next_prompt, completed_prompt)
        res = precision_recall_json(completed_prompt, [correct_answers[i]])
        results.append(res)
        bios.append(completed_prompt)
        
    return results, bios


def run_experiment_memory(bedrock, trajectory, model='claude', starting_prompt = "", memory_prompt = "", style_prompt = ""):
    if model == 'claude':
        call_model = anthropic_complete_prompt
    elif model == 'llama':
        call_model = llama_complete_prompt
    else:
        raise NotImplementedError("This method is not implemented yet.")

    all_bios, preferences, q_list, correct_answers, feedbacks = trajectory
    name = all_bios[0].split('was born')[0].strip()

    prompt = starting_prompt.format(bios='\n\n'.join(all_bios))
    
    scratchpad = memory_prompt
    
    next_prompt = prompt.format(bios='\n\n'.join(all_bios))
    notes_content = ""
    results, bios,full, pads = [], [], [], []

    feedback_indexes = [a[0] for a in feedbacks]
    feedbacks = [a[1] for a in feedbacks]

    for i in range(len(q_list)):
        if i in feedback_indexes:
            feedback = feedbacks[feedback_indexes.index(i)]
            feedback += style_prompt
            
            next_prompt = add_prompt(next_prompt, feedback)

        q = questions[q_list[i]].format(name=name)
        next_prompt = add_prompt(next_prompt, q)
        
        call_prompt = next_prompt + scratchpad.format(content=notes_content)
        completed_prompt = call_model(call_prompt, bedrock, temperature=0)
        
        # Extract new scratchpad content
        try:
            new_notes = completed_prompt.split("Notes:")[1].split("Answer:")[0].strip()
            notes_content = new_notes
        except:
            pass

        next_prompt = add_answer(next_prompt, completed_prompt)
        try:
            res = precision_recall_json(completed_prompt, [correct_answers[i]])
        except:
            res = [0,0]
            print(completed_prompt)#res = precision_recall_json(completed_prompt, [correct_answers[i]])
        results.append(res)
        bios.append(completed_prompt)
        pads.append(notes_content)
    return results, bios, pads



def run_experiment_on_dataset(bedrock, dataset, starting_prompt, style_prompt, memory_prompt = "", name = "", models=['claude', 'llama'], output_dir = 'exp_results/bios/'):

    all_results = {model: [] for model in models}
    all_docs = {model: [] for model in models}

    #  make them run in parallel to save time. 
    from concurrent.futures import ThreadPoolExecutor

    def process_model(model, dataset, memory_prompt, starting_prompt, style_prompt, name):
        model_results, model_docs = [], []
        for traj in tqdm(dataset, desc=f"Running {name} experiment with {model}..."):
            if memory_prompt:
                results, docs = run_experiment_memory(bedrock, traj, model=model,\
                                            starting_prompt = starting_prompt,\
                                            memory_prompt = memory_prompt,\
                                            style_prompt = style_prompt )
            else:
                results, docs = run_experiment(bedrock, traj, model=model,\
                                            starting_prompt = starting_prompt,\
                                            style_prompt = style_prompt )
                
            model_results.append(results)
            model_docs.append(docs)
        return model, model_results, model_docs

    with ThreadPoolExecutor() as executor:
        futures = []
        for model in models:
            future = executor.submit(process_model, model, dataset, memory_prompt, 
                                   starting_prompt, style_prompt, name)
            futures.append(future)
        
        for future in futures:
            model, model_results, model_docs = future.result()
            all_results[model] = model_results
            all_docs[model] = model_docs

    models_str = '_'.join(models)
     
    output_dir = os.path.join(output_dir, f'{name}_{models_str}')
    with open(f'{output_dir}.npy', 'wb') as f:
        np.save(f, all_results)
            
    with open(f'{output_dir}_docs.pkl', 'wb') as f:
        pickle.dump(all_docs, f)

    return all_results, all_docs


def get_results_from_file(file_path, methods = None):

    if methods is None:
        methods = ['full_memory', 'icl', 'cot', 'chain_of_notes', 'scratchpad', 'coreset']

    models = ['claude', 'llama']
    results = {}
    for method in methods:
        with open(f'{file_path}{method}_claude_llama.npy', 'rb') as f:
            method_results = np.load(f)
        
        results[method] = method_results
        print(f"-------- {method} --------")
        for model in models:
            res = results[method][model]
            res = np.array(res)
            prec = np.mean(res[:,:,0])
            rec = np.mean(res[:,:,1])
            print(f"{model}: {prec:.3f}, {rec:.3f}")
        
    return results

def main():

    import boto3

    # Set up AWS credentials
    session = boto3.Session()
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

    with open('bios_dataset_100.pkl', 'rb') as f:
        dataset = pickle.load(f)
        print(f"Loaded datastet with {len(dataset)} trajectories, each with {len(dataset[0][2])} questions.")

    file_path = 'exp_results/bios/'
    # create the file path if it doesn't exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # normal
    run_experiment_on_dataset(bedrock, dataset, full_memory_prompt, full_memory_style_prompt, 
                              name = "full_memory", models=['claude', 'llama'], output_dir = file_path)

    # in context learning
    run_experiment_on_dataset(bedrock, dataset, in_context_prompt, in_context_style_prompt, 
                              name = "icl", models=['claude', 'llama'], output_dir = file_path)
    
    # chain of thought
    run_experiment_on_dataset(bedrock, dataset, cot_prompt, cot_style_prompt, 
                              name = "cot", models=['claude', 'llama'], output_dir = file_path)
    
    # chain of notes
    run_experiment_on_dataset(bedrock, dataset, chain_of_notes_prompt, chain_of_notes_style_prompt, 
                              name = "chain_of_notes", models=['claude', 'llama'], output_dir = file_path)
    
    # scratchpad
    run_experiment_on_dataset(bedrock, dataset, scratchpad_prompt, scratchpad_style_prompt, 
                              memory_prompt = scratchpad_content,
                              name = "scratchpad", models=['claude', 'llama'], 
                              output_dir = file_path)

    # coreset
    run_experiment_on_dataset(bedrock, dataset, coreset_prompt, coreset_style_prompt, 
                              memory_prompt = coreset_content,
                              name = "coreset", models=['claude', 'llama'], 
                              output_dir = file_path)

    get_results_from_file(file_path, methods = ['full_memory', 'icl', 'cot', 'chain_of_notes', 'scratchpad', 'coreset'])

if __name__ == "__main__":
    main()