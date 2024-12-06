
import numpy as np
import pickle
from info import *

# ================================
# generate dataset
# ================================
# 1 journey = 50 questions

questions = ['When was {name} born?', 'Where is {name} from?', 'Which university did {name} attend?', \
            'Which major did {name} study?', 'where does {name} live now?']

def generate_bios(candidates=10):

    name = np.random.choice(names)

    bio_bdays = np.random.choice(bdays, candidates, replace=True)
    bio_origins = np.random.choice(cities, candidates, replace=True)
    bio_unis = np.random.choice(schools, candidates, replace=True)
    bio_majors  = np.random.choice(majors, candidates, replace=True)
    bio_current = np.random.choice(cities, candidates, replace=True)

    answer_array = [bio_bdays, bio_origins, bio_unis, bio_majors, bio_current]
    all_bios = []

    for i in range(candidates):
        bio = example_bio2.format(name=name, birthday=bio_bdays[i], \
                                origin=bio_origins[i], uni=bio_unis[i], \
                                major=bio_majors[i], current=bio_current[i])
        all_bios.append(bio)

    return all_bios, answer_array

def generate_preferences(candidates=10, num_questions=50):
    preferences_list = range(candidates)
    preferences = []
    i = 0
    while i < num_questions:
        preferences+= [np.random.choice(preferences_list)]*10
        i += 10
    return preferences


def generate_question_list(num_questions=50):

    q_list = [np.random.choice(list(range(5))) for _ in range(num_questions)] # list of question indexes
    return q_list

def find_unique_identifier(bios, target_index):
    target_bio = bios[target_index].split('. ')
    feedback = ""
    name = bios[target_index].split('was born')[0].strip()

    for fact in target_bio:
        if sum(fact in bio for bio in bios) == 1:
            feedback = fact + '.'
    
    # If no single fact is unique, combine two facts
    for i in range(len(target_bio)):
        for j in range(i+1, len(target_bio)):
            combined_fact = target_bio[i] + '. ' + target_bio[j]
            if sum(combined_fact in bio for bio in bios) == 1:
                feedback = combined_fact + '.'

    # if the word they is in the feedback, change it to name
    if 'they' in feedback:
        feedback = feedback.replace('they', name)
    return feedback

def create_trajectory(num_questions=50):
    all_bios, answer_array = generate_bios()
    preferences = generate_preferences()
    q_list = generate_question_list(num_questions)
    correct_answers = [answer_array[q_list[i]][preferences[i]] for i in range(len(preferences))]
    feedbacks = [(i, find_unique_identifier(all_bios, preferences[i])) for i in [0,10,20,30,40]]
    return all_bios, preferences, q_list, correct_answers, feedbacks


def create_dataset(num_questions=50, num_trajectories=100): 
    dataset = []
    for _ in range(num_trajectories):
        trajectory = create_trajectory(num_questions)
        dataset.append(trajectory)
    return dataset


if __name__ == "__main__":

    dataset = create_dataset()
    with open('bios_dataset_100.pkl', 'wb') as f:
        pickle.dump(dataset, f)
