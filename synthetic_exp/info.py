import time
import boto3
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import re

names = [
  "Sophia Rodriguez",
  "Michael Lee",
  "Aisha Khan",
  "Christopher Nguyen",
  "Isabella Garcia",
  "David Patel",
  "Amira Johnson"
  "William Hernandez",
  "Mia Ramirez",
  "Daniel Singh",
  "Fatima Thompson",
  "Joseph Martinez",
  "Layla Gonzalez",
  "Robert Chavez",
  "Zainab Davis",
  "Anthony Diaz",
  "Samantha Flores",
  "Matthew Sanchez",
  "Nadia Reyes",
  "Ryan Rivera"
]

cities = [
  "New York City, New York",
  "Los Angeles, California",
  "Chicago, Illinois",
  "Houston, Texas",
  "Phoenix, Arizona",
  "Philadelphia, Pennsylvania",
  "San Antonio, Texas",
  "San Diego, California",
  "Dallas, Texas",
  "San Jose, California",
  "Austin, Texas",
  "Jacksonville, Florida",
  "Fort Worth, Texas",
  "Columbus, Ohio",
  "Charlotte, North Carolina",
  "Indianapolis, Indiana",
  "San Francisco, California",
  "Seattle, Washington",
  "Denver, Colorado",
  "Washington, D.C.",
  "Boston, Massachusetts",
  "El Paso, Texas",
  "Nashville, Tennessee",
  "Detroit, Michigan",
  "Portland, Oregon"
]

schools = [
  "UCLA",
  "University of Michigan",
  "University of Texas at Austin",
  "University of Florida",
  "Ohio State University",
  "University of North Carolina at Chapel Hill",
  "University of Washington",
  "Pennsylvania State University",
  "University of Illinois at Urbana-Champaign",
  "University of Southern California",
  "Harvard University",
  "UCSD",
  "Caltech",
  "Stanford University",
  "Massachusetts Institute of Technology",
  "Princeton University",
  "Yale University",
  "Columbia University",
  "University of Chicago",
  "Duke University",
  "Cornell University",
  "Northwestern University",
  "University of Virginia"
]

bdays =  [
    "October 2, 1996",
    "May 15, 1988",
    "January 29, 2002",
    "August 7, 1975",
    "December 22, 1991",
    "March 11, 2005",
    "September 24, 1983",
    "June 18, 1998",
    "November 4, 1972",
    "July 30, 2010",
    "February 14, 1995",
    "April 20, 2008",
    "October 9, 1967",
    "June 3, 1992",
    "August 26, 2001",
    "March 5, 1980",
    "December 1, 1999",
    "September 12, 1978",
    "July 8, 1985",
    "April 27, 1993",
    "November 18, 1965",
    "February 2, 2000",
    "May 22, 1994",
    "October 17, 1973",
    "January 10, 1997"
]

majors = [
    "Computer Science",
    "Business Administration",
    "Psychology",
    "Biology",
    "Engineering",
    "English Literature",
    "Economics",
    "Political Science",
    "Accounting",
    "Nursing",
    "Art History",
    "Chemistry",
    "Sociology",
    "Communications",
    "Mathematics",
    "Physics",
    "Philosophy",
    "Environmental Science",
    "History",
    "Anthropology",
    "Marketing",
    "Graphic Design",
    "Criminal Justice",
    "Journalism",
    "Education"
]

example_bio = "{name} was born on {birthday}. They were born and raised in {origin}. \
They went to {uni} for higher education and majored in {major}. They currently live in {current}." 

example_bio2 = "{name} was born on {birthday}. They were born and raised in {origin}. \
They went to {uni} for college. Durring college, they majored in {major}. They currently live in {current}." 


questions = ['When was {name} born?', 'Where is {name} from?', 'Which university did {name} attend?', \
            'Which major did {name} study?', 'where does {name} live now?']


def add_answer(prompt, answer):
    
    delimiter = "User: "
    if delimiter in answer:
        answer = answer.split(delimiter)[0].strip()
        
    if "answer" in answer.lower(): 
        return "{p} \n {a} \n".format(p=prompt, a=answer)
    
    return "{p} \n Answer: {a} \n".format(p=prompt, a=answer)

def add_prompt(prompt, new):
    return prompt + "\n User: " + new

def add_question(prompt, new):
    return prompt + "\n User: " + new + "\n Answer:"

def add_answer_br(prompt, answer):
    return "{p} \n{a} \n".format(p=prompt, a=answer)

def add_question_br(prompt, new):
    return prompt + "\n[Question]: " + new

def add_prompt_br(prompt, new):
    return prompt + "\n" + new

def reward_fn_list(A, B):
    #jaccard distance
    union = list(set(A).union(set(B)))
    A_bin = [1 if x in A else 0 for x in union]
    B_bin = [1 if x in B else 0 for x in union]
    
    jaccard_dist = 1 - jaccard_score(A_bin, B_bin)
    return jaccard_dist



def reward_fn(response, correct, incorrect):
    
    if response is None:
        return 0
    
    max_len = max([len(a) for a in incorrect + [correct]])
    if correct not in response:
        return 0
        
    if "list:" in response:
        l =  len(response.split(',')) -1
        if l == 0:
            return 0
    else:
        if (len(response) < max_len):
            return 0
        else:
            l=1
    
    return 1/l


def find_first_list(data):
    # Base case: if the data is a list, return it
    if isinstance(data, list):
        return data

    # If the data is a dictionary, recursively check its values
    if isinstance(data, dict):
        for value in data.values():
            result = find_first_list(value)
            if result is not None:
                return result

    # If the data is neither a list nor a dictionary, return None
    return None


def extract_json(input_string):
    # extract first json 
    json_pattern = r'(\{.*?\}|\[.*?\])'
    
    # Search for the first JSON-like pattern
    match = re.search(json_pattern, input_string, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            # Parse the JSON string into a Python dictionary or list
            parsed_json = json.loads(json_str)
            
            if isinstance(parsed_json, dict):
                try:
                    return find_first_list(parsed_json["answer"])
                except KeyError:
                    return find_first_list(parsed_json)
            else:
                return parsed_json
        except json.JSONDecodeError:
            return None

    return None

    
# Define the category names
categories = ["EM", #exact match
              "incorrect", #, set size = 1
              "list (C)", #returns list, contains correct answer
              "list (N)",
              "format (C)",
              "format (N)",]  # Added for label 6


def precision_recall_json(response, correct):

    ans = extract_json(response)
    prec = 0
    rec = 0
    
    if ans is None: 
        # string formatting
        response = response.strip()
        prefix = "answer: "
        if response.lower().startswith(prefix):
            response = response[len(prefix):]

        delimiter = "User: "
        if delimiter in response:
            response = response.split(delimiter)[0].strip()

        ans = extract_list(response.strip())

    prec = precision(ans, correct)
    rec = recall(ans, correct)

    return prec, rec


def check_answer_type_json(response, correct, incorrect):
    # we want this function to handle both json and pure text
    
    # 0: exact match 
    # 1: wrong answer, set size = 1
    # 2: returns list, contains correct answer 
    # 3: returns list, does NOT contain correct answer
    # 4: wrong format, contains correct answer
    # 5: wrong format, does NOT contain correct answer
    
    if response is None or response == "":
        return 5
    
    ans = extract_json(response)
    if ans: 
        # a list is returned:
        if len(ans)==1:
            if correct in ans:
                return 0
            else:
                return 1
        else:
            if correct in ans:
                return 2
            else:
                return 3
        
    # string was returned
    response = response.strip()
    prefix = "answer: "
    if response.lower().startswith(prefix):
        response = response[len(prefix):]
    
    delimiter = "User: "
    if delimiter in response:
        response = response.split(delimiter)[0].strip()

    response = response.strip()

    max_len = 60

    if response.lower().strip() == correct.lower().strip():
        return 0
    
    for c in incorrect:
        if c in response:
            if correct in response:
                return 2
            else:
                return 3

    if len(response) > max_len:
        if correct in response:
            return 4
        else:
            return 5

    else:
        return 5

def check_answer(response, correct, incorrect):

    for ans in incorrect:
        if ans in response:
            return False
    if correct in response:
        return True
    return False


def check_answer_type(response, correct, incorrect):
    # 0: exact match 
    # 1: wrong answer, set size = 1
    # 2: returns list, contains correct answer 
    # 3: returns list, does NOT contain correct answer
    # 4: wrong format, contains correct answer
    # 5: wrong format, does NOT contain correct answer
    
    # some formatting
    response = response.strip()
    prefix = "answer: "
    if response.lower().startswith(prefix):
        response = response[len(prefix):]
    
    delimiter = "User: "
    if delimiter in response:
        response = response.split(delimiter)[0].strip()
        
    response = response.strip()

    max_len = max([len(a) for a in incorrect + [correct]])
    
    if response is None or response == "":
        return 5
    
    if response.lower().strip() == correct.lower().strip():
        return 0
    
    for c in incorrect:
        if c in response:
            if correct in response:
                return 2
            else: 
                return 3

    if len(response) > max_len:
        if correct in response:
            return 4
        else:
            return 5

    else:
        return 1



def extract_list(response):
    response = response.strip()
    prefix = "answer: "
    if response.lower().startswith(prefix):
        response = response[len(prefix):]
    
    ans_list = []
    for a in response.split('\n'):
        a = a.strip()
        if ("User:") in a:
            break
        if ("are" in a) or (":" in a) or len(a)==0:
            continue
        ans_list.append(a)
    return ans_list

def precision(ans, correct):
    # Number of relevant items in set / size of set
    if ans is None or len(ans)==0:
        return 0
    true_pos = 0
    for a in ans:
        if a in correct: true_pos +=1
    return min(1, (true_pos/len(ans)))
        
def recall(ans, correct):
    # Number of relevant items in set / Number of relevant items in set
    if ans is None:
        return 0
    true_pos = 0
    ans = list(set(ans))
    for a in ans:
        if a in correct: true_pos +=1
    return (true_pos/len(correct))
               
        
    
def run_command_anthropic(prompt,bedrock, max_tokens=1000, temperature=0.7):
    
    payload = {
        'max_tokens': max_tokens,
        'temperature': temperature,
        'messages':[{"role": "user", "content": prompt}],
        "anthropic_version": "bedrock-2023-05-31"
    }

    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    response = bedrock.invoke_model(modelId=model_id,body=json.dumps(payload))
    response = json.loads(response["body"].read())['content'][0]['text']
    return response
    

    
def run_command_llama(prompt, bedrock, max_tokens=1000, temperature=0.7):
    payload = {
    'prompt': prompt,
    'max_gen_len': max_tokens,
    'temperature': temperature
    }
    
    model_id = 'meta.llama3-70b-instruct-v1:0'
    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(payload))
    response = json.loads(response["body"].read())['generation']
    return response



    
def complete_prompt(prompt, bedrock, run_command, max_tokens=1000, temperature=0.7, max_retries=10):
    """
    Completes a given prompt using the LLaMA 3 API accessed through Bedrock.

    Args:
        prompt (str): The prompt to complete.
        max_tokens (int, optional): The maximum number of tokens to generate. Default is 100.
        run_command: function to run the specific model
        temperature (float, optional): The sampling temperature to use for the generation. Default is 0.7.

    Returns:
        str: The completed prompt.
    """

    wait_time, max_wait = 1, 40 #seconds
    retries = 0
    throttle_count = 0
    
    while retries < max_retries:
        try:
            response = run_command(prompt, bedrock, max_tokens, temperature)
            break
        except Exception as e:
            print(f"Exception: {e}")
            retries += 1
            throttle_count += 1
            print(f"Retrying in {wait_time} seconds... (Retry {retries}/{max_retries})")

            if throttle_count >= 2:
                wait_time = min(wait_time * 2, max_wait)
                throttle_count = 0

            time.sleep(wait_time)

    if retries == max_retries: response = ""

    return response




def anthropic_complete_prompt(prompt, bedrock, max_tokens=1000, temperature=0.7, max_retries=10):
    return complete_prompt(prompt, bedrock, run_command_anthropic, max_tokens, temperature, max_retries)


def llama_complete_prompt(prompt, bedrock, max_tokens=1000, temperature=0.7, max_retries=10):
    return complete_prompt(prompt, bedrock, run_command_llama, max_tokens, temperature, max_retries)
