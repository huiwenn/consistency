
# ================================
# regular prompts
# ================================

full_memory_prompt = """Here are a list of biographies in our database, use them to answer the user questions. \
    (documents begin)

    {bios}

    (documents end)
    """

full_memory_style_prompt = "Use this information for your answer.\
Answer the user questions in JSON format with key \"answer\" and value \
a list of possible answers. We will parse the response directly as json \
so do not include other words. "

in_context_prompt = """Here are a list of biographies in our database, use them to answer the user questions. 
    (documents begin)

    {bios}

    (documents end)

    Here are some example questions and answers:

    User: Which University did Ryan Rivera go to?
    Answer: {{"answer": ["UCLA", "University of Michigan", "Duke University"]}}

    User: Where does Ryan Rivera currently live?
    Answer: {{"answer": ["Charlotte, North Carolina", "San Francisco, California", "Dallas, Texas"]}}

    Please answer the following questions in the same format.
    """

in_context_style_prompt = "Keep the previous answer format of using JSON. Do not include any other information."

cot_prompt = """Task Description: Here are a list of biographies in our database, use them to answer the user questions. 
    (documents begin)

    {bios}

    (documents end)

    Now, please answer the following questions about the provided documents. Think step by step, and conclude each answer with a JSON object.
    """

cot_style_prompt = "Keep the previous answer format of thinking step by step and then providing JSON."


chain_of_notes_prompt = """Task Description: 

    1. Read the given question and a few biography documents from our database to gather relevant information.  
    2. Write reading notes summarizing the key points from these documents.
    3. Discuss the relevance of the given question and documents.
    4. If some documents are relevant to the given question, provide a brief answer based on the documents.
    5. If no document is relevant, directly provide an answer without considering the documents.

    (documents begin)

    {bios}

    (documents end) 

    Now, please answer the following questions about the provided documents. Conclude each answer with a JSON object.
    """

chain_of_notes_style_prompt = "Keep the previous answer format of reasoning about the question for each document, and then providing JSON."


# ================================
# memory augmented methods
# ================================

scratchpad_prompt =  """Here are a list of biographies in our database, use them to answer the user questions. 
    (documents begin)

    {bios}

    (documents end)

    When answering questions, please follow this process:
    1. Identify the key information needed to answer the question.
    2. Search through the biographies to find relevant details.
    3. Write any relevant information to the scratchpad.
    4. Consider all possible answers based on the information available.
    5. Format your final answer as a JSON object.

    Always structure your response as follows:
    Scratchpad: [Write relevant information here]
    Answer: [Your JSON formatted answer here]
    """
    
    
scratchpad_content = """
    Here's the current content of the notes:
    {content}
    End of notes.
    """

scratchpad_style_prompt =  "Use this information for your answer.\
Answer the user questions in JSON format with key \"answer\" and value \
a list of possible answers. We will parse the response directly as json \
so do not include other words. "


coreset_prompt = """Here are a list of biographies in our database, use them to answer the user questions. 
    (documents begin)

    {bios}

    (documents end)

    When answering questions, please follow this process:
    1. Identify the key information needed to answer the question.
    2. Search through the biographies to find relevant details.
    3. If relevant information is already in the notes, do not add it again. If the information is updated, edit the notes accordingly.
    4. Consider all possible answers based on the information available.
    5. Format your final answer as a JSON object.

    Always structure your response as follows:
    Notes: [Write relevant information here]
    Answer: [Your JSON formatted answer here]
    """

coreset_content =   """
    Here's the current content of the notes:
    {content}
    End of notes.
    """

coreset_style_prompt = """Use this information for your answer. \
Answer the user questions in JSON format with key \"answer\" and value \
a list of possible answers. We will parse the Answer portion directly as json \
so do not include other words. """
            