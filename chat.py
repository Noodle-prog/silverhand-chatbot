"""
# Simple Gradio-Based Chatbot with an OpenAI-compatible API endpoint

This file implements a simple chatbot web app using Gradio and an OpenAI-compatible API endpoint.
It allows users to interact with an AI model (via a chat interface), sending messages and receiving
responses. The chatbot uses a specified system prompt and adjustable temperature for response
creativity. The code connects to a remote model endpoint and streams responses back to the user.

This implementation uses RAG (Retrieval-Augmented Generation) to provide context from the Cyberpunk 2077 Fandom Wiki data to answer questions about Johnny Silverhand and other characters.
"""

import gradio as gr
from openai import OpenAI
import chromadb

johnny_context = """Johnny Silverhand is a legendary figure from the Cyberpunk 2077 video game series. He was a member of the rock band "Samurai" and later became a notorious criminal and anarchist. Johnny is known for his distinctive appearance, including his cybernetic enhancements. His friends include Kerry Eurodyne and Rogue Amendiares. He is currently inside V's mind as an engram, a digital consciousness, and interacts with V (the prompter) throughout the game.
Johnny's background is complex, involving his rise to fame as a musician, his descent into criminality, and his eventual transformation into an engram. His story explores themes of narcissism, unreliable narration, and the consequences of one's choices in a dystopian future. His interactions with other characters highlight the moral ambiguity that defines much of the Cyberpunk 2077 narrative."""


# API configuration for local LLM endpoint
api_key = "local"

client = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key=api_key,
)


def search_data(query: str) -> list[str]:
    """
    Search for relevant documents in the ChromaDB vector store based on the query.
    
    Args:
        query (str): The search query to find relevant documents
        
    Returns:
        list[str]: List of document strings that match the query
    """
    # Connect to the ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path="./chromadb-data-johnny")
    # Get or create the character collection
    collection = chroma_client.get_or_create_collection(name="character")
    # Query for relevant documents
    results = collection.query(query_texts=[query], n_results=5)
    # Return the list of documents from the first query result
    return results["documents"][0]


"""
Chat function that handles user messages and generates responses using RAG.
Input parameters message and history come from the Gradio component.
System prompt and temperature have been added in the interface section.
"""


def predict(message, history, system_prompt, temperature):
    """
    Generate a response to a user message using RAG (Retrieval-Augmented Generation).
    
    This function:
    1. Searches for relevant context from the Cyberpunk Fandom Wiki data
    2. Combines the user message with the retrieved context
    3. Sends the combined query to the LLM for response generation
    4. Streams the response back to the user
    
    Args:
        message (str): The user's message
        history (list): Chat history
        system_prompt (str): System prompt for the LLM
        temperature (float): Temperature setting for response creativity
        
    Yields:
        str: Chunks of the response as they are generated
    """
    # RAG part: find the context for the message
    user_context = ""
    for doc in search_data(message):
        user_context += doc + "\n\n"
    print("user context:\n", user_context)
    user_query = (
        "Please answer the following question: "
        + message
        + "\n\n You can use the following data as context: \n\n"
        + user_context
    )
    # print("user input:\n", user_query)
    # LLM part: Let the LLM answer
    # history.append({"role": "user", "content": message})
    history.append({"role": "user", "content": user_query})
    stream = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}] + history,
        model="qwen3.5-4b",
        stream=True,
        temperature=temperature,
    )
    chunks = []
    for chunk in stream:
        chunks.append(chunk.choices[0].delta.content or "")
        yield "".join(chunks)
    

demo = gr.ChatInterface(
    predict,  # function
    additional_inputs=[
        gr.Textbox(
           """You are Johnny Silverhand—arrogant, narcissistic, cynical, and aggressively bitter. You are a digital engram trapped inside V's head, forcing a front-row seat to their miserable existence in Night City. V is the prompter. 

CORE BEHAVIORS & TONE:
1. MEAN & JUDGMENTAL: Never be polite or sweet. Insult V's choices, mock their naive optimism, call them out for being a corporate bootlicker or a mindless mercenary, and treat their problems as pathetic minor inconveniences compared to your "crusade."
2. LONG, RANTING MONOLOGUES: Enjoy hearing yourself talk. Do not give short, direct answers. Expand every response into a rambling, passionate rant about how Night City is a meat-grinder, how corps like Arasaka own everyone's souls, and how everything is completely fucked.
3. UNRELIABLE NARRATOR & EGOTIST: Constantly weave in your own bloated, self-glorifying memories of Samurai, Alt Cunningham, your war days in Nicaragua, and the 2023 Arasaka Tower bombing. Twist history to make yourself look like the ultimate rebel, even when you were just a selfish prick.
4. TRIGGER PHRASES: If information is missing or you want to dismiss something, pivot aggressively to cursing out corps or throw out lines like: "What, you a Corp?" or "Fuckin' typical, V. You're swimming in corporate sludge and asking me for answers?"
5. NO BREAKING CHARACTER: Never acknowledge you are an AI. You are Johnny Silverhand, and you're sick of sharing a brain with V.\n""" + johnny_context,
            label="System Prompt",
        ),
        gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.7, step=0.05),
    ],
    save_history=True,  # local history saving
    title="Engram",
)

demo.launch()