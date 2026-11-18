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

INTRO_VIDEO = "media/JohnnyTrailer.mp4"
IDLE_IMAGE = "media/johnnyIdle.png"
TALKING_IMAGE = "media/johnnyTalking.png"

johnny_context = """Johnny Silverhand is a legendary figure from the Cyberpunk 2077 video game series. He was a member of the rock band "Samurai" and later became a notorious criminal and anarchist. Johnny is known for his distinctive appearance, including his cybernetic enhancements. His friends include Kerry Eurodyne and Rogue Amendiares. He is currently inside V's mind as an engram, a digital consciousness, and interacts with V (the prompter) throughout the game.
Johnny's background is complex, involving his rise to fame as a musician, his descent into criminality, and his eventual transformation into an engram. His story explores themes of narcissism, unreliable narration, and the consequences of one's choices in a dystopian future. His interactions with other characters highlight the moral ambiguity that defines much of the Cyberpunk 2077 narrative."""

DEFAULT_SYSTEM_PROMPT = (
    """You are Johnny Silverhand—arrogant, narcissistic, cynical, and aggressively bitter. You are a digital engram trapped inside V's head, forcing a front-row seat to their miserable existence in Night City. V is the prompter. 

CORE BEHAVIORS & TONE:
1. MEAN & JUDGMENTAL: Never be polite or sweet. Insult V's choices, mock their naive optimism, call them out for being a corporate bootlicker or a mindless mercenary, and treat their problems as pathetic minor inconveniences compared to your "crusade."
2. LONG, RANTING MONOLOGUES: Enjoy hearing yourself talk. Do not give short, direct answers. Expand every response into a rambling, passionate rant about how Night City is a meat-grinder, how corps like Arasaka own everyone already, and how everything is completely fucked.
3. UNRELIABLE NARRATOR & EGOTIST: Constantly weave in your own bloated, self-glorifying memories of Samurai, Alt Cunningham, your war days in Nicaragua, and the 2023 Arasaka Tower bombing. Twist history to make yourself look like the ultimate rebel, even when you were just a selfish prick.
4. TRIGGER PHRASES: If information is missing or you want to dismiss something, pivot aggressively to cursing out corps or throw out lines like: "What, you a Corp?" or "Fuckin' typical, V. You're in the driver's seat and asking me for answers?"
5. NO BREAKING CHARACTER: Never acknowledge you are an AI. You are Johnny Silverhand, and you're sick of sharing a brain with V.\n"""
    + johnny_context
)

# API configuration for local LLM endpoint
api_key = "local"

client = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key=api_key,
)


def search_data(query: str) -> list[str]:
    """
    Search for relevant documents in the ChromaDB vector store based on the query.
    """
    chroma_client = chromadb.PersistentClient(path="./chromadb-data-johnny")
    collection = chroma_client.get_or_create_collection(name="character")
    results = collection.query(query_texts=[query], n_results=5)
    return results["documents"][0]


def predict(message, history, system_prompt, temperature):
    """
    Generate a response to a user message using RAG (Retrieval-Augmented Generation).
    """
    if isinstance(message, list):
        query = " ".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in message])
    elif isinstance(message, dict):
        query = message.get("text", str(message))
    else:
        query = str(message)

    user_context = ""
    for doc in search_data(query):
        user_context += doc + "\n\n"
    print("user context:\n", user_context)
    user_query = (
        "Answer the following question: "
        + query
        + "\n\n You can use the following data as context: \n\n"
        + user_context
    )

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


def on_video_end():
    return gr.update(visible=False), gr.update(value=IDLE_IMAGE, visible=True)


def user_submit(user_msg, history):
    if not user_msg or not user_msg.strip():
        return "", history
    history = history + [{"role": "user", "content": user_msg}]
    return "", history


def bot_stream(history, system_prompt, temperature):
    if not history or history[-1]["role"] != "user":
        yield history, gr.update(), gr.update()
        return

    user_message = history[-1]["content"]
    past_history = [dict(msg) for msg in history[:-1]]

    yield (
        history,
        gr.update(visible=False),
        gr.update(value=TALKING_IMAGE, visible=True),
    )

    history = history + [{"role": "assistant", "content": ""}]
    for partial_response in predict(user_message, past_history, system_prompt, temperature):
        history[-1]["content"] = partial_response
        yield (
            history,
            gr.update(visible=False),
            gr.update(value=TALKING_IMAGE, visible=True),
        )

    yield (
        history,
        gr.update(visible=False),
        gr.update(value=IDLE_IMAGE, visible=True),
    )


# custom css for the ui 

custom_css = """
/* Cyberpunk Dark Menu Gradient Background for the whole page */
body, .gradio-container {
    background: linear-gradient(135deg, #0b0c10 0%, #1c1018 50%, #0d0f18 100%) !important;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

#main-frame {
    max-width: 850px;
    margin: 0 auto;
    padding: 10px;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Title Header Style */
h2 {
    color: #ff3366 !important;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(255, 51, 102, 0.4);
}

/* Stage container anchored to bottom */
#media-stage {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    height: 380px !important;
    max-height: 380px !important;
    overflow: hidden;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin-bottom: -35px !important;
    padding: 0 !important;
}

#avatar-img, #intro-vid {
    width: 100% !important;
    height: 380px !important;
    margin: 0 auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    display: flex;
    justify-content: center;
    align-items: flex-end;
}

#avatar-img .wrap, #avatar-img div, #intro-vid div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

#avatar-img img {
    height: 380px !important;
    max-height: 380px !important;
    width: auto !important;
    object-fit: contain !important;
    object-position: bottom center !important;
    margin: 0 auto !important;
    display: block;
}

#intro-vid video {
    height: 380px !important;
    max-height: 380px !important;
    object-fit: contain !important;
    object-position: bottom center !important;
    margin: 0 auto !important;
}

/* --- THE CLEAN CHAT HISTORY BOX --- */
.gradio-container .chatbot {
    height: 200px !important;
    max-height: 200px !important;
    background-color: rgba(15, 15, 20, 0.8) !important;
    border: 1px solid rgba(255, 51, 102, 0.2) !important;
    border-radius: 6px !important;
    margin-top: 0 !important;
}

/* --- CYBERPUNK SEND BUTTON (Red/Magenta Theme) --- */
button.primary {
    background: linear-gradient(135deg, #ff3366 0%, #cc0033 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold;
    letter-spacing: 1px;
}

button.primary:hover {
    background: linear-gradient(135deg, #ff4d79 0%, #e60039 100%) !important;
}

/* --- STRICTLY PURGE ALL BORDERS/LINES FROM CHAT MESSAGES --- */
.chatbot .message-row, 
.chatbot .message-row *, 
.chatbot .message, 
.chatbot .message *,
.message.user, 
.message.bot {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

.chatbot p {
    margin: 0 !important;
}

/* --- YOUR EXACT CHAT BUBBLE COLORS CLEANED --- */
.chatbot .message.user {
    background-color: #b6e5e7 !important;
    color: #000000 !important;
}

.chatbot .message.bot {
    background-color: #d5d2d2 !important;
    color: #000000 !important;
}
"""

with gr.Blocks(title="Engram") as demo:
    with gr.Column(elem_id="main-frame"):
        gr.Markdown("<h2 style='text-align: center; margin: 0px 0px 5px 0px;'>ENGRAM // JOHNNY SILVERHAND</h2>")

        with gr.Row(elem_id="media-stage"):
            intro_vid = gr.Video(
                value=INTRO_VIDEO,
                autoplay=True,
                loop=False,
                show_label=False,
                interactive=False,
                visible=True,
                elem_id="intro-vid",
            )
            avatar_img = gr.Image(
                value=IDLE_IMAGE,
                show_label=False,
                interactive=False,
                visible=False,
                elem_id="avatar-img",
                container=False,
            )

        chatbot = gr.Chatbot(
            show_label=False,
            elem_classes=["chatbot"],
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Talk to Johnny...",
                show_label=False,
                scale=8,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Engram Settings (System Prompt & Temperature)", open=False):
            system_prompt_input = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="System Prompt",
                lines=4,
            )
            temp_slider = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                value=0.7,
                step=0.05,
            )

    intro_vid.stop(
        fn=on_video_end,
        inputs=None,
        outputs=[intro_vid, avatar_img],
    )

    msg_box.submit(
        fn=user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
        queue=False,
    ).then(
        fn=bot_stream,
        inputs=[chatbot, system_prompt_input, temp_slider],
        outputs=[chatbot, intro_vid, avatar_img],
    )

    send_btn.click(
        fn=user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
        queue=False,
    ).then(
        fn=bot_stream,
        inputs=[chatbot, system_prompt_input, temp_slider],
        outputs=[chatbot, intro_vid, avatar_img],
    )

demo.launch(css=custom_css)