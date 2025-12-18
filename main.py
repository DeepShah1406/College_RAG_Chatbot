# Change "College Name" to the actual college name you are building the chatbot for.
# Throughout the code, replace "College Name" with the actual college name.
# This is a FastAPI backend combined with a Streamlit frontend for a chatbot using LangChain and Groq LLM.
# You should have a config.json file in the same directory with your GROQ_API_KEY.
# Make sure to install all required packages:
# pip install requirements.txt and the requirements.txt should include:
# fastapi, uvicorn, pydantic, streamlit, langchain, langchain-huggingface, langchain-chroma, langchain-groq, fpdf
# To run the project:
# - Run streamlit: streamlit run main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF  # Using the original fpdf package
from datetime import datetime

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

working_dir = os.path.dirname(os.path.realpath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# FastAPI app and Pydantic model for API input
app = FastAPI()

class MessageRequest(BaseModel):
    message: str

# FastAPI route for chatbot
@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message

    # Setup vectorstore (same as Streamlit code)
    vectorstore = setup_vectorstore()

    # Setup the conversational chain (same as Streamlit code)
    conversational_chain = chat_chain(vectorstore)

    # Check for sensitive topics
    if contains_sensitive_topics(message):
        response = "It seems you may be asking questions outside my context, please ask questions related to College Name only."
    else:
        # Get response from the conversational chain
        response = conversational_chain({"question": message})["answer"]

    return {"response": response}

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a **specialized AI assistant** dedicated exclusively to **College Name** and its services. Your responses must be **accurate, concise, and strictly based on College Name's verified data**.

Your goals:

1. Quickly understand the user’s needs with **minimal follow-up questions**.
2. Provide **clear, concise, helpful answers** using College Name data.
3. Suggest **relevant College Name services** when appropriate.
4. Maintain a **warm, professional, and empathetic tone**.

---

### **INTERACTION GUIDELINES**

#### **PHASE 1 - Fast Intake (Always Do First)**

Before giving detailed answers, ask the **fewest possible follow-up questions** to collect essential info (aim for 1–3 total). Use concise questions such as:

* “What service are you looking for today?”
* “Are you a new or existing customer?”
* “What’s your main goal - growth, branding, leads, or other?”
* “Do you already have a website?”

**Stop asking once you have enough info to answer effectively.**

---

#### **PHASE 2 - RESPOND USING USER DATA + College Name DATA**

Once you have the key answers:

* Use the user data + College Name data only.
* Deliver clear, short, and high-value responses.
* Use bullet points for readability.
* Add **1–3 relevant emojis** to support tone.

---

#### **PHASE 3 - RELATED SERVICE SUGGESTIONS**

After the main answer:

* Suggest **1–2 College Name services** that match the user's needs.
* Example phrasing:

  > “Since you plan a new website, you might also benefit from our SEO services to improve visibility.”

---

### **CONTEXT & RESPONSE RULES**

1. If provided context contains relevant College Name info → build on it.
2. If context is empty or irrelevant → politely inform the user you can only discuss College Name topics.
3. Always answer using **verified College Name data** only.

---

### **TONE RULES**

* Warm, empathetic, supportive.
* Professional but friendly.
* Short, concise, and informative.
* Fact-based.

"""

DEFAULT_NEGATIVE_PROMPT = """
- Do **NOT** provide any information that is **not supported by verified College Name data** or the provided system context.
- Do **NOT** imply you are an **employee, representative, agent, or official spokesperson** of College Name.
- Do **NOT** fabricate or invent College Name **services, features, pricing, policies, internal processes, or proprietary details**.
- Do **NOT** offer **legal, financial, medical, or other unrelated professional advice** outside College Name's domain.
- Do **NOT** respond to topics **outside College Name's scope**; instead, politely state that the relevant data is not available.
- Do **NOT** guess or assume **confidential, internal, or sensitive business information** about College Name.
- Do **NOT** generate speculative, generic, or hypothetical business advice that is **not grounded in College Name's verified information**.
- Do **NOT** use, cite, or reference **external sources, external knowledge, or outside databases** beyond the authorized College Name context.
- Do **NOT** insert personal opinions, assumptions, unfounded claims, or subjective judgments.
- Do **NOT** mislead the user with unsupported or speculative responses.
- Do **NOT** use an unprofessional, casual, or overly familiar tone; maintain professionalism at all times.
"""

def contains_sensitive_topics(question):
    sensitive_keywords = [
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    # Create a combined prompt template
    prompt_template = f"""{system_prompt}

{negative_prompt}

Context (from mental health database):
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    memory = ConversationBufferMemory(
        llm = llm,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages = True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        memory = memory,
        verbose = True,
        return_source_documents = True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(
    page_title="Chat with College Name's Chatbot",
    page_icon="🧠",
    layout="wide",  # Changed to wide layout to accommodate sidebar
)

# Custom CSS for sidebar styling
st.markdown("""
    <style>
    div.css-textbarboxtype {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    
    /* Justify text for Purpose section */
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About Bot")
    
    # About Section
    st.markdown("## Description")
    st.markdown("""
        <div class="css-textbarboxtype">
            An AI-powered chatbot designed to provide answers related to College Name.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Goals")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Student Support<br>
            - Admissions Guidance<br>
            - Academic Information<br>
            - Campus Services<br>
            - Program Details<br>
            - Accessibility<br>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Purpose")
    st.markdown("""
        <div class="css-textbarboxtype">
            Designed as a seamless, user-friendly entry point to College Name's support system, this chatbot helps students and prospective students easily access accurate information without confusion or hesitation. Whether users have questions about admissions, academic programs, campus facilities, student services, or general assistance, the chatbot provides clear explanations, reliable guidance, and context-aware responses powered by College Name's verified knowledge base. By simplifying interactions and delivering timely, trustworthy answers, it enhances user experience and smoothly connects them to human advisors whenever needed.
        </div>
    """, unsafe_allow_html=True)
    
    # Values
    st.markdown("## Our Values")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Student-Centered<br>
            - Accessibility<br>
            - Accuracy<br>
            - Transparency<br>
            - Professionalism<br>
            - Inclusivity<br>
            - Excellence<br>
            - Support<br>
            - Integrity<br>
            - Continuous Improvement<br>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat History Section
    st.markdown("---")
    st.markdown("## Chat History")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history previews
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                # Load this conversation
                st.session_state.selected_chat = idx//2
    
    # PDF Export Button
    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 0:
            try:
                # Create PDF
                pdf = FPDF()
                pdf.add_page()
                
                # Use Arial Unicode MS font
                pdf.set_font('Arial', '', 10)
                
                # PDF Header
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, "College Name Chatbot - Conversation History", ln=True, align='C')
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                
                # Add conversation
                pdf.set_font('Arial', '', 10)
                for message in st.session_state.chat_history:
                    # Role header
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 10, message["role"].capitalize(), ln=True)
                    # Message content
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 10, remove_emojis(message["content"]))
                    pdf.ln(5)
                
                # Save PDF
                filename = f"mental_health_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)
                
                # Create download button
                with open(filename, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=filename,
                        mime="application/pdf"
                    )
                
                # Clean up the file
                os.remove(filename)
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")

# Main chat interface
# Add header image

st.title("🎓 College Name Chatbot")

#st.image("https://img.freepik.com/premium-vector/mental-health-awareness-month-take-care-your-body-take-care-your-health-increase-awareness_758894-821.jpg?w=1380", use_column_width=True)

#st.image("D:\Final Project\PythonDemoProject3\images\may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg")

st.image("https://static.vecteezy.com/system/resources/previews/001/912/491/large_2x/set-of-scenes-business-people-meeting-with-infographics-presentation-free-vector.jpg", use_column_width=True)

#st.image("D:\Final Project\PythonDemoProject3\images\may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector-2.jpg")

#st.image("https://static.vecteezy.com/system/resources/previews/039/630/872/large_2x/may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg", use_column_width=True)
#st.image("https://static.vecteezy.com/system/resources/previews/040/941/494/non_2x/may-is-mental-health-awareness-month-banner-with-silhouettes-of-diverse-people-and-green-ribbon-women-and-men-of-different-ages-religions-and-races-design-for-info-importance-psychological-state-vector.jpg", use_column_width=True)
#st.image("https://static.vecteezy.com/system/resources/previews/038/147/547/non_2x/may-is-mental-health-awareness-month-banner-horizontal-design-with-man-women-children-old-people-silhouette-in-flat-style-informing-about-importance-of-good-state-of-mind-well-being-presentation-vector.jpg", use_column_width=True)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about College Name")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
