# import streamlit as st
# import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from groq import Groq
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Set API keys (use environment variables in production)
# os.environ["PINECONE_API_KEY"] = "pcsk_5CsWGm_DTETbjaHK7ZP6P2eQaMNL2JdUTKitPSuGC3Ntx3nwJNjcWLGsjwopHmUrV58r5D"
# os.environ["GROQ_API_KEY"] = "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"

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load existing Pinecone index
# doc_store = PineconeVectorStore.from_existing_index(
#     index_name="bible-help",
#     embedding=embedding_model,
# )

# def get_response(query, doc_store, chat_history=[], num_scriptures=5):
#     # Use the correct method for LangChain's HuggingFaceEmbeddings
#     results = doc_store.similarity_search_with_score(query, k=num_scriptures)
#     scriptures = [doc.page_content for doc, _ in results]
    
#     history_text = "\n".join([f"User: {m['content']}\nAssistant: {m['content']}" 
#                              for m in chat_history[-6:] if m['role'] in ['user', 'assistant']])
    
#     prompt = f"""Previous conversation:
# {history_text}
# Bible Verses on the topic: {query}
# Retrieved Scriptures:
# {chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

# User asked: {query}
# Provide exactly {num_scriptures} Bible verses related to this topic. 
# Format each scripture with its reference (book, chapter, verse) and the text.
# Assistant:"""
    
#     client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
#     completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a helpful Bible assistant. Provide relevant Bible verses on the requested topics. Always include the scripture reference (book, chapter, verse) with each verse."},
#             {"role": "user", "content": prompt}
#         ],
#         model="gemma2-9b-it",
#         max_tokens=1000
#     )
    
#     return completion.choices[0].message.content, scriptures

# # Streamlit UI
# st.title("Bible Verses Explorer")
# st.write("Ask for scriptures on any topic or theme!")

# # Initialize chat history if not already
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Parse the number of verses requested
# def parse_scripture_count(query):
#     # Default to 5 scriptures
#     num_scriptures = 5
    
#     # Check if the query explicitly asks for a specific number
#     if "give me" in query.lower() and "scriptures" in query.lower():
#         parts = query.lower().split("give me")
#         if len(parts) > 1:
#             for word in parts[1].split():
#                 if word.isdigit():
#                     num_scriptures = int(word)
#                     break
    
#     return num_scriptures

# # User input
# user_input = st.chat_input("Ask for scriptures on a topic...")
# if user_input:
#     # Display user message
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
        
#     # Show spinner animation while processing the query
#     with st.spinner("Finding Bible verses for you..."):
#         # Determine how many scriptures to return
#         num_scriptures = parse_scripture_count(user_input)
        
#         # Extract the topic from the query
#         topic = user_input
#         if "scriptures on" in user_input.lower():
#             topic = user_input.lower().split("scriptures on")[1].strip()
#         elif "verses on" in user_input.lower():
#             topic = user_input.lower().split("verses on")[1].strip()
#         elif "about" in user_input.lower():
#             topic = user_input.lower().split("about")[1].strip()
        
#         # Get AI response
#         ai_response, source_scriptures = get_response(topic, doc_store, st.session_state.chat_history, num_scriptures)
        
#         # Display AI response
#         st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
#         with st.chat_message("assistant"):
#             st.markdown(ai_response)
            
#         # Option to show additional verses
#         if st.button("Show me more verses on this topic"):
#             more_verses, _ = get_response(topic, doc_store, st.session_state.chat_history, num_scriptures=5)
#             st.session_state.chat_history.append({"role": "assistant", "content": more_verses})
#             with st.chat_message("assistant"):
#                 st.markdown(more_verses)

# import streamlit as st
# import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from groq import Groq
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Set API keys (use environment variables in production)
# os.environ["PINECONE_API_KEY"] = "pcsk_5CsWGm_DTETbjaHK7ZP6P2eQaMNL2JdUTKitPSuGC3Ntx3nwJNjcWLGsjwopHmUrV58r5D"
# os.environ["GROQ_API_KEY"] = "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"

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load existing Pinecone index
# doc_store = PineconeVectorStore.from_existing_index(
#     index_name="bible-help",
#     embedding=embedding_model,
# )

# def get_scriptures(query, doc_store, chat_history=[], num_scriptures=5):
#     # Use similarity search to find relevant scriptures
#     results = doc_store.similarity_search_with_score(query, k=num_scriptures)
#     scriptures = [doc.page_content for doc, _ in results]
    
#     history_text = "\n".join([f"User: {m['content']}\nAssistant: {m['content']}" 
#                              for m in chat_history[-6:] if m['role'] in ['user', 'assistant']])
    
#     prompt = f"""Previous conversation:
# {history_text}
# Bible Verses on the topic: {query}
# Retrieved Scriptures:
# {chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

# User asked for verses about: {query}
# Provide exactly {num_scriptures} Bible verses related to this topic. 
# Format each scripture with its reference (book, chapter, verse) and the text.
# Assistant:"""
    
#     client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
#     completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a helpful Bible assistant. Provide relevant Bible verses on the requested topics. Always include the scripture reference (book, chapter, verse) with each verse."},
#             {"role": "user", "content": prompt}
#         ],
#         model="gemma2-9b-it",
#         max_tokens=1000
#     )
    
#     return completion.choices[0].message.content, scriptures

# def explain_topic(query, doc_store, chat_history=[]):
#     # Retrieve more context for topic explanation
#     results = doc_store.similarity_search_with_score(query, k=8)
#     scriptures = [doc.page_content for doc, _ in results]
    
#     history_text = "\n".join([f"User: {m['content']}\nAssistant: {m['content']}" 
#                              for m in chat_history[-6:] if m['role'] in ['user', 'assistant']])
    
#     prompt = f"""Previous conversation:
# {history_text}
# Biblical Context on the topic: {query}
# Retrieved Scripture Context:
# {chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

# User asked to learn more about the biblical concept of: {query}
# Based ONLY on the retrieved scripture context above, provide a detailed explanation of this biblical concept.
# Include key aspects, how it's understood in biblical context, and how different passages relate to it.
# Cite specific verses from the retrieved context to support your explanation.
# Assistant:"""
    
#     client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
#     completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a knowledgeable Bible teacher. Provide in-depth explanations of biblical concepts using only the scripture context provided. Cite verses to support your explanations."},
#             {"role": "user", "content": prompt}
#         ],
#         model="gemma2-9b-it",
#         max_tokens=1500
#     )
    
#     return completion.choices[0].message.content

# # Streamlit UI
# st.title("Bible Verses Explorer")
# st.write("Ask for scriptures on any topic or theme!")

# # Initialize session state variables
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
    
# if "current_topic" not in st.session_state:
#     st.session_state.current_topic = ""
    
# if "show_more_clicked" not in st.session_state:
#     st.session_state.show_more_clicked = False
    
# if "explain_topic_clicked" not in st.session_state:
#     st.session_state.explain_topic_clicked = False

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Parse the number of verses requested
# def parse_scripture_count(query):
#     # Default to 5 scriptures
#     num_scriptures = 5
    
#     # Check if the query explicitly asks for a specific number
#     if "give me" in query.lower() and "scriptures" in query.lower():
#         parts = query.lower().split("give me")
#         if len(parts) > 1:
#             for word in parts[1].split():
#                 if word.isdigit():
#                     num_scriptures = int(word)
#                     break
    
#     return num_scriptures

# # Function to handle "Show more verses" button click
# def on_show_more_verses():
#     st.session_state.show_more_clicked = True

# # Function to handle "Explain this topic" button click
# def on_explain_topic():
#     st.session_state.explain_topic_clicked = True

# # User input
# user_input = st.chat_input("Ask for scriptures on a topic...")
# if user_input:
#     # Reset button states when new query is entered
#     st.session_state.show_more_clicked = False
#     st.session_state.explain_topic_clicked = False
    
#     # Display user message
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
        
#     # Show spinner animation while processing the query
#     with st.spinner("Finding Bible verses for you..."):
#         # Determine how many scriptures to return
#         num_scriptures = parse_scripture_count(user_input)
        
#         # Extract the topic from the query
#         topic = user_input
#         if "scriptures on" in user_input.lower():
#             topic = user_input.lower().split("scriptures on")[1].strip()
#         elif "verses on" in user_input.lower():
#             topic = user_input.lower().split("verses on")[1].strip()
#         elif "verses about" in user_input.lower():
#             topic = user_input.lower().split("verses about")[1].strip()
#         elif "about" in user_input.lower():
#             topic = user_input.lower().split("about")[1].strip()
        
#         # Store current topic for buttons
#         st.session_state.current_topic = topic
        
#         # Get AI response
#         ai_response, source_scriptures = get_scriptures(topic, doc_store, st.session_state.chat_history, num_scriptures)
        
#         # Display AI response
#         st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
#         with st.chat_message("assistant"):
#             st.markdown(ai_response)
            
#             # Add buttons for additional actions
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.button("Show more verses on this topic", on_click=on_show_more_verses, key="more_verses")
#             with col2:
#                 st.button(f"Tell me more about {topic}", on_click=on_explain_topic, key="explain_topic")

# # Handle button clicks outside the user input block
# if st.session_state.show_more_clicked:
#     st.session_state.show_more_clicked = False  # Reset the state
#     with st.spinner("Finding more Bible verses..."):
#         more_verses, _ = get_scriptures(st.session_state.current_topic, doc_store, st.session_state.chat_history, num_scriptures=5)
#         st.session_state.chat_history.append({"role": "assistant", "content": more_verses})
#         with st.chat_message("assistant"):
#             st.markdown(more_verses)

# if st.session_state.explain_topic_clicked:
#     st.session_state.explain_topic_clicked = False  # Reset the state
#     with st.spinner(f"Preparing explanation about {st.session_state.current_topic}..."):
#         explanation = explain_topic(st.session_state.current_topic, doc_store, st.session_state.chat_history)
#         st.session_state.chat_history.append({"role": "assistant", "content": explanation})
#         with st.chat_message("assistant"):
#             st.markdown(explanation)

import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set API keys (use environment variables in production)
# IMPORTANT: Don't hardcode API keys in production code
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if API keys are available
if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("API keys missing. Please check your .env file or environment variables.")
    st.stop()

try:
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing Pinecone index
    doc_store = PineconeVectorStore.from_existing_index(
        index_name="bible-help",
        embedding=embedding_model,
    )
except Exception as e:
    st.error(f"Error initializing services: {str(e)}")
    st.stop()

def get_scriptures(query, doc_store, chat_history=[], num_scriptures=5):
    try:
        # Use similarity search to find relevant scriptures
        results = doc_store.similarity_search_with_score(query, k=num_scriptures)
        scriptures = [doc.page_content for doc, _ in results]
        
        # Format previous conversation for context
        history_text = "\n".join([f"User: {m['content']}\nAssistant: {m['content']}" 
                                for m in chat_history[-6:] if m['role'] in ['user', 'assistant']])
        
        prompt = f"""Previous conversation:
{history_text}
Bible Verses on the topic: {query}
Retrieved Scriptures:
{chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

User asked for verses about: {query}
Provide exactly {num_scriptures} Bible verses related to this topic. 
Format each scripture with its reference (book, chapter, verse) and the text.
Assistant:"""
        
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful Bible assistant. Provide relevant Bible verses on the requested topics. Always include the scripture reference (book, chapter, verse) with each verse."},
                {"role": "user", "content": prompt}
            ],
            model="gemma2-9b-it",
            max_tokens=1000
        )
        
        return completion.choices[0].message.content, scriptures
    except Exception as e:
        return f"Sorry, I encountered an error while fetching scriptures: {str(e)}", []

def explain_topic(query, doc_store, chat_history=[]):
    try:
        # Retrieve more context for topic explanation
        results = doc_store.similarity_search_with_score(query, k=8)
        scriptures = [doc.page_content for doc, _ in results]
        
        history_text = "\n".join([f"User: {m['content']}\nAssistant: {m['content']}" 
                                for m in chat_history[-6:] if m['role'] in ['user', 'assistant']])
        
        prompt = f"""Previous conversation:
{history_text}
Biblical Context on the topic: {query}
Retrieved Scripture Context:
{chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

User asked to learn more about the biblical concept of: {query}
Based ONLY on the retrieved scripture context above, provide a detailed explanation of this biblical concept.
Include key aspects, how it's understood in biblical context, and how different passages relate to it.
Cite specific verses from the retrieved context to support your explanation.
Assistant:"""
        
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable Bible teacher. Provide in-depth explanations of biblical concepts using only the scripture context provided. Cite verses to support your explanations."},
                {"role": "user", "content": prompt}
            ],
            model="gemma2-9b-it",
            max_tokens=1500
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error while explaining this topic: {str(e)}"

# Streamlit UI
st.title("Bible Verses Explorer")

st.write("Ask for scriptures on any topic or Question you might Have")
st.title("Remember Jeremiah 1:12 says  for I (God) am watching to see that my word is fulfilled.")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
    
if "show_more_clicked" not in st.session_state:
    st.session_state.show_more_clicked = False
    
if "explain_topic_clicked" not in st.session_state:
    st.session_state.explain_topic_clicked = False

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Parse the number of verses requested
def parse_scripture_count(query):
    # Default to 5 scriptures
    num_scriptures = 5
    
    # Check if the query explicitly asks for a specific number
    if "give me" in query.lower() and "scriptures" in query.lower():
        parts = query.lower().split("give me")
        if len(parts) > 1:
            for word in parts[1].split():
                if word.isdigit():
                    num_scriptures = int(word)
                    break
    
    return min(max(1, num_scriptures), 10)  # Ensure between 1 and 10 verses

# Function to handle "Show more verses" button click
def on_show_more_verses():
    st.session_state.show_more_clicked = True

# Function to handle "Explain this topic" button click
def on_explain_topic():
    st.session_state.explain_topic_clicked = True

# User input
user_input = st.chat_input("Ask for scriptures on a topic...")

# Process user input
if user_input:
    # Reset button states when new query is entered
    st.session_state.show_more_clicked = False
    st.session_state.explain_topic_clicked = False
    
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Show spinner animation while processing the query
    with st.spinner("Finding Bible verses for you..."):
        try:
            # Determine how many scriptures to return
            num_scriptures = parse_scripture_count(user_input)
            
            # Extract the topic from the query
            topic = user_input
            for phrase in ["scriptures on", "verses on", "verses about", "about"]:
                if phrase in user_input.lower():
                    topic = user_input.lower().split(phrase, 1)[1].strip()
                    break
            
            # Store current topic for buttons
            st.session_state.current_topic = topic
            
            # Get AI response
            ai_response, source_scriptures = get_scriptures(topic, doc_store, st.session_state.chat_history, num_scriptures)
            
            # Display AI response
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                
                # Only show buttons if we successfully retrieved scriptures
                if source_scriptures:
                    # Add buttons for additional actions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Show more verses on this topic", on_click=on_show_more_verses, key="more_verses")
                    with col2:
                        st.button(f"Tell me more about {topic}", on_click=on_explain_topic, key="explain_topic")
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"I'm sorry, something went wrong: {str(e)}")

# Handle button clicks outside the user input block
if st.session_state.show_more_clicked:
    st.session_state.show_more_clicked = False  # Reset the state
    with st.spinner("Finding more Bible verses..."):
        more_verses, _ = get_scriptures(st.session_state.current_topic, doc_store, st.session_state.chat_history, num_scriptures=5)
        st.session_state.chat_history.append({"role": "assistant", "content": more_verses})
        with st.chat_message("assistant"):
            st.markdown(more_verses)

if st.session_state.explain_topic_clicked:
    st.session_state.explain_topic_clicked = False  # Reset the state
    with st.spinner(f"Preparing explanation about {st.session_state.current_topic}..."):
        explanation = explain_topic(st.session_state.current_topic, doc_store, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": explanation})
        with st.chat_message("assistant"):
            st.markdown(explanation)
