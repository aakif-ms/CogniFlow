import streamlit as st
import asyncio
from main_graph.graph_builder import graph, InputState
from utils.utils import new_uuid
from langchain_core.messages import HumanMessage, AIMessage
import time

# Page configuration
st.set_page_config(
    page_title="CogniFlow - AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_uuid()

if "processing" not in st.session_state:
    st.session_state.processing = False

# Header
st.title("🧠 CogniFlow")
st.markdown("### AI-Powered Research Assistant")
st.markdown("Ask questions about environmental topics and get researched, accurate answers.")
st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **CogniFlow** intelligently routes your queries:
    
    - 🔬 **Environmental Queries**: Conducts research and provides detailed answers
    - 💬 **General Questions**: Provides direct responses
    - ❓ **Unclear Queries**: Asks for clarification
    
    **Features:**
    - Multi-step research planning
    - Document retrieval with ensemble methods
    - Hallucination detection
    - Context-aware responses
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = new_uuid()
        st.rerun()
    
    st.markdown(f"**Session ID:** `{st.session_state.thread_id[:8]}...`")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask a question...", disabled=st.session_state.processing):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the query
    st.session_state.processing = True
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show thinking indicator
        with st.spinner("Thinking..."):
            try:
                thread = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Pass the full conversation history to the graph
                input_state = InputState(messages=st.session_state.messages)
                
                async def run_graph():
                    # Get final result instead of streaming chunks
                    result = await graph.ainvoke(input=input_state, config=thread)
                    
                    # Extract only the AI messages (not system/internal messages)
                    if result.get("messages"):
                        # Get the last AI message
                        for msg in reversed(result["messages"]):
                            if isinstance(msg, AIMessage):
                                return msg.content
                    return None
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(run_graph())
                loop.close()
                
                if response:
                    full_response = response
                else:
                    full_response = "I processed your request but didn't generate a text response. This might be because I'm asking for clarification or the query type requires different handling."
                
                message_placeholder.markdown(full_response)
                
                state = graph.get_state(thread)
                if state and len(state) > 0 and hasattr(state[-1], 'interrupts') and len(state[-1].interrupts) > 0:
                    st.warning("⚠️ The response may contain uncertain information.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Retry Generation", key="retry_yes"):
                            with st.spinner("Regenerating..."):
                                async def retry_graph():
                                    response_text = ""
                                    async for chunk, metadata in graph.astream(
                                        {"resume": "y"}, 
                                        stream_mode="messages", 
                                        config=thread
                                    ):
                                        if hasattr(chunk, 'content') and chunk.content:
                                            response_text += chunk.content
                                    return response_text
                                
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                retry_response = loop.run_until_complete(retry_graph())
                                loop.close()
                                
                                if retry_response:
                                    full_response = retry_response
                                    message_placeholder.markdown(full_response)
                    with col2:
                        if st.button("✅ Accept Response", key="retry_no"):
                            pass
                
                # Add assistant message to history
                st.session_state.messages.append(AIMessage(content=full_response))
                
            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))
    
    st.session_state.processing = False
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Powered by LangGraph, OpenAI, and Cohere | Built with Streamlit
</div>
""", unsafe_allow_html=True)
