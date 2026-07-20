import streamlit as st
from agent import agent

st.title("arXiv Research Analyst")
st.caption("Ask questions about recently indexed AI papers")

# session_state persists data across reruns - without this, chat history
# would reset every time you send a new message
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the full conversation history on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# st.chat_input renders a chat-style text box at the bottom of the page
question = st.chat_input("Ask a question about the indexed papers...")

if question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the LangGraph agent and show the answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking..."):
            result = agent.invoke({"question": question})
            answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})