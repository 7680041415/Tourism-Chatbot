import streamlit as st


# Streamlit UI
st.title("Tourism Information Chatbot")
with st.sidebar:
    st.header("Get Help with Tourism")
    st.write("Ask questions about different countries and their tourism information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for the chatbot
if user_input := st.chat_input("Ask about tourism information:"):

    # Display user's message
    st.chat_message("user").markdown(user_input)

    # Add to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Use the chain to get the answer from the tourism data
    with st.spinner("Searching for the answer..."):
        response = chain({"question": user_input})["answer"]

    # Display the chatbot's response
    st.chat_message("assistant").markdown(response)

    # Add Like and Dislike buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like", key=f"like_{user_input}"):
            st.session_state.feedback = "Glad to hear you found this helpful!"
    with col2:
        if st.button("👎 Dislike", key=f"dislike_{user_input}"):
            st.session_state.feedback = "Sorry this didn’t meet your expectations. For more information, please visit [Wikipedia](https://en.wikipedia.org/)."

# Display feedback message if it exists
if "feedback" in st.session_state:
    st.write(st.session_state.feedback)

print(documents)
