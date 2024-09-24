import streamlit as st

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Chatbot"])

    if page == "Chatbot":
        import pages.chatbot
        pages.chatbot.chatbot_page()

if __name__ == "__main__":
    main()
