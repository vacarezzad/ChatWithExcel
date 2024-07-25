


import streamlit as st

from functions import ChatManager, DataSystem

# ================================================================
# Main Function
# ================================================================

def main():
    """Main function to run the Streamlit app."""

    # Set up Streamlit page configuration
    st.set_page_config(page_title="Data Analysis Chat", page_icon="📊")
    st.title("📊 Data Analysis Chat")

    # File uploader widget for CSV or Excel files
    uploaded_files = st.sidebar.file_uploader(
        label="Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload your CSV or Excel file to continue")
        st.stop()

    # Initialize chat history
    ChatManager.initialize_chat_history()

    # Button to clear chat history
    if st.button("Clear Chat"):
        st.session_state['chat_history'] = []

    # Create DataSystem instance
    data_system = DataSystem(uploaded_files)

    # Display DataFrame column data types (transposed, with index hidden)
    st.subheader("Data Types")
    if not data_system.df.empty:
        data_types_transposed = data_system.data_types_df.set_index("Column").T  # Transpose the DataFrame
        st.markdown(data_types_transposed.to_html(index=False, header=True), unsafe_allow_html=True)
    else:
        st.write("Data not available")

    # Display a sample of the data
    st.subheader("Data Sample")
    if not data_system.df.empty:
        st.dataframe(data_system.sample_df)
    else:
        st.write("Data not available")

    # Input field for user query
    user_query = st.chat_input(placeholder="Ask me anything about the data")

    if user_query:
        # Store user message and get response
        ChatManager.add_message_to_history("user", user_query)
        response = data_system.process_user_query(user_query)
        ChatManager.add_message_to_history("assistant", response)
    
    # Display chat history
    ChatManager.display_chat_history()

# ================================================================
# Run the main function
# ================================================================

if __name__ == "__main__":
    main()
