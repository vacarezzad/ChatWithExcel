import os
import pathlib
import re
import tempfile
from typing import Any, List

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ================================================================
# Custom Exceptions and DataLoader
# ================================================================

class DataLoaderException(Exception):
    """Custom exception for errors related to data loading."""
    pass

class DataLoader:
    """Class for loading data from CSV and Excel files."""

    @staticmethod
    def load_data(temp_filepath: str) -> pd.DataFrame:
        """Load a CSV or Excel file and return it as a DataFrame."""
        ext = pathlib.Path(temp_filepath).suffix
        if ext == ".csv":
            return pd.read_csv(temp_filepath)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(temp_filepath)
        else:
            raise DataLoaderException(f"Invalid file extension {ext}, cannot load this type of file")

# ================================================================
# Data Analyzer
# ================================================================

class DataAnalyzer:
    """Class for performing data analysis on DataFrames."""

    @staticmethod
    def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns in the DataFrame to numeric types if possible; otherwise, keep as object."""
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype('object')
        return df

    @staticmethod
    def infer_data_type(series: pd.Series) -> str:
        """Infer the data type of a pandas Series."""
        if pd.api.types.is_integer_dtype(series):
            return 'int'
        elif pd.api.types.is_float_dtype(series):
            return 'float'
        elif pd.api.types.is_object_dtype(series):
            return 'object'
        else:
            return 'unknown'

    @staticmethod
    def summarize_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Generate a summary of data types for the DataFrame."""
        if df.empty:
            return pd.DataFrame(columns=["Column", "Type"])
        
        df_with_numeric = DataAnalyzer.convert_to_numeric(df)
        data_types = {
            "Column": df_with_numeric.columns,
            "Type": [DataAnalyzer.infer_data_type(df_with_numeric[col]) for col in df_with_numeric.columns]
        }
        return pd.DataFrame(data_types)

    @staticmethod
    def get_sample(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """Get a sample of the DataFrame."""
        if df.empty:
            return pd.DataFrame(columns=df.columns)
        return df.sample(n=min(n, len(df)))

# ================================================================
# Data System
# ================================================================

class DataSystem:
    """Class to handle data system operations, including file loading and query processing."""

    def __init__(self, uploaded_files):
        """Initialize DataSystem with uploaded files."""
        self.uploaded_files = uploaded_files
        self.df = self.load_data_from_files()
        self.data_types_df = DataAnalyzer.summarize_data_types(self.df)
        self.sample_df = DataAnalyzer.get_sample(self.df)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
        self.pandas_df_agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            handle_parsing_errors=True,
            agent_type="zero-shot-react-description",
            verbose=True,
            return_intermediate_steps=False,
            allow_dangerous_code=True
        )

    def load_data_from_files(self) -> pd.DataFrame:
        """Load data from the uploaded files and combine into a single DataFrame."""
        dfs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in self.uploaded_files:
                temp_filepath = os.path.join(temp_dir, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())
                df = DataLoader.load_data(temp_filepath)
                dfs.append(df)
        
        # Combine all dataframes into one
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no files

    def process_user_query(self, query: str) -> str:
        """Process the user's query and return the result of the data analysis."""
        st_cb = StreamlitCallbackHandler(st.chat_message("assistant"))
        
        template = PromptTemplate(
            input_variables=["query", "language"],
            template="""Based on the following question: {query}, answer it in {language}."""
        )
        prompt = template.format(query=query, language="Spanish")
        response = self.pandas_df_agent.invoke(prompt, callbacks=[st_cb])
        
        if 'output' in response:
            return response['output']
        else:
            return "Unexpected response format."

# ================================================================
# Chat Management
# ================================================================

class ChatManager:
    """Class for managing chat interactions within the Streamlit app."""

    @staticmethod
    def initialize_chat_history():
        """Initialize chat history in the Streamlit session if it doesn't exist."""
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

    @staticmethod
    def add_message_to_history(role, message):
        """Add a message to the chat history."""
        st.session_state['chat_history'].append((role, message))

    @staticmethod
    def display_chat_history():
        """Display the chat history in the Streamlit app."""
        for role, message in st.session_state['chat_history']:
            if role == "user":
                st.chat_message("user").markdown(message)
            else:
                st.chat_message("assistant").markdown(message)

# ================================================================
# Environment Setup
# ================================================================

def set_environment():
    """Set environment variables for API keys and IDs."""
    for key, value in globals().items():
        if "API" in key or "ID" in key:
            os.environ[key] = value

GOOGLE_API_KEY = 'your_google_gemini_api_key'
set_environment()
