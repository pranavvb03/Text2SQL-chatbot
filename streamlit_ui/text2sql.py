import streamlit as st
st.set_page_config(layout="wide")
import atexit
import pandas as pd
import re
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.messages  import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from typing import List
from datetime import datetime
import tempfile
import plotly.express as px
import plotly.graph_objects as go

for key in ['messages', 'db_path', 'vector_store', 'table_info', 'chat_history', 'current_chat', 'df_preview']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['messages', 'chat_history'] else None


# Set your Google API key
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

def save_chat_history():
    """Save current chat to history"""
    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat = {
            'timestamp': timestamp,
            'messages': st.session_state.messages,
            'table_info': st.session_state.table_info
        }
        st.session_state.chat_history.append(chat)

def load_chat(chat):
    """Load selected chat"""
    st.session_state.messages = chat['messages']
    st.session_state.table_info = chat['table_info']

def clear_chat():
    """Clear current chat"""
    st.session_state.messages = []

def create_db_from_csv(csv_file) -> str:
    """Create SQLite database from uploaded CSV file"""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = temp_db.name
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Save a preview of the dataframe
    st.session_state.df_preview = df.head(10)
    
    # Clean column names: remove spaces and special characters
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Create SQLite connection
    conn = sqlite3.connect(db_path)
    
    # Save DataFrame to SQLite
    table_name = 'data_table'
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    conn.close()
    return db_path

def get_table_info(db_path: str) -> str:
    """Get comprehensive table information"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("PRAGMA table_info(data_table);")
    columns = cursor.fetchall()
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM data_table;")
    row_count = cursor.fetchone()[0]
    
    # Get column information and sample data
    column_info = []
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM data_table;")
        distinct_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT {col_name} FROM data_table LIMIT 3;")
        samples = [str(row[0]) for row in cursor.fetchall()]
        column_info.append(f"Column '{col_name}' (Type: {col_type}, Distinct Values: {distinct_count}, Examples: {', '.join(samples)})")
    
    conn.close()
    
    return f"""
    Table Name: data_table
    Total Rows: {row_count}
    
    Schema Information:
    {'\n'.join(column_info)}
    """

def clean_sql_query(query: str) -> str:
    """Clean and extract SQL query from the model's response"""
    # Remove markdown code block formatting (backticks)
    query = re.sub(r'^```sql|^```|```$', '', query, flags=re.MULTILINE)
    
    # Remove any trailing semicolons and extra whitespace
    query = query.strip().rstrip(';')
    
    # Add semicolon back for consistency
    return query + ';'

def execute_sql_query(db_path: str, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e

def create_vector_store(text: str, embeddings) -> FAISS:
    """Create FAISS vector store from text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(text)
    return FAISS.from_texts(texts, embeddings)

def create_chart(df, chart_type):
    """Create visualization based on dataframe and chart type"""
    try:
        if df.empty or len(df.columns) < 1:
            return None, "Cannot create chart: Not enough data"
        
        # Try to identify good candidates for x and y axes
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # For charts that need category/time data for x-axis
        x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
        
        # For charts that need numeric data for y-axis
        y_col = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {y_col} by {x_col}")
            return fig, None
        
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} over {x_col}")
            return fig, None
        
        elif chart_type == "pie":
            if len(numeric_cols) > 0:
                fig = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart: {y_col} Distribution by {x_col}")
                return fig, None
            else:
                return None, "Cannot create pie chart: No numeric columns found for values"
        
        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter Plot: {numeric_cols[1]} vs {numeric_cols[0]}")
                return fig, None
            else:
                return None, "Cannot create scatter plot: Need at least 2 numeric columns"
        
        elif chart_type == "histogram":
            if numeric_cols:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Histogram of {numeric_cols[0]}")
                return fig, None
            else:
                return None, "Cannot create histogram: No numeric columns found"
        
        else:
            return None, f"Unsupported chart type: {chart_type}"
    
    except Exception as e:
        return None, f"Error creating chart: {str(e)}"

def detect_visualization_request(question):
    """Detect if user is asking for visualization"""
    vis_keywords = [
        "visualize", "visualization", "plot", "chart", "graph", "diagram",
        "show me a chart", "display a graph", "create a plot", "draw a",
        "bar chart", "pie chart", "line graph", "histogram", "scatter plot"
    ]
    
    question_lower = question.lower()
    for keyword in vis_keywords:
        if keyword in question_lower:
            return True
    return False

def identify_chart_type(question):
    """Identify the type of chart requested"""
    question_lower = question.lower()
    
    chart_types = {
        "bar": ["bar chart", "bar graph", "column chart"],
        "pie": ["pie chart", "pie graph", "donut chart"],
        "line": ["line chart", "line graph", "trend line"],
        "scatter": ["scatter plot", "scatter graph", "scatter chart"],
        "histogram": ["histogram", "distribution chart"]
    }
    
    for chart_type, keywords in chart_types.items():
        for keyword in keywords:
            if keyword in question_lower:
                return chart_type
    
    # Default to bar chart if visualization is requested but type is not specified
    return "bar"

# Create Streamlit interface
st.title("Interactive Text2SQL Chatbot")
st.write("Upload a CSV file to create a database and ask questions in natural language!")


# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    menu_choice = st.radio("Menu", ["New Chat", "Chat History", "About"])
    
    if menu_choice == "Chat History":
        st.subheader("Previous Chats")
        for idx, chat in enumerate(st.session_state.chat_history):
            if st.button(f"Chat {idx + 1} - {chat['timestamp']}"):
                load_chat(chat)
        
        if st.button("Clear All History"):
            st.session_state.chat_history = []
    
    elif menu_choice == "About":
        st.markdown("""
        ### Text-to-SQL Chatbot
        - Upload CSV files
        - Ask questions in natural language
        - Get SQL queries and results
        - Maintains chat history
        - View database structure
        - Create visualizations
        """)

# Main content area
if menu_choice == "New Chat" or menu_choice == "Chat History":
    st.title("SQL Chatbot")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None and st.session_state.db_path is None:
        st.session_state.db_path = create_db_from_csv(uploaded_file)
        st.session_state.table_info = get_table_info(st.session_state.db_path)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vector_store = create_vector_store(st.session_state.table_info, embeddings)
        st.success("Database created!")
        
    if st.session_state.df_preview is not None:
        st.subheader("Database Preview")
        st.dataframe(st.session_state.df_preview)

    if st.session_state.db_path:           
        chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a SQL expert. Generate a SQL query based on the following database information and question.
            Return ONLY the SQL query without any explanations or decorations.
            If you cannot generate a valid query, respond with "I cannot answer this question with the available data."
            
            Database Information:
            {context}
            
            User Question: {question}
            
            Response:
            """
        )
        
        chain = LLMChain(llm=chat_model, prompt=prompt)
        
        # Chat interface with history management
        col1, col2 = st.columns([3, 1])
        with col1:
            if question := st.chat_input("Ask a question"):
                st.session_state.messages.append({"role": "user", "content": question})
                
                try:
                    docs = st.session_state.vector_store.similarity_search(question)
                    context = "\n".join([doc.page_content for doc in docs])
                    sql_response = chain.run(context=context, question=question)
                    
                    if "cannot answer" in sql_response.lower():
                        st.session_state.messages.append({"role": "assistant", "content": sql_response})
                    else:
                        cleaned_query = clean_sql_query(sql_response)
                        results = execute_sql_query(st.session_state.db_path, cleaned_query)
                        
                        # Check if visualization is requested
                        viz_requested = detect_visualization_request(question)
                        
                        if viz_requested:
                            chart_type = identify_chart_type(question)
                            fig, error = create_chart(results, chart_type)
                            
                            if fig:
                                response = f"SQL Query:\n```sql\n{cleaned_query}\n```\n\nResults:\n{results.to_markdown()}\n\n"
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                
                                # Store chart information in the message
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": f"Here's a {chart_type} chart visualization of your data:",
                                    "chart": {
                                        "type": chart_type,
                                        "data": results.to_dict('records')
                                    }
                                })
                            else:
                                response = f"SQL Query:\n```sql\n{cleaned_query}\n```\n\nResults:\n{results.to_markdown()}\n\nCouldn't create visualization: {error}"
                                st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            response = f"SQL Query:\n```sql\n{cleaned_query}\n```\n\nResults:\n{results.to_markdown()}"
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        save_chat_history()
                        
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
        
        with col2:
            if st.button("Clear Chat"):
                clear_chat()
        
        # Display chat with visualization support
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If the message has a chart, display it
                if "chart" in message:
                    chart_data = pd.DataFrame.from_records(message["chart"]["data"])
                    chart_type = message["chart"]["type"]
                    fig, _ = create_chart(chart_data, chart_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

# Cleanup
def cleanup():
    if st.session_state.db_path and os.path.exists(st.session_state.db_path):
        os.unlink(st.session_state.db_path)

atexit.register(cleanup)
