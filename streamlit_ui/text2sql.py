import streamlit as st
st.set_page_config(layout="wide")
import atexit
import numpy as np
import pandas as pd
import re
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from typing import List, Dict, Any
from datetime import datetime
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import json
import time

USE_CASES = {
    "Sales Analysis": {
        "description": "Analyze sales data to identify trends, top-performing products, and customer behavior.",
        "prompt_template": """
        You are a sales analyst. Generate a SQL query to analyze sales data based on the following database information, conversation history, and question.
        
        Database Information:
        {context}
        
        Previous Conversation:
        {history}
        
        User Question: {question}
        
        Focus on:
        - Sales trends over time
        - Top-performing products or categories
        - Customer segmentation by purchase behavior
        - Revenue analysis by region or channel
        
        Return ONLY the SQL query without any explanations or decorations.
        If you cannot generate a valid query, respond with "I cannot answer this question with the available data."
        
        Response:
        """
    },
    "Customer Segmentation": {
        "description": "Segment customers based on their behavior, demographics, and purchase history.",
        "prompt_template": """
        You are a marketing analyst. Generate a SQL query to segment customers based on the following database information, conversation history, and question.
        
        Database Information:
        {context}
        
        Previous Conversation:
        {history}
        
        User Question: {question}
        
        Focus on:
        - RFM (Recency, Frequency, Monetary) analysis
        - Demographic segmentation (age, gender, location)
        - Behavioral segmentation (purchase frequency, product preferences)
        - Customer lifetime value (CLV) analysis
        
        Return ONLY the SQL query without any explanations or decorations.
        If you cannot generate a valid query, respond with "I cannot answer this question with the available data."

        Response:
        """
    },
    "Inventory Management": {
        "description": "Manage and analyze inventory data to optimize stock levels.",
        "prompt_template": """
        You are an inventory manager. Generate a SQL query to analyze inventory data based on the following database information, conversation history, and question.
        
        Database Information:
        {context}
        
        Previous Conversation:
        {history}
        
        User Question: {question}
        
        Focus on:
        - Stock levels and reorder points
        - Inventory turnover rates
        - Deadstock or obsolete inventory
        - Supplier performance analysis
        
        Return ONLY the SQL query without any explanations or decorations.
        If you cannot generate a valid query, respond with "I cannot answer this question with the available data."

        Response:
        """
    },
    "Healthcare Analytics": {
        "description": "Analyze healthcare data for patient outcomes, treatment efficacy, and operational efficiency.",
        "prompt_template": """
        You are a healthcare analyst. Generate a SQL query to analyze healthcare data based on the following database information, conversation history, and question.
        
        Database Information:
        {context}
        
        Previous Conversation:
        {history}
        
        User Question: {question}
        
        Focus on:
        - Patient outcomes and treatment efficacy
        - Hospital readmission rates
        - Operational efficiency (bed occupancy, wait times)
        - Disease prevalence and trends
        
        Return ONLY the SQL query without any explanations or decorations.
        If you cannot generate a valid query, respond with "I cannot answer this question with the available data."

        Response:
        """
    },
    "Finance Analytics": {
        "description": "Analyze financial transactions, detect fraud, and evaluate profitability.",
        "prompt_template": """
        You are a finance analyst. Generate a SQL query to analyze financial data based on the following database information, conversation history, and question.
        
        Database Information:
        {context}
        
        Previous Conversation:
        {history}
        
        User Question: {question}
        
        Focus on:
        - Transaction analysis
        - Fraud detection
        - Revenue trend analysis
        - Expense breakdown
        - ROI analysis
        - Customer profitability
        - Budget variance analysis
        
        Return ONLY the SQL query without any explanations or decorations.
        If you cannot generate a valid query, respond with "I cannot answer this question with the available data."

        Response:
        """
    }
}
# Initialize session state variables
session_state_keys = [
    'messages', 'db_path', 'vector_store', 'table_info', 'chat_history', 
    'current_chat', 'df_preview', 'query_history', 'favorites', 'schema_visualization',
    'query_explanation', 'execution_time', 'execution_plan', 'query_templates', 'use_case'
]

for key in session_state_keys:
    if key not in st.session_state:
        if key in ['messages', 'chat_history', 'query_history', 'favorites', 'query_templates']:
            st.session_state[key] = []
        else:
            st.session_state[key] = None

# Initialize query templates
if not st.session_state.query_templates:
    st.session_state.query_templates = [
        {"name": "Top N Records", "template": "Show me the top {n} records by {column}"},
        {"name": "Group By Summary", "template": "Summarize {measure} by {dimension}"},
        {"name": "Time Trend", "template": "Show {metric} trend over {time_column}"},
        {"name": "Filter and Sort", "template": "Show data where {condition} sorted by {column}"},
        {"name": "Aggregation", "template": "Calculate {aggregate_function} of {column} grouped by {dimension}"}
    ]

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

def save_to_favorites(query, result=None):
    """Save query to favorites"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    favorite = {
        'timestamp': timestamp,
        'query': query,
        'result': result.to_dict('records') if result is not None else None
    }
    st.session_state.favorites.append(favorite)
    return "Query saved to favorites!"

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
    column_stats = []
    
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        
        # Get distinct count
        cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM data_table;")
        distinct_count = cursor.fetchone()[0]
        
        # Get samples
        cursor.execute(f"SELECT {col_name} FROM data_table LIMIT 3;")
        samples = [str(row[0]) for row in cursor.fetchall()]
        
        # Get min, max for numeric columns
        min_val = max_val = avg_val = None
        if col_type in ['INTEGER', 'REAL', 'FLOAT', 'DOUBLE']:
            try:
                cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM data_table;")
                min_val, max_val, avg_val = cursor.fetchone()
                stats = f"Min: {min_val}, Max: {max_val}, Avg: {round(avg_val, 2) if avg_val is not None else 'N/A'}"
            except:
                stats = "Stats not available"
        else:
            stats = "Non-numeric column"
        
        column_info.append(f"Column '{col_name}' (Type: {col_type}, Distinct Values: {distinct_count}, Examples: {', '.join(samples)})")
        column_stats.append({
            "name": col_name,
            "type": col_type,
            "distinct_count": distinct_count,
            "samples": samples,
            "stats": stats
        })
    
    # Create a graph representation of the schema for visualization
    schema_viz = {
        "table_name": "data_table",
        "row_count": row_count,
        "columns": [{"name": col[1], "type": col[2]} for col in columns]
    }
    
    st.session_state.schema_visualization = schema_viz
    conn.close()
    
    return f"""
    Table Name: data_table
    Total Rows: {row_count}
    
    Schema Information:
    {'\n'.join(column_info)}
    """, column_stats

def get_query_explanation(query: str) -> str:
    """Get natural language explanation of SQL query"""
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    explain_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Explain the following SQL query in simple terms, breaking down each component:
        
        ```sql
        {query}
        ```
        
        Provide a clear explanation that a non-technical person could understand.
        """
    )
    
    explain_chain = LLMChain(llm=chat_model, prompt=explain_prompt)
    
    try:
        explanation = explain_chain.run(query=query)
        return explanation
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"

def optimize_query(query: str) -> str:
    """Get optimization suggestions for SQL query"""
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    optimize_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Analyze the following SQL query and suggest optimizations:
        
        ```sql
        {query}
        ```
        
        Provide specific suggestions to improve performance, readability, and efficiency.
        """
    )
    
    optimize_chain = LLMChain(llm=chat_model, prompt=optimize_prompt)
    
    try:
        optimization = optimize_chain.run(query=query)
        return optimization
    except Exception as e:
        return f"Could not generate optimization suggestions: {str(e)}"

def get_execution_plan(db_path: str, query: str) -> str:
    """Get execution plan for SQL query"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = cursor.fetchall()
        conn.close()
        
        plan_text = "Execution Plan:\n"
        for step in plan:
            plan_text += f"- {step[3]}\n"
        
        return plan_text
    except Exception as e:
        conn.close()
        return f"Could not generate execution plan: {str(e)}"

def clean_sql_query(query: str) -> str:
    """Clean and extract SQL query from the model's response"""
    # Remove markdown code block formatting (backticks)
    query = re.sub(r'^```sql|^```|```$', '', query, flags=re.MULTILINE)
    
    # Remove any trailing semicolons and extra whitespace
    query = query.strip().rstrip(';')
    
    # Add semicolon back for consistency
    return query + ';'

def execute_sql_query(db_path: str, query: str) -> tuple:
    """Execute SQL query and return results as DataFrame along with execution time"""
    conn = sqlite3.connect(db_path)
    start_time = time.time()
    
    try:
        df = pd.read_sql_query(query, conn)
        execution_time = round(time.time() - start_time, 4)
        
        # Save to query history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.query_history.append({
            'timestamp': timestamp,
            'query': query,
            'execution_time': execution_time,
            'row_count': len(df)
        })
        
        conn.close()
        return df, execution_time
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
                
        elif chart_type == "heatmap":
            if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 2:
                pivot = df.pivot_table(
                    values=numeric_cols[0], 
                    index=non_numeric_cols[0], 
                    columns=non_numeric_cols[1], 
                    aggfunc='mean'
                )
                fig = px.imshow(pivot, title=f"Heatmap of {numeric_cols[0]} by {non_numeric_cols[0]} and {non_numeric_cols[1]}")
                return fig, None
            else:
                return None, "Cannot create heatmap: Need at least 1 numeric column and 2 categorical columns"
        
        elif chart_type == "bubble":
            if len(numeric_cols) >= 3:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], size=numeric_cols[2],
                               title=f"Bubble Chart: {numeric_cols[1]} vs {numeric_cols[0]} (size: {numeric_cols[2]})")
                return fig, None
            else:
                return None, "Cannot create bubble chart: Need at least 3 numeric columns"
        
        else:
            return None, f"Unsupported chart type: {chart_type}"
    
    except Exception as e:
        return None, f"Error creating chart: {str(e)}"

def detect_visualization_request(question):
    """Detect if user is asking for visualization"""
    vis_keywords = [
        "visualize", "visualization", "plot", "chart", "graph", "diagram",
        "show me a chart", "display a graph", "create a plot", "draw a",
        "bar chart", "pie chart", "line graph", "histogram", "scatter plot",
        "heatmap", "bubble chart"
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
        "histogram": ["histogram", "distribution chart"],
        "heatmap": ["heatmap", "heat map", "correlation matrix"],
        "bubble": ["bubble chart", "bubble plot", "bubble graph"]
    }
    
    for chart_type, keywords in chart_types.items():
        for keyword in keywords:
            if keyword in question_lower:
                return chart_type
    
    # Default to bar chart if visualization is requested but type is not specified
    return "bar"

def extract_sql_refinement_intent(question):
    """Determine if user wants to refine previous SQL query"""
    refinement_keywords = [
        "modify", "change", "refine", "update", "adjust", "previous query",
        "last query", "that query", "fix", "improve", "edit", "adapt"
    ]
    
    question_lower = question.lower()
    for keyword in refinement_keywords:
        if keyword in question_lower:
            return True
    return False

def create_schema_diagram():
    """Create an interactive data relationship visualization using Plotly"""
    if st.session_state.db_path is None:
        return None
    
    try:
        # Connect to the database and load data
        conn = sqlite3.connect(st.session_state.db_path)
        df = pd.read_sql_query("SELECT * FROM data_table", conn)
        conn.close()
        
        # Create visualization controls
        st.subheader("Interactive Data Explorer")
        
        # Get only numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns for visualizations")
            return None
            
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["Correlation Matrix", "Scatter Plot Matrix"])
        
        with viz_tabs[0]:  # Correlation Matrix
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                
                # Create correlation heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='Viridis',
                    zmin=-1, zmax=1
                ))
                
                fig.update_layout(
                    title="Correlation Matrix",
                    height=500,
                    width=700
                )
                
                st.plotly_chart(fig)
                
                # Show strongest correlations
                st.subheader("Strongest Relationships")
                
                # Get the upper triangle of the correlation matrix
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                
                # Find the top 5 strongest correlations
                strongest = upper.unstack().sort_values(kind="quicksort", ascending=False).dropna()
                strongest = strongest.head(5)
                
                # Display the strongest correlations
                for i, (cols, val) in enumerate(strongest.items()):
                    st.write(f"{i+1}. **{cols[0]}** and **{cols[1]}**: {val:.3f}")
        
        with viz_tabs[1]:  # Scatter Plot Matrix
            if len(numeric_cols) > 1:
                # Let user select columns for scatter plot matrix (max 4 for readability)
                selected_cols = st.multiselect(
                    "Select columns for scatter plot matrix (max 4 recommended)",
                    options=numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )
                
                if selected_cols and len(selected_cols) >= 2:
                    fig = px.scatter_matrix(
                        df, 
                        dimensions=selected_cols,
                        color=df[selected_cols[0]],
                        opacity=0.7
                    )
                    fig.update_layout(height=700, width=700)
                    st.plotly_chart(fig)
                else:
                    st.info("Select at least 2 columns for scatter plot matrix")
        return True
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None

def generate_follow_up_questions(context, question, result_df):
    """Generate follow-up questions based on current question and results"""
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    
    # Create a description of the result data
    result_description = "No results" if result_df is None or result_df.empty else f"Result with {len(result_df)} rows and columns: {', '.join(result_df.columns.tolist())}"
    
    follow_up_prompt = PromptTemplate(
        input_variables=["context", "question", "result_description"],
        template="""
        Based on the following database information, the user's current question, and the query results,
        suggest 3 natural follow-up questions the user might want to ask next. Make them conversational and directly
        related to the current question or results. Return them as a comma-separated list.
        
        Database Information:
        {context}
        
        User's Current Question: {question}
        
        Query Result Information: {result_description}
        
        3 Follow-up Questions (comma-separated):
        """
    )
    
    follow_up_chain = LLMChain(llm=chat_model, prompt=follow_up_prompt)
    
    retries = 3
    for attempt in range(retries):
        try:
            follow_ups = follow_up_chain.run(
                context=context, 
                question=question, 
                result_description=result_description
            )
            return [q.strip() for q in follow_ups.split(',') if q.strip()]
        except Exception as e:       
            if "ResourceExhausted" in str(e) or "429" in str(e):
                wait_time = (attempt + 1) * 2  # Exponential backoff (2s, 4s, 6s)
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                break
    return ["Would you like to refine your query?", "Do you want to include more columns?", "Should I use different filtering criteria?"]
    
def interpret_natural_query(question):
    """Advanced query interpretation to extract dimensions, metrics, etc."""
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    interpret_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Analyze this natural language query and extract the key elements:
        
        "{question}"
        
        Return a JSON object with the following structure:
        {{
          "intent": "filter|aggregation|comparison|trend|visualization|other",
          "dimensions": ["dimension1", "dimension2"],
          "metrics": ["metric1", "metric2"],
          "filters": ["filter1", "filter2"],
          "time_period": "time period if specified"
        }}
        
        Return ONLY the JSON without any explanation.
        """
    )
    
    interpret_chain = LLMChain(llm=chat_model, prompt=interpret_prompt)
    
    try:
        interpretation = interpret_chain.run(question=question)
        # Extract just the JSON part
        json_match = re.search(r'({.*})', interpretation, re.DOTALL)
        if json_match:
            interpretation = json_match.group(1)
        return json.loads(interpretation)
    except Exception as e:
        return {
            "intent": "other",
            "dimensions": [],
            "metrics": [],
            "filters": [],
            "time_period": ""
        }

# Create Streamlit interface
st.title("Advanced Text2SQL Chatbot")
st.write("Upload a CSV file to create a database and ask questions in natural language!")

if 'use_case' not in st.session_state or st.session_state.use_case is None:
    st.write("### Select a Use Case")
    st.write("Choose a use case to get started.")

    # Define use case cards in a grid layout
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("""
                <div style="padding: 16px; border-radius: 8px; background-color: #fefbd8; margin-bottom: 16px;">
                    <h3>Sales Analysis</h3>
                    <p>Analyze sales data to identify trends, top-performing products, and customer behavior.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Sales Analysis", key="sales_analysis"):
                st.session_state.use_case = "Sales Analysis"
                st.rerun()

        with st.container():
            st.markdown("""
                <div style="padding: 16px; border-radius: 8px; background-color: #fefbd8; margin-bottom: 16px;">
                    <h3>Customer Segmentation</h3>
                    <p>Segment customers based on their behavior, demographics, and purchase history.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Customer Segmentation", key="customer_segmentation"):
                st.session_state.use_case = "Customer Segmentation"
                st.rerun()

    with col2:
        with st.container():
            st.markdown("""
                <div style="padding: 16px; border-radius: 8px; background-color: #fefbd8; margin-bottom: 16px;">
                    <h3>Inventory Management</h3>
                    <p>Manage and analyze inventory data to optimize stock levels.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Inventory Management", key="inventory_management"):
                st.session_state.use_case = "Inventory Management"
                st.rerun()

        with st.container():
            st.markdown("""
                <div style="padding: 16px; border-radius: 8px; background-color: #fefbd8; margin-bottom: 16px;">
                    <h3>Healthcare Analytics</h3>
                    <p>Analyze healthcare data for patient outcomes, treatment efficacy, and operational efficiency.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Healthcare Analytics", key="healthcare_analytics"):
                st.session_state.use_case = "Healthcare Analytics"
                st.rerun()

        with st.container():
            st.markdown("""
                <div style="padding: 16px; border-radius: 8px; background-color: #fefbd8; margin-bottom: 16px;">
                    <h3>Finance Analytics</h3>
                    <p>Analyze financial transactions, detect fraud, and evaluate profitability.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Finance Analytics", key="finance_analytics"):
                st.session_state.use_case = "Finance Analytics"
                st.rerun()

# if st.session_state.use_case:
#     prompt_template = USE_CASES[st.session_state.use_case]['prompt_template']
# else:
#     prompt_template = """
#     You are a SQL expert. Generate a SQL query based on the following database information, conversation history, and question:
    
#     Database Information:
#     {context}
    
#     Previous Conversation:
#     {history}
    
#     User Question: {question}
    
#     Return ONLY the SQL query without any explanations or decorations.
#     If you cannot generate a valid query, respond with "I cannot answer this question with the available data."
#     """
    
# Sidebar for navigation and features
with st.sidebar:
    st.title("Navigation")
    menu_choice = st.radio("Menu", ["New Chat", "Chat History", "Query History", "Favorites", "Templates", "Analytics", "About"])
    
    if menu_choice == "Chat History":
        st.subheader("Previous Chats")
        for idx, chat in enumerate(st.session_state.chat_history):
            if st.button(f"Chat {idx + 1} - {chat['timestamp']}"):
                load_chat(chat)
        
        if st.button("Clear All History"):
            st.session_state.chat_history = []
    
    elif menu_choice == "Query History":
        st.subheader("Recent Queries")
        for idx, query in enumerate(st.session_state.query_history):
            with st.expander(f"Query {idx + 1} - {query['timestamp']}"):
                st.code(query['query'], language="sql")
                st.text(f"Execution time: {query['execution_time']}s, Rows: {query['row_count']}")
                if st.button(f"Reuse Query #{idx + 1}"):
                    st.session_state.messages.append({"role": "user", "content": f"Execute this SQL: {query['query']}"})
        
        if st.button("Clear Query History"):
            st.session_state.query_history = []
    
    elif menu_choice == "Favorites":
        st.subheader("Favorite Queries")
        for idx, fav in enumerate(st.session_state.favorites):
            with st.expander(f"Favorite {idx + 1} - {fav['timestamp']}"):
                st.code(fav['query'], language="sql")
                if st.button(f"Reuse Favorite #{idx + 1}"):
                    st.session_state.messages.append({"role": "user", "content": f"Execute this SQL: {fav['query']}"})
        
        if st.button("Clear Favorites"):
            st.session_state.favorites = []
    
    elif menu_choice == "Templates":
        st.subheader("Query Templates")
        for idx, template in enumerate(st.session_state.query_templates):
            with st.expander(f"{template['name']}"):
                st.text(template['template'])
                if st.button(f"Use Template #{idx + 1}"):
                    st.session_state.messages.append({"role": "user", "content": template['template']})
   
    elif menu_choice == "Analytics":
        st.subheader("Query Analytics")
        
        if st.session_state.query_history:
            # Extract execution times and timestamps for trending
            times = [q['execution_time'] for q in st.session_state.query_history]
            timestamps = [q['timestamp'] for q in st.session_state.query_history]
            
            # Display average execution time
            st.metric("Average Query Time", f"{sum(times)/len(times):.4f}s")
            
            # Create a trend chart
            trend_df = pd.DataFrame({
                'Timestamp': timestamps,
                'Execution Time (s)': times
            })
            st.line_chart(trend_df.set_index('Timestamp'))
        else:
            st.info("No query data available yet")
    
    elif menu_choice == "About":
        st.markdown("""
        ### Advanced Text-to-SQL Chatbot
        - Upload CSV files
        - Ask questions in natural language
        - Get SQL queries and results
        - Query explanations
        - Performance optimization
        - Schema visualization
        - Interactive query refinement
        - Query templates and favorites
        - Advanced visualizations
        - Query history analytics
        """)

if st.session_state.use_case:
    st.write(f"**Selected Use Case:** {st.session_state.use_case}")
    if menu_choice == "New Chat" or menu_choice == "Chat History":
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None and st.session_state.db_path is None:
            st.session_state.db_path = create_db_from_csv(uploaded_file)
            st.session_state.table_info, column_stats = get_table_info(st.session_state.db_path)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.vector_store = create_vector_store(st.session_state.table_info, embeddings)
            st.success("Database created!")
            
        # Database info expander section
        if st.session_state.db_path:
            with st.expander("Database Information"):
                tabs = st.tabs(["Preview", "Schema", "Statistics", "Visualization"])
                
                with tabs[0]:  # Preview tab
                    if st.session_state.df_preview is not None:
                        st.dataframe(st.session_state.df_preview)
                
                with tabs[1]:  # Schema tab
                    if st.session_state.table_info:
                        st.text(st.session_state.table_info)
                
                with tabs[2]:  # Statistics tab
                    if st.session_state.db_path:
                        conn = sqlite3.connect(st.session_state.db_path)
                        df = pd.read_sql_query("SELECT * FROM data_table", conn)
                        conn.close()
                        
                        st.write("Numeric Column Statistics:")
                        st.dataframe(df.describe())
                
                with tabs[3]:  # Visualization tab
                    schema_fig = create_schema_diagram()
                    if not schema_fig:
                        st.info("Upload a CSV file with numeric data to see visualizations")
    
        if st.session_state.db_path:           
            chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
            prompt = PromptTemplate(
                input_variables=["context", "question", "history"],
                template=USE_CASES[st.session_state.use_case]['prompt_template']
            )
            
            chain = LLMChain(llm=chat_model, prompt=prompt)
            
            # Chat interface with history management
            col1, col2 = st.columns([3, 1])
            with col1:
                if question := st.chat_input("Ask a question"):
                    # Check if this is a refinement request
                    is_refinement = extract_sql_refinement_intent(question)
                    
                    # Get previous SQL query if refinement is requested
                    previous_query = None
                    if is_refinement:
                        for msg in reversed(st.session_state.messages):
                            if msg["role"] == "assistant" and "SQL Query:" in msg["content"]:
                                query_match = re.search(r'```sql\n(.*?)\n```', msg["content"], re.DOTALL)
                                if query_match:
                                    previous_query = query_match.group(1)
                                    break
                    
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # Create chat history context for the model
                    history_text = ""
                    for msg in st.session_state.messages[-6:-1]:  # Last 5 messages excluding current
                        if msg["role"] == "user":
                            history_text += f"User: {msg['content']}\n"
                        else:
                            # Extract just the query part if it's an assistant message with SQL
                            if "SQL Query:" in msg["content"]:
                                query_match = re.search(r'```sql\n(.*?)\n```', msg["content"], re.DOTALL)
                                if query_match:
                                    history_text += f"Assistant: Generated SQL: {query_match.group(1)}\n"
                                else:
                                    history_text += f"Assistant: {msg['content']}\n"
                            else:
                                history_text += f"Assistant: {msg['content']}\n"
                    
                    try:
                        # Analyze the query for advanced understanding
                        query_analysis = interpret_natural_query(question)
                        
                        # Get relevant database context
                        docs = st.session_state.vector_store.similarity_search(question)
                        context = "\n".join([doc.page_content for doc in docs])
                        
                        if is_refinement and previous_query:
                            question = f"Refine this SQL query: {previous_query}\nNew requirements: {question}"
                        
                        # Generate SQL query
                        sql_response = chain.run(context=context, question=question, history=history_text)
                        
                        if "cannot answer" in sql_response.lower():
                            st.session_state.messages.append({"role": "assistant", "content": sql_response})
                        else:
                            cleaned_query = clean_sql_query(sql_response)
                            
                            # Get query explanation
                            explanation = get_query_explanation(cleaned_query)
                            
                            # Get query execution plan
                            execution_plan = get_execution_plan(st.session_state.db_path, cleaned_query)
                            
                            # Execute query with timing
                            results, exec_time = execute_sql_query(st.session_state.db_path, cleaned_query)
                            
                            # Generate optimization suggestions
                            optimization = optimize_query(cleaned_query)
                            
                            # Generate follow-up questions
                            follow_up_questions = generate_follow_up_questions(context, question, results)
                            
                            # Check if visualization is requested
                            viz_requested = detect_visualization_request(question)
                            
                            if viz_requested:
                                chart_type = identify_chart_type(question)
                                fig, error = create_chart(results, chart_type)
                                
                                if fig:
                                    # Build response with query details
                                    response = f"""
SQL Query:
```sql
{cleaned_query}
```

Execution Time: {exec_time}s

Results:
{results.to_markdown()}

Explanation:
{explanation}
                                    """
                                    
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": "**Results:**",
                                    "dataframe": results
                                    })
                                    
                                    # Store chart information in the message
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": f"Here's a {chart_type} chart visualization of your data:",
                                        "chart": {
                                            "type": chart_type,
                                            "data": results.to_dict('records')
                                        }
                                    })
                                    
                                    # Add follow-up questions if available
                                    if follow_up_questions:
                                        follow_up_text = "You might want to ask:\n\n" + "\n".join([f"- {q}" for q in follow_up_questions])
                                        st.session_state.messages.append({"role": "assistant", "content": follow_up_text})
                                    
                                else:
                                    response = f"""
SQL Query:
```sql
{cleaned_query}
```

Execution Time: {exec_time}s

Results:
{results.to_markdown()}

Explanation:
{explanation}

Couldn't create visualization: {error}
                                    """
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                # Build response with query details
                                response = f"""
SQL Query:
```sql
{cleaned_query}
```

Execution Time: {exec_time}s

Results:
{results.to_markdown()}

Explanation:
{explanation}
                                """
                                
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "**Results:**",
                                "dataframe": results
                                })
                                # Add follow-up questions if available
                                if follow_up_questions:
                                    follow_up_text = "You might want to ask:\n\n" + "\n".join([f"- {q}" for q in follow_up_questions])
                                    st.session_state.messages.append({"role": "assistant", "content": follow_up_text})
                            
                            # Store execution plan and optimization in session state
                            st.session_state.execution_plan = execution_plan
                            st.session_state.query_optimization = optimization
                            
                            save_chat_history()
                            
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            
            with col2:
                col2_btns = st.columns(2)
                with col2_btns[0]:
                    if st.button("Clear Chat"):
                        clear_chat()
                with col2_btns[1]:
                    if st.button("Save Query"):
                        # Find the last SQL query in the chat
                        last_query = None
                        for msg in reversed(st.session_state.messages):
                            if msg["role"] == "assistant" and "SQL Query:" in msg["content"]:
                                query_match = re.search(r'```sql\n(.*?)\n```', msg["content"], re.DOTALL)
                                if query_match:
                                    last_query = query_match.group(1)
                                    save_msg = save_to_favorites(last_query)
                                    st.success(save_msg)
                                    break
                        if not last_query:
                            st.error("No SQL query found to save")
                
                # Add optimization and execution plan expanders
                if st.session_state.execution_plan:
                    with st.expander("Execution Plan"):
                        st.text(st.session_state.execution_plan)
                
                if hasattr(st.session_state, 'query_optimization') and st.session_state.query_optimization:
                    with st.expander("Query Optimization"):
                        st.markdown(st.session_state.query_optimization)
            
            # Display chat with visualization support
            for idx, message in enumerate(st.session_state.messages):
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
