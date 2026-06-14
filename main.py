import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def encode_categorical_data(file_path: str) -> str:
    """Encodes categorical variables (like ad_type) into numbers."""
    df = pd.read_csv(file_path)
    le = LabelEncoder()
    if 'ad_type' in df.columns:
        df['ad_type_encoded'] = le.fit_transform(df['ad_type'])
        df.to_csv(file_path, index=False)
        return "Successfully encoded 'ad_type' to 'ad_type_encoded'."
    return "Column 'ad_type' not found."

@tool
def standardize_data(file_path: str) -> str:
    """Standardizes numerical features to have a mean of 0 and variance of 1."""
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    if 'time_spent' in df.columns:
        df['time_spent_scaled'] = scaler.fit_transform(df[['time_spent']])
        df.to_csv(file_path, index=False)
        return "Successfully standardized 'time_spent' to 'time_spent_scaled'."
    return "Column 'time_spent' not found."

@tool
def train_ctr_model(file_path: str) -> str:
    """Trains a Random Forest model on the dataset and saves predictions."""
    df = pd.read_csv(file_path)
    
    if 'ad_type_encoded' not in df.columns or 'time_spent_scaled' not in df.columns:
        return "Error: Data must be preprocessed before training."

    X = df[['ad_type_encoded', 'time_spent_scaled']]
    y = df['ctr']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    df['predicted_ctr'] = model.predict(X)
    df.to_csv(file_path, index=False)
    
    mse = mean_squared_error(y, df['predicted_ctr'])
    return f"Model trained successfully. Mean Squared Error: {mse:.4f}. Predictions saved to dataset."

@tool
def plot_predictions(file_path: str) -> str:
    """Generates a scatter plot comparing ground truth CTR to predicted CTR."""
    df = pd.read_csv(file_path)
    
    if 'predicted_ctr' not in df.columns:
        return "Error: No predictions found. Train the model first."

    plt.figure(figsize=(8, 6))
    plt.scatter(df['ctr'], df['predicted_ctr'], color='blue', alpha=0.7)
    plt.plot([df['ctr'].min(), df['ctr'].max()],
             [df['ctr'].min(), df['ctr'].max()], 'r--', lw=2)
    plt.title("Ground Truth vs. Predicted CTR")
    plt.xlabel("Actual CTR")
    plt.ylabel("Predicted CTR")
    plt.grid(True)
    plt.savefig("ctr_scatter_plot.png")
    return "Scatter plot generated and saved as 'ctr_scatter_plot.png'."

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    file_path: str

def eda_agent(state: AgentState):
    print("--- EDA AGENT WORKING ---")
    encode_res = encode_categorical_data.invoke({"file_path": state["file_path"]})
    scale_res = standardize_data.invoke({"file_path": state["file_path"]})
    return {"messages": [f"EDA done: {encode_res} {scale_res}"]}

def stats_agent(state: AgentState):
    print("--- STATISTICIAN AGENT WORKING ---")
    train_res = train_ctr_model.invoke({"file_path": state["file_path"]})
    return {"messages": [f"Stats done: {train_res}"]}

def viz_agent(state: AgentState):
    print("--- VISUALIZATION AGENT WORKING ---")
    plot_res = plot_predictions.invoke({"file_path": state["file_path"]})
    return {"messages": [f"Viz done: {plot_res}"]}

workflow = StateGraph(AgentState)

workflow.add_node("EDA_Expert", eda_agent)
workflow.add_node("Statistician", stats_agent)
workflow.add_node("Visualization_Expert", viz_agent)

workflow.set_entry_point("EDA_Expert")
workflow.add_edge("EDA_Expert", "Statistician")
workflow.add_edge("Statistician", "Visualization_Expert")
workflow.add_edge("Visualization_Expert", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_state = {
        "messages": ["Start the process."],
        "file_path": "ctr-prediction-dataset.csv"
    }
    
    print("Starting Multi-Agent System...\n")
    final_state = app.invoke(initial_state)
    
    print("\n--- FINAL SYSTEM LOG ---")
    for msg in final_state["messages"]:
        print(msg)
    print("\nCheck your directory for the updated CSV and 'ctr_scatter_plot.png'!")
