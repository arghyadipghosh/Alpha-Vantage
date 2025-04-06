import os
import yfinance as yf
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
from dotenv import load_dotenv
import logging
import requests
load_dotenv()

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def get_tickers_from_api(company_names):
    """
    Fetches ticker symbols for multiple company names using the API:
    https://ticker-2e1ica8b9.now.sh/keyword/{company_name}
    
    Args:
        company_names (str): A comma-separated string of company names.
    
    Returns:
        list: A list of ticker symbols corresponding to the input company names.
              If a ticker is not found for a company, it will be replaced with 'None'.
    """
    company_list = [name.strip() for name in company_names.split(",")]
    tickers = []

    for company_name in company_list:
        try:
            logger.info("Fetching ticker for company: %s", company_name)
            url = f"https://ticker-2e1ica8b9.now.sh/keyword/{company_name}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    # Assuming the API returns a list of results, take the first match
                    ticker = data[0].get("symbol")
                    tickers.append(ticker)
                    logger.info("Found ticker for company '%s': %s", company_name, ticker)
                else:
                    logger.warning("No results found for company: %s", company_name)
                    tickers.append(None)
            else:
                logger.error("Failed to fetch ticker for company '%s'. HTTP Status: %s", company_name, response.status_code)
                tickers.append(None)
        except Exception as e:
            logger.error("Error fetching ticker for company '%s': %s", company_name, str(e))
            tickers.append(None)

    return tickers

# Function to fetch and compare stock data
def compare_stocks(symbols):
    data = {}
    for symbol in symbols:
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")  # Fetch last 6 months' data
            
            if hist.empty:
                st.warning(f"No data found for {symbol}, skipping it.")
                continue  # Skip this ticker if no data found
            
            # Calculate overall % change
            data[symbol] = hist['Close'].pct_change().sum()
        
        except Exception as e:
            st.error(f"Could not retrieve data for {symbol}. Reason: {str(e)}")
            continue  # Skip this ticker if an error occurs

    return data

# Define the Market Analyst Agent
market_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Analyzes and compares stock performance over time.",
    instructions=[
        "Retrieve and compare stock performance from Yahoo Finance.",
        "Calculate percentage change over a 6-month period.",
        "Rank stocks based on their relative performance."
    ],
    show_tool_calls=True,
    markdown=True
)

# Function to get market analysis
def get_market_analysis(symbols):
    performance_data = compare_stocks(symbols)
    print(f'performance data: {performance_data}')
    
    if not performance_data:
        return "No valid stock data found for the given symbols."

    analysis = market_analyst.run(f"Compare these stock performances: {performance_data}")
    print(f'analysis report: {analysis}')
    if not analysis:
        return "No analysis report generated."
    return analysis.content

# Function to fetch company information
def get_company_info(symbol):
    stock = yf.Ticker(symbol)
    return {
        "name": stock.info.get("longName", "N/A"),
        "sector": stock.info.get("sector", "N/A"),
        "market_cap": stock.info.get("marketCap", "N/A"),
        "summary": stock.info.get("longBusinessSummary", "N/A"),
    }

# Function to fetch company news
def get_company_news(symbol):
    stock = yf.Ticker(symbol)
    try:
        news = stock.news[:5]  # Get latest 5 news articles
        return news
    except AttributeError:
        return []

# Define the Company Researcher Agent
company_researcher = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Fetches company profiles, financials, and latest news.",
    instructions=[
        "Retrieve company information from Yahoo Finance.",
        "Summarize latest company news relevant to investors.",
        "Provide sector, market cap, and business overview."
    ],
    markdown=True
)

# Function to analyze a company
def get_company_analysis(symbol):
    info = get_company_info(symbol)
    news = get_company_news(symbol)
    response = company_researcher.run(
        f"Provide an analysis for {info['name']} in the {info['sector']} sector.\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n"
        f"Latest News: {news}"
    )
    return response.content

# Define the Stock Strategist Agent
stock_strategist = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Provides investment insights and recommends top stocks.",
    instructions=[
        "Analyze stock performance trends and company fundamentals.",
        "Evaluate risk-reward potential and industry trends.",
        "Provide top stock recommendations for investors."
    ],
    markdown=True
)

# Function to get stock recommendations
def get_stock_recommendations(symbols):
    market_analysis = get_market_analysis(symbols)
    data = {}
    for symbol in symbols:
        data[symbol] = get_company_analysis(symbol)
    recommendations = stock_strategist.run(
        f"Based on the market analysis: {market_analysis}, and company news {data} "
        f"which stocks would you recommend for investment?"
    )
    return recommendations.content

# Define the Team Lead Agent
team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Aggregates stock analysis, company research, and investment strategy.",
    instructions=[
        "Compile stock performance, company analysis, and recommendations.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis."
    ],
    markdown=True
)

# Function to generate the final investment report
def get_final_investment_report(symbols):
    market_analysis = get_market_analysis(symbols)
    company_analyses = [get_company_analysis(symbol) for symbol in symbols]
    print(f'company analyses: {company_analyses}')
    stock_recommendations = get_stock_recommendations(symbols)
    print(f'stock recommendations: {stock_recommendations}')

    final_report = team_lead.run(
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses:\n{company_analyses}\n\n"
        f"Stock Recommendations:\n{stock_recommendations}\n\n"
        f"Provide the full analysis of each stock with Fundamentals and market news. "
        f"Generate a final ranked list in ascending order on which should I buy."
    )
    return final_report.content

# Streamlit page configuration
st.set_page_config(page_title="Alpha Vantage", page_icon="📈", layout="wide")

# Title and header
st.markdown("""
    <h1 style="text-align: center; color: #000000;">Alpha Vantage Bot</h3>
""", unsafe_allow_html=True)

# Sidebar Styling
st.sidebar.markdown("""
    <h2 style="color: #343a40;">Configuration</h2>
    <p style="color: #6c757d;">Enter the stock symbols you want to analyze. The AI will provide detailed insights, performance reports, and top recommendations.</p>
""", unsafe_allow_html=True)

# Stock symbols input
input_symbols = st.sidebar.text_input("Enter Company Names (separated by commas)", "Apple, Tesla, Alphabet")
api_key = st.sidebar.text_input("Enter your API Key (optional)", type="password")

# Parse the company names input
company_names = [name.strip() for name in input_symbols.split(",")]

# Generate Investment Report button
if st.sidebar.button("Generate Investment Report"):
    if not company_names:
        st.sidebar.warning("Please enter at least one company name.")
    else:
        # Fetch tickers from company names
        tickers = get_tickers_from_api(", ".join(company_names))
        
        # Filter out None values from the tickers list
        valid_tickers = [ticker for ticker in tickers if ticker is not None]
        
        if not valid_tickers:
            st.sidebar.warning("No valid tickers found for the given company names.")
        else:
            logger.info("Valid tickers: %s", valid_tickers)
            
            # Generate the final report using the valid tickers
            report = get_final_investment_report(valid_tickers)
            
            # Display the report
            st.subheader("Investment Report")
            st.markdown(report)

            # Interactive Stock Performance Chart
            st.markdown("### 📊 Stock Performance (6-Months)")
            stock_data = yf.download(valid_tickers, period="6mo")['Close']

            fig = go.Figure()
            for symbol in valid_tickers:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[symbol], mode='lines', name=symbol))

            fig.update_layout(title="Stock Performance Over the Last 6 Months",
                              xaxis_title="Date",
                              yaxis_title="Price (in USD)",
                              template="plotly_dark")
            st.plotly_chart(fig)

        st.info("This report provides detailed insights, including market performance, company analysis, and investment recommendations.")
