import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import schedule
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InvestmentAI:
    def __init__(self, news_api_key, alpha_vantage_key=None):
        self.news_api_key = news_api_key
        self.alpha_vantage_key = alpha_vantage_key
        self.model = None
        self.scaler = StandardScaler()
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.market_data = pd.DataFrame()
        self.news_data = pd.DataFrame()
        self.investment_recommendations = pd.DataFrame()
        
        logger.info("Investment AI initialized")
    
    def fetch_news(self, days=7, category='business', country='us', language='en'):
        """Fetch news articles from NewsAPI"""
        logger.info(f"Fetching news for the past {days} days")
        
        url = f"https://newsapi.org/v2/top-headlines?category={category}&sortBy=publishedAt&apiKey={self.news_api_key}&language={language}&country={country}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json()['articles']
                
                # Process articles
                processed_articles = []
                for article in articles:
                    if article['title'] and article['description']:
                        # Calculate sentiment scores
                        title_sentiment = self.sentiment_analyzer.polarity_scores(article['title'])
                        desc_sentiment = self.sentiment_analyzer.polarity_scores(article['description'])
                        
                        processed_articles.append({
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'publishedAt': article['publishedAt'],
                            'source': article['source']['name'],
                            'title_sentiment': title_sentiment['compound'],
                            'desc_sentiment': desc_sentiment['compound'],
                            'overall_sentiment': (title_sentiment['compound'] + desc_sentiment['compound']) / 2
                        })
                
                news_df = pd.DataFrame(processed_articles)
                if not news_df.empty:
                    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
                    self.news_data = news_df
                    logger.info(f"Fetched {len(news_df)} news articles")
                    return news_df
                else:
                    logger.warning("No news articles found")
                    return pd.DataFrame()
            else:
                logger.error(f"Failed to fetch news: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return pd.DataFrame()

    def fetch_market_data(self, symbols=['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'], period='3mo'):
        """Fetch market data using yfinance with flattened columns and a longer period."""
        logger.info(f"Fetching market data for {symbols}")
        
        all_data = []
        
        for symbol in symbols:
            stock_data = yf.download(symbol, period=period)
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            
            if stock_data.empty:
                logger.warning(f"No data for {symbol}. Skipping...")
                continue
            
            stock_data['Symbol'] = symbol
            stock_data['Return'] = stock_data['Close'].pct_change()
            stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
            stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
            stock_data['Volatility'] = stock_data['Close'].rolling(window=5).std()
            
            stock_data['Target'] = stock_data['Return'].shift(-1) > 0
            
            all_data.append(stock_data)
        
        if all_data:
            all_data = pd.concat(all_data)
            all_data = all_data.dropna()
            self.market_data = all_data
            logger.info(f"Fetched market data with {len(all_data)} entries")
            return all_data
        else:
            logger.warning("No market data fetched")
            return pd.DataFrame()

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def merge_data(self):
        """Merge market and news data based on dates"""
        logger.info("Merging market and news data")

        market_data = self.market_data.copy().reset_index()
        market_data['Date'] = pd.to_datetime(market_data['Date']).dt.date

        news_agg = self.news_data.groupby(pd.Grouper(key='publishedAt', freq='D')).agg({
            'overall_sentiment': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'news_count'})
        news_agg.index = news_agg.index.date
        
        merged = pd.merge(
            market_data, 
            news_agg, 
            left_on='Date', 
            right_index=True, 
            how='left'
        )
        
        merged['overall_sentiment'] = merged['overall_sentiment'].fillna(0)
        merged['news_count'] = merged['news_count'].fillna(0)
        
        logger.info(f"Merged data shape: {merged.shape}")
        return merged

    def preprocess_data(self, data):
        """Preprocess data for model training"""
        logger.info("Preprocessing data for model training")
        
        if data.empty:
            logger.warning("Cannot preprocess: data is empty")
            return None, None
        
        try:
            features = [
                'Open', 'High', 'Low', 'Close', 'Volume', 
                'Return', 'MA_5', 'MA_20', 'RSI', 'Volatility',
                'overall_sentiment', 'news_count'
            ]
            
            X = data[features].copy()
            y = data['Target'].copy()
            
            X = X.dropna()
            y = y.loc[X.index]
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            logger.info(f"Preprocessed data shape: X={X_scaled.shape}, y={y.shape}")
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None, None

    def train_model(self):
        """Train the prediction model"""
        logger.info("Training investment prediction model")
        
        try:
            merged_data = self.merge_data()
            X, y = self.preprocess_data(merged_data)
            
            if X is None or y is None:
                logger.warning("Cannot train model: preprocessing failed")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            logger.info(f"Model trained - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
            
            self.model = model
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def get_news_for_symbol(self, symbol):
        """Return news articles that mention the symbol using related keywords"""
        stock_keywords = {
            'SPY': ['SPY', 'S&P'],
            'QQQ': ['QQQ', 'Nasdaq'],
            'AAPL': ['Apple', 'AAPL'],
            'MSFT': ['Microsoft', 'MSFT'],
            'AMZN': ['Amazon', 'AMZN'],
            'GOOGL': ['Google', 'Alphabet', 'GOOGL'],
            'TSLA': ['Tesla', 'TSLA']
        }
        keywords = stock_keywords.get(symbol, [symbol])
        pattern = '|'.join(keywords)
        mask = self.news_data['title'].str.contains(pattern, case=False, na=False) | \
               self.news_data['description'].str.contains(pattern, case=False, na=False)
        return self.news_data[mask]

    def generate_recommendations(self, symbols=['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']):
        """Generate investment recommendations"""
        logger.info("Generating investment recommendations")
        
        if self.model is None:
            logger.warning("Cannot generate recommendations: model not trained")
            return pd.DataFrame()
        
        try:
            recommendations = []
            
            for symbol in symbols:
                symbol_data = self.market_data[self.market_data['Symbol'] == symbol].iloc[-1:].copy()
                
                if symbol_data.empty:
                    continue
                
                latest_sentiment = self.news_data['overall_sentiment'].mean() if not self.news_data.empty else 0
                symbol_data['overall_sentiment'] = latest_sentiment
                symbol_data['news_count'] = len(self.news_data) if not self.news_data.empty else 0
                
                features = [
                    'Open', 'High', 'Low', 'Close', 'Volume', 
                    'Return', 'MA_5', 'MA_20', 'RSI', 'Volatility',
                    'overall_sentiment', 'news_count'
                ]
                X = symbol_data[features]
                X_scaled = pd.DataFrame(self.scaler.transform(X), columns=features, index=X.index)
                
                prediction_proba = self.model.predict_proba(X_scaled)[0]
                predicted_direction = "UP" if prediction_proba[1] > 0.5 else "DOWN"
                confidence = prediction_proba[1] if predicted_direction == "UP" else prediction_proba[0]
                
                current_price = symbol_data['Close'].values[0]
                rsi = symbol_data['RSI'].values[0]
                
                if predicted_direction == "UP" and confidence > 0.65:
                    action = "BUY"
                elif predicted_direction == "DOWN" and confidence > 0.65:
                    action = "SELL"
                else:
                    action = "HOLD"
                
                technical_strength = 0
                if symbol_data['MA_5'].values[0] > symbol_data['MA_20'].values[0]:
                    technical_strength += 1
                if rsi < 30:
                    technical_strength += 1
                elif rsi > 70: 
                    technical_strength -= 1
                
                sentiment_impact = "Positive" if latest_sentiment > 0.1 else "Negative" if latest_sentiment < -0.1 else "Neutral"
                
                recommendation = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'predicted_direction': predicted_direction,
                    'confidence': confidence,
                    'recommendation': action,
                    'rsi': rsi,
                    'sentiment': latest_sentiment,
                    'sentiment_impact': sentiment_impact,
                    'technical_strength': technical_strength,
                    'timestamp': datetime.now()
                }
                
                recommendations.append(recommendation)
            
            recommendations_df = pd.DataFrame(recommendations)
            self.investment_recommendations = recommendations_df
            
            logger.info(f"Generated {len(recommendations_df)} investment recommendations")
            return recommendations_df
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame()

    def visualize_data(self, symbol):
        """Visualize stock data with indicators and sentiment"""
        logger.info(f"Visualizing data for {symbol}")
        
        try:
            symbol_data = self.market_data[self.market_data['Symbol'] == symbol].copy()
            
            if symbol_data.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            ax1.plot(symbol_data.index, symbol_data['Close'], label='Close Price', color='blue')
            ax1.plot(symbol_data.index, symbol_data['MA_5'], label='5-day MA', linestyle='--', color='orange', alpha=0.7)
            ax1.plot(symbol_data.index, symbol_data['MA_20'], label='20-day MA', linestyle='--', color='green', alpha=0.7)
            ax1.set_title(f'{symbol} Price Movement')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(symbol_data.index, symbol_data['RSI'], color='purple', alpha=0.7)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            if 'overall_sentiment' in symbol_data.columns:
                ax3.bar(symbol_data.index, symbol_data['overall_sentiment'], color='blue', alpha=0.5)
                ax3.set_ylabel('News Sentiment')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing data: {str(e)}")
            return None

    def display_recommendations(self):
        """Display detailed investment recommendations along with charts and related news"""
        if self.investment_recommendations.empty:
            return "No recommendations available. Please generate recommendations first."
        
        rec_df = self.investment_recommendations.copy()
        rec_df = rec_df.sort_values(by='confidence', ascending=False)
        
        output = "INVESTMENT RECOMMENDATIONS\n"
        output += "=" * 50 + "\n\n"
        
        os.makedirs("charts", exist_ok=True)
        
        for _, row in rec_df.iterrows():
            output += f"SYMBOL: {row['symbol']}\n"
            output += f"RECOMMENDATION: {row['recommendation']} (Confidence: {row['confidence']:.2f})\n"
            output += f"Current Price: ${row['current_price']:.2f}\n"
            output += f"Predicted Direction: {row['predicted_direction']}\n"
            output += f"RSI: {row['rsi']:.2f} | Sentiment: {row['sentiment']:.2f} ({row['sentiment_impact']})\n"
            output += f"Technical Strength: {row['technical_strength']}\n"
            
            fig = self.visualize_data(row['symbol'])
            chart_file = f"charts/{row['symbol']}_chart.png"
            if fig:
                fig.savefig(chart_file)
                plt.close(fig)
                output += f"Chart: {chart_file}\n"
            else:
                output += "Chart: Not available\n"
            
            news_articles = self.get_news_for_symbol(row['symbol'])
            if not news_articles.empty:
                output += "Related News:\n"
                for idx, news in news_articles.iterrows():
                    output += f" - {news['title']}\n"
                    snippet = news['description'][:100] + "..." if len(news['description']) > 100 else news['description']
                    output += f"   {snippet}\n"
                    output += f"   URL: {news['url']}\n"
            else:
                output += "Related News: None found\n"
            
            output += "-" * 50 + "\n"
        
        return output

    def run_daily_analysis(self):
        """Run daily analysis and generate recommendations"""
        logger.info("Running daily investment analysis")
        
        self.fetch_news(days=7)
        self.fetch_market_data(period='3mo')
        
        success = self.train_model()
        
        if success:
            recommendations = self.generate_recommendations()
            
            if not recommendations.empty:
                logger.info("Daily analysis completed successfully")
                return self.display_recommendations()
            else:
                logger.warning("Daily analysis completed but no recommendations generated")
                return "Analysis completed but no recommendations could be generated."
        else:
            logger.error("Daily analysis failed")
            return "Analysis failed. Please check the logs for details."

    def schedule_daily_runs(self, time='16:30'):
        """Schedule daily analysis runs"""
        logger.info(f"Scheduling daily analysis for {time}")
        
        schedule.every().day.at(time).do(self.run_daily_analysis)
        
        logger.info(f"Daily analysis scheduled for {time}")
        print(f"Daily analysis scheduled for {time}. Press Ctrl+C to exit.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60) 
        except KeyboardInterrupt:
            logger.info("Scheduled runs stopped by user")
            print("Scheduled runs stopped.")

if __name__ == "__main__":
    investment_ai = InvestmentAI(news_api_key, alpha_vantage_key)
    
    investment_ai.fetch_news()
    investment_ai.fetch_market_data()
    
    investment_ai.train_model()
    
    recommendations = investment_ai.generate_recommendations()
    
    print(investment_ai.display_recommendations())
    
    # Schedule daily runs (uncomment to enable)
    # investment_ai.schedule_daily_runs()
