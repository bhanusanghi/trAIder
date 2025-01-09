from analyst.trAIder import process_request
from typing import Optional


def run_analyst(ticker: str, timeframe: str, extra_query: Optional[str] = None):
    response = process_request(ticker, timeframe, extra_query)
    print(response)


run_analyst(
    "APTUSDT",
    "15m",
    "should I open a 3x long position on APTUSDT? what should be my tp and sl?",
)
