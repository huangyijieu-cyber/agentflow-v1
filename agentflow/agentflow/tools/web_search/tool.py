import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from agentflow.tools.base import BaseTool
from agentflow.tools.page_summary import PageSummarizer, check_cancelled, extract_page_text
from agentflow.tools.network_retry import (
    MAX_NETWORK_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
    RETRYABLE_HTTP_STATUSES,
    retry_wait_seconds,
)
from agentflow.engine.factory import create_llm_engine

load_dotenv()

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Web_RAG_Search_Tool"

LIMITATION = f"""
The {TOOL_NAME} has several limitations: 
1) Requires valid URLs that are accessible and contain text content. 
2) May not work with JavaScript-heavy websites or those requiring authentication. 
3) Performance depends on the quality and relevance of the website content. 
4) May return incomplete or inaccurate information if the website content is not comprehensive. 
5) Long pages require multiple model calls and may take longer to process.
6) Summaries can omit details; preserve and verify important source evidence.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1) Use specific, targeted queries rather than broad questions.
2) Ensure the URL is accessible and contains relevant information.
3) Prefer websites with well-structured, text-rich content.
4) For complex queries, break them down into smaller, specific questions.
5) Verify important information from multiple sources when possible.
6) Use it as part of a multi-step research process rather than a single source of truth.
7) It is highly recommended to use this tool after calling other web-based tools (e.g., Google_Search_Tool, Wiki_Search_Tool, etc.) to get the real, accessible URLs.
"""


class Web_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Reads the full extracted text of a given URL and summarizes evidence relevant to the query. Long pages are read in segments and their evidence is merged.",
            tool_version="2.0.0",
            input_types={
                "query": "str - The search query for the website.",
                "url": "str - The URL of the website to retrieve information from.",
            },
            output_type="str - The answer to the user's query based on the information gathered from the website.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the exact mass in kg of the moon?", url="https://en.wikipedia.org/wiki/Moon")',
                    "description": "Retrieve information about the moon's mass from Wikipedia."
                },
                {
                    "command": 'execution = tool.execute(query="What are the main features of Python programming language?", url="https://www.python.org/about/apps/")',
                    "description": "Get information about Python features from the official website."
                }
            ],
            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )

        self.model_string = model_string
        print(f"Initializing Website Summary Tool with model: {self.model_string}")
        self._summarizer = None

        # NOTE: deterministic mode
        self.temperature = 0.0
        self.llm_engine = llm_engine or create_llm_engine(
            model_string=self.model_string, 
            temperature=self.temperature, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )

    def _get_website_content(self, url):
        """ 
        Extracts all text from the given URL.

        Parameters:
            url (str): The URL from which to extract text.

        Returns:
            str: The extracted text.
        """
        url = url.replace("arxiv.org/pdf", "arxiv.org/abs")

        # Add headers to mimic a real browser request
        # NOTE: this is a workaround to avoid being blocked by the website
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Session.get bypasses Wikipedia's global requests.get retry patch.
        # This keeps one Web RAG fetch within four HTTP requests even for wiki URLs.
        with requests.Session() as session:
            for attempt in range(MAX_NETWORK_RETRIES + 1):
                try:
                    response = session.get(url, headers=headers, timeout=10, verify=False)
                    if response.status_code in RETRYABLE_HTTP_STATUSES and attempt < MAX_NETWORK_RETRIES:
                        wait_time = retry_wait_seconds(
                            attempt,
                            status_code=response.status_code,
                            retry_after=response.headers.get("Retry-After"),
                        )
                        if wait_time <= MAX_RETRY_WAIT_SECONDS:
                            print(f"[Web RAG HTTP] {response.status_code}; retrying in {wait_time:.1f}s "
                                  f"({attempt + 1}/{MAX_NETWORK_RETRIES})")
                            response.close()
                            time.sleep(wait_time)
                            continue
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    return extract_page_text(soup)
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < MAX_NETWORK_RETRIES:
                        wait_time = retry_wait_seconds(attempt)
                        print(f"[Web RAG Network] {type(e).__name__}; retrying in {wait_time:.1f}s "
                              f"({attempt + 1}/{MAX_NETWORK_RETRIES})")
                        time.sleep(wait_time)
                        continue
                    return f"Error fetching URL: {str(e)}"
                except requests.RequestException as e:
                    return f"Error fetching URL: {str(e)}"
                except Exception as e:
                    return f"Error extracting text: {str(e)}"

    def execute(self, query, url):
        check_cancelled()
        website_content = self._get_website_content(url)
        check_cancelled()
        if website_content.startswith("Error"):
            return website_content
        if not website_content.strip():
            return "Error: No text content could be extracted from the website."
        if self._summarizer is None:
            self._summarizer = PageSummarizer(self.llm_engine)
        return self._summarizer.summarize(query, url, website_content)

    def get_metadata(self):
        metadata = super().get_metadata()
        # metadata['require_llm_engine'] = self.require_llm_engine
        return metadata


def test_web_data():
    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = "https://en.wikipedia.org/wiki/Moon"

    # 创建请求并添加 User-Agent
    req = urllib.request.Request(
        url, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    )

    response = urllib.request.urlopen(req, context=ctx)
    # print(f"response: {response}")
    # print(response.read().decode('utf-8'))

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/web_search
    python tool.py
    """

    test_web_data()

    import json

    # Example usage of the Web_Search_Tool
    # tool = Web_Search_Tool(model_string="gpt-4o-mini") # NOTE: strong LLM for tool
    # tool = Web_Search_Tool(model_string="gemini-1.5-flash") # NOTE: weak 8B model for tool
    # tool = Web_Search_Tool(model_string="dashscope") # NOTE: weak Qwen2.5-7B model for tool

    tool = Web_Search_Tool(model_string="vllm-Qwen3-30B-A3B-Instruct-2507")

    # Get tool metadata
    metadata = tool.get_metadata()
    # print("Tool Metadata:")
    # print(json.dumps(metadata, indent=4))

    examples = [
        {
            "query": "What is the exact mass in kg of the moon?", 
            "url": "https://en.wikipedia.org/wiki/Moon"
        }
        # {
        #     "query": "What is the capital of France?", 
        #     "url": "https://en.wikipedia.org/wiki/France"
        # },
        # {
        #     "query": "What are the main features of Python programming language?", 
        #     "url": "https://www.python.org/about/apps/"
        # }
    ]

    for example in examples:
        # try:
            # Execute the tool with example query
            execution = tool.execute(**example)
            print("\nGenerated Response:")
            print(execution)
            print("\n")
        # except Exception as e:
        #     print(f"Execution failed: {e}")


    print("\nDone!")
