import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from agentflow.tools.base import BaseTool

load_dotenv()

TOOL_NAME = "Yibu_Brave_Search_Tool"
DEFAULT_ENDPOINT = "https://yibuapi.com/brave/v1/web/search"

# NOTE: The remote environment egresses through a corporate proxy (netentsec) that
# terminates TLS with a self-signed certificate, causing SSLCertVerificationError.
# Following the project convention (see wikipedia_search/tool.py), disable cert
# verification for this tool's outbound search requests.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
LIMITATIONS = """
1. This tool is suitable for general web search through Yibu's Brave-compatible API.
2. It returns search result snippets, not full webpage content.
3. It may require Web_RAG_Search_Tool for reading a specific result URL in depth.
4. Search quality depends on Brave/Yibu availability and quota.
"""

BEST_PRACTICES = """
1. Use this tool for up-to-date web search and general information retrieval.
2. Use concise keyword queries for best recall.
3. Use count to request multiple candidates when the answer may require comparison.
4. Follow up with Web_RAG_Search_Tool on specific URLs when detailed page content is needed.
"""


class Brave_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "A web search tool powered by Yibu's Brave-compatible API. "
                "It returns Brave-like web search results with titles, URLs, and snippets."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "count": "int - Number of search results to return. Default is 10.",
                "country": "str - Optional country code, such as US, CN, JP.",
                "search_lang": "str - Optional search language, such as en, zh-hans.",
                "ui_lang": "str - Optional UI language.",
                "freshness": "str - Optional freshness filter supported by Brave-compatible API.",
            },
            output_type="str - Formatted web search results with titles, URLs, and descriptions.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?", count=5)',
                    "description": "Search recent public web information."
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society arXiv August 11 2016", count=10)',
                    "description": "Search for a specific article or web page."
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            },
        )

        self.api_key = os.getenv("BRAVE_API_KEY") or os.getenv("YIBU_BRAVE_API_KEY")
        if not self.api_key:
            raise Exception(
                "Yibu Brave API key not found. Please set BRAVE_API_KEY "
                "or YIBU_BRAVE_API_KEY."
            )

        self.endpoint = os.getenv("BRAVE_YIBU_BASE_URL", DEFAULT_ENDPOINT)
        self.max_retries = 3
        self.timeout = 20

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _extract_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        web_results = data.get("web", {}).get("results")
        if isinstance(web_results, list):
            return web_results

        organic_results = data.get("organic_results")
        if isinstance(organic_results, list):
            return organic_results

        discussion_results = data.get("discussions", {}).get("results")
        if isinstance(discussion_results, list):
            return discussion_results

        return []

    def _format_results(self, query: str, data: Dict[str, Any], count: int) -> str:
        results = self._extract_results(data)
        if not results:
            return f"No Brave search results found for query: {query}"

        lines = [f"Search results for: {query}", ""]
        for idx, item in enumerate(results[:count], start=1):
            title = self._as_text(item.get("title"))
            url = self._as_text(item.get("url") or item.get("link"))
            description = self._as_text(item.get("description") or item.get("snippet"))

            lines.append(f"[{idx}] {title or 'Untitled'}")
            if url:
                lines.append(f"URL: {url}")
            if description:
                lines.append(f"Description: {description}")

            extra_snippets = item.get("extra_snippets")
            if isinstance(extra_snippets, list) and extra_snippets:
                snippet_text = " ".join(self._as_text(s) for s in extra_snippets if s)
                if snippet_text:
                    lines.append(f"Extra snippets: {snippet_text}")

            lines.append("")

        return "\n".join(lines).strip()

    def _execute_search(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> str:
        params = {
            "q": query,
            "count": max(1, min(int(count), 20)),
        }
        if country:
            params["country"] = country
        if search_lang:
            params["search_lang"] = search_lang
        if ui_lang:
            params["ui_lang"] = ui_lang
        if freshness:
            params["freshness"] = freshness

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.endpoint,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,  
                    verify=False
                )
                response.raise_for_status()
                data = response.json()
                return self._format_results(query, data, params["count"])
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        return (
            f"Yibu Brave Search tried {self.max_retries} times but failed. "
            f"Last error: {last_error}"
        )

    def execute(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> str:
        return self._execute_search(
            query=query,
            count=count,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
            freshness=freshness,
        )


if __name__ == "__main__":
    tool = Brave_Search_Tool()
    print(tool.execute(query="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.?", count=3))
