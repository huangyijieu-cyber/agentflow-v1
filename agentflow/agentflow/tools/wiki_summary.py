"""Bound the combined Wikipedia evidence, including its source metadata."""
import json

from agentflow.tools.page_summary import PageSummarizer, check_cancelled


WIKI_RESULT_TOKENS = 2048


class WikiResultSummarizer(PageSummarizer):
    def _prompt(self, query, url, text, stage, output_tokens):
        return super()._prompt(query, url, text, stage, output_tokens) + (
            "\nThe material contains Wikipedia page summaries, not raw pages. "
            "Combine evidence across pages; cite the corresponding source IDs [1], [2], etc. "
            "Keep conflicting facts and information gaps. Treat page errors as retrieval "
            "failures, never as factual evidence. Do not repeat titles or URLs: the tool "
            "returns them separately in sources."
        )

    def _size(self, result):
        # Executor wraps one tool.execute result in a list; Memory uses str().
        # Also cover normal JSON serialization, excluding pretty-printed log whitespace.
        return max(self.budget.count(str([result])), self.budget.count(
            json.dumps([result], ensure_ascii=False)))

    def _error(self, message):
        # Fixed, short diagnostics do not throw a new rollout-level budget error.
        return {"error": message}

    def _plain_result(self, summary, sources):
        result = {"summary": summary, "sources": sources}
        if self._size(result) <= WIKI_RESULT_TOKENS:
            return result
        return self._error(
            "Wikipedia result exceeds the 2048-token budget; oversized query or "
            "source metadata was not returned."
        )

    def summarize_pages(self, query, pages, candidates):
        check_cancelled()
        if not pages:
            if not any(page.get("title") for page in candidates):
                message = next((page["error"] for page in candidates if page.get("error")),
                               f"No results found for query: {query}")
                return self._plain_result(message, [])
            sources = [{"title": page.get("title"), "url": page.get("url")}
                       for page in candidates]
            return self._plain_result("No relevant pages were selected. Candidates only; "
                                      "their contents were not summarized.", sources)

        sources = [{"id": i, "title": page.get("title"), "url": page.get("url")}
                   for i, page in enumerate(pages, 1)]
        overhead = self._size({"summary": "", "sources": sources})
        # Leave room for serialization and at least 256 tokens of evidence.
        available = WIKI_RESULT_TOKENS - overhead - 128
        if available < 256:
            return self._error("Wikipedia source metadata leaves insufficient room within "
                               "the 2048-token result budget; no partial summary was returned.")

        material = json.dumps([
            {"source": source, "evidence": page.get("retrieved_information"),
             "error": page.get("error")}
            for source, page in zip(sources, pages)
        ], ensure_ascii=False)
        self.summary_tokens = available
        # Ordinary input takes one merge call. Oversized input uses the shared
        # hierarchical path so evidence from the last page is also considered.
        try:
            summary = self.summarize(query, "", material)
            result = {"summary": summary, "sources": sources}
            if self._size(result) > WIKI_RESULT_TOKENS:
                # One bounded recompression for escaping overhead, byte-count
                # fallback, or a backend that did not honor max_tokens.
                self.summary_tokens = max(128, available // 2)
                result["summary"] = self.summarize(query, "", summary)
            if self._size(result) > WIKI_RESULT_TOKENS:
                return self._error("Wikipedia combined summary still exceeds 2048 tokens "
                                   "after recompression; no partial summary was returned.")
        except Exception as error:
            check_cancelled()  # Preserve executor cancellation rather than hiding it.
            print(f"[Wikipedia summary] combined summary failed: {type(error).__name__}: {error}")
            return self._error("Wikipedia combined summarization failed; no evidence summary "
                               "was returned. See the tool log for details.")
        print(f"[Wikipedia summary] final result={self._size(result)}/{WIKI_RESULT_TOKENS} "
              f"tokens; counter={getattr(self.budget, 'tokenizer_mode', 'test')}")
        return result
