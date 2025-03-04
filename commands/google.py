import json
from duckduckgo_search import ddg
from Config import Config
from Commands import Commands

CFG = Config()

class google(Commands):
    def __init__(self):
        super().__init__()
        self.commands = {
            "Google Search": self.google_search,
            "Google Official Search": self.google_official_search
        }

    def google_search(self, query: str, num_results: int = 8) -> str:
        search_results = []
        if not query:
            return json.dumps(search_results)
        results = ddg(query, max_results=num_results)
        if not results:
            return json.dumps(search_results)
        for j in results:
            search_results.append(j)
        return json.dumps(search_results, ensure_ascii=False, indent=4)

    def google_official_search(self, query: str, num_results: int = 8) -> str | list[str]:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        try:
            # Get the Google API key and Custom Search Engine ID from the config file
            api_key = CFG.google_api_key
            custom_search_engine_id = CFG.custom_search_engine_id

            # Initialize the Custom Search API service
            service = build("customsearch", "v1", developerKey=api_key)

            # Send the search query and retrieve the results
            result = (
                service.cse()
                .list(q=query, cx=custom_search_engine_id, num=num_results)
                .execute()
            )

            # Extract the search result items from the response
            search_results = result.get("items", [])

            # Create a list of only the URLs from the search results
            search_results_links = [item["link"] for item in search_results]

        except HttpError as e:
            # Handle errors in the API call
            error_details = json.loads(e.content.decode())

            # Check if the error is related to an invalid or missing API key
            if error_details.get("error", {}).get(
                "code"
            ) == 403 and "invalid API key" in error_details.get("error", {}).get(
                "message", ""
            ):
                return "Error: The provided Google API key is invalid or missing."
            else:
                return f"Error: {e}"

        # Return the list of search result URLs
        return search_results_links
