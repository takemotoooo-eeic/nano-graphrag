class TokenTracker:
    """Track token usage for LLM calls."""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def add_usage(self, token_counts):
        """Add token usage from one LLM call.

        Args:
            token_counts: A dictionary containing prompt_tokens, completion_tokens, total_tokens
        """
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)

        # If total_tokens is provided, use it directly; otherwise calculate the sum
        if "total_tokens" in token_counts:
            self.total_tokens += token_counts["total_tokens"]
        else:
            self.total_tokens += token_counts.get(
                "prompt_tokens", 0
            ) + token_counts.get("completion_tokens", 0)

        self.call_count += 1

    def get_usage(self):
        """Get current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def __str__(self):
        usage = self.get_usage()
        return (
            f"LLM call count: {usage['call_count']}, "
            f"Prompt tokens: {usage['prompt_tokens']}, "
            f"Completion tokens: {usage['completion_tokens']}, "
            f"Total tokens: {usage['total_tokens']}"
        )
