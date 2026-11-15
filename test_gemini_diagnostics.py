import unittest

from gemini_diagnostics import explain_gemini_error


class FakeAPIError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class GeminiDiagnosticTests(unittest.TestCase):
    def test_404_explains_model_access(self):
        message = explain_gemini_error(
            FakeAPIError(404, "model not found"),
            "gemini-3.5-flash-lite",
        )
        self.assertIn("not available", message)
        self.assertIn("model not found", message)

    def test_429_explains_project_level_quota(self):
        message = explain_gemini_error(
            FakeAPIError(429, "resource exhausted"),
            "gemini-3.5-flash-lite",
        )
        self.assertIn("same project does not reset project quotas", message)

    def test_network_errors_keep_the_original_detail(self):
        message = explain_gemini_error(
            TimeoutError("connection timed out"),
            "gemini-3.5-flash-lite",
        )
        self.assertIn("connection timed out", message)


if __name__ == "__main__":
    unittest.main()
