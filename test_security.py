"""
Security Unit Tests for Sentinel Protection
"""

import os
import unittest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestPathTraversalSecurity(unittest.TestCase):
    def test_download_report_json_path_traversal(self):
        # Attempt path traversal
        response = client.get("/api/runs/..%2F..%2Fetc%2Fpasswd/download/json")
        self.assertIn(response.status_code, [400, 404])
        self.assertNotIn("root:", response.text)

    def test_download_report_markdown_path_traversal(self):
        response = client.get("/api/runs/..%2F..%2Fetc%2Fpasswd/download/markdown")
        self.assertIn(response.status_code, [400, 404])
        self.assertNotIn("root:", response.text)

    def test_download_report_docx_path_traversal(self):
        response = client.get("/api/runs/..%2F..%2Fetc%2Fpasswd/download/docx")
        self.assertIn(response.status_code, [400, 404])
        self.assertNotIn("root:", response.text)

    def test_download_batch_markdown_path_traversal(self):
        response = client.get("/api/batch/..%2F..%2Fetc%2Fpasswd/download/markdown")
        self.assertIn(response.status_code, [400, 404])
        self.assertNotIn("root:", response.text)


if __name__ == "__main__":
    unittest.main()
