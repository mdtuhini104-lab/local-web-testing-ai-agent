#!/usr/bin/env python3
"""
Core Setup & Diagnostics Verification Script
Validates Google Gemini Vision API packages and Playwright Chromium installation.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")


async def check_gemini() -> bool:
    print("🔍 Checking Google Gemini Vision SDK installation...")
    try:
        from google import genai
        print("  ✅ Google GenAI SDK (`google-genai`) is installed!")
    except ImportError:
        try:
            import google.generativeai as genai
            print("  ✅ Legacy Google GenerativeAI SDK (`google-generativeai`) is installed!")
        except ImportError:
            print("  ❌ Google Gemini SDK missing. Please run `pip install google-genai`.")
            return False

    if GEMINI_KEY:
        print(f"  🔑 GEMINI_API_KEY detected in environment! ({GEMINI_KEY[:6]}...)")
    else:
        print("  ℹ️ No GEMINI_API_KEY found in .env (You can enter it directly in the UI dashboard).")

    return True


async def check_playwright() -> bool:
    print("\n🔍 Checking Playwright Chromium installation...")
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--ignore-certificate-errors"],
            )
            page = await browser.new_page(ignore_https_errors=True)
            await page.goto("about:blank")
            await browser.close()
            print("  ✅ Playwright Chromium launched successfully!")
            return True
    except Exception as e:
        print(f"  ❌ Playwright check failed: {e}")
        print("     Please run: `playwright install chromium`")
        return False


async def main():
    print("=" * 65)
    print("🚀 Google Gemini Vision Autonomous Web QA Agent - Diagnostics")
    print("=" * 65)

    gemini_ok = await check_gemini()
    playwright_ok = await check_playwright()

    print("\n" + "=" * 65)
    if gemini_ok and playwright_ok:
        print("🎉 Environment verified! Google Gemini Vision QA Agent is ready.")
    else:
        print("⚠️ Environment setup check failed. Check recommendations above.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
