import asyncio
import sys
import json
import os

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent_runner import AgentRunner

async def main():
    runner = AgentRunner(
        target_url="http://localhost:3000/login",
        username="admin",
        password="password123",
        max_steps=20,
        headless=True,
    )
    result = await runner.run()
    print(">>> QA Audit Execution Finished Successfully <<<")
    summary = result.get("summary") if isinstance(result, dict) else {}
    print("Overall Rating:", summary.get("overall_ux_rating", "N/A"))
    print("Completed Reason:", summary.get("completed_reason", "N/A"))
    print("Total Steps:", summary.get("total_steps", 0))
    print("Critical Bugs:", len(summary.get("critical_bugs", [])))
    print("Console Errors:", len(summary.get("all_console_errors", [])))
    print("Network Errors:", len(summary.get("all_network_errors", [])))
    print("Uncaught Exceptions:", len(summary.get("all_uncaught_exceptions", [])))

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
