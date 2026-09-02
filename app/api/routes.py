"""
REST API Routes for launching test runs, retrieving history, downloading reports, and querying Gemini model status.
"""

import asyncio
import datetime
import json
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent_runner import AgentRunner, generate_master_batch_report
from app.config import settings
from app.db import database
from app.api.websocket import ws_manager

router = APIRouter(prefix="/api")


class RunRequest(BaseModel):
    target_url: str = Field(..., example="http://localhost:3000")
    username: Optional[str] = Field(None, example="admin@example.com")
    password: Optional[str] = Field(None, example="secret123")
    max_steps: int = Field(200, ge=1, le=5000)
    model_name: Optional[str] = Field(None, example="microsoft/Florence-2-base")
    headless: bool = Field(False)


class AppTarget(BaseModel):
    target_url: str = Field(..., example="http://localhost:3000")
    username: Optional[str] = Field(None, example="admin@example.com")
    password: Optional[str] = Field(None, example="secret123")


class BatchRunRequest(BaseModel):
    apps: List[AppTarget] = Field(..., description="List of applications to test in parallel")
    max_steps: int = Field(200, ge=1, le=5000)
    model_name: Optional[str] = Field(None, example="microsoft/Florence-2-base")
    headless: bool = Field(False)


async def execute_agent_task(
    run_id: str,
    target_url: str,
    username: Optional[str],
    password: Optional[str],
    max_steps: int,
    model_name: str,
    headless: bool,
) -> Dict:
    """Background task wrapper executing AgentRunner and persisting results to DB."""
    await database.create_test_run(run_id, target_url, username)
    await ws_manager.broadcast({
        "type": "RUN_STARTED",
        "run_id": run_id,
        "target_url": target_url,
    })

    main_loop = asyncio.get_running_loop()

    def step_callback(data: Dict):
        # Broadcast step progress live via WebSocket with run_id context
        data["run_id"] = run_id
        current_step_num = data.get("step", 0)
        data["total_steps"] = current_step_num
        data["total_steps_executed"] = current_step_num

        main_loop.call_soon_threadsafe(
            main_loop.create_task,
            ws_manager.broadcast({
                "type": "STEP_UPDATE",
                "run_id": run_id,
                "step": current_step_num,
                "total_steps": current_step_num,
                "total_steps_executed": current_step_num,
                "data": data,
            })
        )
        # Also persist step into DB asynchronously
        main_loop.call_soon_threadsafe(
            main_loop.create_task,
            database.add_test_step(run_id, {
                "step_number": current_step_num,
                "action": data.get("action"),
                "target_selector": data.get("target_selector", ""),
                "reasoning": data.get("reasoning", ""),
                "screenshot_path": os.path.basename(data.get("screenshot_url", "")) if data.get("screenshot_url") else "",
                "observed_issues": data.get("issues", []),
                "ux_feedback": data.get("ux_feedback", []),
            })
        )

    def state_callback(state_msg: str):
        # Broadcast inner agent state updates live
        main_loop.call_soon_threadsafe(
            main_loop.create_task,
            ws_manager.broadcast({
                "type": "AGENT_STATE",
                "run_id": run_id,
                "message": state_msg,
            })
        )

    runner = AgentRunner(
        target_url=target_url,
        username=username,
        password=password,
        max_steps=max_steps,
        run_id=run_id,
        headless=headless,
        model_name=model_name or settings.LOCAL_MODEL_NAME,
        storage_dir=settings.STORAGE_DIR,
        step_callback=step_callback,
        state_callback=state_callback,
    )

    def _run_agent_in_thread():
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(runner.run())
        except Exception as e:
            print(f"⚠️ Runner thread error: {e}")
            return None
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    try:
        report_data = await asyncio.to_thread(_run_agent_in_thread)
        summary = (report_data or {}).get("summary", {}) if isinstance(report_data, dict) else {}
        comp_reason = str((summary or {}).get("completed_reason", "") or "")

        is_failed = any(term in comp_reason.lower() for term in ["failed", "exception", "error", "missing"])

        status = f"failed: {comp_reason}" if is_failed else "completed"
        ws_event = "RUN_FAILED" if is_failed else "RUN_COMPLETED"

        await database.update_test_run_status(
            run_id=run_id,
            status=status,
            duration_seconds=(summary or {}).get("duration_seconds", 0.0),
            total_steps=(summary or {}).get("total_steps", 0),
            overall_ux_rating=(summary or {}).get("overall_ux_rating", "N/A"),
            summary_data=summary,
        )

        await ws_manager.broadcast({
            "type": ws_event,
            "run_id": run_id,
            "target_url": target_url,
            "summary": summary,
            "error": comp_reason if is_failed else None,
        })
        return summary
    except Exception as e:
        err_msg = str(e)
        await database.update_test_run_status(run_id=run_id, status=f"failed: {err_msg}")
        await ws_manager.broadcast({
            "type": "RUN_FAILED",
            "run_id": run_id,
            "target_url": target_url,
            "error": err_msg,
        })
        return {"run_id": run_id, "target_url": target_url, "completed_reason": f"failed: {err_msg}", "critical_bugs": [err_msg]}


async def execute_batch_agent_tasks(batch_id: str, req: BatchRunRequest):
    """Executes multiple AgentRunners in parallel using asyncio.gather() and generates Master Batch Report."""
    await ws_manager.broadcast({
        "type": "BATCH_STARTED",
        "batch_id": batch_id,
        "total_apps": len(req.apps),
    })

    model = req.model_name or settings.LOCAL_MODEL_NAME

    async def _run_single(idx: int, app: AppTarget):
        # Generate isolated run_id per app
        clean_name = app.target_url.replace("http://", "").replace("https://", "").replace(":", "_").replace("/", "_")[:20]
        run_id = f"{batch_id}_app{idx+1}_{clean_name}"
        return await execute_agent_task(
            run_id=run_id,
            target_url=app.target_url,
            username=app.username,
            password=app.password,
            max_steps=req.max_steps,
            model_name=model,
            headless=req.headless,
        )

    # Run all apps concurrently
    summaries = await asyncio.gather(*[_run_single(idx, app) for idx, app in enumerate(req.apps)])
    
    # Generate Master Batch Audit Report
    master_report = generate_master_batch_report(batch_id, list(summaries), settings.STORAGE_DIR)

    await ws_manager.broadcast({
        "type": "BATCH_COMPLETED",
        "batch_id": batch_id,
        "master_report": master_report,
    })


active_tasks: Dict[str, asyncio.Task] = {}


@router.post("/runs")
async def start_test_run(req: RunRequest, background_tasks: BackgroundTasks):
    """Starts a new local autonomous test run powered by Local Vision."""
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model = req.model_name or settings.LOCAL_MODEL_NAME

    task = asyncio.create_task(
        execute_agent_task(
            run_id=run_id,
            target_url=req.target_url,
            username=req.username,
            password=req.password,
            max_steps=req.max_steps,
            model_name=model,
            headless=req.headless,
        )
    )
    active_tasks[run_id] = task

    return {
        "status": "started",
        "run_id": run_id,
        "message": f"Autonomous agent run [{run_id}] started for {req.target_url}",
    }


@router.post("/runs/batch")
async def start_batch_test_run(req: BatchRunRequest, background_tasks: BackgroundTasks):
    """Launches parallel autonomous test runs across multiple target applications simultaneously."""
    if not req.apps:
        raise HTTPException(status_code=400, detail="At least one target application must be provided.")

    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    task = asyncio.create_task(
        execute_batch_agent_tasks(
            batch_id=batch_id,
            req=req,
        )
    )
    active_tasks[batch_id] = task

    return {
        "status": "started",
        "batch_id": batch_id,
        "total_apps": len(req.apps),
        "message": f"Parallel batch audit [{batch_id}] started for {len(req.apps)} applications.",
    }


@router.get("/runs")
async def list_runs():
    """Returns list of past test runs."""
    runs = await database.get_test_runs()
    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    """Fetches details & step history of a specific run."""
    run_dict = await database.get_test_run_details(run_id)
    if not run_dict:
        raise HTTPException(status_code=404, detail="Test run not found")
    return run_dict


@router.get("/runs/{run_id}/download/json")
async def download_report_json(run_id: str):
    """Download JSON report file."""
    file_path = os.path.join(settings.STORAGE_DIR, "runs", run_id, "report.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="JSON report file not found")
    return FileResponse(file_path, media_type="application/json", filename=f"{run_id}_report.json")


@router.get("/runs/{run_id}/download/markdown")
async def download_report_markdown(run_id: str):
    """Download Markdown report file."""
    file_path = os.path.join(settings.STORAGE_DIR, "runs", run_id, "report.md")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Markdown report file not found")
    return FileResponse(file_path, media_type="text/markdown", filename=f"{run_id}_report.md")


@router.get("/runs/{run_id}/download/docx")
@router.get("/reports/{run_id}/docx")
async def download_report_docx(run_id: str):
    """Download Microsoft Word (.docx) report file with fallback generation."""
    possible_paths = [
        os.path.join(settings.STORAGE_DIR, "runs", run_id, f"{run_id}_audit_report.docx"),
        os.path.join(settings.STORAGE_DIR, "reports", f"{run_id}_audit_report.docx"),
        os.path.join("./storage/reports", f"{run_id}_audit_report.docx"),
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return FileResponse(
                p,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename=f"{run_id}_audit_report.docx"
            )

    # Try dynamic generation from report.json / summary.json / DB
    run_dir = os.path.join(settings.STORAGE_DIR, "runs", run_id)
    report_json_file = os.path.join(run_dir, "report.json")
    summary_file = os.path.join(run_dir, "summary.json")
    target_json = report_json_file if os.path.exists(report_json_file) else (summary_file if os.path.exists(summary_file) else None)

    summary_dict = {}
    steps_list = []

    if target_json and os.path.exists(target_json):
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            if isinstance(s_data, dict):
                summary_dict = s_data.get("summary", s_data)
                steps_list = s_data.get("steps", [])
        except Exception as e:
            pass

    # Fallback to DB if JSON loading yielded nothing
    if not summary_dict:
        try:
            run_data = await database.get_test_run_details(run_id)
            if run_data and isinstance(run_data, dict):
                summary_dict = run_data.get("summary") or {
                    "run_id": run_id,
                    "target_url": run_data.get("target_url", "N/A"),
                    "start_time": run_data.get("created_at", "N/A"),
                    "duration_seconds": run_data.get("duration_seconds", 0),
                    "total_steps": run_data.get("total_steps", 0),
                    "overall_ux_rating": run_data.get("overall_ux_rating", "N/A"),
                    "completed_reason": run_data.get("status", "Completed"),
                    "critical_bugs": [],
                }
                steps_list = run_data.get("steps", [])
        except Exception:
            pass

    # Final fallback if2 neither file nor DB entry exist
    if not summary_dict:
        summary_dict = {
            "run_id": run_id,
            "target_url": "N/A",
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": 0,
            "total_steps": len(steps_list),
            "overall_ux_rating": "N/A",
            "completed_reason": "Fallback Report Generated",
            "critical_bugs": [],
        }

    # Generate the DOCX file
    out_docx = possible_paths[0]
    os.makedirs(os.path.dirname(out_docx), exist_ok=True)
    try:
        from agent_runner import generate_docx_report_from_dict
        generate_docx_report_from_dict(summary_dict, out_docx, steps_list)
        return FileResponse(
            out_docx,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=f"{run_id}_audit_report.docx"
        )
    except Exception as err:
        # Fallback to minimal docx generation if template generation throws an error
        try:
            from docx import Document
            doc = Document()
            doc.add_heading(f"Audit Report - {run_id}", 0)
            doc.add_paragraph(f"Report Generation Note: Partial data available for {run_id}.")
            doc.save(out_docx)
            return FileResponse(
                out_docx,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename=f"{run_id}_audit_report.docx"
            )
        except Exception as final_err:
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(final_err)}")


@router.get("/models")
async def get_gemini_models():
    """Returns available local models and status."""
    local_models = [
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
        "google/paligemma-3b-pt-224",
    ]
    return {
        "status": "ok",
        "has_env_key": True, # For compatibility with old UI logic if any
        "models": local_models,
        "default_model": settings.LOCAL_MODEL_NAME,
    }


@router.get("/batch/{batch_id}")
async def get_batch_details(batch_id: str):
    """Fetches details & master report of a batch test run."""
    import json
    file_path = os.path.join(settings.STORAGE_DIR, "runs", f"{batch_id}_master_report.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Batch report not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/batch/{batch_id}/download/markdown")
async def download_batch_markdown(batch_id: str):
    """Download Master Batch Markdown report file."""
    file_path = os.path.join(settings.STORAGE_DIR, "runs", f"{batch_id}_master_report.md")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Master batch markdown report file not found")
    return FileResponse(file_path, media_type="text/markdown", filename=f"{batch_id}_master_report.md")
