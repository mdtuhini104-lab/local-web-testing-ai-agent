"""
Knowledge Base Manager (knowledge_manager.py).
Manages persistent storage and retrieval of learned QA testing strategies,
edge-case patterns, and app-specific vulnerability knowledge across autonomous test runs.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("knowledge_manager")

DEFAULT_KNOWLEDGE_STORE = {
    "global_qa_strategies": [
        {
            "category": "form_validation",
            "pattern": "Empty required input fields",
            "strategy": "Submit form with empty inputs to verify explicit client/server validation feedback messages.",
        },
        {
            "category": "numeric_boundary",
            "pattern": "Price/Quantity inputs",
            "strategy": "Test negative numbers (-1), zero (0), and extreme values (9999999) to verify price calculation and inventory guardrails.",
        },
        {
            "category": "string_injection",
            "pattern": "Text/Search inputs",
            "strategy": "Inject special characters (<script>alert(1)</script>, ' OR '1'='1) to test XSS and SQL injection robustness.",
        },
        {
            "category": "authentication",
            "pattern": "Login/Register forms",
            "strategy": "Test invalid email formats and weak passwords to verify security error messaging.",
        },
        {
            "category": "accounting_precision",
            "pattern": "Invoice & Billing tables",
            "strategy": "Verify that subtotal + tax/VAT - discount equals grand total with 2-decimal precision.",
        },
        {
            "category": "workflow_sequence",
            "pattern": "Automotive Workshop Business Process Sequence",
            "strategy": "Enforce strict workflow sequence: Vehicle Inspection -> Quotation -> Job Card / Work Order -> Invoice & Billing. Flag any flow forcing Quotation creation BEFORE Vehicle Inspection as a Logical Process Mismatch.",
        },
    ],
    "app_specific_learnings": [],
    "discovered_vulnerabilities": [],
}


class QAKnowledgeManager:
    def __init__(self, store_path: str = "./storage/qa_knowledge_store.json"):
        self.store_path = os.path.abspath(store_path)
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.knowledge: Dict[str, Any] = self._load_store()
        logger.info(f"🧠 Persistent Knowledge Base Warm-Start: Loaded {len(self.knowledge.get('global_qa_strategies', []))} global strategies & {len(self.knowledge.get('step_experiences', []))} learned rules from '{self.store_path}'")

    def _load_store(self) -> Dict[str, Any]:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    store_data = json.load(f)
                    logger.info(f"⚡ Knowledge Store pre-warmed directly from disk at {self.store_path}")
                    return store_data
            except Exception as e:
                logger.warning(f"⚠️ Failed to read knowledge store at {self.store_path}: {e}")

        # Save default store if file missing or unparseable
        self._save_store(DEFAULT_KNOWLEDGE_STORE)
        return DEFAULT_KNOWLEDGE_STORE

    def _save_store(self, data: Optional[Dict[str, Any]] = None) -> None:
        to_save = data if data is not None else self.knowledge
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge store to {self.store_path}: {e}")

    def retrieve_relevant_strategies(self, page_title: str, url: str, elements: List[Dict[str, Any]]) -> List[str]:
        """
        Retrieves matching QA strategies based on current page elements and context.
        """
        strategies = []
        combined_text = (page_title + " " + url + " " + " ".join([e.get("text", "") for e in elements if e.get("text")])).lower()

        for item in self.knowledge.get("global_qa_strategies", []):
            strategies.append(f"[{item.get('category', 'general').upper()}] {item.get('strategy')}")

        for app_learn in self.knowledge.get("app_specific_learnings", []):
            if any(k in combined_text for k in app_learn.get("context_keywords", [])):
                strategies.append(f"[LEARNED LESSON] {app_learn.get('lesson')}")

        return strategies[:5]

    def record_learning(self, run_id: str, target_url: str, new_bug: str, strategy_used: str) -> None:
        """
        Persists a newly discovered bug and successful testing strategy into the knowledge store.
        """
        entry = {
            "run_id": run_id,
            "target_url": target_url,
            "bug": new_bug,
            "strategy": strategy_used,
        }
        self.knowledge.setdefault("discovered_vulnerabilities", []).append(entry)

        # Check if new app learning should be added
        if any(k in new_bug.lower() for k in ["form", "404", "500", "input", "validation"]):
            self.knowledge.setdefault("app_specific_learnings", []).append({
                "context_keywords": [target_url.lower()],
                "lesson": f"On target '{target_url}', issue encountered: {new_bug[:100]}. Ensure rigorous validation check on input fields.",
            })

        self._save_store()

    def record_step_experience(
        self,
        target_url: str,
        action: str,
        selector: str,
        outcome: str,
        root_cause: Optional[str] = None,
        fix_rule: Optional[str] = None,
    ) -> None:
        """
        Persists step action execution experience, selector resolutions, failure analysis, and RAG corrective rules.
        """
        entry = {
            "target_url": target_url,
            "action": action,
            "selector": selector,
            "outcome": outcome,
            "root_cause": root_cause or "Normal execution",
            "fix_rule": fix_rule or "Proceed with standard element interaction",
        }
        self.knowledge.setdefault("step_experiences", []).append(entry)

        if outcome.lower() in ["failed", "stalled", "error"] and fix_rule:
            self.knowledge.setdefault("app_specific_learnings", []).append({
                "context_keywords": [selector.lower(), target_url.lower()],
                "lesson": f"Corrective RAG Rule for '{selector}': {fix_rule}",
            })

        self._save_store()

    def retrieve_rag_insights(self, target_url: str, dom_snapshot: Dict[str, Any]) -> List[str]:
        """
        Retrieves past learned RAG experiences and corrective rules for current DOM snapshot.
        """
        url_low = target_url.lower()
        title_low = str(dom_snapshot.get("title", "")).lower()
        insights = []

        for exp in (self.knowledge.get("step_experiences", []) or [])[-20:]:
            if isinstance(exp, dict):
                if (exp.get("target_url") or "").lower() == url_low and exp.get("fix_rule"):
                    insights.append(f"🧠 RAG Past Insight [{(exp.get('action') or '').upper()}]: {exp.get('fix_rule')}")

        for app_learn in (self.knowledge.get("app_specific_learnings", []) or []):
            if isinstance(app_learn, dict):
                keywords = app_learn.get("context_keywords") or []
                if any(k in url_low or k in title_low for k in keywords):
                    insights.append(f"⚡ RAG Adaptive Rule: {app_learn.get('lesson')}")

        if not insights:
            insights.append("🧠 RAG Memory: Form exploration initialized. Complete mandatory input fields, textareas, and select dropdowns prior to form submission.")

        return list(dict.fromkeys(insights))[:4]


# Alias for backward compatibility
KnowledgeManager = QAKnowledgeManager
