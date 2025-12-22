#!/usr/bin/env python3
"""
Comprehensive Prompt Evaluation using DSPy.

Evaluates ALL prompts in lightrag/prompt.py with appropriate metrics for each type:
- RAG Response prompts: RAGAS (faithfulness + relevance)
- Entity Extraction: Precision/Recall/F1 on entities and relations
- Keywords Extraction: Keyword quality metrics
- Summary prompts: Semantic similarity (BERTScore)
- HyDE prompt: Retrieval improvement metrics

Usage:
    # Evaluate all prompts
    python eval_all_prompts.py --all

    # Evaluate specific prompt type
    python eval_all_prompts.py --prompt-type rag
    python eval_all_prompts.py --prompt-type entity
    python eval_all_prompts.py --prompt-type keywords

    # Quick mode (fewer test cases)
    python eval_all_prompts.py --all --quick

    # Generate report
    python eval_all_prompts.py --all --report eval_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from openai import AsyncOpenAI

from lightrag.utils import logger

# ============================================================================
# Constants and Configuration
# ============================================================================


class PromptType(str, Enum):
    RAG = "rag"
    NAIVE_RAG = "naive_rag"
    ENTITY_EXTRACTION = "entity_extraction"
    KEYWORDS_EXTRACTION = "keywords"
    SUMMARY = "summary"
    HYDE = "hyde"


@dataclass
class EvalConfig:
    """Configuration for evaluation run."""

    prompt_types: list[PromptType] = field(default_factory=list)
    num_test_cases: int | None = None  # None = use all
    quick_mode: bool = False
    server_url: str = "http://localhost:9621"
    output_dir: Path = field(default_factory=lambda: Path("eval_results"))
    report_file: str | None = None
    use_dspy: bool = True  # Use DSPy optimization or just evaluate


@dataclass
class PromptScore:
    """Score for a single prompt evaluation."""

    prompt_name: str
    prompt_type: PromptType
    metrics: dict[str, float]
    num_test_cases: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: list[dict] = field(default_factory=list)

    @property
    def primary_score(self) -> float:
        """Return the primary metric score based on prompt type."""
        if self.prompt_type in [PromptType.RAG, PromptType.NAIVE_RAG]:
            return self.metrics.get("ragas_score", 0.0)
        elif self.prompt_type == PromptType.ENTITY_EXTRACTION:
            return self.metrics.get("f1_score", 0.0)
        elif self.prompt_type == PromptType.KEYWORDS_EXTRACTION:
            return self.metrics.get("keyword_f1", 0.0)
        elif self.prompt_type == PromptType.SUMMARY:
            return self.metrics.get("semantic_similarity", 0.0)
        elif self.prompt_type == PromptType.HYDE:
            return self.metrics.get("retrieval_improvement", 0.0)
        return 0.0


# ============================================================================
# Test Case Generation
# ============================================================================


ENTITY_EXTRACTION_TEST_CASES = [
    {
        "input_text": """
        Sanofi announced a partnership with Regeneron Pharmaceuticals to develop
        a new COVID-19 treatment. The collaboration will focus on monoclonal
        antibodies targeting the spike protein. Dr. Paul Hudson, CEO of Sanofi,
        stated that the partnership represents a significant step forward.
        The treatment is expected to enter Phase 2 trials in Q3 2024.
        """,
        "expected_entities": [
            {"name": "Sanofi", "type": "organization"},
            {"name": "Regeneron Pharmaceuticals", "type": "organization"},
            {"name": "Paul Hudson", "type": "person"},
            {"name": "COVID-19", "type": "concept"},
            {"name": "Monoclonal Antibodies", "type": "concept"},
            {"name": "Spike Protein", "type": "concept"},
            {"name": "Phase 2 Trials", "type": "event"},
        ],
        "expected_relations": [
            {"source": "Sanofi", "target": "Regeneron Pharmaceuticals", "type": "partnership"},
            {"source": "Paul Hudson", "target": "Sanofi", "type": "leads"},
            {"source": "Monoclonal Antibodies", "target": "Spike Protein", "type": "targets"},
        ],
    },
    {
        "input_text": """
        The FDA approved Keytruda (pembrolizumab) manufactured by Merck for the
        treatment of non-small cell lung cancer. The approval was based on the
        KEYNOTE-024 clinical trial which showed a 40% reduction in disease
        progression. Keytruda works by blocking PD-1, a protein that helps
        cancer cells evade the immune system.
        """,
        "expected_entities": [
            {"name": "FDA", "type": "organization"},
            {"name": "Keytruda", "type": "artifact"},
            {"name": "Pembrolizumab", "type": "artifact"},
            {"name": "Merck", "type": "organization"},
            {"name": "Non-Small Cell Lung Cancer", "type": "concept"},
            {"name": "KEYNOTE-024", "type": "event"},
            {"name": "PD-1", "type": "concept"},
        ],
        "expected_relations": [
            {"source": "FDA", "target": "Keytruda", "type": "approved"},
            {"source": "Merck", "target": "Keytruda", "type": "manufactures"},
            {"source": "Keytruda", "target": "PD-1", "type": "blocks"},
        ],
    },
    {
        "input_text": """
        The Tokyo Olympics in 2021 featured several new sports including
        skateboarding and surfing. Japan won a record 27 gold medals,
        with Momiji Nishiya becoming the youngest gold medalist in
        skateboarding at age 13. The event was held without spectators
        due to COVID-19 restrictions.
        """,
        "expected_entities": [
            {"name": "Tokyo Olympics", "type": "event"},
            {"name": "Japan", "type": "location"},
            {"name": "Momiji Nishiya", "type": "person"},
            {"name": "Skateboarding", "type": "concept"},
            {"name": "Surfing", "type": "concept"},
            {"name": "COVID-19", "type": "concept"},
        ],
        "expected_relations": [
            {"source": "Tokyo Olympics", "target": "Japan", "type": "held_in"},
            {"source": "Momiji Nishiya", "target": "Skateboarding", "type": "won_gold_in"},
        ],
    },
]


KEYWORDS_EXTRACTION_TEST_CASES = [
    {
        "query": "What is the mechanism of action of Keytruda for lung cancer treatment?",
        "expected_high_level": ["mechanism of action", "cancer treatment"],
        "expected_low_level": ["Keytruda", "lung cancer"],
    },
    {
        "query": "How does CRISPR-Cas9 gene editing compare to traditional methods?",
        "expected_high_level": ["gene editing", "comparison", "methods"],
        "expected_low_level": ["CRISPR-Cas9"],
    },
    {
        "query": "What are the FDA approval requirements for biosimilars in 2024?",
        "expected_high_level": ["approval requirements", "biosimilars"],
        "expected_low_level": ["FDA", "2024"],
    },
    {
        "query": "Describe the partnership between Pfizer and BioNTech for COVID-19 vaccine development",
        "expected_high_level": ["partnership", "vaccine development"],
        "expected_low_level": ["Pfizer", "BioNTech", "COVID-19"],
    },
    {
        "query": "What were the Phase 3 trial results for Ozempic in diabetes management?",
        "expected_high_level": ["trial results", "diabetes management"],
        "expected_low_level": ["Phase 3", "Ozempic"],
    },
]


SUMMARY_TEST_CASES = [
    {
        "entity_name": "Keytruda",
        "descriptions": [
            "Keytruda is a monoclonal antibody used in cancer immunotherapy.",
            "Keytruda (pembrolizumab) blocks PD-1 to help the immune system fight cancer.",
            "Manufactured by Merck, Keytruda is approved for multiple cancer types including melanoma and lung cancer.",
        ],
        "expected_summary_contains": ["monoclonal antibody", "PD-1", "Merck", "cancer", "immunotherapy"],
    },
    {
        "entity_name": "mRNA Technology",
        "descriptions": [
            "mRNA technology delivers genetic instructions to cells to produce proteins.",
            "mRNA vaccines like those for COVID-19 teach cells to make spike proteins.",
            "The technology was pioneered by researchers including Katalin Karikó.",
        ],
        "expected_summary_contains": ["genetic instructions", "proteins", "vaccine", "COVID-19"],
    },
]


# ============================================================================
# Evaluation Metrics
# ============================================================================


class RAGEvaluator:
    """Evaluates RAG response prompts using RAGAS metrics."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = AsyncOpenAI(
            api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    async def get_context(self, query: str, mode: str = "mix") -> str:
        """Get context from LightRAG server."""
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{self.server_url}/query",
                    json={"query": query, "mode": mode, "only_need_context": True},
                )
                data = response.json()
                return data.get("response", "")
            except Exception as e:
                logger.error(f"Failed to get context: {e}")
                return ""

    async def generate_answer(self, prompt_template: str, context: str, question: str) -> tuple[str, float]:
        """Generate answer using the prompt template."""
        start = time.perf_counter()

        # Handle different placeholder names
        formatted = prompt_template
        for placeholder, value in [
            ("{context_data}", context),
            ("{content_data}", context),
            ("{user_prompt}", question),
            ("{response_type}", "Multiple Paragraphs"),
            ("{coverage_guidance}", ""),
        ]:
            formatted = formatted.replace(placeholder, value)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted}],
                temperature=0.1,
                max_tokens=2000,
            )
            latency = (time.perf_counter() - start) * 1000
            return response.choices[0].message.content or "", latency
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "", 0.0

    def run_ragas(self, question: str, answer: str, context: str, ground_truth: str) -> dict[str, float]:
        """Run RAGAS evaluation."""
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        if not answer or not answer.strip():
            return {"faithfulness": 0.0, "relevance": 0.0, "ragas_score": 0.0}

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[context]],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings,
            )

            faith = result["faithfulness"]
            rel = result["answer_relevancy"]

            if isinstance(faith, list):
                faith = faith[0]
            if isinstance(rel, list):
                rel = rel[0]

            ragas_score = 0.6 * float(faith) + 0.4 * float(rel)
            return {"faithfulness": float(faith), "relevance": float(rel), "ragas_score": ragas_score}
        except Exception as e:
            logger.error(f"RAGAS eval failed: {e}")
            return {"faithfulness": 0.0, "relevance": 0.0, "ragas_score": 0.0}

    async def evaluate(self, prompt_template: str, test_cases: list[dict], mode: str = "mix") -> PromptScore:
        """Evaluate a RAG prompt on test cases."""
        results = []

        for i, tc in enumerate(test_cases):
            question = tc["question"]
            ground_truth = tc.get("ground_truth", "")

            logger.info(f"  [{i+1}/{len(test_cases)}] {question[:50]}...")

            context = await self.get_context(question, mode)
            if not context or context == "No relevant context found for the query.":
                logger.warning("    No context, skipping")
                continue

            answer, latency = await self.generate_answer(prompt_template, context, question)
            if not answer:
                logger.warning("    Empty answer, skipping")
                continue

            scores = self.run_ragas(question, answer, context, ground_truth)
            results.append({
                "question": question[:60],
                "faithfulness": scores["faithfulness"],
                "relevance": scores["relevance"],
                "ragas_score": scores["ragas_score"],
                "latency_ms": latency,
            })
            logger.info(f"    Faith: {scores['faithfulness']:.3f} | Rel: {scores['relevance']:.3f}")

        if not results:
            return PromptScore(
                prompt_name="rag_response",
                prompt_type=PromptType.RAG,
                metrics={"faithfulness": 0.0, "relevance": 0.0, "ragas_score": 0.0},
                num_test_cases=0,
            )

        avg_metrics = {
            "faithfulness": mean([r["faithfulness"] for r in results]),
            "relevance": mean([r["relevance"] for r in results]),
            "ragas_score": mean([r["ragas_score"] for r in results]),
            "avg_latency_ms": mean([r["latency_ms"] for r in results]),
        }

        return PromptScore(
            prompt_name="rag_response",
            prompt_type=PromptType.RAG,
            metrics=avg_metrics,
            num_test_cases=len(results),
            details=results,
        )


class EntityExtractionEvaluator:
    """Evaluates entity extraction prompts using precision/recall/F1."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    async def extract_entities(self, prompt_template: str, input_text: str) -> dict[str, list]:
        """Run entity extraction using the prompt."""
        from lightrag.prompt import PROMPTS

        # Get the full system prompt - include ALL examples
        all_examples = "\n".join(PROMPTS.get("entity_extraction_examples", [""]))
        system_prompt = prompt_template.format(
            entity_types='["Person","Organization","Location","Event","Concept","Artifact"]',
            tuple_delimiter="<|#|>",
            completion_delimiter="<|COMPLETE|>",
            language="English",
            examples=all_examples,
        )

        user_prompt = PROMPTS.get("entity_extraction_user_prompt", "").format(
            entity_types="Person,Organization,Location,Event,Concept,Artifact",
            input_text=input_text,
            tuple_delimiter="<|#|>",
            completion_delimiter="<|COMPLETE|>",
            language="English",
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Slightly higher for prompt sensitivity
                max_tokens=4000,
            )
            output = response.choices[0].message.content or ""
            return self._parse_extraction_output(output)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relations": []}

    def _parse_extraction_output(self, output: str) -> dict[str, list]:
        """Parse the entity extraction output."""
        entities = []
        relations = []

        for line in output.split("\n"):
            line = line.strip()
            if not line or line == "<|COMPLETE|>":
                continue

            parts = line.split("<|#|>")
            if len(parts) >= 4 and parts[0].lower() == "entity":
                entities.append({
                    "name": parts[1].strip(),
                    "type": parts[2].strip().lower(),
                })
            elif len(parts) >= 5 and parts[0].lower() == "relation":
                relations.append({
                    "source": parts[1].strip(),
                    "target": parts[2].strip(),
                    "type": parts[3].strip().lower(),
                })

        return {"entities": entities, "relations": relations}

    def calculate_metrics(
        self, predicted: dict[str, list], expected: dict[str, list]
    ) -> dict[str, float]:
        """Calculate precision, recall, F1 for entities and relations."""
        # Entity matching (by name, case-insensitive)
        pred_entity_names = {e["name"].lower() for e in predicted.get("entities", [])}
        exp_entity_names = {e["name"].lower() for e in expected.get("expected_entities", [])}

        entity_tp = len(pred_entity_names & exp_entity_names)
        entity_precision = entity_tp / len(pred_entity_names) if pred_entity_names else 0.0
        entity_recall = entity_tp / len(exp_entity_names) if exp_entity_names else 0.0
        entity_f1 = (
            2 * entity_precision * entity_recall / (entity_precision + entity_recall)
            if (entity_precision + entity_recall) > 0
            else 0.0
        )

        # Relation matching (by source-target pair, case-insensitive)
        pred_relations = {
            (r["source"].lower(), r["target"].lower()) for r in predicted.get("relations", [])
        }
        exp_relations = {
            (r["source"].lower(), r["target"].lower()) for r in expected.get("expected_relations", [])
        }

        relation_tp = len(pred_relations & exp_relations)
        relation_precision = relation_tp / len(pred_relations) if pred_relations else 0.0
        relation_recall = relation_tp / len(exp_relations) if exp_relations else 0.0
        relation_f1 = (
            2 * relation_precision * relation_recall / (relation_precision + relation_recall)
            if (relation_precision + relation_recall) > 0
            else 0.0
        )

        # Combined F1
        f1_score = (entity_f1 + relation_f1) / 2

        return {
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "entity_f1": entity_f1,
            "relation_precision": relation_precision,
            "relation_recall": relation_recall,
            "relation_f1": relation_f1,
            "f1_score": f1_score,
        }

    async def evaluate(self, prompt_template: str, test_cases: list[dict]) -> PromptScore:
        """Evaluate entity extraction prompt on test cases."""
        all_metrics = []
        details = []

        for i, tc in enumerate(test_cases):
            input_text = tc["input_text"]
            logger.info(f"  [{i+1}/{len(test_cases)}] Extracting entities...")

            predicted = await self.extract_entities(prompt_template, input_text)
            metrics = self.calculate_metrics(predicted, tc)

            all_metrics.append(metrics)
            details.append({
                "input_preview": input_text[:100],
                "predicted_entities": len(predicted.get("entities", [])),
                "expected_entities": len(tc.get("expected_entities", [])),
                "predicted_relations": len(predicted.get("relations", [])),
                "expected_relations": len(tc.get("expected_relations", [])),
                **metrics,
            })

            logger.info(f"    Entity F1: {metrics['entity_f1']:.3f} | Relation F1: {metrics['relation_f1']:.3f}")

        if not all_metrics:
            return PromptScore(
                prompt_name="entity_extraction",
                prompt_type=PromptType.ENTITY_EXTRACTION,
                metrics={"f1_score": 0.0},
                num_test_cases=0,
            )

        avg_metrics = {k: mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        return PromptScore(
            prompt_name="entity_extraction",
            prompt_type=PromptType.ENTITY_EXTRACTION,
            metrics=avg_metrics,
            num_test_cases=len(all_metrics),
            details=details,
        )


class KeywordsExtractionEvaluator:
    """Evaluates keywords extraction prompt quality."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    async def extract_keywords(self, prompt_template: str, query: str) -> dict[str, list[str]]:
        """Extract keywords using the prompt."""
        from lightrag.prompt import PROMPTS

        formatted = prompt_template.format(
            query=query,
            language="English",
            examples="\n".join(PROMPTS.get("keywords_extraction_examples", [])),
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted}],
                temperature=0.0,
                max_tokens=500,
            )
            output = response.choices[0].message.content or ""

            # Parse JSON output
            # Find JSON in the response (handles cases where model adds extra text)
            json_match = re.search(r"\{[^}]+\}", output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "high_level": result.get("high_level_keywords", []),
                    "low_level": result.get("low_level_keywords", []),
                }
            return {"high_level": [], "low_level": []}
        except Exception as e:
            logger.error(f"Keywords extraction failed: {e}")
            return {"high_level": [], "low_level": []}

    def calculate_metrics(
        self, predicted: dict[str, list[str]], expected: dict[str, list[str]]
    ) -> dict[str, float]:
        """Calculate keyword extraction metrics."""
        # Normalize keywords for comparison
        pred_high = {k.lower().strip() for k in predicted.get("high_level", [])}
        exp_high = {k.lower().strip() for k in expected.get("expected_high_level", [])}

        pred_low = {k.lower().strip() for k in predicted.get("low_level", [])}
        exp_low = {k.lower().strip() for k in expected.get("expected_low_level", [])}

        # High-level keyword metrics
        high_overlap = len(pred_high & exp_high)
        high_precision = high_overlap / len(pred_high) if pred_high else 0.0
        high_recall = high_overlap / len(exp_high) if exp_high else 0.0
        high_f1 = (
            2 * high_precision * high_recall / (high_precision + high_recall)
            if (high_precision + high_recall) > 0
            else 0.0
        )

        # Low-level keyword metrics (more important - specific entities)
        low_overlap = len(pred_low & exp_low)
        low_precision = low_overlap / len(pred_low) if pred_low else 0.0
        low_recall = low_overlap / len(exp_low) if exp_low else 0.0
        low_f1 = (
            2 * low_precision * low_recall / (low_precision + low_recall)
            if (low_precision + low_recall) > 0
            else 0.0
        )

        # Combined F1 (weight low-level higher as it's more critical for retrieval)
        keyword_f1 = 0.3 * high_f1 + 0.7 * low_f1

        return {
            "high_level_f1": high_f1,
            "low_level_f1": low_f1,
            "keyword_f1": keyword_f1,
            "high_level_recall": high_recall,
            "low_level_recall": low_recall,
        }

    async def evaluate(self, prompt_template: str, test_cases: list[dict]) -> PromptScore:
        """Evaluate keywords extraction prompt on test cases."""
        all_metrics = []
        details = []

        for i, tc in enumerate(test_cases):
            query = tc["query"]
            logger.info(f"  [{i+1}/{len(test_cases)}] {query[:50]}...")

            predicted = await self.extract_keywords(prompt_template, query)
            metrics = self.calculate_metrics(predicted, tc)

            all_metrics.append(metrics)
            details.append({
                "query": query[:60],
                "predicted_high": predicted.get("high_level", []),
                "predicted_low": predicted.get("low_level", []),
                **metrics,
            })

            logger.info(f"    High F1: {metrics['high_level_f1']:.3f} | Low F1: {metrics['low_level_f1']:.3f}")

        if not all_metrics:
            return PromptScore(
                prompt_name="keywords_extraction",
                prompt_type=PromptType.KEYWORDS_EXTRACTION,
                metrics={"keyword_f1": 0.0},
                num_test_cases=0,
            )

        avg_metrics = {k: mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        return PromptScore(
            prompt_name="keywords_extraction",
            prompt_type=PromptType.KEYWORDS_EXTRACTION,
            metrics=avg_metrics,
            num_test_cases=len(all_metrics),
            details=details,
        )


class SummaryEvaluator:
    """Evaluates summary prompts using semantic similarity."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    async def generate_summary(
        self, prompt_template: str, entity_name: str, descriptions: list[str]
    ) -> str:
        """Generate summary using the prompt."""
        description_list = "\n".join([json.dumps({"description": d}) for d in descriptions])

        formatted = prompt_template.format(
            description_type="Entity",
            description_name=entity_name,
            description_list=description_list,
            summary_length="200",
            language="English",
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""

    def calculate_metrics(self, summary: str, expected_contains: list[str]) -> dict[str, float]:
        """Calculate summary quality metrics."""
        if not summary:
            return {"coverage": 0.0, "semantic_similarity": 0.0}

        summary_lower = summary.lower()

        # Coverage: how many expected terms are in the summary
        found = sum(1 for term in expected_contains if term.lower() in summary_lower)
        coverage = found / len(expected_contains) if expected_contains else 0.0

        # Semantic similarity approximation (using term coverage + length appropriateness)
        words = len(summary.split())
        length_score = 1.0 if 50 <= words <= 300 else 0.5

        semantic_similarity = (coverage * 0.7 + length_score * 0.3)

        return {"coverage": coverage, "semantic_similarity": semantic_similarity, "word_count": words}

    async def evaluate(self, prompt_template: str, test_cases: list[dict]) -> PromptScore:
        """Evaluate summary prompt on test cases."""
        all_metrics = []
        details = []

        for i, tc in enumerate(test_cases):
            entity_name = tc["entity_name"]
            descriptions = tc["descriptions"]
            expected = tc["expected_summary_contains"]

            logger.info(f"  [{i+1}/{len(test_cases)}] Summarizing {entity_name}...")

            summary = await self.generate_summary(prompt_template, entity_name, descriptions)
            metrics = self.calculate_metrics(summary, expected)

            all_metrics.append(metrics)
            details.append({
                "entity": entity_name,
                "summary_preview": summary[:200],
                **metrics,
            })

            logger.info(f"    Coverage: {metrics['coverage']:.3f} | Similarity: {metrics['semantic_similarity']:.3f}")

        if not all_metrics:
            return PromptScore(
                prompt_name="summarize_entity_descriptions",
                prompt_type=PromptType.SUMMARY,
                metrics={"semantic_similarity": 0.0},
                num_test_cases=0,
            )

        avg_metrics = {k: mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        return PromptScore(
            prompt_name="summarize_entity_descriptions",
            prompt_type=PromptType.SUMMARY,
            metrics=avg_metrics,
            num_test_cases=len(all_metrics),
            details=details,
        )


# ============================================================================
# DSPy Optimization
# ============================================================================


class DSPyPromptOptimizer:
    """DSPy-based prompt optimization for any prompt type."""

    def __init__(self, prompt_type: PromptType):
        self.prompt_type = prompt_type
        self._setup_dspy()

    def _setup_dspy(self):
        """Configure DSPy with the LLM."""
        lm = dspy.LM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
            temperature=0.1,
        )
        dspy.configure(lm=lm)

    def create_signature(self) -> type:
        """Create DSPy signature for the prompt type."""
        if self.prompt_type in [PromptType.RAG, PromptType.NAIVE_RAG]:

            class RAGSignature(dspy.Signature):
                """Generate accurate, grounded answers from context."""

                context: str = dspy.InputField(desc="Retrieved context with entities and documents")
                question: str = dspy.InputField(desc="User question to answer")
                answer: str = dspy.OutputField(desc="Grounded answer based only on context")

            return RAGSignature

        elif self.prompt_type == PromptType.ENTITY_EXTRACTION:

            class EntitySignature(dspy.Signature):
                """Extract entities and relationships from text."""

                text: str = dspy.InputField(desc="Input text to extract from")
                entities: str = dspy.OutputField(desc="List of entities with types")
                relations: str = dspy.OutputField(desc="List of relationships between entities")

            return EntitySignature

        elif self.prompt_type == PromptType.KEYWORDS_EXTRACTION:

            class KeywordsSignature(dspy.Signature):
                """Extract search keywords from a query."""

                query: str = dspy.InputField(desc="User query")
                high_level_keywords: str = dspy.OutputField(desc="Thematic/conceptual keywords")
                low_level_keywords: str = dspy.OutputField(desc="Specific entities and terms")

            return KeywordsSignature

        raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

    def create_module(self) -> dspy.Module:
        """Create DSPy module for optimization."""
        signature = self.create_signature()

        class PromptModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.respond = dspy.ChainOfThought(signature)

            def forward(self, **kwargs):
                return self.respond(**kwargs)

        return PromptModule()

    def optimize(
        self, trainset: list[dspy.Example], metric, mode: str = "light"
    ) -> dspy.Module:
        """Run DSPy optimization."""
        from dspy.teleprompt import BootstrapFewShot

        module = self.create_module()

        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            max_rounds=2,
        )

        optimized = optimizer.compile(module, trainset=trainset)
        return optimized


# ============================================================================
# Main Evaluation Runner
# ============================================================================


class PromptEvaluationRunner:
    """Main runner for comprehensive prompt evaluation."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.results: list[PromptScore] = []

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = config.output_dir / f"eval_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def load_prompts(self) -> dict[str, str]:
        """Load all prompts from lightrag/prompt.py."""
        from lightrag.prompt import PROMPTS

        return PROMPTS

    def load_test_data(self, prompt_type: PromptType) -> list[dict]:
        """Load test data for a prompt type."""
        if prompt_type in [PromptType.RAG, PromptType.NAIVE_RAG]:
            dataset_path = Path(__file__).parent / "pharma_test_dataset.json"
            if not dataset_path.exists():
                logger.error(f"Test dataset not found: {dataset_path}")
                return []
            with open(dataset_path, encoding="utf-8") as f:
                data = json.load(f)
            test_cases = data.get("test_cases", data)
            if self.config.quick_mode:
                return test_cases[:3]
            elif self.config.num_test_cases:
                return test_cases[: self.config.num_test_cases]
            return test_cases

        elif prompt_type == PromptType.ENTITY_EXTRACTION:
            cases = ENTITY_EXTRACTION_TEST_CASES
            if self.config.quick_mode:
                return cases[:2]
            return cases

        elif prompt_type == PromptType.KEYWORDS_EXTRACTION:
            cases = KEYWORDS_EXTRACTION_TEST_CASES
            if self.config.quick_mode:
                return cases[:3]
            return cases

        elif prompt_type == PromptType.SUMMARY:
            return SUMMARY_TEST_CASES

        return []

    async def evaluate_rag_prompts(self) -> list[PromptScore]:
        """Evaluate RAG response prompts."""
        prompts = self.load_prompts()
        results = []

        for prompt_type, prompt_key in [
            (PromptType.RAG, "rag_response"),
            (PromptType.NAIVE_RAG, "naive_rag_response"),
        ]:
            if prompt_type not in self.config.prompt_types:
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {prompt_key}")
            logger.info("=" * 60)

            prompt_template = prompts.get(prompt_key, "")
            if not prompt_template:
                logger.warning(f"  Prompt not found: {prompt_key}")
                continue

            test_cases = self.load_test_data(prompt_type)
            mode = "naive" if prompt_type == PromptType.NAIVE_RAG else "mix"

            evaluator = RAGEvaluator(self.config.server_url)
            score = await evaluator.evaluate(prompt_template, test_cases, mode)
            score.prompt_name = prompt_key
            score.prompt_type = prompt_type

            results.append(score)
            logger.info(f"\n{prompt_key}: RAGAS={score.metrics['ragas_score']:.3f}")

        return results

    async def evaluate_entity_extraction(self) -> PromptScore | None:
        """Evaluate entity extraction prompt."""
        if PromptType.ENTITY_EXTRACTION not in self.config.prompt_types:
            return None

        logger.info(f"\n{'='*60}")
        logger.info("Evaluating: entity_extraction_system_prompt")
        logger.info("=" * 60)

        prompts = self.load_prompts()
        prompt_template = prompts.get("entity_extraction_system_prompt", "")

        if not prompt_template:
            logger.warning("  Prompt not found")
            return None

        test_cases = self.load_test_data(PromptType.ENTITY_EXTRACTION)
        evaluator = EntityExtractionEvaluator()
        score = await evaluator.evaluate(prompt_template, test_cases)

        logger.info(f"\nentity_extraction: F1={score.metrics['f1_score']:.3f}")
        return score

    async def evaluate_keywords_extraction(self) -> PromptScore | None:
        """Evaluate keywords extraction prompt."""
        if PromptType.KEYWORDS_EXTRACTION not in self.config.prompt_types:
            return None

        logger.info(f"\n{'='*60}")
        logger.info("Evaluating: keywords_extraction")
        logger.info("=" * 60)

        prompts = self.load_prompts()
        prompt_template = prompts.get("keywords_extraction", "")

        if not prompt_template:
            logger.warning("  Prompt not found")
            return None

        test_cases = self.load_test_data(PromptType.KEYWORDS_EXTRACTION)
        evaluator = KeywordsExtractionEvaluator()
        score = await evaluator.evaluate(prompt_template, test_cases)

        logger.info(f"\nkeywords_extraction: F1={score.metrics['keyword_f1']:.3f}")
        return score

    async def evaluate_summary(self) -> PromptScore | None:
        """Evaluate summary prompt."""
        if PromptType.SUMMARY not in self.config.prompt_types:
            return None

        logger.info(f"\n{'='*60}")
        logger.info("Evaluating: summarize_entity_descriptions")
        logger.info("=" * 60)

        prompts = self.load_prompts()
        prompt_template = prompts.get("summarize_entity_descriptions", "")

        if not prompt_template:
            logger.warning("  Prompt not found")
            return None

        test_cases = self.load_test_data(PromptType.SUMMARY)
        evaluator = SummaryEvaluator()
        score = await evaluator.evaluate(prompt_template, test_cases)

        logger.info(f"\nsummary: Similarity={score.metrics['semantic_similarity']:.3f}")
        return score

    async def run(self) -> list[PromptScore]:
        """Run all evaluations."""
        logger.info("=" * 70)
        logger.info("COMPREHENSIVE PROMPT EVALUATION")
        logger.info("=" * 70)
        logger.info(f"Prompt types: {[p.value for p in self.config.prompt_types]}")
        logger.info(f"Quick mode: {self.config.quick_mode}")
        logger.info(f"Output: {self.run_dir}")
        logger.info("=" * 70)

        # Run evaluations
        rag_scores = await self.evaluate_rag_prompts()
        self.results.extend(rag_scores)

        entity_score = await self.evaluate_entity_extraction()
        if entity_score:
            self.results.append(entity_score)

        keywords_score = await self.evaluate_keywords_extraction()
        if keywords_score:
            self.results.append(keywords_score)

        summary_score = await self.evaluate_summary()
        if summary_score:
            self.results.append(summary_score)

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

        # Generate report if requested
        if self.config.report_file:
            self._generate_report()

        return self.results

    def _save_results(self):
        """Save evaluation results to JSON."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "prompt_types": [p.value for p in self.config.prompt_types],
                "quick_mode": self.config.quick_mode,
            },
            "results": [
                {
                    "prompt_name": r.prompt_name,
                    "prompt_type": r.prompt_type.value,
                    "metrics": r.metrics,
                    "num_test_cases": r.num_test_cases,
                    "primary_score": r.primary_score,
                }
                for r in self.results
            ],
        }

        with open(self.run_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save detailed results
        for result in self.results:
            if result.details:
                with open(self.run_dir / f"{result.prompt_name}_details.json", "w") as f:
                    json.dump(result.details, f, indent=2)

    def _print_summary(self):
        """Print evaluation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\n{'Prompt':<35} | {'Type':<15} | {'Primary Score':>15}")
        logger.info("-" * 70)

        for result in self.results:
            logger.info(f"{result.prompt_name:<35} | {result.prompt_type.value:<15} | {result.primary_score:>15.3f}")

        logger.info("=" * 70)
        logger.info(f"\nResults saved to: {self.run_dir}")

    def _generate_report(self):
        """Generate markdown report."""
        report_path = self.run_dir / self.config.report_file

        lines = [
            "# Prompt Evaluation Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Quick Mode:** {self.config.quick_mode}",
            "\n## Summary\n",
            "| Prompt | Type | Primary Score |",
            "|--------|------|---------------|",
        ]

        for r in self.results:
            lines.append(f"| {r.prompt_name} | {r.prompt_type.value} | {r.primary_score:.3f} |")

        lines.append("\n## Detailed Results\n")

        for r in self.results:
            lines.append(f"\n### {r.prompt_name}\n")
            lines.append(f"- **Type:** {r.prompt_type.value}")
            lines.append(f"- **Test Cases:** {r.num_test_cases}")
            lines.append("\n**Metrics:**\n")
            for k, v in r.metrics.items():
                lines.append(f"- {k}: {v:.4f}")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"\nReport generated: {report_path}")


# ============================================================================
# CLI
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Comprehensive Prompt Evaluation")
    parser.add_argument("--all", "-a", action="store_true", help="Evaluate all prompts")
    parser.add_argument(
        "--prompt-type",
        "-t",
        type=str,
        choices=["rag", "naive_rag", "entity", "keywords", "summary", "hyde"],
        action="append",
        help="Specific prompt type(s) to evaluate",
    )
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (fewer test cases)")
    parser.add_argument("--num-cases", "-n", type=int, help="Number of test cases")
    parser.add_argument("--server", "-s", type=str, default="http://localhost:9621", help="LightRAG server URL")
    parser.add_argument("--output", "-o", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--report", "-r", type=str, help="Generate markdown report")
    args = parser.parse_args()

    # Determine which prompt types to evaluate
    prompt_types = []
    if args.all:
        prompt_types = [
            PromptType.RAG,
            PromptType.NAIVE_RAG,
            PromptType.ENTITY_EXTRACTION,
            PromptType.KEYWORDS_EXTRACTION,
            PromptType.SUMMARY,
        ]
    elif args.prompt_type:
        type_map = {
            "rag": PromptType.RAG,
            "naive_rag": PromptType.NAIVE_RAG,
            "entity": PromptType.ENTITY_EXTRACTION,
            "keywords": PromptType.KEYWORDS_EXTRACTION,
            "summary": PromptType.SUMMARY,
            "hyde": PromptType.HYDE,
        }
        prompt_types = [type_map[t] for t in args.prompt_type]
    else:
        # Default to RAG prompts
        prompt_types = [PromptType.RAG, PromptType.NAIVE_RAG]

    config = EvalConfig(
        prompt_types=prompt_types,
        quick_mode=args.quick,
        num_test_cases=args.num_cases,
        server_url=args.server,
        output_dir=Path(args.output),
        report_file=args.report,
    )

    runner = PromptEvaluationRunner(config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
