"""
Hallucination & Faithfulness Assessment Module.
Verifies factual consistency between LLM-generated answers and retrieved context,
detecting ungrounded claims and calculating hallucination risk.
"""

from typing import Dict, Any, List
import re


class HallucinationDetector:
    """
    Evaluates whether statements in an LLM generated response are grounded
    in the retrieved context documents.
    """

    def split_into_claims(self, text: str) -> List[str]:
        """
        Splits paragraph into individual declarative statements/claims.
        """
        # Split on period, exclamation, or newline while preserving sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        cleaned = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
        return cleaned

    def assess_faithfulness(self, context: str, generated_answer: str) -> Dict[str, Any]:
        """
        Scores factual alignment between context and answer.
        """
        claims = self.split_into_claims(generated_answer)
        if not claims:
            return {
                "faithfulness_score": 1.0,
                "hallucination_score": 0.0,
                "status": "PASS",
                "total_claims": 0,
                "grounded_claims": [],
                "unsupported_claims": []
            }

        context_lower = context.lower()
        context_words = set(re.findall(r"\b\w{4,}\b", context_lower))

        grounded = []
        unsupported = []

        for claim in claims:
            claim_lower = claim.lower()
            claim_words = re.findall(r"\b\w{4,}\b", claim_lower)

            if not claim_words:
                continue

            # Calculate word grounding ratio
            matched_count = sum(1 for w in claim_words if w in context_words)
            grounding_ratio = matched_count / float(len(claim_words))

            # If more than 40% of substantive words are grounded in context
            if grounding_ratio >= 0.40:
                grounded.append({
                    "claim": claim,
                    "grounding_ratio": round(grounding_ratio, 2)
                })
            else:
                unsupported.append({
                    "claim": claim,
                    "grounding_ratio": round(grounding_ratio, 2),
                    "reason": "Insufficient lexical or semantic support in retrieved context"
                })

        total = len(grounded) + len(unsupported)
        faithfulness = round(len(grounded) / max(1, total), 2)
        hallucination_score = round(1.0 - faithfulness, 2)

        status = "GROUNDED" if faithfulness >= 0.80 else "MODERATE_RISK" if faithfulness >= 0.50 else "HALLUCINATION_DETECTED"

        return {
            "faithfulness_percent": round(faithfulness * 100, 1),
            "hallucination_score": hallucination_score,
            "status": status,
            "total_claims_evaluated": total,
            "supported_claims_count": len(grounded),
            "unsupported_claims_count": len(unsupported),
            "grounded_claims": grounded,
            "unsupported_claims": unsupported
        }


def assess_hallucination(context: str, generated_answer: str) -> Dict[str, Any]:
    detector = HallucinationDetector()
    return detector.assess_faithfulness(context, generated_answer)
