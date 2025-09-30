"""
Common utilities for Automated Reasoning policy testing.

This module provides shared enums and result extraction logic used across
all AR demo implementations.
"""

from enum import Enum
from typing import Dict, Any, List


class TestResult(Enum):
    """Expected test results from automated reasoning"""

    VALID = "VALID"
    INVALID = "INVALID"
    SATISFIABLE = "SATISFIABLE"


class GuardrailAction(Enum):
    """Guardrail intervention actions"""

    GUARDRAIL_INTERVENED = "GUARDRAIL_INTERVENED"
    NONE = "NONE"


def extract_automated_reasoning_results(
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract automated reasoning policy results from guardrail response.

    This function handles different response structures from:
    - ApplyGuardrail API (assessments field)
    - Converse API (trace.guardrail.outputAssessments field)

    Args:
        response: The guardrail API response

    Returns:
        List of automated reasoning findings with result, finding details, and rules
    """
    findings = []

    # Handle ApplyGuardrail API response format
    if "assessments" in response:
        for assessment in response.get("assessments", []):
            if "automatedReasoningPolicy" in assessment:
                automated_reasoning_policy = assessment["automatedReasoningPolicy"]
                for finding in automated_reasoning_policy.get("findings", []):
                    findings.append(_parse_finding(finding))

    # Handle Converse API response format (trace.guardrail.outputAssessments)
    elif "trace" in response and "guardrail" in response["trace"]:
        guardrail_trace = response["trace"]["guardrail"]
        if "outputAssessments" in guardrail_trace:
            for assessments_list in guardrail_trace["outputAssessments"].values():
                for assessment in assessments_list:
                    if "automatedReasoningPolicy" in assessment:
                        automated_reasoning_policy = assessment[
                            "automatedReasoningPolicy"
                        ]
                        for finding in automated_reasoning_policy.get("findings", []):
                            findings.append(_parse_finding(finding))

    return findings


def _parse_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single AR finding to extract result type and relevant information.

    Args:
        finding: A single finding from the automated reasoning policy

    Returns:
        Dictionary with result type, full finding, and associated rules
    """
    # Determine the result type based on the finding structure
    result = None
    if "satisfiable" in finding:
        result = "SATISFIABLE"
    elif "valid" in finding:
        result = "VALID"
    elif "invalid" in finding:
        result = "INVALID"
    elif "translationAmbiguous" in finding:
        result = "AMBIGUOUS"

    return {
        "result": result,
        "finding": finding,
        "rules": finding.get("supportingRules", [])
        + finding.get("contradictingRules", []),
    }
