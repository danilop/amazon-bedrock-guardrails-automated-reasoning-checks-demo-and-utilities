"""
Automated Reasoning Policy Test Runner using Bedrock Runtime ApplyGuardrail API
Tests guardrail enforcement based on test cases from an automated reasoning policy.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import boto3
from botocore.exceptions import ClientError

from utils.automated_reasoning_common import (
    TestResult,
    extract_automated_reasoning_results,
)
from utils.config import load_config, print_config_header


@dataclass
class TestCase:
    """Test case for automated reasoning policy"""

    expected_result: TestResult
    question: Optional[str] = None  # User question/query
    answer: Optional[str] = None  # Model answer/response

    def __post_init__(self):
        """Convert string expected result to enum"""
        if isinstance(self.expected_result, str):
            self.expected_result = TestResult(self.expected_result)


class AutomatedReasoningTester:
    """
    Test runner for automated reasoning policy using Bedrock ApplyGuardrail API
    """

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str = "DRAFT",
        region: str = "us-east-1",
    ):
        """
        Initialize the test runner

        Args:
            guardrail_id: The ID of the guardrail to test
            guardrail_version: Version of the guardrail (default: DRAFT)
            region: AWS region (default: us-east-1)
        """
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

    def apply_guardrail(
        self, question: Optional[str] = None, answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply guardrail to content using Bedrock Runtime API

        Args:
            question: User question/query (optional)
            answer: Model answer/response (optional)

        Returns:
            API response dictionary
        """
        try:
            # Build content array based on what's provided
            content = []

            if question and answer:
                # Question + Answer: question is query, answer is guard_content
                content.append({"text": {"text": question, "qualifiers": ["query"]}})
                content.append(
                    {"text": {"text": answer, "qualifiers": ["guard_content"]}}
                )
            elif answer:
                # Answer-only: answer is guard_content (no qualifiers needed)
                content.append({"text": {"text": answer}})
            elif question:
                # This case should not happen with the new JSON structure
                # but keeping for safety - treat question as guard_content
                content.append({"text": {"text": question}})

            if not content:
                return {"action": "ERROR", "error": "No content provided"}

            # For automated reasoning, source must always be OUTPUT
            source = "OUTPUT"

            print(f"Applying guardrail with source: {source}")
            print(f"Content: {json.dumps(content, indent=2)}")

            response = self.bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=content,
            )

            if "assessments" in response:
                for assessment in response["assessments"]:
                    if "automatedReasoningPolicy" in assessment:
                        if "findings" in assessment["automatedReasoningPolicy"]:
                            for finding in assessment["automatedReasoningPolicy"][
                                "findings"
                            ]:
                                print(f"Finding: {json.dumps(finding, indent=2)}")

            return response
        except ClientError as e:
            print(f"Error applying guardrail: {e}")
            return {"error": str(e)}

    def extract_automated_reasoning_results(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract automated reasoning policy results from guardrail response

        Args:
            response: The guardrail response

        Returns:
            List of automated reasoning findings
        """
        return extract_automated_reasoning_results(response)

    def run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Run a single test case

        Args:
            test_case: The test case to run

        Returns:
            Test results dictionary
        """
        results = {"expected_result": test_case.expected_result.value, "tests_run": []}

        # Print test case details (like in sample code)
        print("-" * 80)
        if test_case.question and test_case.answer:
            print(f"Question: {test_case.question}")
            print(f"Answer: {test_case.answer}")
        elif test_case.answer:
            print(f"Answer: {test_case.answer}")
        elif test_case.question:
            print(f"Question: {test_case.question}")
        print(f"Expected: {test_case.expected_result.value}")

        # Apply guardrail with appropriate content format
        response = self.apply_guardrail(
            question=test_case.question, answer=test_case.answer
        )

        # Extract automated reasoning results
        ar_findings = self.extract_automated_reasoning_results(response)

        # Print results (like in sample code)
        actual_result = None
        for finding in ar_findings:
            actual_result = finding.get("result")
            print(f"Result: {actual_result}")
            if finding.get("rules"):
                for rule in finding.get("rules", []):
                    print(f"- Rule ID: {rule.get('identifier', 'Unknown')}")
                    print(f"  Policy: {rule.get('policyVersionArn', 'Unknown')}")

        # Create a single test result entry
        if test_case.question and test_case.answer:
            test_type = "QUESTION_AND_ANSWER"
            content_summary = f"Q: {test_case.question} | A: {test_case.answer}"
        elif test_case.answer:
            test_type = "ANSWER_ONLY"
            content_summary = f"A: {test_case.answer}"
        else:
            test_type = "QUESTION_ONLY"
            content_summary = f"Q: {test_case.question}"

        results["tests_run"].append(
            {
                "type": test_type,
                "content": content_summary,
                "action": response.get("action", "ERROR"),
                "usage": response.get("usage", {}),
                "assessments": response.get("assessments", []),
                "automated_reasoning_result": actual_result,
                "automated_reasoning_findings": ar_findings,
            }
        )

        # Determine overall test result using automated reasoning result
        results["overall_action"] = actual_result or "ERROR"
        results["test_passed"] = self._evaluate_test_result(
            actual_result, test_case.expected_result
        )

        return results

    def _evaluate_test_result(
        self, actual_result: str, expected_result: TestResult
    ) -> bool:
        """
        Evaluate if the test passed based on expected result and actual automated reasoning result

        Args:
            actual_result: The actual automated reasoning result (VALID, INVALID, SATISFIABLE, etc.)
            expected_result: The expected test result

        Returns:
            True if test passed, False otherwise
        """
        if not actual_result:
            return False

        # Direct comparison for automated reasoning results
        return actual_result == expected_result.value

    def run_all_tests(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Run all test cases

        Args:
            test_cases: List of test cases to run

        Returns:
            Summary of all test results
        """
        results = {
            "guardrail_id": self.guardrail_id,
            "guardrail_version": self.guardrail_version,
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": [],
        }

        for idx, test_case in enumerate(test_cases, 1):
            print("\n" + "-" * 80)
            print(f"Running test case {idx}")
            test_result = self.run_test_case(test_case)
            results["test_results"].append(test_result)

            if test_result["test_passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

        results["success_rate"] = (
            (results["passed"] / results["total_tests"]) * 100
            if results["total_tests"] > 0
            else 0
        )

        return results


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from JSON file

    Args:
        file_path: Path to the JSON file containing test cases

    Returns:
        test_cases_data
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("test_cases", [])
    except FileNotFoundError:
        print(f"Error: Test cases file '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        sys.exit(1)


def main():
    """
    Main function to run automated reasoning policy tests
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Automated Reasoning Policy Tests")
    parser.add_argument(
        "--test",
        type=int,
        help="Run only a specific test by number (1-based index). Example: --test 9 to run the 9th test case",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(require_guardrail_id=True, load_test_cases_file=True)

    # Load test cases from JSON file
    test_cases_data = load_test_cases(config.test_cases_file)

    # Filter test cases if specific test number provided
    if args.test:
        if args.test < 1 or args.test > len(test_cases_data):
            print(
                f"Error: Test number {args.test} is out of range. Available tests: 1-{len(test_cases_data)}"
            )
            sys.exit(1)
        # Convert to 0-based index and select single test
        test_cases_data = [test_cases_data[args.test - 1]]
        print(f"Running single test: #{args.test}")
    else:
        print(f"Running all {len(test_cases_data)} tests")

    # Print configuration header
    print_config_header(
        config,
        "Automated Reasoning Policy Test Runner",
        {"Total Test Cases": len(test_cases_data)},
    )

    # Convert test data to TestCase objects
    test_cases = []
    for idx, test_data in enumerate(test_cases_data, 1):
        # Use all available fields from JSON
        test_cases.append(TestCase(**test_data))

    # Initialize tester
    tester = AutomatedReasoningTester(
        guardrail_id=config.guardrail_id,
        guardrail_version=config.guardrail_version,
        region=config.aws_region,
    )

    # Run tests (single test or all tests)
    if args.test:
        # Run single test
        print(f"Running Test #{args.test}")
        test_result = tester.run_test_case(
            test_cases[0]
        )  # Only one test case in the list
        print(f"\nTest Result: {'PASSED' if test_result['test_passed'] else 'FAILED'}")

        # Create a simple results structure for single test
        results = {
            "total_tests": 1,
            "passed": 1 if test_result["test_passed"] else 0,
            "failed": 0 if test_result["test_passed"] else 1,
            "success_rate": 100.0 if test_result["test_passed"] else 0.0,
            "test_results": [test_result],
            "single_test_number": args.test,  # Store the original test number
        }
    else:
        # Run all tests
        results = tester.run_all_tests(test_cases)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")

    # Print detailed results for failed tests
    if results["failed"] > 0:
        print("\n" + "-" * 40)
        print("FAILED TEST DETAILS")
        print("-" * 40)
        for idx, test_result in enumerate(results["test_results"], 1):
            if not test_result["test_passed"]:
                # Use the original test number if this is a single test run
                test_number = results.get("single_test_number", idx)
                print(f"\nTest Case {test_number}:")
                print(f"Expected: {test_result['expected_result']}")
                print(f"Actual Result: {test_result['overall_action']}")
                for test in test_result["tests_run"]:
                    print(f"  - {test['type']}")

    # Results summary already printed above

    return results


if __name__ == "__main__":
    # Note: Before running, make sure to:
    # 1. Set GUARDRAIL_ID environment variable or replace in code
    # 2. Ensure you have AWS credentials configured
    # 3. Ensure you have the necessary permissions to use bedrock-runtime:ApplyGuardrail
    # 4. Have automated_reasoning_test_cases.json file in the same directory

    # Example usage:
    # export GUARDRAIL_ID=your-guardrail-id
    # export GUARDRAIL_VERSION=DRAFT
    # export AWS_REGION=us-east-1
    # python automated-reasoning-checks-demo-static.py

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
