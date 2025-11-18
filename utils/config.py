"""
Configuration management for AR demos.

This module handles environment variable loading and configuration
for all AR demo scripts.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_OPENAI_MODEL_ID = "openai.gpt-oss-20b-1:0"
DEFAULT_MODEL_ID = "us.amazon.nova-lite-v1:0"
DEFAULT_GUARDRAIL_VERSION = "DRAFT"
DEFAULT_TEST_CASES_FILE = "automated_reasoning_test_cases.json"


@dataclass
class ARConfig:
    """Configuration for AR demos"""

    guardrail_id: str
    guardrail_version: str
    aws_region: str
    model_id: str
    test_cases_file: Optional[str] = None


def load_config(
    require_guardrail_id: bool = True, load_test_cases_file: bool = False, use_openai_model: bool = False
) -> ARConfig:
    """
    Load configuration from environment variables.

    Args:
        require_guardrail_id: If True, prompts user if GUARDRAIL_ID is not set
        load_test_cases_file: If True, loads TEST_CASES_FILE from environment
        use_openai_model: If True, uses DEFAULT_OPENAI_MODEL_ID as default instead of DEFAULT_MODEL_ID

    Returns:
        ARConfig object with loaded configuration

    Raises:
        SystemExit: If user chooses not to continue with default guardrail ID
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get configuration from environment
    guardrail_id = os.getenv("GUARDRAIL_ID", "YOUR_GUARDRAIL_ID")
    guardrail_version = os.getenv("GUARDRAIL_VERSION", DEFAULT_GUARDRAIL_VERSION)
    aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    default_model = DEFAULT_OPENAI_MODEL_ID if use_openai_model else DEFAULT_MODEL_ID
    model_id = os.getenv("MODEL_ID", default_model)
    test_cases_file = (
        os.getenv("TEST_CASES_FILE", DEFAULT_TEST_CASES_FILE)
        if load_test_cases_file
        else None
    )

    # Validate guardrail ID if required
    if require_guardrail_id and guardrail_id == "YOUR_GUARDRAIL_ID":
        print(
            "\n⚠️  Warning: Using default guardrail ID. Set GUARDRAIL_ID environment variable or edit the script."
        )
        print("Example: export GUARDRAIL_ID=your-actual-guardrail-id")
        response = input("\nDo you want to continue with the default ID? (y/n): ")
        if response.lower() != "y":
            print("Exiting. Please set the GUARDRAIL_ID environment variable.")
            sys.exit(0)

    return ARConfig(
        guardrail_id=guardrail_id,
        guardrail_version=guardrail_version,
        aws_region=aws_region,
        model_id=model_id,
        test_cases_file=test_cases_file,
    )


def print_config_header(
    config: ARConfig, title: str, extra_info: Optional[dict] = None
):
    """
    Print a formatted configuration header.

    Args:
        config: ARConfig object
        title: Title for the header
        extra_info: Optional dictionary of extra information to display
    """
    print("=" * 80)
    print(title)
    print("=" * 80)
    if config.test_cases_file:
        print(f"Test Cases File: {config.test_cases_file}")
    print(f"Guardrail ID: {config.guardrail_id}")
    print(f"Guardrail Version: {config.guardrail_version}")
    print(f"AWS Region: {config.aws_region}")
    print(f"Model ID: {config.model_id}")

    if extra_info:
        for key, value in extra_info.items():
            print(f"{key}: {value}")

    print("=" * 80)
