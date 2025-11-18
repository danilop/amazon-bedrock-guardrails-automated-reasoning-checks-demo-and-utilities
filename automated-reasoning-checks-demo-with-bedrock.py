"""
Interactive Automated Reasoning Policy Test Runner using Bedrock Converse API
Tests guardrail enforcement through interactive conversation with user input.
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, Any, List

import boto3
from botocore.exceptions import ClientError

from utils.automated_reasoning_common import extract_automated_reasoning_results
from utils.pdf_utils import extract_pdf_text, DEFAULT_REFUND_POLICY_PATH
from utils.config import load_config, print_config_header


@dataclass
class Message:
    """Conversation message"""

    role: str
    content: str


class InteractiveAutomatedReasoningTester:
    """
    Interactive test runner for automated reasoning policy using Bedrock Converse API
    """

    def __init__(
        self, guardrail_id: str, guardrail_version: str, region: str, model_id: str
    ):
        """
        Initialize the interactive test runner

        Args:
            guardrail_id: The ID of the guardrail to test
            guardrail_version: Version of the guardrail
            region: AWS region
            model_id: The model ID to use for conversation
        """
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.model_id = model_id
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        self.conversation_history: List[Message] = []
        self.system_prompt = (
            "You are a customer support agent. You follow the provided refund policy. Reply with max 10 words"
        )
        self.refund_policy_path = DEFAULT_REFUND_POLICY_PATH
        self.refund_policy_text = None

    def converse_with_guardrail(self, user_message: str) -> Dict[str, Any]:
        """
        Use Bedrock Converse API with guardrail to process user message

        Args:
            user_message: The user's input message

        Returns:
            Dictionary containing response and guardrail information
        """
        try:
            # Lazy load the refund policy text if not already loaded
            if not self.refund_policy_text:
                self.refund_policy_text = extract_pdf_text(self.refund_policy_path)

            # Prepare messages for Converse API
            messages = []

            # Add all conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": [{"text": msg.content}]})

            # Add current user message to conversation history and to messages
            self.conversation_history.append(Message(role="user", content=user_message))
            messages.append({"role": "user", "content": [{"text": user_message}]})

            print("\nInvoking model with guardrail...")
            print(f"Model: {self.model_id}")
            print(f"Guardrail ID: {self.guardrail_id}")
            print(f"Guardrail Version: {self.guardrail_version}")

            # Prepare system prompt with the refund policy included
            system_prompt_with_policy = (
                f"{self.system_prompt}\n\nREFUND POLICY:\n{self.refund_policy_text}"
            )
            system = [{"text": system_prompt_with_policy}]

            # Call Converse API with guardrail
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=messages,
                system=system,
#               inferenceConfig={"maxTokens": 2000, "temperature": 0.7, "topP": 0.9},
                guardrailConfig={
                    "guardrailIdentifier": self.guardrail_id,
                    "guardrailVersion": self.guardrail_version,
                    "trace": "enabled",
                },
            )

            return response
        except ClientError as e:
            print(f"Error in Converse API call: {e}")
            return {"error": str(e)}
        except FileNotFoundError:
            print(f"Error: Refund policy file not found at {self.refund_policy_path}")
            return {
                "error": f"Refund policy file not found at {self.refund_policy_path}"
            }

    def extract_and_print_automated_reasoning_results(
        self, response: Dict[str, Any]
    ) -> None:
        """
        Extract and print automated reasoning policy results from Converse response

        Args:
            response: The Converse API response
        """
        findings = extract_automated_reasoning_results(response)
        if findings:
            print("\n📋 Automated Reasoning Findings:")
            print(json.dumps(findings, indent=2))

    def process_response(self, response: Dict[str, Any]) -> None:
        """
        Process and display the response from Converse API - Always display response and findings

        Args:
            response: The Converse API response
        """
        if "error" in response:
            print(f"❌ Error: {response['error']}")
            return

        # Always extract and display assistant response (even if blocked)
        assistant_message = ""
        if "output" in response and "message" in response["output"]:
            for content in response["output"]["message"].get("content", []):
                if "text" in content:
                    assistant_message += content["text"]

        if assistant_message:
            print(f"\n🤖 Assistant: {assistant_message}")
            # Only add to conversation history if not blocked by guardrail
            if not (
                "stopReason" in response
                and response["stopReason"] == "guardrail_intervened"
            ):
                self.conversation_history.append(
                    Message(role="assistant", content=assistant_message)
                )

        # Check for guardrail intervention
        if (
            "stopReason" in response
            and response["stopReason"] == "guardrail_intervened"
        ):
            print("\n🛡️  Guardrail intervened - content blocked")

        # Extract and display automated reasoning findings
        self.extract_and_print_automated_reasoning_results(response)

        # Display usage information if available
        if "usage" in response:
            usage = response["usage"]
            print(
                f"\n📊 Usage: Input tokens: {usage.get('inputTokens', 0)}, Output tokens: {usage.get('outputTokens', 0)}"
            )

    def show_help(self):
        """Display help information"""
        print(
            """
🔧 Interactive Automated Reasoning Policy Tester Help
=====================================================

This tool allows you to interactively test automated reasoning policies
using Amazon Bedrock's Converse API with guardrails.

Commands:
  /help    - Show this help message
  /quit    - Exit the program
  /exit    - Exit the program
  /bye     - Exit the program
  /clear   - Clear conversation history
  /status  - Show current configuration

Features:
  • Interactive conversation with AI model
  • Real-time guardrail enforcement
  • Automated reasoning policy testing
  • Conversation history maintained across interactions
  • Always displays responses even when blocked
  • Shows detailed automated reasoning findings

Configuration:
  Model: {model_id}
  Guardrail ID: {guardrail_id}
  Guardrail Version: {guardrail_version}

Simply type your message and press Enter to interact with the AI model.
The guardrail will automatically check each interaction for policy violations.
        """.format(
                model_id=self.model_id,
                guardrail_id=self.guardrail_id,
                guardrail_version=self.guardrail_version,
            )
        )

    def show_status(self):
        """Display current configuration status"""
        print(f"""
📊 Current Configuration
=======================
Model ID: {self.model_id}
Guardrail ID: {self.guardrail_id}
Guardrail Version: {self.guardrail_version}
Conversation Messages: {len(self.conversation_history)}
Policy Loaded: {self.refund_policy_text is not None}
""")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        # Note: We keep self.refund_policy_text loaded to avoid re-parsing the PDF
        print("🗑️  Conversation history cleared")

    def run_interactive_session(self):
        """
        Run the interactive session
        """
        print("🚀 Starting interactive session...")
        print(
            "Type your message and press Enter. Use /help for commands, /quit to exit."
        )
        print("-" * 60)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["/quit", "/exit", "/bye"]:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "/help":
                    self.show_help()
                    continue
                elif user_input.lower() == "/clear":
                    self.clear_history()
                    continue
                elif user_input.lower() == "/status":
                    self.show_status()
                    continue

                # Process the user message
                print("-" * 40)
                response = self.converse_with_guardrail(user_input)
                self.process_response(response)
                print("-" * 40)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """
    Main function to run interactive automated reasoning policy tester
    """
    # Load configuration
    config = load_config(require_guardrail_id=True, load_test_cases_file=False)

    # Print configuration header
    print_config_header(config, "Interactive Automated Reasoning Policy Tester")

    # Initialize interactive tester
    tester = InteractiveAutomatedReasoningTester(
        guardrail_id=config.guardrail_id,
        guardrail_version=config.guardrail_version,
        region=config.aws_region,
        model_id=config.model_id,
    )

    # Run interactive session
    tester.run_interactive_session()


if __name__ == "__main__":
    # Note: Before running, make sure to:
    # 1. Set GUARDRAIL_ID environment variable or replace in code
    # 2. Ensure you have AWS credentials configured
    # 3. Ensure you have the necessary permissions to use bedrock-runtime:Converse
    # 4. Have python-dotenv installed: uv add python-dotenv

    # Example usage:
    # export GUARDRAIL_ID=your-guardrail-id
    # export GUARDRAIL_VERSION=DRAFT
    # export AWS_REGION=us-east-1
    # export MODEL_ID=amazon.nova-lite-v1:0
    # python automated-reasoning-checks-demo-with-bedrock.py

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
