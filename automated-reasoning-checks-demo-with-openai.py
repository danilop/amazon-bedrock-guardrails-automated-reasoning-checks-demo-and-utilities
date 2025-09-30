"""
Interactive Automated Reasoning Policy Test Runner using OpenAI SDK with AWS Bedrock
Tests guardrail enforcement through interactive conversation with user input.
Guardrails are applied separately on input and output using ApplyGuardrail API.
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from openai import OpenAI
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
    Interactive test runner for automated reasoning policy using OpenAI SDK with AWS Bedrock
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
        self.region = region

        # Initialize Bedrock Runtime client for guardrail operations
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

        # Configure OpenAI client for AWS Bedrock
        self.openai_client = OpenAI()

        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = (
            "You are a customer support agent. You follow the provided refund policy."
        )
        self.refund_policy_path = DEFAULT_REFUND_POLICY_PATH
        self.refund_policy_text = None

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
                # Answer-only: answer is guard_content
                content.append({"text": {"text": answer}})
            elif question:
                # Question-only: question is query
                content.append({"text": {"text": question}})
            else:
                raise ValueError("At least one of question or answer must be provided")

            # Apply guardrail
            response = self.bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="OUTPUT" if answer else "INPUT",
                content=content,
            )

            return response

        except ClientError as e:
            print(f"Error applying guardrail: {e}")
            return {"error": str(e)}

    def extract_and_print_automated_reasoning_results(
        self, guardrail_response: Dict[str, Any]
    ) -> None:
        """
        Extract and print automated reasoning policy results from guardrail response

        Args:
            guardrail_response: The ApplyGuardrail API response
        """
        findings = extract_automated_reasoning_results(guardrail_response)
        if findings:
            print("\n📋 Automated Reasoning Findings:")
            print(json.dumps(findings, indent=2))

    def chat_with_model(self, user_message: str) -> Dict[str, Any]:
        """
        Use OpenAI SDK to chat with the model, applying guardrails separately

        Args:
            user_message: The user's input message

        Returns:
            Dictionary containing response and guardrail information
        """
        try:
            # Lazy load the refund policy text if not already loaded
            if not self.refund_policy_text:
                self.refund_policy_text = extract_pdf_text(self.refund_policy_path)

            print("\n=�  Applying input guardrail...")
            # Apply guardrail to input (question only)
            input_guardrail_response = self.apply_guardrail(question=user_message)

            # Check if input was blocked
            if (
                "action" in input_guardrail_response
                and input_guardrail_response["action"] == "GUARDRAIL_INTERVENED"
            ):
                print("=� Input blocked by guardrail")
                return {
                    "blocked": True,
                    "stage": "input",
                    "guardrail_response": input_guardrail_response,
                    "message": user_message,
                }

            print(" Input passed guardrail check")

            # Prepare messages for OpenAI API
            messages = []

            # Add system message with refund policy
            system_prompt_with_policy = (
                f"{self.system_prompt}\n\nREFUND POLICY:\n{self.refund_policy_text}"
            )
            messages.append({"role": "system", "content": system_prompt_with_policy})

            # Add conversation history
            messages.extend(self.conversation_history)

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            print(f"\n> Invoking model: {self.model_id}")

            # Call OpenAI API
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.model_id, messages=messages
                )

                # Extract assistant response
                assistant_message = completion.choices[0].message.content

                print("\n=�  Applying output guardrail...")
                # Apply guardrail to output (question + answer)
                output_guardrail_response = self.apply_guardrail(
                    question=user_message, answer=assistant_message
                )

                # Check if output was blocked
                if (
                    "action" in output_guardrail_response
                    and output_guardrail_response["action"] == "GUARDRAIL_INTERVENED"
                ):
                    print("=� Output blocked by guardrail")
                    return {
                        "blocked": True,
                        "stage": "output",
                        "guardrail_response": output_guardrail_response,
                        "message": assistant_message,
                        "usage": {
                            "input_tokens": completion.usage.prompt_tokens
                            if completion.usage
                            else 0,
                            "output_tokens": completion.usage.completion_tokens
                            if completion.usage
                            else 0,
                            "total_tokens": completion.usage.total_tokens
                            if completion.usage
                            else 0,
                        },
                    }

                print(" Output passed guardrail check")

                # Return successful response
                return {
                    "blocked": False,
                    "message": assistant_message,
                    "input_guardrail_response": input_guardrail_response,
                    "output_guardrail_response": output_guardrail_response,
                    "usage": {
                        "input_tokens": completion.usage.prompt_tokens
                        if completion.usage
                        else 0,
                        "output_tokens": completion.usage.completion_tokens
                        if completion.usage
                        else 0,
                        "total_tokens": completion.usage.total_tokens
                        if completion.usage
                        else 0,
                    },
                }

            except Exception as e:
                print(f"Error calling OpenAI API: {e}")
                return {"error": f"OpenAI API error: {str(e)}"}

        except FileNotFoundError:
            print(f"Error: Refund policy file not found at {self.refund_policy_path}")
            return {
                "error": f"Refund policy file not found at {self.refund_policy_path}"
            }
        except Exception as e:
            print(f"Error in chat processing: {e}")
            return {"error": str(e)}

    def process_response(self, response: Dict[str, Any], user_message: str) -> None:
        """
        Process and display the response

        Args:
            response: The response dictionary from chat_with_model
            user_message: The original user message
        """
        if "error" in response:
            print(f"L Error: {response['error']}")
            return

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Display assistant response
        if "message" in response:
            print(f"\n=� Assistant: {response['message']}")

            # Only add to conversation history if not blocked
            if not response.get("blocked", False):
                self.conversation_history.append(
                    {"role": "assistant", "content": response["message"]}
                )

        # Display if blocked
        if response.get("blocked", False):
            stage = response.get("stage", "unknown")
            print(f"\n=�  Guardrail intervened at {stage} stage - content blocked")

            # Extract automated reasoning findings from the blocking guardrail response
            if "guardrail_response" in response:
                self.extract_and_print_automated_reasoning_results(
                    response["guardrail_response"]
                )
        else:
            # Extract automated reasoning findings from both guardrail responses
            if "input_guardrail_response" in response:
                print("\n=� Input Guardrail Assessment:")
                self.extract_and_print_automated_reasoning_results(
                    response["input_guardrail_response"]
                )

            if "output_guardrail_response" in response:
                print("\n=� Output Guardrail Assessment:")
                self.extract_and_print_automated_reasoning_results(
                    response["output_guardrail_response"]
                )

        # Display usage information
        if "usage" in response:
            usage = response["usage"]
            print(
                f"\n=� Usage: Input tokens: {usage.get('input_tokens', 0)}, Output tokens: {usage.get('output_tokens', 0)}, Total: {usage.get('total_tokens', 0)}"
            )

    def show_help(self):
        """Display help information"""
        print(
            """
=' Interactive Automated Reasoning Policy Tester (OpenAI SDK) Help
===================================================================

This tool allows you to interactively test automated reasoning policies
using the OpenAI SDK with AWS Bedrock endpoint and separate guardrail checks.

Commands:
  /help    - Show this help message
  /quit    - Exit the program
  /exit    - Exit the program
  /bye     - Exit the program
  /clear   - Clear conversation history
  /status  - Show current configuration

Features:
  " Interactive conversation using OpenAI SDK with AWS Bedrock
  " Separate input and output guardrail enforcement
  " Automated reasoning policy testing
  " Conversation history maintained across interactions
  " Shows detailed automated reasoning findings
  " Token usage tracking

Configuration:
  Model: {model_id}
  Guardrail ID: {guardrail_id}
  Guardrail Version: {guardrail_version}
  AWS Region: {region}
  Endpoint: https://bedrock-runtime.{region}.amazonaws.com/openai/v1

Simply type your message and press Enter to interact with the AI model.
The guardrail will check both input and output for policy violations.
        """.format(
                model_id=self.model_id,
                guardrail_id=self.guardrail_id,
                guardrail_version=self.guardrail_version,
                region=self.region,
            )
        )

    def show_status(self):
        """Display current configuration status"""
        print(f"""
=� Current Configuration
========================
Model ID: {self.model_id}
Guardrail ID: {self.guardrail_id}
Guardrail Version: {self.guardrail_version}
AWS Region: {self.region}
OpenAI Endpoint: https://bedrock-runtime.{self.region}.amazonaws.com/openai/v1
Conversation Messages: {len(self.conversation_history)}
Policy Loaded: {self.refund_policy_text is not None}
""")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        # Note: We keep self.refund_policy_text loaded to avoid re-parsing the PDF
        print("=�  Conversation history cleared")

    def run_interactive_session(self):
        """
        Run the interactive session
        """
        print("=� Starting interactive session with OpenAI SDK + AWS Bedrock...")
        print(
            "Type your message and press Enter. Use /help for commands, /quit to exit."
        )
        print("-" * 60)

        while True:
            try:
                # Get user input
                user_input = input("\n=d You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["/quit", "/exit", "/bye"]:
                    print("=K Goodbye!")
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
                response = self.chat_with_model(user_input)
                self.process_response(response, user_input)
                print("-" * 40)

            except KeyboardInterrupt:
                print("\n\n=K Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"L Error: {e}")


def main():
    """
    Main function to run interactive automated reasoning policy tester with OpenAI SDK
    """
    # Load configuration
    config = load_config(require_guardrail_id=True, load_test_cases_file=False)

    # Print configuration header
    print_config_header(
        config,
        "Interactive Automated Reasoning Policy Tester (OpenAI SDK)",
        {
            "OpenAI Endpoint": f"https://bedrock-runtime.{config.aws_region}.amazonaws.com/openai/v1"
        },
    )

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
    # 2. Ensure you have AWS credentials configured OR set AWS_BEARER_TOKEN_BEDROCK
    # 3. Ensure you have the necessary permissions to use:
    #    - bedrock-runtime:InvokeModel (through OpenAI endpoint)
    #    - bedrock-runtime:ApplyGuardrail
    # 4. Have required packages installed: uv add openai python-dotenv pymupdf boto3

    # Example usage:
    # export GUARDRAIL_ID=your-guardrail-id
    # export GUARDRAIL_VERSION=DRAFT
    # export AWS_REGION=us-east-1
    # export MODEL_ID=openai.gpt-oss-20b-1:0
    # export AWS_BEARER_TOKEN_BEDROCK=your-api-key (optional)
    # python automated-reasoning-checks-demo-with-openai.py

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
