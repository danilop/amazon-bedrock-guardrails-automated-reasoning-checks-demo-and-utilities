"""
Interactive Automated Reasoning Policy Test Runner using Strands Agents
Tests guardrail enforcement through interactive conversation with user input.
Uses Strands Agents with AWS Bedrock integration for AI interactions with guardrails.
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, Any

from strands import Agent
from strands.models import BedrockModel

from utils.pdf_utils import extract_pdf_text, DEFAULT_REFUND_POLICY_PATH
from utils.config import load_config, print_config_header


@dataclass
class Message:
    """Conversation message"""

    role: str
    content: str


class InteractiveAutomatedReasoningTester:
    """
    Interactive test runner for automated reasoning policy using Strands Agents
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

        self.refund_policy_path = DEFAULT_REFUND_POLICY_PATH
        self.refund_policy_text = None

        # Initialize Strands Agent with Bedrock model and guardrails
        self.bedrock_model = BedrockModel(
            model_id=model_id,
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            guardrail_trace="enabled",
            region_name=region,
            streaming=False,
        )

        # System prompt will be set after loading the refund policy
        self.agent = None

    def initialize_agent(self):
        """Initialize the Strands agent with system prompt including refund policy"""
        if not self.refund_policy_text:
            self.refund_policy_text = extract_pdf_text(self.refund_policy_path)

        system_prompt = f"""You are a customer support agent. You follow the provided refund policy.

REFUND POLICY:
{self.refund_policy_text}"""

        self.agent = Agent(system_prompt=system_prompt, model=self.bedrock_model)

    def chat_with_agent(self, user_message: str) -> Dict[str, Any]:
        """
        Use Strands Agent to process user message with guardrails

        Args:
            user_message: The user's input message

        Returns:
            Dictionary containing response and guardrail information
        """
        # Initialize agent if not already done
        if self.agent is None:
            self.initialize_agent()

        print("\n🔍 Invoking Strands Agent with guardrail...")
        print(f"Model: {self.model_id}")
        print(f"Guardrail ID: {self.guardrail_id}")
        print(f"Guardrail Version: {self.guardrail_version}")

        print(
            f"System prompt: {json.dumps(self.agent.system_prompt, indent=2, default=str)}"
        )
        print(f"Messages: {json.dumps(self.agent.messages, indent=2, default=str)}")
        print(f"User message: {user_message}")

        # Use the agent to process the message
        response = self.agent(user_message)

        print(f"Response: {json.dumps(response, indent=2, default=str)}")

        # Extract response information
        assistant_message = str(response)

        return {
            "message": assistant_message,
        }

    def show_help(self):
        """Display help information"""
        print(f"""
🔧 Interactive Automated Reasoning Policy Tester (Strands Agents) Help
======================================================================

This tool allows you to interactively test automated reasoning policies
using Strands Agents with AWS Bedrock guardrails integration.

Commands:
  /help    - Show this help message
  /quit    - Exit the program
  /exit    - Exit the program
  /bye     - Exit the program
  /clear   - Clear conversation history
  /status  - Show current configuration

Features:
  • Interactive conversation using Strands Agents
  • Integrated AWS Bedrock guardrail enforcement
  • Automated reasoning policy testing
  • Conversation history maintained across interactions
  • Shows detailed conversation flow and guardrail information

Configuration:
  Model: {self.model_id}
  Guardrail ID: {self.guardrail_id}
  Guardrail Version: {self.guardrail_version}
  AWS Region: {self.region}
  Interface: Strands Agents SDK

Simply type your message and press Enter to interact with the AI model.
The guardrail will automatically check interactions for policy violations.
        """)

    def show_status(self):
        """Display current configuration status"""
        print(f"""
📊 Current Configuration
========================
Model ID: {self.model_id}
Guardrail ID: {self.guardrail_id}
Guardrail Version: {self.guardrail_version}
AWS Region: {self.region}
Interface: Strands Agents SDK
Agent Initialized: {self.agent is not None}
Policy Loaded: {self.refund_policy_text is not None}
Conversation Messages: {len(self.agent.messages) if self.agent else 0}
""")

    def clear_history(self):
        """Clear conversation history"""
        if self.agent:
            # Reinitialize the agent to clear history
            self.initialize_agent()
        print("🗑️  Conversation history cleared")

    def run_interactive_session(self):
        """
        Run the interactive session
        """
        print("🚀 Starting interactive session with Strands Agents...")
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
                response = self.chat_with_agent(user_input)
                print(f"Response: {json.dumps(response, indent=2, default=str)}")
                print("-" * 40)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """
    Main function to run interactive automated reasoning policy tester with Strands Agents
    """
    # Load configuration
    config = load_config(require_guardrail_id=True, load_test_cases_file=False)

    # Print configuration header
    print_config_header(
        config,
        "Interactive Automated Reasoning Policy Tester (Strands Agents)",
        {"Interface": "Strands Agents SDK"},
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
