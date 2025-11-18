"""
Interactive Automated Reasoning Policy Test Runner using Strands Agents
Tests guardrail enforcement through interactive conversation with user input.
Uses Strands Agents with AWS Bedrock integration and hooks for guardrail checking.
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, Any

import boto3
from strands import Agent
from strands.models import BedrockModel
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent, AfterInvocationEvent

from utils.automated_reasoning_common import extract_automated_reasoning_results
from utils.pdf_utils import extract_pdf_text, DEFAULT_REFUND_POLICY_PATH
from utils.config import load_config, print_config_header


@dataclass
class Message:
    """Conversation message"""

    role: str
    content: str


class AutomatedReasoningGuardrailHook(HookProvider):
    """
    Hook for checking automated reasoning policy using Bedrock ApplyGuardrail API.
    Evaluates both user input and assistant output for policy violations.
    """

    def __init__(self, guardrail_id: str, guardrail_version: str, region: str):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        self.last_input_findings = []
        self.last_output_findings = []

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hooks for checking user input and assistant output"""
        registry.add_callback(MessageAddedEvent, self.check_user_input)
        registry.add_callback(AfterInvocationEvent, self.check_assistant_response)

    def evaluate_content(self, content: str, source: str = "INPUT") -> Dict[str, Any]:
        """
        Evaluate content using Bedrock ApplyGuardrail API.

        Args:
            content: The content to evaluate
            source: Either "INPUT" or "OUTPUT"

        Returns:
            The guardrail API response
        """
        try:
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=[{"text": {"text": content}}]
            )
            return response
        except Exception as e:
            print(f"[GUARDRAIL] Evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_content_with_context(
        self, content: str, qualifying_query: str, source: str = "OUTPUT"
    ) -> Dict[str, Any]:
        """
        Evaluate content using Bedrock ApplyGuardrail API with question-answer context.
        This is used to evaluate assistant responses with the user's question as context.

        Args:
            content: The content to evaluate (assistant response)
            qualifying_query: The user's question that provides context
            source: Either "INPUT" or "OUTPUT"

        Returns:
            The guardrail API response
        """
        try:
            # Build content array with qualifiers for question-answer pair
            # question is "query", answer is "guard_content"
            content_array = [
                {"text": {"text": qualifying_query, "qualifiers": ["query"]}},
                {"text": {"text": content, "qualifiers": ["guard_content"]}}
            ]
            
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=content_array
            )
            return response
        except Exception as e:
            print(f"[GUARDRAIL] Evaluation failed: {e}")
            return {"error": str(e)}

    def check_user_input(self, event: MessageAddedEvent) -> None:
        """Check user input using guardrail"""
        if event.message.get("role") == "user":
            content = "".join(block.get("text", "") for block in event.message.get("content", []))
            if content:
                print("\n🔍 Checking user input with guardrail...")
                response = self.evaluate_content(content, "INPUT")
                
                # Extract and store automated reasoning findings
                self.last_input_findings = extract_automated_reasoning_results(response)
                
                if response.get("action") == "GUARDRAIL_INTERVENED":
                    print("🛡️  Guardrail intervened on user input")
                
                # Display automated reasoning findings
                if self.last_input_findings:
                    print("\n📋 Automated Reasoning Findings (Input):")
                    print(json.dumps(self.last_input_findings, indent=2))

    def check_assistant_response(self, event: AfterInvocationEvent) -> None:
        """Check assistant response using guardrail with user question context"""
        if event.agent.messages and len(event.agent.messages) >= 2:
            # Get the last assistant message
            assistant_message = event.agent.messages[-1]
            if assistant_message.get("role") == "assistant":
                assistant_content = "".join(
                    block.get("text", "") for block in assistant_message.get("content", [])
                )
                
                # Get the corresponding user message (question)
                user_message = event.agent.messages[-2]
                user_content = ""
                if user_message.get("role") == "user":
                    user_content = "".join(
                        block.get("text", "") for block in user_message.get("content", [])
                    )
                
                if assistant_content:
                    print("\n🔍 Checking assistant output with guardrail...")
                    print(f"User question: {user_content}")
                    print(f"Assistant answer: {assistant_content}")
                    
                    # Send both question and answer to guardrail
                    # Use qualifyingQuery to provide the user's question as context
                    response = self.evaluate_content_with_context(
                        content=assistant_content,
                        qualifying_query=user_content,
                        source="OUTPUT"
                    )
                    
                    # Extract and store automated reasoning findings
                    self.last_output_findings = extract_automated_reasoning_results(response)
                    
                    if response.get("action") == "GUARDRAIL_INTERVENED":
                        print("🛡️  Guardrail intervened on assistant output")
                    
                    # Display automated reasoning findings
                    if self.last_output_findings:
                        print("\n📋 Automated Reasoning Findings (Output):")
                        print(json.dumps(self.last_output_findings, indent=2))


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

        # Initialize Bedrock model without direct guardrail integration
        self.bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            streaming=False,
        )

        # Create guardrail hook for automated reasoning checks
        self.guardrail_hook = AutomatedReasoningGuardrailHook(
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            region=region
        )

        # System prompt will be set after loading the refund policy
        self.agent = None

    def initialize_agent(self):
        """Initialize the Strands agent with system prompt including refund policy and guardrail hook"""
        if not self.refund_policy_text:
            self.refund_policy_text = extract_pdf_text(self.refund_policy_path)

        system_prompt = f"""You are a customer support agent. You follow the provided refund policy. Reply with max 10 words.

REFUND POLICY:
{self.refund_policy_text}"""

        self.agent = Agent(
            system_prompt=system_prompt, 
            model=self.bedrock_model,
            hooks=[self.guardrail_hook]
        )

    def chat_with_agent(self, user_message: str) -> Dict[str, Any]:
        """
        Use Strands Agent to process user message with guardrail hooks

        Args:
            user_message: The user's input message

        Returns:
            Dictionary containing response and guardrail information
        """
        # Initialize agent if not already done
        if self.agent is None:
            self.initialize_agent()

        print("\n🔍 Invoking Strands Agent with guardrail hook...")
        print(f"Model: {self.model_id}")
        print(f"Guardrail ID: {self.guardrail_id}")
        print(f"Guardrail Version: {self.guardrail_version}")

        # Use the agent to process the message
        # The guardrail hook will automatically check input and output
        response = self.agent(user_message)

        # Extract response information
        assistant_message = str(response)

        print(f"\n🤖 Assistant: {assistant_message}")

        return {
            "message": assistant_message,
            "input_findings": self.guardrail_hook.last_input_findings,
            "output_findings": self.guardrail_hook.last_output_findings,
        }

    def show_help(self):
        """Display help information"""
        print(f"""
🔧 Interactive Automated Reasoning Policy Tester (Strands Agents) Help
======================================================================

This tool allows you to interactively test automated reasoning policies
using Strands Agents with AWS Bedrock guardrails via hooks.

Commands:
  /help    - Show this help message
  /quit    - Exit the program
  /exit    - Exit the program
  /bye     - Exit the program
  /clear   - Clear conversation history
  /status  - Show current configuration

Features:
  • Interactive conversation using Strands Agents
  • Guardrail enforcement via hooks using ApplyGuardrail API
  • Automated reasoning policy testing on both input and output
  • Conversation history maintained across interactions
  • Shows detailed automated reasoning findings

Configuration:
  Model: {self.model_id}
  Guardrail ID: {self.guardrail_id}
  Guardrail Version: {self.guardrail_version}
  AWS Region: {self.region}
  Interface: Strands Agents SDK with Hooks

Simply type your message and press Enter to interact with the AI model.
The guardrail hook will automatically check both user input and assistant
output for policy violations and display automated reasoning findings.
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
Interface: Strands Agents SDK with Hooks
Agent Initialized: {self.agent is not None}
Policy Loaded: {self.refund_policy_text is not None}
Conversation Messages: {len(self.agent.messages) if self.agent else 0}
Last Input Findings: {len(self.guardrail_hook.last_input_findings)}
Last Output Findings: {len(self.guardrail_hook.last_output_findings)}
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
        {"Interface": "Strands Agents SDK with Hooks"},
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
