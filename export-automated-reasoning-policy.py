#!/usr/bin/env python3
"""
Export Automated Reasoning Policy from AWS Bedrock

This script exports an automated reasoning policy configuration from AWS Bedrock
and generates a CloudFormation template to recreate it.

Usage:
    # Interactive mode (browse and select policies)
    python export-automated-reasoning-policy.py --region us-east-1
    python export-automated-reasoning-policy.py --region us-east-1 --output-dir ./exports

    # Direct mode (specify policy directly)
    python export-automated-reasoning-policy.py --policy-arn <arn> [--output-dir <dir>]
    python export-automated-reasoning-policy.py --policy-id <id> --version <version> --region <region> [--output-dir <dir>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


def export_policy_version(
    bedrock_client,
    policy_arn: Optional[str] = None,
    policy_id: Optional[str] = None,
    policy_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export automated reasoning policy version from Bedrock.

    Args:
        bedrock_client: Boto3 Bedrock client
        policy_arn: Full ARN of the policy version (if provided, policy_id and policy_version are ignored)
        policy_id: Policy identifier
        policy_version: Policy version

    Returns:
        Policy configuration dictionary

    Raises:
        ClientError: If API call fails
    """
    try:
        if policy_arn:
            response = bedrock_client.export_automated_reasoning_policy_version(
                policyArn=policy_arn
            )
        elif policy_id and policy_version:
            response = bedrock_client.export_automated_reasoning_policy_version(
                policyIdentifier=policy_id, policyVersion=policy_version
            )
        else:
            raise ValueError(
                "Must provide either policy_arn or both policy_id and policy_version"
            )

        return response

    except ClientError as e:
        print(f"Error exporting policy: {e}")
        raise


def generate_cloudformation_template(
    policy_config: Dict[str, Any], policy_name: str, include_tags: bool = True
) -> Dict[str, Any]:
    """
    Generate CloudFormation template from policy configuration.

    Args:
        policy_config: Policy configuration from export API
        policy_name: Name for the CloudFormation resource
        include_tags: Whether to include tags in the template

    Returns:
        CloudFormation template dictionary
    """
    # Extract policy definition (contains rules, types, variables)
    policy_definition = policy_config.get("policyDefinition", {})

    # Build CloudFormation resource properties
    properties = {"Name": policy_name}

    # Add policy definition if present
    if policy_definition:
        properties["PolicyDefinition"] = policy_definition

    # Add optional properties if present
    if "description" in policy_config:
        properties["Description"] = policy_config["description"]

    if include_tags and "tags" in policy_config:
        # Convert tags format if needed
        tags = policy_config.get("tags", [])
        if tags:
            properties["Tags"] = [
                {"Key": tag["key"], "Value": tag["value"]} for tag in tags
            ]

    # Create CloudFormation template
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"CloudFormation template for Automated Reasoning Policy: {policy_name}",
        "Resources": {
            "AutomatedReasoningPolicy": {
                "Type": "AWS::Bedrock::AutomatedReasoningPolicy",
                "Properties": properties,
            }
        },
        "Outputs": {
            "PolicyId": {
                "Description": "The ID of the created Automated Reasoning Policy",
                "Value": {"Fn::GetAtt": ["AutomatedReasoningPolicy", "PolicyId"]},
            },
            "PolicyArn": {
                "Description": "The ARN of the created Automated Reasoning Policy",
                "Value": {"Fn::GetAtt": ["AutomatedReasoningPolicy", "PolicyArn"]},
            },
        },
    }

    return template


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file with pretty formatting."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved: {file_path}")


def list_policies(bedrock_client) -> List[Dict[str, Any]]:
    """
    List all automated reasoning policies in the account.

    Args:
        bedrock_client: Boto3 Bedrock client

    Returns:
        List of policy summaries
    """
    try:
        policies = []
        paginator = bedrock_client.get_paginator("list_automated_reasoning_policies")

        for page in paginator.paginate():
            policies.extend(page.get("automatedReasoningPolicySummaries", []))

        return policies
    except ClientError as e:
        print(f"Error listing policies: {e}")
        raise


def display_policies(policies: List[Dict[str, Any]]) -> None:
    """Display policies in a formatted table."""
    if not policies:
        print("No policies found in this region.")
        return

    print("\n" + "=" * 110)
    print("Available Automated Reasoning Policies")
    print("=" * 110)
    print(
        f"{'#':<4} {'Policy ID':<20} {'Name':<30} {'Version':<10} {'Created':<20} {'Updated':<20}"
    )
    print("-" * 110)

    for idx, policy in enumerate(policies, 1):
        policy_id = policy.get("policyId", "N/A")
        policy_name = policy.get("name", "N/A")
        version = policy.get("version", "N/A")
        created_at = policy.get("createdAt", "N/A")
        updated_at = policy.get("updatedAt", "N/A")

        if isinstance(created_at, datetime):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(updated_at, datetime):
            updated_at = updated_at.strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"{idx:<4} {policy_id:<20} {policy_name:<30} {version:<10} {str(created_at):<20} {str(updated_at):<20}"
        )

    print("=" * 110)


def interactive_policy_selection(
    bedrock_client, region: str
) -> Optional[Dict[str, str]]:
    """
    Interactive menu to select a policy.

    Args:
        bedrock_client: Boto3 Bedrock client
        region: AWS region

    Returns:
        Dictionary with policy_arn, policy_id, version, and policy_name, or None if cancelled
    """
    print(f"\n🔍 Fetching policies from region: {region}...")

    try:
        policies = list_policies(bedrock_client)
    except Exception as e:
        print(f"❌ Failed to list policies: {e}")
        return None

    if not policies:
        print("\n❌ No automated reasoning policies found in this region.")
        return None

    # Display policies
    display_policies(policies)

    # Select policy
    while True:
        try:
            choice = input("\nSelect a policy by number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                print("👋 Cancelled.")
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(policies):
                selected_policy = policies[idx]
                break
            else:
                print(
                    f"❌ Invalid selection. Please enter a number between 1 and {len(policies)}."
                )
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q'.")

    policy_arn = selected_policy["policyArn"]
    policy_id = selected_policy["policyId"]
    policy_name = selected_policy.get("name", policy_id)
    version = selected_policy.get("version", "DRAFT")

    print(f"\n✅ Selected: {policy_name} (ID: {policy_id}, Version: {version})")

    return {
        "policy_arn": policy_arn,
        "policy_id": policy_id,
        "version": version,
        "policy_name": policy_name,
    }


def parse_policy_arn(arn: str) -> Dict[str, str]:
    """
    Parse policy ARN to extract components.

    ARN format: arn:aws:bedrock:region:account-id:automated-reasoning-policy/policy-id/version
    """
    parts = arn.split(":")
    if len(parts) < 6:
        raise ValueError(f"Invalid ARN format: {arn}")

    resource_parts = parts[5].split("/")
    if len(resource_parts) < 3:
        raise ValueError(f"Invalid policy ARN format: {arn}")

    return {
        "region": parts[3],
        "account_id": parts[4],
        "policy_id": resource_parts[1],
        "version": resource_parts[2],
    }


def main():
    """Main function to export policy and generate CloudFormation template."""
    parser = argparse.ArgumentParser(
        description="Export Automated Reasoning Policy and generate CloudFormation template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (browse and select)
  python export-automated-reasoning-policy.py --region us-east-1
  python export-automated-reasoning-policy.py --region us-east-1 --output-dir ./exports

  # Direct mode - Using policy ARN
  python export-automated-reasoning-policy.py --policy-arn arn:aws:bedrock:us-east-1:123456789012:automated-reasoning-policy/abc123/1

  # Direct mode - Using policy ID and version
  python export-automated-reasoning-policy.py --policy-id abc123 --version 1 --region us-east-1

  # Specify output directory
  python export-automated-reasoning-policy.py --policy-arn arn:... --output-dir ./exports
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--policy-arn", help="Full ARN of the policy version (direct mode)"
    )
    input_group.add_argument(
        "--policy-id", help="Policy identifier (use with --version, direct mode)"
    )

    parser.add_argument(
        "--version", help="Policy version (use with --policy-id, direct mode)"
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (default: uses AWS default configuration)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--policy-name",
        help="Name for the policy in CloudFormation template (default: derived from policy ID)",
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        help="Exclude tags from CloudFormation template",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.policy_id and not args.version:
        parser.error("--version is required when using --policy-id")

    region = args.region

    # Initialize Bedrock client
    if region:
        print(f"\n🔧 Connecting to AWS Bedrock in {region}...")
        bedrock_client = boto3.client("bedrock", region_name=region)
    else:
        print("\n🔧 Connecting to AWS Bedrock (using default region)...")
        bedrock_client = boto3.client("bedrock")
        # Get actual region from client
        region = bedrock_client.meta.region_name
        print(f"   Using region: {region}")

    # Determine mode: interactive or direct
    if args.policy_arn:
        # Direct mode with ARN
        try:
            arn_parts = parse_policy_arn(args.policy_arn)
            policy_id = arn_parts["policy_id"]
            version = arn_parts["version"]
            policy_name = None
            print("📋 Parsed ARN:")
            print(f"   Policy ID: {policy_id}")
            print(f"   Version: {version}")
            print(f"   Region: {region}")
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    elif args.policy_id and args.version:
        # Direct mode with ID and version
        policy_id = args.policy_id
        version = args.version
        policy_name = None
    else:
        # Interactive mode
        print("\n🎯 Interactive Mode - Browse and select policy")
        selection = interactive_policy_selection(bedrock_client, region)
        if not selection:
            sys.exit(0)

        policy_arn = selection["policy_arn"]
        policy_id = selection["policy_id"]
        version = selection["version"]
        policy_name = selection.get("policy_name")

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export policy
    print(f"📤 Exporting policy {policy_id} version {version}...")
    try:
        # Always use ARN if available (from interactive or parsed from args)
        if args.policy_arn or "policy_arn" in locals():
            export_arn = args.policy_arn if args.policy_arn else policy_arn
            policy_config = export_policy_version(bedrock_client, policy_arn=export_arn)
        else:
            policy_config = export_policy_version(
                bedrock_client, policy_id=policy_id, policy_version=version
            )
    except Exception as e:
        print(f"❌ Failed to export policy: {e}")
        sys.exit(1)

    # Generate timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw policy configuration
    json_filename = f"policy_{policy_id}_v{version}_{timestamp}.json"
    json_path = args.output_dir / json_filename
    save_json_file(policy_config, json_path)

    # Generate CloudFormation template
    print("\n🏗️  Generating CloudFormation template...")
    cfn_policy_name = (
        args.policy_name or policy_name or f"AutomatedReasoningPolicy{policy_id}"
    )

    cfn_template = generate_cloudformation_template(
        policy_config, cfn_policy_name, include_tags=not args.no_tags
    )

    # Save CloudFormation template
    cfn_filename = f"policy_{policy_id}_v{version}_{timestamp}_cfn.yaml"
    cfn_json_filename = f"policy_{policy_id}_v{version}_{timestamp}_cfn.json"

    # Save JSON version
    cfn_json_path = args.output_dir / cfn_json_filename
    save_json_file(cfn_template, cfn_json_path)

    # Save YAML version
    try:
        import yaml

        cfn_yaml_path = args.output_dir / cfn_filename
        with open(cfn_yaml_path, "w") as f:
            yaml.dump(cfn_template, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Saved: {cfn_yaml_path}")
    except ImportError:
        print(
            "ℹ️  PyYAML not installed. YAML format not available. Install with: uv add pyyaml"
        )

    # Print summary
    print("\n" + "=" * 80)
    print("📊 Export Summary")
    print("=" * 80)
    print(f"Policy ID: {policy_id}")
    print(f"Version: {version}")
    print(f"Region: {region}")
    print("\nGenerated Files:")
    print(f"  1. Policy Configuration (JSON): {json_path}")
    print(f"  2. CloudFormation Template (JSON): {cfn_json_path}")
    if "yaml" in sys.modules:
        print(f"  3. CloudFormation Template (YAML): {cfn_yaml_path}")
    print("\n💡 To deploy the CloudFormation template:")
    print("   aws cloudformation create-stack \\")
    print(f"     --stack-name {cfn_policy_name.lower()}-stack \\")
    print(f"     --template-body file://{cfn_json_path} \\")
    print(f"     --region {region}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
