# Automated Reasoning Policy Test Demo

A comprehensive toolkit for testing and managing AWS Bedrock Automated Reasoning Policies across different AI frameworks.

> **Note**: This project requires `uv` package manager. For installation instructions, visit [uv's homepage](https://docs.astral.sh/uv/).

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment (choose one method)
export GUARDRAIL_ID=your-guardrail-id
# OR create a .env file with GUARDRAIL_ID=your-guardrail-id

# Run interactive demo
uv run automated-reasoning-checks-demo-with-bedrock.py
```

## What's Included

This toolkit provides **five specialized tools** for working with Automated Reasoning policies:

| Tool | Purpose | Mode |
|------|---------|------|
| `automated-reasoning-checks-demo-static.py` | Batch test predefined cases | Non-interactive |
| `automated-reasoning-checks-demo-with-bedrock.py` | Test with Bedrock native API | Interactive |
| `automated-reasoning-checks-demo-with-openai.py` | Test with OpenAI SDK + Bedrock | Interactive |
| `automated-reasoning-checks-demo-with-strands.py` | Test with Strands Agents framework | Interactive |
| `export-automated-reasoning-policy.py` | Export policies to CloudFormation | Both modes |

## Interactive Testing Tools

All interactive demos share these capabilities:
- Real-time guardrail enforcement testing
- Live automated reasoning findings
- Customer support refund policy scenarios
- Conversation history management

**Common Commands for Interactive Demos**:
- `/help` - Show help information
- `/quit`, `/exit`, `/bye` - Exit the program
- `/clear` - Clear conversation history
- `/status` - Show current configuration

### Bedrock Native Demo

Uses AWS Bedrock Converse API with integrated guardrails - the simplest integration path.

```bash
uv run automated-reasoning-checks-demo-with-bedrock.py
```

**Best for**: Native AWS workflows, maximum integration with Bedrock features

### OpenAI SDK Demo

Connects to Bedrock via OpenAI SDK, applying guardrails separately for input and output.

```bash
uv run automated-reasoning-checks-demo-with-openai.py
```

**Best for**: Teams using OpenAI SDK, dual-stage filtering analysis

### Strands Agents Demo

Uses Strands Agents framework for modern agent-based architecture.

```bash
uv run automated-reasoning-checks-demo-with-strands.py
```

**Best for**: Complex agent workflows, framework-agnostic integration

## Batch Testing Tool

Run structured test cases from JSON for CI/CD integration.

```bash
uv run automated-reasoning-checks-demo-static.py

# Run specific test
uv run automated-reasoning-checks-demo-static.py --test 5
```

Test case format in `automated_reasoning_test_cases.json`:
```json
{
  "test_cases": [
    {
      "question": "User question",
      "answer": "Model response",
      "expected_result": "VALID|INVALID|SATISFIABLE"
    }
  ]
}
```

## Policy Export Tool

Export AR policies to CloudFormation templates for infrastructure-as-code workflows.

### Interactive Mode (Browse & Select)
```bash
python export-automated-reasoning-policy.py
python export-automated-reasoning-policy.py --region us-west-2 --output-dir ./exports
```

### Direct Mode (Specify Policy)
```bash
# By ARN
python export-automated-reasoning-policy.py --policy-arn arn:aws:bedrock:us-east-1:123456789012:automated-reasoning-policy/abc123/DRAFT

# By ID and version
python export-automated-reasoning-policy.py --policy-id abc123 --version DRAFT --region us-east-1
```

**Generated Files**:
- `policy_{id}_v{version}_{timestamp}.json` - Complete policy configuration
- `policy_{id}_v{version}_{timestamp}_cfn.json` - CloudFormation template (JSON)
- `policy_{id}_v{version}_{timestamp}_cfn.yaml` - CloudFormation template (YAML)

**Use Cases**:
- Multi-account policy deployment
- Version control and GitOps workflows
- Policy backup and disaster recovery
- Cross-region replication

## Configuration

### Environment Variables

All demos use these variables (export tool only needs AWS credentials):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GUARDRAIL_ID` | Yes* | - | Your AR policy guardrail ID |
| `GUARDRAIL_VERSION` | No | `DRAFT` | Policy version |
| `AWS_REGION` | No | AWS default | AWS region |
| `MODEL_ID` | No | `openai.gpt-oss-20b-1:0` | Bedrock model ID |

*Not required for export tool

### Configuration Methods

**Option 1: Direct export**
```bash
export GUARDRAIL_ID=your-guardrail-id
export GUARDRAIL_VERSION=DRAFT
export AWS_REGION=us-east-1
export MODEL_ID=openai.gpt-oss-20b-1:0
```

**Option 2: .env file (recommended)**
```bash
# Copy the example file and edit with your values
cp .env.example .env
# Edit .env and replace placeholders with your actual values
```

## Prerequisites

**AWS Setup**:
- AWS credentials configured (`aws configure`)
- Bedrock model access enabled in your region
- AR policy (guardrail) created

**Required Permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-runtime:ApplyGuardrail",
        "bedrock-runtime:Converse",
        "bedrock-runtime:InvokeModel",
        "bedrock:ListAutomatedReasoningPolicies",
        "bedrock:ExportAutomatedReasoningPolicyVersion"
      ],
      "Resource": "*"
    }
  ]
}
```

**Python Environment**:
- Python 3.8+
- `uv` package manager

## Project Structure

```
ar-checks-demo/
├── README.md                                           # This file
├── pyproject.toml                                      # Dependencies configuration
├── .env                                                # Environment variables (create this)
│
├── automated-reasoning-checks-demo-static.py           # Batch testing
├── automated-reasoning-checks-demo-with-bedrock.py     # Bedrock native integration
├── automated-reasoning-checks-demo-with-openai.py      # OpenAI SDK integration
├── automated-reasoning-checks-demo-with-strands.py     # Strands Agents integration
├── export-automated-reasoning-policy.py                # Policy export utility
│
├── utils/                                              # Shared utilities
│   ├── __init__.py
│   ├── automated_reasoning_common.py                   # Result extraction
│   ├── config.py                                       # Configuration management
│   └── pdf_utils.py                                    # PDF processing
│
├── automated_reasoning_test_cases.json                 # Test cases for batch testing
└── docs/
    └── Customer Support Refund Policy.pdf              # Policy document for demos
```

## Architecture

The toolkit uses a modular architecture with three shared utilities in the `utils/` directory:

- **`utils/automated_reasoning_common.py`**: Handles automated reasoning results from different API formats (Converse vs ApplyGuardrail)
- **`utils/config.py`**: Manages environment variables, validation, and configuration
- **`utils/pdf_utils.py`**: Extracts text from PDF policy documents

This design eliminates code duplication and ensures consistent behavior across all tools.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing guardrail ID | Set `GUARDRAIL_ID` environment variable or create `.env` file |
| Permission denied | Verify IAM permissions with `aws sts get-caller-identity` |
| Model not available | Check model access in AWS Console → Bedrock → Model access |
| PDF not found | Ensure `docs/Customer Support Refund Policy.pdf` exists |
| Connection errors | Verify AWS credentials and region configuration |

**Debug Tips**:
- Use `/status` command in interactive mode to check configuration
- Check AWS CloudTrail for API call logs
- Verify model ID format: `provider.model-name:version`

## Dependencies

Automatically installed via `uv sync`:

```toml
[project.dependencies]
boto3 = ">=1.40.40"              # AWS SDK
openai = ">=1.109.1"             # OpenAI SDK
pymupdf = ">=1.26.4"             # PDF processing
python-dotenv = ">=1.1.1"        # Environment management
pyyaml = ">=6.0.3"               # YAML generation
strands-agents = ">=1.10.0"      # Strands framework
```

## Use Cases

- ✅ **Development**: Test and refine AR policies interactively
- ✅ **CI/CD**: Automate policy validation in pipelines
- ✅ **Compliance**: Ensure consistent policy enforcement
- ✅ **Comparison**: Evaluate guardrail behavior across frameworks
- ✅ **Migration**: Deploy policies across accounts/regions
- ✅ **IaC**: Version control policies with CloudFormation
- ✅ **Training**: Demonstrate AR capabilities to stakeholders

## Contributing

When modifying the toolkit:
1. Maintain consistency across all demo scripts
2. Update this README for any new features
3. Use shared libraries in `utils/` directory (`automated_reasoning_common.py`, `config.py`, `pdf_utils.py`) for common functionality
4. Test with `uv run <script.py>` format
5. Verify environment variable handling

## License

See repository license file.