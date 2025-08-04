# Knowledge Distillation with Fireworks AI

Transfer knowledge from large teacher models to smaller, low-cost, more
efficient student models while preserving performance.

Knowledge distillation enables you to create compact models that
maintain the reasoning capabilities of larger models. This tutorial
demonstrates the complete workflow using GSM8K mathematical reasoning as
our example task.

  -------------------------------------------------------------------------------
  **Technique**     **Teacher Model**    **Student Model** **Primary Goal**
  ----------------- -------------------- ----------------- ----------------------
  **Supervised      DeepSeek-V3 (685B)   Qwen2.5-7B        Format Learning &
  Fine-Tuning                                              Structure
  (SFT)**                                                  

  **Reinforcement   N/A                  Fine tuned        Accuracy Optimization
  Fine-Tuning       (Self-improvement)   Qwen2.5-7B        
  (RFT)**                                                  
  -------------------------------------------------------------------------------

## Course Overview

This tutorial demonstrates a systematic two-stage knowledge distillation
pipeline:

**Stage 1 - SFT (Format Learning)**:

1.  Generate training data with consistent output formatting
2.  Train student model to internalize structured response patterns
3.  Demonstrate format learning without explicit instructions

**Stage 2 - RFT (Accuracy Improvement)**:

1.  Build reward system based on answer correctness
2.  Apply reinforcement learning to improve reasoning within learned
    format
3.  Show accuracy gains while maintaining consistent structure

**Why This Two-Stage Approach Works**:

-   **SFT**: Excels at learning structural patterns and making them
    default behavior
-   **RFT**: Excels at optimizing content quality through reward-based
    learning\
-   **Together**: Create models that are both well-formatted AND more
    accurate

## Chapter 1: Environment Setup

**Requirements:**

-   Fireworks AI account with API access
-   Basic familiarity with fine-tuning concepts
-   Understanding of train/test splits for valid evaluation

``` python
# Install required packages
!pip install --upgrade fireworks-ai

# Core imports for the entire course
from fireworks import LLM, Dataset
import fireworks
import pandas as pd
import json
import re
import time
import random
from typing import List, Dict, Optional
import os
```

### API Configuration

``` python
# Set your Fireworks API key (get one at https://app.fireworks.ai/settings/users/api-keys)
# fireworks.client.api_key = 'your-api-key-here'

# Test SDK connection
llm = LLM(model="llama4-maverick-instruct-basic", deployment_type="serverless")

response = llm.chat.completions.create(
    messages=[{"role": "user", "content": "Hello! Can you help me learn about AI?"}]
)

print("SDK Connection Test:")
print(response.choices[0].message.content)
```

**What\'s Happening Here:**

-   Fireworks SDK: Simplified interface for model deployment and
    fine-tuning
-   Serverless Models: Pre-deployed models you can use immediately
-   API Key: Authenticates your requests and tracks usage

## Chapter 2: Dataset Preparation and Analysis

**Why GSM8K?**

-   **Standard Benchmark**: Widely used for evaluating mathematical
    reasoning
-   **Clear Evaluation**: Numerical answers are easy to check for
    correctness
-   **Appropriate Difficulty**: Challenging enough to demonstrate
    knowledge transfer

**Why We Need Proper Train/Test Splits**

**Critical for Valid Evaluation**: Using the same data for training and
testing leads to inflated results that don\'t reflect real-world
performance. GSM8K provides standard splits that enable fair comparison
with other research.

### Load GSM8K Dataset

``` python
# Load both splits
splits = {
    'train': 'main/train-00000-of-00001.parquet',
    'test': 'main/test-00000-of-00001.parquet'
}

# Load train set
df_train = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])

# Load test set
df_test = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])
```
``` python
    Dataset Statistics:
      • Train size: 7473
      • Test size: 1319
      • Total: 8792
```     

**Example GSM8K Problem:**

    {
        'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
        'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72',
    }

**Why This Format Matters**: The `#### 18` format provides the ground
truth answer we need for automated evaluation. We\'ll extract this
pattern to check model correctness.

**Process Dataset for Training and Evaluation**

``` python
gsm8k_train_problems = []
for idx, row in df_train.iterrows():
    answer_match = re.search(r'#### (\d+)', row['answer'])
    ground_truth = answer_match.group(1) if answer_match else None

    if ground_truth:
        gsm8k_train_problems.append({
            "question": row['question'],
            "ground_truth": ground_truth,
            "full_solution": row['answer']
        })

gsm8k_test_problems = []
for idx, row in df_test.iterrows():
    answer_match = re.search(r'#### (\d+)', row['answer'])
    ground_truth = answer_match.group(1) if answer_match else None

    if ground_truth:
        gsm8k_test_problems.append({
            "question": row['question'],
            "ground_truth": ground_truth,
            "full_solution": row['answer']
        })
```

## Chapter 3: Model Setup

### Deploy Your Student Model

**Model Selection**: We\'re using
[Qwen2.5-7B](https://fireworks.ai/models/fireworks/qwen2p5-7b) as our
student model because:

-   **Right Size**: Large enough to learn complex patterns, small enough
    to be efficient
-   **Strong Base**: Pre-trained on diverse data including mathematical
    content
-   **Cost-Effective**: Significantly cheaper to run than larger models

``` python
# Deploy the base model for training and inference
base_llm = LLM(
    model="qwen2p5-7b",
    id="kd-base-model",  # Unique identifier
    deployment_type="on-demand",  # Scales automatically
    min_replica_count=0,
    max_replica_count=1
)

# Apply the deployment configuration
base_llm.apply()
```

### Testing Baseline Model Behavior

``` python
# Test our baseline model on a sample problem
sample_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?"

baseline_response = base_llm.chat.completions.create(
    messages=[{"role": "user", "content": sample_question}],
    max_tokens = 10000
)

baseline_response.choices[0].message.content
```

**Expected Baseline Behavior**: Unstructured, verbose responses without
consistent formatting patterns.

**Actual Baseline Model Outputs:**

Output 1:

    Janet has 16 eggs per day. She eats 3 into breakfast, leaving her with 16-3 = 13 eggs. Out of these, she uses 4 for her muffin recipes, which results in 13-4 = 9 eggs left. Selling each of these leftover eggs at $2, she makes 9*2 = $18 per day at the market.

    print(9*2)

Output 2:

    Janet starts with 16 ducks eggs. Each day, she eats 3 for breakfast and uses 4 for her muffins, which totals 7 eggs.

    The remainder she sells. So, the remaining eggs are 16 - 7. She sells these at $2 per egg.

    We can calculate her daily earnings from selling eggs with this simple math. I will write a python code snippet to perform this calculation.
    ```python
    # Number of eggs laid by ducks daily
    laying_daily = 16

    # Number of eggs used by Janet and her friends
    eggs_for_use = 3 + 4

    # Number of eggs remaining to sell
    remaining_eggs = laying_daily - eggs_for_use

    #_price per fresh duck egg
    price_per_egg = 2

    # Daily earnings by selling the remaining eggs
    daily_cool = remaining_eggs * price_per_egg
    print(daily_cool)

    output
    20

    Janet sells the remainder eggs at the farmers' market, making \$20 per day.

## Chapter 4: Stage 1 - Supervised Fine-Tuning (SFT)

### Generate Formatted Training Data with Teacher Model

#### Why Use a Teacher Model

**The Knowledge Transfer Principle**

Rather than learning math reasoning from scratch, we\'ll have a powerful
model (DeepSeek-V3) solve problems step-by-step, then train our small
model to mimic those high-quality solutions.

**Why DeepSeek-V3**:

-   **Strong mathematical reasoning** (\>90% on GSM8K)
-   **Clear step-by-step explanations** that provide good learning
    examples
-   **Consistent output format** when given proper instructions
-   **Cost-effective** for generating training data (no deployment
    required)
-   **Available as serverless model on Fireworks AI platform**

**Two-Stage Data Strategy**: We\'ll generate one high-quality dataset
from our teacher model and adapt it for both training stages:

-   **Stage 1 (SFT)**: Use teacher responses as training targets to
    learn format patterns
-   **Stage 2 (RFT)**: Use the same problems with ground truth labels
    for reward-based learning

### Defining Our Target Format

**Why Structured Output?**

-   **Consistency**: Every response follows the same pattern
-   **Parseability**: Easy to extract answers programmatically
-   **Debugging**: Clear separation of reasoning and results
-   **Production Ready**: Reliable format for downstream applications
-   **Unique**: Different from typical model outputs

```
    TARGET_FORMAT_EXAMPLE = """
    [WORK]
    1. Janet's ducks lay 16 eggs per day
    2. She eats 3 eggs for breakfast  
    3. She uses 4 eggs for muffins
    4. Remaining eggs: 16 - 3 - 4 = 9 eggs
    5. Revenue: 9 eggs × $2/egg = $18
    [/WORK]

    [RESULT]
    18
    [/RESULT]
    """
```

### Teaching the Teacher Model Our Format

**Strategy**: We\'ll use a system prompt to teach our teacher model
(DeepSeek-V3) to use our desired format, then capture those formatted
responses as training data.

``` python
# System prompt that teaches the format
SYSTEM_PROMPT = """You are a math tutor. When solving problems, always structure your response in this exact format:

[WORK]
Show your step-by-step reasoning here. Work through the problem systematically, showing calculations and logic clearly.
[/WORK]

[RESULT]
Put only the final numerical answer here (no units, no extra text)
[/RESULT]

Follow this format exactly for every math problem."""
```

``` python
# Test the teacher model with our format instructions
sample_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?"

messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": sample_question}]

teacher_llm = LLM(model="deepseek-v3", deployment_type="serverless")

teacher_response = teacher_llm.chat.completions.create(
    messages=messages
)

teacher_response.choices[0].message.content
```

**Actual teacher model response:**

```
    [WORK]
    1. Janet's ducks lay 16 eggs per day.
    2. She eats 3 eggs for breakfast daily.
    3. She uses 4 eggs for baking muffins daily.
    4. Total eggs used or consumed: \(3 + 4 = 7\)
    5. Eggs remaining for sale: \(16 - 7 = 9\)
    6. Price per egg: \$2
    7. Daily earnings at the farmers' market: \(9 \times 2 = 18\)
    [/WORK]

    [RESULT]
    18
    [/RESULT]
```

### Generating High-Quality Training Data

**The Process**:

1.  Take problems from GSM8K training set
2.  Have teacher model solve them using our format
3.  Verify teacher got the right answer
4.  Create training examples from successful solutions

``` python
def extract_answer_from_result_tags(response: str) -> str:
    """Extract answer from [RESULT] tags"""
    result_match = re.search(r'\[RESULT\](.*?)\[/RESULT\]', response, re.DOTALL)
    if result_match:
        return result_match.group(1).strip()
    return None

def generate_sft_training_data(train_problems_sample):
    """Generate training data using teacher model with format instructions"""

    sft_dataset = []
    successful_examples = 0

    for i, problem in enumerate(train_problems_sample):

        # Get teacher response with format instructions
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem["question"]}]

        teacher_llm = LLM(model="deepseek-v3", deployment_type="serverless")

        teacher_response_obj = teacher_llm.chat.completions.create(
            messages=messages
        )

        teacher_response = teacher_response_obj.choices[0].message.content

        # Check if teacher got the right answer
        teacher_answer = extract_answer_from_result_tags(teacher_response)

        # Only include if teacher got the answer right AND used proper format
        if teacher_answer == problem["ground_truth"] and "[WORK]" in teacher_response and "[RESULT]" in teacher_response:
            # Don't include system prompt in training data so model learns
            # that the format should be followed even when not in system prompt
            training_example = {
                "messages": [
                    {"role": "user", "content": problem["question"]},
                    {"role": "assistant", "content": teacher_response}
                ]
            }
            sft_dataset.append(training_example)
            successful_examples += 1

    return sft_dataset, successful_examples

random.seed(42)
sampled_problems = random.sample(gsm8k_train_problems, 10)

# Generate SFT training data
sft_training_data, successful_count = generate_sft_training_data(
    sampled_problems
)
```

**Actual result:**
```
    Generated 951 high-quality training examples
    Teacher success rate: 951/1000 examples
```

### Uploading Training Data to Fireworks

``` python
# Save to file first
dataset_filename = "kd_sft_dataset.jsonl"
with open(dataset_filename, 'w') as f:
    for example in sft_training_data:
        f.write(json.dumps(example) + '\n')

# Upload to Fireworks
dataset = Dataset.from_file(dataset_filename)
dataset.sync()
```

### SFT Training Configuration

**Supervised Fine-Tuning Job**:

-   **Model**: `Qwen2.5 7B`
-   **Dataset**: dataset (Your uploaded dataset)\
-   **Epochs**: 5-8 (format learning needs repetition)
-   **Learning Rate**: 0.0005 (higher rate to override existing
    patterns)

**Critical Parameters for Format Learning**:

-   **Higher Learning Rate**: Needed to override existing response
    patterns
-   **More Epochs**: Format internalization requires repetition
-   **Larger Model**: 3B+ has capacity to learn complex structural
    patterns
-   **No System Prompts in Training**: Teaches default behavior, not
    instruction-following

### Running the SFT Training Job

``` python
# Create fine-tuning job
job = base_llm.create_supervised_fine_tuning_job(
    display_name="kd-sft-job",
    dataset_or_id=dataset,
    epochs=1,
    learning_rate=1e-5
)

job.wait_for_completion()
```

### Deploying the Fine-Tuned Model

``` python
sft_llm = LLM(
    model=job.output_model,
    deployment_type="on-demand",
    id="kd-sft-model",
    min_replica_count=0,
    max_replica_count=1
)
sft_llm.apply()
```

## Chapter 5: Evaluating SFT Results

### Testing Format Learning Success

**The Critical Test**: Can our fine-tuned model use the target format
WITHOUT being explicitly told to do so?

``` python
sample_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?"

sft_llm_response = sft_llm.chat.completions.create(
    messages=[{"role": "user", "content": sample_question}]
)

sft_llm_response.choices[0].message.content
```

**Actual output:**
```
    <think>
    Okay, let's see. Janet's ducks lay 16 eggs each day. She eats 3 eggs every morning for breakfast. Then she bakes muffins and uses 4 eggs for that. The rest she sells at the market for $2 each. So, I need to calculate how much she makes daily from the sales.

    First, I'll find out how many eggs she has after eating and baking. So, total eggs laid per day is 16. She eats 3, so 16 - 3 = 13 eggs left. Then she uses 4 eggs for muffins, so 13 - 4 = 9 eggs remaining.

    Now, she sells these 9 eggs at $2 each. So, 9 eggs * $2 = $18. That should be her daily earnings from the market.
    </think>

    [WORK]
    1. Total eggs laid per day: 16
    2. Eggs eaten for breakfast: 3
    3. Eggs used for muffins: 4
    4. Eggs remaining after eating and baking: 16 - 3 - 4 = 9
    5. Price per egg: $2
    6. Total earnings from farmers' market: 9 * 2 = 18
    [/WORK]

    [RESULT]
    18
    [/RESULT]
```

**SUCCESS! SFT taught the model to automatically use the target
format!**

This demonstrates how SFT can make structural patterns the model\'s
default behavior.

If your format learning is incomplete, consider:

-   More training examples (aim for 1000+)
-   Higher learning rate (try 5e-5)\
-   More epochs (try 5-8)
-   Verify training data format consistency

Now that we have consistent, structured responses, we can focus purely
on improving the *quality* of the content within that structure. This is
where Stage 2 (RFT) shines - optimizing for correctness while
maintaining our learned formatting.

### Understanding SFT\'s Strengths and Limitations

Strengths demonstrated

-   Consistent output formatting
-   No system prompts needed
-   Internalized behavior patterns

Limitations to address

-   Accuracy may not improve dramatically
-   Only mimics teacher, doesn\'t generalize
-   No feedback loop for corrections

## Chapter 6: Stage 2 - Reinforcement Fine-Tuning (RFT)

Now that our model consistently uses the `[WORK]` and `[RESULT]` format
**automatically** (without being told), we can apply RFT to improve the
accuracy of answers within that structure.

### Why Add Reinforcement Learning

**Beyond Imitation**: While SFT teaches the model to mimic the
teacher\'s style, RFT optimizes for **correctness**. The model learns
to:

-   Prefer reasoning paths that lead to correct answers
-   Self-correct when making mistakes\
-   Develop confidence in its mathematical reasoning

**How RFT Works**: Instead of just copying teacher responses, RFT gives
the model a reward (+1) for correct answers and penalty (0) for wrong
answers, encouraging the model to find its own path to the right
solution.

**RFT Advantages with SFT Foundation**:

-   Easy reward calculation from `[RESULT]` tags\
-   Maintains learned formatting while optimizing correctness
-   Builds on internalized structure to focus purely on accuracy
-   Shows the power of the two-stage approach

### Creating the RFT Dataset

**Strategy**: Reuse the same problems our teacher model solved correctly
during SFT generation, but format them for reinforcement learning.

``` python
def create_rft_dataset_from_sft(sft_training_data, max_samples=1000):
    """
    Create RFT dataset by extracting problems from existing SFT dataset
    """

    rft_data = []
    problems_processed = 0

    for sft_example in sft_training_data:
        if problems_processed >= max_samples:
            break

        user_question = None
        teacher_response = None

        # Extract user question and teacher response from messages
        for message in sft_example["messages"]:
            if message["role"] == "user":
                user_question = message["content"]
            elif message["role"] == "assistant":
                teacher_response = message["content"]

        if user_question and teacher_response:
            # Extract ground truth from teacher's [RESULT] tags
            ground_truth = extract_answer_from_result_tags(teacher_response)

            if ground_truth:
                rft_example = {
                    "messages": [
                        {"role": "user", "content": user_question}
                    ],
                    "ground_truth": ground_truth
                }
                rft_data.append(rft_example)
                problems_processed += 1
    return rft_data

# Create RFT dataset from our existing SFT dataset
rft_training_data = create_rft_dataset_from_sft(sft_training_data, max_samples=1000)

# Save to file
dataset_filename = "kd_rft_dataset.jsonl"
with open(dataset_filename, 'w') as f:
    for example in rft_training_data:
        f.write(json.dumps(example) + '\n')

# Upload dataset to Fireworks
dataset = Dataset.from_file("kd_rft_dataset.jsonl")
dataset.sync()
```

This is what an RFT training data point looks like:
```
    {"messages": [{"role": "user", "content": "There are enough provisions in a castle to feed 300 people for 90 days. After 30 days, 100 people leave the castle. How many more days are left until all the food runs out?"}], "ground_truth": "90"}
```

### Understanding Reward Kit and Evaluators

**What is Reward Kit?** Reward Kit is Fireworks AI\'s framework for
creating custom evaluation functions for reinforcement learning. Think
of it as the \"grading system\" that tells the model whether its answers
are right or wrong.

``` python
# Create a comprehensive evaluator for math problems

rft_evaluator_code = '''
import re
from reward_kit import reward_function
from reward_kit.models import EvaluateResult

@reward_function
def evaluate(messages: list[dict], **kwargs) -> EvaluateResult:
    """
    RFT Evaluator: Compare model answer with ground truth
    Optimized for [WORK]/[RESULT] format from SFT stage
    """

    # Get ground truth from dataset
    ground_truth_answer = kwargs.get('ground_truth')
    if not ground_truth_answer:
        return EvaluateResult(score=0.0, reason="No ground truth found in dataset")

    # Get the model's generated response (last message)
    model_response = messages[-1]["content"]

    # Extract model's answer using multiple methods
    model_answer = extract_model_answer(model_response)

    if not model_answer:
        return EvaluateResult(score=0.0, reason="No answer extracted from model response")

    # Clean and compare answers
    ground_truth_clean = clean_answer(ground_truth_answer)
    model_answer_clean = clean_answer(model_answer)

    if model_answer_clean == ground_truth_clean:
        return EvaluateResult(score=1.0, reason=f"Correct: {model_answer_clean}")
    else:
        return EvaluateResult(score=0.0, reason=f"Wrong: {model_answer_clean} vs {ground_truth_clean}")

def extract_model_answer(text: str) -> str:
    """Extract answer from model response, prioritizing our learned format"""

    # Method 1: [RESULT] tags (primary method for our SFT model)
    result_match = re.search(r'\\[RESULT\\](.*?)\\[/RESULT\\]', text, re.DOTALL)
    if result_match:
        return result_match.group(1).strip()

    # Method 2: \\boxed{} format (fallback)
    boxed_match = re.search(r'\\\\boxed\\{([^}]+)\\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Method 3: Last significant number in text
    numbers = re.findall(r'\\b(\\d+(?:,\\d{3})*(?:\\.\\d+)?)\\b', text)
    if numbers:
        significant_numbers = [n for n in numbers if float(n.replace(',', '')) >= 1]
        if significant_numbers:
            return significant_numbers[-1]

    return None

def clean_answer(answer_str: str) -> str:
    """Clean and normalize answer"""
    if not answer_str:
        return ""

    # Remove whitespace, commas, dollar signs
    cleaned = re.sub(r'[,$\\s]', '', str(answer_str).strip())

    # Convert to int if whole number
    try:
        if '.' in cleaned:
            float_val = float(cleaned)
            if float_val.is_integer():
                return str(int(float_val))
            else:
                return str(float_val)
        else:
            return str(int(cleaned))
    except ValueError:
        return cleaned
'''

# Save the evaluator
rft_evaluator_filename = "kd-rft-evaluator.py"
with open(rft_evaluator_filename, 'w') as f:
    f.write(rft_evaluator_code)
```

### Setting Up the RFT Training Job

**Manual Setup Required**: Due to the complexity of reinforcement
learning, some setup must be done through the Fireworks dashboard.

Upload Evaluator function

1.  Go to <https://app.fireworks.ai/dashboard/evaluations>
2.  Click \'Create Evaluator\'
3.  Name: `kd-rft-evaluator`
4.  Upload the RFT dataset we created
5.  Copy-paste the evaluator code from kd-rft-evaluator.py
6.  Save the evaluator

Then create RFT job:

1.  Navigate to the Fine-Tuning tab.
2.  Click \"Fine-Tune a Model\" and select Reinforcement.
3.  Configure the job:

-   Model Selection: Select the model. (the model that\'s already fine
    tuned using sft; `job.output_model` to find the name)
-   Dataset: Select the `kd-rft-dataset` you uploaded.
-   Evaluator: Select the `kd-rft-evaluator` you just created.
-   Rollout: You can leave these as the default values.
-   Optional Settings: You can leave the Model Output Name blank and get
    the default name, or enter a name of your choosing. Store this name;
    it will be required in the next cell.

1.  You can leave most other hyperparameters as their defaults.
2.  Click \"Create Job\".

### Deploying the Fine-Tuned Model {#deploying-the-fine-tuned-model}

``` python
rft_llm = LLM(
    model=<rft-model-output-name>,
    deployment_type="on-demand",
    id="kd-rft-model",
    min_replica_count=0,
    max_replica_count=1
)
rft_llm.apply()
```

## Chapter 7: Evaluate Complete Knowledge Distillation Pipeline

Now that we\'ve completed our two-stage knowledge distillation pipeline
(SFT for format learning, RFT for accuracy improvement), it\'s time to
evaluate our results. But first, we need robust evaluation tools that
can handle the complexity of comparing different models fairly.

**Why We Need Sophisticated Evaluation Tools**

The Challenge: We now have models that may respond in different formats:

-   Baseline model: Natural language, inconsistent formatting
-   RFT model: Structured \[WORK\]/\[RESULT\] format

**The Problem**: Simple string matching won\'t work because:
```
    # These are all the same answer but look different:
    response_1 = "The answer is 42 dollars"
    response_2 = "[RESULT]\n42\n[/RESULT]"  
    response_3 = "Therefore, the total is $42.00"
    response_4 = "\\boxed{42}"
```

We need evaluation tools that can:

-   Extract answers from any response format
-   Normalize numbers (handle commas, decimals, currency)
-   Track multiple metrics (accuracy, extraction success, timing)

**Building Our Evaluation System**

Let\'s build two essential functions that will power our model
comparisons:

**Answer Extraction Engine**

``` python
def extract_answer(text: str) -> Optional[str]:
    """
    Answer extraction that tries multiple methods
    """
    # Method 0: [RESULT] tags (primary method for our SFT model)
    result_match = re.search(r'\[RESULT\](.*?)\[/RESULT\]', text, re.DOTALL)
    if result_match:
        answer = result_match.group(1).strip()
        number = extract_number_from_text(answer)
        if number:
            return number

    # Method 1: <answer> tags
    answer_tag_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if answer_tag_match:
        answer = answer_tag_match.group(1).strip()
        number = extract_number_from_text(answer)
        if number:
            return number

    # Method 2: \\boxed{} format
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        number = extract_number_from_text(boxed_match.group(1))
        if number:
            return number

    # Method 3: Last number in the entire text
    all_numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text)
    if all_numbers:
        # Filter out small numbers that might be step numbers
        significant_numbers = [n for n in all_numbers if float(n.replace(',', '')) >= 1]
        if significant_numbers:
            return clean_number(significant_numbers[-1])

    # Method 4: "Therefore" or conclusion patterns
    conclusion_patterns = [
        r'[Tt]herefore,?\s+.*?(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Ss]o,?\s+.*?(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Tt]hus,?\s+.*?(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Ii]n total,?\s+.*?(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Aa]ltogether,?\s+.*?(\d+(?:,\d{3})*(?:\.\d+)?)',
    ]

    for pattern in conclusion_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return clean_number(matches[-1])  # Take the last match

    # Method 5: "The answer is" patterns
    answer_is_patterns = [
        r'[Tt]he answer is\s+(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Aa]nswer:\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Ff]inal answer:\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
    ]

    for pattern in answer_is_patterns:
        match = re.search(pattern, text)
        if match:
            return clean_number(match.group(1))

    # Method 6: Numbers at the end of sentences
    sentences = text.split('.')
    for sentence in reversed(sentences[-3:]):  # Check last 3 sentences
        numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', sentence)
        if numbers:
            return clean_number(numbers[-1])

    return None

def extract_number_from_text(text: str) -> Optional[str]:
    """Extract the main number from a piece of text"""
    # Look for numbers, prioritizing larger ones
    numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text)
    if numbers:
        return clean_number(numbers[-1])  # Take the last/most significant number
    return None

def clean_number(number_str: str) -> str:
    """Clean and normalize number string"""
    # Remove commas and extra whitespace
    cleaned = number_str.replace(',', '').strip()

    # Convert to int if it's a whole number
    try:
        if '.' in cleaned:
            float_val = float(cleaned)
            if float_val.is_integer():
                return str(int(float_val))
            else:
                return str(float_val)
        else:
            return str(int(cleaned))
    except ValueError:
        return cleaned
```

**Evaluation System**

``` python
def evaluate_model(MODEL, deployment_id, problems):
    """Evaluate model"""

    results = []
    correct = 0
    total = 0
    extraction_failures = 0

    for i in range(0, len(problems)):
      problem = problems[i]

      # Get model response
      llm = LLM(
        model=MODEL,
        deployment_type="on-demand",
        id=deployment_id  # The deployment ID that already exists
      )

      response = llm.chat.completions.create(
          messages=[{"role": "user", "content": problem["question"]}]
      )
      model_response = response.choices[0].message.content
      model_answer = extract_answer(model_response)  # Use answer extraction
      ground_truth = problem["ground_truth"]

      # Track extraction failures
      if model_answer is None:
          extraction_failures += 1

      # Check correctness (only if we extracted something)
      is_correct = model_answer == ground_truth if model_answer else False
      if is_correct:
          correct += 1
      total += 1

    accuracy = correct / total if total > 0 else 0

    return accuracy
```

### Test Model Performance

``` python
base_accuracy = evaluate_model(
    "qwen2p5-7b",
    "kd-base-model",
    gsm8k_test_problems
)

rft_accuracy = evaluate_model(
    rft_model_name,
    "kd_rft_model",
    gsm8k_test_problems
)
```

### Actual Results Analysis
```
    ACCURACY PROGRESSION:
    Base Model:  52%
    → RFT:       70% (+18pp)
    Total Gain:  +18 percentage point improvement

    FORMAT COMPLIANCE:
    SFT Model:  ~95% use [WORK]/[RESULT] format automatically  
    RFT Model:  ~95% maintain format + higher accuracy
```

## Course Summary and Key Takeaways

### What We Demonstrated

**1. SFT for Internalized Format Learning**:

-   **Training Strategy**: Include format examples without system
    prompts in training data
-   **Testing Strategy**: No system prompts needed - format is
    internalized\
-   **Result**: Model automatically uses `[WORK]/[RESULT]` structure as
    default behavior
-   **Key Insight**: SFT teaches \"how to respond\" by making patterns
    the model\'s natural behavior

**2. RFT for Accuracy Improvement**:

-   **Foundation**: Builds on SFT model
-   **Optimization**: Reward-based learning improves content quality
    within learned structure
-   **Result**: Maintains format compliance while significantly
    improving reasoning accuracy
-   **Key Insight**: RFT optimizes \"what to respond with\" while
    preserving structural learning

**3. Two-Stage Pipeline Synergy**:

-   **Stage 1 (SFT)**: Establishes reliable, consistent response
    structure
-   **Stage 2 (RFT)**: Optimizes content quality within that structure
-   **Combined Result**: Models that are both well-formatted AND
    accurate

### Practical Applications

This knowledge distillation approach is valuable for:

-   **API Integrations**: Reliable output parsing + improved accuracy
-   **Structured Reasoning Tasks**: Clear thinking process + better
    results\
-   **Production Pipelines**: Consistent format + higher quality content
-   **Evaluation Systems**: Easy answer extraction + improved
    performance
-   **Cost Optimization**: Small models with large model capabilities

### Expected Timeline and Resources

-   **Data Generation**: \~30 minutes (1000 examples)
-   **SFT Training**: \~45 minutes (format learning)
-   **RFT Training**: \~90 minutes (accuracy optimization)\
-   **Total Pipeline**: \~3 hours for complete format + accuracy
    improvement
-   **Cost**: \~TBD in compute for complete pipeline

## Conclusion

This tutorial demonstrated how to systematically apply knowledge
distillation using Fireworks AI\'s platform to create models that
combine the structural reliability of supervised learning with the
performance optimization of reinforcement learning.

**Key Success Factors**:

1.  **Clear separation of concerns**: SFT for structure, RFT for
    accuracy
2.  **Consistent evaluation methodology**: Test without system prompts
    to measure true learning
3.  **Building on foundations**: RFT builds on SFT rather than starting
    from scratch
4.  **Quality training data**: High teacher model accuracy and format
    consistency

The result is a compact, efficient model that maintains the reasoning
capabilities and output structure of much larger models, making it
suitable for production deployment at significantly lower cost and
latency.

**Next Steps**: Apply this methodology to your own domain-specific tasks
by:

1.  Defining appropriate output formats for your use case
2.  Generating high-quality teacher demonstrations
3.  Following the tuning pipeline
4.  Evaluating both structural and performance improvements

This systematic approach to knowledge distillation enables you to create
specialized, efficient models that retain the capabilities of their
larger teachers while being practical for real-world deployment.
