"""
Agentic Code/Kernel Generation System
Implements: Generator -> Executor -> Reflector -> Optimizer loop
"""

import os
import sys
import traceback
import subprocess
from typing import Dict, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# ====================
# AGENTIC KERNEL GENERATOR
# ====================
class AgenticKernelGenerator:
    """
    Multi-agent system for autonomous code generation with self-correction.
    Based on patterns from KernelFalcon and GEAK frameworks.
    """

    def __init__(self, api_key: str = None, model: str = "openai/gpt-oss-120b"):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self.max_iterations = 5
        self.conversation_history = []

    def generator_agent(self, task_description: str, previous_attempt: str = None, 
                       error_feedback: str = None) -> str:
        """
        Generator Agent: Creates or refines code based on task and feedback.
        """
        if previous_attempt is None:
            # Initial generation
            prompt = f"""You are an expert Python/GPU kernel programmer. Generate clean, efficient, and well-documented code.

TASK: {task_description}

REQUIREMENTS:
- Write complete, executable Python code
- Include proper imports
- Add a main execution block for testing
- Handle edge cases
- Use type hints where appropriate

Generate the code now:"""
        else:
            # Refinement after error
            prompt = f"""You are an expert Python programmer. The previous code attempt failed.

ORIGINAL TASK: {task_description}

PREVIOUS CODE:
```python
{previous_attempt}
```

ERROR FEEDBACK:
{error_feedback}

FIX THE CODE:
- Analyze the error carefully
- Fix the specific issue
- Ensure the code handles edge cases
- Keep the solution simple and correct

Generate the corrected code now:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower temperature for more deterministic code
        )

        print(f"\n[Generator Agent] Generated code snippet...")
        print(response.choices[0].message.content)

        generated_code = response.choices[0].message.content

        # Extract code from markdown blocks if present
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        return generated_code

    def executor_agent(self, code: str) -> Tuple[bool, str, str]:
        """
        Executor Agent: Runs code in isolated subprocess and captures results.
        Returns: (success, stdout, stderr)
        """
        try:
            # Write code to temporary file
            temp_file = "/tmp/agent_generated_code.py"
            with open(temp_file, "w") as f:
                f.write(code)

            # Execute in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Error: Code execution timed out (>10s)"
        except Exception as e:
            return False, "", f"Execution error: {str(e)}"

    def reflector_agent(self, code: str, error_output: str) -> str:
        """
        Reflector Agent: Analyzes errors and provides structured feedback.
        """
        prompt = f"""You are a code debugging expert. Analyze the following error and provide concise feedback.

CODE:
```python
{code}
```

ERROR:
{error_output}

Provide a clear, actionable analysis:
1. What is the root cause of the error?
2. What specific changes are needed?
3. Are there any edge cases to consider?

Keep your response under 200 words and focus on actionable fixes."""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        print(f"\n[Reflector Agent] Generated feedback...")
        print(response.choices[0].message.content)

        return response.choices[0].message.content

    def generate(self, task_description: str, verbose: bool = True) -> Dict:
        """
        Main orchestrator: Coordinates all agents in iterative loop.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TASK: {task_description}")
            print(f"{'='*60}\n")

        current_code = None
        error_feedback = None

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- ITERATION {iteration + 1}/{self.max_iterations} ---")

            # Step 1: Generate/Refine Code
            if verbose:
                print("🤖 Generator Agent: Creating code...")
            current_code = self.generator_agent(
                task_description, 
                current_code, 
                error_feedback
            )

            if verbose:
                print("\n" + "-"*40)
                print(f"\n📝 Generated Code (excerpt):")
                print(current_code)
                print("\n" + "-"*40)

            # Step 2: Execute Code
            if verbose:
                print("\n⚙️  Executor Agent: Running code...")
            success, stdout, stderr = self.executor_agent(current_code)

            if success:
                if verbose:
                    print("✅ SUCCESS! Code executed without errors.")
                    if stdout:
                        print("\n" + "-"*40)
                        print(f"\n📤 Output:\n{stdout}")
                        print("\n" + "-"*40)

                return {
                    "success": True,
                    "code": current_code,
                    "iterations": iteration + 1,
                    "output": stdout
                }

            # Step 3: Reflect on Error
            if verbose:
                print("\n" + "-"*40)
                print(f"❌ Execution failed. Error:\n{stderr}")
                print("\n🔍 Reflector Agent: Analyzing error...")
                print("\n" + "-"*40)


            error_feedback = self.reflector_agent(current_code, stderr)

            if verbose:
                print("\n" + "-"*40)
                print(f"\n💡 Feedback: {error_feedback}")
                print("\n" + "-"*40)

        # Max iterations reached
        return {
            "success": False,
            "code": current_code,
            "iterations": self.max_iterations,
            "error": "Max iterations reached without successful execution"
        }


# ====================
# EXAMPLE USAGE
# ====================

if __name__ == "__main__":
    # Initialize agent system
    agent = AgenticKernelGenerator()

    # Example 1: Simple optimization task
    print("\n" + "="*60)
    print("EXAMPLE 1: Matrix Operations")
    print("="*60)

    task1 = task_triton_softmax = """
Generate a Triton GPU kernel for a 'Fused Softmax' operation.
Requirements:
1. Input: A 2D tensor (rows x cols).
2. Operation: Apply Softmax along the last dimension (rows).
   - Step A: Find Max of row (for numerical stability).
   - Step B: Calculate Exponentials: exp(x - max).
   - Step C: Sum Exponentials.
   - Step D: Divide by Sum.
3. OPTIMIZATION: Do this all in one kernel load (fused) to minimize HBM access.
4. Use `tl.load` with masking to handle rows that aren't multiples of BLOCK_SIZE.
5. Include a `triton.jit` decorator.
"""



    result1 = agent.generate(task1, verbose=True)

    # print("result1 =", result1["code"])

    # # Example 2: More complex task with potential errors
    # print("\n\n" + "="*60)
    # print("EXAMPLE 2: Performance Benchmark")
    # print("="*60)

    # task2 = """
    # Create a Python script that:
    # 1. Implements a simple moving average function
    # 2. Benchmarks it with timeit on random data (1000 elements)
    # 3. Prints the execution time in milliseconds
    # 4. Validates correctness with a test case
    # """

    # result2 = agent.generate(task2, verbose=True)

    # print("result2 =", result2["code"])

    # Display final results
    print("\n\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Task 1: {'SUCCESS' if result1['success'] else 'FAILED'} in {result1['iterations']} iterations")
    # print(f"Task 2: {'SUCCESS' if result2['success'] else 'FAILED'} in {result2['iterations']} iterations")