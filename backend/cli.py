"""
cli.py -- interactive terminal chat over the same agent the API uses.

    python cli.py --data data/sample_sales.csv
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from agent.data_loader import load_csv  # noqa: E402
from agent.llm_client import LLMError  # noqa: E402
from agent.orchestrator import run_agent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a CSV dataset.")
    parser.add_argument("--data", required=True, help="Path to a CSV file.")
    args = parser.parse_args()

    df = load_csv(args.data)
    print(f"Loaded '{args.data}': {len(df)} rows, {len(df.columns)} columns.")
    print("Columns:", ", ".join(map(str, df.columns)))
    print("\nAsk questions about this data. Type 'exit' to quit.\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            result = run_agent(df, question, verbose=True)
        except (RuntimeError, LLMError) as e:
            print(f"\n{e}\n")
            continue
        print(f"\n{result['answer']}")
        for path in result["plot_files"]:
            print(f"(chart saved to: {path})")
        print()


if __name__ == "__main__":
    main()
