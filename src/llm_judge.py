import os
import json
import logging
import argparse
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

# Datapizza imports
from datapizza.clients.google import GoogleClient
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.pipeline import DagPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Wrapper to adapt GoogleClient to Pipeline expected interface
class GeneratorWrapper:
    def __init__(self, client):
        self.client = client
    
    def run(self, input, **kwargs):
        prompt_text = input
        # Simple extraction if it's a Memory object or similar
        if hasattr(input, "to_string"):
             prompt_text = input.to_string()
        elif hasattr(input, "get_string"):
             prompt_text = input.get_string()
        elif hasattr(input, "messages"):
             prompt_text = "\n".join([str(m) for m in input.messages])
        elif not isinstance(input, str):
             prompt_text = str(input)

        return self.client.invoke(input=prompt_text)

    def __call__(self, input, **kwargs):
        return self.run(input, **kwargs)

class SustainabilityJudge:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")

        # 1. LLM Client
        self.llm = GoogleClient(
            model="gemini-2.0-flash", 
            api_key=self.api_key
        )

        # 2. Pipeline Construction
        self.pipeline = DagPipeline()

        # Step A: Prompt Template
        # We place the dynamic content in the 'user_prompt' variable which is standard for Datapizza
        user_template_str = """
        You are an expert sustainability analyst evaluating mobile games.
        
        TASK: Analyze the following app description to determine how it relates to sustainability education.
        
        {{ user_prompt }}

        Determine the following two DISTINCT classifications:

        1. DIRECTLY EDUCATIONAL (Explicit Intent):
           - YES if the app's PRIMARY GOAL is to *teach* specific sustainability concepts, facts, or behaviors (e.g. quizzes about recycling, "how-to" guides, educational simulators explicitly marketed for learning).
           - NO if it is a game primarily for entertainment, even if it has a nature theme.

        2. INDIRECTLY EDUCATIONAL (Embedded in Gameplay):
           - YES if the gameplay mechanics *themselves* require managing resources, balancing ecosystems, or solving pollution problems, which might incidently foster sustainability awareness or skills (e.g. a strategy tycoon where pollution hurts your city rating, or a survival game where you must protect nature to win).
           - NO if sustainability is just a visual theme/background (e.g. a Sudoku with forest pictures) or if the gameplay has zero ecological cause-and-effect.

        Note: An app can be BOTH (e.g. a fun game that is also explicitly educational) or NEITHER.

        Return ONLY a JSON object with no markdown formatting:
        {
          "DIRECTLY": "YES/NO",
          "INDIRECTLY": "YES/NO",
          "REASON_WHY": "Concise explanation distinguishing the intent vs mechanics."
        }
        """

        self.pipeline.add_module("prompt", ChatPromptTemplate(
            user_prompt_template=user_template_str,
            retrieval_prompt_template=""
        ))

        # Step B: Generator
        self.pipeline.add_module("generator", GeneratorWrapper(self.llm))

        # Connect Pipeline
        # Prompt (dict input) -> Generator (text response)
        self.pipeline.connect("prompt", "generator", target_key="input")

    def evaluate(self, title: str, description: str, score: str) -> Dict[str, str]:
        try:
            # Pre-format the prompt string ourselves since ChatPromptTemplate might be strict
            # or pass them as part of the 'user_prompt' input if supported.
            # Based on the error, ChatPromptTemplate.format() likely only accepts 'user_prompt' and 'retrieved_context'
            
            # WORKAROUND: We format the template strings in python, then pass the result as 'user_prompt'
            formatted_prompt = f"""
            App Title: {title}
            Description: {description}
            Context: A rules-based algorithm gave this a sustainability relevance score of {score} (0-1).
            """
            
            result = self.pipeline.run({
                "prompt": {
                    "user_prompt": formatted_prompt
                }
            })
            
            response_text = result["generator"].content
            
            # Handle case where content is a list (e.g. multiple candidates or chunks)
            if isinstance(response_text, list):
                # Join if it's a list of strings, or take first if it's objects?
                # Usually it's a list of strings or a single string.
                # Let's try to join if they are strings.
                try:
                    response_text = "".join([str(x) for x in response_text])
                except:
                    response_text = str(response_text)
            
            # Ensure it is a string
            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Clean up response if it contains markdown code blocks
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                # Try parsing JSON
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to find JSON blob if extra text exists
                import re
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"Error analyzing '{title}': {e}")
            return {
                "DIRECTLY": "ERROR",
                "INDIRECTLY": "ERROR",
                "REASON_WHY": f"Error: {str(e)}"
            }

def main():
    parser = argparse.ArgumentParser(description="LLM Judge for Sustainability")
    parser.add_argument("--infile", default="results_expanded_lexicon.csv", help="Input CSV file")
    parser.add_argument("--outfile", default="results_with_llm_judge.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows for testing (0 = no limit)")
    
    args = parser.parse_args()
    
    infile_path = args.infile
    if not os.path.exists(infile_path):
        # Fallback to older file if expanded one doesn't exist
        old_file = "results_excel_schema_with_sustainability.csv"
        if os.path.exists(old_file):
            print(f"File {infile_path} not found, falling back to {old_file}")
            infile_path = old_file
        else:
            print(f"Input file {infile_path} not found.")
            return

    # Use pandas for reading to handle the semicolon separator from lexicon.py
    try:
        # Try reading with semicolon first (lexicon.py output)
        df = pd.read_csv(infile_path, sep=";", encoding="utf-8")
        if "Description" not in df.columns:
            # If that fails to find columns, try comma
            df = pd.read_csv(infile_path, sep=",", encoding="utf-8")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Loaded {len(df)} rows from {infile_path}")

    judge = SustainabilityJudge()
    
    # Process rows
    llm_directly = []
    llm_indirectly = []
    llm_reason = []
    
    processed_count = 0
    
    for index, row in df.iterrows():
        if args.limit > 0 and processed_count >= args.limit:
            # Fill remaining with skip
            llm_directly.append("SKIPPED")
            llm_indirectly.append("SKIPPED")
            llm_reason.append("Limit reached")
            continue
            
        title = str(row.get("Title", ""))
        desc = str(row.get("Description", ""))
        score = str(row.get("Sustainability_Score", "0"))
        
        # Skip if description is empty or NaN
        if not desc or desc.lower() == "nan":
            llm_directly.append("N/A")
            llm_indirectly.append("N/A")
            llm_reason.append("No description")
        else:
            print(f"[{processed_count+1}] Analyzing: {title[:40]}...")
            res = judge.evaluate(title, desc, score)
            llm_directly.append(res.get("DIRECTLY", "N/A"))
            llm_indirectly.append(res.get("INDIRECTLY", "N/A"))
            llm_reason.append(res.get("REASON_WHY", "N/A"))
            
        processed_count += 1
        
        # Fill remaining if loop breaks early (not needed here but good practice)
        
    # Assign new columns
    # Ensure lists are same length as df
    while len(llm_directly) < len(df):
        llm_directly.append("SKIPPED")
        llm_indirectly.append("SKIPPED")
        llm_reason.append("Limit reached")
        
    df["LLM_Directly"] = llm_directly
    df["LLM_Indirectly"] = llm_indirectly
    df["LLM_Reason"] = llm_reason

    # Save output with semicolon to maintain format
    df.to_csv(args.outfile, sep=";", index=False, encoding="utf-8", decimal=",")
    print(f"Analysis complete. Results saved to {args.outfile}")

if __name__ == "__main__":
    main()
