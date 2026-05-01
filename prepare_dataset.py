import csv
import json
import os
from pathlib import Path

def main():
    # Setup paths
    current_dir = Path(__file__).parent
    # Check for the csv in root or inside customer-support-agents/data
    csv_file = current_dir / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    if not csv_file.exists():
        csv_file = current_dir / "customer-support-agents" / "data" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    output_file = current_dir / "training_data.jsonl"
    
    print(f"Reading from: {csv_file}")
    print(f"Writing to: {output_file}")
    
    count = 0
    with open(csv_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        for row in reader:
            category = row.get('category', '').strip()
            intent = row.get('intent', '').strip()
            instruction = row.get('instruction', '').strip()
            response = row.get('response', '').strip()
            
            if not instruction or not response:
                continue
                
            # Build proper RAG-like context for the model
            system_prompt = (
                "You are an expert customer support specialist. Context:\n"
                f"- Category: {category}\n"
                f"- Intent: {intent}\n\n"
                "Provide clear, actionable solutions to customer issues."
            )
            
            # Azure Mistral Chat Format
            message_obj = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
            
            f_out.write(json.dumps(message_obj) + '\n')
            count += 1
            
    print(f"Successfully processed {count} records and created {output_file.name}")

if __name__ == "__main__":
    main()
