#!/usr/bin/env python3
"""
Convert ImProver prompts output to Agora-compatible Lean project.
Creates Targets/ directory with sorry-fied theorems marked with @[target].
"""
import json
import os
from pathlib import Path

PROMPTS_DIR = Path("/home/shivansg/ImProver/prompts/inftheory_prompts/src")
OUTPUT_DIR = Path("/data/user_data/shivansg/eval_InformationTheory")
TOPIC = "InformationTheory"

def create_lean_file(theorems: list, module_name: str) -> str:
    """Create a Lean file with @[target] annotated sorry-fied theorems."""
    lines = [
        "import VerifiedAgora.tagger",
        f"-- Auto-generated from {module_name}",
        f"-- Total theorems: {len(theorems)}",
        "",
    ]
    
    for thm in theorems:
        content_sorry = thm.get("content_sorry", "").strip()
        if not content_sorry:
            continue
        
        # Get prescopes and postscopes for context
        prescopes = thm.get("prescopes", "").strip()
        postscopes = thm.get("postscopes", "").strip()
        
        # Add prescopes if present (namespace, variables, etc.)
        if prescopes and prescopes not in "\n".join(lines):
            lines.append(f"\n{prescopes}\n")
        
        # Add the sorry-fied theorem with @[target] annotation
        # Check if it already has decorators
        if content_sorry.startswith("@["):
            # Merge target into existing decorator
            content_sorry = content_sorry.replace("@[", "@[target, ", 1)
        else:
            # Add @[target] before theorem/lemma
            for kw in ["theorem", "lemma", "def"]:
                if kw in content_sorry:
                    content_sorry = content_sorry.replace(kw, f"@[target]\n{kw}", 1)
                    break
        
        lines.append(content_sorry)
        lines.append("")
        
        # Add postscopes if present and not already added
        if postscopes and postscopes not in "\n".join(lines):
            lines.append(f"\n{postscopes}\n")
    
    return "\n".join(lines)

def main():
    print(f"Converting ImProver prompts to Agora format...")
    print(f"Source: {PROMPTS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    targets_dir = OUTPUT_DIR / "Targets" / TOPIC
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    target_imports = []
    total_theorems = 0
    
    for json_file in PROMPTS_DIR.rglob("*.json"):
        with open(json_file) as f:
            theorems = json.load(f)
        
        if not theorems:
            continue
        
        # Determine output path
        rel_path = json_file.relative_to(PROMPTS_DIR / "Mathlib")
        lean_path = targets_dir / rel_path.with_suffix(".lean")
        lean_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get module name from first theorem
        module_name = theorems[0].get("id", {}).get("module", str(rel_path))
        
        # Create Lean content
        content = create_lean_file(theorems, module_name)
        lean_path.write_text(content)
        
        # Track imports
        module = f"Targets.{TOPIC}.{str(rel_path.with_suffix('')).replace('/', '.')}"
        target_imports.append(f"import {module}")
        total_theorems += len(theorems)
        
        print(f"  Created {lean_path.name} with {len(theorems)} theorems")
    
    # Write Targets.lean
    (OUTPUT_DIR / "Targets.lean").write_text(
        "-- Auto-generated import file for Targets\n" + "\n".join(target_imports) + "\n"
    )
    
    # Write lakefile
    lakefile = f'''import Lake
open Lake DSL

package eval_{TOPIC} {{
  -- Package configuration
}}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.17.0"

require "leanprover-community" / "batteries" @ git "v4.17.0"

require "leanprover" / "Cli" @ git "v4.17.0"

require VerifiedAgora from git
  "https://github.com/stagiralabs/VerifiedAgora.git" @ "v4.17.0"

@[default_target]
lean_lib Targets where

lean_lib Library where
'''
    (OUTPUT_DIR / "lakefile.lean").write_text(lakefile)
    
    # Write toolchain
    (OUTPUT_DIR / "lean-toolchain").write_text("leanprover/lean4:v4.17.0\n")
    
    # Write empty Library.lean
    (OUTPUT_DIR / "Library.lean").write_text("-- Auto-generated import file for Library\n")
    (OUTPUT_DIR / "Library").mkdir(exist_ok=True)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Total theorems: {total_theorems}")
    print(f"Target files: {len(target_imports)}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
