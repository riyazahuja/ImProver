import re
import argparse
import duckdb
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Hard-coded metric specifications (value: "min" or "max")
METRIC_SPECIFICATIONS = {
    "length": "min",
    "declarativity": "max",
    "completion": "min",
    "dependency": "min",
    "readability": "max"
}

def get_db_connection(db_path):
    """Establishes a connection to a DuckDB database."""
    try:
        return duckdb.connect(database=str(db_path), read_only=False) # Read_only False to allow creating tables if raw.duckdb
    except Exception as e:
        print(f"Error connecting to database {db_path}: {e}")
        return None

def load_config(config_path):
    """Loads the JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if "metric" not in config or "n" not in config:
            print(f"Error: 'metric' or 'n' not found in config file {config_path}")
            return None
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None

def parse_json_field_as_float(value):
    """Safely parses a JSON field (which might be a string or number) as float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        # If it's a JSON string representing a number, try to parse that
        try:
            return float(json.loads(value))
        except (json.JSONDecodeError, ValueError, TypeError):
            # print(f"Warning: Could not parse value '{value}' as float.")
            return None # Or handle as an error, or return 0.0?

# --- Experimental Measures ---

def calculate_accuracy(rows):
    """Calculates accuracy: number of rows with new_correct as true / total rows."""
    if not rows:
        return 0.0
    correct_count = sum(1 for row in rows if row.get('new_correct'))
    return correct_count / len(rows)

def calculate_nonzero_accuracy(rows, metric_name, metric_spec_map):
    """
    Calculates nonzero accuracy:
    Number of rows with new_correct as true AND (delta < 0 if min, else delta > 0) / total rows.
    """
    if not rows:
        return 0.0
    
    metric_objective = metric_spec_map.get(metric_name)
    if not metric_objective:
        print(f"Warning: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS for nonzero_accuracy.")
        return 0.0 # Or handle error

    relevant_rows = 0
    for row in rows:
        delta = parse_json_field_as_float(row.get('delta'))
        if row.get('new_correct') and delta is not None:
            if metric_objective == "min" and delta < 0:
                relevant_rows += 1
            elif metric_objective == "max" and delta > 0:
                relevant_rows += 1
    return relevant_rows / len(rows)

def calculate_improvement(rows, metric_name, metric_spec_map):
    """
    Calculates improvement: average delta.
    If delta is null, set to zero.
    If delta > 0 and metric is min, delta for that row is zero (and vice versa for max).
    """
    if not rows:
        return 0.0
    
    metric_objective = metric_spec_map.get(metric_name)
    if not metric_objective:
        print(f"Warning: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS for improvement.")
        return 0.0

    effective_deltas = []
    for row in rows:
        delta = parse_json_field_as_float(row.get('delta'))
        if delta is None:
            delta = 0.0
        
        if metric_objective == "min":
            effective_delta = delta if delta < 0 else 0.0
        elif metric_objective == "max":
            effective_delta = delta if delta > 0 else 0.0
        else: # Should not happen if metric_objective is validated
            effective_delta = 0.0 
        effective_deltas.append(effective_delta)
        
    return sum(effective_deltas) / len(rows) if effective_deltas else 0.0

def calculate_nonzero_improvement(rows, metric_name, metric_spec_map):
    """
    Calculates nonzero improvement:
    Average delta value across rows with delta not null AND (delta < 0 if min, else delta > 0).
    """
    metric_objective = metric_spec_map.get(metric_name)
    if not metric_objective:
        print(f"Warning: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS for nonzero_improvement.")
        return 0.0

    relevant_deltas = []
    for row in rows:
        delta = parse_json_field_as_float(row.get('delta'))
        if delta is not None:
            if metric_objective == "min" and delta < 0:
                relevant_deltas.append(delta)
            elif metric_objective == "max" and delta > 0:
                relevant_deltas.append(delta)
                
    return sum(relevant_deltas) / len(relevant_deltas) if relevant_deltas else 0.0

# --- Analysis Commands ---

def run_best_of_n_analysis(run_id, run_dir_path, db_con, config):
    """Performs the Best-of-N analysis."""
    print("Starting Best-of-N analysis...")
    metric_name = config['metric']
    n_config_val = int(config['n'])
    metric_objective = METRIC_SPECIFICATIONS.get(metric_name)

    if not metric_objective:
        print(f"Error: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS.")
        return False

    analysis_base_path = run_dir_path / run_id / "analysis" / "BoN"
    analysis_base_path.mkdir(parents=True, exist_ok=True)
    
    raw_db_path = analysis_base_path / "raw.duckdb"
    plot_path = analysis_base_path / "BoN.png"
    csv_path = analysis_base_path / "data.csv"

    tick_size = max(math.floor(n_config_val / 16), 1)
    
    # Ensure '1' is the first tick, and N_config is the last.
    # Intermediate ticks are t, 2t, ..., up to 15t.
    tick_n_values = [1] 
    for i in range(1, 16): # t, 2t, ..., 15t
        val = i * tick_size
        if val < n_config_val and val not in tick_n_values : # ensure distinct and less than N
             tick_n_values.append(val)
    if n_config_val not in tick_n_values:
         tick_n_values.append(n_config_val)
    tick_n_values = sorted(list(set(tick_n_values))) # Unique, sorted

    print(f"Config: metric='{metric_name}' (objective: {metric_objective}), N={n_config_val}, tick_size={tick_size}")
    print(f"Tick n values for analysis: {tick_n_values}")

    try:
        distinct_pairs_query = "SELECT DISTINCT decl, module FROM evaluation_results"
        distinct_pairs = db_con.execute(distinct_pairs_query).fetchall()
        if not distinct_pairs:
            print("No (decl, module) pairs found in eval.duckdb. Aborting BoN.")
            return False
        
        # Fetch all data once to avoid multiple queries per (decl,module) inside loop
        # Assuming rowid or an implicit order gives us the "L[0]...L[n]" behavior
        all_data_query = "SELECT original_prompt,og_score,og_raw,list_transform(og_errors, x -> CAST(x as VARCHAR))::VARCHAR[] AS og_errors,new_trimmed, new_score,new_raw,list_transform(new_errors, x -> CAST(x as VARCHAR))::VARCHAR[] AS new_errors,new_correct, module, delta, decl,rowid FROM evaluation_results ORDER BY module, decl, rowid" # Added rowid for stable slicing
        all_data_df = db_con.execute(all_data_query).fetchdf()
        # print(all_data_df)
        # Convert to list of dicts for easier processing as in original plan
        all_data_rows = all_data_df.to_dict('records')
    except Exception as e:
        print(f"Database error during BoN setup: {e}")
        return False

    graph_data_points = []
    collected_rows_for_raw_db = [] # For the final N_config tick

    # Pre-group data by (decl, module)
    grouped_data = {}
    for row in all_data_rows:
        pair = (row['decl'], row['module'])
        if pair not in grouped_data:
            grouped_data[pair] = []
        grouped_data[pair].append(row)


    for n_val in tick_n_values:
        current_tick_best_rows_for_measures = [] # Rows for calculating measures at this n_val

        for decl_val, module_val in distinct_pairs:
            pair_key = (decl_val, module_val)
            # L_dm = db_con.execute(f"SELECT * FROM eval WHERE decl = ? AND module = ? ORDER BY rowid LIMIT ?", # Assuming rowid implies order
            #                       [decl_val, module_val, n_val]).fetchall()
            # L_dm_dicts = [dict(zip([col[0] for col in db_con.description], row)) for row in L_dm]
            
            L_dm_all_for_pair = grouped_data.get(pair_key, [])
            sub_list_for_dm = L_dm_all_for_pair[:n_val]


            if not sub_list_for_dm:
                # print(f"No data for ({decl_val}, {module_val}) at n_val={n_val}")
                continue

            correct_rows = [r for r in sub_list_for_dm if r.get('new_correct')]
            
            best_row_in_sub_list = None
            if not correct_rows:
                best_row_in_sub_list = sub_list_for_dm[0] # Arbitrary
            else:
                # Find best by new_score
                # Assuming new_score is a simple numeric value or can be cast
                best_row_in_sub_list = correct_rows[0] # Default if scores are problematic
                best_score_val = parse_json_field_as_float(best_row_in_sub_list.get('new_score'))

                for row_cr in correct_rows[1:]:
                    current_score_val = parse_json_field_as_float(row_cr.get('new_score'))
                    if current_score_val is None: continue # Skip if score unparsable

                    if best_score_val is None: # First parsable score becomes best
                         best_score_val = current_score_val
                         best_row_in_sub_list = row_cr
                         continue

                    if metric_objective == "min" and current_score_val < best_score_val:
                        best_score_val = current_score_val
                        best_row_in_sub_list = row_cr
                    elif metric_objective == "max" and current_score_val > best_score_val:
                        best_score_val = current_score_val
                        best_row_in_sub_list = row_cr
            
            if best_row_in_sub_list:
                current_tick_best_rows_for_measures.append(best_row_in_sub_list)
                if n_val == n_config_val: # Collect for raw.duckdb only at the final N_config tick
                    collected_rows_for_raw_db.append(best_row_in_sub_list)
        
        # Calculate measures for the current_tick_best_rows_for_measures
        if current_tick_best_rows_for_measures:
            acc = calculate_accuracy(current_tick_best_rows_for_measures)
            non_zero_acc = calculate_nonzero_accuracy(current_tick_best_rows_for_measures, metric_name, METRIC_SPECIFICATIONS)
            impr = calculate_improvement(current_tick_best_rows_for_measures, metric_name, METRIC_SPECIFICATIONS)
            non_zero_impr = calculate_nonzero_improvement(current_tick_best_rows_for_measures, metric_name, METRIC_SPECIFICATIONS)
            
            graph_data_points.append({
                'n_value': n_val,
                'accuracy': acc,
                'nonzero_accuracy': non_zero_acc,
                'improvement': impr,
                'nonzero_improvement': non_zero_impr
            })
            print(f"  n_val={n_val}: Acc={acc:.3f}, NonZeroAcc={non_zero_acc:.3f}, Impr={impr:.3f}, NonZeroImpr={non_zero_impr:.3f} (from {len(current_tick_best_rows_for_measures)} best rows)")
        else:
            print(f"  n_val={n_val}: No best rows found to calculate measures.")
            graph_data_points.append({
                'n_value': n_val, 'accuracy': 0, 'nonzero_accuracy': 0, 'improvement': 0, 'nonzero_improvement': 0
            })


    # Save collected_rows_for_raw_db to raw.duckdb
    if collected_rows_for_raw_db:
        print(f"Saving {len(collected_rows_for_raw_db)} rows to {raw_db_path}...")
        # Need to get column names and types from original table or ensure dict keys match
        # For simplicity, create a DataFrame then write to DuckDB
        # Remove 'rowid' if it was added and not part of original schema for raw.duckdb
        df_raw = pd.DataFrame([ {k:v for k,v in row.items() if k != 'rowid'} for row in collected_rows_for_raw_db])
        
        try:
            # If raw_db_path exists, DuckDB might error on connect if it's not a valid DB
            # It's safer to delete if exists, or use a new table name if appending
            if raw_db_path.exists():
                raw_db_path.unlink() # Remove old raw.duckdb to ensure clean write

            con_raw = duckdb.connect(database=str(raw_db_path))
            # Infer schema from DataFrame; ensure it matches original if necessary
            # For now, let DuckDB infer from DataFrame
            con_raw.execute("CREATE TABLE best_results AS SELECT * FROM df_raw")
            con_raw.close()
            print(f"Successfully saved to {raw_db_path}")
        except Exception as e:
            print(f"Error saving to raw.duckdb: {e}")
            return False # Indicate BoN failed if raw_db cannot be created
    else:
        print("No rows collected for raw.duckdb.")
        # This might be an issue if training command expects this file.
        # Create an empty table? Or let training command handle missing file.
        # For now, if it's empty, the file might not be created or will be empty.

    # Create and save CSV and plot
    if graph_data_points:
        df_graph = pd.DataFrame(graph_data_points)
        try:
            df_graph.to_csv(csv_path, index=False)
            print(f"Analysis data saved to {csv_path}")
            
            
            
            plt.figure(figsize=(12, 7))
            plt.plot(df_graph['n_value'], df_graph['accuracy'], marker='o', label='Accuracy')
            plt.plot(df_graph['n_value'], df_graph['nonzero_accuracy'], marker='s', label='Nonzero Accuracy')
            plt.plot(df_graph['n_value'], -df_graph['improvement'], marker='^', label='Improvement')
            plt.plot(df_graph['n_value'], -df_graph['nonzero_improvement'], marker='x', label='Nonzero Improvement')
            
            plt.xlabel("n Value (Number of samples considered per (decl,module))")
            plt.ylabel("Metric Value")
            plt.title(f"Best-of-N Analysis (Metric: {metric_name}, N_config: {n_config_val})")
            plt.legend()
            plt.grid(True)
            plt.xticks(tick_n_values) # Ensure all tick points are shown
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Analysis plot saved to {plot_path}")
            # plt.show() # Optionally show plot
            plt.close()

        except Exception as e:
            print(f"Error generating CSV/plot: {e}")
            # BoN might still be considered partially successful if raw_db was made
    else:
        print("No data points for graph and CSV.")

    print("Best-of-N analysis finished.")
    return True # Indicate success


def extract_improved_content(text):
    start_tag = "<IMPROVED>"
    end_tag = "</IMPROVED>"

    start_index = text.find(start_tag)
    end_index = text.find(end_tag)

    if start_index != -1 and end_index != -1 and end_index > start_index:
        # Both tags exist and properly ordered
        return text[start_index + len(start_tag):end_index]
    elif start_index != -1:
        # Only start tag found
        return text[start_index + len(start_tag):]
    elif end_index != -1:
        # Only end tag found
        return text[:end_index]
    else:
        # Neither tag found
        return text

def run_training_analysis(run_id, run_dir_path, config):
    """Performs the training data extraction analysis."""
    print("Starting training data extraction...")
    metric_name = config['metric']
    metric_objective = METRIC_SPECIFICATIONS.get(metric_name)

    if not metric_objective:
        print(f"Error: Metric '{metric_name}' not found in METRIC_SPECIFICATIONS.")
        return

    analysis_base_path = run_dir_path / run_id / "analysis" / "BoN"
    raw_db_path = analysis_base_path / "raw.duckdb"

    if not raw_db_path.exists():
        print(f"{raw_db_path} does not exist. Running Best-of-N analysis first...")
        # Need a connection to the original eval.duckdb for BoN
        eval_db_path = run_dir_path / run_id / "eval.duckdb"
        if not eval_db_path.exists():
            print(f"Error: eval.duckdb not found at {eval_db_path} for BoN pre-run.")
            return
        
        db_con_eval = get_db_connection(eval_db_path)
        if not db_con_eval:
            return 
        
        bon_success = run_best_of_n_analysis(run_id, run_dir_path, db_con_eval, config)
        db_con_eval.close()
        if not bon_success or not raw_db_path.exists():
            print("Best-of-N analysis failed or did not produce raw.duckdb. Cannot proceed with training analysis.")
            return
    
    # Connect to the raw_db_path (which should now exist)
    db_con_raw = get_db_connection(raw_db_path)
    if not db_con_raw:
        print(f"Failed to connect to {raw_db_path} for training analysis.")
        return

    try:
        # Query the 'best_results' table (assuming this is the table name used in BoN)
        query = "SELECT decl, module, new_raw, new_correct, delta, original_prompt FROM best_results"
        results = db_con_raw.execute(query).fetchall()
        
        # Convert to list of dicts for easier processing
        cols = [desc[0] for desc in db_con_raw.description]
        result_dicts = [dict(zip(cols, row)) for row in results]

        print("\n--- Training Data Candidates ---")
        count = 0
        pairs = []
        errors =[]
        for row in result_dicts:
            if row.get('new_correct'):
                delta = parse_json_field_as_float(row.get('delta'))
                if delta is not None:
                    condition_met = False
                    if metric_objective == "min" and delta < 0:
                        condition_met = True
                    elif metric_objective == "max" and delta > 0:
                        condition_met = True
                    
                    if condition_met:
                        print(f"DECL: {row['decl']}")
                        # print(f"MODULE: {row['module']}")
                        print(f"NEW_RAW: {row['new_raw']}")
                        print("---")
                        prompt = re.sub(r'\<\uFF5C.*?\uFF5C\>', '', row['original_prompt'])
                        new = extract_improved_content(row['new_raw'])
                        first = new.strip().split("\n")[0] if new else ""
                        if row['decl'] not in first and row['decl'].split(".")[-1] not in first:
                            new = row['new_raw']
                            errors.append(row['decl'])
                        # print(f"NEW: {new}")
                        pair = {"instruction": prompt.strip(), "output": new.strip() + "\n</IMPROVED>"}
                        pairs.append(pair)
                        
                        print(f"Pair: {new.strip()}")
                        # print(f"PROMPT: {re.sub(r'\<\uFF5C.*?\uFF5C\>', '', row['original_prompt'])}")
                        print("===")
                        count +=1
        print(f"Found {count} training data candidates.")
        json_output_path = analysis_base_path / "train.jsonl"
        if pairs:
            # print(pairs[:5])  # Print first 5 pairs for verification
            with open(json_output_path, 'w') as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
        print(f"Training data candidates saved to {json_output_path}")
        print(f"{len(errors)} errors in decls: {errors}")
    except Exception as e:
        print(f"Error during training data extraction: {e}")
    finally:
        db_con_raw.close()
    
    print("Training data extraction finished.")


def main():
    parser = argparse.ArgumentParser(description="Perform analysis on experimental run data.")
    parser.add_argument("RunID", help="Identifier for the run.")
    parser.add_argument("--run_dir", help="Path to the run directory.", default="runs/")
    args = parser.parse_args()

    run_id = args.RunID
    run_dir_path = Path(args.run_dir)

    db_path = run_dir_path / run_id / "eval.duckdb"
    config_path = run_dir_path / run_id / "config.json"

    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        return
    
    config = load_config(config_path)
    if not config:
        return

    # Main DB connection for eval.duckdb (used by BoN if called directly)
    # Training command might re-open it if BoN needs to be run first.
    # BoN itself also opens a connection to raw.duckdb.
    
    # For BoN, we pass the main eval.duckdb connection.
    # For Training, it handles its own connections (to raw.duckdb, and to eval.duckdb if BoN needs to run).

    while True:
        try:
            command = input("\nEnter command (Best-of-n, training, exit): ").strip().lower()
            if command == "best-of-n":
                # BoN needs connection to eval.duckdb
                db_con_eval = get_db_connection(db_path)
                if db_con_eval:
                    run_best_of_n_analysis(run_id, run_dir_path, db_con_eval, config)
                    db_con_eval.close()
                else:
                    print("Could not establish DB connection for Best-of-N.")
            elif command == "training":
                run_training_analysis(run_id, run_dir_path, config)
            elif command == "exit":
                print("Exiting.")
                break
            else:
                print("Unknown command. Available commands: Best-of-n, training, exit")
        except KeyboardInterrupt:
            print("\nExiting due to user interruption.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")


if __name__ == "__main__":
    main()
