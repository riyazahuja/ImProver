import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import os
# HuggingFace tokenizer fork‑safety
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
import json
import datetime
import multiprocessing
import argparse
import duckdb
from transformers import AutoTokenizer
import re



def _parse_json(val):
    if val is None:
        return []
    if isinstance(val, str):
        import json
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return val



def run_inference(df, args):
    # assuming gpus sit behind different PCIe host bridges on separate
    # NUMA sockets (i.e. nvidia-smi topo -m shows SYS between gpus)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    ray.init(
        num_cpus=args.cpus, num_gpus=args.gpus
    )  # , _temp_dir='/home/riyaza/ray_tmp')
    DataContext.get_current().wait_for_min_actors_s = 1800
    ctx = DataContext.get_current()
    # ctx.progress_bar = True
    # ctx.execution_options.verbose_progress = True

    assert Version(ray.__version__) >= Version(
        "2.44.1"
    ), "Ray version must be at least 2.44.1"
    # print(df)
    # ds = ray.data.from_pandas(df)
    

    # Create a new dataframe with duplicated rows, each with a unique prompt_idx
    df2_parts = []
    for i in range(args.n):
        df_copy = df.copy()
        df_copy["prompt_idx"] = i
        df2_parts.append(df_copy)

    df2 = pd.concat(df2_parts, ignore_index=True)
    # Use df2 instead of df for the Ray dataset
    ds = ray.data.from_pandas(df2).drop_columns(
        [
            "C1Dependencies",
            "C2Dependencies",
            "C3Dependencies",
            "text",
            "informalStatement",
            "informalProof",
            "errorMsgs",
            "isExtracted",
        ]
    ).repartition(args.gpus * 8)
    # ds = ray.data.from_pandas(df).repartition(args.gpus * 4)
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    print(ds.schema())

    size = ds.count()
    print(f"Size of dataset: {size} prompts")

    # ctx.execution_options = ExecutionOptions(task_extra_resources={"CPU": 0.25})

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_resources={"CPU": args.cpus // args.gpus, "GPU": 1},
        concurrency=args.gpus,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "enable_chunked_prefill": True,
            "max_model_len": 16384,
            "max_num_batched_tokens": 65536,
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
        },
        max_concurrent_batches=32,
        batch_size=32,
    )

    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["raw_prompt"]}],
            sampling_params=dict(
                # n=args.n,
                truncate_prompt_tokens=16384-512,
                # temperature=0.3,
                max_tokens=512,
            ),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )
    ds = vllm_processor(ds).materialize()

    # id = f"RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = os.path.join("knowledge_graphs", args.KG_id)
    os.makedirs(run_output_dir, exist_ok=True)

    output_path = os.path.join(run_output_dir, "filtered_data")

    config = {
        # "dataset": args.dataset_path,
        # "split": args.split,
        "n": args.n,
        "model": args.model,
    }

    config_path = os.path.join(run_output_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    ds.repartition(16).write_parquet(f"local://{output_path}")

    # con = duckdb.connect(os.path.join(run_output_dir, "filter_data.duckdb"))


    return run_output_dir


def get_thm_prompt(thm):

    if thm.get("isExtracted") or not thm.get("isOriginal", False):
        return None

    # Dependencies (be forgiving about NULL / malformed JSON)
    c1 = _parse_json(thm.get("C1Dependencies"))
    c2 = _parse_json(thm.get("C2Dependencies"))
    all_deps = c1 + c2
    
    dependencies_raw = [t["content"].strip() for t in all_deps]

    dependencies = "\n\n".join(
        [
            f"<DEPENDENCY index={i}>\n{dep}\n</DEPENDENCY>"
            for i, dep in enumerate(dependencies_raw)
        ]
    )

    example = r"""
<EXAMPLE>

// INPUT:

<CURRENT>

/-- If $H$ is a finite subgroup of $G$, and $\rho(U_H) \leq r$, then there exists $t$ such
that $|A \cap (H+t)| \geq e^{-r} \sqrt{|A||H|}$, and $|H|/|A| \in [e^{-2r}, e^{2r}]$. -/
lemma rho_of_subgroup [IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G}
    (hunif : IsUniform H U μ) {A : Finset G} (hA : A.Nonempty) (hU : Measurable U)
    (r : ℝ) (hr : ρ[U ; μ # A] ≤ r) :
    ∃ t : G,
      exp (-r) * Nat.card A ^ (1/2 : ℝ) * Nat.card H ^ (1/2 : ℝ) ≤
        Nat.card ↑(↑A ∩ (t +ᵥ (H : Set G)))
      ∧ Nat.card A ≤ exp (2 * r) * Nat.card H
      ∧ Nat.card H ≤ exp (2 * r) * Nat.card A := by
  have hr' : ρ[U ; μ # A] ≤ r := hr
  have Hpos : 0 < (Nat.card H : ℝ) := by exact_mod_cast Nat.card_pos
  have : Nonempty A := hA.to_subtype
  have Apos : 0 < (Nat.card A : ℝ) := by exact_mod_cast Nat.card_pos
  simp only [rho] at hr
  rw [rhoMinus_of_subgroup hunif hA hU, rhoPlus_of_subgroup hunif hA hU] at hr
  rcases exists_card_inter_add_eq_sSup (A := A) H hA with ⟨t, ht, hpos⟩
  rw [← ht] at hr
  have Rm : 0 ≤ ρ⁻[U ; μ # A] := rhoMinus_nonneg hU
  have RM : 0 ≤ ρ⁺[U ; μ # A] := by
    rw [rhoPlus_of_subgroup hunif hA hU, ← ht, sub_nonneg]
    apply log_le_log (mod_cast hpos)
    norm_cast
    have : Nat.card (t +ᵥ (H : Set G) : Set G) = Nat.card H := by
      apply Nat.card_image_of_injective (add_right_injective t)
    rw [← this]
    exact Nat.card_mono (toFinite _) inter_subset_right
  have I : |log (Nat.card H) - log (Nat.card A)| ≤ 2 * r := calc
    |log (Nat.card H) - log (Nat.card A)|
    _ = |H[U ; μ] - log (Nat.card A)| := by rw [hunif.entropy_eq' (toFinite _) hU]; rfl
    _ = |ρ⁺[U ; μ # A] - ρ⁻[U ; μ # A]| := by congr 1; simp [rhoPlus]; abel
    _ ≤ ρ⁺[U ; μ # A] + ρ⁻[U ; μ # A] :=
      (abs_sub _ _).trans_eq (by simp [abs_of_nonneg, Rm, RM])
    _ = 2 * ρ[U ; μ # A] := by simp [rho]; ring
    _ ≤ 2 * r := by linarith
  refine ⟨t, ?_, ?_, ?_⟩
  · have : - r + (log (Nat.card A) + log (Nat.card H)) * (1 / 2 : ℝ) ≤
      log (Nat.card (A ∩ (t +ᵥ (H : Set G)) : Set G)) := by linarith
    have := exp_monotone this
    rwa [exp_add, exp_log (mod_cast hpos), exp_mul, exp_add,
      exp_log Hpos, exp_log Apos, mul_rpow, ← mul_assoc] at this <;> positivity
  · have : log (Nat.card A) ≤ 2 * r + log (Nat.card H) := by
      linarith [(abs_sub_le_iff.1 I).2]
    have := exp_monotone this
    rwa [exp_log Apos, exp_add, exp_log Hpos] at this
  · have : log (Nat.card H) ≤ 2 * r + log (Nat.card A) := by
      linarith [(abs_sub_le_iff.1 I).1]
    have := exp_monotone this
    rwa [exp_log Hpos, exp_add, exp_log Apos] at this

</CURRENT>

<DEPENDENCY index=0>
theorem abs_sub_le_iff : |a - b| ≤ c ↔ a - b ≤ c ∧ b - a ≤ c := by
  rw [abs_le, neg_le_sub_iff_le_add, sub_le_iff_le_add', and_comm, sub_le_iff_le_add']
</DEPENDENCY>

<DEPENDENCY index=1>
@[to_additive]
theorem mul_assoc : ∀ a b c : G, a * b * c = a * (b * c) :=
  Semigroup.mul_assoc
</DEPENDENCY>

<DEPENDENCY index=2>
@[to_additive] lemma mabs_of_one_le (h : 1 ≤ a) : |a|ₘ = a :=
</DEPENDENCY>

<DEPENDENCY index=3>
theorem abs_sub (a b : α) : |a - b| ≤ |a| + |b| := by
  rw [sub_eq_add_neg, ← abs_neg b]
  exact abs_add a _
</DEPENDENCY>

<DEPENDENCY index=4>
lemma hU [IsProbabilityMeasure (ℙ : Measure Ω)] : H[U] = H[X₁' + X₂'] :=
  IdentDistrib.entropy_eq (h₁.add h₂
    (h_indep.indepFun (show (0 : Fin 4) ≠ 1 by norm_cast))
     (h_indep.indepFun (show (2 : Fin 4) ≠ 3 by norm_cast)))
</DEPENDENCY>

<DEPENDENCY index=5>
lemma card_mono (ht : t.Finite) (h : s ⊆ t) : Nat.card s ≤ Nat.card t :=
  toNat_le_toNat (mk_le_mk_of_subset h) ht.lt_aleph0
</DEPENDENCY>

<DEPENDENCY index=6>
@[to_additive]
</DEPENDENCY>

<DEPENDENCY index=7>
lemma card_image_of_injective {f : α → β} (hf : Injective f) (s : Set α) :
    Nat.card (f '' s) = Nat.card s := card_image_of_injOn hf.injOn
</DEPENDENCY>

<DEPENDENCY index=8>
@[to_additive (attr := simp) sub_nonneg]
</DEPENDENCY>

<DEPENDENCY index=9>
@[simp] lemma card_pos [Nonempty α] [Finite α] : 0 < Nat.card α := card_pos_iff.2 ⟨‹_›, ‹_›⟩
</DEPENDENCY>

<DEPENDENCY index=10>
theorem extracted_1.{u_1, uG} {G : Type uG} [inst : AddCommGroup G] [hGm : MeasurableSpace G]
  [inst_1 : DiscreteMeasurableSpace G] {Ω : Type u_1} [inst_2 : MeasurableSpace Ω] {μ : Measure Ω}
  [inst_3 : IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G} (hunif : IsUniform (↑H) U μ) {A : Finset G}
  (hA : A.Nonempty) (hU : Measurable U) (r : ℝ) (hr' : ρ[U ; μ # A] ≤ r) (Hpos : 0 < ↑(Nat.card ↥H))
  (this : Nonempty { x // x ∈ A }) (Apos : 0 < ↑(Nat.card { x // x ∈ A })) (t : G)
  (hr :
    (log ↑(Nat.card { x // x ∈ A }) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) +
          (log ↑(Nat.card ↥H) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))))) /
        2 ≤
      r)
  (ht : Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = sSup {x | ∃ t, Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = x})
  (hpos : 0 < Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) (Rm : 0 ≤ ρ⁻[U ; μ # A]) (RM : 0 ≤ ρ⁺[U ; μ # A]) :
  |log ↑(Nat.card ↥H) - log ↑(Nat.card { x // x ∈ A })| = |H[U ; μ] - log ↑(Nat.card { x // x ∈ A })| := by
  hunif.entropy_eq' (toFinite _) hU]; rfl
</DEPENDENCY>

<DEPENDENCY index=11>
theorem extracted_1.{u_1, uG} {G : Type uG} [inst : AddCommGroup G] [hGm : MeasurableSpace G]
  [inst_1 : DiscreteMeasurableSpace G] {Ω : Type u_1} [inst_2 : MeasurableSpace Ω] {μ : Measure Ω}
  [inst_3 : IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G} (hunif : IsUniform (↑H) U μ) {A : Finset G}
  (hA : A.Nonempty) (hU : Measurable U) (r : ℝ) (hr' : ρ[U ; μ # A] ≤ r) (Hpos : 0 < ↑(Nat.card ↥H))
  (this : Nonempty { x // x ∈ A }) (Apos : 0 < ↑(Nat.card { x // x ∈ A })) (t : G)
  (hr :
    (log ↑(Nat.card { x // x ∈ A }) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) +
          (log ↑(Nat.card ↥H) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))))) /
        2 ≤
      r)
  (ht : Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = sSup {x | ∃ t, Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = x})
  (hpos : 0 < Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) (Rm : 0 ≤ ρ⁻[U ; μ # A]) (RM : 0 ≤ ρ⁺[U ; μ # A]) :
  |H[U ; μ] - log ↑(Nat.card { x // x ∈ A })| = |ρ⁺[U ; μ # A] - ρ⁻[U ; μ # A]| := by
  congr 1; simp [rhoPlus]; abel
</DEPENDENCY>

<DEPENDENCY index=12>
theorem extracted_1.{u_1, uG} {G : Type uG} [inst : AddCommGroup G] [hGm : MeasurableSpace G]
  [inst_1 : DiscreteMeasurableSpace G] {Ω : Type u_1} [inst_2 : MeasurableSpace Ω] {μ : Measure Ω}
  [inst_3 : IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G} (hunif : IsUniform (↑H) U μ) {A : Finset G}
  (hA : A.Nonempty) (hU : Measurable U) (r : ℝ) (hr' : ρ[U ; μ # A] ≤ r) (Hpos : 0 < ↑(Nat.card ↥H))
  (this : Nonempty { x // x ∈ A }) (Apos : 0 < ↑(Nat.card { x // x ∈ A })) (t : G)
  (hr :
    (log ↑(Nat.card { x // x ∈ A }) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) +
          (log ↑(Nat.card ↥H) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))))) /
        2 ≤
      r)
  (ht : Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = sSup {x | ∃ t, Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = x})
  (hpos : 0 < Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) (Rm : 0 ≤ ρ⁻[U ; μ # A]) (RM : 0 ≤ ρ⁺[U ; μ # A]) :
  |ρ⁺[U ; μ # A]| + |ρ⁻[U ; μ # A]| = ρ⁺[U ; μ # A] + ρ⁻[U ; μ # A] := by
  simp [abs_of_nonneg, Rm, RM]
</DEPENDENCY>

<DEPENDENCY index=13>
theorem extracted_1.{u_1, uG} {G : Type uG} [inst : AddCommGroup G] [hGm : MeasurableSpace G]
  [inst_1 : DiscreteMeasurableSpace G] {Ω : Type u_1} [inst_2 : MeasurableSpace Ω] {μ : Measure Ω}
  [inst_3 : IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G} (hunif : IsUniform (↑H) U μ) {A : Finset G}
  (hA : A.Nonempty) (hU : Measurable U) (r : ℝ) (hr' : ρ[U ; μ # A] ≤ r) (Hpos : 0 < ↑(Nat.card ↥H))
  (this : Nonempty { x // x ∈ A }) (Apos : 0 < ↑(Nat.card { x // x ∈ A })) (t : G)
  (hr :
    (log ↑(Nat.card { x // x ∈ A }) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) +
          (log ↑(Nat.card ↥H) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))))) /
        2 ≤
      r)
  (ht : Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = sSup {x | ∃ t, Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = x})
  (hpos : 0 < Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) (Rm : 0 ≤ ρ⁻[U ; μ # A]) (RM : 0 ≤ ρ⁺[U ; μ # A]) :
  ρ⁺[U ; μ # A] + ρ⁻[U ; μ # A] = 2 * ρ[U ; μ # A] := by
  simp [rho]; ring
</DEPENDENCY>

<DEPENDENCY index=14>
theorem extracted_1.{u_1, uG} {G : Type uG} [inst : AddCommGroup G] [hGm : MeasurableSpace G]
  [inst_1 : DiscreteMeasurableSpace G] {Ω : Type u_1} [inst_2 : MeasurableSpace Ω] {μ : Measure Ω}
  [inst_3 : IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G} (hunif : IsUniform (↑H) U μ) {A : Finset G}
  (hA : A.Nonempty) (hU : Measurable U) (r : ℝ) (hr' : ρ[U ; μ # A] ≤ r) (Hpos : 0 < ↑(Nat.card ↥H))
  (this : Nonempty { x // x ∈ A }) (Apos : 0 < ↑(Nat.card { x // x ∈ A })) (t : G)
  (hr :
    (log ↑(Nat.card { x // x ∈ A }) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) +
          (log ↑(Nat.card ↥H) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))))) /
        2 ≤
      r)
  (ht : Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = sSup {x | ∃ t, Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = x})
  (hpos : 0 < Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) (Rm : 0 ≤ ρ⁻[U ; μ # A]) (RM : 0 ≤ ρ⁺[U ; μ # A]) :
  2 * ρ[U ; μ # A] ≤ 2 * r := by
  linarith
</DEPENDENCY>

<DEPENDENCY index=15>
theorem extracted_1.{u_1, uG} {G : Type uG} [inst : AddCommGroup G] [hGm : MeasurableSpace G]
  [inst_1 : DiscreteMeasurableSpace G] {Ω : Type u_1} [inst_2 : MeasurableSpace Ω] {μ : Measure Ω}
  [inst_3 : IsProbabilityMeasure μ] {H : AddSubgroup G} {U : Ω → G} (hunif : IsUniform (↑H) U μ) {A : Finset G}
  (hA : A.Nonempty) (hU : Measurable U) (r : ℝ) (hr' : ρ[U ; μ # A] ≤ r) (Hpos : 0 < ↑(Nat.card ↥H))
  (this : Nonempty { x // x ∈ A }) (Apos : 0 < ↑(Nat.card { x // x ∈ A })) (t : G)
  (hr :
    (log ↑(Nat.card { x // x ∈ A }) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) +
          (log ↑(Nat.card ↥H) - log ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))))) /
        2 ≤
      r)
  (ht : Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = sSup {x | ∃ t, Nat.card ↑(↑A ∩ (t +ᵥ ↑H)) = x})
  (hpos : 0 < Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) (Rm : 0 ≤ ρ⁻[U ; μ # A]) (RM : 0 ≤ ρ⁺[U ; μ # A])
  (I : |log ↑(Nat.card ↥H) - log ↑(Nat.card { x // x ∈ A })| ≤ 2 * r) :
  ∃ t,
    rexp (-r) * ↑(Nat.card { x // x ∈ A }) ^ (1 / 2) * ↑(Nat.card ↥H) ^ (1 / 2) ≤ ↑(Nat.card ↑(↑A ∩ (t +ᵥ ↑H))) ∧
      ↑(Nat.card { x // x ∈ A }) ≤ rexp (2 * r) * ↑(Nat.card ↥H) ∧
        ↑(Nat.card ↥H) ≤ rexp (2 * r) * ↑(Nat.card { x // x ∈ A }) := by
  refine ⟨t, ?_, ?_, ?_⟩
    · have : - r + (log (Nat.card A) + log (Nat.card H)) * (1 / 2 : ℝ) ≤
        log (Nat.card (A ∩ (t +ᵥ (H : Set G)) : Set G)) := by linarith
      have := exp_monotone this
      rwa [exp_add, exp_log (mod_cast hpos), exp_mul, exp_add,
        exp_log Hpos, exp_log Apos, mul_rpow, ← mul_assoc] at this <;> positivity
    · have : log (Nat.card A) ≤ 2 * r + log (Nat.card H) := by
        linarith [(abs_sub_le_iff.1 I).2]
      have := exp_monotone this
      rwa [exp_log Apos, exp_add, exp_log Hpos] at this
    · have : log (Nat.card H) ≤ 2 * r + log (Nat.card A) := by
        linarith [(abs_sub_le_iff.1 I).1]
      have := exp_monotone this
      rwa [exp_log Hpos, exp_add, exp_log Apos
</DEPENDENCY>




//OUTPUT:
<CORE_DEPENDENCIES>4,15</CORE_DEPENDENCIES>
</EXAMPLE>
"""

    prompt = f"""

Given the following Lean4 theorem (wrapped with <CURRENT>...</CURRENT>), and all its dependencies/lemmas (each wrapped in <DEPENDENCY>, with an index) return the indices of the "core" lemmas. Namely, a core lemma is a dependency that intuitively embodies fundamental ideas relevant to the proof of the current theorem. This is in contrast to purely technical or helper lemmas.

Return your answer as a comma-seperated list of indices, wrapped in <CORE_DEPENDENCIES>...</CORE_DEPENDENCIES> tags.


For example, consider the following example input within the <EXAMPLE>...</EXAMPLE> tags. The correct answer is given in the <EXAMPLE_ANSWER>...</EXAMPLE_ANSWER> tags:

{example.strip()}

Note that this is a heuristic, and so your output should be given by reasoning at a high level about the structure of the proof to determine dependencies, rather than by only looking at the specific details of the code. After your informal reasoning, be sure to output your answer as a comma-separated list of indices, wrapped in <CORE_DEPENDENCIES>...</CORE_DEPENDENCIES> tags. Now, here is the current theorem and its dependencies that you must reason about and examine:


<CURRENT>

{thm['text'].strip()}

</CURRENT>

{dependencies}

"""

    return {"raw_prompt": prompt, **thm}



def construct_KG_data(con, args):

  output_path = os.path.join("knowledge_graphs", args.KG_id, "filter_data")
      #safe mode
    # con.execute(
    #     f"""
    #     CREATE TABLE IF NOT EXISTS run_data AS
    #     SELECT * FROM read_parquet('{output_path}/*.parquet');
    # """
    # )
  
  #unsafe but probably better lol
  con.execute("DROP TABLE IF EXISTS run_data;")
  con.execute(f"CREATE TABLE run_data AS SELECT * FROM read_parquet('{output_path}/*.parquet');")
  # Add a core_dependencies column initialized to "[]" if it doesn't exist
  con.execute("""
    ALTER TABLE run_data ADD COLUMN IF NOT EXISTS core_dependencies JSON DEFAULT '[]';
  """)
  
  # Step 1: Get all unique (name, module) pairs
  pairs = con.execute("""
    SELECT DISTINCT name, module
    FROM run_data
  """).fetchall()
  
  # Process each (name, module) pair
  for name, module in pairs:
    # print(f"Processing ({name}, {module})")
    # Get all rows for this (name, module) pair
    rows = con.execute(f"""
      SELECT answer
      FROM run_data
      WHERE name = '{name.replace("'","''")}' AND module = '{module.replace("'","''")}'
    """).fetchall()
    
    # Skip if we don't have the expected number of rows
    if len(rows) != args.n:
      print(f"Warning: Expected {args.n} rows for ({name}, {module}) but found {len(rows)}")
      continue
    
    # Process the answers for this group
    core_dependencies = []
    valid_format = False
    
    for row in rows:
      answer = row[-1]  # Assuming answer is the last column
      # Look for the pattern <CORE_DEPENDENCIES>...</CORE_DEPENDENCIES>
      match = re.search(r"<CORE_DEPENDENCIES>([\d+,\s]+)</CORE_DEPENDENCIES>", answer)
      if match:
        # Extract the comma-separated list of integers
        deps_str = match.group(1).strip()
        try:
          # Parse the comma-separated list into integers
          deps = [int(d.strip()) for d in deps_str.split(',') if d.strip()]
          core_dependencies = deps
          valid_format = True
          break
        except ValueError:
          continue
    
    # Update all rows for this (name, module) with the core_dependencies
    if valid_format:
      # Convert list to JSON string for storage
      deps_json = json.dumps(core_dependencies)

      # Now update the column with the core dependencies
      con.execute(f"""
        UPDATE run_data
        SET core_dependencies = '{deps_json}'
        WHERE name = '{name.replace("'","''")}' AND module = '{module.replace("'","''")}'
      """)
      
      
def get_training_dataset(con, args):

  # Get rows with non-empty core_dependencies, selecting just one per (name, module)
  query = """
    SELECT 
      ANY_VALUE(raw_prompt) as raw_prompt, 
      ANY_VALUE(answer) as answer,
      name,
      module
    FROM run_data
    WHERE 
      json_array_length(core_dependencies) > 0 OR 
      core_dependencies != '[]'
    GROUP BY name, module
  """
  
  rows = con.execute(query).fetchall()
  print(f"Found {len(rows)} rows with non-empty core_dependencies")
  
  # Create a list of dictionary entries for the JSONL file
  jsonl_entries = []
  for row in rows:
    raw_prompt, answer, name, module = row
    jsonl_entries.append({
      "instruction": raw_prompt,
      "output": answer
    })
  
  # Write the JSONL file
  output_path = os.path.join("knowledge_graphs", args.KG_id, "training.jsonl")
  with open(output_path, "w") as f:
    for entry in jsonl_entries:
      f.write(json.dumps(entry) + "\n")
  
  print(f"Wrote {len(jsonl_entries)} entries to {output_path}")


def main(args):
  
    MAX_PROMPT_TOKENS = 16384 - 512   # model context minus generation tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)


    if args.run_inference:
      
      combined_dataset_path = os.path.join("knowledge_graphs", args.KG_id, "combined.duckdb")
      con = duckdb.connect(combined_dataset_path)
      
      df_raw = con.execute("SELECT * FROM theorems").df()
      
      
      
      prompts = []
      truncation_count = 0
      for _,row in df_raw.iterrows():
        thm_prompt = get_thm_prompt(row.to_dict())
        if thm_prompt is None:
          continue
                      

        # Token‑level truncation / filtering
        tokens = tokenizer.encode(thm_prompt["raw_prompt"],
                                  add_special_tokens=False)
        if len(tokens) > MAX_PROMPT_TOKENS:
            # ----- OPTION A: truncate to last MAX_PROMPT_TOKENS tokens
            tokens = tokens[-MAX_PROMPT_TOKENS:]
            thm_prompt["raw_prompt"] = tokenizer.decode(tokens)
            truncation_count += 1

            # ----- OPTION B: drop the prompt entirely instead
            # continue    # ← uncomment this line & delete the two lines above to drop

        prompts.append(thm_prompt)

      # Convert the list of prompts to a pandas DataFrame
      print(f"Total prompts: {len(prompts)}")
      print(f"Truncated prompts: {truncation_count}")
      df = pd.DataFrame(prompts)

      output_path = run_inference(df, args)
      print(f"Run inference complete. Output saved to {output_path}")
    
    database_path = os.path.join("knowledge_graphs", args.KG_id, "filtered_data.duckdb")
    con = duckdb.connect(database_path)
    if args.augment_DB:
      construct_KG_data(con, args)
      print("Constructed KG data and updated run_data table.")
    if args.training_data:
      get_training_dataset(con, args)
      print("Generated training dataset and saved to training.jsonl.")
    
    
    
    
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter KG for ImProver")
    parser.add_argument("KG_id", type=str)

    # parser.add_argument(
    #     "--KG_dir",
    #     type=str,
    #     default=".knowledge_graphs",
    #     help="Directory to get KG (default: .knowledge_graphs)",
    # )
    parser.add_argument(
        "--heuristic_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model to use",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use (default: all available)",
    )

    try:
        available_gpus = torch.cuda.device_count()
    except (ImportError, AttributeError):
        available_gpus = 0

    parser.add_argument(
        "--gpus",
        type=int,
        default=available_gpus,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Majority vote value (default: 1)"
    )
    parser.add_argument(
        "--run_inference", type=bool, action=argparse.BooleanOptionalAction , default=True, help="Whether to run inference (default: True)"
    )
    parser.add_argument(
        "--augment_DB",  action=argparse.BooleanOptionalAction , type=bool, default=True, help="Whether to augment filteredDB (default: True)"
    )
    parser.add_argument(
        "--training_data", action=argparse.BooleanOptionalAction , type=bool, default=True, help="Whether to generate training data (default: True)"
    )

    args = parser.parse_args()


    main(args)
