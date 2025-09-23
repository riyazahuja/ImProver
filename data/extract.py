#!/usr/bin/env python3
"""
list_unexpected_files.py

Recursively walk a directory, find every file whose *base name* is **not**
in the allow-list below, and write those paths to a JSON file.
"""

from pathlib import Path
import json
import argparse
import sys

# ─── 1  EDIT THIS LIST ──────────────────────────────────────────────────────────
# Put *file names* (not paths) that you want to keep/ignore here.
allow_raw = [
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/Basic.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.ι_commute_fieldOpFreeAlgebra_superCommuteF_ofCrAnOpF_ofCrAnOpF"
        ]
      },
      {
        "file": "HepLean/Lorentz/SL2C/SelfAdjoint.lean",
        "theorems": [
          "Lorentz.SL2C.toSelfAdjointMap_mul"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/Sign/Join.lean",
        "theorems": [
          "WickContraction.join_singleton_sign_right"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldStatistics/Basic.lean",
        "theorems": [
          "FieldStatistic.ofList_freeMonoid"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/UncontractedList.lean",
        "theorems": [
          "WickContraction.fin_list_sorted_indexOf_mem"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/TimeOrder.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.timeOrderF_timeOrderF_left"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/InsertAndContract.lean",
        "theorems": [
          "WickContraction.insertAndContract_isSome_getDual?_self"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/NormalOrder/Basic.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.ι_normalOrderF_zero_of_mem_ideal"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/NormalOrder/Lemmas.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.normalOrder_ofFieldOpList_mul_anPart_swap"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/Grading.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.bosonicProjF_of_fermionic_part"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/SuperCommute.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.superCommute_ofCrAnList_ofFieldOpList"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/Basic.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.mulLinearMap_apply"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/Grading.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.fermionicProjFree_zero_of_ι_zero"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/NormalOrder/Lemmas.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.normalOrder_superCommute_eq_zero"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/SubContraction.lean",
        "theorems": [
          "WickContraction.quotContraction_fstFieldOfContract_uncontractedListEmd"
        ]
      },
      {
        "file": "HepLean/Mathematics/List/InsertIdx.lean",
        "theorems": [
          "HepLean.List.get_eq_insertIdx_succAbove"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/Join.lean",
        "theorems": [
          "WickContraction.join_fstFieldOfContract_joinLiftLeft"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/NormalOrder/Basic.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.ι_normalOrderF_superCommuteF_ofCrAnListF_ofCrAnListF_eq_zero_mul"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/TimeOrder.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.ι_timeOrderF_superCommuteF_neq_time"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/ExtractEquiv.lean",
        "theorems": [
          "WickContraction.extractEquiv_equiv"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/TimeOrder.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.ι_timeOrderF_zero_of_mem_ideal"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/Sign/Join.lean",
        "theorems": [
          "WickContraction.join_singleton_signFinset_eq_filter"
        ]
      },
      {
        "file": "HepLean/Mathematics/Fin/Involutions.lean",
        "theorems": [
          "HepLean.Fin.involutionAddEquiv_cast"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/Koszul/KoszulSign.lean",
        "theorems": [
          "Wick.koszulSign_perm_eq_append"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/SuperCommute.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.ofFieldOp_mul_ofFieldOp_eq_superCommute"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldSpecification/Filters.lean",
        "theorems": [
          "FieldSpecification.annihilateFilter_cons_create"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/SuperCommute.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.superCommuteF_ofCrAnListF_ofCrAnListF"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/NormTimeOrder.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.normTimeOrder_ofCrAnListF"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/SuperCommute.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.superCommute_ofCrAnList_ofFieldOpList_eq_sum"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/TimeCond.lean",
        "theorems": [
          "WickContraction.eqTimeContractSet_of_not_haveEqTime"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/TimeOrder.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.timeOrder_ofFieldOpList_singleton"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/Grading.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.fermionicProj_mem_bosonic"
        ]
      },
      {
        "file": "HepLean/Mathematics/Fin/Involutions.lean",
        "theorems": [
          "HepLean.Fin.involutionNoFixed_card_succ"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/Sign/InsertNone.lean",
        "theorems": [
          "WickContraction.signInsertNone_eq_filter_map"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/NormalOrder.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.normalOrderF_superCommuteF_annihilate_create"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/Basic.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.fermionicProjF_mem_ideal"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/Basic.lean",
        "theorems": [
          "WickContraction.sndFieldOfContract_mem"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/Singleton.lean",
        "theorems": [
          "WickContraction.of_singleton_eq"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/NormalOrder/Lemmas.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.normalOrder_normalOrder_left"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/SuperCommute.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.summerCommute_jacobi_ofCrAnListF"
        ]
      },
      {
        "file": "HepLean/Mathematics/Fin/Involutions.lean",
        "theorems": [
          "HepLean.Fin.involutionNoFixed_card_odd"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/SubContraction.lean",
        "theorems": [
          "WickContraction.mem_subContraction_or_quotContraction"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/UncontractedList.lean",
        "theorems": [
          "WickContraction.uncontractedList_getElem_uncontractedIndexEquiv_symm"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpAlgebra/Grading.lean",
        "theorems": [
          "FieldSpecification.FieldOpAlgebra.bosonicProj_bosonicProj_eq_bosonicProj"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/Koszul/KoszulSign.lean",
        "theorems": [
          "Wick.koszulSign_singleton"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/InsertAndContractNat.lean",
        "theorems": [
          "WickContraction.insertLift_injective"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/CreateAnnihilate.lean",
        "theorems": [
          "CreateAnnihilate.sum_eq"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldOpFreeAlgebra/SuperCommute.lean",
        "theorems": [
          "FieldSpecification.FieldOpFreeAlgebra.anPartF_mul_crPartF_eq_superCommuteF"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/FieldStatistics/Basic.lean",
        "theorems": [
          "FieldStatistic.ofList_append"
        ]
      },
      {
        "file": "HepLean/PerturbationTheory/WickContraction/UncontractedList.lean",
        "theorems": [
          "WickContraction.uncontractedListEmd_mem_uncontracted"
        ]
      }]


ALLOW_LIST = set([
      item['file'] for item in allow_raw
])
# ────────────────────────────────────────────────────────────────────────────────

def collect_extra_files(root: Path, allowed: set[str]) -> list[str]:
    """
    Return relative paths (POSIX style) of every file under `root`
    whose *name* is not in `allowed`.
    """
    extras: list[str] = []
    for path in root.rglob("*"):           # recursive glob
        if path.is_file() and path.name not in allowed:
            # store paths relative to the root, using forward slashes
            extras.append("HepLean/"+path.relative_to(root).as_posix())
    return sorted(extras)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find files not in an allow-list and save them as JSON."
    )
    ap.add_argument(
        "directory",
        type=Path,
        help="Directory to scan (walks sub-directories recursively)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="extra_files.json",
        type=Path,
        help="Where to write the resulting JSON file (default: %(default)s)",
    )
    args = ap.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"Error: {args.directory} is not a directory")

    extras = collect_extra_files(args.directory, ALLOW_LIST)

    # Write the list to disk
    args.out.write_text(json.dumps(extras, indent=2))
    print(f"Wrote {len(extras)} paths to {args.out}")

if __name__ == "__main__":
    main()