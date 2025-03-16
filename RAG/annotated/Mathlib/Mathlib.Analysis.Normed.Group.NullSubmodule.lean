variable (M) in
/-- The null subgroup with respect to the norm. -/
@[to_additive "The additive null subgroup with respect to the norm."]
def nullSubgroup : Subgroup M where
  carrier := {x : M | ‖x‖ = 0}
  mul_mem' {x y} (hx : ‖x‖ = 0) (hy : ‖y‖ = 0) := by
    /-
      M : Type u_1
      inst✝ : SeminormedCommGroup M
      x y : M
      hx : Eq (Norm.norm x) 0
      hy : Eq (Norm.norm y) 0
      ⊢ Membership.mem (setOf fun x => Eq (Norm.norm x) 0) (HMul.hMul x y)
    -/
    apply le_antisymm _ (norm_nonneg' _)
    /-
      M : Type u_1
      inst✝ : SeminormedCommGroup M
      x y : M
      hx : Eq (Norm.norm x) 0
      hy : Eq (Norm.norm y) 0
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) 0
    -/
    refine (norm_mul_le' x y).trans_eq ?_
    /-
      M : Type u_1
      inst✝ : SeminormedCommGroup M
      x y : M
      hx : Eq (Norm.norm x) 0
      hy : Eq (Norm.norm y) 0
      ⊢ Eq (HAdd.hAdd (Norm.norm x) (Norm.norm y)) 0
    -/
    rw [hx, hy, add_zero]
    /-
      🎉 no goals
    -/
  one_mem' := norm_one'
                                    /-
                                      M : Type u_1
                                      inst✝ : SeminormedCommGroup M
                                      x : M
                                      hx : Eq (Norm.norm x) 0
                                      ⊢ Membership.mem { carrier := setOf fun x => Eq (Norm.norm x) 0, mul_mem' := ⋯ …
                                    -/
  inv_mem' {x} (hx : ‖x‖ = 0) := by simpa only [Set.mem_setOf_eq, norm_inv'] using hx
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive]
lemma isClosed_nullSubgroup : IsClosed (nullSubgroup M : Set M) := by
  /-
    M : Type u_1
    inst✝ : SeminormedCommGroup M
    ⊢ IsClosed ↑(nullSubgroup M)
  -/
  apply isClosed_singleton.preimage continuous_norm'
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma mem_nullSubgroup_iff {x : M} : x ∈ nullSubgroup M ↔ ‖x‖ = 0 := Iff.rfl


variable (𝕜 E) in
/-- The null space with respect to the norm. -/
def nullSubmodule : Submodule 𝕜 E where
  __ := nullAddSubgroup E
  smul_mem' c x (hx : ‖x‖ = 0) := by
    /-
      M : Type u_1
      inst✝⁴ : SeminormedCommGroup M
      𝕜 : Type u_2
      E : Type u_3
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      x : E
      hx : Eq (Norm.norm x) 0
      ⊢ Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul c x)
    -/
    apply le_antisymm _ (norm_nonneg _)
    /-
      M : Type u_1
      inst✝⁴ : SeminormedCommGroup M
      𝕜 : Type u_2
      E : Type u_3
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      x : E
      hx : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm (HSMul.hSMul c x)) 0
    -/
    refine (norm_smul_le _ _).trans_eq ?_
    /-
      M : Type u_1
      inst✝⁴ : SeminormedCommGroup M
      𝕜 : Type u_2
      E : Type u_3
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      c : 𝕜
      x : E
      hx : Eq (Norm.norm x) 0
      ⊢ Eq (HMul.hMul (Norm.norm c) (Norm.norm x)) 0
    -/
    rw [hx, mul_zero]
    /-
      🎉 no goals
    -/


lemma isClosed_nullSubmodule : IsClosed (nullSubmodule 𝕜 E : Set E) := isClosed_nullAddSubgroup


@[simp]
lemma mem_nullSubmodule_iff {x : E} : x ∈ nullSubmodule 𝕜 E ↔ ‖x‖ = 0 := Iff.rfl

