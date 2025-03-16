/-- The product of two subalgebras is a subalgebra. -/
def prod : Subalgebra R (A × B) :=
  { S.toSubsemiring.prod S₁.toSubsemiring with
    carrier := S ×ˢ S₁
    algebraMap_mem' := fun _ => ⟨algebraMap_mem _ _, algebraMap_mem _ _⟩ }


@[simp]
theorem coe_prod : (prod S S₁ : Set (A × B)) = (S : Set A) ×ˢ (S₁ : Set B) :=
  rfl


open Subalgebra in
theorem prod_toSubmodule : toSubmodule (S.prod S₁) = (toSubmodule S).prod (toSubmodule S₁) := rfl


@[simp]
theorem mem_prod {S : Subalgebra R A} {S₁ : Subalgebra R B} {x : A × B} :
    x ∈ prod S S₁ ↔ x.1 ∈ S ∧ x.2 ∈ S₁ := Set.mem_prod


@[simp]
                                                               /-
                                                                 R : Type u_1
                                                                 A : Type u_2
                                                                 B : Type u_3
                                                                 inst✝⁴ : CommSemiring R
                                                                 inst✝³ : Semiring A
                                                                 inst✝² : Algebra R A
                                                                 inst✝¹ : Semiring B
                                                                 inst✝ : Algebra R B
                                                                 ⊢ Eq (Top.top.prod Top.top) Top.top
                                                               -/
theorem prod_top : (prod ⊤ ⊤ : Subalgebra R (A × B)) = ⊤ := by ext; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem prod_mono {S T : Subalgebra R A} {S₁ T₁ : Subalgebra R B} :
    S ≤ T → S₁ ≤ T₁ → prod S S₁ ≤ prod T T₁ :=
  Set.prod_mono


@[simp]
theorem prod_inf_prod {S T : Subalgebra R A} {S₁ T₁ : Subalgebra R B} :
    S.prod S₁ ⊓ T.prod T₁ = (S ⊓ T).prod (S₁ ⊓ T₁) :=
  SetLike.coe_injective Set.prod_inter_prod


