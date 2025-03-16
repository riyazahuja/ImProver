/-- The boundary of an element of a co-Heyting algebra is the intersection of its Heyting negation
with itself. Note that this is always `⊥` for a boolean algebra. -/
def boundary (a : α) : α :=
  a ⊓ ￢a


/-- The boundary of an element of a co-Heyting algebra. -/
scoped[Heyting] prefix:120 "∂ " => Coheyting.boundary
-- Porting note: Should the notation be automatically included in the current scope?

theorem inf_hnot_self (a : α) : a ⊓ ￢a = ∂ a :=
  rfl


theorem boundary_le : ∂ a ≤ a :=
  inf_le_left


theorem boundary_le_hnot : ∂ a ≤ ￢a :=
  inf_le_right


@[simp]
theorem boundary_bot : ∂ (⊥ : α) = ⊥ := bot_inf_eq _


@[simp]
                                           /-
                                             α : Type u_1
                                             inst✝ : CoheytingAlgebra α
                                             ⊢ Eq (Coheyting.boundary Top.top) Bot.bot
                                           -/
theorem boundary_top : ∂ (⊤ : α) = ⊥ := by rw [boundary, hnot_top, inf_bot_eq]
                                           /-
                                             🎉 no goals
                                           -/


theorem boundary_hnot_le (a : α) : ∂ (￢a) ≤ ∂ a :=
  (inf_comm _ _).trans_le <| inf_le_inf_right _ hnot_hnot_le


@[simp]
theorem boundary_hnot_hnot (a : α) : ∂ (￢￢a) = ∂ (￢a) := by
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a : α
    ⊢ Eq (Coheyting.boundary (HNot.hnot (HNot.hnot a))) (Coheyting.boundary (HNot. …
  -/
  simp_rw [boundary, hnot_hnot_hnot, inf_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                               /-
                                                 α : Type u_1
                                                 inst✝ : CoheytingAlgebra α
                                                 a : α
                                                 ⊢ Eq (HNot.hnot (Coheyting.boundary a)) Top.top
                                               -/
theorem hnot_boundary (a : α) : ￢∂ a = ⊤ := by rw [boundary, hnot_inf_distrib, sup_hnot_self]
                                               /-
                                                 🎉 no goals
                                               -/


/-- **Leibniz rule** for the co-Heyting boundary. -/
theorem boundary_inf (a b : α) : ∂ (a ⊓ b) = ∂ a ⊓ b ⊔ a ⊓ ∂ b := by
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Eq (Coheyting.boundary (Min.min a b)) (Max.max (Min.min (Coheyting.boundary  …
  -/
  unfold boundary
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Eq (Min.min (Min.min a b) (HNot.hnot (Min.min a b))) (Max.max (Min.min (Min. …
  -/
  rw [hnot_inf_distrib, inf_sup_left, inf_right_comm, ← inf_assoc]
  /-
    🎉 no goals
  -/


theorem boundary_inf_le : ∂ (a ⊓ b) ≤ ∂ a ⊔ ∂ b :=
  (boundary_inf _ _).trans_le <| sup_le_sup inf_le_left inf_le_right


theorem boundary_sup_le : ∂ (a ⊔ b) ≤ ∂ a ⊔ ∂ b := by
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ LE.le (Coheyting.boundary (Max.max a b)) (Max.max (Coheyting.boundary a) (Co …
  -/
  rw [boundary, inf_sup_right]
  exact
    sup_le_sup (inf_le_inf_left _ <| hnot_anti le_sup_left)
      (inf_le_inf_left _ <| hnot_anti le_sup_right)

/- The intuitionistic version of `Coheyting.boundary_le_boundary_sup_sup_boundary_inf_left`. Either
proof can be obtained from the other using the equivalence of Heyting algebras and intuitionistic
logic and duality between Heyting and co-Heyting algebras. It is crucial that the following proof be
intuitionistic. -/

theorem boundary_le_boundary_sup_sup_boundary_inf_left : ∂ a ≤ ∂ (a ⊔ b) ⊔ ∂ (a ⊓ b) := by
  -- Porting note: the following simp generates the same term as mathlib3 if you remove
  -- sup_inf_right from both. With sup_inf_right included, mathlib4 and mathlib3 generate
  -- different terms
  simp only [boundary, sup_inf_left, sup_inf_right, sup_right_idem, le_inf_iff, sup_assoc,
    sup_comm _ a]
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ And (And (And (LE.le (Min.min a (HNot.hnot a)) (Max.max a (Max.max a b))) (L …
  -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  refine ⟨⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩, ?_, ?_⟩ <;> try { exact le_sup_of_le_left inf_le_left } <;>
    /-
      case refine_4
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (Min.min a (HNot.hnot a)) (Max.max (HNot.hnot (Max.max a b)) b)
    -/
    refine inf_le_of_right_le ?_
    /-
      case refine_4
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (HNot.hnot a) (Max.max (HNot.hnot (Max.max a b)) b)
    -/
  · rw [hnot_le_iff_codisjoint_right, codisjoint_left_comm]
    /-
      case refine_4
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ Codisjoint (HNot.hnot (Max.max a b)) (Max.max a b)
    -/
    exact codisjoint_hnot_left
    /-
      🎉 no goals
    -/
    /-
      case refine_6
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (HNot.hnot a) (Max.max (HNot.hnot (Max.max a b)) (HNot.hnot (Min.min a …
    -/
  · refine le_sup_of_le_right ?_
    /-
      case refine_6
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (HNot.hnot a) (HNot.hnot (Min.min a b))
    -/
    rw [hnot_le_iff_codisjoint_right]
    /-
      case refine_6
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ Codisjoint a (HNot.hnot (Min.min a b))
    -/
    exact codisjoint_hnot_right.mono_right (hnot_anti inf_le_left)
    /-
      🎉 no goals
    -/


theorem boundary_le_boundary_sup_sup_boundary_inf_right : ∂ b ≤ ∂ (a ⊔ b) ⊔ ∂ (a ⊓ b) := by
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ LE.le (Coheyting.boundary b) (Max.max (Coheyting.boundary (Max.max a b)) (Co …
  -/
  rw [sup_comm a, inf_comm]
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ LE.le (Coheyting.boundary b) (Max.max (Coheyting.boundary (Max.max b a)) (Co …
  -/
  exact boundary_le_boundary_sup_sup_boundary_inf_left
  /-
    🎉 no goals
  -/


theorem boundary_sup_sup_boundary_inf (a b : α) : ∂ (a ⊔ b) ⊔ ∂ (a ⊓ b) = ∂ a ⊔ ∂ b :=
  le_antisymm (sup_le boundary_sup_le boundary_inf_le) <|
    sup_le boundary_le_boundary_sup_sup_boundary_inf_left
      boundary_le_boundary_sup_sup_boundary_inf_right


@[simp]
                                                  /-
                                                    α : Type u_1
                                                    inst✝ : CoheytingAlgebra α
                                                    a : α
                                                    ⊢ Eq (Coheyting.boundary (Coheyting.boundary a)) (Coheyting.boundary a)
                                                  -/
theorem boundary_idem (a : α) : ∂ ∂ a = ∂ a := by rw [boundary, hnot_boundary, inf_top_eq]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem hnot_hnot_sup_boundary (a : α) : ￢￢a ⊔ ∂ a = a := by
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a : α
    ⊢ Eq (Max.max (HNot.hnot (HNot.hnot a)) (Coheyting.boundary a)) a
  -/
  rw [boundary, sup_inf_left, hnot_sup_self, inf_top_eq, sup_eq_right]
  /-
    α : Type u_1
    inst✝ : CoheytingAlgebra α
    a : α
    ⊢ LE.le (HNot.hnot (HNot.hnot a)) a
  -/
  exact hnot_hnot_le
  /-
    🎉 no goals
  -/


theorem hnot_eq_top_iff_exists_boundary : ￢a = ⊤ ↔ ∃ b, ∂ b = a :=
                   /-
                     α : Type u_1
                     inst✝ : CoheytingAlgebra α
                     a : α
                     h : Eq (HNot.hnot a) Top.top
                     ⊢ Eq (Coheyting.boundary a) a
                   -/
  ⟨fun h => ⟨a, by rw [boundary, h, inf_top_eq]⟩, by
                   /-
                     🎉 no goals
                   -/
    /-
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      a : α
      ⊢ (Exists fun b => Eq (Coheyting.boundary b) a) → Eq (HNot.hnot a) Top.top
    -/
    rintro ⟨b, rfl⟩
    /-
      case intro
      α : Type u_1
      inst✝ : CoheytingAlgebra α
      b : α
      ⊢ Eq (HNot.hnot (Coheyting.boundary b)) Top.top
    -/
    exact hnot_boundary _⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem Coheyting.boundary_eq_bot (a : α) : ∂ a = ⊥ :=
  inf_compl_eq_bot


