lemma chain'_ofFn {α : Type*} {n : ℕ} {f : Fin n → α} {r : α → α → Prop} :
    (ofFn f).Chain' r ↔ ∀ (i) (hi : i + 1 < n), r (f ⟨i, lt_of_succ_lt hi⟩) (f ⟨i + 1, hi⟩) := by
  /-
    α : Type u_1
    n : Nat
    f : Fin n → α
    r : α → α → Prop
    ⊢ Iff (List.Chain' r (List.ofFn f)) (∀ (i : Nat) (hi : LT.lt (HAdd.hAdd i 1) n …
  -/
  simp_rw [chain'_iff_get, get_ofFn, length_ofFn]
  /-
    α : Type u_1
    n : Nat
    f : Fin n → α
    r : α → α → Prop
    ⊢ Iff (∀ (i : Nat) (h : LT.lt i (HSub.hSub n 1)), r (f (Fin.cast ⋯ ⟨i, ⋯⟩)) (f …
  -/
  exact ⟨fun h i hi ↦ h i (by omega), fun h i hi ↦ h i (by omega)⟩
  /-
    🎉 no goals
  -/


