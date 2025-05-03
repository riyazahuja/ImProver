lemma even_sum_iff_even_card_odd {s : Finset ι} (f : ι → ℕ) :
    Even (∑ i ∈ s, f i) ↔ Even (s.filter fun x ↦ Odd (f x)).card := by
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Nat
    ⊢ Iff (Even (s.sum fun i => f i)) (Even (Finset.filter (fun x => Odd (f x)) s) …
  -/
  rw [← Finset.sum_filter_add_sum_filter_not _ (fun x ↦ Even (f x)), Nat.even_add]
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Nat
    ⊢ Iff (Iff (Even ((Finset.filter (fun x => Even (f x)) s).sum fun x => f x)) ( …
  -/
  simp only [Finset.mem_filter, and_imp, imp_self, implies_true, Finset.even_sum, true_iff]
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Nat
    ⊢ Iff (Even ((Finset.filter (fun x => Not (Even (f x))) s).sum fun x => f x))  …
  -/
  rw [Nat.even_iff, Finset.sum_nat_mod, Finset.sum_filter]
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Nat
    ⊢ Iff (Eq (HMod.hMod (s.sum fun a => ite (Not (Even (f a))) (HMod.hMod (f a) 2 …
  -/
  simp +contextual only [Nat.not_even_iff_odd, Nat.odd_iff.mp]
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Nat
    ⊢ Iff (Eq (HMod.hMod (s.sum fun x => ite (Odd (f x)) 1 0) 2) 0) (Even (Finset. …
  -/
  simp_rw [← Finset.sum_filter, ← Nat.even_iff, Finset.card_eq_sum_ones]
  /-
    🎉 no goals
  -/


lemma odd_sum_iff_odd_card_odd {s : Finset ι} (f : ι → ℕ) :
    Odd (∑ i ∈ s, f i) ↔ Odd (s.filter fun x ↦ Odd (f x)).card := by
  /-
    ι : Type u_1
    s : Finset ι
    f : ι → Nat
    ⊢ Iff (Odd (s.sum fun i => f i)) (Odd (Finset.filter (fun x => Odd (f x)) s).c …
  -/
  simp only [← Nat.not_even_iff_odd, even_sum_iff_even_card_odd]
  /-
    🎉 no goals
  -/


