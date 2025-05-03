/-- The nth-harmonic number defined as a finset sum of consecutive reciprocals. -/
def harmonic : ℕ → ℚ := fun n => ∑ i ∈ Finset.range n, (↑(i + 1))⁻¹


@[simp]
lemma harmonic_zero : harmonic 0 = 0 :=
  rfl


@[simp]
lemma harmonic_succ (n : ℕ) : harmonic (n + 1) = harmonic n + (↑(n + 1))⁻¹ :=
  Finset.sum_range_succ ..


lemma harmonic_pos {n : ℕ} (Hn : n ≠ 0) : 0 < harmonic n := by
  /-
    n : Nat
    Hn : Ne n 0
    ⊢ LT.lt 0 (harmonic n)
  -/
  unfold harmonic
  /-
    n : Nat
    Hn : Ne n 0
    ⊢ LT.lt 0 ((Finset.range n).sum fun i => Inv.inv ↑(HAdd.hAdd i 1))
  -/
  rw [← Finset.nonempty_range_iff] at Hn
  /-
    n : Nat
    Hn : (Finset.range n).Nonempty
    ⊢ LT.lt 0 ((Finset.range n).sum fun i => Inv.inv ↑(HAdd.hAdd i 1))
  -/
  positivity
  /-
    🎉 no goals
  -/



lemma harmonic_eq_sum_Icc {n : ℕ} :  harmonic n = ∑ i ∈ Finset.Icc 1 n, (↑i)⁻¹ := by
  /-
    n : Nat
    ⊢ Eq (harmonic n) ((Finset.Icc 1 n).sum fun i => Inv.inv ↑i)
  -/
  rw [harmonic, Finset.range_eq_Ico, Finset.sum_Ico_add' (fun (i : ℕ) ↦ (i : ℚ)⁻¹) 0 n (c := 1)]
  -- It might be better to restate `Nat.Ico_succ_right` in terms of `+ 1`,
  -- as we try to move away from `Nat.succ`.
  /-
    n : Nat
    ⊢ Eq ((Finset.Ico (HAdd.hAdd 0 1) (HAdd.hAdd n 1)).sum fun x => Inv.inv ↑x) (( …
  -/
  simp only [Nat.add_one, Nat.Ico_succ_right]
  /-
    🎉 no goals
  -/

