instance : DecidablePred (derangements α) := fun _ => Fintype.decidableForallFintype

-- Porting note: used to use the tactic delta_instance

instance : Fintype (derangements α) := Subtype.fintype (fun (_ : Perm α) => ∀ (x_1 : α), ¬_ = x_1)


theorem card_derangements_invariant {α β : Type*} [Fintype α] [DecidableEq α] [Fintype β]
    [DecidableEq β] (h : card α = card β) : card (derangements α) = card (derangements β) :=
  Fintype.card_congr (Equiv.derangementsCongr <| equivOfCardEq h)


theorem card_derangements_fin_add_two (n : ℕ) :
    card (derangements (Fin (n + 2))) =
      (n + 1) * card (derangements (Fin n)) + (n + 1) * card (derangements (Fin (n + 1))) := by
  -- get some basic results about the size of Fin (n+1) plus or minus an element
  have h1 : ∀ a : Fin (n + 1), card ({a}ᶜ : Set (Fin (n + 1))) = card (Fin n) := by
    intro a
    simp only
      [card_ofFinset (s := Finset.filter (fun x => x ∈ ({a}ᶜ : Set (Fin (n + 1)))) Finset.univ),
      Set.mem_compl_singleton_iff, Finset.filter_ne' _ a,
      Finset.card_erase_of_mem (Finset.mem_univ a), Finset.card_fin, add_tsub_cancel_right,
      card_fin]
  /-
    n : Nat
    h1 : ∀ (a : Fin (HAdd.hAdd n 1)), Eq (Fintype.card ↑(HasCompl.compl (Singleton …
    ⊢ Eq (Fintype.card ↑(derangements (Fin (HAdd.hAdd n 2)))) (HAdd.hAdd (HMul.hMu …
  -/
  have h2 : card (Fin (n + 2)) = card (Option (Fin (n + 1))) := by simp only [card_fin, card_option]
  -- rewrite the LHS and substitute in our fintype-level equivalence
  simp only [card_derangements_invariant h2,
    card_congr
      (@derangementsRecursionEquiv (Fin (n + 1))
        _),-- push the cardinality through the Σ and ⊕ so that we can use `card_n`
    card_sigma,
    card_sum, card_derangements_invariant (h1 _), Finset.sum_const, nsmul_eq_mul, Finset.card_fin,
    mul_add, Nat.cast_id]


/-- The number of derangements of an `n`-element set. -/
def numDerangements : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | n + 2 => (n + 1) * (numDerangements n + numDerangements (n + 1))


@[simp]
theorem numDerangements_zero : numDerangements 0 = 1 :=
  rfl


@[simp]
theorem numDerangements_one : numDerangements 1 = 0 :=
  rfl


theorem numDerangements_add_two (n : ℕ) :
    numDerangements (n + 2) = (n + 1) * (numDerangements n + numDerangements (n + 1)) :=
  rfl


theorem numDerangements_succ (n : ℕ) :
    (numDerangements (n + 1) : ℤ) = (n + 1) * (numDerangements n : ℤ) - (-1) ^ n := by
  induction n with
  | zero => rfl
  | succ n hn =>
    simp only [numDerangements_add_two, hn, pow_succ, Int.ofNat_mul, Int.ofNat_add]
    ring


theorem card_derangements_fin_eq_numDerangements {n : ℕ} :
    card (derangements (Fin n)) = numDerangements n := by
  /-
    n : Nat
    ⊢ Eq (Fintype.card ↑(derangements (Fin n))) (numDerangements n)
  -/
  induction' n using Nat.strong_induction_on with n hyp
  /-
    case h
    n : Nat
    hyp : ∀ (m : Nat), LT.lt m n → Eq (Fintype.card ↑(derangements (Fin m))) (numD …
    ⊢ Eq (Fintype.card ↑(derangements (Fin n))) (numDerangements n)
  -/
  rcases n with _ | _ | n
  -- knock out cases 0 and 1
    /-
      case h.zero
      hyp : ∀ (m : Nat), LT.lt m 0 → Eq (Fintype.card ↑(derangements (Fin m))) (numD …
      ⊢ Eq (Fintype.card ↑(derangements (Fin 0))) (numDerangements 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.succ.zero
      hyp : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → Eq (Fintype.card ↑(derangements ( …
      ⊢ Eq (Fintype.card ↑(derangements (Fin (HAdd.hAdd 0 1)))) (numDerangements (HA …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  -- now we have n ≥ 2. rewrite everything in terms of card_derangements, so that we can use
  -- `card_derangements_fin_add_two`
  /-
    case h.succ.succ
    n : Nat
    hyp : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → Eq (Fintype.card ↑( …
    ⊢ Eq (Fintype.card ↑(derangements (Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)))) (numDe …
  -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
  rw [numDerangements_add_two, card_derangements_fin_add_two, mul_add, hyp, hyp] <;> omega
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem card_derangements_eq_numDerangements (α : Type*) [Fintype α] [DecidableEq α] :
    card (derangements α) = numDerangements (card α) := by
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ⊢ Eq (Fintype.card ↑(derangements α)) (numDerangements (Fintype.card α))
  -/
  rw [← card_derangements_invariant (card_fin _)]
  /-
    α : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ⊢ Eq (Fintype.card ↑(derangements (Fin (Fintype.card α)))) (numDerangements (F …
  -/
  exact card_derangements_fin_eq_numDerangements
  /-
    🎉 no goals
  -/


theorem numDerangements_sum (n : ℕ) :
    (numDerangements n : ℤ) =
      ∑ k ∈ Finset.range (n + 1), (-1 : ℤ) ^ k * Nat.ascFactorial (k + 1) (n - k) := by
  /-
    n : Nat
    ⊢ Eq (↑(numDerangements n)) ((Finset.range (HAdd.hAdd n 1)).sum fun k => HMul. …
  -/
  induction' n with n hn; · rfl
                            /-
                              🎉 no goals
                            -/
  rw [Finset.sum_range_succ, numDerangements_succ, hn, Finset.mul_sum, tsub_self,
    Nat.ascFactorial_zero, Int.ofNat_one, mul_one, pow_succ', neg_one_mul, sub_eq_add_neg,
    add_left_inj, Finset.sum_congr rfl]
  -- show that (n + 1) * (-1)^x * asc_fac x (n - x) = (-1)^x * asc_fac x (n.succ - x)
  /-
    case succ
    n : Nat
    hn : Eq (↑(numDerangements n)) ((Finset.range (HAdd.hAdd n 1)).sum fun k => HM …
    ⊢ ∀ (x : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) x → Eq (HMul.hMul …
  -/
  intro x hx
  /-
    case succ
    n : Nat
    hn : Eq (↑(numDerangements n)) ((Finset.range (HAdd.hAdd n 1)).sum fun k => HM …
    x : Nat
    hx : Membership.mem (Finset.range (HAdd.hAdd n 1)) x
    ⊢ Eq (HMul.hMul (HAdd.hAdd (↑n) 1) (HMul.hMul (HPow.hPow (-1) x) ↑((HAdd.hAdd  …
  -/
  have h_le : x ≤ n := Finset.mem_range_succ_iff.mp hx
  rw [Nat.succ_sub h_le, Nat.ascFactorial_succ, add_right_comm, add_tsub_cancel_of_le h_le,
    Int.ofNat_mul, Int.ofNat_add, mul_left_comm, Nat.cast_one]

