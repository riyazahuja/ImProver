theorem Prime.dvd_factorial : ∀ {n p : ℕ} (_ : Prime p), p ∣ n ! ↔ p ≤ n
  | 0, _, hp => iff_of_false hp.not_dvd_one (not_le_of_lt hp.pos)
  | n + 1, p, hp => by
    /-
      n p : Nat
      hp : Nat.Prime p
      ⊢ Iff (Dvd.dvd p (HAdd.hAdd n 1).factorial) (LE.le p (HAdd.hAdd n 1))
    -/
    rw [factorial_succ, hp.dvd_mul, Prime.dvd_factorial hp]
    exact
      ⟨fun h => h.elim (le_of_dvd (succ_pos _)) le_succ_of_le, fun h =>
        (_root_.lt_or_eq_of_le h).elim (Or.inr ∘ le_of_lt_succ) fun h => Or.inl <| by rw [h]⟩


