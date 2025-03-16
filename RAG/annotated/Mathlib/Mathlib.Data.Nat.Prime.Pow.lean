theorem pow_minFac {n k : ℕ} (hk : k ≠ 0) : (n ^ k).minFac = n.minFac := by
  /-
    n k : Nat
    hk : Ne k 0
    ⊢ Eq (HPow.hPow n k).minFac n.minFac
  -/
  rcases eq_or_ne n 1 with (rfl | hn)
    /-
      case inl
      k : Nat
      hk : Ne k 0
      ⊢ Eq (HPow.hPow 1 k).minFac (Nat.minFac 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n k : Nat
    hk : Ne k 0
    hn : Ne n 1
    ⊢ Eq (HPow.hPow n k).minFac n.minFac
  -/
  have hnk : n ^ k ≠ 1 := fun hk' => hn ((pow_eq_one_iff hk).1 hk')
  /-
    case inr
    n k : Nat
    hk : Ne k 0
    hn : Ne n 1
    hnk : Ne (HPow.hPow n k) 1
    ⊢ Eq (HPow.hPow n k).minFac n.minFac
  -/
  apply (minFac_le_of_dvd (minFac_prime hn).two_le ((minFac_dvd n).pow hk)).antisymm
  apply
    minFac_le_of_dvd (minFac_prime hnk).two_le
      ((minFac_prime hnk).dvd_of_dvd_pow (minFac_dvd _))


theorem Prime.pow_minFac {p k : ℕ} (hp : p.Prime) (hk : k ≠ 0) : (p ^ k).minFac = p := by
  /-
    p k : Nat
    hp : Nat.Prime p
    hk : Ne k 0
    ⊢ Eq (HPow.hPow p k).minFac p
  -/
  rw [Nat.pow_minFac hk, hp.minFac_eq]
  /-
    🎉 no goals
  -/


