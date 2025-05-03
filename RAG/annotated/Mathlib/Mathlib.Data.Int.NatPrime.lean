theorem not_prime_of_int_mul {a b : ℤ} {c : ℕ} (ha : a.natAbs ≠ 1) (hb : b.natAbs ≠ 1)
    (hc : a * b = (c : ℤ)) : ¬Nat.Prime c :=
  not_prime_mul' (natAbs_mul_natAbs_eq hc) ha hb


theorem succ_dvd_or_succ_dvd_of_succ_sum_dvd_mul {p : ℕ} (p_prime : Nat.Prime p) {m n : ℤ}
    {k l : ℕ} (hpm : ↑(p ^ k) ∣ m) (hpn : ↑(p ^ l) ∣ n) (hpmn : ↑(p ^ (k + l + 1)) ∣ m * n) :
    ↑(p ^ (k + 1)) ∣ m ∨ ↑(p ^ (l + 1)) ∣ n :=
  have hpm' : p ^ k ∣ m.natAbs := Int.natCast_dvd_natCast.1 <| Int.dvd_natAbs.2 hpm
  have hpn' : p ^ l ∣ n.natAbs := Int.natCast_dvd_natCast.1 <| Int.dvd_natAbs.2 hpn
  have hpmn' : p ^ (k + l + 1) ∣ m.natAbs * n.natAbs := by
    /-
      p : Nat
      p_prime : Nat.Prime p
      m n : Int
      k l : Nat
      hpm : Dvd.dvd (↑(HPow.hPow p k)) m
      hpn : Dvd.dvd (↑(HPow.hPow p l)) n
      hpmn : Dvd.dvd (↑(HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1))) (HMul.hMul m n)
      hpm' : Dvd.dvd (HPow.hPow p k) m.natAbs
      hpn' : Dvd.dvd (HPow.hPow p l) n.natAbs
      ⊢ Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul m.natAbs n.na …
    -/
    rw [← Int.natAbs_mul]; apply Int.natCast_dvd_natCast.1 <| Int.dvd_natAbs.2 hpmn
                           /-
                             🎉 no goals
                           -/
  let hsd := Nat.succ_dvd_or_succ_dvd_of_succ_sum_dvd_mul p_prime hpm' hpn' hpmn'
                                   /-
                                     p : Nat
                                     p_prime : Nat.Prime p
                                     m n : Int
                                     k l : Nat
                                     hpm : Dvd.dvd (↑(HPow.hPow p k)) m
                                     hpn : Dvd.dvd (↑(HPow.hPow p l)) n
                                     hpmn : Dvd.dvd (↑(HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1))) (HMul.hMul m n)
                                     hpm' : Dvd.dvd (HPow.hPow p k) m.natAbs
                                     hpn' : Dvd.dvd (HPow.hPow p l) n.natAbs
                                     hpmn' : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul m.natAb …
                                     hsd : Or (Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) m.natAbs) (Dvd.dvd (HPow.hPow  …
                                     hsd1 : Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) m.natAbs
                                     ⊢ Dvd.dvd (↑(HPow.hPow p (HAdd.hAdd k 1))) m
                                   -/
  hsd.elim (fun hsd1 => Or.inl (by apply Int.dvd_natAbs.1; apply Int.natCast_dvd_natCast.2 hsd1))
                                                           /-
                                                             🎉 no goals
                                                           -/
                           /-
                             p : Nat
                             p_prime : Nat.Prime p
                             m n : Int
                             k l : Nat
                             hpm : Dvd.dvd (↑(HPow.hPow p k)) m
                             hpn : Dvd.dvd (↑(HPow.hPow p l)) n
                             hpmn : Dvd.dvd (↑(HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1))) (HMul.hMul m n)
                             hpm' : Dvd.dvd (HPow.hPow p k) m.natAbs
                             hpn' : Dvd.dvd (HPow.hPow p l) n.natAbs
                             hpmn' : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul m.natAb …
                             hsd : Or (Dvd.dvd (HPow.hPow p (HAdd.hAdd k 1)) m.natAbs) (Dvd.dvd (HPow.hPow  …
                             hsd2 : Dvd.dvd (HPow.hPow p (HAdd.hAdd l 1)) n.natAbs
                             ⊢ Dvd.dvd (↑(HPow.hPow p (HAdd.hAdd l 1))) n
                           -/
    fun hsd2 => Or.inr (by apply Int.dvd_natAbs.1; apply Int.natCast_dvd_natCast.2 hsd2)
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem Prime.dvd_natAbs_of_coe_dvd_sq {p : ℕ} (hp : p.Prime) (k : ℤ) (h : (p : ℤ) ∣ k ^ 2) :
    p ∣ k.natAbs := by
  /-
    p : Nat
    hp : Nat.Prime p
    k : Int
    h : Dvd.dvd (↑p) (HPow.hPow k 2)
    ⊢ Dvd.dvd p k.natAbs
  -/
  apply @Nat.Prime.dvd_of_dvd_pow _ _ 2 hp
  /-
    p : Nat
    hp : Nat.Prime p
    k : Int
    h : Dvd.dvd (↑p) (HPow.hPow k 2)
    ⊢ Dvd.dvd p (HPow.hPow k.natAbs 2)
  -/
  rwa [sq, ← natAbs_mul, ← natCast_dvd, ← sq]
  /-
    🎉 no goals
  -/


