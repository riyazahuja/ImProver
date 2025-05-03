theorem dvd_choose_add (hp : Prime p) (hap : a < p) (hbp : b < p) (h : p ≤ a + b) :
    p ∣ choose (a + b) a := by
  /-
    p a b : Nat
    hp : Nat.Prime p
    hap : LT.lt a p
    hbp : LT.lt b p
    h : LE.le p (HAdd.hAdd a b)
    ⊢ Dvd.dvd p ((HAdd.hAdd a b).choose a)
  -/
  have h₁ : p ∣ (a + b)! := hp.dvd_factorial.2 h
  rw [← add_choose_mul_factorial_mul_factorial, ← choose_symm_add, hp.dvd_mul, hp.dvd_mul,
    hp.dvd_factorial, hp.dvd_factorial] at h₁
  /-
    p a b : Nat
    hp : Nat.Prime p
    hap : LT.lt a p
    hbp : LT.lt b p
    h : LE.le p (HAdd.hAdd a b)
    h₁ : Or (Or (Dvd.dvd p ((HAdd.hAdd a b).choose a)) (LE.le p a)) (LE.le p b)
    ⊢ Dvd.dvd p ((HAdd.hAdd a b).choose a)
  -/
  exact (h₁.resolve_right hbp.not_le).resolve_right hap.not_le
  /-
    🎉 no goals
  -/


lemma dvd_choose (hp : Prime p) (ha : a < p) (hab : b - a < p) (h : p ≤ b) : p ∣ choose b a :=
  have : a + (b - a) = b := Nat.add_sub_of_le (ha.le.trans h)
  this ▸ hp.dvd_choose_add ha hab (this.symm ▸ h)


lemma dvd_choose_self (hp : Prime p) (hk : k ≠ 0) (hkp : k < p) : p ∣ choose p k :=
  hp.dvd_choose hkp (sub_lt ((zero_le _).trans_lt hkp) <| zero_lt_of_ne_zero hk) le_rfl


