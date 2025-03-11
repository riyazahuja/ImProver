/-- `a ≡ b [ZMOD n]` when `a % n = b % n`. -/
def ModEq (n a b : ℤ) :=
  a % n = b % n


@[inherit_doc]
notation:50 a " ≡ " b " [ZMOD " n "]" => ModEq n a b


instance : Decidable (ModEq n a b) := decEq (a % n) (b % n)


@[refl, simp]
protected theorem refl (a : ℤ) : a ≡ a [ZMOD n] :=
  @rfl _ _


protected theorem rfl : a ≡ a [ZMOD n] :=
  ModEq.refl _


instance : IsRefl _ (ModEq n) :=
  ⟨ModEq.refl⟩


@[symm]
protected theorem symm : a ≡ b [ZMOD n] → b ≡ a [ZMOD n] :=
  Eq.symm


@[trans]
protected theorem trans : a ≡ b [ZMOD n] → b ≡ c [ZMOD n] → a ≡ c [ZMOD n] :=
  Eq.trans


instance : IsTrans ℤ (ModEq n) where
  trans := @Int.ModEq.trans n


protected theorem eq : a ≡ b [ZMOD n] → a % n = b % n := id


theorem modEq_comm : a ≡ b [ZMOD n] ↔ b ≡ a [ZMOD n] := ⟨ModEq.symm, ModEq.symm⟩


theorem natCast_modEq_iff {a b n : ℕ} : a ≡ b [ZMOD n] ↔ a ≡ b [MOD n] := by
  /-
    a b n : Nat
    ⊢ Iff ((↑n).ModEq ↑a ↑b) (n.ModEq a b)
  -/
  unfold ModEq Nat.ModEq; rw [← Int.ofNat_inj]; simp [natCast_mod]
                                                /-
                                                  🎉 no goals
                                                -/


theorem modEq_zero_iff_dvd : a ≡ 0 [ZMOD n] ↔ n ∣ a := by
  /-
    n a : Int
    ⊢ Iff (n.ModEq a 0) (Dvd.dvd n a)
  -/
  rw [ModEq, zero_emod, dvd_iff_emod_eq_zero]
  /-
    🎉 no goals
  -/


theorem _root_.Dvd.dvd.modEq_zero_int (h : n ∣ a) : a ≡ 0 [ZMOD n] :=
  modEq_zero_iff_dvd.2 h


theorem _root_.Dvd.dvd.zero_modEq_int (h : n ∣ a) : 0 ≡ a [ZMOD n] :=
  h.modEq_zero_int.symm


theorem modEq_iff_dvd : a ≡ b [ZMOD n] ↔ n ∣ b - a := by
  /-
    n a b : Int
    ⊢ Iff (n.ModEq a b) (Dvd.dvd n (HSub.hSub b a))
  -/
  rw [ModEq, eq_comm]
  /-
    n a b : Int
    ⊢ Iff (Eq (HMod.hMod b n) (HMod.hMod a n)) (Dvd.dvd n (HSub.hSub b a))
  -/
  simp [emod_eq_emod_iff_emod_sub_eq_zero, dvd_iff_emod_eq_zero]
  /-
    🎉 no goals
  -/


theorem modEq_iff_add_fac {a b n : ℤ} : a ≡ b [ZMOD n] ↔ ∃ t, b = a + n * t := by
  /-
    a b n : Int
    ⊢ Iff (n.ModEq a b) (Exists fun t => Eq b (HAdd.hAdd a (HMul.hMul n t)))
  -/
  rw [modEq_iff_dvd]
  /-
    a b n : Int
    ⊢ Iff (Dvd.dvd n (HSub.hSub b a)) (Exists fun t => Eq b (HAdd.hAdd a (HMul.hMu …
  -/
  exact exists_congr fun t => sub_eq_iff_eq_add'
  /-
    🎉 no goals
  -/


alias ⟨ModEq.dvd, modEq_of_dvd⟩ := modEq_iff_dvd


theorem mod_modEq (a n) : a % n ≡ a [ZMOD n] :=
  emod_emod _ _


@[simp]
theorem neg_modEq_neg : -a ≡ -b [ZMOD n] ↔ a ≡ b [ZMOD n] := by
  /-
    n a b : Int
    ⊢ Iff (n.ModEq (Neg.neg a) (Neg.neg b)) (n.ModEq a b)
  -/
  simp only [modEq_iff_dvd, (by omega : -b - -a = -(b - a)), Int.dvd_neg]
  /-
    🎉 no goals
  -/


@[simp]
                                                           /-
                                                             n a b : Int
                                                             ⊢ Iff ((Neg.neg n).ModEq a b) (n.ModEq a b)
                                                           -/
theorem modEq_neg : a ≡ b [ZMOD -n] ↔ a ≡ b [ZMOD n] := by simp [modEq_iff_dvd]
                                                           /-
                                                             🎉 no goals
                                                           -/


protected theorem of_dvd (d : m ∣ n) (h : a ≡ b [ZMOD n]) : a ≡ b [ZMOD m] :=
  modEq_iff_dvd.2 <| d.trans h.dvd


protected theorem mul_left' (h : a ≡ b [ZMOD n]) : c * a ≡ c * b [ZMOD c * n] := by
  /-
    n a b c : Int
    h : n.ModEq a b
    ⊢ (HMul.hMul c n).ModEq (HMul.hMul c a) (HMul.hMul c b)
  -/
  obtain hc | rfl | hc := lt_trichotomy c 0
    /-
      case inl
      n a b c : Int
      h : n.ModEq a b
      hc : LT.lt c 0
      ⊢ (HMul.hMul c n).ModEq (HMul.hMul c a) (HMul.hMul c b)
    -/
  · rw [← neg_modEq_neg, ← modEq_neg, ← Int.neg_mul, ← Int.neg_mul, ← Int.neg_mul]
    /-
      case inl
      n a b c : Int
      h : n.ModEq a b
      hc : LT.lt c 0
      ⊢ (HMul.hMul (Neg.neg c) n).ModEq (HMul.hMul (Neg.neg c) a) (HMul.hMul (Neg.ne …
    -/
    simp only [ModEq, mul_emod_mul_of_pos _ _ (neg_pos.2 hc), h.eq]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      n a b : Int
      h : n.ModEq a b
      ⊢ (HMul.hMul 0 n).ModEq (HMul.hMul 0 a) (HMul.hMul 0 b)
    -/
  · simp only [Int.zero_mul, ModEq.rfl]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      n a b c : Int
      h : n.ModEq a b
      hc : LT.lt 0 c
      ⊢ (HMul.hMul c n).ModEq (HMul.hMul c a) (HMul.hMul c b)
    -/
  · simp only [ModEq, mul_emod_mul_of_pos _ _ hc, h.eq]
    /-
      🎉 no goals
    -/


protected theorem mul_right' (h : a ≡ b [ZMOD n]) : a * c ≡ b * c [ZMOD n * c] := by
  /-
    n a b c : Int
    h : n.ModEq a b
    ⊢ (HMul.hMul n c).ModEq (HMul.hMul a c) (HMul.hMul b c)
  -/
  rw [mul_comm a, mul_comm b, mul_comm n]; exact h.mul_left'
                                           /-
                                             🎉 no goals
                                           -/


@[gcongr]
protected theorem add (h₁ : a ≡ b [ZMOD n]) (h₂ : c ≡ d [ZMOD n]) : a + c ≡ b + d [ZMOD n] :=
                        /-
                          n a b c d : Int
                          h₁ : n.ModEq a b
                          h₂ : n.ModEq c d
                          ⊢ Dvd.dvd n (HSub.hSub (HAdd.hAdd b d) (HAdd.hAdd a c))
                        -/
  modEq_iff_dvd.2 <| by convert Int.dvd_add h₁.dvd h₂.dvd using 1; omega
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[gcongr] protected theorem add_left (c : ℤ) (h : a ≡ b [ZMOD n]) : c + a ≡ c + b [ZMOD n] :=
  ModEq.rfl.add h


@[gcongr] protected theorem add_right (c : ℤ) (h : a ≡ b [ZMOD n]) : a + c ≡ b + c [ZMOD n] :=
  h.add ModEq.rfl


protected theorem add_left_cancel (h₁ : a ≡ b [ZMOD n]) (h₂ : a + c ≡ b + d [ZMOD n]) :
    c ≡ d [ZMOD n] :=
                                                 /-
                                                   n a b c d : Int
                                                   h₁ : n.ModEq a b
                                                   h₂ : n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
                                                   ⊢ Eq (HSub.hSub d c) (HSub.hSub (HSub.hSub (HAdd.hAdd b d) (HAdd.hAdd a c)) (H …
                                                 -/
  have : d - c = b + d - (a + c) - (b - a) := by omega
                                                 /-
                                                   🎉 no goals
                                                 -/
  modEq_iff_dvd.2 <| by
    /-
      n a b c d : Int
      h₁ : n.ModEq a b
      h₂ : n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
      this : Eq (HSub.hSub d c) (HSub.hSub (HSub.hSub (HAdd.hAdd b d) (HAdd.hAdd a c …
      ⊢ Dvd.dvd n (HSub.hSub d c)
    -/
    rw [this]
    /-
      n a b c d : Int
      h₁ : n.ModEq a b
      h₂ : n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
      this : Eq (HSub.hSub d c) (HSub.hSub (HSub.hSub (HAdd.hAdd b d) (HAdd.hAdd a c …
      ⊢ Dvd.dvd n (HSub.hSub (HSub.hSub (HAdd.hAdd b d) (HAdd.hAdd a c)) (HSub.hSub  …
    -/
    exact Int.dvd_sub h₂.dvd h₁.dvd
    /-
      🎉 no goals
    -/


protected theorem add_left_cancel' (c : ℤ) (h : c + a ≡ c + b [ZMOD n]) : a ≡ b [ZMOD n] :=
  ModEq.rfl.add_left_cancel h


protected theorem add_right_cancel (h₁ : c ≡ d [ZMOD n]) (h₂ : a + c ≡ b + d [ZMOD n]) :
    a ≡ b [ZMOD n] := by
  /-
    n a b c d : Int
    h₁ : n.ModEq c d
    h₂ : n.ModEq (HAdd.hAdd a c) (HAdd.hAdd b d)
    ⊢ n.ModEq a b
  -/
  rw [add_comm a, add_comm b] at h₂
  /-
    n a b c d : Int
    h₁ : n.ModEq c d
    h₂ : n.ModEq (HAdd.hAdd c a) (HAdd.hAdd d b)
    ⊢ n.ModEq a b
  -/
  exact h₁.add_left_cancel h₂
  /-
    🎉 no goals
  -/


protected theorem add_right_cancel' (c : ℤ) (h : a + c ≡ b + c [ZMOD n]) : a ≡ b [ZMOD n] :=
  ModEq.rfl.add_right_cancel h


@[gcongr] protected theorem neg (h : a ≡ b [ZMOD n]) : -a ≡ -b [ZMOD n] :=
                        /-
                          n a b : Int
                          h : n.ModEq a b
                          ⊢ n.ModEq (HAdd.hAdd a (Neg.neg a)) (HAdd.hAdd b (Neg.neg b))
                        -/
  h.add_left_cancel (by simp_rw [← sub_eq_add_neg, sub_self]; rfl)
                                                              /-
                                                                🎉 no goals
                                                              -/


@[gcongr]
protected theorem sub (h₁ : a ≡ b [ZMOD n]) (h₂ : c ≡ d [ZMOD n]) : a - c ≡ b - d [ZMOD n] := by
  /-
    n a b c d : Int
    h₁ : n.ModEq a b
    h₂ : n.ModEq c d
    ⊢ n.ModEq (HSub.hSub a c) (HSub.hSub b d)
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg]
  /-
    n a b c d : Int
    h₁ : n.ModEq a b
    h₂ : n.ModEq c d
    ⊢ n.ModEq (HAdd.hAdd a (Neg.neg c)) (HAdd.hAdd b (Neg.neg d))
  -/
  exact h₁.add h₂.neg
  /-
    🎉 no goals
  -/


@[gcongr] protected theorem sub_left (c : ℤ) (h : a ≡ b [ZMOD n]) : c - a ≡ c - b [ZMOD n] :=
  ModEq.rfl.sub h


@[gcongr] protected theorem sub_right (c : ℤ) (h : a ≡ b [ZMOD n]) : a - c ≡ b - c [ZMOD n] :=
  h.sub ModEq.rfl


@[gcongr] protected theorem mul_left (c : ℤ) (h : a ≡ b [ZMOD n]) : c * a ≡ c * b [ZMOD n] :=
  h.mul_left'.of_dvd <| dvd_mul_left _ _


@[gcongr] protected theorem mul_right (c : ℤ) (h : a ≡ b [ZMOD n]) : a * c ≡ b * c [ZMOD n] :=
  h.mul_right'.of_dvd <| dvd_mul_right _ _


@[gcongr]
protected theorem mul (h₁ : a ≡ b [ZMOD n]) (h₂ : c ≡ d [ZMOD n]) : a * c ≡ b * d [ZMOD n] :=
  (h₂.mul_left _).trans (h₁.mul_right _)


@[gcongr] protected theorem pow (m : ℕ) (h : a ≡ b [ZMOD n]) : a ^ m ≡ b ^ m [ZMOD n] := by
  /-
    n a b : Int
    m : Nat
    h : n.ModEq a b
    ⊢ n.ModEq (HPow.hPow a m) (HPow.hPow b m)
  -/
  induction' m with d hd; · rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n a b : Int
    h : n.ModEq a b
    d : Nat
    hd : n.ModEq (HPow.hPow a d) (HPow.hPow b d)
    ⊢ n.ModEq (HPow.hPow a (HAdd.hAdd d 1)) (HPow.hPow b (HAdd.hAdd d 1))
  -/
  rw [pow_succ, pow_succ]
  /-
    case succ
    n a b : Int
    h : n.ModEq a b
    d : Nat
    hd : n.ModEq (HPow.hPow a d) (HPow.hPow b d)
    ⊢ n.ModEq (HMul.hMul (HPow.hPow a d) a) (HMul.hMul (HPow.hPow b d) b)
  -/
  exact hd.mul h
  /-
    🎉 no goals
  -/


lemma of_mul_left (m : ℤ) (h : a ≡ b [ZMOD m * n]) : a ≡ b [ZMOD n] := by
  /-
    n a b m : Int
    h : (HMul.hMul m n).ModEq a b
    ⊢ n.ModEq a b
  -/
  rw [modEq_iff_dvd] at *; exact (dvd_mul_left n m).trans h
                           /-
                             🎉 no goals
                           -/


lemma of_mul_right (m : ℤ) : a ≡ b [ZMOD n * m] → a ≡ b [ZMOD n] :=
  mul_comm m n ▸ of_mul_left _


/-- To cancel a common factor `c` from a `ModEq` we must divide the modulus `m` by `gcd m c`. -/
theorem cancel_right_div_gcd (hm : 0 < m) (h : a * c ≡ b * c [ZMOD m]) :
    a ≡ b [ZMOD m / gcd m c] := by
  /-
    m a b c : Int
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul a c) (HMul.hMul b c)
    ⊢ (HDiv.hDiv m ↑(m.gcd c)).ModEq a b
  -/
  letI d := gcd m c
  /-
    m a b c : Int
    hm : LT.lt 0 m
    h : m.ModEq (HMul.hMul a c) (HMul.hMul b c)
    d : Nat := m.gcd c
    ⊢ (HDiv.hDiv m ↑(m.gcd c)).ModEq a b
  -/
  rw [modEq_iff_dvd] at h ⊢
  -- Porting note: removed `show` due to https://github.com/leanprover-community/mathlib4/issues/3305
  /-
    m a b c : Int
    hm : LT.lt 0 m
    h : Dvd.dvd m (HSub.hSub (HMul.hMul b c) (HMul.hMul a c))
    d : Nat := m.gcd c
    ⊢ Dvd.dvd (HDiv.hDiv m ↑(m.gcd c)) (HSub.hSub b a)
  -/
  refine Int.dvd_of_dvd_mul_right_of_gcd_one (?_ : m / d ∣ c / d * (b - a)) ?_
    /-
      case refine_1
      m a b c : Int
      hm : LT.lt 0 m
      h : Dvd.dvd m (HSub.hSub (HMul.hMul b c) (HMul.hMul a c))
      d : Nat := m.gcd c
      ⊢ Dvd.dvd (HDiv.hDiv m ↑d) (HMul.hMul (HDiv.hDiv c ↑d) (HSub.hSub b a))
    -/
  · rw [mul_comm, ← Int.mul_ediv_assoc (b - a) gcd_dvd_right, Int.sub_mul]
    /-
      case refine_1
      m a b c : Int
      hm : LT.lt 0 m
      h : Dvd.dvd m (HSub.hSub (HMul.hMul b c) (HMul.hMul a c))
      d : Nat := m.gcd c
      ⊢ Dvd.dvd (HDiv.hDiv m ↑d) (HDiv.hDiv (HSub.hSub (HMul.hMul b c) (HMul.hMul a  …
    -/
    exact Int.ediv_dvd_ediv gcd_dvd_left h
    /-
      🎉 no goals
    -/
  · rw [gcd_div gcd_dvd_left gcd_dvd_right, natAbs_ofNat,
      Nat.div_self (gcd_pos_of_ne_zero_left c hm.ne')]


/-- To cancel a common factor `c` from a `ModEq` we must divide the modulus `m` by `gcd m c`. -/
theorem cancel_left_div_gcd (hm : 0 < m) (h : c * a ≡ c * b [ZMOD m]) : a ≡ b [ZMOD m / gcd m c] :=
                                /-
                                  m a b c : Int
                                  hm : LT.lt 0 m
                                  h : m.ModEq (HMul.hMul c a) (HMul.hMul c b)
                                  ⊢ m.ModEq (HMul.hMul a c) (HMul.hMul b c)
                                -/
  cancel_right_div_gcd hm <| by simpa [mul_comm] using h
                                /-
                                  🎉 no goals
                                -/


theorem of_div (h : a / c ≡ b / c [ZMOD m / c]) (ha : c ∣ a) (ha : c ∣ b) (ha : c ∣ m) :
                         /-
                           m a b c : Int
                           h : (HDiv.hDiv m c).ModEq (HDiv.hDiv a c) (HDiv.hDiv b c)
                           ha✝¹ : Dvd.dvd c a
                           ha✝ : Dvd.dvd c b
                           ha : Dvd.dvd c m
                           ⊢ m.ModEq a b
                         -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    a ≡ b [ZMOD m] := by convert h.mul_left' <;> rwa [Int.mul_ediv_cancel']
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem modEq_one : a ≡ b [ZMOD 1] :=
  modEq_of_dvd (one_dvd _)


theorem modEq_sub (a b : ℤ) : a ≡ b [ZMOD a - b] :=
  (modEq_of_dvd dvd_rfl).symm


@[simp]
                                                      /-
                                                        a b : Int
                                                        ⊢ Iff (Int.ModEq 0 a b) (Eq a b)
                                                      -/
theorem modEq_zero_iff : a ≡ b [ZMOD 0] ↔ a = b := by rw [ModEq, emod_zero, emod_zero]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                                                   /-
                                                                                     n a : Int
                                                                                     ⊢ Dvd.dvd n (HSub.hSub (HAdd.hAdd n a) a)
                                                                                   -/
theorem add_modEq_left : n + a ≡ a [ZMOD n] := ModEq.symm <| modEq_iff_dvd.2 <| by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp]
                                                                                    /-
                                                                                      n a : Int
                                                                                      ⊢ Dvd.dvd n (HSub.hSub (HAdd.hAdd a n) a)
                                                                                    -/
theorem add_modEq_right : a + n ≡ a [ZMOD n] := ModEq.symm <| modEq_iff_dvd.2 <| by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem modEq_and_modEq_iff_modEq_mul {a b m n : ℤ} (hmn : m.natAbs.Coprime n.natAbs) :
    a ≡ b [ZMOD m] ∧ a ≡ b [ZMOD n] ↔ a ≡ b [ZMOD m * n] :=
  ⟨fun h => by
    /-
      a b m n : Int
      hmn : m.natAbs.Coprime n.natAbs
      h : And (m.ModEq a b) (n.ModEq a b)
      ⊢ (HMul.hMul m n).ModEq a b
    -/
    rw [modEq_iff_dvd, modEq_iff_dvd] at h
    /-
      a b m n : Int
      hmn : m.natAbs.Coprime n.natAbs
      h : And (Dvd.dvd m (HSub.hSub b a)) (Dvd.dvd n (HSub.hSub b a))
      ⊢ (HMul.hMul m n).ModEq a b
    -/
    rw [modEq_iff_dvd, ← natAbs_dvd, ← dvd_natAbs, natCast_dvd_natCast, natAbs_mul]
    /-
      a b m n : Int
      hmn : m.natAbs.Coprime n.natAbs
      h : And (Dvd.dvd m (HSub.hSub b a)) (Dvd.dvd n (HSub.hSub b a))
      ⊢ Dvd.dvd (HMul.hMul m.natAbs n.natAbs) (HSub.hSub b a).natAbs
    -/
    refine hmn.mul_dvd_of_dvd_of_dvd ?_ ?_ <;>
      /-
        case refine_1
        a b m n : Int
        hmn : m.natAbs.Coprime n.natAbs
        h : And (Dvd.dvd m (HSub.hSub b a)) (Dvd.dvd n (HSub.hSub b a))
        ⊢ Dvd.dvd m.natAbs (HSub.hSub b a).natAbs
      -/
      rw [← natCast_dvd_natCast, natAbs_dvd, dvd_natAbs] <;>
      /-
        case refine_1
        a b m n : Int
        hmn : m.natAbs.Coprime n.natAbs
        h : And (Dvd.dvd m (HSub.hSub b a)) (Dvd.dvd n (HSub.hSub b a))
        ⊢ Dvd.dvd m (HSub.hSub b a)
      -/
      /-
        🎉 no goals
      -/
      tauto,
      /-
        🎉 no goals
      -/
    fun h => ⟨h.of_mul_right _, h.of_mul_left _⟩⟩


theorem gcd_a_modEq (a b : ℕ) : (a : ℤ) * Nat.gcdA a b ≡ Nat.gcd a b [ZMOD b] := by
  /-
    a b : Nat
    ⊢ (↑b).ModEq (HMul.hMul (↑a) (a.gcdA b)) ↑(a.gcd b)
  -/
  rw [← add_zero ((a : ℤ) * _), Nat.gcd_eq_gcd_ab]
  /-
    a b : Nat
    ⊢ (↑b).ModEq (HAdd.hAdd (HMul.hMul (↑a) (a.gcdA b)) 0) (HAdd.hAdd (HMul.hMul ( …
  -/
  exact (dvd_mul_right _ _).zero_modEq_int.add_left _
  /-
    🎉 no goals
  -/


theorem modEq_add_fac {a b n : ℤ} (c : ℤ) (ha : a ≡ b [ZMOD n]) : a + n * c ≡ b [ZMOD n] :=
  calc
    a + n * c ≡ b + n * c [ZMOD n] := ha.add_right _
    _ ≡ b + 0 [ZMOD n] := (dvd_mul_right _ _).modEq_zero_int.add_left _
                         /-
                           a b n c : Int
                           ha : n.ModEq a b
                           ⊢ n.ModEq (HAdd.hAdd b 0) b
                         -/
    _ ≡ b [ZMOD n] := by rw [add_zero]
                         /-
                           🎉 no goals
                         -/


theorem modEq_sub_fac {a b n : ℤ} (c : ℤ) (ha : a ≡ b [ZMOD n]) : a - n * c ≡ b [ZMOD n] := by
  /-
    a b n c : Int
    ha : n.ModEq a b
    ⊢ n.ModEq (HSub.hSub a (HMul.hMul n c)) b
  -/
  convert Int.modEq_add_fac (-c) ha using 1; rw [Int.mul_neg, sub_eq_add_neg]
                                             /-
                                               🎉 no goals
                                             -/


theorem modEq_add_fac_self {a t n : ℤ} : a + n * t ≡ a [ZMOD n] :=
  modEq_add_fac _ ModEq.rfl


theorem mod_coprime {a b : ℕ} (hab : Nat.Coprime a b) : ∃ y : ℤ, a * y ≡ 1 [ZMOD b] :=
  ⟨Nat.gcdA a b,
    have hgcd : Nat.gcd a b = 1 := Nat.Coprime.gcd_eq_one hab
    calc
      ↑a * Nat.gcdA a b ≡ ↑a * Nat.gcdA a b + ↑b * Nat.gcdB a b [ZMOD ↑b] :=
        ModEq.symm <| modEq_add_fac _ <| ModEq.refl _
                            /-
                              a b : Nat
                              hab : a.Coprime b
                              hgcd : Eq (a.gcd b) 1
                              ⊢ (↑b).ModEq (HAdd.hAdd (HMul.hMul (↑a) (a.gcdA b)) (HMul.hMul (↑b) (a.gcdB b) …
                            -/
      _ ≡ 1 [ZMOD ↑b] := by rw [← Nat.gcd_eq_gcd_ab, hgcd]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/
      ⟩


theorem existsUnique_equiv (a : ℤ) {b : ℤ} (hb : 0 < b) :
    ∃ z : ℤ, 0 ≤ z ∧ z < b ∧ z ≡ a [ZMOD b] :=
  ⟨a % b, emod_nonneg _ (ne_of_gt hb),
    by
      /-
        a b : Int
        hb : LT.lt 0 b
        ⊢ LT.lt (HMod.hMod a b) b
      -/
      have : a % b < |b| := emod_lt _ (ne_of_gt hb)
      /-
        a b : Int
        hb : LT.lt 0 b
        this : LT.lt (HMod.hMod a b) (abs b)
        ⊢ LT.lt (HMod.hMod a b) b
      -/
      /-
        🎉 no goals
      -/
      rwa [abs_of_pos hb] at this, by simp [ModEq]⟩
                                      /-
                                        🎉 no goals
                                      -/


@[deprecated (since := "2024-12-17")] alias exists_unique_equiv := existsUnique_equiv


theorem existsUnique_equiv_nat (a : ℤ) {b : ℤ} (hb : 0 < b) : ∃ z : ℕ, ↑z < b ∧ ↑z ≡ a [ZMOD b] :=
  let ⟨z, hz1, hz2, hz3⟩ := existsUnique_equiv a hb
  ⟨z.natAbs, by
    /-
      a b : Int
      hb : LT.lt 0 b
      z : Int
      hz1 : LE.le 0 z
      hz2 : LT.lt z b
      hz3 : b.ModEq z a
      ⊢ And (LT.lt (↑z.natAbs) b) (b.ModEq (↑z.natAbs) a)
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    constructor <;> rw [natAbs_of_nonneg hz1] <;> assumption⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[deprecated (since := "2024-12-17")] alias exists_unique_equiv_nat := existsUnique_equiv_nat


theorem mod_mul_right_mod (a b c : ℤ) : a % (b * c) % b = a % b :=
  (mod_modEq _ _).of_mul_right _


theorem mod_mul_left_mod (a b c : ℤ) : a % (b * c) % c = a % c :=
  (mod_modEq _ _).of_mul_left _


@[deprecated (since := "2024-04-02")] alias coe_nat_modEq_iff := natCast_modEq_iff


