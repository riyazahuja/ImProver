theorem IsNilpotent.neg [Ring R] (h : IsNilpotent x) : IsNilpotent (-x) := by
  /-
    R : Type u_1
    x : R
    inst✝ : Ring R
    h : IsNilpotent x
    ⊢ IsNilpotent (Neg.neg x)
  -/
  obtain ⟨n, hn⟩ := h
  /-
    case intro
    R : Type u_1
    x : R
    inst✝ : Ring R
    n : Nat
    hn : Eq (HPow.hPow x n) 0
    ⊢ IsNilpotent (Neg.neg x)
  -/
  use n
  /-
    case h
    R : Type u_1
    x : R
    inst✝ : Ring R
    n : Nat
    hn : Eq (HPow.hPow x n) 0
    ⊢ Eq (HPow.hPow (Neg.neg x) n) 0
  -/
  rw [neg_pow, hn, mul_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem isNilpotent_neg_iff [Ring R] : IsNilpotent (-x) ↔ IsNilpotent x :=
  ⟨fun h => neg_neg x ▸ h.neg, fun h => h.neg⟩


lemma IsNilpotent.smul [MonoidWithZero R] [MonoidWithZero S] [MulActionWithZero R S]
    [SMulCommClass R S S] [IsScalarTower R S S] {a : S} (ha : IsNilpotent a) (t : R) :
    IsNilpotent (t • a) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    inst✝² : MulActionWithZero R S
    inst✝¹ : SMulCommClass R S S
    inst✝ : IsScalarTower R S S
    a : S
    ha : IsNilpotent a
    t : R
    ⊢ IsNilpotent (HSMul.hSMul t a)
  -/
  obtain ⟨k, ha⟩ := ha
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    inst✝² : MulActionWithZero R S
    inst✝¹ : SMulCommClass R S S
    inst✝ : IsScalarTower R S S
    a : S
    t : R
    k : Nat
    ha : Eq (HPow.hPow a k) 0
    ⊢ IsNilpotent (HSMul.hSMul t a)
  -/
  use k
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    inst✝² : MulActionWithZero R S
    inst✝¹ : SMulCommClass R S S
    inst✝ : IsScalarTower R S S
    a : S
    t : R
    k : Nat
    ha : Eq (HPow.hPow a k) 0
    ⊢ Eq (HPow.hPow (HSMul.hSMul t a) k) 0
  -/
  rw [smul_pow, ha, smul_zero]
  /-
    🎉 no goals
  -/


theorem IsNilpotent.isUnit_sub_one [Ring R] {r : R} (hnil : IsNilpotent r) : IsUnit (r - 1) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    r : R
    hnil : IsNilpotent r
    ⊢ IsUnit (HSub.hSub r 1)
  -/
  obtain ⟨n, hn⟩ := hnil
  /-
    case intro
    R : Type u_1
    inst✝ : Ring R
    r : R
    n : Nat
    hn : Eq (HPow.hPow r n) 0
    ⊢ IsUnit (HSub.hSub r 1)
  -/
  refine ⟨⟨r - 1, -∑ i ∈ Finset.range n, r ^ i, ?_, ?_⟩, rfl⟩
    /-
      case intro.refine_1
      R : Type u_1
      inst✝ : Ring R
      r : R
      n : Nat
      hn : Eq (HPow.hPow r n) 0
      ⊢ Eq (HMul.hMul (HSub.hSub r 1) (Neg.neg ((Finset.range n).sum fun i => HPow.h …
    -/
  · simp [mul_geom_sum, hn]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      R : Type u_1
      inst✝ : Ring R
      r : R
      n : Nat
      hn : Eq (HPow.hPow r n) 0
      ⊢ Eq (HMul.hMul (Neg.neg ((Finset.range n).sum fun i => HPow.hPow r i)) (HSub. …
    -/
  · simp [geom_sum_mul, hn]
    /-
      🎉 no goals
    -/


theorem IsNilpotent.isUnit_one_sub [Ring R] {r : R} (hnil : IsNilpotent r) : IsUnit (1 - r) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    r : R
    hnil : IsNilpotent r
    ⊢ IsUnit (HSub.hSub 1 r)
  -/
  rw [← IsUnit.neg_iff, neg_sub]
  /-
    R : Type u_1
    inst✝ : Ring R
    r : R
    hnil : IsNilpotent r
    ⊢ IsUnit (HSub.hSub r 1)
  -/
  exact isUnit_sub_one hnil
  /-
    🎉 no goals
  -/


theorem IsNilpotent.isUnit_add_one [Ring R] {r : R} (hnil : IsNilpotent r) : IsUnit (r + 1) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    r : R
    hnil : IsNilpotent r
    ⊢ IsUnit (HAdd.hAdd r 1)
  -/
  rw [← IsUnit.neg_iff, neg_add']
  /-
    R : Type u_1
    inst✝ : Ring R
    r : R
    hnil : IsNilpotent r
    ⊢ IsUnit (HSub.hSub (Neg.neg r) 1)
  -/
  exact isUnit_sub_one hnil.neg
  /-
    🎉 no goals
  -/


theorem IsNilpotent.isUnit_one_add [Ring R] {r : R} (hnil : IsNilpotent r) : IsUnit (1 + r) :=
  add_comm r 1 ▸ isUnit_add_one hnil


theorem IsNilpotent.isUnit_add_left_of_commute [Ring R] {r u : R}
    (hnil : IsNilpotent r) (hu : IsUnit u) (h_comm : Commute r u) :
    IsUnit (u + r) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    r u : R
    hnil : IsNilpotent r
    hu : IsUnit u
    h_comm : Commute r u
    ⊢ IsUnit (HAdd.hAdd u r)
  -/
  rw [← Units.isUnit_mul_units _ hu.unit⁻¹, add_mul, IsUnit.mul_val_inv]
  /-
    R : Type u_1
    inst✝ : Ring R
    r u : R
    hnil : IsNilpotent r
    hu : IsUnit u
    h_comm : Commute r u
    ⊢ IsUnit (HAdd.hAdd 1 (HMul.hMul r ↑(Inv.inv hu.unit)))
  -/
  replace h_comm : Commute r (↑hu.unit⁻¹) := Commute.units_inv_right h_comm
  /-
    R : Type u_1
    inst✝ : Ring R
    r u : R
    hnil : IsNilpotent r
    hu : IsUnit u
    h_comm : Commute r ↑(Inv.inv hu.unit)
    ⊢ IsUnit (HAdd.hAdd 1 (HMul.hMul r ↑(Inv.inv hu.unit)))
  -/
  refine IsNilpotent.isUnit_one_add ?_
  /-
    R : Type u_1
    inst✝ : Ring R
    r u : R
    hnil : IsNilpotent r
    hu : IsUnit u
    h_comm : Commute r ↑(Inv.inv hu.unit)
    ⊢ IsNilpotent (HMul.hMul r ↑(Inv.inv hu.unit))
  -/
  exact (hu.unit⁻¹.isUnit.isNilpotent_mul_unit_of_commute_iff h_comm).mpr hnil
  /-
    🎉 no goals
  -/


theorem IsNilpotent.isUnit_add_right_of_commute [Ring R] {r u : R}
    (hnil : IsNilpotent r) (hu : IsUnit u) (h_comm : Commute r u) :
    IsUnit (r + u) :=
  add_comm r u ▸ hnil.isUnit_add_left_of_commute hu h_comm


lemma IsUnit.not_isNilpotent [Ring R] [Nontrivial R] {x : R} (hx : IsUnit x) :
    ¬ IsNilpotent x := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    x : R
    hx : IsUnit x
    ⊢ Not (IsNilpotent x)
  -/
  intro H
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    x : R
    hx : IsUnit x
    H : IsNilpotent x
    ⊢ False
  -/
  simpa using H.isUnit_add_right_of_commute hx.neg (by simp)
  /-
    🎉 no goals
  -/


lemma IsNilpotent.not_isUnit [Ring R] [Nontrivial R] {x : R} (hx : IsNilpotent x) :
    ¬ IsUnit x :=
                                /-
                                  R : Type u_1
                                  inst✝¹ : Ring R
                                  inst✝ : Nontrivial R
                                  x : R
                                  hx : IsNilpotent x
                                  ⊢ Not (Not (IsNilpotent x))
                                -/
  mt IsUnit.not_isNilpotent (by simpa only [not_not] using hx)
                                /-
                                  🎉 no goals
                                -/


lemma IsIdempotentElem.eq_zero_of_isNilpotent [MonoidWithZero R] {e : R}
    (idem : IsIdempotentElem e) (nilp : IsNilpotent e) : e = 0 := by
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    e : R
    idem : IsIdempotentElem e
    nilp : IsNilpotent e
    ⊢ Eq e 0
  -/
  obtain ⟨rfl | n, hn⟩ := nilp
    /-
      case intro.zero
      R : Type u_1
      inst✝ : MonoidWithZero R
      e : R
      idem : IsIdempotentElem e
      hn : Eq (HPow.hPow e 0) 0
      ⊢ Eq e 0
    -/
  · rw [pow_zero] at hn; rw [← one_mul e, hn, zero_mul]
                         /-
                           🎉 no goals
                         -/
    /-
      case intro.succ
      R : Type u_1
      inst✝ : MonoidWithZero R
      e : R
      idem : IsIdempotentElem e
      n : Nat
      hn : Eq (HPow.hPow e (HAdd.hAdd n 1)) 0
      ⊢ Eq e 0
    -/
  · rw [← hn, idem.pow_succ_eq]
    /-
      🎉 no goals
    -/


alias IsNilpotent.eq_zero_of_isIdempotentElem := IsIdempotentElem.eq_zero_of_isNilpotent


instance [Zero R] [Pow R ℕ] [Zero S] [Pow S ℕ] [IsReduced R] [IsReduced S] : IsReduced (R × S) where
  eq_zero _ := fun ⟨n, hn⟩ ↦ have hn := Prod.ext_iff.1 hn
    Prod.ext (IsReduced.eq_zero _ ⟨n, hn.1⟩) (IsReduced.eq_zero _ ⟨n, hn.2⟩)


theorem Prime.isRadical [CommMonoidWithZero R] {y : R} (hy : Prime y) : IsRadical y :=
  fun _ _ ↦ hy.dvd_of_dvd_pow


theorem zero_isRadical_iff [MonoidWithZero R] : IsRadical (0 : R) ↔ IsReduced R := by
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    ⊢ Iff (IsRadical 0) (IsReduced R)
  -/
  simp_rw [isReduced_iff, IsNilpotent, exists_imp, ← zero_dvd_iff]
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    ⊢ Iff (IsRadical 0) (∀ (x : R) (x_1 : Nat), Dvd.dvd 0 (HPow.hPow x x_1) → Dvd. …
  -/
  exact forall_swap
  /-
    🎉 no goals
  -/


theorem isReduced_iff_pow_one_lt [MonoidWithZero R] (k : ℕ) (hk : 1 < k) :
    IsReduced R ↔ ∀ x : R, x ^ k = 0 → x = 0 := by
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    k : Nat
    hk : LT.lt 1 k
    ⊢ Iff (IsReduced R) (∀ (x : R), Eq (HPow.hPow x k) 0 → Eq x 0)
  -/
  simp_rw [← zero_isRadical_iff, isRadical_iff_pow_one_lt k hk, zero_dvd_iff]
  /-
    🎉 no goals
  -/


theorem IsRadical.of_dvd [CancelCommMonoidWithZero R] {x y : R} (hy : IsRadical y) (h0 : y ≠ 0)
    (hxy : x ∣ y) : IsRadical x := (isRadical_iff_pow_one_lt 2 one_lt_two).2 <| by
  /-
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x y : R
    hy : IsRadical y
    h0 : Ne y 0
    hxy : Dvd.dvd x y
    ⊢ ∀ (x_1 : R), Dvd.dvd x (HPow.hPow x_1 2) → Dvd.dvd x x_1
  -/
  obtain ⟨z, rfl⟩ := hxy
  /-
    case intro
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x z : R
    hy : IsRadical (HMul.hMul x z)
    h0 : Ne (HMul.hMul x z) 0
    ⊢ ∀ (x_1 : R), Dvd.dvd x (HPow.hPow x_1 2) → Dvd.dvd x x_1
  -/
  refine fun w dvd ↦ ((mul_dvd_mul_iff_right <| right_ne_zero_of_mul h0).mp <| hy 2 _ ?_)
  /-
    case intro
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    x z : R
    hy : IsRadical (HMul.hMul x z)
    h0 : Ne (HMul.hMul x z) 0
    w : R
    dvd : Dvd.dvd x (HPow.hPow w 2)
    ⊢ Dvd.dvd (HMul.hMul x z) (HPow.hPow (HMul.hMul w z) 2)
  -/
  rw [mul_pow, sq z]; exact mul_dvd_mul dvd (dvd_mul_left z z)
                      /-
                        🎉 no goals
                      -/


theorem add_pow_eq_zero_of_add_le_succ_of_pow_eq_zero (h_comm : Commute x y) {m n k : ℕ}
    (hx : x ^ m = 0) (hy : y ^ n = 0) (h : m + n ≤ k + 1) :
    (x + y) ^ k = 0 := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) k) 0
  -/
  rw [h_comm.add_pow']
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal k).sum fun m => HSMul.hSMul (k.choo …
  -/
  apply Finset.sum_eq_zero
  /-
    case h
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    ⊢ ∀ (x_1 : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal  …
  -/
  rintro ⟨i, j⟩ hij
  /-
    case h.mk
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) { fst := i, snd : …
    ⊢ Eq (HSMul.hSMul (k.choose { fst := i, snd := j }.1) (HMul.hMul (HPow.hPow x  …
  -/
  suffices x ^ i * y ^ j = 0 by simp only [this, nsmul_eq_mul, mul_zero]
  /-
    case h.mk
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) { fst := i, snd : …
    ⊢ Eq (HMul.hMul (HPow.hPow x i) (HPow.hPow y j)) 0
  -/
  by_cases hi : m ≤ i
    /-
      case pos
      R : Type u_1
      x y : R
      inst✝ : Semiring R
      h_comm : Commute x y
      m n k : Nat
      hx : Eq (HPow.hPow x m) 0
      hy : Eq (HPow.hPow y n) 0
      h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
      i j : Nat
      hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) { fst := i, snd : …
      hi : LE.le m i
      ⊢ Eq (HMul.hMul (HPow.hPow x i) (HPow.hPow y j)) 0
    -/
  · rw [pow_eq_zero_of_le hi hx, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) { fst := i, snd : …
    hi : Not (LE.le m i)
    ⊢ Eq (HMul.hMul (HPow.hPow x i) (HPow.hPow y j)) 0
  -/
  rw [pow_eq_zero_of_le ?_ hy, mul_zero]
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    m n k : Nat
    hx : Eq (HPow.hPow x m) 0
    hy : Eq (HPow.hPow y n) 0
    h : LE.le (HAdd.hAdd m n) (HAdd.hAdd k 1)
    i j : Nat
    hij : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) { fst := i, snd : …
    hi : Not (LE.le m i)
    ⊢ LE.le n j
  -/
  linarith [Finset.mem_antidiagonal.mp hij]
  /-
    🎉 no goals
  -/


theorem add_pow_add_eq_zero_of_pow_eq_zero (h_comm : Commute x y) {m n : ℕ}
    (hx : x ^ m = 0) (hy : y ^ n = 0) :
    (x + y) ^ (m + n - 1) = 0 :=
                                                                   /-
                                                                     R : Type u_1
                                                                     x y : R
                                                                     inst✝ : Semiring R
                                                                     h_comm : Commute x y
                                                                     m n : Nat
                                                                     hx : Eq (HPow.hPow x m) 0
                                                                     hy : Eq (HPow.hPow y n) 0
                                                                     ⊢ LE.le (HAdd.hAdd m n) (HAdd.hAdd (HSub.hSub (HAdd.hAdd m n) 1) 1)
                                                                   -/
  h_comm.add_pow_eq_zero_of_add_le_succ_of_pow_eq_zero hx hy <| by rw [← Nat.sub_le_iff_le_add]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem isNilpotent_add (h_comm : Commute x y) (hx : IsNilpotent x) (hy : IsNilpotent y) :
    IsNilpotent (x + y) := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hx : IsNilpotent x
    hy : IsNilpotent y
    ⊢ IsNilpotent (HAdd.hAdd x y)
  -/
  obtain ⟨n, hn⟩ := hx
  /-
    case intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hy : IsNilpotent y
    n : Nat
    hn : Eq (HPow.hPow x n) 0
    ⊢ IsNilpotent (HAdd.hAdd x y)
  -/
  obtain ⟨m, hm⟩ := hy
  /-
    case intro.intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    n : Nat
    hn : Eq (HPow.hPow x n) 0
    m : Nat
    hm : Eq (HPow.hPow y m) 0
    ⊢ IsNilpotent (HAdd.hAdd x y)
  -/
  exact ⟨_, add_pow_add_eq_zero_of_pow_eq_zero h_comm hn hm⟩
  /-
    🎉 no goals
  -/


protected lemma isNilpotent_sum {ι : Type*} {s : Finset ι} {f : ι → R}
    (hnp : ∀ i ∈ s, IsNilpotent (f i)) (h_comm : ∀ i j, i ∈ s → j ∈ s → Commute (f i) (f j)) :
    IsNilpotent (∑ i ∈ s, f i) := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | @insert j s hj ih => ?_
  rw [Finset.sum_insert hj]
  apply Commute.isNilpotent_add
  · exact Commute.sum_right _ _ _ (fun i hi ↦ h_comm _ _ (by simp) (by simp [hi]))
  · apply hnp; simp
  · exact ih (fun i hi ↦ hnp i (by simp [hi]))
      (fun i j hi hj ↦ h_comm i j (by simp [hi]) (by simp [hj]))


protected lemma isNilpotent_mul_left_iff (h_comm : Commute x y) (hy : y ∈ nonZeroDivisorsLeft R) :
    IsNilpotent (x * y) ↔ IsNilpotent x := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hy : Membership.mem (nonZeroDivisorsLeft R) y
    ⊢ Iff (IsNilpotent (HMul.hMul x y)) (IsNilpotent x)
  -/
  refine ⟨?_, h_comm.isNilpotent_mul_left⟩
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hy : Membership.mem (nonZeroDivisorsLeft R) y
    ⊢ IsNilpotent (HMul.hMul x y) → IsNilpotent x
  -/
  rintro ⟨k, hk⟩
  /-
    case intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hy : Membership.mem (nonZeroDivisorsLeft R) y
    k : Nat
    hk : Eq (HPow.hPow (HMul.hMul x y) k) 0
    ⊢ IsNilpotent x
  -/
  rw [mul_pow h_comm] at hk
  /-
    case intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hy : Membership.mem (nonZeroDivisorsLeft R) y
    k : Nat
    hk : Eq (HMul.hMul (HPow.hPow x k) (HPow.hPow y k)) 0
    ⊢ IsNilpotent x
  -/
  exact ⟨k, (nonZeroDivisorsLeft R).pow_mem hy k _ hk⟩
  /-
    🎉 no goals
  -/


protected lemma isNilpotent_mul_right_iff (h_comm : Commute x y) (hx : x ∈ nonZeroDivisorsRight R) :
    IsNilpotent (x * y) ↔ IsNilpotent y := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hx : Membership.mem (nonZeroDivisorsRight R) x
    ⊢ Iff (IsNilpotent (HMul.hMul x y)) (IsNilpotent y)
  -/
  refine ⟨?_, h_comm.isNilpotent_mul_right⟩
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hx : Membership.mem (nonZeroDivisorsRight R) x
    ⊢ IsNilpotent (HMul.hMul x y) → IsNilpotent y
  -/
  rintro ⟨k, hk⟩
  /-
    case intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hx : Membership.mem (nonZeroDivisorsRight R) x
    k : Nat
    hk : Eq (HPow.hPow (HMul.hMul x y) k) 0
    ⊢ IsNilpotent y
  -/
  rw [mul_pow h_comm] at hk
  /-
    case intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    hx : Membership.mem (nonZeroDivisorsRight R) x
    k : Nat
    hk : Eq (HMul.hMul (HPow.hPow x k) (HPow.hPow y k)) 0
    ⊢ IsNilpotent y
  -/
  exact ⟨k, (nonZeroDivisorsRight R).pow_mem hx k _ hk⟩
  /-
    🎉 no goals
  -/


theorem isNilpotent_sub (h_comm : Commute x y) (hx : IsNilpotent x) (hy : IsNilpotent y) :
    IsNilpotent (x - y) := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Ring R
    h_comm : Commute x y
    hx : IsNilpotent x
    hy : IsNilpotent y
    ⊢ IsNilpotent (HSub.hSub x y)
  -/
  rw [← neg_right_iff] at h_comm
  /-
    R : Type u_1
    x y : R
    inst✝ : Ring R
    h_comm : Commute x (Neg.neg y)
    hx : IsNilpotent x
    hy : IsNilpotent y
    ⊢ IsNilpotent (HSub.hSub x y)
  -/
  rw [← isNilpotent_neg_iff] at hy
  /-
    R : Type u_1
    x y : R
    inst✝ : Ring R
    h_comm : Commute x (Neg.neg y)
    hx : IsNilpotent x
    hy : IsNilpotent (Neg.neg y)
    ⊢ IsNilpotent (HSub.hSub x y)
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u_1
    x y : R
    inst✝ : Ring R
    h_comm : Commute x (Neg.neg y)
    hx : IsNilpotent x
    hy : IsNilpotent (Neg.neg y)
    ⊢ IsNilpotent (HAdd.hAdd x (Neg.neg y))
  -/
  exact h_comm.isNilpotent_add hx hy
  /-
    🎉 no goals
  -/


lemma isNilpotent_sum {ι : Type*} {s : Finset ι} {f : ι → R}
    (hnp : ∀ i ∈ s, IsNilpotent (f i)) :
    IsNilpotent (∑ i ∈ s, f i) :=
  Commute.isNilpotent_sum hnp fun _ _ _ _ ↦ Commute.all _ _


lemma NoZeroSMulDivisors.isReduced (R M : Type*)
    [MonoidWithZero R] [Zero M] [MulActionWithZero R M] [Nontrivial M] [NoZeroSMulDivisors R M] :
    IsReduced R := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝⁴ : MonoidWithZero R
    inst✝³ : Zero M
    inst✝² : MulActionWithZero R M
    inst✝¹ : Nontrivial M
    inst✝ : NoZeroSMulDivisors R M
    ⊢ IsReduced R
  -/
  refine ⟨fun x ⟨k, hk⟩ ↦ ?_⟩
  /-
    R : Type u_3
    M : Type u_4
    inst✝⁴ : MonoidWithZero R
    inst✝³ : Zero M
    inst✝² : MulActionWithZero R M
    inst✝¹ : Nontrivial M
    inst✝ : NoZeroSMulDivisors R M
    x : R
    x✝ : IsNilpotent x
    k : Nat
    hk : Eq (HPow.hPow x k) 0
    ⊢ Eq x 0
  -/
  induction' k with k ih
    /-
      case zero
      R : Type u_3
      M : Type u_4
      inst✝⁴ : MonoidWithZero R
      inst✝³ : Zero M
      inst✝² : MulActionWithZero R M
      inst✝¹ : Nontrivial M
      inst✝ : NoZeroSMulDivisors R M
      x : R
      x✝ : IsNilpotent x
      hk : Eq (HPow.hPow x 0) 0
      ⊢ Eq x 0
    -/
  · rw [pow_zero] at hk
    /-
      case zero
      R : Type u_3
      M : Type u_4
      inst✝⁴ : MonoidWithZero R
      inst✝³ : Zero M
      inst✝² : MulActionWithZero R M
      inst✝¹ : Nontrivial M
      inst✝ : NoZeroSMulDivisors R M
      x : R
      x✝ : IsNilpotent x
      hk : Eq 1 0
      ⊢ Eq x 0
    -/
    exact eq_zero_of_zero_eq_one hk.symm x
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_3
      M : Type u_4
      inst✝⁴ : MonoidWithZero R
      inst✝³ : Zero M
      inst✝² : MulActionWithZero R M
      inst✝¹ : Nontrivial M
      inst✝ : NoZeroSMulDivisors R M
      x : R
      x✝ : IsNilpotent x
      k : Nat
      ih : Eq (HPow.hPow x k) 0 → Eq x 0
      hk : Eq (HPow.hPow x (HAdd.hAdd k 1)) 0
      ⊢ Eq x 0
    -/
  · obtain ⟨m : M, hm : m ≠ 0⟩ := exists_ne (0 : M)
    /-
      case succ.intro
      R : Type u_3
      M : Type u_4
      inst✝⁴ : MonoidWithZero R
      inst✝³ : Zero M
      inst✝² : MulActionWithZero R M
      inst✝¹ : Nontrivial M
      inst✝ : NoZeroSMulDivisors R M
      x : R
      x✝ : IsNilpotent x
      k : Nat
      ih : Eq (HPow.hPow x k) 0 → Eq x 0
      hk : Eq (HPow.hPow x (HAdd.hAdd k 1)) 0
      m : M
      hm : Ne m 0
      ⊢ Eq x 0
    -/
    have : x ^ (k + 1) • m = 0 := by simp only [hk, zero_smul]
    /-
      case succ.intro
      R : Type u_3
      M : Type u_4
      inst✝⁴ : MonoidWithZero R
      inst✝³ : Zero M
      inst✝² : MulActionWithZero R M
      inst✝¹ : Nontrivial M
      inst✝ : NoZeroSMulDivisors R M
      x : R
      x✝ : IsNilpotent x
      k : Nat
      ih : Eq (HPow.hPow x k) 0 → Eq x 0
      hk : Eq (HPow.hPow x (HAdd.hAdd k 1)) 0
      m : M
      hm : Ne m 0
      this : Eq (HSMul.hSMul (HPow.hPow x (HAdd.hAdd k 1)) m) 0
      ⊢ Eq x 0
    -/
    rw [pow_succ', mul_smul] at this
    /-
      case succ.intro
      R : Type u_3
      M : Type u_4
      inst✝⁴ : MonoidWithZero R
      inst✝³ : Zero M
      inst✝² : MulActionWithZero R M
      inst✝¹ : Nontrivial M
      inst✝ : NoZeroSMulDivisors R M
      x : R
      x✝ : IsNilpotent x
      k : Nat
      ih : Eq (HPow.hPow x k) 0 → Eq x 0
      hk : Eq (HPow.hPow x (HAdd.hAdd k 1)) 0
      m : M
      hm : Ne m 0
      this : Eq (HSMul.hSMul x (HSMul.hSMul (HPow.hPow x k) m)) 0
      ⊢ Eq x 0
    -/
    rcases eq_zero_or_eq_zero_of_smul_eq_zero this with rfl | hx
      /-
        case succ.intro.inl
        R : Type u_3
        M : Type u_4
        inst✝⁴ : MonoidWithZero R
        inst✝³ : Zero M
        inst✝² : MulActionWithZero R M
        inst✝¹ : Nontrivial M
        inst✝ : NoZeroSMulDivisors R M
        k : Nat
        m : M
        hm : Ne m 0
        x✝ : IsNilpotent 0
        ih : Eq (HPow.hPow 0 k) 0 → Eq 0 0
        hk : Eq (HPow.hPow 0 (HAdd.hAdd k 1)) 0
        this : Eq (HSMul.hSMul 0 (HSMul.hSMul (HPow.hPow 0 k) m)) 0
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case succ.intro.inr
        R : Type u_3
        M : Type u_4
        inst✝⁴ : MonoidWithZero R
        inst✝³ : Zero M
        inst✝² : MulActionWithZero R M
        inst✝¹ : Nontrivial M
        inst✝ : NoZeroSMulDivisors R M
        x : R
        x✝ : IsNilpotent x
        k : Nat
        ih : Eq (HPow.hPow x k) 0 → Eq x 0
        hk : Eq (HPow.hPow x (HAdd.hAdd k 1)) 0
        m : M
        hm : Ne m 0
        this : Eq (HSMul.hSMul x (HSMul.hSMul (HPow.hPow x k) m)) 0
        hx : Eq (HSMul.hSMul (HPow.hPow x k) m) 0
        ⊢ Eq x 0
      -/
    · exact ih <| (eq_zero_or_eq_zero_of_smul_eq_zero hx).resolve_right hm
      /-
        🎉 no goals
      -/

