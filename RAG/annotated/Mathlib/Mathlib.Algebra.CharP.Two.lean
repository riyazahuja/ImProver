theorem two_eq_zero [CharP R 2] : (2 : R) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    inst✝ : CharP R 2
    ⊢ Eq 2 0
  -/
  rw [← Nat.cast_two, CharP.cast_eq_zero]
  /-
    🎉 no goals
  -/


/-- The only hypotheses required to build a `CharP R 2` instance are `1 ≠ 0` and `2 = 0`. -/
theorem of_one_ne_zero_of_two_eq_zero (h₁ : (1 : R) ≠ 0) (h₂ : (2 : R) = 0) : CharP R 2 where
  cast_eq_zero_iff' n := by
    /-
      R : Type u_1
      inst✝ : AddMonoidWithOne R
      h₁ : Ne 1 0
      h₂ : Eq 2 0
      n : Nat
      ⊢ Iff (Eq (↑n) 0) (Dvd.dvd 2 n)
    -/
    obtain hn | hn := Nat.even_or_odd n
      /-
        case inl
        R : Type u_1
        inst✝ : AddMonoidWithOne R
        h₁ : Ne 1 0
        h₂ : Eq 2 0
        n : Nat
        hn : Even n
        ⊢ Iff (Eq (↑n) 0) (Dvd.dvd 2 n)
      -/
    · simp_rw [hn.two_dvd, iff_true]
      /-
        case inl
        R : Type u_1
        inst✝ : AddMonoidWithOne R
        h₁ : Ne 1 0
        h₂ : Eq 2 0
        n : Nat
        hn : Even n
        ⊢ Eq (↑n) 0
      -/
      exact natCast_eq_zero_of_even_of_two_eq_zero hn h₂
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u_1
        inst✝ : AddMonoidWithOne R
        h₁ : Ne 1 0
        h₂ : Eq 2 0
        n : Nat
        hn : Odd n
        ⊢ Iff (Eq (↑n) 0) (Dvd.dvd 2 n)
      -/
    · simp_rw [hn.not_two_dvd_nat, iff_false]
      /-
        case inr
        R : Type u_1
        inst✝ : AddMonoidWithOne R
        h₁ : Ne 1 0
        h₂ : Eq 2 0
        n : Nat
        hn : Odd n
        ⊢ Not (Eq (↑n) 0)
      -/
      rwa [natCast_eq_one_of_odd_of_two_eq_zero hn h₂]
      /-
        🎉 no goals
      -/


@[scoped simp]
                                                   /-
                                                     R : Type u_1
                                                     inst✝¹ : Semiring R
                                                     inst✝ : CharP R 2
                                                     x : R
                                                     ⊢ Eq (HAdd.hAdd x x) 0
                                                   -/
theorem add_self_eq_zero (x : R) : x + x = 0 := by rw [← two_smul R x, two_eq_zero, zero_smul]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[scoped simp]
                                                      /-
                                                        R : Type u_1
                                                        inst✝¹ : Semiring R
                                                        inst✝ : CharP R 2
                                                        x : R
                                                        ⊢ Eq (HSMul.hSMul 2 x) 0
                                                      -/
protected theorem two_nsmul (x : R) : 2 • x = 0 := by rw [two_smul, add_self_eq_zero]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[scoped simp]
theorem neg_eq (x : R) : -x = x := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R 2
    x : R
    ⊢ Eq (Neg.neg x) x
  -/
  rw [neg_eq_iff_add_eq_zero, add_self_eq_zero]
  /-
    🎉 no goals
  -/


theorem neg_eq' : Neg.neg = (id : R → R) :=
  funext neg_eq


@[scoped simp]
                                                   /-
                                                     R : Type u_1
                                                     inst✝¹ : Ring R
                                                     inst✝ : CharP R 2
                                                     x y : R
                                                     ⊢ Eq (HSub.hSub x y) (HAdd.hAdd x y)
                                                   -/
theorem sub_eq_add (x y : R) : x - y = x + y := by rw [sub_eq_add_neg, neg_eq]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated sub_eq_add (since := "2024-10-24")]
theorem sub_eq_add' : HSub.hSub = (· + · : R → R → R) :=
  funext₂ sub_eq_add


theorem add_eq_iff_eq_add {a b c : R} : a + b = c ↔ a = c + b := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R 2
    a b c : R
    ⊢ Iff (Eq (HAdd.hAdd a b) c) (Eq a (HAdd.hAdd c b))
  -/
  rw [← sub_eq_iff_eq_add, sub_eq_add]
  /-
    🎉 no goals
  -/


theorem eq_add_iff_add_eq {a b c : R} : a = b + c ↔ a + c = b := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R 2
    a b c : R
    ⊢ Iff (Eq a (HAdd.hAdd b c)) (Eq (HAdd.hAdd a c) b)
  -/
  rw [← eq_sub_iff_add_eq, sub_eq_add]
  /-
    🎉 no goals
  -/


@[scoped simp]
protected theorem two_zsmul (x : R) : (2 : ℤ) • x = 0 := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R 2
    x : R
    ⊢ Eq (HSMul.hSMul 2 x) 0
  -/
  rw [two_zsmul, add_self_eq_zero]
  /-
    🎉 no goals
  -/


theorem add_sq (x y : R) : (x + y) ^ 2 = x ^ 2 + y ^ 2 :=
  add_pow_char _ _ _


theorem add_mul_self (x y : R) : (x + y) * (x + y) = x * x + y * y := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CharP R 2
    x y : R
    ⊢ Eq (HMul.hMul (HAdd.hAdd x y) (HAdd.hAdd x y)) (HAdd.hAdd (HMul.hMul x x) (H …
  -/
  rw [← pow_two, ← pow_two, ← pow_two, add_sq]
  /-
    🎉 no goals
  -/


theorem list_sum_sq (l : List R) : l.sum ^ 2 = (l.map (· ^ 2)).sum :=
  list_sum_pow_char _ _


theorem list_sum_mul_self (l : List R) : l.sum * l.sum = (List.map (fun x => x * x) l).sum := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CharP R 2
    l : List R
    ⊢ Eq (HMul.hMul l.sum l.sum) (List.map (fun x => HMul.hMul x x) l).sum
  -/
  simp_rw [← pow_two, list_sum_sq]
  /-
    🎉 no goals
  -/


theorem multiset_sum_sq (l : Multiset R) : l.sum ^ 2 = (l.map (· ^ 2)).sum :=
  multiset_sum_pow_char _ _


theorem multiset_sum_mul_self (l : Multiset R) :
                                                                /-
                                                                  R : Type u_1
                                                                  inst✝¹ : CommSemiring R
                                                                  inst✝ : CharP R 2
                                                                  l : Multiset R
                                                                  ⊢ Eq (HMul.hMul l.sum l.sum) (Multiset.map (fun x => HMul.hMul x x) l).sum
                                                                -/
    l.sum * l.sum = (Multiset.map (fun x => x * x) l).sum := by simp_rw [← pow_two, multiset_sum_sq]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem sum_sq (s : Finset ι) (f : ι → R) : (∑ i ∈ s, f i) ^ 2 = ∑ i ∈ s, f i ^ 2 :=
  sum_pow_char _ _ _


theorem sum_mul_self (s : Finset ι) (f : ι → R) :
                                                               /-
                                                                 R : Type u_1
                                                                 ι : Type u_2
                                                                 inst✝¹ : CommSemiring R
                                                                 inst✝ : CharP R 2
                                                                 s : Finset ι
                                                                 f : ι → R
                                                                 ⊢ Eq (HMul.hMul (s.sum fun i => f i) (s.sum fun i => f i)) (s.sum fun i => HMu …
                                                               -/
    ((∑ i ∈ s, f i) * ∑ i ∈ s, f i) = ∑ i ∈ s, f i * f i := by simp_rw [← pow_two, sum_sq]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem neg_one_eq_one_iff [Nontrivial R] : (-1 : R) = 1 ↔ ringChar R = 2 := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    ⊢ Iff (Eq (-1) 1) (Eq (ringChar R) 2)
  -/
  refine ⟨fun h => ?_, fun h => @CharTwo.neg_eq _ _ (ringChar.of_eq h) 1⟩
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    h : Eq (-1) 1
    ⊢ Eq (ringChar R) 2
  -/
  rw [eq_comm, ← sub_eq_zero, sub_neg_eq_add, ← Nat.cast_one, ← Nat.cast_add] at h
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    h : Eq (↑(HAdd.hAdd 1 1)) 0
    ⊢ Eq (ringChar R) 2
  -/
  exact ((Nat.dvd_prime Nat.prime_two).mp (ringChar.dvd h)).resolve_left CharP.ringChar_ne_one
  /-
    🎉 no goals
  -/


@[simp]
theorem orderOf_neg_one [Nontrivial R] : orderOf (-1 : R) = if ringChar R = 2 then 1 else 2 := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    ⊢ Eq (orderOf (-1)) (ite (Eq (ringChar R) 2) 1 2)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : Nontrivial R
      h : Eq (ringChar R) 2
      ⊢ Eq (orderOf (-1)) 1
    -/
  · rw [neg_one_eq_one_iff.2 h, orderOf_one]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    h : Not (Eq (ringChar R) 2)
    ⊢ Eq (orderOf (-1)) 2
  -/
  apply orderOf_eq_prime
    /-
      case neg.hg
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : Nontrivial R
      h : Not (Eq (ringChar R) 2)
      ⊢ Eq (HPow.hPow (-1) 2) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case neg.hg1
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    h : Not (Eq (ringChar R) 2)
    ⊢ Ne (-1) 1
  -/
  simpa [neg_one_eq_one_iff] using h
  /-
    🎉 no goals
  -/


