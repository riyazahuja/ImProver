/--
For any field `F` and natural numbers `d` and `k`, `Polynomial.preHilbertPoly F d k`
is defined as `(d.factorial : F)⁻¹ • ((ascPochhammer F d).comp (X - (C (k : F)) + 1))`.
This is the most basic form of Hilbert polynomials. `Polynomial.preHilbertPoly ℚ d 0`
is exactly the Hilbert polynomial of the polynomial ring `ℚ[X_0,...,X_d]` viewed as
a graded module over itself. In fact, `Polynomial.preHilbertPoly F d k` is the
same as `Polynomial.hilbertPoly ((X : F[X]) ^ k) (d + 1)` for any field `F` and
`d k : ℕ` (see the lemma `Polynomial.hilbertPoly_X_pow_succ`). See also the lemma
`Polynomial.preHilbertPoly_eq_choose_sub_add`, which states that if `CharZero F`,
then for any `d k n : ℕ` with `k ≤ n`, `(Polynomial.preHilbertPoly F d k).eval (n : F)`
equals `(n - k + d).choose d`.
-/
noncomputable def preHilbertPoly (d k : ℕ) : F[X] :=
  (d.factorial : F)⁻¹ • ((ascPochhammer F d).comp (Polynomial.X - (C (k : F)) + 1))


lemma natDegree_preHilbertPoly [CharZero F] (d k : ℕ) :
    (preHilbertPoly F d k).natDegree = d := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    d k : Nat
    ⊢ Eq (Polynomial.preHilbertPoly F d k).natDegree d
  -/
  have hne : (d ! : F) ≠ 0 := by norm_cast; positivity
  rw [preHilbertPoly, natDegree_smul _ (inv_ne_zero hne), natDegree_comp, ascPochhammer_natDegree,
    add_comm_sub, ← C_1, ← map_sub, natDegree_add_C, natDegree_X, mul_one]


lemma coeff_preHilbertPoly_self [CharZero F] (d k : ℕ) :
    (preHilbertPoly F d k).coeff d = (d ! : F)⁻¹ := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    d k : Nat
    ⊢ Eq ((Polynomial.preHilbertPoly F d k).coeff d) (Inv.inv ↑d.factorial)
  -/
  delta preHilbertPoly
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    d k : Nat
    ⊢ Eq ((HSMul.hSMul (Inv.inv ↑d.factorial) ((ascPochhammer F d).comp (HAdd.hAdd …
  -/
  have hne : (d ! : F) ≠ 0 := by norm_cast; positivity
  have heq : d = ((ascPochhammer F d).comp (X - C (k : F) + 1)).natDegree :=
    (natDegree_preHilbertPoly F d k).symm.trans (natDegree_smul _ (inv_ne_zero hne))
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    d k : Nat
    hne : Ne (↑d.factorial) 0
    heq : Eq d ((ascPochhammer F d).comp (HAdd.hAdd (HSub.hSub Polynomial.X (Polyn …
    ⊢ Eq ((HSMul.hSMul (Inv.inv ↑d.factorial) ((ascPochhammer F d).comp (HAdd.hAdd …
  -/
  nth_rw 3 [heq]
  calc
  _ = (d ! : F)⁻¹ • ((ascPochhammer F d).comp (X - C ((k : F) - 1))).leadingCoeff := by
    simp only [sub_add, ← C_1, ← map_sub, coeff_smul, coeff_natDegree]
  _ = (d ! : F)⁻¹ := by
    simp only [leadingCoeff_comp (ne_of_eq_of_ne (natDegree_X_sub_C _) one_ne_zero), Monic.def.1
      (monic_ascPochhammer _ _), one_mul, leadingCoeff_X_sub_C, one_pow, smul_eq_mul, mul_one]


lemma leadingCoeff_preHilbertPoly [CharZero F] (d k : ℕ) :
    (preHilbertPoly F d k).leadingCoeff = (d ! : F)⁻¹ := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    d k : Nat
    ⊢ Eq (Polynomial.preHilbertPoly F d k).leadingCoeff (Inv.inv ↑d.factorial)
  -/
  rw [leadingCoeff, natDegree_preHilbertPoly, coeff_preHilbertPoly_self]
  /-
    🎉 no goals
  -/


lemma preHilbertPoly_eq_choose_sub_add [CharZero F] (d : ℕ) {k n : ℕ} (hkn : k ≤ n):
    (preHilbertPoly F d k).eval (n : F) = (n - k + d).choose d := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    d k n : Nat
    hkn : LE.le k n
    ⊢ Eq (Polynomial.eval (↑n) (Polynomial.preHilbertPoly F d k)) ↑((HAdd.hAdd (HS …
  -/
  have : (d ! : F) ≠ 0 := by norm_cast; positivity
  calc
  _ = (↑d !)⁻¹ * eval (↑(n - k + 1)) (ascPochhammer F d) := by simp [cast_sub hkn, preHilbertPoly]
  _ = (n - k + d).choose d := by
    rw [ascPochhammer_nat_eq_natCast_ascFactorial];
    field_simp [ascFactorial_eq_factorial_mul_choose]


/--
`Polynomial.hilbertPoly p 0 = 0`; for any `d : ℕ`, `Polynomial.hilbertPoly p (d + 1)`
is defined as `∑ i in p.support, (p.coeff i) • Polynomial.preHilbertPoly F d i`. If
`M` is a graded module whose Poincaré series can be written as `p(X)/(1 - X)ᵈ` for some
`p : ℚ[X]` with integer coefficients, then `Polynomial.hilbertPoly p d` is the Hilbert
polynomial of `M`. See also `Polynomial.coeff_mul_invOneSubPow_eq_hilbertPoly_eval`,
which says that `PowerSeries.coeff F n (p * PowerSeries.invOneSubPow F d)` equals
`(Polynomial.hilbertPoly p d).eval (n : F)` for any large enough `n : ℕ`.
-/
noncomputable def hilbertPoly (p : F[X]) : (d : ℕ) → F[X]
  | 0 => 0
  | d + 1 => ∑ i in p.support, (p.coeff i) • preHilbertPoly F d i


lemma hilbertPoly_zero_left (d : ℕ) : hilbertPoly (0 : F[X]) d = 0 := by
  /-
    F : Type u_1
    inst✝ : Field F
    d : Nat
    ⊢ Eq (Polynomial.hilbertPoly 0 d) 0
  -/
  delta hilbertPoly; induction d with
  | zero => simp only
  | succ d _ => simp only [coeff_zero, zero_smul, Finset.sum_const_zero]


lemma hilbertPoly_zero_right (p : F[X]) : hilbertPoly p 0 = 0 := rfl


lemma hilbertPoly_succ (p : F[X]) (d : ℕ) :
    hilbertPoly p (d + 1) = ∑ i in p.support, (p.coeff i) • preHilbertPoly F d i := rfl


lemma hilbertPoly_X_pow_succ (d k : ℕ) :
    hilbertPoly ((X : F[X]) ^ k) (d + 1) = preHilbertPoly F d k := by
  /-
    F : Type u_1
    inst✝ : Field F
    d k : Nat
    ⊢ Eq ((HPow.hPow Polynomial.X k).hilbertPoly (HAdd.hAdd d 1)) (Polynomial.preH …
  -/
  delta hilbertPoly; simp
                     /-
                       🎉 no goals
                     -/


lemma hilbertPoly_add_left (p q : F[X]) (d : ℕ) :
    hilbertPoly (p + q) d = hilbertPoly p d + hilbertPoly q d := by
  /-
    F : Type u_1
    inst✝ : Field F
    p q : Polynomial F
    d : Nat
    ⊢ Eq ((HAdd.hAdd p q).hilbertPoly d) (HAdd.hAdd (p.hilbertPoly d) (q.hilbertPo …
  -/
  delta hilbertPoly
  induction d with
  | zero => simp only [add_zero]
  | succ d _ =>
      simp only [← coeff_add]
      rw [← sum_def _ fun _ r => r • _]
      exact sum_add_index _ _ _ (fun _ => zero_smul ..) (fun _ _ _ => add_smul ..)


lemma hilbertPoly_smul (a : F) (p : F[X]) (d : ℕ) :
    hilbertPoly (a • p) d = a • hilbertPoly p d := by
  /-
    F : Type u_1
    inst✝ : Field F
    a : F
    p : Polynomial F
    d : Nat
    ⊢ Eq ((HSMul.hSMul a p).hilbertPoly d) (HSMul.hSMul a (p.hilbertPoly d))
  -/
  delta hilbertPoly
  induction d with
  | zero => simp only [smul_zero]
  | succ d _ =>
      simp only
      rw [← sum_def _ fun _ r => r • _, ← sum_def _ fun _ r => r • _, Polynomial.smul_sum,
        sum_smul_index' _ _ _ fun i => zero_smul F (preHilbertPoly F d i)]
      simp only [smul_assoc]


variable (F) in
/--
The function that sends any `p : F[X]` to `Polynomial.hilbertPoly p d` is an `F`-linear map from
`F[X]` to `F[X]`.
-/
noncomputable def hilbertPoly_linearMap (d : ℕ) : F[X] →ₗ[F] F[X] where
  toFun p := hilbertPoly p d
  map_add' p q := hilbertPoly_add_left p q d
  map_smul' r p := hilbertPoly_smul r p d


/--
The key property of Hilbert polynomials. If `F` is a field with characteristic `0`, `p : F[X]` and
`d : ℕ`, then for any large enough `n : ℕ`, `(Polynomial.hilbertPoly p d).eval (n : F)` equals the
coefficient of `Xⁿ` in the power series expansion of `p/(1 - X)ᵈ`.
-/
theorem coeff_mul_invOneSubPow_eq_hilbertPoly_eval
    {p : F[X]} (d : ℕ) {n : ℕ} (hn : p.natDegree < n) :
    PowerSeries.coeff F n (p * invOneSubPow F d) = (hilbertPoly p d).eval (n : F) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d n : Nat
    hn : LT.lt p.natDegree n
    ⊢ Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(PowerSeries.invOneSubPow F d)))  …
  -/
  delta hilbertPoly; induction d with
  | zero => simp only [invOneSubPow_zero, Units.val_one, mul_one, coeff_coe, eval_zero]
            exact coeff_eq_zero_of_natDegree_lt hn
  | succ d hd =>
      simp only [eval_finset_sum, eval_smul, smul_eq_mul]
      rw [← Finset.sum_coe_sort]
      have h_le (i : p.support) : (i : ℕ) ≤ n :=
        le_trans (le_natDegree_of_ne_zero <| mem_support_iff.1 i.2) hn.le
      have h (i : p.support) : eval ↑n (preHilbertPoly F d ↑i) = (n + d - ↑i).choose d := by
        rw [preHilbertPoly_eq_choose_sub_add _ _ (h_le i), Nat.sub_add_comm (h_le i)]
      simp_rw [h]
      rw [Finset.sum_coe_sort _ (fun x => (p.coeff ↑x) * (_ + d - ↑x).choose _),
        PowerSeries.coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk,
        invOneSubPow_val_eq_mk_sub_one_add_choose_of_pos _ _ (zero_lt_succ d)]
      simp only [coeff_coe, coeff_mk]
      symm
      refine Finset.sum_subset_zero_on_sdiff (fun s hs ↦ ?_) (fun x hx ↦ ?_) (fun x hx ↦ ?_)
      · rw [Finset.mem_range_succ_iff]
        exact h_le ⟨s, hs⟩
      · simp only [Finset.mem_sdiff, mem_support_iff, not_not] at hx
        rw [hx.2, zero_mul]
      · rw [add_comm, Nat.add_sub_assoc (h_le ⟨x, hx⟩), succ_eq_add_one, add_tsub_cancel_right]


/--
The polynomial satisfying the key property of `Polynomial.hilbertPoly p d` is unique.
-/
theorem existsUnique_hilbertPoly (p : F[X]) (d : ℕ) :
    ∃! h : F[X], ∃ N : ℕ, ∀ n > N,
    PowerSeries.coeff F n (p * invOneSubPow F d) = h.eval (n : F) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    ⊢ ExistsUnique fun h => Exists fun N => ∀ (n : Nat), GT.gt n N → Eq ((PowerSer …
  -/
  use hilbertPoly p d; constructor
    /-
      case h.left
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      ⊢ (fun h => Exists fun N => ∀ (n : Nat), GT.gt n N → Eq ((PowerSeries.coeff F  …
    -/
  · use p.natDegree
    /-
      case h
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      ⊢ ∀ (n : Nat), GT.gt n p.natDegree → Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p …
    -/
    exact fun n => coeff_mul_invOneSubPow_eq_hilbertPoly_eval d
    /-
      🎉 no goals
    -/
    /-
      case h.right
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      ⊢ ∀ (y : Polynomial F), (fun h => Exists fun N => ∀ (n : Nat), GT.gt n N → Eq  …
    -/
  · rintro h ⟨N, hhN⟩
    /-
      case h.right.intro
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      h : Polynomial F
      N : Nat
      hhN : ∀ (n : Nat), GT.gt n N → Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(Pow …
      ⊢ Eq h (p.hilbertPoly d)
    -/
    apply eq_of_infinite_eval_eq h (hilbertPoly p d)
    /-
      case h.right.intro
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      h : Polynomial F
      N : Nat
      hhN : ∀ (n : Nat), GT.gt n N → Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(Pow …
      ⊢ (setOf fun x => Eq (Polynomial.eval x h) (Polynomial.eval x (p.hilbertPoly d …
    -/
    apply ((Set.Ioi_infinite (max N p.natDegree)).image cast_injective.injOn).mono
    /-
      case h.right.intro
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      h : Polynomial F
      N : Nat
      hhN : ∀ (n : Nat), GT.gt n N → Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(Pow …
      ⊢ HasSubset.Subset (Set.image Nat.cast (Set.Ioi (Max.max N p.natDegree))) (set …
    -/
    rintro x ⟨n, hn, rfl⟩
    /-
      case h.right.intro.intro.intro
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      h : Polynomial F
      N : Nat
      hhN : ∀ (n : Nat), GT.gt n N → Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(Pow …
      n : Nat
      hn : Membership.mem (Set.Ioi (Max.max N p.natDegree)) n
      ⊢ Membership.mem (setOf fun x => Eq (Polynomial.eval x h) (Polynomial.eval x ( …
    -/
    simp only [Set.mem_Ioi, sup_lt_iff, Set.mem_setOf_eq] at hn ⊢
    /-
      case h.right.intro.intro.intro
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      h : Polynomial F
      N : Nat
      hhN : ∀ (n : Nat), GT.gt n N → Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(Pow …
      n : Nat
      hn : And (LT.lt N n) (LT.lt p.natDegree n)
      ⊢ Eq (Polynomial.eval (↑n) h) (Polynomial.eval (↑n) (p.hilbertPoly d))
    -/
    rw [← coeff_mul_invOneSubPow_eq_hilbertPoly_eval d hn.2, hhN n hn.1]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-17")] alias exists_unique_hilbertPoly := existsUnique_hilbertPoly


/--
If `h : F[X]` and there exists some `N : ℕ` such that for any number `n : ℕ` bigger than `N`
we have `PowerSeries.coeff F n (p * invOneSubPow F d) = h.eval (n : F)`, then `h` is exactly
`Polynomial.hilbertPoly p d`.
-/
theorem eq_hilbertPoly_of_forall_coeff_eq_eval
    {p h : F[X]} {d : ℕ} (N : ℕ) (hhN : ∀ n > N,
    PowerSeries.coeff F n (p * invOneSubPow F d) = h.eval (n : F)) :
    h = hilbertPoly p d :=
  ExistsUnique.unique (existsUnique_hilbertPoly p d) ⟨N, hhN⟩
    ⟨p.natDegree, fun _ x => coeff_mul_invOneSubPow_eq_hilbertPoly_eval d x⟩


lemma hilbertPoly_mul_one_sub_succ (p : F[X]) (d : ℕ) :
    hilbertPoly (p * (1 - X)) (d + 1) = hilbertPoly p d := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    ⊢ Eq ((HMul.hMul p (HSub.hSub 1 Polynomial.X)).hilbertPoly (HAdd.hAdd d 1)) (p …
  -/
  apply eq_hilbertPoly_of_forall_coeff_eq_eval (p * (1 - X)).natDegree
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    ⊢ ∀ (n : Nat), GT.gt n (HMul.hMul p (HSub.hSub 1 Polynomial.X)).natDegree → Eq …
  -/
  intro n hn
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d n : Nat
    hn : GT.gt n (HMul.hMul p (HSub.hSub 1 Polynomial.X)).natDegree
    ⊢ Eq ((PowerSeries.coeff F n) (HMul.hMul ↑p ↑(PowerSeries.invOneSubPow F d)))  …
  -/
  have heq : 1 - PowerSeries.X = ((1 - X : F[X]) : F⟦X⟧) := by simp only [coe_sub, coe_one, coe_X]
  rw [← one_sub_pow_mul_invOneSubPow_val_add_eq_invOneSubPow_val F d 1, pow_one, ← mul_assoc, heq,
    ← coe_mul, coeff_mul_invOneSubPow_eq_hilbertPoly_eval (d + 1) hn]


lemma hilbertPoly_mul_one_sub_pow_add (p : F[X]) (d e : ℕ) :
    hilbertPoly (p * (1 - X) ^ e) (d + e) = hilbertPoly p d := by
  induction e with
  | zero => simp
  | succ e he => rw [pow_add, pow_one, ← mul_assoc, ← add_assoc, hilbertPoly_mul_one_sub_succ, he]


lemma hilbertPoly_eq_zero_of_le_rootMultiplicity_one
    {p : F[X]} {d : ℕ} (hdp : d ≤ p.rootMultiplicity 1) :
    hilbertPoly p d = 0 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    hdp : LE.le d (Polynomial.rootMultiplicity 1 p)
    ⊢ Eq (p.hilbertPoly d) 0
  -/
  by_cases hp : p = 0
    /-
      case pos
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hdp : LE.le d (Polynomial.rootMultiplicity 1 p)
      hp : Eq p 0
      ⊢ Eq (p.hilbertPoly d) 0
    -/
  · rw [hp, hilbertPoly_zero_left]
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hdp : LE.le d (Polynomial.rootMultiplicity 1 p)
      hp : Not (Eq p 0)
      ⊢ Eq (p.hilbertPoly d) 0
    -/
  · rcases exists_eq_pow_rootMultiplicity_mul_and_not_dvd p hp 1 with ⟨q, hq1, hq2⟩
    have heq : p = q * (- 1) ^ p.rootMultiplicity 1 * (1 - X) ^ p.rootMultiplicity 1 := by
      simp only [mul_assoc, ← mul_pow, neg_mul, one_mul, neg_sub]
      exact hq1.trans (mul_comm _ _)
    rw [heq, ← zero_add d, ← Nat.sub_add_cancel hdp, pow_add (1 - X), ← mul_assoc,
      hilbertPoly_mul_one_sub_pow_add, hilbertPoly]


theorem natDegree_hilbertPoly_of_ne_zero_of_rootMultiplicity_lt
    {p : F[X]} {d : ℕ} (hp : p ≠ 0) (hpd : p.rootMultiplicity 1 < d) :
    (hilbertPoly p d).natDegree = d - p.rootMultiplicity 1 - 1 := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    hp : Ne p 0
    hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
    ⊢ Eq (p.hilbertPoly d).natDegree (HSub.hSub (HSub.hSub d (Polynomial.rootMulti …
  -/
  rcases exists_eq_pow_rootMultiplicity_mul_and_not_dvd p hp 1 with ⟨q, hq1, hq2⟩
  have heq : p = q * (- 1) ^ p.rootMultiplicity 1 * (1 - X) ^ p.rootMultiplicity 1 := by
    simp only [mul_assoc, ← mul_pow, neg_mul, one_mul, neg_sub]
    exact hq1.trans (mul_comm _ _)
  nth_rw 1 [heq, ← Nat.sub_add_cancel (le_of_lt hpd), hilbertPoly_mul_one_sub_pow_add,
    ← Nat.sub_add_cancel (Nat.le_sub_of_add_le' <| add_one_le_of_lt hpd)]
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    hp : Ne p 0
    hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
    q : Polynomial F
    hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
    hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
    heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
    ⊢ Eq ((HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicity 1 p))).hilbert …
  -/
  delta hilbertPoly
  /-
    case intro.intro
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    hp : Ne p 0
    hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
    q : Polynomial F
    hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
    hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
    heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
    ⊢ Eq (Polynomial.hilbertPoly.match_1 (fun x => Polynomial F) (HAdd.hAdd (HSub. …
  -/
  apply natDegree_eq_of_le_of_coeff_ne_zero
    /-
      case intro.intro.pn
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hp : Ne p 0
      hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
      q : Polynomial F
      hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
      hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
      heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
      ⊢ LE.le (Polynomial.hilbertPoly.match_1 (fun x => Polynomial F) (HAdd.hAdd (HS …
    -/
  · apply natDegree_sum_le_of_forall_le _ _ <| fun _ _ => ?_
    /-
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hp : Ne p 0
      hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
      q : Polynomial F
      hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
      hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
      heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
      x✝¹ : Nat
      x✝ : Membership.mem (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicity  …
      ⊢ LE.le (HSMul.hSMul ((HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
    -/
    apply le_trans (natDegree_smul_le _ _)
    /-
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hp : Ne p 0
      hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
      q : Polynomial F
      hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
      hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
      heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
      x✝¹ : Nat
      x✝ : Membership.mem (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicity  …
      ⊢ LE.le (Polynomial.preHilbertPoly F (HSub.hSub (HSub.hSub d (Polynomial.rootM …
    -/
    rw [natDegree_preHilbertPoly]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.p1
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hp : Ne p 0
      hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
      q : Polynomial F
      hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
      hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
      heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
      ⊢ Ne ((Polynomial.hilbertPoly.match_1 (fun x => Polynomial F) (HAdd.hAdd (HSub …
    -/
  · have : (fun (x : ℕ) (a : F) => a) = fun x a => a * 1 ^ x := by simp only [one_pow, mul_one]
    simp only [finset_sum_coeff, coeff_smul, smul_eq_mul, coeff_preHilbertPoly_self,
      ← Finset.sum_mul, ← sum_def _ (fun _ a => a), this, ← eval_eq_sum, eval_mul, eval_pow,
      eval_neg, eval_one, _root_.mul_eq_zero, pow_eq_zero_iff', neg_eq_zero, one_ne_zero, ne_eq,
      false_and, or_false, inv_eq_zero, cast_eq_zero, not_or]
    /-
      case intro.intro.p1
      F : Type u_1
      inst✝¹ : Field F
      inst✝ : CharZero F
      p : Polynomial F
      d : Nat
      hp : Ne p 0
      hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
      q : Polynomial F
      hq1 : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C 1)) (Po …
      hq2 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 1)) q)
      heq : Eq p (HMul.hMul (HMul.hMul q (HPow.hPow (-1) (Polynomial.rootMultiplicit …
      this : Eq (fun x a => a) fun x a => HMul.hMul a (HPow.hPow 1 x)
      ⊢ And (Not (Eq (Polynomial.eval 1 q) 0)) (Not (Eq (HSub.hSub (HSub.hSub d (Pol …
    -/
    exact ⟨(not_iff_not.2 dvd_iff_isRoot).1 hq2, factorial_ne_zero _⟩
    /-
      🎉 no goals
    -/


theorem natDegree_hilbertPoly_of_ne_zero
    {p : F[X]} {d : ℕ} (hh : hilbertPoly p d ≠ 0) :
    (hilbertPoly p d).natDegree = d - p.rootMultiplicity 1 - 1 := by
  have hp : p ≠ 0 := by
    intro h
    rw [h] at hh
    exact hh (hilbertPoly_zero_left d)
  have hpd : p.rootMultiplicity 1 < d := by
    by_contra h
    exact hh (hilbertPoly_eq_zero_of_le_rootMultiplicity_one <| not_lt.1 h)
  /-
    F : Type u_1
    inst✝¹ : Field F
    inst✝ : CharZero F
    p : Polynomial F
    d : Nat
    hh : Ne (p.hilbertPoly d) 0
    hp : Ne p 0
    hpd : LT.lt (Polynomial.rootMultiplicity 1 p) d
    ⊢ Eq (p.hilbertPoly d).natDegree (HSub.hSub (HSub.hSub d (Polynomial.rootMulti …
  -/
  exact natDegree_hilbertPoly_of_ne_zero_of_rootMultiplicity_lt hp hpd
  /-
    🎉 no goals
  -/


