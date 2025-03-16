/-- Suppose `x : B`, where `B` is an `A`-algebra.

The minimal polynomial `minpoly A x` of `x`
is a monic polynomial with coefficients in `A` of smallest degree that has `x` as its root,
if such exists (`IsIntegral A x`) or zero otherwise.

For example, if `V` is a `𝕜`-vector space for some field `𝕜` and `f : V →ₗ[𝕜] V` then
the minimal polynomial of `f` is `minpoly 𝕜 f`.
-/
@[stacks 09GM]
noncomputable def minpoly (x : B) : A[X] :=
  if hx : IsIntegral A x then degree_lt_wf.min _ hx else 0


/-- A minimal polynomial is monic. -/
theorem monic (hx : IsIntegral A x) : Monic (minpoly A x) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    ⊢ (minpoly A x).Monic
  -/
  delta minpoly
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    ⊢ (dite (IsIntegral A x) (fun hx => ⋯.min (fun x_1 => And x_1.Monic (Eq (Polyn …
  -/
  rw [dif_pos hx]
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    ⊢ (⋯.min (fun x_1 => And x_1.Monic (Eq (Polynomial.eval₂ (algebraMap A B) x x_ …
  -/
  exact (degree_lt_wf.min_mem _ hx).1
  /-
    🎉 no goals
  -/


/-- A minimal polynomial is nonzero. -/
theorem ne_zero [Nontrivial A] (hx : IsIntegral A x) : minpoly A x ≠ 0 :=
  (monic hx).ne_zero


theorem eq_zero (hx : ¬IsIntegral A x) : minpoly A x = 0 :=
  dif_neg hx


theorem ne_zero_iff [Nontrivial A] : minpoly A x ≠ 0 ↔ IsIntegral A x :=
  ⟨fun h => of_not_not <| eq_zero.mt h, ne_zero⟩


theorem algHom_eq (f : B →ₐ[A] B') (hf : Function.Injective f) (x : B) :
    minpoly A (f x) = minpoly A x := by
  simp_rw [minpoly, isIntegral_algHom_iff _ hf, ← Polynomial.aeval_def, aeval_algHom,
    AlgHom.comp_apply, _root_.map_eq_zero_iff f hf]


theorem algebraMap_eq {B} [CommRing B] [Algebra A B] [Algebra B B'] [IsScalarTower A B B']
    (h : Function.Injective (algebraMap B B')) (x : B) :
    minpoly A (algebraMap B B' x) = minpoly A x :=
  algHom_eq (IsScalarTower.toAlgHom A B B') h x


@[simp]
theorem algEquiv_eq (f : B ≃ₐ[A] B') (x : B) : minpoly A (f x) = minpoly A x :=
  algHom_eq (f : B →ₐ[A] B') f.injective x


/-- An element is a root of its minimal polynomial. -/
@[simp]
theorem aeval : aeval x (minpoly A x) = 0 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    ⊢ Eq ((Polynomial.aeval x) (minpoly A x)) 0
  -/
  delta minpoly
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    ⊢ Eq ((Polynomial.aeval x) (dite (IsIntegral A x) (fun hx => ⋯.min (fun x_1 => …
  -/
  split_ifs with hx
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝² : CommRing A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      hx : IsIntegral A x
      ⊢ Eq ((Polynomial.aeval x) (⋯.min (fun x_1 => And x_1.Monic (Eq (Polynomial.ev …
    -/
  · exact (degree_lt_wf.min_mem _ hx).2
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝² : CommRing A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      hx : Not (IsIntegral A x)
      ⊢ Eq ((Polynomial.aeval x) 0) 0
    -/
  · exact aeval_zero _
    /-
      🎉 no goals
    -/


/-- Given any `f : B →ₐ[A] B'` and any `x : L`, the minimal polynomial of `x` vanishes at `f x`. -/
@[simp]
theorem aeval_algHom (f : B →ₐ[A] B') (x : B) : (Polynomial.aeval (f x)) (minpoly A x) = 0 := by
  /-
    A : Type u_1
    B : Type u_2
    B' : Type u_3
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Ring B'
    inst✝¹ : Algebra A B
    inst✝ : Algebra A B'
    f : AlgHom A B B'
    x : B
    ⊢ Eq ((Polynomial.aeval (f x)) (minpoly A x)) 0
  -/
  rw [Polynomial.aeval_algHom, AlgHom.coe_comp, comp_apply, aeval, map_zero]
  /-
    🎉 no goals
  -/


/-- A minimal polynomial is not `1`. -/
theorem ne_one [Nontrivial B] : minpoly A x ≠ 1 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    ⊢ Ne (minpoly A x) 1
  -/
  intro h
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    h : Eq (minpoly A x) 1
    ⊢ False
  -/
  refine (one_ne_zero : (1 : B) ≠ 0) ?_
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    h : Eq (minpoly A x) 1
    ⊢ Eq 1 0
  -/
  simpa using congr_arg (Polynomial.aeval x) h
  /-
    🎉 no goals
  -/


theorem map_ne_one [Nontrivial B] {R : Type*} [Semiring R] [Nontrivial R] (f : A →+* R) :
    (minpoly A x).map f ≠ 1 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁵ : CommRing A
    inst✝⁴ : Ring B
    inst✝³ : Algebra A B
    x : B
    inst✝² : Nontrivial B
    R : Type u_4
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    f : RingHom A R
    ⊢ Ne (Polynomial.map f (minpoly A x)) 1
  -/
  by_cases hx : IsIntegral A x
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : Ring B
      inst✝³ : Algebra A B
      x : B
      inst✝² : Nontrivial B
      R : Type u_4
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      f : RingHom A R
      hx : IsIntegral A x
      ⊢ Ne (Polynomial.map f (minpoly A x)) 1
    -/
  · exact mt ((monic hx).eq_one_of_map_eq_one f) (ne_one A x)
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : Ring B
      inst✝³ : Algebra A B
      x : B
      inst✝² : Nontrivial B
      R : Type u_4
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      f : RingHom A R
      hx : Not (IsIntegral A x)
      ⊢ Ne (Polynomial.map f (minpoly A x)) 1
    -/
  · rw [eq_zero hx, Polynomial.map_zero]
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝⁵ : CommRing A
      inst✝⁴ : Ring B
      inst✝³ : Algebra A B
      x : B
      inst✝² : Nontrivial B
      R : Type u_4
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      f : RingHom A R
      hx : Not (IsIntegral A x)
      ⊢ Ne 0 1
    -/
    exact zero_ne_one
    /-
      🎉 no goals
    -/


/-- A minimal polynomial is not a unit. -/
theorem not_isUnit [Nontrivial B] : ¬IsUnit (minpoly A x) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    ⊢ Not (IsUnit (minpoly A x))
  -/
  haveI : Nontrivial A := (algebraMap A B).domain_nontrivial
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    this : Nontrivial A
    ⊢ Not (IsUnit (minpoly A x))
  -/
  by_cases hx : IsIntegral A x
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : Ring B
      inst✝¹ : Algebra A B
      x : B
      inst✝ : Nontrivial B
      this : Nontrivial A
      hx : IsIntegral A x
      ⊢ Not (IsUnit (minpoly A x))
    -/
  · exact mt (monic hx).eq_one_of_isUnit (ne_one A x)
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : Ring B
      inst✝¹ : Algebra A B
      x : B
      inst✝ : Nontrivial B
      this : Nontrivial A
      hx : Not (IsIntegral A x)
      ⊢ Not (IsUnit (minpoly A x))
    -/
  · rw [eq_zero hx]
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : Ring B
      inst✝¹ : Algebra A B
      x : B
      inst✝ : Nontrivial B
      this : Nontrivial A
      hx : Not (IsIntegral A x)
      ⊢ Not (IsUnit 0)
    -/
    exact not_isUnit_zero
    /-
      🎉 no goals
    -/


theorem mem_range_of_degree_eq_one (hx : (minpoly A x).degree = 1) :
    x ∈ (algebraMap A B).range := by
  have h : IsIntegral A x := by
    by_contra h
    rw [eq_zero h, degree_zero, ← WithBot.coe_one] at hx
    exact ne_of_lt (show ⊥ < ↑1 from WithBot.bot_lt_coe 1) hx
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    hx : Eq (minpoly A x).degree 1
    h : IsIntegral A x
    ⊢ Membership.mem (algebraMap A B).range x
  -/
  have key := minpoly.aeval A x
  rw [eq_X_add_C_of_degree_eq_one hx, (minpoly.monic h).leadingCoeff, C_1, one_mul, aeval_add,
    aeval_C, aeval_X, ← eq_neg_iff_add_eq_zero, ← RingHom.map_neg] at key
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    hx : Eq (minpoly A x).degree 1
    h : IsIntegral A x
    key : Eq x ((algebraMap A B) (Neg.neg ((minpoly A x).coeff 0)))
    ⊢ Membership.mem (algebraMap A B).range x
  -/
  exact ⟨-(minpoly A x).coeff 0, key.symm⟩
  /-
    🎉 no goals
  -/


/-- The defining property of the minimal polynomial of an element `x`:
it is the monic polynomial with smallest degree that has `x` as its root. -/
theorem min {p : A[X]} (pmonic : p.Monic) (hp : Polynomial.aeval x p = 0) :
    degree (minpoly A x) ≤ degree p := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    ⊢ LE.le (minpoly A x).degree p.degree
  -/
  delta minpoly; split_ifs with hx
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝² : CommRing A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      p : Polynomial A
      pmonic : p.Monic
      hp : Eq ((Polynomial.aeval x) p) 0
      hx : IsIntegral A x
      ⊢ LE.le (⋯.min (fun x_1 => And x_1.Monic (Eq (Polynomial.eval₂ (algebraMap A B …
    -/
  · exact le_of_not_lt (degree_lt_wf.not_lt_min _ hx ⟨pmonic, hp⟩)
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      B : Type u_2
      inst✝² : CommRing A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      p : Polynomial A
      pmonic : p.Monic
      hp : Eq ((Polynomial.aeval x) p) 0
      hx : Not (IsIntegral A x)
      ⊢ LE.le (Polynomial.degree 0) p.degree
    -/
  · simp only [degree_zero, bot_le]
    /-
      🎉 no goals
    -/


theorem unique' {p : A[X]} (hm : p.Monic) (hp : Polynomial.aeval x p = 0)
    (hl : ∀ q : A[X], degree q < degree p → q = 0 ∨ Polynomial.aeval x q ≠ 0) :
    p = minpoly A x := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    ⊢ Eq p (minpoly A x)
  -/
  nontriviality A
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    ⊢ Eq p (minpoly A x)
  -/
  have hx : IsIntegral A x := ⟨p, hm, hp⟩
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    hx : IsIntegral A x
    ⊢ Eq p (minpoly A x)
  -/
  obtain h | h := hl _ ((minpoly A x).degree_modByMonic_lt hm)
  /-
    case inl
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    hx : IsIntegral A x
    h : Eq ((minpoly A x).modByMonic p) 0
    ⊢ Eq p (minpoly A x)
  -/
  swap
    /-
      case inr
      A : Type u_1
      B : Type u_2
      inst✝² : CommRing A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      p : Polynomial A
      hm : p.Monic
      hp : Eq ((Polynomial.aeval x) p) 0
      hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
      a✝ : Nontrivial A
      hx : IsIntegral A x
      h : Ne ((Polynomial.aeval x) ((minpoly A x).modByMonic p)) 0
      ⊢ Eq p (minpoly A x)
    -/
  · exact (h <| (aeval_modByMonic_eq_self_of_root hm hp).trans <| aeval A x).elim
    /-
      🎉 no goals
    -/
  /-
    case inl
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    hx : IsIntegral A x
    h : Eq ((minpoly A x).modByMonic p) 0
    ⊢ Eq p (minpoly A x)
  -/
  obtain ⟨r, hr⟩ := (modByMonic_eq_zero_iff_dvd hm).1 h
  /-
    case inl.intro
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    hx : IsIntegral A x
    h : Eq ((minpoly A x).modByMonic p) 0
    r : Polynomial A
    hr : Eq (minpoly A x) (HMul.hMul p r)
    ⊢ Eq p (minpoly A x)
  -/
  rw [hr]
  /-
    case inl.intro
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    hx : IsIntegral A x
    h : Eq ((minpoly A x).modByMonic p) 0
    r : Polynomial A
    hr : Eq (minpoly A x) (HMul.hMul p r)
    ⊢ Eq p (HMul.hMul p r)
  -/
  have hlead := congr_arg leadingCoeff hr
  /-
    case inl.intro
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hm : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    hl : ∀ (q : Polynomial A), LT.lt q.degree p.degree → Or (Eq q 0) (Ne ((Polynom …
    a✝ : Nontrivial A
    hx : IsIntegral A x
    h : Eq ((minpoly A x).modByMonic p) 0
    r : Polynomial A
    hr : Eq (minpoly A x) (HMul.hMul p r)
    hlead : Eq (minpoly A x).leadingCoeff (HMul.hMul p r).leadingCoeff
    ⊢ Eq p (HMul.hMul p r)
  -/
  rw [mul_comm, leadingCoeff_mul_monic hm, (monic hx).leadingCoeff] at hlead
  have : natDegree r ≤ 0 := by
    have hr0 : r ≠ 0 := by
      rintro rfl
      exact ne_zero hx (mul_zero p ▸ hr)
    apply_fun natDegree at hr
    rw [hm.natDegree_mul' hr0] at hr
    apply Nat.le_of_add_le_add_left
    rw [add_zero]
    exact hr.symm.trans_le (natDegree_le_natDegree <| min A x hm hp)
  rw [eq_C_of_natDegree_le_zero this, ← Nat.eq_zero_of_le_zero this, ← leadingCoeff, ← hlead, C_1,
    mul_one]


@[nontriviality]
theorem subsingleton [Subsingleton B] : minpoly A x = 1 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Subsingleton B
    ⊢ Eq (minpoly A x) 1
  -/
  nontriviality A
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Subsingleton B
    a✝ : Nontrivial A
    ⊢ Eq (minpoly A x) 1
  -/
  have := minpoly.min A x monic_one (Subsingleton.elim _ _)
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Subsingleton B
    a✝ : Nontrivial A
    this : LE.le (minpoly A x).degree (Polynomial.degree 1)
    ⊢ Eq (minpoly A x) 1
  -/
  rw [degree_one] at this
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Subsingleton B
    a✝ : Nontrivial A
    this : LE.le (minpoly A x).degree 0
    ⊢ Eq (minpoly A x) 1
  -/
  rcases le_or_lt (minpoly A x).degree 0 with h | h
  · rwa [(monic ⟨1, monic_one, by simp [eq_iff_true_of_subsingleton]⟩ :
           (minpoly A x).Monic).degree_le_zero_iff_eq_one] at h
    /-
      case inr
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : Ring B
      inst✝¹ : Algebra A B
      x : B
      inst✝ : Subsingleton B
      a✝ : Nontrivial A
      this : LE.le (minpoly A x).degree 0
      h : LT.lt 0 (minpoly A x).degree
      ⊢ Eq (minpoly A x) 1
    -/
  · exact (this.not_lt h).elim
    /-
      🎉 no goals
    -/


/-- The degree of a minimal polynomial, as a natural number, is positive. -/
theorem natDegree_pos [Nontrivial B] (hx : IsIntegral A x) : 0 < natDegree (minpoly A x) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    hx : IsIntegral A x
    ⊢ LT.lt 0 (minpoly A x).natDegree
  -/
  rw [pos_iff_ne_zero]
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    hx : IsIntegral A x
    ⊢ Ne (minpoly A x).natDegree 0
  -/
  intro ndeg_eq_zero
  have eq_one : minpoly A x = 1 := by
    rw [eq_C_of_natDegree_eq_zero ndeg_eq_zero]
    convert C_1 (R := A)
    simpa only [ndeg_eq_zero.symm] using (monic hx).leadingCoeff
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    hx : IsIntegral A x
    ndeg_eq_zero : Eq (minpoly A x).natDegree 0
    eq_one : Eq (minpoly A x) 1
    ⊢ False
  -/
  simpa only [eq_one, map_one, one_ne_zero] using aeval A x
  /-
    🎉 no goals
  -/


/-- The degree of a minimal polynomial is positive. -/
theorem degree_pos [Nontrivial B] (hx : IsIntegral A x) : 0 < degree (minpoly A x) :=
  natDegree_pos_iff_degree_pos.mp (natDegree_pos hx)


open Polynomial in
theorem degree_eq_one_iff : (minpoly A x).degree = 1 ↔ x ∈ (algebraMap A B).range := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    ⊢ Iff (Eq (minpoly A x).degree 1) (Membership.mem (algebraMap A B).range x)
  -/
  refine ⟨minpoly.mem_range_of_degree_eq_one _ _, ?_⟩
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    ⊢ Membership.mem (algebraMap A B).range x → Eq (minpoly A x).degree 1
  -/
  rintro ⟨x, rfl⟩
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    inst✝ : Nontrivial B
    x : A
    ⊢ Eq (minpoly A ((algebraMap A B) x)).degree 1
  -/
  haveI := Module.nontrivial A B
  exact (degree_X_sub_C x ▸ minpoly.min A (algebraMap A B x) (monic_X_sub_C x) (by simp)).antisymm
    (Nat.WithBot.add_one_le_of_lt <| minpoly.degree_pos isIntegral_algebraMap)


theorem natDegree_eq_one_iff :
    (minpoly A x).natDegree = 1 ↔ x ∈ (algebraMap A B).range := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    ⊢ Iff (Eq (minpoly A x).natDegree 1) (Membership.mem (algebraMap A B).range x)
  -/
  rw [← Polynomial.degree_eq_iff_natDegree_eq_of_pos zero_lt_one]
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    ⊢ Iff (Eq (minpoly A x).degree ↑1) (Membership.mem (algebraMap A B).range x)
  -/
  exact degree_eq_one_iff
  /-
    🎉 no goals
  -/


theorem two_le_natDegree_iff (int : IsIntegral A x) :
    2 ≤ (minpoly A x).natDegree ↔ x ∉ (algebraMap A B).range := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    int : IsIntegral A x
    ⊢ Iff (LE.le 2 (minpoly A x).natDegree) (Not (Membership.mem (algebraMap A B). …
  -/
  rw [iff_not_comm, ← natDegree_eq_one_iff, not_le]
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    int : IsIntegral A x
    ⊢ Iff (Eq (minpoly A x).natDegree 1) (LT.lt (minpoly A x).natDegree 2)
  -/
  exact ⟨fun h ↦ h.trans_lt one_lt_two, fun h ↦ by linarith only [minpoly.natDegree_pos int, h]⟩
  /-
    🎉 no goals
  -/


theorem two_le_natDegree_subalgebra {B} [CommRing B] [Algebra A B] [Nontrivial B]
    {S : Subalgebra A B} {x : B} (int : IsIntegral S x) : 2 ≤ (minpoly S x).natDegree ↔ x ∉ S := by
  /-
    A : Type u_1
    inst✝³ : CommRing A
    B : Type u_4
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Nontrivial B
    S : Subalgebra A B
    x : B
    int : IsIntegral (Subtype fun x => Membership.mem S x) x
    ⊢ Iff (LE.le 2 (minpoly (Subtype fun x => Membership.mem S x) x).natDegree) (N …
  -/
  rw [two_le_natDegree_iff int, Iff.not]
  /-
    A : Type u_1
    inst✝³ : CommRing A
    B : Type u_4
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Nontrivial B
    S : Subalgebra A B
    x : B
    int : IsIntegral (Subtype fun x => Membership.mem S x) x
    ⊢ Iff (Membership.mem (algebraMap (Subtype fun x => Membership.mem S x) B).ran …
  -/
  apply Set.ext_iff.mp Subtype.range_val_subtype
  /-
    🎉 no goals
  -/


/-- If `B/A` is an injective ring extension, and `a` is an element of `A`,
then the minimal polynomial of `algebraMap A B a` is `X - C a`. -/
theorem eq_X_sub_C_of_algebraMap_inj (a : A) (hf : Function.Injective (algebraMap A B)) :
    minpoly A (algebraMap A B a) = X - C a := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    ⊢ Eq (minpoly A ((algebraMap A B) a)) (HSub.hSub Polynomial.X (Polynomial.C a))
  -/
  nontriviality A
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    a✝ : Nontrivial A
    ⊢ Eq (minpoly A ((algebraMap A B) a)) (HSub.hSub Polynomial.X (Polynomial.C a))
  -/
  refine (unique' A _ (monic_X_sub_C a) ?_ ?_).symm
    /-
      case refine_1
      A : Type u_1
      B : Type u_2
      inst✝² : CommRing A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      a : A
      hf : Function.Injective ⇑(algebraMap A B)
      a✝ : Nontrivial A
      ⊢ Eq ((Polynomial.aeval ((algebraMap A B) a)) (HSub.hSub Polynomial.X (Polynom …
    -/
  · rw [map_sub, aeval_C, aeval_X, sub_self]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    a✝ : Nontrivial A
    ⊢ ∀ (q : Polynomial A), LT.lt q.degree (HSub.hSub Polynomial.X (Polynomial.C a …
  -/
  simp_rw [or_iff_not_imp_left]
  /-
    case refine_2
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    a✝ : Nontrivial A
    ⊢ ∀ (q : Polynomial A), LT.lt q.degree (HSub.hSub Polynomial.X (Polynomial.C a …
  -/
  intro q hl h0
  /-
    case refine_2
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    a✝ : Nontrivial A
    q : Polynomial A
    hl : LT.lt q.degree (HSub.hSub Polynomial.X (Polynomial.C a)).degree
    h0 : Not (Eq q 0)
    ⊢ Ne ((Polynomial.aeval ((algebraMap A B) a)) q) 0
  -/
  rw [← natDegree_lt_natDegree_iff h0, natDegree_X_sub_C, Nat.lt_one_iff] at hl
  /-
    case refine_2
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    a✝ : Nontrivial A
    q : Polynomial A
    hl : Eq q.natDegree 0
    h0 : Not (Eq q 0)
    ⊢ Ne ((Polynomial.aeval ((algebraMap A B) a)) q) 0
  -/
  rw [eq_C_of_natDegree_eq_zero hl] at h0 ⊢
  /-
    case refine_2
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    a : A
    hf : Function.Injective ⇑(algebraMap A B)
    a✝ : Nontrivial A
    q : Polynomial A
    hl : Eq q.natDegree 0
    h0 : Not (Eq (Polynomial.C (q.coeff 0)) 0)
    ⊢ Ne ((Polynomial.aeval ((algebraMap A B) a)) (Polynomial.C (q.coeff 0))) 0
  -/
  rwa [aeval_C, map_ne_zero_iff _ hf, ← C_ne_zero]
  /-
    🎉 no goals
  -/


/-- If `a` strictly divides the minimal polynomial of `x`, then `x` cannot be a root for `a`. -/
theorem aeval_ne_zero_of_dvdNotUnit_minpoly {a : A[X]} (hx : IsIntegral A x) (hamonic : a.Monic)
    (hdvd : DvdNotUnit a (minpoly A x)) : Polynomial.aeval x a ≠ 0 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    hdvd : DvdNotUnit a (minpoly A x)
    ⊢ Ne ((Polynomial.aeval x) a) 0
  -/
  refine fun ha => (min A x hamonic ha).not_lt (degree_lt_degree ?_)
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    hdvd : DvdNotUnit a (minpoly A x)
    ha : Eq ((Polynomial.aeval x) a) 0
    ⊢ LT.lt a.natDegree (minpoly A x).natDegree
  -/
  obtain ⟨_, c, hu, he⟩ := hdvd
  /-
    case intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    ha : Eq ((Polynomial.aeval x) a) 0
    left✝ : Ne a 0
    c : Polynomial A
    hu : Not (IsUnit c)
    he : Eq (minpoly A x) (HMul.hMul a c)
    ⊢ LT.lt a.natDegree (minpoly A x).natDegree
  -/
  have hcm := hamonic.of_mul_monic_left (he.subst <| monic hx)
  /-
    case intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    ha : Eq ((Polynomial.aeval x) a) 0
    left✝ : Ne a 0
    c : Polynomial A
    hu : Not (IsUnit c)
    he : Eq (minpoly A x) (HMul.hMul a c)
    hcm : c.Monic
    ⊢ LT.lt a.natDegree (minpoly A x).natDegree
  -/
  rw [he, hamonic.natDegree_mul hcm]
  -- TODO: port Nat.lt_add_of_zero_lt_left from lean3 core
  /-
    case intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    ha : Eq ((Polynomial.aeval x) a) 0
    left✝ : Ne a 0
    c : Polynomial A
    hu : Not (IsUnit c)
    he : Eq (minpoly A x) (HMul.hMul a c)
    hcm : c.Monic
    ⊢ LT.lt a.natDegree (HAdd.hAdd a.natDegree c.natDegree)
  -/
  apply lt_add_of_pos_right
  /-
    case intro.intro.intro.h
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    ha : Eq ((Polynomial.aeval x) a) 0
    left✝ : Ne a 0
    c : Polynomial A
    hu : Not (IsUnit c)
    he : Eq (minpoly A x) (HMul.hMul a c)
    hcm : c.Monic
    ⊢ LT.lt 0 c.natDegree
  -/
  refine (lt_of_not_le fun h => hu ?_)
  rw [eq_C_of_natDegree_le_zero h, ← Nat.eq_zero_of_le_zero h, ← leadingCoeff, hcm.leadingCoeff,
    C_1]
  /-
    case intro.intro.intro.h
    A : Type u_1
    B : Type u_2
    inst✝² : CommRing A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    a : Polynomial A
    hx : IsIntegral A x
    hamonic : a.Monic
    ha : Eq ((Polynomial.aeval x) a) 0
    left✝ : Ne a 0
    c : Polynomial A
    hu : Not (IsUnit c)
    he : Eq (minpoly A x) (HMul.hMul a c)
    hcm : c.Monic
    h : LE.le c.natDegree 0
    ⊢ IsUnit 1
  -/
  exact isUnit_one
  /-
    🎉 no goals
  -/


/-- A minimal polynomial is irreducible. -/
theorem irreducible (hx : IsIntegral A x) : Irreducible (minpoly A x) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Algebra A B
    x : B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    hx : IsIntegral A x
    ⊢ Irreducible (minpoly A x)
  -/
  refine (irreducible_of_monic (monic hx) <| ne_one A x).2 fun f g hf hg he => ?_
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Algebra A B
    x : B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    hx : IsIntegral A x
    f g : Polynomial A
    hf : f.Monic
    hg : g.Monic
    he : Eq (HMul.hMul f g) (minpoly A x)
    ⊢ Or (Eq f 1) (Eq g 1)
  -/
  rw [← hf.isUnit_iff, ← hg.isUnit_iff]
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Algebra A B
    x : B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    hx : IsIntegral A x
    f g : Polynomial A
    hf : f.Monic
    hg : g.Monic
    he : Eq (HMul.hMul f g) (minpoly A x)
    ⊢ Or (IsUnit f) (IsUnit g)
  -/
  by_contra! h
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Algebra A B
    x : B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    hx : IsIntegral A x
    f g : Polynomial A
    hf : f.Monic
    hg : g.Monic
    he : Eq (HMul.hMul f g) (minpoly A x)
    h : And (Not (IsUnit f)) (Not (IsUnit g))
    ⊢ False
  -/
  have heval := congr_arg (Polynomial.aeval x) he
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Algebra A B
    x : B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    hx : IsIntegral A x
    f g : Polynomial A
    hf : f.Monic
    hg : g.Monic
    he : Eq (HMul.hMul f g) (minpoly A x)
    h : And (Not (IsUnit f)) (Not (IsUnit g))
    heval : Eq ((Polynomial.aeval x) (HMul.hMul f g)) ((Polynomial.aeval x) (minpo …
    ⊢ False
  -/
  rw [aeval A x, aeval_mul, mul_eq_zero] at heval
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Ring B
    inst✝² : Algebra A B
    x : B
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    hx : IsIntegral A x
    f g : Polynomial A
    hf : f.Monic
    hg : g.Monic
    he : Eq (HMul.hMul f g) (minpoly A x)
    h : And (Not (IsUnit f)) (Not (IsUnit g))
    heval : Or (Eq ((Polynomial.aeval x) f) 0) (Eq ((Polynomial.aeval x) g) 0)
    ⊢ False
  -/
  cases' heval with heval heval
    /-
      case inl
      A : Type u_1
      B : Type u_2
      inst✝⁴ : CommRing A
      inst✝³ : Ring B
      inst✝² : Algebra A B
      x : B
      inst✝¹ : IsDomain A
      inst✝ : IsDomain B
      hx : IsIntegral A x
      f g : Polynomial A
      hf : f.Monic
      hg : g.Monic
      he : Eq (HMul.hMul f g) (minpoly A x)
      h : And (Not (IsUnit f)) (Not (IsUnit g))
      heval : Eq ((Polynomial.aeval x) f) 0
      ⊢ False
    -/
  · exact aeval_ne_zero_of_dvdNotUnit_minpoly hx hf ⟨hf.ne_zero, g, h.2, he.symm⟩ heval
    /-
      🎉 no goals
    -/
    /-
      case inr
      A : Type u_1
      B : Type u_2
      inst✝⁴ : CommRing A
      inst✝³ : Ring B
      inst✝² : Algebra A B
      x : B
      inst✝¹ : IsDomain A
      inst✝ : IsDomain B
      hx : IsIntegral A x
      f g : Polynomial A
      hf : f.Monic
      hg : g.Monic
      he : Eq (HMul.hMul f g) (minpoly A x)
      h : And (Not (IsUnit f)) (Not (IsUnit g))
      heval : Eq ((Polynomial.aeval x) g) 0
      ⊢ False
    -/
  · refine aeval_ne_zero_of_dvdNotUnit_minpoly hx hg ⟨hg.ne_zero, f, h.1, ?_⟩ heval
    /-
      case inr
      A : Type u_1
      B : Type u_2
      inst✝⁴ : CommRing A
      inst✝³ : Ring B
      inst✝² : Algebra A B
      x : B
      inst✝¹ : IsDomain A
      inst✝ : IsDomain B
      hx : IsIntegral A x
      f g : Polynomial A
      hf : f.Monic
      hg : g.Monic
      he : Eq (HMul.hMul f g) (minpoly A x)
      h : And (Not (IsUnit f)) (Not (IsUnit g))
      heval : Eq ((Polynomial.aeval x) g) 0
      ⊢ Eq (minpoly A x) (HMul.hMul g f)
    -/
    rw [mul_comm, he]
    /-
      🎉 no goals
    -/


