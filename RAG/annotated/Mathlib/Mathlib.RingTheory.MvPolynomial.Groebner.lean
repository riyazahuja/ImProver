variable (m) in
/-- Delete the leading term in a multivariate polynomial (for some monomial order) -/
noncomputable def subLTerm (f : MvPolynomial σ R) : MvPolynomial σ R :=
  f - monomial (m.degree f) (m.lCoeff f)


theorem degree_sub_LTerm_le (f : MvPolynomial σ R) :
    m.degree (m.subLTerm f) ≼[m] m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree (m.subLTerm f))) (m.toSyn (m.degree f))
  -/
  apply le_trans degree_sub_le
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    ⊢ LE.le (Max.max (m.toSyn (m.degree f)) (m.toSyn (m.degree ((MvPolynomial.mono …
  -/
  simp only [sup_le_iff, le_refl, true_and]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    ⊢ LE.le (m.toSyn (m.degree ((MvPolynomial.monomial (m.degree f)) (m.lCoeff f)) …
  -/
  apply degree_monomial_le
  /-
    🎉 no goals
  -/


theorem degree_sub_LTerm_lt {f : MvPolynomial σ R} (hf : m.degree f ≠ 0) :
    m.degree (m.subLTerm f) ≺[m] m.degree f := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    hf : Ne (m.degree f) 0
    ⊢ LT.lt (m.toSyn (m.degree (m.subLTerm f))) (m.toSyn (m.degree f))
  -/
  rw [lt_iff_le_and_ne]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f : MvPolynomial σ R
    hf : Ne (m.degree f) 0
    ⊢ And (LE.le (m.toSyn (m.degree (m.subLTerm f))) (m.toSyn (m.degree f))) (Ne ( …
  -/
  refine ⟨degree_sub_LTerm_le f, ?_⟩
  classical
  intro hf'
  simp only [EmbeddingLike.apply_eq_iff_eq] at hf'
  have : m.subLTerm f ≠ 0 := by
    intro h
    simp only [h, degree_zero] at hf'
    exact hf hf'.symm
  rw [← coeff_degree_ne_zero_iff (m := m), hf'] at this
  apply this
  simp [subLTerm, coeff_monomial, lCoeff]


variable (m) in
/-- Reduce a polynomial modulo a polynomial with unit leading term (for some monomial order) -/
noncomputable def reduce {b : MvPolynomial σ R} (hb : IsUnit (m.lCoeff b)) (f : MvPolynomial σ R) :
    MvPolynomial σ R :=
 f - monomial (m.degree f - m.degree b) (hb.unit⁻¹ * m.lCoeff f) * b


theorem degree_reduce_lt {f b : MvPolynomial σ R} (hb : IsUnit (m.lCoeff b))
    (hbf : m.degree b ≤ m.degree f) (hf : m.degree f ≠ 0) :
    m.degree (m.reduce hb f) ≺[m] m.degree f := by
  have H : m.degree f =
    m.degree ((monomial (m.degree f - m.degree b)) (hb.unit⁻¹ * m.lCoeff f)) +
      m.degree b := by
    classical
    rw [degree_monomial, if_neg]
    · ext d
      rw [tsub_add_cancel_of_le hbf]
    · simp only [Units.mul_right_eq_zero, lCoeff_eq_zero_iff]
      intro hf0
      apply hf
      simp [hf0]
  have H' : coeff (m.degree f) (m.reduce hb f) = 0 := by
    simp only [reduce, coeff_sub, sub_eq_zero]
    nth_rewrite 2 [H]
    rw [coeff_mul_of_degree_add (m := m), lCoeff_monomial]
    rw [mul_comm, ← mul_assoc]
    simp only [IsUnit.mul_val_inv, one_mul]
    rfl
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f b : MvPolynomial σ R
    hb : IsUnit (m.lCoeff b)
    hbf : LE.le (m.degree b) (m.degree f)
    hf : Ne (m.degree f) 0
    H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
    H' : Eq (MvPolynomial.coeff (m.degree f) (m.reduce hb f)) 0
    ⊢ LT.lt (m.toSyn (m.degree (m.reduce hb f))) (m.toSyn (m.degree f))
  -/
  rw [lt_iff_le_and_ne]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    f b : MvPolynomial σ R
    hb : IsUnit (m.lCoeff b)
    hbf : LE.le (m.degree b) (m.degree f)
    hf : Ne (m.degree f) 0
    H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
    H' : Eq (MvPolynomial.coeff (m.degree f) (m.reduce hb f)) 0
    ⊢ And (LE.le (m.toSyn (m.degree (m.reduce hb f))) (m.toSyn (m.degree f))) (Ne  …
  -/
  constructor
  · classical
    apply le_trans degree_sub_le
    simp only [sup_le_iff, le_refl, true_and]
    apply le_of_le_of_eq degree_mul_le
    rw [m.toSyn.injective.eq_iff]
    exact H.symm
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      H' : Eq (MvPolynomial.coeff (m.degree f) (m.reduce hb f)) 0
      ⊢ Ne (m.toSyn (m.degree (m.reduce hb f))) (m.toSyn (m.degree f))
    -/
  · intro K
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      H' : Eq (MvPolynomial.coeff (m.degree f) (m.reduce hb f)) 0
      K : Eq (m.toSyn (m.degree (m.reduce hb f))) (m.toSyn (m.degree f))
      ⊢ False
    -/
    simp only [EmbeddingLike.apply_eq_iff_eq] at K
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      H' : Eq (MvPolynomial.coeff (m.degree f) (m.reduce hb f)) 0
      K : Eq (m.degree (m.reduce hb f)) (m.degree f)
      ⊢ False
    -/
    nth_rewrite 1 [← K] at H'
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      H' : Eq (MvPolynomial.coeff (m.degree (m.reduce hb f)) (m.reduce hb f)) 0
      K : Eq (m.degree (m.reduce hb f)) (m.degree f)
      ⊢ False
    -/
    change lCoeff m _ = 0 at H'
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      K : Eq (m.degree (m.reduce hb f)) (m.degree f)
      H' : Eq (m.lCoeff (m.reduce hb f)) 0
      ⊢ False
    -/
    rw [lCoeff_eq_zero_iff] at H'
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      K : Eq (m.degree (m.reduce hb f)) (m.degree f)
      H' : Eq (m.reduce hb f) 0
      ⊢ False
    -/
    rw [H', degree_zero] at K
    /-
      case right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      f b : MvPolynomial σ R
      hb : IsUnit (m.lCoeff b)
      hbf : LE.le (m.degree b) (m.degree f)
      hf : Ne (m.degree f) 0
      H : Eq (m.degree f) (HAdd.hAdd (m.degree ((MvPolynomial.monomial (HSub.hSub (m …
      K : Eq 0 (m.degree f)
      H' : Eq (m.reduce hb f) 0
      ⊢ False
    -/
    exact hf K.symm
    /-
      🎉 no goals
    -/


theorem div {ι : Type*} {b : ι → MvPolynomial σ R}
    (hb : ∀ i, IsUnit (m.lCoeff (b i))) (f : MvPolynomial σ R) :
    ∃ (g : ι →₀ (MvPolynomial σ R)) (r : MvPolynomial σ R),
      f = Finsupp.linearCombination _ b g + r ∧
        (∀ i, m.degree (b i * (g i)) ≼[m] m.degree f) ∧
        (∀ c ∈ r.support, ∀ i, ¬ (m.degree (b i) ≤ c)) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    ι : Type u_3
    b : ι → MvPolynomial σ R
    hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
    f : MvPolynomial σ R
    ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
  -/
  by_cases hb' : ∃ i, m.degree (b i) = 0
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : Exists fun i => Eq (m.degree (b i)) 0
      ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
    -/
  · obtain ⟨i, hb0⟩ := hb'
    /-
      case pos.intro
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      i : ι
      hb0 : Eq (m.degree (b i)) 0
      ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
    -/
    use Finsupp.single i ((hb i).unit⁻¹ • f), 0
    /-
      case h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      i : ι
      hb0 : Eq (m.degree (b i)) 0
      ⊢ And (Eq f (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomial σ R) b) (Fins …
    -/
    constructor
      /-
        case h.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        ⊢ Eq f (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomial σ R) b) (Finsupp.s …
      -/
    · simp only [Finsupp.linearCombination_single, smul_eq_mul, add_zero]
      /-
        case h.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        ⊢ Eq f (HMul.hMul (HSMul.hSMul (Inv.inv ⋯.unit) f) (b i))
      -/
      simp only [smul_mul_assoc, ← smul_eq_iff_eq_inv_smul, Units.smul_isUnit]
      /-
        case h.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        ⊢ Eq (HSMul.hSMul (m.lCoeff (b i)) f) (HMul.hMul f (b i))
      -/
      nth_rewrite 2 [eq_C_of_degree_eq_zero hb0]
      /-
        case h.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        ⊢ Eq (HSMul.hSMul (m.lCoeff (b i)) f) (HMul.hMul f (MvPolynomial.C (m.lCoeff ( …
      -/
      rw [mul_comm, smul_eq_C_mul]
      /-
        🎉 no goals
      -/
    /-
      case h.right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      i : ι
      hb0 : Eq (m.degree (b i)) 0
      ⊢ And (∀ (i_1 : ι), LE.le (m.toSyn (m.degree (HMul.hMul (b i_1) ((Finsupp.sing …
    -/
    constructor
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        ⊢ ∀ (i_1 : ι), LE.le (m.toSyn (m.degree (HMul.hMul (b i_1) ((Finsupp.single i  …
      -/
    · intro j
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        j : ι
        ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b j) ((Finsupp.single i (HSMul.hSMul (I …
      -/
      by_cases hj : j = i
        /-
          case pos
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Eq j i
          ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b j) ((Finsupp.single i (HSMul.hSMul (I …
        -/
      · apply le_trans degree_mul_le
        /-
          case pos
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Eq j i
          ⊢ LE.le (m.toSyn (HAdd.hAdd (m.degree (b j)) (m.degree ((Finsupp.single i (HSM …
        -/
        simp only [hj, hb0, Finsupp.single_eq_same, zero_add]
        /-
          case pos
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Eq j i
          ⊢ LE.le (m.toSyn (m.degree (HSMul.hSMul (Inv.inv ⋯.unit) f))) (m.toSyn (m.degr …
        -/
        apply le_of_eq
        /-
          case pos.hab
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Eq j i
          ⊢ Eq (m.toSyn (m.degree (HSMul.hSMul (Inv.inv ⋯.unit) f))) (m.toSyn (m.degree  …
        -/
        simp only [EmbeddingLike.apply_eq_iff_eq]
        /-
          case pos.hab
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Eq j i
          ⊢ Eq (m.degree (HSMul.hSMul (Inv.inv ⋯.unit) f)) (m.degree f)
        -/
        apply degree_smul (Units.isRegular _)
        /-
          🎉 no goals
        -/
        /-
          case neg
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Not (Eq j i)
          ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b j) ((Finsupp.single i (HSMul.hSMul (I …
        -/
      · simp only [Finsupp.single_eq_of_ne (Ne.symm hj), mul_zero, degree_zero, map_zero]
        /-
          case neg
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          i : ι
          hb0 : Eq (m.degree (b i)) 0
          j : ι
          hj : Not (Eq j i)
          ⊢ LE.le 0 (m.toSyn (m.degree f))
        -/
        apply bot_le
        /-
          🎉 no goals
        -/
      /-
        case h.right.right
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        i : ι
        hb0 : Eq (m.degree (b i)) 0
        ⊢ ∀ (c : Finsupp σ Nat), Membership.mem (MvPolynomial.support 0) c → ∀ (i : ι) …
      -/
    · simp
      /-
        🎉 no goals
      -/
  /-
    case neg
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    ι : Type u_3
    b : ι → MvPolynomial σ R
    hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
    f : MvPolynomial σ R
    hb' : Not (Exists fun i => Eq (m.degree (b i)) 0)
    ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
  -/
  push_neg at hb'
  /-
    case neg
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    ι : Type u_3
    b : ι → MvPolynomial σ R
    hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
    f : MvPolynomial σ R
    hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
    ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
  -/
  by_cases hf0 : f = 0
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Eq f 0
      ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
    -/
  · refine ⟨0, 0, by simp [hf0], ?_, by simp⟩
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Eq f 0
      ⊢ ∀ (i : ι), LE.le (m.toSyn (m.degree (HMul.hMul (b i) (0 i)))) (m.toSyn (m.de …
    -/
    intro b
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b✝ : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b✝ i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b✝ i)) 0
      hf0 : Eq f 0
      b : ι
      ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b✝ b) (0 b)))) (m.toSyn (m.degree f))
    -/
    simp only [Finsupp.coe_zero, Pi.zero_apply, mul_zero, degree_zero, map_zero]
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b✝ : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b✝ i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b✝ i)) 0
      hf0 : Eq f 0
      b : ι
      ⊢ LE.le 0 (m.toSyn (m.degree f))
    -/
    exact bot_le
    /-
      🎉 no goals
    -/
  /-
    case neg
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    ι : Type u_3
    b : ι → MvPolynomial σ R
    hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
    f : MvPolynomial σ R
    hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
    hf0 : Not (Eq f 0)
    ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
  -/
  by_cases hf : ∃ i, m.degree (b i) ≤ m.degree f
    /-
      case pos
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      hf : Exists fun i => LE.le (m.degree (b i)) (m.degree f)
      ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
    -/
  · obtain ⟨i, hf⟩ := hf
    have deg_reduce : m.degree (m.reduce (hb i) f) ≺[m] m.degree f := by
      apply degree_reduce_lt (hb i) hf
      intro hf0'
      apply hb' i
      simpa [hf0'] using hf
    /-
      case pos.intro
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      i : ι
      hf : LE.le (m.degree (b i)) (m.degree f)
      deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
      ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
    -/
    obtain ⟨g', r', H'⟩ := div hb (m.reduce (hb i) f)
    use g' +
      Finsupp.single i (monomial (m.degree f - m.degree (b i)) ((hb i).unit⁻¹ * m.lCoeff f))
    /-
      case h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      i : ι
      hf : LE.le (m.degree (b i)) (m.degree f)
      deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
      g' : Finsupp ι (MvPolynomial σ R)
      r' : MvPolynomial σ R
      H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
      ⊢ Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
    -/
    use r'
    /-
      case h
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      i : ι
      hf : LE.le (m.degree (b i)) (m.degree f)
      deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
      g' : Finsupp ι (MvPolynomial σ R)
      r' : MvPolynomial σ R
      H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
      ⊢ And (Eq f (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomial σ R) b) (HAdd …
    -/
    constructor
      /-
        case h.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        ⊢ Eq f (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomial σ R) b) (HAdd.hAdd …
      -/
    · rw [map_add, add_assoc, add_comm _ r', ← add_assoc, ← H'.1]
      /-
        case h.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        ⊢ Eq f (HAdd.hAdd (m.reduce ⋯ f) ((Finsupp.linearCombination (MvPolynomial σ R …
      -/
      simp [reduce]
      /-
        🎉 no goals
      -/
    /-
      case h.right
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      i : ι
      hf : LE.le (m.degree (b i)) (m.degree f)
      deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
      g' : Finsupp ι (MvPolynomial σ R)
      r' : MvPolynomial σ R
      H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
      ⊢ And (∀ (i_1 : ι), LE.le (m.toSyn (m.degree (HMul.hMul (b i_1) ((HAdd.hAdd g' …
    -/
    constructor
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        ⊢ ∀ (i_1 : ι), LE.le (m.toSyn (m.degree (HMul.hMul (b i_1) ((HAdd.hAdd g' (Fin …
      -/
    · rintro j
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        j : ι
        ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b j) ((HAdd.hAdd g' (Finsupp.single i ( …
      -/
      simp only [Finsupp.coe_add, Pi.add_apply]
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        j : ι
        ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b j) (HAdd.hAdd (g' j) ((Finsupp.single …
      -/
      rw [mul_add]
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        j : ι
        ⊢ LE.le (m.toSyn (m.degree (HAdd.hAdd (HMul.hMul (b j) (g' j)) (HMul.hMul (b j …
      -/
      apply le_trans degree_add_le
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        j : ι
        ⊢ LE.le (Max.max (m.toSyn (m.degree (HMul.hMul (b j) (g' j)))) (m.toSyn (m.deg …
      -/
      simp only [sup_le_iff]
      /-
        case h.right.left
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        j : ι
        ⊢ And (LE.le (m.toSyn (m.degree (HMul.hMul (b j) (g' j)))) (m.toSyn (m.degree  …
      -/
      constructor
        /-
          case h.right.left.left
          σ : Type u_1
          m : MonomialOrder σ
          R : Type u_2
          inst✝ : CommRing R
          ι : Type u_3
          b : ι → MvPolynomial σ R
          hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
          f : MvPolynomial σ R
          hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
          hf0 : Not (Eq f 0)
          i : ι
          hf : LE.le (m.degree (b i)) (m.degree f)
          deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
          g' : Finsupp ι (MvPolynomial σ R)
          r' : MvPolynomial σ R
          H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
          j : ι
          ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b j) (g' j)))) (m.toSyn (m.degree f))
        -/
      · exact le_trans (H'.2.1 _) (le_of_lt deg_reduce)
        /-
          🎉 no goals
        -/
      · classical
        rw [Finsupp.single_apply]
        split_ifs with hc
        · apply le_trans degree_mul_le
          simp only [map_add]
          apply le_of_le_of_eq (add_le_add_left (degree_monomial_le _) _)
          simp only [← hc]
          rw [← map_add, m.toSyn.injective.eq_iff]
          rw [add_tsub_cancel_of_le]
          exact hf
        · simp only [mul_zero, degree_zero, map_zero]
          exact bot_le
      /-
        case h.right.right
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        i : ι
        hf : LE.le (m.degree (b i)) (m.degree f)
        deg_reduce : LT.lt (m.toSyn (m.degree (m.reduce ⋯ f))) (m.toSyn (m.degree f))
        g' : Finsupp ι (MvPolynomial σ R)
        r' : MvPolynomial σ R
        H' : And (Eq (m.reduce ⋯ f) (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomi …
        ⊢ ∀ (c : Finsupp σ Nat), Membership.mem r'.support c → ∀ (i : ι), Not (LE.le ( …
      -/
    · exact H'.2.2
      /-
        🎉 no goals
      -/
    /-
      case neg
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      hf : Not (Exists fun i => LE.le (m.degree (b i)) (m.degree f))
      ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
    -/
  · push_neg at hf
    suffices ∃ (g' : ι →₀ MvPolynomial σ R), ∃ r',
        (m.subLTerm f = Finsupp.linearCombination (MvPolynomial σ R) b g' + r') ∧
        (∀ i, m.degree ((b  i) * (g' i)) ≼[m] m.degree (m.subLTerm f)) ∧
        (∀ c ∈ r'.support, ∀ i, ¬ m.degree (b i) ≤ c) by
      obtain ⟨g', r', H'⟩ := this
      use g', r' +  monomial (m.degree f) (m.lCoeff f)
      constructor
      · simp [← add_assoc, ← H'.1, subLTerm]
      constructor
      · exact fun b ↦ le_trans (H'.2.1 b) (degree_sub_LTerm_le f)
      · intro c hc i
        by_cases hc' : c ∈ r'.support
        · exact H'.2.2 c hc' i
        · convert hf i
          classical
          have := MvPolynomial.support_add hc
          rw [Finset.mem_union, Classical.or_iff_not_imp_left] at this
          simpa only [Finset.mem_singleton] using support_monomial_subset (this hc')
    /-
      case neg
      σ : Type u_1
      m : MonomialOrder σ
      R : Type u_2
      inst✝ : CommRing R
      ι : Type u_3
      b : ι → MvPolynomial σ R
      hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
      f : MvPolynomial σ R
      hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
      hf0 : Not (Eq f 0)
      hf : ∀ (i : ι), Not (LE.le (m.degree (b i)) (m.degree f))
      ⊢ Exists fun g' => Exists fun r' => And (Eq (m.subLTerm f) (HAdd.hAdd ((Finsup …
    -/
    by_cases hf'0 : m.subLTerm f = 0
      /-
        case pos
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        hf : ∀ (i : ι), Not (LE.le (m.degree (b i)) (m.degree f))
        hf'0 : Eq (m.subLTerm f) 0
        ⊢ Exists fun g' => Exists fun r' => And (Eq (m.subLTerm f) (HAdd.hAdd ((Finsup …
      -/
    · refine ⟨0, 0, by simp [hf'0], ?_, by simp⟩
      /-
        case pos
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        hf : ∀ (i : ι), Not (LE.le (m.degree (b i)) (m.degree f))
        hf'0 : Eq (m.subLTerm f) 0
        ⊢ ∀ (i : ι), LE.le (m.toSyn (m.degree (HMul.hMul (b i) (0 i)))) (m.toSyn (m.de …
      -/
      intro b
      /-
        case pos
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b✝ : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b✝ i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b✝ i)) 0
        hf0 : Not (Eq f 0)
        hf : ∀ (i : ι), Not (LE.le (m.degree (b✝ i)) (m.degree f))
        hf'0 : Eq (m.subLTerm f) 0
        b : ι
        ⊢ LE.le (m.toSyn (m.degree (HMul.hMul (b✝ b) (0 b)))) (m.toSyn (m.degree (m.su …
      -/
      simp only [Finsupp.coe_zero, Pi.zero_apply, mul_zero, degree_zero, map_zero]
      /-
        case pos
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b✝ : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b✝ i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b✝ i)) 0
        hf0 : Not (Eq f 0)
        hf : ∀ (i : ι), Not (LE.le (m.degree (b✝ i)) (m.degree f))
        hf'0 : Eq (m.subLTerm f) 0
        b : ι
        ⊢ LE.le 0 (m.toSyn (m.degree (m.subLTerm f)))
      -/
      exact bot_le
      /-
        🎉 no goals
      -/
      /-
        case neg
        σ : Type u_1
        m : MonomialOrder σ
        R : Type u_2
        inst✝ : CommRing R
        ι : Type u_3
        b : ι → MvPolynomial σ R
        hb : ∀ (i : ι), IsUnit (m.lCoeff (b i))
        f : MvPolynomial σ R
        hb' : ∀ (i : ι), Ne (m.degree (b i)) 0
        hf0 : Not (Eq f 0)
        hf : ∀ (i : ι), Not (LE.le (m.degree (b i)) (m.degree f))
        hf'0 : Not (Eq (m.subLTerm f) 0)
        ⊢ Exists fun g' => Exists fun r' => And (Eq (m.subLTerm f) (HAdd.hAdd ((Finsup …
      -/
    · exact (div hb) (m.subLTerm f)
      /-
        🎉 no goals
      -/
termination_by WellFounded.wrap
  ((isWellFounded_iff m.syn fun x x_1 ↦ x < x_1).mp m.wf) (m.toSyn (m.degree f))
decreasing_by
· exact deg_reduce
· apply degree_sub_LTerm_lt
  intro hf0
  apply hf'0
  simp only [subLTerm, sub_eq_zero]
  nth_rewrite 1 [eq_C_of_degree_eq_zero hf0, hf0]
  simp


theorem div_set {B : Set (MvPolynomial σ R)}
    (hB : ∀ b ∈ B, IsUnit (m.lCoeff b)) (f : MvPolynomial σ R) :
    ∃ (g : B →₀ (MvPolynomial σ R)) (r : MvPolynomial σ R),
      f = Finsupp.linearCombination _ (fun (b : B) ↦ (b : MvPolynomial σ R)) g + r ∧
        (∀ (b : B), m.degree ((b : MvPolynomial σ R) * (g b)) ≼[m] m.degree f) ∧
        (∀ c ∈ r.support, ∀ b ∈ B, ¬ (m.degree b ≤ c)) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    B : Set (MvPolynomial σ R)
    hB : ∀ (b : MvPolynomial σ R), Membership.mem B b → IsUnit (m.lCoeff b)
    f : MvPolynomial σ R
    ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
  -/
  obtain ⟨g, r, H⟩ := m.div (b := fun (p : B) ↦ p) (fun b ↦ hB b b.prop) f
  /-
    case intro.intro
    σ : Type u_1
    m : MonomialOrder σ
    R : Type u_2
    inst✝ : CommRing R
    B : Set (MvPolynomial σ R)
    hB : ∀ (b : MvPolynomial σ R), Membership.mem B b → IsUnit (m.lCoeff b)
    f : MvPolynomial σ R
    g : Finsupp (↑B) (MvPolynomial σ R)
    r : MvPolynomial σ R
    H : And (Eq f (HAdd.hAdd ((Finsupp.linearCombination (MvPolynomial σ R) fun p  …
    ⊢ Exists fun g => Exists fun r => And (Eq f (HAdd.hAdd ((Finsupp.linearCombina …
  -/
  exact ⟨g, r, H.1, H.2.1, fun c hc b hb ↦ H.2.2 c hc ⟨b, hb⟩⟩
  /-
    🎉 no goals
  -/


