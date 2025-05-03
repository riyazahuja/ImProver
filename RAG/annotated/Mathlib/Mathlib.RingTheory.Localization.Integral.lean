open scoped Classical in
/-- `coeffIntegerNormalization p` gives the coefficients of the polynomial
`integerNormalization p` -/
noncomputable def coeffIntegerNormalization (p : S[X]) (i : ℕ) : R :=
  if hi : i ∈ p.support then
    Classical.choose
      (Classical.choose_spec (exist_integer_multiples_of_finset M (p.support.image p.coeff))
        (p.coeff i) (Finset.mem_image.mpr ⟨i, hi, rfl⟩))
  else 0


theorem coeffIntegerNormalization_of_not_mem_support (p : S[X]) (i : ℕ) (h : coeff p i = 0) :
    coeffIntegerNormalization M p i = 0 := by
  simp only [coeffIntegerNormalization, h, mem_support_iff, eq_self_iff_true, not_true, Ne,
    dif_neg, not_false_iff]


theorem coeffIntegerNormalization_mem_support (p : S[X]) (i : ℕ)
    (h : coeffIntegerNormalization M p i ≠ 0) : i ∈ p.support := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    p : Polynomial S
    i : Nat
    h : Ne (IsLocalization.coeffIntegerNormalization M p i) 0
    ⊢ Membership.mem p.support i
  -/
  contrapose h
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    p : Polynomial S
    i : Nat
    h : Not (Membership.mem p.support i)
    ⊢ Not (Ne (IsLocalization.coeffIntegerNormalization M p i) 0)
  -/
  rw [Ne, Classical.not_not, coeffIntegerNormalization, dif_neg h]
  /-
    🎉 no goals
  -/


/-- `integerNormalization g` normalizes `g` to have integer coefficients
by clearing the denominators -/
noncomputable def integerNormalization (p : S[X]) : R[X] :=
  ∑ i ∈ p.support, monomial i (coeffIntegerNormalization M p i)


@[simp]
theorem integerNormalization_coeff (p : S[X]) (i : ℕ) :
    (integerNormalization M p).coeff i = coeffIntegerNormalization M p i := by
  simp +contextual [integerNormalization, coeff_monomial,
    coeffIntegerNormalization_of_not_mem_support]


theorem integerNormalization_spec (p : S[X]) :
    ∃ b : M, ∀ i, algebraMap R S ((integerNormalization M p).coeff i) = (b : R) • p.coeff i := by
  classical
  use Classical.choose (exist_integer_multiples_of_finset M (p.support.image p.coeff))
  intro i
  rw [integerNormalization_coeff, coeffIntegerNormalization]
  split_ifs with hi
  · exact
      Classical.choose_spec
        (Classical.choose_spec (exist_integer_multiples_of_finset M (p.support.image p.coeff))
          (p.coeff i) (Finset.mem_image.mpr ⟨i, hi, rfl⟩))
  · rw [RingHom.map_zero, not_mem_support_iff.mp hi, smul_zero]
    -- Porting note: was `convert (smul_zero _).symm, ...`


theorem integerNormalization_map_to_map (p : S[X]) :
    ∃ b : M, (integerNormalization M p).map (algebraMap R S) = (b : R) • p :=
  let ⟨b, hb⟩ := integerNormalization_spec M p
  ⟨b,
    Polynomial.ext fun i => by
      /-
        R : Type u_1
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : Polynomial S
        b : Subtype fun x => Membership.mem M x
        hb : ∀ (i : Nat), Eq ((algebraMap R S) ((IsLocalization.integerNormalization M …
        i : Nat
        ⊢ Eq ((Polynomial.map (algebraMap R S) (IsLocalization.integerNormalization M  …
      -/
      rw [coeff_map, coeff_smul]
      /-
        R : Type u_1
        inst✝³ : CommRing R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommRing S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        p : Polynomial S
        b : Subtype fun x => Membership.mem M x
        hb : ∀ (i : Nat), Eq ((algebraMap R S) ((IsLocalization.integerNormalization M …
        i : Nat
        ⊢ Eq ((algebraMap R S) ((IsLocalization.integerNormalization M p).coeff i)) (H …
      -/
      exact hb i⟩
      /-
        🎉 no goals
      -/


theorem integerNormalization_eval₂_eq_zero (g : S →+* R') (p : S[X]) {x : R'}
    (hx : eval₂ g x p = 0) : eval₂ (g.comp (algebraMap R S)) x (integerNormalization M p) = 0 :=
  let ⟨b, hb⟩ := integerNormalization_map_to_map M p
  _root_.trans (eval₂_map (algebraMap R S) g x).symm
        /-
          R : Type u_1
          inst✝⁴ : CommRing R
          M : Submonoid R
          S : Type u_2
          inst✝³ : CommRing S
          inst✝² : Algebra R S
          inst✝¹ : IsLocalization M S
          R' : Type u_3
          inst✝ : CommRing R'
          g : RingHom S R'
          p : Polynomial S
          x : R'
          hx : Eq (Polynomial.eval₂ g x p) 0
          b : Subtype fun x => Membership.mem M x
          hb : Eq (Polynomial.map (algebraMap R S) (IsLocalization.integerNormalization  …
          ⊢ Eq (Polynomial.eval₂ g x (Polynomial.map (algebraMap R S) (IsLocalization.in …
        -/
    (by rw [hb, ← IsScalarTower.algebraMap_smul S (b : R) p, eval₂_smul, hx, mul_zero])
        /-
          🎉 no goals
        -/


theorem integerNormalization_aeval_eq_zero [Algebra R R'] [Algebra S R'] [IsScalarTower R S R']
    (p : S[X]) {x : R'} (hx : aeval x p = 0) : aeval x (integerNormalization M p) = 0 := by
  rw [aeval_def, IsScalarTower.algebraMap_eq R S R',
    integerNormalization_eval₂_eq_zero _ (algebraMap _ _) _ hx]


theorem integerNormalization_eq_zero_iff {p : K[X]} :
    integerNormalization (nonZeroDivisors A) p = 0 ↔ p = 0 := by
  /-
    A : Type u_3
    K : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial K
    ⊢ Iff (Eq (IsLocalization.integerNormalization (nonZeroDivisors A) p) 0) (Eq p …
  -/
  refine Polynomial.ext_iff.trans (Polynomial.ext_iff.trans ?_).symm
  /-
    A : Type u_3
    K : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial K
    ⊢ Iff (∀ (n : Nat), Eq (p.coeff n) (Polynomial.coeff 0 n)) (∀ (n : Nat), Eq (( …
  -/
  obtain ⟨⟨b, nonzero⟩, hb⟩ := integerNormalization_spec (nonZeroDivisors A) p
  /-
    case intro.mk
    A : Type u_3
    K : Type u_4
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Field K
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    p : Polynomial K
    b : A
    nonzero : Membership.mem (nonZeroDivisors A) b
    hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
    ⊢ Iff (∀ (n : Nat), Eq (p.coeff n) (Polynomial.coeff 0 n)) (∀ (n : Nat), Eq (( …
  -/
  constructor <;> intro h i
  · -- Porting note: avoided some defeq abuse
    /-
      case intro.mk.mp
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h : ∀ (n : Nat), Eq (p.coeff n) (Polynomial.coeff 0 n)
      i : Nat
      ⊢ Eq ((IsLocalization.integerNormalization (nonZeroDivisors A) p).coeff i) (Po …
    -/
    rw [coeff_zero, ← to_map_eq_zero_iff (K := K), hb i, h i, coeff_zero, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.mpr
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h : ∀ (n : Nat), Eq ((IsLocalization.integerNormalization (nonZeroDivisors A)  …
      i : Nat
      ⊢ Eq (p.coeff i) (Polynomial.coeff 0 i)
    -/
  · have hi := h i
    /-
      case intro.mk.mpr
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h : ∀ (n : Nat), Eq ((IsLocalization.integerNormalization (nonZeroDivisors A)  …
      i : Nat
      hi : Eq ((IsLocalization.integerNormalization (nonZeroDivisors A) p).coeff i)  …
      ⊢ Eq (p.coeff i) (Polynomial.coeff 0 i)
    -/
    rw [Polynomial.coeff_zero, ← @to_map_eq_zero_iff A _ K, hb i, Algebra.smul_def] at hi
    /-
      case intro.mk.mpr
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h : ∀ (n : Nat), Eq ((IsLocalization.integerNormalization (nonZeroDivisors A)  …
      i : Nat
      hi : Eq (HMul.hMul ((algebraMap A K) ↑⟨b, nonzero⟩) (p.coeff i)) 0
      ⊢ Eq (p.coeff i) (Polynomial.coeff 0 i)
    -/
    apply Or.resolve_left (eq_zero_or_eq_zero_of_mul_eq_zero hi)
    /-
      case intro.mk.mpr
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h : ∀ (n : Nat), Eq ((IsLocalization.integerNormalization (nonZeroDivisors A)  …
      i : Nat
      hi : Eq (HMul.hMul ((algebraMap A K) ↑⟨b, nonzero⟩) (p.coeff i)) 0
      ⊢ Not (Eq ((algebraMap A K) ↑⟨b, nonzero⟩) 0)
    -/
    intro h
    /-
      case intro.mk.mpr
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h✝ : ∀ (n : Nat), Eq ((IsLocalization.integerNormalization (nonZeroDivisors A) …
      i : Nat
      hi : Eq (HMul.hMul ((algebraMap A K) ↑⟨b, nonzero⟩) (p.coeff i)) 0
      h : Eq ((algebraMap A K) ↑⟨b, nonzero⟩) 0
      ⊢ False
    -/
    apply mem_nonZeroDivisors_iff_ne_zero.mp nonzero
    /-
      case intro.mk.mpr
      A : Type u_3
      K : Type u_4
      inst✝⁴ : CommRing A
      inst✝³ : IsDomain A
      inst✝² : Field K
      inst✝¹ : Algebra A K
      inst✝ : IsFractionRing A K
      p : Polynomial K
      b : A
      nonzero : Membership.mem (nonZeroDivisors A) b
      hb : ∀ (i : Nat), Eq ((algebraMap A K) ((IsLocalization.integerNormalization ( …
      h✝ : ∀ (n : Nat), Eq ((IsLocalization.integerNormalization (nonZeroDivisors A) …
      i : Nat
      hi : Eq (HMul.hMul ((algebraMap A K) ↑⟨b, nonzero⟩) (p.coeff i)) 0
      h : Eq ((algebraMap A K) ↑⟨b, nonzero⟩) 0
      ⊢ Eq b 0
    -/
    exact to_map_eq_zero_iff.mp h
    /-
      🎉 no goals
    -/


/-- An element of a ring is algebraic over the ring `A` iff it is algebraic
over the field of fractions of `A`.
-/
theorem isAlgebraic_iff [Algebra A C] [Algebra K C] [IsScalarTower A K C] {x : C} :
    IsAlgebraic A x ↔ IsAlgebraic K x := by
  /-
    A : Type u_3
    K : Type u_4
    C : Type u_5
    inst✝⁸ : CommRing A
    inst✝⁷ : IsDomain A
    inst✝⁶ : Field K
    inst✝⁵ : Algebra A K
    inst✝⁴ : IsFractionRing A K
    inst✝³ : CommRing C
    inst✝² : Algebra A C
    inst✝¹ : Algebra K C
    inst✝ : IsScalarTower A K C
    x : C
    ⊢ Iff (IsAlgebraic A x) (IsAlgebraic K x)
  -/
  constructor <;> rintro ⟨p, hp, px⟩
    /-
      case mp.intro.intro
      A : Type u_3
      K : Type u_4
      C : Type u_5
      inst✝⁸ : CommRing A
      inst✝⁷ : IsDomain A
      inst✝⁶ : Field K
      inst✝⁵ : Algebra A K
      inst✝⁴ : IsFractionRing A K
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra K C
      inst✝ : IsScalarTower A K C
      x : C
      p : Polynomial A
      hp : Ne p 0
      px : Eq ((Polynomial.aeval x) p) 0
      ⊢ IsAlgebraic K x
    -/
  · refine ⟨p.map (algebraMap A K), fun h => hp (Polynomial.ext fun i => ?_), ?_⟩
    · have : algebraMap A K (p.coeff i) = 0 :=
        _root_.trans (Polynomial.coeff_map _ _).symm (by simp [h])
      /-
        case mp.intro.intro.refine_1
        A : Type u_3
        K : Type u_4
        C : Type u_5
        inst✝⁸ : CommRing A
        inst✝⁷ : IsDomain A
        inst✝⁶ : Field K
        inst✝⁵ : Algebra A K
        inst✝⁴ : IsFractionRing A K
        inst✝³ : CommRing C
        inst✝² : Algebra A C
        inst✝¹ : Algebra K C
        inst✝ : IsScalarTower A K C
        x : C
        p : Polynomial A
        hp : Ne p 0
        px : Eq ((Polynomial.aeval x) p) 0
        h : Eq (Polynomial.map (algebraMap A K) p) 0
        i : Nat
        this : Eq ((algebraMap A K) (p.coeff i)) 0
        ⊢ Eq (p.coeff i) (Polynomial.coeff 0 i)
      -/
      exact to_map_eq_zero_iff.mp this
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.refine_2
        A : Type u_3
        K : Type u_4
        C : Type u_5
        inst✝⁸ : CommRing A
        inst✝⁷ : IsDomain A
        inst✝⁶ : Field K
        inst✝⁵ : Algebra A K
        inst✝⁴ : IsFractionRing A K
        inst✝³ : CommRing C
        inst✝² : Algebra A C
        inst✝¹ : Algebra K C
        inst✝ : IsScalarTower A K C
        x : C
        p : Polynomial A
        hp : Ne p 0
        px : Eq ((Polynomial.aeval x) p) 0
        ⊢ Eq ((Polynomial.aeval x) (Polynomial.map (algebraMap A K) p)) 0
      -/
    · exact (Polynomial.aeval_map_algebraMap K _ _).trans px
      /-
        🎉 no goals
      -/
  · exact
      ⟨integerNormalization _ p, mt integerNormalization_eq_zero_iff.mp hp,
        integerNormalization_aeval_eq_zero _ p px⟩


/-- A ring is algebraic over the ring `A` iff it is algebraic over the field of fractions of `A`.
-/
theorem comap_isAlgebraic_iff [Algebra A C] [Algebra K C] [IsScalarTower A K C] :
    Algebra.IsAlgebraic A C ↔ Algebra.IsAlgebraic K C :=
  ⟨fun h => ⟨fun x => (isAlgebraic_iff A K C).mp (h.isAlgebraic x)⟩,
   fun h => ⟨fun x => (isAlgebraic_iff A K C).mpr (h.isAlgebraic x)⟩⟩


theorem RingHom.isIntegralElem_localization_at_leadingCoeff {R S : Type*} [CommRing R] [CommRing S]
    (f : R →+* S) (x : S) (p : R[X]) (hf : p.eval₂ f x = 0) (M : Submonoid R)
    (hM : p.leadingCoeff ∈ M) {Rₘ Sₘ : Type*} [CommRing Rₘ] [CommRing Sₘ] [Algebra R Rₘ]
    [IsLocalization M Rₘ] [Algebra S Sₘ] [IsLocalization (M.map f : Submonoid S) Sₘ] :
    (map Sₘ f M.le_comap_map : Rₘ →+* _).IsIntegralElem (algebraMap S Sₘ x) := by
  /-
    R : Type u_5
    S : Type u_6
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    x : S
    p : Polynomial R
    hf : Eq (Polynomial.eval₂ f x p) 0
    M : Submonoid R
    hM : Membership.mem M p.leadingCoeff
    Rₘ : Type u_7
    Sₘ : Type u_8
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map f M) Sₘ
    ⊢ (IsLocalization.map Sₘ f ⋯).IsIntegralElem ((algebraMap S Sₘ) x)
  -/
  by_cases triv : (1 : Rₘ) = 0
    /-
      case pos
      R : Type u_5
      S : Type u_6
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      x : S
      p : Polynomial R
      hf : Eq (Polynomial.eval₂ f x p) 0
      M : Submonoid R
      hM : Membership.mem M p.leadingCoeff
      Rₘ : Type u_7
      Sₘ : Type u_8
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra S Sₘ
      inst✝ : IsLocalization (Submonoid.map f M) Sₘ
      triv : Eq 1 0
      ⊢ (IsLocalization.map Sₘ f ⋯).IsIntegralElem ((algebraMap S Sₘ) x)
    -/
  · exact ⟨0, ⟨_root_.trans leadingCoeff_zero triv.symm, eval₂_zero _ _⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_5
    S : Type u_6
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    x : S
    p : Polynomial R
    hf : Eq (Polynomial.eval₂ f x p) 0
    M : Submonoid R
    hM : Membership.mem M p.leadingCoeff
    Rₘ : Type u_7
    Sₘ : Type u_8
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map f M) Sₘ
    triv : Not (Eq 1 0)
    ⊢ (IsLocalization.map Sₘ f ⋯).IsIntegralElem ((algebraMap S Sₘ) x)
  -/
  haveI : Nontrivial Rₘ := nontrivial_of_ne 1 0 triv
  /-
    case neg
    R : Type u_5
    S : Type u_6
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    x : S
    p : Polynomial R
    hf : Eq (Polynomial.eval₂ f x p) 0
    M : Submonoid R
    hM : Membership.mem M p.leadingCoeff
    Rₘ : Type u_7
    Sₘ : Type u_8
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map f M) Sₘ
    triv : Not (Eq 1 0)
    this : Nontrivial Rₘ
    ⊢ (IsLocalization.map Sₘ f ⋯).IsIntegralElem ((algebraMap S Sₘ) x)
  -/
  obtain ⟨b, hb⟩ := isUnit_iff_exists_inv.mp (map_units Rₘ ⟨p.leadingCoeff, hM⟩)
  /-
    case neg.intro
    R : Type u_5
    S : Type u_6
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    x : S
    p : Polynomial R
    hf : Eq (Polynomial.eval₂ f x p) 0
    M : Submonoid R
    hM : Membership.mem M p.leadingCoeff
    Rₘ : Type u_7
    Sₘ : Type u_8
    inst✝⁵ : CommRing Rₘ
    inst✝⁴ : CommRing Sₘ
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : Algebra S Sₘ
    inst✝ : IsLocalization (Submonoid.map f M) Sₘ
    triv : Not (Eq 1 0)
    this : Nontrivial Rₘ
    b : Rₘ
    hb : Eq (HMul.hMul ((algebraMap R Rₘ) ↑⟨p.leadingCoeff, hM⟩) b) 1
    ⊢ (IsLocalization.map Sₘ f ⋯).IsIntegralElem ((algebraMap S Sₘ) x)
  -/
  refine ⟨p.map (algebraMap R Rₘ) * C b, ⟨?_, ?_⟩⟩
    /-
      case neg.intro.refine_1
      R : Type u_5
      S : Type u_6
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      x : S
      p : Polynomial R
      hf : Eq (Polynomial.eval₂ f x p) 0
      M : Submonoid R
      hM : Membership.mem M p.leadingCoeff
      Rₘ : Type u_7
      Sₘ : Type u_8
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra S Sₘ
      inst✝ : IsLocalization (Submonoid.map f M) Sₘ
      triv : Not (Eq 1 0)
      this : Nontrivial Rₘ
      b : Rₘ
      hb : Eq (HMul.hMul ((algebraMap R Rₘ) ↑⟨p.leadingCoeff, hM⟩) b) 1
      ⊢ (HMul.hMul (Polynomial.map (algebraMap R Rₘ) p) (Polynomial.C b)).Monic
    -/
  · refine monic_mul_C_of_leadingCoeff_mul_eq_one ?_
    /-
      case neg.intro.refine_1
      R : Type u_5
      S : Type u_6
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      x : S
      p : Polynomial R
      hf : Eq (Polynomial.eval₂ f x p) 0
      M : Submonoid R
      hM : Membership.mem M p.leadingCoeff
      Rₘ : Type u_7
      Sₘ : Type u_8
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra S Sₘ
      inst✝ : IsLocalization (Submonoid.map f M) Sₘ
      triv : Not (Eq 1 0)
      this : Nontrivial Rₘ
      b : Rₘ
      hb : Eq (HMul.hMul ((algebraMap R Rₘ) ↑⟨p.leadingCoeff, hM⟩) b) 1
      ⊢ Eq (HMul.hMul (Polynomial.map (algebraMap R Rₘ) p).leadingCoeff b) 1
    -/
    rwa [leadingCoeff_map_of_leadingCoeff_ne_zero (algebraMap R Rₘ)]
    refine fun hfp => zero_ne_one
      (_root_.trans (zero_mul b).symm (hfp ▸ hb) : (0 : Rₘ) = 1)
    /-
      case neg.intro.refine_2
      R : Type u_5
      S : Type u_6
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      x : S
      p : Polynomial R
      hf : Eq (Polynomial.eval₂ f x p) 0
      M : Submonoid R
      hM : Membership.mem M p.leadingCoeff
      Rₘ : Type u_7
      Sₘ : Type u_8
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra S Sₘ
      inst✝ : IsLocalization (Submonoid.map f M) Sₘ
      triv : Not (Eq 1 0)
      this : Nontrivial Rₘ
      b : Rₘ
      hb : Eq (HMul.hMul ((algebraMap R Rₘ) ↑⟨p.leadingCoeff, hM⟩) b) 1
      ⊢ Eq (Polynomial.eval₂ (IsLocalization.map Sₘ f ⋯) ((algebraMap S Sₘ) x) (HMul …
    -/
  · refine eval₂_mul_eq_zero_of_left _ _ _ ?_
    /-
      case neg.intro.refine_2
      R : Type u_5
      S : Type u_6
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      x : S
      p : Polynomial R
      hf : Eq (Polynomial.eval₂ f x p) 0
      M : Submonoid R
      hM : Membership.mem M p.leadingCoeff
      Rₘ : Type u_7
      Sₘ : Type u_8
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra S Sₘ
      inst✝ : IsLocalization (Submonoid.map f M) Sₘ
      triv : Not (Eq 1 0)
      this : Nontrivial Rₘ
      b : Rₘ
      hb : Eq (HMul.hMul ((algebraMap R Rₘ) ↑⟨p.leadingCoeff, hM⟩) b) 1
      ⊢ Eq (Polynomial.eval₂ (IsLocalization.map Sₘ f ⋯) ((algebraMap S Sₘ) x) (Poly …
    -/
    rw [eval₂_map, IsLocalization.map_comp, ← hom_eval₂ _ f (algebraMap S Sₘ) x]
    /-
      case neg.intro.refine_2
      R : Type u_5
      S : Type u_6
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      f : RingHom R S
      x : S
      p : Polynomial R
      hf : Eq (Polynomial.eval₂ f x p) 0
      M : Submonoid R
      hM : Membership.mem M p.leadingCoeff
      Rₘ : Type u_7
      Sₘ : Type u_8
      inst✝⁵ : CommRing Rₘ
      inst✝⁴ : CommRing Sₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra S Sₘ
      inst✝ : IsLocalization (Submonoid.map f M) Sₘ
      triv : Not (Eq 1 0)
      this : Nontrivial Rₘ
      b : Rₘ
      hb : Eq (HMul.hMul ((algebraMap R Rₘ) ↑⟨p.leadingCoeff, hM⟩) b) 1
      ⊢ Eq ((algebraMap S Sₘ) (Polynomial.eval₂ f x p)) 0
    -/
    exact _root_.trans (congr_arg (algebraMap S Sₘ) hf) (RingHom.map_zero _)
    /-
      🎉 no goals
    -/


/-- Given a particular witness to an element being algebraic over an algebra `R → S`,
We can localize to a submonoid containing the leading coefficient to make it integral.
Explicitly, the map between the localizations will be an integral ring morphism -/
theorem is_integral_localization_at_leadingCoeff {x : S} (p : R[X]) (hp : aeval x p = 0)
    (hM : p.leadingCoeff ∈ M) :
    (map Sₘ (algebraMap R S)
            (show _ ≤ (Algebra.algebraMapSubmonoid S M).comap _ from M.le_comap_map) :
          Rₘ →+* _).IsIntegralElem
      (algebraMap S Sₘ x) :=
  -- Porting note: added `haveI`
  haveI : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ :=
    inferInstanceAs (IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ)
  (algebraMap R S).isIntegralElem_localization_at_leadingCoeff x p hp M hM


/-- If `R → S` is an integral extension, `M` is a submonoid of `R`,
`Rₘ` is the localization of `R` at `M`,
and `Sₘ` is the localization of `S` at the image of `M` under the extension map,
then the induced map `Rₘ → Sₘ` is also an integral extension -/
theorem isIntegral_localization [Algebra.IsIntegral R S] :
    (map Sₘ (algebraMap R S)
          (show _ ≤ (Algebra.algebraMapSubmonoid S M).comap _ from M.le_comap_map) :
        Rₘ →+* _).IsIntegral := by
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁶ : CommRing Rₘ
    inst✝⁵ : CommRing Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra S Sₘ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝ : Algebra.IsIntegral R S
    ⊢ (IsLocalization.map Sₘ (algebraMap R S) ⋯).IsIntegral
  -/
  intro x
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁶ : CommRing Rₘ
    inst✝⁵ : CommRing Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra S Sₘ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝ : Algebra.IsIntegral R S
    x : Sₘ
    ⊢ (IsLocalization.map Sₘ (algebraMap R S) ⋯).IsIntegralElem x
  -/
  obtain ⟨⟨s, ⟨u, hu⟩⟩, hx⟩ := surj (Algebra.algebraMapSubmonoid S M) x
  /-
    case intro.mk.mk
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁶ : CommRing Rₘ
    inst✝⁵ : CommRing Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra S Sₘ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝ : Algebra.IsIntegral R S
    x : Sₘ
    s u : S
    hu : Membership.mem (Algebra.algebraMapSubmonoid S M) u
    hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, hu⟩ }.2)) ((al …
    ⊢ (IsLocalization.map Sₘ (algebraMap R S) ⋯).IsIntegralElem x
  -/
  obtain ⟨v, hv⟩ := hu
  /-
    case intro.mk.mk.intro
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁶ : CommRing Rₘ
    inst✝⁵ : CommRing Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra S Sₘ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝ : Algebra.IsIntegral R S
    x : Sₘ
    s u : S
    v : R
    hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
    hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
    ⊢ (IsLocalization.map Sₘ (algebraMap R S) ⋯).IsIntegralElem x
  -/
  obtain ⟨v', hv'⟩ := isUnit_iff_exists_inv'.1 (map_units Rₘ ⟨v, hv.1⟩)
  /-
    case intro.mk.mk.intro.intro
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁸ : CommRing S
    inst✝⁷ : Algebra R S
    Rₘ : Type u_3
    Sₘ : Type u_4
    inst✝⁶ : CommRing Rₘ
    inst✝⁵ : CommRing Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra S Sₘ
    inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
    inst✝ : Algebra.IsIntegral R S
    x : Sₘ
    s u : S
    v : R
    hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
    hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
    v' : Rₘ
    hv' : Eq (HMul.hMul v' ((algebraMap R Rₘ) ↑⟨v, ⋯⟩)) 1
    ⊢ (IsLocalization.map Sₘ (algebraMap R S) ⋯).IsIntegralElem x
  -/
  refine @IsIntegral.of_mul_unit Rₘ _ _ _ (localizationAlgebra M S) x (algebraMap S Sₘ u) v' ?_ ?_
    /-
      case intro.mk.mk.intro.intro.refine_1
      R : Type u_1
      inst✝⁹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁸ : CommRing S
      inst✝⁷ : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁶ : CommRing Rₘ
      inst✝⁵ : CommRing Sₘ
      inst✝⁴ : Algebra R Rₘ
      inst✝³ : IsLocalization M Rₘ
      inst✝² : Algebra S Sₘ
      inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝ : Algebra.IsIntegral R S
      x : Sₘ
      s u : S
      v : R
      hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
      hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
      v' : Rₘ
      hv' : Eq (HMul.hMul v' ((algebraMap R Rₘ) ↑⟨v, ⋯⟩)) 1
      ⊢ Eq (HMul.hMul ((algebraMap Rₘ Sₘ) v') ((algebraMap S Sₘ) u)) 1
    -/
  · replace hv' := congr_arg (@algebraMap Rₘ Sₘ _ _ (localizationAlgebra M S)) hv'
    /-
      case intro.mk.mk.intro.intro.refine_1
      R : Type u_1
      inst✝⁹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁸ : CommRing S
      inst✝⁷ : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁶ : CommRing Rₘ
      inst✝⁵ : CommRing Sₘ
      inst✝⁴ : Algebra R Rₘ
      inst✝³ : IsLocalization M Rₘ
      inst✝² : Algebra S Sₘ
      inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝ : Algebra.IsIntegral R S
      x : Sₘ
      s u : S
      v : R
      hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
      hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
      v' : Rₘ
      hv' : Eq ((algebraMap Rₘ Sₘ) (HMul.hMul v' ((algebraMap R Rₘ) ↑⟨v, ⋯⟩))) ((alg …
      ⊢ Eq (HMul.hMul ((algebraMap Rₘ Sₘ) v') ((algebraMap S Sₘ) u)) 1
    -/
    rw [RingHom.map_mul, RingHom.map_one, ← RingHom.comp_apply _ (algebraMap R Rₘ)] at hv'
    -- Porting note: added argument
    erw [IsLocalization.map_comp
      (show _ ≤ (Algebra.algebraMapSubmonoid S M).comap _ from M.le_comap_map)] at hv'
    /-
      case intro.mk.mk.intro.intro.refine_1
      R : Type u_1
      inst✝⁹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁸ : CommRing S
      inst✝⁷ : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁶ : CommRing Rₘ
      inst✝⁵ : CommRing Sₘ
      inst✝⁴ : Algebra R Rₘ
      inst✝³ : IsLocalization M Rₘ
      inst✝² : Algebra S Sₘ
      inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝ : Algebra.IsIntegral R S
      x : Sₘ
      s u : S
      v : R
      hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
      hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
      v' : Rₘ
      hv' : Eq (HMul.hMul ((algebraMap Rₘ Sₘ) v') (((algebraMap S Sₘ).comp (algebraM …
      ⊢ Eq (HMul.hMul ((algebraMap Rₘ Sₘ) v') ((algebraMap S Sₘ) u)) 1
    -/
    exact hv.2 ▸ hv'
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.mk.intro.intro.refine_2
      R : Type u_1
      inst✝⁹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁸ : CommRing S
      inst✝⁷ : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁶ : CommRing Rₘ
      inst✝⁵ : CommRing Sₘ
      inst✝⁴ : Algebra R Rₘ
      inst✝³ : IsLocalization M Rₘ
      inst✝² : Algebra S Sₘ
      inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝ : Algebra.IsIntegral R S
      x : Sₘ
      s u : S
      v : R
      hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
      hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
      v' : Rₘ
      hv' : Eq (HMul.hMul v' ((algebraMap R Rₘ) ↑⟨v, ⋯⟩)) 1
      ⊢ IsIntegral Rₘ (HMul.hMul x ((algebraMap S Sₘ) u))
    -/
  · obtain ⟨p, hp⟩ := Algebra.IsIntegral.isIntegral (R := R) s
    /-
      case intro.mk.mk.intro.intro.refine_2.intro
      R : Type u_1
      inst✝⁹ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁸ : CommRing S
      inst✝⁷ : Algebra R S
      Rₘ : Type u_3
      Sₘ : Type u_4
      inst✝⁶ : CommRing Rₘ
      inst✝⁵ : CommRing Sₘ
      inst✝⁴ : Algebra R Rₘ
      inst✝³ : IsLocalization M Rₘ
      inst✝² : Algebra S Sₘ
      inst✝¹ : IsLocalization (Algebra.algebraMapSubmonoid S M) Sₘ
      inst✝ : Algebra.IsIntegral R S
      x : Sₘ
      s u : S
      v : R
      hv : And (Membership.mem (↑M) v) (Eq ((algebraMap R S) v) u)
      hx : Eq (HMul.hMul x ((algebraMap S Sₘ) ↑{ fst := s, snd := ⟨u, ⋯⟩ }.2)) ((alg …
      v' : Rₘ
      hv' : Eq (HMul.hMul v' ((algebraMap R Rₘ) ↑⟨v, ⋯⟩)) 1
      p : Polynomial R
      hp : And p.Monic (Eq (Polynomial.eval₂ (algebraMap R S) s p) 0)
      ⊢ IsIntegral Rₘ (HMul.hMul x ((algebraMap S Sₘ) u))
    -/
    exact hx.symm ▸ is_integral_localization_at_leadingCoeff p hp.2 (hp.1.symm ▸ M.one_mem)
    /-
      🎉 no goals
    -/


@[nolint unusedHavesSuffices] -- It claims the `have : IsLocalization` line is unnecessary,
                              -- but remove it and the proof won't work.
theorem isIntegral_localization' {R S : Type*} [CommRing R] [CommRing S] {f : R →+* S}
    (hf : f.IsIntegral) (M : Submonoid R) :
    (map (Localization (M.map (f : R →* S))) f
          (M.le_comap_map : _ ≤ Submonoid.comap (f : R →* S) _) :
        Localization M →+* _).IsIntegral :=
  -- Porting note: added
  let _ := f.toAlgebra
  have : Algebra.IsIntegral R S := ⟨hf⟩
  have : IsLocalization (Algebra.algebraMapSubmonoid S M)
    (Localization (Submonoid.map (f : R →* S) M)) := Localization.isLocalization
  isIntegral_localization


theorem IsLocalization.scaleRoots_commonDenom_mem_lifts (p : Rₘ[X])
    (hp : p.leadingCoeff ∈ (algebraMap R Rₘ).range) :
    p.scaleRoots (algebraMap R Rₘ <| IsLocalization.commonDenom M p.support p.coeff) ∈
      Polynomial.lifts (algebraMap R Rₘ) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    Rₘ : Type u_3
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    p : Polynomial Rₘ
    hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
    ⊢ Membership.mem (Polynomial.lifts (algebraMap R Rₘ)) (p.scaleRoots ((algebraM …
  -/
  rw [Polynomial.lifts_iff_coeff_lifts]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    Rₘ : Type u_3
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    p : Polynomial Rₘ
    hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
    ⊢ ∀ (n : Nat), Membership.mem (Set.range ⇑(algebraMap R Rₘ)) ((p.scaleRoots (( …
  -/
  intro n
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    Rₘ : Type u_3
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    p : Polynomial Rₘ
    hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
    n : Nat
    ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) ((p.scaleRoots ((algebraMap R  …
  -/
  rw [Polynomial.coeff_scaleRoots]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    Rₘ : Type u_3
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    p : Polynomial Rₘ
    hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
    n : Nat
    ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul (p.coeff n) (HPow.h …
  -/
  by_cases h₁ : n ∈ p.support
  /-
    case pos
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    Rₘ : Type u_3
    inst✝² : CommRing Rₘ
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    p : Polynomial Rₘ
    hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
    n : Nat
    h₁ : Membership.mem p.support n
    ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul (p.coeff n) (HPow.h …
  -/
  on_goal 1 => by_cases h₂ : n = p.natDegree
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Membership.mem p.support n
      h₂ : Eq n p.natDegree
      ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul (p.coeff n) (HPow.h …
    -/
  · rwa [h₂, Polynomial.coeff_natDegree, tsub_self, pow_zero, _root_.mul_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Membership.mem p.support n
      h₂ : Not (Eq n p.natDegree)
      ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul (p.coeff n) (HPow.h …
    -/
  · have : n + 1 ≤ p.natDegree := lt_of_le_of_ne (Polynomial.le_natDegree_of_mem_supp _ h₁) h₂
    rw [← tsub_add_cancel_of_le (le_tsub_of_add_le_left this), pow_add, pow_one, mul_comm,
      _root_.mul_assoc, ← map_pow]
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Membership.mem p.support n
      h₂ : Not (Eq n p.natDegree)
      this : LE.le (HAdd.hAdd n 1) p.natDegree
      ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul ((algebraMap R Rₘ)  …
    -/
    change _ ∈ (algebraMap R Rₘ).range
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Membership.mem p.support n
      h₂ : Not (Eq n p.natDegree)
      this : LE.le (HAdd.hAdd n 1) p.natDegree
      ⊢ Membership.mem (algebraMap R Rₘ).range (HMul.hMul ((algebraMap R Rₘ) (HPow.h …
    -/
    apply mul_mem
      /-
        case neg.a
        R : Type u_1
        inst✝³ : CommRing R
        M : Submonoid R
        Rₘ : Type u_3
        inst✝² : CommRing Rₘ
        inst✝¹ : Algebra R Rₘ
        inst✝ : IsLocalization M Rₘ
        p : Polynomial Rₘ
        hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
        n : Nat
        h₁ : Membership.mem p.support n
        h₂ : Not (Eq n p.natDegree)
        this : LE.le (HAdd.hAdd n 1) p.natDegree
        ⊢ Membership.mem (algebraMap R Rₘ).range ((algebraMap R Rₘ) (HPow.hPow (↑(IsLo …
      -/
    · exact RingHom.mem_range_self _ _
      /-
        🎉 no goals
      -/
      /-
        case neg.a
        R : Type u_1
        inst✝³ : CommRing R
        M : Submonoid R
        Rₘ : Type u_3
        inst✝² : CommRing Rₘ
        inst✝¹ : Algebra R Rₘ
        inst✝ : IsLocalization M Rₘ
        p : Polynomial Rₘ
        hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
        n : Nat
        h₁ : Membership.mem p.support n
        h₂ : Not (Eq n p.natDegree)
        this : LE.le (HAdd.hAdd n 1) p.natDegree
        ⊢ Membership.mem (algebraMap R Rₘ).range (HMul.hMul ((algebraMap R Rₘ) ↑(IsLoc …
      -/
    · rw [← Algebra.smul_def]
      /-
        case neg.a
        R : Type u_1
        inst✝³ : CommRing R
        M : Submonoid R
        Rₘ : Type u_3
        inst✝² : CommRing Rₘ
        inst✝¹ : Algebra R Rₘ
        inst✝ : IsLocalization M Rₘ
        p : Polynomial Rₘ
        hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
        n : Nat
        h₁ : Membership.mem p.support n
        h₂ : Not (Eq n p.natDegree)
        this : LE.le (HAdd.hAdd n 1) p.natDegree
        ⊢ Membership.mem (algebraMap R Rₘ).range (HSMul.hSMul (↑(IsLocalization.common …
      -/
      exact ⟨_, IsLocalization.map_integerMultiple M p.support p.coeff ⟨n, h₁⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Not (Membership.mem p.support n)
      ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul (p.coeff n) (HPow.h …
    -/
  · rw [Polynomial.not_mem_support_iff] at h₁
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Eq (p.coeff n) 0
      ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) (HMul.hMul (p.coeff n) (HPow.h …
    -/
    rw [h₁, zero_mul]
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      Rₘ : Type u_3
      inst✝² : CommRing Rₘ
      inst✝¹ : Algebra R Rₘ
      inst✝ : IsLocalization M Rₘ
      p : Polynomial Rₘ
      hp : Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
      n : Nat
      h₁ : Eq (p.coeff n) 0
      ⊢ Membership.mem (Set.range ⇑(algebraMap R Rₘ)) 0
    -/
    exact zero_mem (algebraMap R Rₘ).range
    /-
      🎉 no goals
    -/


theorem IsIntegral.exists_multiple_integral_of_isLocalization [Algebra Rₘ S] [IsScalarTower R Rₘ S]
    (x : S) (hx : IsIntegral Rₘ x) : ∃ m : M, IsIntegral R (m • x) := by
  /-
    R : Type u_1
    inst✝⁷ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    Rₘ : Type u_3
    inst✝⁴ : CommRing Rₘ
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : Algebra Rₘ S
    inst✝ : IsScalarTower R Rₘ S
    x : S
    hx : IsIntegral Rₘ x
    ⊢ Exists fun m => IsIntegral R (HSMul.hSMul m x)
  -/
  cases' subsingleton_or_nontrivial Rₘ with _ nontriv
    /-
      case inl
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      hx : IsIntegral Rₘ x
      h✝ : Subsingleton Rₘ
      ⊢ Exists fun m => IsIntegral R (HSMul.hSMul m x)
    -/
  · haveI := (_root_.algebraMap Rₘ S).codomain_trivial
    /-
      case inl
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      hx : IsIntegral Rₘ x
      h✝ : Subsingleton Rₘ
      this : Subsingleton S
      ⊢ Exists fun m => IsIntegral R (HSMul.hSMul m x)
    -/
    exact ⟨1, Polynomial.X, Polynomial.monic_X, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝⁷ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    Rₘ : Type u_3
    inst✝⁴ : CommRing Rₘ
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : Algebra Rₘ S
    inst✝ : IsScalarTower R Rₘ S
    x : S
    hx : IsIntegral Rₘ x
    nontriv : Nontrivial Rₘ
    ⊢ Exists fun m => IsIntegral R (HSMul.hSMul m x)
  -/
  obtain ⟨p, hp₁, hp₂⟩ := hx
  -- Porting note: obtain doesn't support side goals
  have :=
    lifts_and_natDegree_eq_and_monic (IsLocalization.scaleRoots_commonDenom_mem_lifts M p ?_) ?_
    /-
      case inr.intro.intro.refine_3
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      nontriv : Nontrivial Rₘ
      p : Polynomial Rₘ
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap Rₘ S) x p) 0
      this : Exists fun q => And (Eq (Polynomial.map (algebraMap R Rₘ) q) (p.scaleRo …
      ⊢ Exists fun m => IsIntegral R (HSMul.hSMul m x)
    -/
  · obtain ⟨p', hp'₁, -, hp'₂⟩ := this
    /-
      case inr.intro.intro.refine_3.intro.intro.intro
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      nontriv : Nontrivial Rₘ
      p : Polynomial Rₘ
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap Rₘ S) x p) 0
      p' : Polynomial R
      hp'₁ : Eq (Polynomial.map (algebraMap R Rₘ) p') (p.scaleRoots ((algebraMap R R …
      hp'₂ : p'.Monic
      ⊢ Exists fun m => IsIntegral R (HSMul.hSMul m x)
    -/
    refine ⟨IsLocalization.commonDenom M p.support p.coeff, p', hp'₂, ?_⟩
    rw [IsScalarTower.algebraMap_eq R Rₘ S, ← Polynomial.eval₂_map, hp'₁, Submonoid.smul_def,
      Algebra.smul_def, IsScalarTower.algebraMap_apply R Rₘ S]
    /-
      case inr.intro.intro.refine_3.intro.intro.intro
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      nontriv : Nontrivial Rₘ
      p : Polynomial Rₘ
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap Rₘ S) x p) 0
      p' : Polynomial R
      hp'₁ : Eq (Polynomial.map (algebraMap R Rₘ) p') (p.scaleRoots ((algebraMap R R …
      hp'₂ : p'.Monic
      ⊢ Eq (Polynomial.eval₂ (algebraMap Rₘ S) (HMul.hMul ((algebraMap Rₘ S) ((algeb …
    -/
    exact Polynomial.scaleRoots_eval₂_eq_zero _ hp₂
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.refine_1
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      nontriv : Nontrivial Rₘ
      p : Polynomial Rₘ
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap Rₘ S) x p) 0
      ⊢ Membership.mem (algebraMap R Rₘ).range p.leadingCoeff
    -/
  · rw [hp₁.leadingCoeff]
    /-
      case inr.intro.intro.refine_1
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      nontriv : Nontrivial Rₘ
      p : Polynomial Rₘ
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap Rₘ S) x p) 0
      ⊢ Membership.mem (algebraMap R Rₘ).range 1
    -/
    exact one_mem _
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.refine_2
      R : Type u_1
      inst✝⁷ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      Rₘ : Type u_3
      inst✝⁴ : CommRing Rₘ
      inst✝³ : Algebra R Rₘ
      inst✝² : IsLocalization M Rₘ
      inst✝¹ : Algebra Rₘ S
      inst✝ : IsScalarTower R Rₘ S
      x : S
      nontriv : Nontrivial Rₘ
      p : Polynomial Rₘ
      hp₁ : p.Monic
      hp₂ : Eq (Polynomial.eval₂ (algebraMap Rₘ S) x p) 0
      ⊢ (p.scaleRoots ((algebraMap R Rₘ) ↑(IsLocalization.commonDenom M p.support p. …
    -/
  · rwa [Polynomial.monic_scaleRoots_iff]
    /-
      🎉 no goals
    -/


/-- If the field `L` is an algebraic extension of the integral domain `A`,
the integral closure `C` of `A` in `L` has fraction field `L`. -/
theorem isFractionRing_of_algebraic [Algebra.IsAlgebraic A L]
    (inj : ∀ x, algebraMap A L x = 0 → x = 0) : IsFractionRing C L :=
  { map_units' := fun ⟨y, hy⟩ =>
      IsUnit.mk0 _
        (show algebraMap C L y ≠ 0 from fun h =>
          mem_nonZeroDivisors_iff_ne_zero.mp hy
            ((injective_iff_map_eq_zero (algebraMap C L)).mp (algebraMap_injective C A L) _ h))
    surj' := fun z =>
      let ⟨x, hx, int⟩ := (Algebra.IsAlgebraic.isAlgebraic z).exists_integral_multiple
        ((injective_iff_map_eq_zero _).mpr inj)
      ⟨⟨mk' C _ int, algebraMap _ _ x, mem_nonZeroDivisors_of_ne_zero fun h ↦
                        /-
                          A : Type u_3
                          inst✝⁹ : CommRing A
                          L : Type u_5
                          inst✝⁸ : Field L
                          inst✝⁷ : Algebra A L
                          C : Type u_6
                          inst✝⁶ : CommRing C
                          inst✝⁵ : IsDomain C
                          inst✝⁴ : Algebra C L
                          inst✝³ : IsIntegralClosure C A L
                          inst✝² : Algebra A C
                          inst✝¹ : IsScalarTower A C L
                          inst✝ : Algebra.IsAlgebraic A L
                          inj : ∀ (x : A), Eq ((algebraMap A L) x) 0 → Eq x 0
                          z : L
                          x : A
                          hx : Ne x 0
                          int : IsIntegral A (HSMul.hSMul x z)
                          h : Eq ((algebraMap A C) x) 0
                          ⊢ Eq ((algebraMap A L) x) 0
                        -/
        hx (inj _ <| by rw [IsScalarTower.algebraMap_apply A C L, h, RingHom.map_zero])⟩, by
                        /-
                          🎉 no goals
                        -/
        /-
          A : Type u_3
          inst✝⁹ : CommRing A
          L : Type u_5
          inst✝⁸ : Field L
          inst✝⁷ : Algebra A L
          C : Type u_6
          inst✝⁶ : CommRing C
          inst✝⁵ : IsDomain C
          inst✝⁴ : Algebra C L
          inst✝³ : IsIntegralClosure C A L
          inst✝² : Algebra A C
          inst✝¹ : IsScalarTower A C L
          inst✝ : Algebra.IsAlgebraic A L
          inj : ∀ (x : A), Eq ((algebraMap A L) x) 0 → Eq x 0
          z : L
          x : A
          hx : Ne x 0
          int : IsIntegral A (HSMul.hSMul x z)
          ⊢ Eq (HMul.hMul z ((algebraMap C L) ↑{ fst := IsIntegralClosure.mk' C (HSMul.h …
        -/
        rw [algebraMap_mk', ← IsScalarTower.algebraMap_apply A C L, Algebra.smul_def, mul_comm]⟩
        /-
          🎉 no goals
        -/
                                          /-
                                            A : Type u_3
                                            inst✝⁹ : CommRing A
                                            L : Type u_5
                                            inst✝⁸ : Field L
                                            inst✝⁷ : Algebra A L
                                            C : Type u_6
                                            inst✝⁶ : CommRing C
                                            inst✝⁵ : IsDomain C
                                            inst✝⁴ : Algebra C L
                                            inst✝³ : IsIntegralClosure C A L
                                            inst✝² : Algebra A C
                                            inst✝¹ : IsScalarTower A C L
                                            inst✝ : Algebra.IsAlgebraic A L
                                            inj : ∀ (x : A), Eq ((algebraMap A L) x) 0 → Eq x 0
                                            x y : C
                                            h : Eq ((algebraMap C L) x) ((algebraMap C L) y)
                                            ⊢ Eq (HMul.hMul (↑1) x) (HMul.hMul (↑1) y)
                                          -/
    exists_of_eq := fun {x y} h => ⟨1, by simpa using algebraMap_injective C A L h⟩ }
                                          /-
                                            🎉 no goals
                                          -/


/-- If the field `L` is a finite extension of the fraction field of the integral domain `A`,
the integral closure `C` of `A` in `L` has fraction field `L`. -/
theorem isFractionRing_of_finite_extension [IsDomain A] [Algebra K L] [IsScalarTower A K L]
    [FiniteDimensional K L] : IsFractionRing C L :=
  have : Algebra.IsAlgebraic A L := IsFractionRing.comap_isAlgebraic_iff.mpr
    (inferInstanceAs (Algebra.IsAlgebraic K L))
  isFractionRing_of_algebraic A C
    fun _ hx =>
    IsFractionRing.to_map_eq_zero_iff.mp
      ((map_eq_zero <| algebraMap K L).mp <| (IsScalarTower.algebraMap_apply _ _ _ _).symm.trans hx)


/-- If the field `L` is an algebraic extension of the integral domain `A`,
the integral closure of `A` in `L` has fraction field `L`. -/
theorem isFractionRing_of_algebraic [Algebra A L] [Algebra.IsAlgebraic A L]
    (inj : ∀ x, algebraMap A L x = 0 → x = 0) : IsFractionRing (integralClosure A L) L :=
  IsIntegralClosure.isFractionRing_of_algebraic A (integralClosure A L) inj


/-- If the field `L` is a finite extension of the fraction field of the integral domain `A`,
the integral closure of `A` in `L` has fraction field `L`. -/
theorem isFractionRing_of_finite_extension [IsDomain A] [Algebra A L] [Algebra K L]
    [IsScalarTower A K L] [FiniteDimensional K L] : IsFractionRing (integralClosure A L) L :=
  IsIntegralClosure.isFractionRing_of_finite_extension A K L (integralClosure A L)


/-- `S` is algebraic over `R` iff a fraction ring of `S` is algebraic over `R` -/
theorem isAlgebraic_iff' [Field K] [IsDomain R] [IsDomain S] [Algebra R K] [Algebra S K]
    [NoZeroSMulDivisors R K] [IsFractionRing S K] [IsScalarTower R S K] :
    Algebra.IsAlgebraic R S ↔ Algebra.IsAlgebraic R K := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    S : Type u_2
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    K : Type u_4
    inst✝⁷ : Field K
    inst✝⁶ : IsDomain R
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R K
    inst✝³ : Algebra S K
    inst✝² : NoZeroSMulDivisors R K
    inst✝¹ : IsFractionRing S K
    inst✝ : IsScalarTower R S K
    ⊢ Iff (Algebra.IsAlgebraic R S) (Algebra.IsAlgebraic R K)
  -/
  simp only [Algebra.isAlgebraic_def]
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    S : Type u_2
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    K : Type u_4
    inst✝⁷ : Field K
    inst✝⁶ : IsDomain R
    inst✝⁵ : IsDomain S
    inst✝⁴ : Algebra R K
    inst✝³ : Algebra S K
    inst✝² : NoZeroSMulDivisors R K
    inst✝¹ : IsFractionRing S K
    inst✝ : IsScalarTower R S K
    ⊢ Iff (∀ (x : S), IsAlgebraic R x) (∀ (x : K), IsAlgebraic R x)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      ⊢ (∀ (x : S), IsAlgebraic R x) → ∀ (x : K), IsAlgebraic R x
    -/
  · intro h x
    /-
      case mp
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      x : K
      ⊢ IsAlgebraic R x
    -/
    letI := FractionRing.liftAlgebra R K
    /-
      case mp
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      x : K
      this : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
      ⊢ IsAlgebraic R x
    -/
    have := FractionRing.isScalarTower_liftAlgebra R K
    /-
      case mp
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      x : K
      this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
      this : IsScalarTower R (FractionRing R) K
      ⊢ IsAlgebraic R x
    -/
    rw [IsFractionRing.isAlgebraic_iff R (FractionRing R) K, isAlgebraic_iff_isIntegral]
    /-
      case mp
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      x : K
      this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
      this : IsScalarTower R (FractionRing R) K
      ⊢ IsIntegral (FractionRing R) x
    -/
    obtain ⟨a : S, b, ha, rfl⟩ := div_surjective (A := S) x
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
      this : IsScalarTower R (FractionRing R) K
      a b : S
      ha : Membership.mem (nonZeroDivisors S) b
      ⊢ IsIntegral (FractionRing R) (HDiv.hDiv ((algebraMap S K) a) ((algebraMap S K …
    -/
    obtain ⟨f, hf₁, hf₂⟩ := h b
    /-
      case mp.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
      this : IsScalarTower R (FractionRing R) K
      a b : S
      ha : Membership.mem (nonZeroDivisors S) b
      f : Polynomial R
      hf₁ : Ne f 0
      hf₂ : Eq ((Polynomial.aeval b) f) 0
      ⊢ IsIntegral (FractionRing R) (HDiv.hDiv ((algebraMap S K) a) ((algebraMap S K …
    -/
    rw [div_eq_mul_inv]
    /-
      case mp.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : S), IsAlgebraic R x
      this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
      this : IsScalarTower R (FractionRing R) K
      a b : S
      ha : Membership.mem (nonZeroDivisors S) b
      f : Polynomial R
      hf₁ : Ne f 0
      hf₂ : Eq ((Polynomial.aeval b) f) 0
      ⊢ IsIntegral (FractionRing R) (HMul.hMul ((algebraMap S K) a) (Inv.inv ((algeb …
    -/
    refine IsIntegral.mul ?_ ?_
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        R : Type u_1
        inst✝¹⁰ : CommRing R
        S : Type u_2
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        K : Type u_4
        inst✝⁷ : Field K
        inst✝⁶ : IsDomain R
        inst✝⁵ : IsDomain S
        inst✝⁴ : Algebra R K
        inst✝³ : Algebra S K
        inst✝² : NoZeroSMulDivisors R K
        inst✝¹ : IsFractionRing S K
        inst✝ : IsScalarTower R S K
        h : ∀ (x : S), IsAlgebraic R x
        this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
        this : IsScalarTower R (FractionRing R) K
        a b : S
        ha : Membership.mem (nonZeroDivisors S) b
        f : Polynomial R
        hf₁ : Ne f 0
        hf₂ : Eq ((Polynomial.aeval b) f) 0
        ⊢ IsIntegral (FractionRing R) ((algebraMap S K) a)
      -/
    · rw [← isAlgebraic_iff_isIntegral]
      refine .extendScalars
        (NoZeroSMulDivisors.algebraMap_injective R (FractionRing R)) ?_
      /-
        case mp.intro.intro.intro.intro.intro.refine_1
        R : Type u_1
        inst✝¹⁰ : CommRing R
        S : Type u_2
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        K : Type u_4
        inst✝⁷ : Field K
        inst✝⁶ : IsDomain R
        inst✝⁵ : IsDomain S
        inst✝⁴ : Algebra R K
        inst✝³ : Algebra S K
        inst✝² : NoZeroSMulDivisors R K
        inst✝¹ : IsFractionRing S K
        inst✝ : IsScalarTower R S K
        h : ∀ (x : S), IsAlgebraic R x
        this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
        this : IsScalarTower R (FractionRing R) K
        a b : S
        ha : Membership.mem (nonZeroDivisors S) b
        f : Polynomial R
        hf₁ : Ne f 0
        hf₂ : Eq ((Polynomial.aeval b) f) 0
        ⊢ IsAlgebraic R ((algebraMap S K) a)
      -/
      exact .algebraMap (h a)
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.refine_2
        R : Type u_1
        inst✝¹⁰ : CommRing R
        S : Type u_2
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        K : Type u_4
        inst✝⁷ : Field K
        inst✝⁶ : IsDomain R
        inst✝⁵ : IsDomain S
        inst✝⁴ : Algebra R K
        inst✝³ : Algebra S K
        inst✝² : NoZeroSMulDivisors R K
        inst✝¹ : IsFractionRing S K
        inst✝ : IsScalarTower R S K
        h : ∀ (x : S), IsAlgebraic R x
        this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
        this : IsScalarTower R (FractionRing R) K
        a b : S
        ha : Membership.mem (nonZeroDivisors S) b
        f : Polynomial R
        hf₁ : Ne f 0
        hf₂ : Eq ((Polynomial.aeval b) f) 0
        ⊢ IsIntegral (FractionRing R) (Inv.inv ((algebraMap S K) b))
      -/
    · rw [← isAlgebraic_iff_isIntegral]
      /-
        case mp.intro.intro.intro.intro.intro.refine_2
        R : Type u_1
        inst✝¹⁰ : CommRing R
        S : Type u_2
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        K : Type u_4
        inst✝⁷ : Field K
        inst✝⁶ : IsDomain R
        inst✝⁵ : IsDomain S
        inst✝⁴ : Algebra R K
        inst✝³ : Algebra S K
        inst✝² : NoZeroSMulDivisors R K
        inst✝¹ : IsFractionRing S K
        inst✝ : IsScalarTower R S K
        h : ∀ (x : S), IsAlgebraic R x
        this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
        this : IsScalarTower R (FractionRing R) K
        a b : S
        ha : Membership.mem (nonZeroDivisors S) b
        f : Polynomial R
        hf₁ : Ne f 0
        hf₂ : Eq ((Polynomial.aeval b) f) 0
        ⊢ IsAlgebraic (FractionRing R) (Inv.inv ((algebraMap S K) b))
      -/
      use (f.map (algebraMap R (FractionRing R))).reverse
      /-
        case h
        R : Type u_1
        inst✝¹⁰ : CommRing R
        S : Type u_2
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        K : Type u_4
        inst✝⁷ : Field K
        inst✝⁶ : IsDomain R
        inst✝⁵ : IsDomain S
        inst✝⁴ : Algebra R K
        inst✝³ : Algebra S K
        inst✝² : NoZeroSMulDivisors R K
        inst✝¹ : IsFractionRing S K
        inst✝ : IsScalarTower R S K
        h : ∀ (x : S), IsAlgebraic R x
        this✝ : Algebra (FractionRing R) K := FractionRing.liftAlgebra R K
        this : IsScalarTower R (FractionRing R) K
        a b : S
        ha : Membership.mem (nonZeroDivisors S) b
        f : Polynomial R
        hf₁ : Ne f 0
        hf₂ : Eq ((Polynomial.aeval b) f) 0
        ⊢ And (Ne (Polynomial.map (algebraMap R (FractionRing R)) f).reverse 0) (Eq (( …
      -/
      constructor
      · rwa [Ne, Polynomial.reverse_eq_zero, ← Polynomial.degree_eq_bot,
          Polynomial.degree_map_eq_of_injective
            (NoZeroSMulDivisors.algebraMap_injective R (FractionRing R)),
          Polynomial.degree_eq_bot]
      · have : Invertible (algebraMap S K b) :=
          IsUnit.invertible
            (isUnit_of_mem_nonZeroDivisors
              (mem_nonZeroDivisors_iff_ne_zero.2 fun h =>
                nonZeroDivisors.ne_zero ha
                  ((injective_iff_map_eq_zero (algebraMap S K)).1
                    (NoZeroSMulDivisors.algebraMap_injective _ _) b h)))
        rw [Polynomial.aeval_def, ← invOf_eq_inv, Polynomial.eval₂_reverse_eq_zero_iff,
          Polynomial.eval₂_map, ← IsScalarTower.algebraMap_eq, ← Polynomial.aeval_def,
          Polynomial.aeval_algebraMap_apply, hf₂, RingHom.map_zero]
    /-
      case mpr
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      ⊢ (∀ (x : K), IsAlgebraic R x) → ∀ (x : S), IsAlgebraic R x
    -/
  · intro h x
    /-
      case mpr
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : K), IsAlgebraic R x
      x : S
      ⊢ IsAlgebraic R x
    -/
    obtain ⟨f, hf₁, hf₂⟩ := h (algebraMap S K x)
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : K), IsAlgebraic R x
      x : S
      f : Polynomial R
      hf₁ : Ne f 0
      hf₂ : Eq ((Polynomial.aeval ((algebraMap S K) x)) f) 0
      ⊢ IsAlgebraic R x
    -/
    use f, hf₁
    /-
      case right
      R : Type u_1
      inst✝¹⁰ : CommRing R
      S : Type u_2
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      K : Type u_4
      inst✝⁷ : Field K
      inst✝⁶ : IsDomain R
      inst✝⁵ : IsDomain S
      inst✝⁴ : Algebra R K
      inst✝³ : Algebra S K
      inst✝² : NoZeroSMulDivisors R K
      inst✝¹ : IsFractionRing S K
      inst✝ : IsScalarTower R S K
      h : ∀ (x : K), IsAlgebraic R x
      x : S
      f : Polynomial R
      hf₁ : Ne f 0
      hf₂ : Eq ((Polynomial.aeval ((algebraMap S K) x)) f) 0
      ⊢ Eq ((Polynomial.aeval x) f) 0
    -/
    rw [Polynomial.aeval_algebraMap_apply] at hf₂
    exact
      (injective_iff_map_eq_zero (algebraMap S K)).1 (NoZeroSMulDivisors.algebraMap_injective _ _) _
        hf₂


/-- If the `S`-multiples of `a` are contained in some `R`-span, then `Frac(S)`-multiples of `a`
are contained in the equivalent `Frac(R)`-span. -/
theorem ideal_span_singleton_map_subset {L : Type*} [IsDomain R] [IsDomain S] [Field K] [Field L]
    [Algebra R K] [Algebra R L] [Algebra S L] [Algebra.IsAlgebraic R S] [IsFractionRing S L]
    [Algebra K L] [IsScalarTower R S L] [IsScalarTower R K L] {a : S} {b : Set S}
    (inj : Function.Injective (algebraMap R L))
    (h : (Ideal.span ({a} : Set S) : Set S) ⊆ Submodule.span R b) :
    (Ideal.span ({algebraMap S L a} : Set L) : Set L) ⊆ Submodule.span K (algebraMap S L '' b) := by
  /-
    R : Type u_1
    inst✝¹⁴ : CommRing R
    S : Type u_2
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : IsDomain R
    inst✝¹⁰ : IsDomain S
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra.IsAlgebraic R S
    inst✝³ : IsFractionRing S L
    inst✝² : Algebra K L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    a : S
    b : Set S
    inj : Function.Injective ⇑(algebraMap R L)
    h : HasSubset.Subset ↑(Ideal.span (Singleton.singleton a)) ↑(Submodule.span R b)
    ⊢ HasSubset.Subset ↑(Ideal.span (Singleton.singleton ((algebraMap S L) a))) ↑( …
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝¹⁴ : CommRing R
    S : Type u_2
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : IsDomain R
    inst✝¹⁰ : IsDomain S
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra.IsAlgebraic R S
    inst✝³ : IsFractionRing S L
    inst✝² : Algebra K L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    a : S
    b : Set S
    inj : Function.Injective ⇑(algebraMap R L)
    h : HasSubset.Subset ↑(Ideal.span (Singleton.singleton a)) ↑(Submodule.span R b)
    x : L
    hx : Membership.mem (↑(Ideal.span (Singleton.singleton ((algebraMap S L) a)))) x
    ⊢ Membership.mem (↑(Submodule.span K (Set.image (⇑(algebraMap S L)) b))) x
  -/
  obtain ⟨x', rfl⟩ := Ideal.mem_span_singleton.mp hx
  /-
    case intro
    R : Type u_1
    inst✝¹⁴ : CommRing R
    S : Type u_2
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : IsDomain R
    inst✝¹⁰ : IsDomain S
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra.IsAlgebraic R S
    inst✝³ : IsFractionRing S L
    inst✝² : Algebra K L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    a : S
    b : Set S
    inj : Function.Injective ⇑(algebraMap R L)
    h : HasSubset.Subset ↑(Ideal.span (Singleton.singleton a)) ↑(Submodule.span R b)
    x' : L
    hx : Membership.mem (↑(Ideal.span (Singleton.singleton ((algebraMap S L) a)))) …
    ⊢ Membership.mem (↑(Submodule.span K (Set.image (⇑(algebraMap S L)) b))) (HMul …
  -/
  obtain ⟨y', z', rfl⟩ := IsLocalization.mk'_surjective S⁰ x'
  obtain ⟨y, z, hz0, yz_eq⟩ :=
    Algebra.IsAlgebraic.exists_smul_eq_mul R y' (nonZeroDivisors.coe_ne_zero z')
  have injRS : Function.Injective (algebraMap R S) := by
    refine
      Function.Injective.of_comp (show Function.Injective (algebraMap S L ∘ algebraMap R S) from ?_)
    rwa [← RingHom.coe_comp, ← IsScalarTower.algebraMap_eq]
  have hz0' : algebraMap R S z ∈ S⁰ :=
    map_mem_nonZeroDivisors (algebraMap R S) injRS (mem_nonZeroDivisors_of_ne_zero hz0)
  have mk_yz_eq : IsLocalization.mk' L y' z' = IsLocalization.mk' L y ⟨_, hz0'⟩ := by
    rw [Algebra.smul_def, mul_comm _ y, mul_comm _ y'] at yz_eq
    exact IsLocalization.mk'_eq_of_eq (by rw [mul_comm _ y, mul_comm _ y', yz_eq])
  suffices hy : algebraMap S L (a * y) ∈ Submodule.span K ((algebraMap S L) '' b) by
    rw [mk_yz_eq, IsFractionRing.mk'_eq_div, ← IsScalarTower.algebraMap_apply,
      IsScalarTower.algebraMap_apply R K L, div_eq_mul_inv, ← mul_assoc, mul_comm, ← map_inv₀, ←
      Algebra.smul_def, ← _root_.map_mul]
    exact (Submodule.span K _).smul_mem _ hy
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹⁴ : CommRing R
    S : Type u_2
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : IsDomain R
    inst✝¹⁰ : IsDomain S
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra.IsAlgebraic R S
    inst✝³ : IsFractionRing S L
    inst✝² : Algebra K L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    a : S
    b : Set S
    inj : Function.Injective ⇑(algebraMap R L)
    h : HasSubset.Subset ↑(Ideal.span (Singleton.singleton a)) ↑(Submodule.span R b)
    y' : S
    z' : Subtype fun x => Membership.mem (nonZeroDivisors S) x
    hx : Membership.mem (↑(Ideal.span (Singleton.singleton ((algebraMap S L) a)))) …
    y : S
    z : R
    hz0 : Ne z 0
    yz_eq : Eq (HSMul.hSMul z y') (HMul.hMul (↑z') y)
    injRS : Function.Injective ⇑(algebraMap R S)
    hz0' : Membership.mem (nonZeroDivisors S) ((algebraMap R S) z)
    mk_yz_eq : Eq (IsLocalization.mk' L y' z') (IsLocalization.mk' L y ⟨(algebraMa …
    ⊢ Membership.mem (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) ((algebr …
  -/
  refine Submodule.span_subset_span R K _ ?_
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹⁴ : CommRing R
    S : Type u_2
    inst✝¹³ : CommRing S
    inst✝¹² : Algebra R S
    K : Type u_4
    L : Type u_5
    inst✝¹¹ : IsDomain R
    inst✝¹⁰ : IsDomain S
    inst✝⁹ : Field K
    inst✝⁸ : Field L
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra.IsAlgebraic R S
    inst✝³ : IsFractionRing S L
    inst✝² : Algebra K L
    inst✝¹ : IsScalarTower R S L
    inst✝ : IsScalarTower R K L
    a : S
    b : Set S
    inj : Function.Injective ⇑(algebraMap R L)
    h : HasSubset.Subset ↑(Ideal.span (Singleton.singleton a)) ↑(Submodule.span R b)
    y' : S
    z' : Subtype fun x => Membership.mem (nonZeroDivisors S) x
    hx : Membership.mem (↑(Ideal.span (Singleton.singleton ((algebraMap S L) a)))) …
    y : S
    z : R
    hz0 : Ne z 0
    yz_eq : Eq (HSMul.hSMul z y') (HMul.hMul (↑z') y)
    injRS : Function.Injective ⇑(algebraMap R S)
    hz0' : Membership.mem (nonZeroDivisors S) ((algebraMap R S) z)
    mk_yz_eq : Eq (IsLocalization.mk' L y' z') (IsLocalization.mk' L y ⟨(algebraMa …
    ⊢ Membership.mem (↑(Submodule.span R (Set.image (⇑(algebraMap S L)) b))) ((alg …
  -/
  rw [Submodule.span_algebraMap_image_of_tower]
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specify the value of `f` here:
  exact Submodule.mem_map_of_mem (f := LinearMap.restrictScalars _ _)
    (h (Ideal.mem_span_singleton.mpr ⟨y, rfl⟩))


lemma isAlgebraic_of_isLocalization {R} [CommRing R] (M : Submonoid R) (S) [CommRing S]
    [Nontrivial R] [Algebra R S] [IsLocalization M S] : Algebra.IsAlgebraic R S := by
  /-
    R : Type u_5
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_6
    inst✝³ : CommRing S
    inst✝² : Nontrivial R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ Algebra.IsAlgebraic R S
  -/
  constructor
  /-
    case isAlgebraic
    R : Type u_5
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_6
    inst✝³ : CommRing S
    inst✝² : Nontrivial R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ ∀ (x : S), IsAlgebraic R x
  -/
  intro x
  /-
    case isAlgebraic
    R : Type u_5
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_6
    inst✝³ : CommRing S
    inst✝² : Nontrivial R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : S
    ⊢ IsAlgebraic R x
  -/
  obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective M x
  /-
    case isAlgebraic.intro.intro
    R : Type u_5
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_6
    inst✝³ : CommRing S
    inst✝² : Nontrivial R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    s : Subtype fun x => Membership.mem M x
    ⊢ IsAlgebraic R (IsLocalization.mk' S x s)
  -/
  by_cases hs : (s : R) = 0
    /-
      case pos
      R : Type u_5
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_6
      inst✝³ : CommRing S
      inst✝² : Nontrivial R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      s : Subtype fun x => Membership.mem M x
      hs : Eq (↑s) 0
      ⊢ IsAlgebraic R (IsLocalization.mk' S x s)
    -/
  · have := IsLocalization.mk'_spec S x s
    /-
      case pos
      R : Type u_5
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_6
      inst✝³ : CommRing S
      inst✝² : Nontrivial R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      s : Subtype fun x => Membership.mem M x
      hs : Eq (↑s) 0
      this : Eq (HMul.hMul (IsLocalization.mk' S x s) ((algebraMap R S) ↑s)) ((algeb …
      ⊢ IsAlgebraic R (IsLocalization.mk' S x s)
    -/
    rw [hs, map_zero, mul_zero] at this
    /-
      case pos
      R : Type u_5
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_6
      inst✝³ : CommRing S
      inst✝² : Nontrivial R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      s : Subtype fun x => Membership.mem M x
      hs : Eq (↑s) 0
      this : Eq 0 ((algebraMap R S) x)
      ⊢ IsAlgebraic R (IsLocalization.mk' S x s)
    -/
    exact ⟨X, X_ne_zero, by simp [IsLocalization.mk'_eq_mul_mk'_one x, ← this]⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_5
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_6
    inst✝³ : CommRing S
    inst✝² : Nontrivial R
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    s : Subtype fun x => Membership.mem M x
    hs : Not (Eq (↑s) 0)
    ⊢ IsAlgebraic R (IsLocalization.mk' S x s)
  -/
  refine ⟨s • X - C x, ?_, ?_⟩
    /-
      case neg.refine_1
      R : Type u_5
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_6
      inst✝³ : CommRing S
      inst✝² : Nontrivial R
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      s : Subtype fun x => Membership.mem M x
      hs : Not (Eq (↑s) 0)
      ⊢ Ne (HSub.hSub (HSMul.hSMul s Polynomial.X) (Polynomial.C x)) 0
    -/
  · intro e; apply hs
    simpa only [coeff_sub, coeff_smul, coeff_X_one, coeff_C_succ, sub_zero, coeff_zero,
      ← Algebra.algebraMap_eq_smul_one, Submonoid.smul_def,
      Algebra.id.map_eq_id, RingHom.id_apply] using congr_arg (Polynomial.coeff · 1) e
  · simp only [map_sub, Algebra.smul_def, Submonoid.smul_def,
      map_mul, AlgHom.commutes, aeval_X, IsLocalization.mk'_spec', aeval_C, sub_self]


open nonZeroDivisors in
lemma isAlgebraic_of_isFractionRing {R S} (K L) [CommRing R] [CommRing S] [Field K] [CommRing L]
    [Algebra R S] [Algebra R K] [Algebra R L] [Algebra S L] [Algebra K L] [IsScalarTower R S L]
    [IsScalarTower R K L] [IsFractionRing S L]
    [Algebra.IsIntegral R S] : Algebra.IsAlgebraic K L := by
  /-
    R : Type u_5
    S : Type u_6
    K : Type u_7
    L : Type u_8
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Field K
    inst✝⁹ : CommRing L
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsFractionRing S L
    inst✝ : Algebra.IsIntegral R S
    ⊢ Algebra.IsAlgebraic K L
  -/
  constructor
  /-
    case isAlgebraic
    R : Type u_5
    S : Type u_6
    K : Type u_7
    L : Type u_8
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Field K
    inst✝⁹ : CommRing L
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsFractionRing S L
    inst✝ : Algebra.IsIntegral R S
    ⊢ ∀ (x : L), IsAlgebraic K x
  -/
  intro x
  /-
    case isAlgebraic
    R : Type u_5
    S : Type u_6
    K : Type u_7
    L : Type u_8
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Field K
    inst✝⁹ : CommRing L
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsFractionRing S L
    inst✝ : Algebra.IsIntegral R S
    x : L
    ⊢ IsAlgebraic K x
  -/
  obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective S⁰ x
  /-
    case isAlgebraic.intro.intro
    R : Type u_5
    S : Type u_6
    K : Type u_7
    L : Type u_8
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Field K
    inst✝⁹ : CommRing L
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsFractionRing S L
    inst✝ : Algebra.IsIntegral R S
    x : S
    s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
    ⊢ IsAlgebraic K (IsLocalization.mk' L x s)
  -/
  apply IsIntegral.isAlgebraic
  /-
    case isAlgebraic.intro.intro.a
    R : Type u_5
    S : Type u_6
    K : Type u_7
    L : Type u_8
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Field K
    inst✝⁹ : CommRing L
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsFractionRing S L
    inst✝ : Algebra.IsIntegral R S
    x : S
    s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
    ⊢ IsIntegral K (IsLocalization.mk' L x s)
  -/
  rw [IsLocalization.mk'_eq_mul_mk'_one]
  /-
    case isAlgebraic.intro.intro.a
    R : Type u_5
    S : Type u_6
    K : Type u_7
    L : Type u_8
    inst✝¹² : CommRing R
    inst✝¹¹ : CommRing S
    inst✝¹⁰ : Field K
    inst✝⁹ : CommRing L
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra R K
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsFractionRing S L
    inst✝ : Algebra.IsIntegral R S
    x : S
    s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
    ⊢ IsIntegral K (HMul.hMul ((algebraMap S L) x) (IsLocalization.mk' L 1 s))
  -/
  apply RingHom.IsIntegralElem.mul
    /-
      case isAlgebraic.intro.intro.a.hx
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ (algebraMap K L).IsIntegralElem ((algebraMap S L) x)
    -/
  · apply IsIntegral.tower_top (R := R)
    /-
      case isAlgebraic.intro.intro.a.hx
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ IsIntegral R ((algebraMap S L) x)
    -/
    apply IsIntegral.map (IsScalarTower.toAlgHom R S L)
    /-
      case isAlgebraic.intro.intro.a.hx
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ IsIntegral R x
    -/
    exact Algebra.IsIntegral.isIntegral x
    /-
      🎉 no goals
    -/
    /-
      case isAlgebraic.intro.intro.a.hy
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ (algebraMap K L).IsIntegralElem (IsLocalization.mk' L 1 s)
    -/
  · show IsIntegral _ _
    /-
      case isAlgebraic.intro.intro.a.hy
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ IsIntegral K (IsLocalization.mk' L 1 s)
    -/
    rw [← isAlgebraic_iff_isIntegral, ← IsAlgebraic.invOf_iff, isAlgebraic_iff_isIntegral]
    /-
      case isAlgebraic.intro.intro.a.hy
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ IsIntegral K (Invertible.invOf (IsLocalization.mk' L 1 s))
    -/
    apply IsIntegral.tower_top (R := R)
    /-
      case isAlgebraic.intro.intro.a.hy
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ IsIntegral R (Invertible.invOf (IsLocalization.mk' L 1 s))
    -/
    apply IsIntegral.map (IsScalarTower.toAlgHom R S L)
    /-
      case isAlgebraic.intro.intro.a.hy
      R : Type u_5
      S : Type u_6
      K : Type u_7
      L : Type u_8
      inst✝¹² : CommRing R
      inst✝¹¹ : CommRing S
      inst✝¹⁰ : Field K
      inst✝⁹ : CommRing L
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R K
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsFractionRing S L
      inst✝ : Algebra.IsIntegral R S
      x : S
      s : Subtype fun x => Membership.mem (nonZeroDivisors S) x
      ⊢ IsIntegral R ↑s
    -/
    exact Algebra.IsIntegral.isIntegral (s : S)
    /-
      🎉 no goals
    -/

