/-- An integral element of an algebra is algebraic. -/
theorem IsIntegral.isAlgebraic [Nontrivial R] {x : A} : IsIntegral R x → IsAlgebraic R x :=
  fun ⟨p, hp, hpx⟩ => ⟨p, hp.ne_zero, hpx⟩


instance Algebra.IsIntegral.isAlgebraic [Nontrivial R] [Algebra.IsIntegral R A] :
    Algebra.IsAlgebraic R A := ⟨fun a ↦ (Algebra.IsIntegral.isIntegral a).isAlgebraic⟩


/-- An element of an algebra over a field is algebraic if and only if it is integral. -/
theorem isAlgebraic_iff_isIntegral {x : A} : IsAlgebraic K x ↔ IsIntegral K x := by
  /-
    K : Type u
    A : Type v
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    x : A
    ⊢ Iff (IsAlgebraic K x) (IsIntegral K x)
  -/
  refine ⟨?_, IsIntegral.isAlgebraic⟩
  /-
    K : Type u
    A : Type v
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    x : A
    ⊢ IsAlgebraic K x → IsIntegral K x
  -/
  rintro ⟨p, hp, hpx⟩
  /-
    case intro.intro
    K : Type u
    A : Type v
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    x : A
    p : Polynomial K
    hp : Ne p 0
    hpx : Eq ((Polynomial.aeval x) p) 0
    ⊢ IsIntegral K x
  -/
  refine ⟨_, monic_mul_leadingCoeff_inv hp, ?_⟩
  /-
    case intro.intro
    K : Type u
    A : Type v
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    x : A
    p : Polynomial K
    hp : Ne p 0
    hpx : Eq ((Polynomial.aeval x) p) 0
    ⊢ Eq (Polynomial.eval₂ (algebraMap K A) x (HMul.hMul p (Polynomial.C (Inv.inv  …
  -/
  rw [← aeval_def, map_mul, hpx, zero_mul]
  /-
    🎉 no goals
  -/


protected theorem Algebra.isAlgebraic_iff_isIntegral :
    Algebra.IsAlgebraic K A ↔ Algebra.IsIntegral K A := by
  rw [Algebra.isAlgebraic_def, Algebra.isIntegral_def,
      forall_congr' fun _ ↦ isAlgebraic_iff_isIntegral]


alias ⟨IsAlgebraic.isIntegral, _⟩ := isAlgebraic_iff_isIntegral


/-- This used to be an `alias` of `Algebra.isAlgebraic_iff_isIntegral` but that would make
`Algebra.IsAlgebraic K A` an explicit parameter instead of instance implicit. -/
protected instance Algebra.IsAlgebraic.isIntegral [Algebra.IsAlgebraic K A] :
    Algebra.IsIntegral K A := Algebra.isAlgebraic_iff_isIntegral.mp ‹_›


theorem Algebra.IsAlgebraic.of_isIntegralClosure (R B C : Type*) [CommRing R] [Nontrivial R]
    [CommRing B] [CommRing C] [Algebra R B] [Algebra R C] [Algebra B C]
    [IsScalarTower R B C] [IsIntegralClosure B R C] : Algebra.IsAlgebraic R B :=
  have := IsIntegralClosure.isIntegral_algebra R (A := B) C
  inferInstance


theorem IsAlgebraic.of_finite (e : A) [FiniteDimensional K A] : IsAlgebraic K e :=
  (IsIntegral.of_finite K e).isAlgebraic


/-- A field extension is algebraic if it is finite. -/
@[stacks 09GG "first part"]
instance Algebra.IsAlgebraic.of_finite [FiniteDimensional K A] : Algebra.IsAlgebraic K A :=
  (IsIntegral.of_finite K A).isAlgebraic


/-- If L is an algebraic field extension of K and A is an algebraic algebra over L,
then A is algebraic over K. -/
@[stacks 09GJ]
protected theorem Algebra.IsAlgebraic.trans
    [L_alg : Algebra.IsAlgebraic K L] [A_alg : Algebra.IsAlgebraic L A] :
    Algebra.IsAlgebraic K A := by
  /-
    K : Type u_1
    L : Type u_2
    A : Type u_5
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Ring A
    inst✝³ : Algebra K L
    inst✝² : Algebra L A
    inst✝¹ : Algebra K A
    inst✝ : IsScalarTower K L A
    L_alg : Algebra.IsAlgebraic K L
    A_alg : Algebra.IsAlgebraic L A
    ⊢ Algebra.IsAlgebraic K A
  -/
  rw [Algebra.isAlgebraic_iff_isIntegral] at L_alg A_alg ⊢
  /-
    K : Type u_1
    L : Type u_2
    A : Type u_5
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : Ring A
    inst✝³ : Algebra K L
    inst✝² : Algebra L A
    inst✝¹ : Algebra K A
    inst✝ : IsScalarTower K L A
    L_alg : Algebra.IsIntegral K L
    A_alg : Algebra.IsIntegral L A
    ⊢ Algebra.IsIntegral K A
  -/
  exact Algebra.IsIntegral.trans L
  /-
    🎉 no goals
  -/


/-- If `K` is a field, `r : A` and `f : K[X]`, then `Polynomial.aeval r f` is
transcendental over `K` if and only if `r` and `f` are both transcendental over `K`.
See also `Transcendental.aeval_of_transcendental` and `Transcendental.of_aeval`. -/
@[simp]
theorem transcendental_aeval_iff {r : A} {f : K[X]} :
    Transcendental K (Polynomial.aeval r f) ↔ Transcendental K r ∧ Transcendental K f := by
  /-
    K : Type u_1
    A : Type u_5
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    r : A
    f : Polynomial K
    ⊢ Iff (Transcendental K ((Polynomial.aeval r) f)) (And (Transcendental K r) (T …
  -/
  refine ⟨fun h ↦ ⟨?_, h.of_aeval⟩, fun ⟨h1, h2⟩ ↦ h1.aeval_of_transcendental h2⟩
  /-
    K : Type u_1
    A : Type u_5
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    r : A
    f : Polynomial K
    h : Transcendental K ((Polynomial.aeval r) f)
    ⊢ Transcendental K r
  -/
  rw [Transcendental] at h ⊢
  /-
    K : Type u_1
    A : Type u_5
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    r : A
    f : Polynomial K
    h : Not (IsAlgebraic K ((Polynomial.aeval r) f))
    ⊢ Not (IsAlgebraic K r)
  -/
  contrapose! h
  /-
    K : Type u_1
    A : Type u_5
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    r : A
    f : Polynomial K
    h : IsAlgebraic K r
    ⊢ IsAlgebraic K ((Polynomial.aeval r) f)
  -/
  rw [isAlgebraic_iff_isIntegral] at h ⊢
  /-
    K : Type u_1
    A : Type u_5
    inst✝² : Field K
    inst✝¹ : Ring A
    inst✝ : Algebra K A
    r : A
    f : Polynomial K
    h : IsIntegral K r
    ⊢ IsIntegral K ((Polynomial.aeval r) f)
  -/
  exact .of_mem_of_fg _ h.fg_adjoin_singleton _ (aeval_mem_adjoin_singleton _ _)
  /-
    🎉 no goals
  -/


theorem AlgHom.bijective [FiniteDimensional K L] (ϕ : L →ₐ[K] L) : Function.Bijective ϕ :=
  (Algebra.IsAlgebraic.of_finite K L).algHom_bijective ϕ


variable (K L) in
/-- Bijection between algebra equivalences and algebra homomorphisms -/
noncomputable abbrev algEquivEquivAlgHom [FiniteDimensional K L] :
    (L ≃ₐ[K] L) ≃* (L →ₐ[K] L) :=
  Algebra.IsAlgebraic.algEquivEquivAlgHom K L


theorem exists_integral_multiple (hz : IsAlgebraic R z)
    (inj : Function.Injective (algebraMap R A)) :
    ∃ y ≠ (0 : R), IsIntegral R (y • z) := by
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    hz : IsAlgebraic R z
    inj : Function.Injective ⇑(algebraMap R A)
    ⊢ Exists fun y => And (Ne y 0) (IsIntegral R (HSMul.hSMul y z))
  -/
  have ⟨p, p_ne_zero, px⟩ := hz
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    hz : IsAlgebraic R z
    inj : Function.Injective ⇑(algebraMap R A)
    p : Polynomial R
    p_ne_zero : Ne p 0
    px : Eq ((Polynomial.aeval z) p) 0
    ⊢ Exists fun y => And (Ne y 0) (IsIntegral R (HSMul.hSMul y z))
  -/
  set a := p.leadingCoeff
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    hz : IsAlgebraic R z
    inj : Function.Injective ⇑(algebraMap R A)
    p : Polynomial R
    p_ne_zero : Ne p 0
    px : Eq ((Polynomial.aeval z) p) 0
    a : R := p.leadingCoeff
    ⊢ Exists fun y => And (Ne y 0) (IsIntegral R (HSMul.hSMul y z))
  -/
  have a_ne_zero : a ≠ 0 := mt Polynomial.leadingCoeff_eq_zero.mp p_ne_zero
  have x_integral : IsIntegral R (algebraMap R A a * z) :=
    ⟨p.integralNormalization, monic_integralNormalization p_ne_zero,
      integralNormalization_aeval_eq_zero px fun _ ↦ (map_eq_zero_iff _ inj).mp⟩
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    hz : IsAlgebraic R z
    inj : Function.Injective ⇑(algebraMap R A)
    p : Polynomial R
    p_ne_zero : Ne p 0
    px : Eq ((Polynomial.aeval z) p) 0
    a : R := p.leadingCoeff
    a_ne_zero : Ne a 0
    x_integral : IsIntegral R (HMul.hMul ((algebraMap R A) a) z)
    ⊢ Exists fun y => And (Ne y 0) (IsIntegral R (HSMul.hSMul y z))
  -/
  exact ⟨_, a_ne_zero, Algebra.smul_def a z ▸ x_integral⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias _root_.exists_integral_multiple := exists_integral_multiple


theorem _root_.Algebra.IsAlgebraic.exists_integral_multiples [NoZeroDivisors R]
    [alg : Algebra.IsAlgebraic R A] (inj : Function.Injective (algebraMap R A)) (s : Finset A) :
    ∃ y ≠ (0 : R), ∀ z ∈ s, IsIntegral R (y • z) := by
  /-
    R : Type u_1
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroDivisors R
    alg : Algebra.IsAlgebraic R A
    inj : Function.Injective ⇑(algebraMap R A)
    s : Finset A
    ⊢ Exists fun y => And (Ne y 0) (∀ (z : A), Membership.mem s z → IsIntegral R ( …
  -/
  have := Algebra.IsAlgebraic.nontrivial R A
  /-
    R : Type u_1
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroDivisors R
    alg : Algebra.IsAlgebraic R A
    inj : Function.Injective ⇑(algebraMap R A)
    s : Finset A
    this : Nontrivial R
    ⊢ Exists fun y => And (Ne y 0) (∀ (z : A), Membership.mem s z → IsIntegral R ( …
  -/
  choose r hr int using fun x ↦ (alg.1 x).exists_integral_multiple inj
  /-
    R : Type u_1
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroDivisors R
    alg : Algebra.IsAlgebraic R A
    inj : Function.Injective ⇑(algebraMap R A)
    s : Finset A
    this : Nontrivial R
    r : A → R
    hr : ∀ (x : A), Ne (r x) 0
    int : ∀ (x : A), IsIntegral R (HSMul.hSMul (r x) x)
    ⊢ Exists fun y => And (Ne y 0) (∀ (z : A), Membership.mem s z → IsIntegral R ( …
  -/
  refine ⟨∏ x ∈ s, r x, Finset.prod_ne_zero_iff.mpr fun _ _ ↦ hr _, fun _ h ↦ ?_⟩
  /-
    R : Type u_1
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroDivisors R
    alg : Algebra.IsAlgebraic R A
    inj : Function.Injective ⇑(algebraMap R A)
    s : Finset A
    this : Nontrivial R
    r : A → R
    hr : ∀ (x : A), Ne (r x) 0
    int : ∀ (x : A), IsIntegral R (HSMul.hSMul (r x) x)
    x✝ : A
    h : Membership.mem s x✝
    ⊢ IsIntegral R (HSMul.hSMul (s.prod fun x => r x) x✝)
  -/
  classical rw [← Finset.prod_erase_mul _ _ h, mul_smul]
  /-
    R : Type u_1
    A : Type u_3
    inst✝³ : CommRing R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroDivisors R
    alg : Algebra.IsAlgebraic R A
    inj : Function.Injective ⇑(algebraMap R A)
    s : Finset A
    this : Nontrivial R
    r : A → R
    hr : ∀ (x : A), Ne (r x) 0
    int : ∀ (x : A), IsIntegral R (HSMul.hSMul (r x) x)
    x✝ : A
    h : Membership.mem s x✝
    ⊢ IsIntegral R (HSMul.hSMul ((s.erase x✝).prod fun x => r x) (HSMul.hSMul (r x …
  -/
  exact (int _).smul _
  /-
    🎉 no goals
  -/


theorem of_smul_isIntegral {y : R} (hy : ¬ IsNilpotent y)
    (h : IsIntegral R (y • z)) : IsAlgebraic R z := by
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    y : R
    hy : Not (IsNilpotent y)
    h : IsIntegral R (HSMul.hSMul y z)
    ⊢ IsAlgebraic R z
  -/
  have ⟨p, monic, eval0⟩ := h
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    y : R
    hy : Not (IsNilpotent y)
    h : IsIntegral R (HSMul.hSMul y z)
    p : Polynomial R
    monic : p.Monic
    eval0 : Eq (Polynomial.eval₂ (algebraMap R A) (HSMul.hSMul y z) p) 0
    ⊢ IsAlgebraic R z
  -/
  refine ⟨p.comp (C y * X), fun h ↦ ?_, by simpa [aeval_comp, Algebra.smul_def] using eval0⟩
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    y : R
    hy : Not (IsNilpotent y)
    h✝ : IsIntegral R (HSMul.hSMul y z)
    p : Polynomial R
    monic : p.Monic
    eval0 : Eq (Polynomial.eval₂ (algebraMap R A) (HSMul.hSMul y z) p) 0
    h : Eq (p.comp (HMul.hMul (Polynomial.C y) Polynomial.X)) 0
    ⊢ False
  -/
  apply_fun (coeff · p.natDegree) at h
  /-
    R : Type u_1
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    z : A
    y : R
    hy : Not (IsNilpotent y)
    h✝ : IsIntegral R (HSMul.hSMul y z)
    p : Polynomial R
    monic : p.Monic
    eval0 : Eq (Polynomial.eval₂ (algebraMap R A) (HSMul.hSMul y z) p) 0
    h : Eq ((p.comp (HMul.hMul (Polynomial.C y) Polynomial.X)).coeff p.natDegree)  …
    ⊢ False
  -/
  have hy0 : y ≠ 0 := by rintro rfl; exact hy .zero
  rw [coeff_zero, ← mul_one p.natDegree, ← natDegree_C_mul_X y hy0,
    coeff_comp_degree_mul_degree, monic, one_mul, leadingCoeff_C_mul_X] at h
    /-
      R : Type u_1
      A : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      z : A
      y : R
      hy : Not (IsNilpotent y)
      h✝ : IsIntegral R (HSMul.hSMul y z)
      p : Polynomial R
      monic : p.Monic
      eval0 : Eq (Polynomial.eval₂ (algebraMap R A) (HSMul.hSMul y z) p) 0
      h : Eq (HPow.hPow y p.natDegree) 0
      hy0 : Ne y 0
      ⊢ False
    -/
  · exact hy ⟨_, h⟩
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      A : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      z : A
      y : R
      hy : Not (IsNilpotent y)
      h✝ : IsIntegral R (HSMul.hSMul y z)
      p : Polynomial R
      monic : p.Monic
      eval0 : Eq (Polynomial.eval₂ (algebraMap R A) (HSMul.hSMul y z) p) 0
      h : Eq ((p.comp (HMul.hMul (Polynomial.C y) Polynomial.X)).coeff (HMul.hMul p. …
      hy0 : Ne y 0
      ⊢ Ne (HMul.hMul (Polynomial.C y) Polynomial.X).natDegree 0
    -/
  · rw [natDegree_C_mul_X _ hy0]; rintro ⟨⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem of_smul {y : R} (hy : y ∈ nonZeroDivisors R)
    (h : IsAlgebraic R (y • z)) : IsAlgebraic R z :=
  have ⟨p, hp, eval0⟩ := h
                                                 /-
                                                   R : Type u_1
                                                   A : Type u_3
                                                   inst✝² : CommRing R
                                                   inst✝¹ : Ring A
                                                   inst✝ : Algebra R A
                                                   z : A
                                                   y : R
                                                   hy : Membership.mem (nonZeroDivisors R) y
                                                   h : IsAlgebraic R (HSMul.hSMul y z)
                                                   p : Polynomial R
                                                   hp : Ne p 0
                                                   eval0 : Eq ((Polynomial.aeval (HSMul.hSMul y z)) p) 0
                                                   ⊢ Eq ((Polynomial.aeval z) (p.comp (HMul.hMul (Polynomial.C y) Polynomial.X))) 0
                                                 -/
  ⟨_, mt (comp_C_mul_X_eq_zero_iff hy).mp hp, by simpa [aeval_comp, Algebra.smul_def] using eval0⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem iff_exists_smul_integral [IsReduced R] (inj : Function.Injective (algebraMap R A)) :
    IsAlgebraic R z ↔ ∃ y ≠ (0 : R), IsIntegral R (y • z) :=
  ⟨(exists_integral_multiple · inj), fun ⟨_, hy, int⟩ ↦
                           /-
                             R : Type u_1
                             A : Type u_3
                             inst✝³ : CommRing R
                             inst✝² : Ring A
                             inst✝¹ : Algebra R A
                             z : A
                             inst✝ : IsReduced R
                             inj : Function.Injective ⇑(algebraMap R A)
                             x✝ : Exists fun y => And (Ne y 0) (IsIntegral R (HSMul.hSMul y z))
                             w✝ : R
                             hy : Ne w✝ 0
                             int : IsIntegral R (HSMul.hSMul w✝ z)
                             ⊢ Not (IsNilpotent w✝)
                           -/
    of_smul_isIntegral (by rwa [isNilpotent_iff_eq_zero]) int⟩
                           /-
                             🎉 no goals
                           -/


theorem restrictScalars_of_isIntegral [int : Algebra.IsIntegral R S]
    {a : A} (h : IsAlgebraic S a) : IsAlgebraic R a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    ⊢ IsAlgebraic R a
  -/
  by_cases hRS : Function.Injective (algebraMap R S)
  on_goal 2 => exact (Algebra.isAlgebraic_of_not_injective
    fun h ↦ hRS <| .of_comp (IsScalarTower.algebraMap_eq R S A ▸ h)).1 _
  /-
    case pos
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    hRS : Function.Injective ⇑(algebraMap R S)
    ⊢ IsAlgebraic R a
  -/
  have := hRS.noZeroDivisors _ (map_zero _) (map_mul _)
  /-
    case pos
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    hRS : Function.Injective ⇑(algebraMap R S)
    this : NoZeroDivisors R
    ⊢ IsAlgebraic R a
  -/
  have ⟨s, hs, int_s⟩ := h.exists_integral_multiple inj
  /-
    case pos
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    hRS : Function.Injective ⇑(algebraMap R S)
    this : NoZeroDivisors R
    s : S
    hs : Ne s 0
    int_s : IsIntegral S (HSMul.hSMul s a)
    ⊢ IsAlgebraic R a
  -/
  cases subsingleton_or_nontrivial R
    /-
      case pos.inl
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : NoZeroDivisors S
      inj : Function.Injective ⇑(algebraMap S A)
      int : Algebra.IsIntegral R S
      a : A
      h : IsAlgebraic S a
      hRS : Function.Injective ⇑(algebraMap R S)
      this : NoZeroDivisors R
      s : S
      hs : Ne s 0
      int_s : IsIntegral S (HSMul.hSMul s a)
      h✝ : Subsingleton R
      ⊢ IsAlgebraic R a
    -/
  · have := Module.subsingleton R S
    /-
      case pos.inl
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : NoZeroDivisors S
      inj : Function.Injective ⇑(algebraMap S A)
      int : Algebra.IsIntegral R S
      a : A
      h : IsAlgebraic S a
      hRS : Function.Injective ⇑(algebraMap R S)
      this✝ : NoZeroDivisors R
      s : S
      hs : Ne s 0
      int_s : IsIntegral S (HSMul.hSMul s a)
      h✝ : Subsingleton R
      this : Subsingleton S
      ⊢ IsAlgebraic R a
    -/
    exact (is_transcendental_of_subsingleton _ _ h).elim
    /-
      🎉 no goals
    -/
  /-
    case pos.inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    hRS : Function.Injective ⇑(algebraMap R S)
    this : NoZeroDivisors R
    s : S
    hs : Ne s 0
    int_s : IsIntegral S (HSMul.hSMul s a)
    h✝ : Nontrivial R
    ⊢ IsAlgebraic R a
  -/
  have ⟨r, hr, _, e⟩ := (int.1 s).isAlgebraic.exists_nonzero_dvd (mem_nonZeroDivisors_of_ne_zero hs)
  /-
    case pos.inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    hRS : Function.Injective ⇑(algebraMap R S)
    this : NoZeroDivisors R
    s : S
    hs : Ne s 0
    int_s : IsIntegral S (HSMul.hSMul s a)
    h✝ : Nontrivial R
    r : R
    hr : Ne r 0
    w✝ : S
    e : Eq ((algebraMap R S) r) (HMul.hMul s w✝)
    ⊢ IsAlgebraic R a
  -/
  refine .of_smul_isIntegral (y := r) (by rwa [isNilpotent_iff_eq_zero]) ?_
  rw [Algebra.smul_def, IsScalarTower.algebraMap_apply R S,
    e, ← Algebra.smul_def, mul_comm, mul_smul]
  /-
    case pos.inr
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    int : Algebra.IsIntegral R S
    a : A
    h : IsAlgebraic S a
    hRS : Function.Injective ⇑(algebraMap R S)
    this : NoZeroDivisors R
    s : S
    hs : Ne s 0
    int_s : IsIntegral S (HSMul.hSMul s a)
    h✝ : Nontrivial R
    r : R
    hr : Ne r 0
    w✝ : S
    e : Eq ((algebraMap R S) r) (HMul.hMul s w✝)
    ⊢ IsIntegral R (HSMul.hSMul w✝ (HSMul.hSMul s a))
  -/
  exact isIntegral_trans _ (int_s.smul _)
  /-
    🎉 no goals
  -/


theorem restrictScalars [Algebra.IsAlgebraic R S]
    {a : A} (h : IsAlgebraic S a) : IsAlgebraic R a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    inst✝ : Algebra.IsAlgebraic R S
    a : A
    h : IsAlgebraic S a
    ⊢ IsAlgebraic R a
  -/
  have ⟨p, hp, eval0⟩ := h
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    inst✝ : Algebra.IsAlgebraic R S
    a : A
    h : IsAlgebraic S a
    p : Polynomial S
    hp : Ne p 0
    eval0 : Eq ((Polynomial.aeval a) p) 0
    ⊢ IsAlgebraic R a
  -/
  by_cases hRS : Function.Injective (algebraMap R S)
  on_goal 2 => exact (Algebra.isAlgebraic_of_not_injective
    fun h ↦ hRS <| .of_comp (IsScalarTower.algebraMap_eq R S A ▸ h)).1 _
  /-
    case pos
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    inst✝ : Algebra.IsAlgebraic R S
    a : A
    h : IsAlgebraic S a
    p : Polynomial S
    hp : Ne p 0
    eval0 : Eq ((Polynomial.aeval a) p) 0
    hRS : Function.Injective ⇑(algebraMap R S)
    ⊢ IsAlgebraic R a
  -/
  have := hRS.noZeroDivisors _ (map_zero _) (map_mul _)
  classical
  have ⟨r, hr, int⟩ := Algebra.IsAlgebraic.exists_integral_multiples hRS (p.support.image (coeff p))
  let p := (r • p).toSubring (integralClosure R S).toSubring fun s hs ↦ by
    obtain ⟨n, hn, rfl⟩ := mem_coeffs_iff.mp hs
    exact int _ (Finset.mem_image_of_mem _ <| support_smul _ _ hn)
  have : IsAlgebraic (integralClosure R S) a := by
    refine ⟨p, ?_, ?_⟩
    · have := NoZeroSMulDivisors.of_algebraMap_injective hRS
      simpa only [← Polynomial.map_ne_zero_iff (f := Subring.subtype _) Subtype.val_injective,
        p, map_toSubring, smul_ne_zero_iff] using And.intro hr hp
    rw [← eval_map_algebraMap, Subalgebra.algebraMap_eq, ← map_map, ← Subalgebra.toSubring_subtype,
      map_toSubring, eval_map_algebraMap, ← AlgHom.restrictScalars_apply R,
      map_smul, AlgHom.restrictScalars_apply, eval0, smul_zero]
  exact restrictScalars_of_isIntegral _ (by exact inj.comp Subtype.val_injective) this


theorem _root_.IsIntegral.trans_isAlgebraic [alg : Algebra.IsAlgebraic R S]
    {a : A} (h : IsIntegral S a) : IsAlgebraic R a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R S
    inst✝³ : Algebra R A
    inst✝² : Algebra S A
    inst✝¹ : IsScalarTower R S A
    inst✝ : NoZeroDivisors S
    inj : Function.Injective ⇑(algebraMap S A)
    alg : Algebra.IsAlgebraic R S
    a : A
    h : IsIntegral S a
    ⊢ IsAlgebraic R a
  -/
  cases subsingleton_or_nontrivial A
    /-
      case inl
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : NoZeroDivisors S
      inj : Function.Injective ⇑(algebraMap S A)
      alg : Algebra.IsAlgebraic R S
      a : A
      h : IsIntegral S a
      h✝ : Subsingleton A
      ⊢ IsAlgebraic R a
    -/
  · have := Algebra.IsAlgebraic.nontrivial R S
    /-
      case inl
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : NoZeroDivisors S
      inj : Function.Injective ⇑(algebraMap S A)
      alg : Algebra.IsAlgebraic R S
      a : A
      h : IsIntegral S a
      h✝ : Subsingleton A
      this : Nontrivial R
      ⊢ IsAlgebraic R a
    -/
    exact Subsingleton.elim a 0 ▸ isAlgebraic_zero
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : NoZeroDivisors S
      inj : Function.Injective ⇑(algebraMap S A)
      alg : Algebra.IsAlgebraic R S
      a : A
      h : IsIntegral S a
      h✝ : Nontrivial A
      ⊢ IsAlgebraic R a
    -/
  · have := Module.nontrivial S A
    /-
      case inr
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Ring A
      inst✝⁴ : Algebra R S
      inst✝³ : Algebra R A
      inst✝² : Algebra S A
      inst✝¹ : IsScalarTower R S A
      inst✝ : NoZeroDivisors S
      inj : Function.Injective ⇑(algebraMap S A)
      alg : Algebra.IsAlgebraic R S
      a : A
      h : IsIntegral S a
      h✝ : Nontrivial A
      this : Nontrivial S
      ⊢ IsAlgebraic R a
    -/
    exact h.isAlgebraic.restrictScalars _ inj
    /-
      🎉 no goals
    -/


protected lemma neg : IsAlgebraic R (-a) :=
  have ⟨p, h, eval0⟩ := ha
                                                                /-
                                                                  R : Type u_1
                                                                  S : Type u_2
                                                                  inst✝² : CommRing R
                                                                  inst✝¹ : CommRing S
                                                                  inst✝ : Algebra R S
                                                                  a : S
                                                                  ha : IsAlgebraic R a
                                                                  p : Polynomial R
                                                                  h : Ne p 0
                                                                  eval0 : Eq ((Polynomial.aeval a) p) 0
                                                                  ⊢ Eq ((Polynomial.aeval (Neg.neg a)) (Polynomial.algEquivAevalNegX p)) 0
                                                                -/
  ⟨algEquivAevalNegX p, EmbeddingLike.map_ne_zero_iff.mpr h, by simpa [← comp_eq_aeval, aeval_comp]⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


protected lemma smul (r : R) : IsAlgebraic R (r • a) :=
  have ⟨_, hp, eval0⟩ := ha
  ⟨_, scaleRoots_ne_zero hp r, Algebra.smul_def r a ▸ scaleRoots_aeval_eq_zero eval0⟩


protected lemma nsmul (n : ℕ) : IsAlgebraic R (n • a) :=
  Nat.cast_smul_eq_nsmul R n a ▸ ha.smul _


protected lemma zsmul (n : ℤ) : IsAlgebraic R (n • a) :=
  Int.cast_smul_eq_zsmul R n a ▸ ha.smul _


protected lemma mul : IsAlgebraic R (a * b) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    ⊢ IsAlgebraic R (HMul.hMul a b)
  -/
  refine (em _).elim (fun h ↦ ?_) fun h ↦ (Algebra.isAlgebraic_of_not_injective h).1 _
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ⊢ IsAlgebraic R (HMul.hMul a b)
  -/
  have ⟨ra, a0, int_a⟩ := ha.exists_integral_multiple h
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    ⊢ IsAlgebraic R (HMul.hMul a b)
  -/
  have ⟨rb, b0, int_b⟩ := hb.exists_integral_multiple h
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    rb : R
    b0 : Ne rb 0
    int_b : IsIntegral R (HSMul.hSMul rb b)
    ⊢ IsAlgebraic R (HMul.hMul a b)
  -/
  refine (IsAlgebraic.iff_exists_smul_integral h).mpr ⟨_, mul_ne_zero a0 b0, ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    rb : R
    b0 : Ne rb 0
    int_b : IsIntegral R (HSMul.hSMul rb b)
    ⊢ IsIntegral R (HSMul.hSMul (HMul.hMul ra rb) (HMul.hMul a b))
  -/
  simp_rw [Algebra.smul_def, map_mul, mul_mul_mul_comm, ← Algebra.smul_def]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    rb : R
    b0 : Ne rb 0
    int_b : IsIntegral R (HSMul.hSMul rb b)
    ⊢ IsIntegral R (HMul.hMul (HSMul.hSMul ra a) (HSMul.hSMul rb b))
  -/
  exact int_a.mul int_b
  /-
    🎉 no goals
  -/


protected lemma add : IsAlgebraic R (a + b) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    ⊢ IsAlgebraic R (HAdd.hAdd a b)
  -/
  refine (em _).elim (fun h ↦ ?_) fun h ↦ (Algebra.isAlgebraic_of_not_injective h).1 _
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ⊢ IsAlgebraic R (HAdd.hAdd a b)
  -/
  have ⟨ra, a0, int_a⟩ := ha.exists_integral_multiple h
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    ⊢ IsAlgebraic R (HAdd.hAdd a b)
  -/
  have ⟨rb, b0, int_b⟩ := hb.exists_integral_multiple h
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    rb : R
    b0 : Ne rb 0
    int_b : IsIntegral R (HSMul.hSMul rb b)
    ⊢ IsAlgebraic R (HAdd.hAdd a b)
  -/
  refine (IsAlgebraic.iff_exists_smul_integral h).mpr ⟨_, mul_ne_zero b0 a0, ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    rb : R
    b0 : Ne rb 0
    int_b : IsIntegral R (HSMul.hSMul rb b)
    ⊢ IsIntegral R (HSMul.hSMul (HMul.hMul rb ra) (HAdd.hAdd a b))
  -/
  rw [smul_add, mul_smul, mul_comm, mul_smul]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    nzd : NoZeroDivisors R
    a b : S
    ha : IsAlgebraic R a
    hb : IsAlgebraic R b
    h : Function.Injective ⇑(algebraMap R S)
    ra : R
    a0 : Ne ra 0
    int_a : IsIntegral R (HSMul.hSMul ra a)
    rb : R
    b0 : Ne rb 0
    int_b : IsIntegral R (HSMul.hSMul rb b)
    ⊢ IsIntegral R (HAdd.hAdd (HSMul.hSMul rb (HSMul.hSMul ra a)) (HSMul.hSMul ra  …
  -/
  exact (int_a.smul _).add (int_b.smul _)
  /-
    🎉 no goals
  -/


protected lemma sub : IsAlgebraic R (a - b) :=
  sub_eq_add_neg a b ▸ ha.add hb.neg


protected lemma pow (n : ℕ) : IsAlgebraic R (a ^ n) :=
  have := ha.nontrivial
  n.rec (pow_zero a ▸ isAlgebraic_one) fun _ h ↦ pow_succ a _ ▸ h.mul ha


/-- Transitivity of algebraicity for algebras over domains. -/
theorem IsAlgebraic.trans' [Algebra.IsAlgebraic R S] [alg : Algebra.IsAlgebraic S A] :
    Algebra.IsAlgebraic R A :=
  ⟨fun _ ↦ (alg.1 _).restrictScalars _ inj⟩


theorem IsIntegral.trans_isAlgebraic [Algebra.IsIntegral R S] [alg : Algebra.IsAlgebraic S A] :
    Algebra.IsAlgebraic R A :=
  ⟨fun _ ↦ (alg.1 _).restrictScalars_of_isIntegral _ inj⟩


theorem IsAlgebraic.trans_isIntegral [Algebra.IsAlgebraic R S] [int : Algebra.IsIntegral S A] :
    Algebra.IsAlgebraic R A :=
  ⟨fun _ ↦ (int.1 _).trans_isAlgebraic _ inj⟩


/-- If `R` is a domain and `S` is an arbitrary `R`-algebra, then the elements of `S`
that are algebraic over `R` form a subalgebra. -/
def Subalgebra.algebraicClosure [IsDomain R] : Subalgebra R S where
  carrier := {s | _root_.IsAlgebraic R s}
  mul_mem' ha hb := ha.mul hb
  add_mem' ha hb := ha.add hb
  algebraMap_mem' := isAlgebraic_algebraMap


theorem integralClosure_le_algebraicClosure [IsDomain R] :
    integralClosure R S ≤ Subalgebra.algebraicClosure R S :=
  fun _ ↦ IsIntegral.isAlgebraic


theorem Subalgebra.algebraicClosure_eq_integralClosure {K} [Field K] [Algebra K S] :
    algebraicClosure K S = integralClosure K S :=
  SetLike.ext fun _ ↦ isAlgebraic_iff_isIntegral


instance [IsDomain R] : Algebra.IsAlgebraic R (Subalgebra.algebraicClosure R S) :=
  (Subalgebra.isAlgebraic_iff _).mp fun _ ↦ id


theorem Algebra.isAlgebraic_adjoin_iff [IsDomain R] {s : Set S} :
    (adjoin R s).IsAlgebraic ↔ ∀ x ∈ s, IsAlgebraic R x :=
  Algebra.adjoin_le_iff (S := Subalgebra.algebraicClosure R S)


theorem Algebra.isAlgebraic_adjoin_of_nonempty [NoZeroDivisors R] {s : Set S} (hs : s.Nonempty) :
    (adjoin R s).IsAlgebraic ↔ ∀ x ∈ s, IsAlgebraic R x :=
  ⟨fun h x hx ↦ h _ (subset_adjoin hx), fun h ↦
    have ⟨x, hx⟩ := hs
    have := (isDomain_iff_noZeroDivisors_and_nontrivial _).mpr ⟨‹_›, (h x hx).nontrivial⟩
    isAlgebraic_adjoin_iff.mpr h⟩


theorem Algebra.isAlgebraic_adjoin_singleton_iff [NoZeroDivisors R] {s : S} :
    (adjoin R {s}).IsAlgebraic ↔ IsAlgebraic R s :=
  (isAlgebraic_adjoin_of_nonempty <| Set.singleton_nonempty s).trans forall_eq


theorem IsAlgebraic.of_mul [NoZeroDivisors R] {y z : S} (hy : y ∈ nonZeroDivisors S)
    (alg_y : IsAlgebraic R y) (alg_yz : IsAlgebraic R (y * z)) : IsAlgebraic R z := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroDivisors R
    y z : S
    hy : Membership.mem (nonZeroDivisors S) y
    alg_y : IsAlgebraic R y
    alg_yz : IsAlgebraic R (HMul.hMul y z)
    ⊢ IsAlgebraic R z
  -/
  have ⟨t, ht, r, hr, eq⟩ := alg_y.exists_nonzero_eq_adjoin_mul hy
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroDivisors R
    y z : S
    hy : Membership.mem (nonZeroDivisors S) y
    alg_y : IsAlgebraic R y
    alg_yz : IsAlgebraic R (HMul.hMul y z)
    t : S
    ht : Membership.mem (Algebra.adjoin R (Singleton.singleton y)) t
    r : R
    hr : Ne r 0
    eq : Eq (HMul.hMul y t) ((algebraMap R S) r)
    ⊢ IsAlgebraic R z
  -/
  have := alg_yz.mul (Algebra.isAlgebraic_adjoin_singleton_iff.mpr alg_y _ ht)
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroDivisors R
    y z : S
    hy : Membership.mem (nonZeroDivisors S) y
    alg_y : IsAlgebraic R y
    alg_yz : IsAlgebraic R (HMul.hMul y z)
    t : S
    ht : Membership.mem (Algebra.adjoin R (Singleton.singleton y)) t
    r : R
    hr : Ne r 0
    eq : Eq (HMul.hMul y t) ((algebraMap R S) r)
    this : IsAlgebraic R (HMul.hMul (HMul.hMul y z) t)
    ⊢ IsAlgebraic R z
  -/
  rw [mul_right_comm, eq, ← Algebra.smul_def] at this
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : NoZeroDivisors R
    y z : S
    hy : Membership.mem (nonZeroDivisors S) y
    alg_y : IsAlgebraic R y
    alg_yz : IsAlgebraic R (HMul.hMul y z)
    t : S
    ht : Membership.mem (Algebra.adjoin R (Singleton.singleton y)) t
    r : R
    hr : Ne r 0
    eq : Eq (HMul.hMul y t) ((algebraMap R S) r)
    this : IsAlgebraic R (HSMul.hSMul r z)
    ⊢ IsAlgebraic R z
  -/
  exact this.of_smul (mem_nonZeroDivisors_of_ne_zero hr)
  /-
    🎉 no goals
  -/


lemma extendScalars_of_isIntegral [NoZeroDivisors S] [Algebra.IsIntegral R S]
    (inj : Function.Injective (algebraMap S A)) : Transcendental S a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    a : A
    ha : Transcendental R a
    inst✝¹ : NoZeroDivisors S
    inst✝ : Algebra.IsIntegral R S
    inj : Function.Injective ⇑(algebraMap S A)
    ⊢ Transcendental S a
  -/
  contrapose ha
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    a : A
    inst✝¹ : NoZeroDivisors S
    inst✝ : Algebra.IsIntegral R S
    inj : Function.Injective ⇑(algebraMap S A)
    ha : Not (Transcendental S a)
    ⊢ Not (Transcendental R a)
  -/
  rw [Transcendental, not_not] at ha ⊢
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    a : A
    inst✝¹ : NoZeroDivisors S
    inst✝ : Algebra.IsIntegral R S
    inj : Function.Injective ⇑(algebraMap S A)
    ha : IsAlgebraic S a
    ⊢ IsAlgebraic R a
  -/
  exact ha.restrictScalars_of_isIntegral _ inj
  /-
    🎉 no goals
  -/


lemma extendScalars [NoZeroDivisors S] [Algebra.IsAlgebraic R S]
    (inj : Function.Injective (algebraMap S A)) : Transcendental S a := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    a : A
    ha : Transcendental R a
    inst✝¹ : NoZeroDivisors S
    inst✝ : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    ⊢ Transcendental S a
  -/
  contrapose ha
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    a : A
    inst✝¹ : NoZeroDivisors S
    inst✝ : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    ha : Not (Transcendental S a)
    ⊢ Not (Transcendental R a)
  -/
  rw [Transcendental, not_not] at ha ⊢
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Ring A
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    a : A
    inst✝¹ : NoZeroDivisors S
    inst✝ : Algebra.IsAlgebraic R S
    inj : Function.Injective ⇑(algebraMap S A)
    ha : IsAlgebraic S a
    ⊢ IsAlgebraic R a
  -/
  exact ha.restrictScalars _ inj
  /-
    🎉 no goals
  -/


protected lemma integralClosure [NoZeroDivisors S] :
    Transcendental (integralClosure R S) a :=
  ha.extendScalars_of_isIntegral Subtype.val_injective


lemma subalgebraAlgebraicClosure [IsDomain R] [NoZeroDivisors S] :
    Transcendental (Subalgebra.algebraicClosure R S) a :=
  ha.extendScalars Subtype.val_injective


