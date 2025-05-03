/-- The two numbers `c`, `d` in the "bottom_row" of `g=[[*,*],[c,d]]` in `SL(2, ℤ)` are coprime. -/
theorem bottom_row_coprime {R : Type*} [CommRing R] (g : SL(2, R)) :
    IsCoprime ((↑g : Matrix (Fin 2) (Fin 2) R) 1 0) ((↑g : Matrix (Fin 2) (Fin 2) R) 1 1) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    g : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ IsCoprime (↑g 1 0) (↑g 1 1)
  -/
  use -(↑g : Matrix (Fin 2) (Fin 2) R) 0 1, (↑g : Matrix (Fin 2) (Fin 2) R) 0 0
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    g : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (↑g 0 1)) (↑g 1 0)) (HMul.hMul (↑g 0 0) (↑ …
  -/
  rw [add_comm, neg_mul, ← sub_eq_add_neg, ← det_fin_two]
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    g : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ Eq (↑g).det 1
  -/
  exact g.det_coe
  /-
    🎉 no goals
  -/


/-- Every pair `![c, d]` of coprime integers is the "bottom_row" of some element `g=[[*,*],[c,d]]`
of `SL(2,ℤ)`. -/
theorem bottom_row_surj {R : Type*} [CommRing R] :
    Set.SurjOn (fun g : SL(2, R) => (↑g : Matrix (Fin 2) (Fin 2) R) 1) Set.univ
      {cd | IsCoprime (cd 0) (cd 1)} := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ Set.SurjOn (fun g => ↑g 1) Set.univ (setOf fun cd => IsCoprime (cd 0) (cd 1))
  -/
  rintro cd ⟨b₀, a, gcd_eqn⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    cd : Fin 2 → R
    b₀ a : R
    gcd_eqn : Eq (HAdd.hAdd (HMul.hMul b₀ (cd 0)) (HMul.hMul a (cd 1))) 1
    ⊢ Membership.mem (Set.image (fun g => ↑g 1) Set.univ) cd
  -/
  let A := of ![![a, -b₀], cd]
  have det_A_1 : det A = 1 := by
    convert gcd_eqn
    rw [det_fin_two]
    simp [A, (by ring : a * cd 1 + b₀ * cd 0 = b₀ * cd 0 + a * cd 1)]
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    cd : Fin 2 → R
    b₀ a : R
    gcd_eqn : Eq (HAdd.hAdd (HMul.hMul b₀ (cd 0)) (HMul.hMul a (cd 1))) 1
    A : Matrix (Fin (Nat.succ 0).succ) (Fin (Nat.succ 0).succ) R := Matrix.of (Mat …
    det_A_1 : Eq A.det 1
    ⊢ Membership.mem (Set.image (fun g => ↑g 1) Set.univ) cd
  -/
  refine ⟨⟨A, det_A_1⟩, Set.mem_univ _, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    cd : Fin 2 → R
    b₀ a : R
    gcd_eqn : Eq (HAdd.hAdd (HMul.hMul b₀ (cd 0)) (HMul.hMul a (cd 1))) 1
    A : Matrix (Fin (Nat.succ 0).succ) (Fin (Nat.succ 0).succ) R := Matrix.of (Mat …
    det_A_1 : Eq A.det 1
    ⊢ Eq ((fun g => ↑g 1) ⟨A, det_A_1⟩) cd
  -/
  ext; simp [A]
       /-
         🎉 no goals
       -/


/-- The function `(c,d) → |cz+d|^2` is proper, that is, preimages of bounded-above sets are finite.
-/
theorem tendsto_normSq_coprime_pair :
    Filter.Tendsto (fun p : Fin 2 → ℤ => normSq ((p 0 : ℂ) * z + p 1)) cofinite atTop := by
  -- using this instance rather than the automatic `Function.module` makes unification issues in
  -- `LinearEquiv.isClosedEmbedding_of_injective` less bad later in the proof.
  /-
    z : UpperHalfPlane
    ⊢ Filter.Tendsto (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p …
  -/
  letI : Module ℝ (Fin 2 → ℝ) := NormedSpace.toModule
  /-
    z : UpperHalfPlane
    this : Module Real (Fin 2 → Real) := NormedSpace.toModule
    ⊢ Filter.Tendsto (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p …
  -/
  let π₀ : (Fin 2 → ℝ) →ₗ[ℝ] ℝ := LinearMap.proj 0
  /-
    z : UpperHalfPlane
    this : Module Real (Fin 2 → Real) := NormedSpace.toModule
    π₀ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 0
    ⊢ Filter.Tendsto (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p …
  -/
  let π₁ : (Fin 2 → ℝ) →ₗ[ℝ] ℝ := LinearMap.proj 1
  /-
    z : UpperHalfPlane
    this : Module Real (Fin 2 → Real) := NormedSpace.toModule
    π₀ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 0
    π₁ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 1
    ⊢ Filter.Tendsto (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p …
  -/
  let f : (Fin 2 → ℝ) →ₗ[ℝ] ℂ := π₀.smulRight (z : ℂ) + π₁.smulRight 1
  have f_def : ⇑f = fun p : Fin 2 → ℝ => (p 0 : ℂ) * ↑z + p 1 := by
    ext1
    dsimp only [π₀, π₁, f, LinearMap.coe_proj, real_smul, LinearMap.coe_smulRight,
      LinearMap.add_apply]
    rw [mul_one]
  have :
    (fun p : Fin 2 → ℤ => normSq ((p 0 : ℂ) * ↑z + ↑(p 1))) =
      normSq ∘ f ∘ fun p : Fin 2 → ℤ => ((↑) : ℤ → ℝ) ∘ p := by
    ext1
    rw [f_def]
    dsimp only [Function.comp_def]
    rw [ofReal_intCast, ofReal_intCast]
  /-
    z : UpperHalfPlane
    this✝ : Module Real (Fin 2 → Real) := NormedSpace.toModule
    π₀ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 0
    π₁ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 1
    f : LinearMap (RingHom.id Real) (Fin 2 → Real) Complex := HAdd.hAdd (π₀.smulRi …
    f_def : Eq ⇑f fun p => HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1)
    this : Eq (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1))) ( …
    ⊢ Filter.Tendsto (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p …
  -/
  rw [this]
  have hf : LinearMap.ker f = ⊥ := by
    let g : ℂ →ₗ[ℝ] Fin 2 → ℝ :=
      LinearMap.pi ![imLm, imLm.comp ((z : ℂ) • ((conjAe : ℂ →ₐ[ℝ] ℂ) : ℂ →ₗ[ℝ] ℂ))]
    suffices ((z : ℂ).im⁻¹ • g).comp f = LinearMap.id by exact LinearMap.ker_eq_bot_of_inverse this
    apply LinearMap.ext
    intro c
    have hz : (z : ℂ).im ≠ 0 := z.2.ne'
    rw [LinearMap.comp_apply, LinearMap.smul_apply, LinearMap.id_apply]
    ext i
    dsimp only [Pi.smul_apply, LinearMap.pi_apply, smul_eq_mul]
    fin_cases i
    · show (z : ℂ).im⁻¹ * (f c).im = c 0
      rw [f_def, add_im, im_ofReal_mul, ofReal_im, add_zero, mul_left_comm, inv_mul_cancel₀ hz,
        mul_one]
    · show (z : ℂ).im⁻¹ * ((z : ℂ) * conj (f c)).im = c 1
      rw [f_def, RingHom.map_add, RingHom.map_mul, mul_add, mul_left_comm, mul_conj, conj_ofReal,
        conj_ofReal, ← ofReal_mul, add_im, ofReal_im, zero_add, inv_mul_eq_iff_eq_mul₀ hz]
      simp only [ofReal_im, ofReal_re, mul_im, zero_add, mul_zero]
  /-
    z : UpperHalfPlane
    this✝ : Module Real (Fin 2 → Real) := NormedSpace.toModule
    π₀ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 0
    π₁ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 1
    f : LinearMap (RingHom.id Real) (Fin 2 → Real) Complex := HAdd.hAdd (π₀.smulRi …
    f_def : Eq ⇑f fun p => HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1)
    this : Eq (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1))) ( …
    hf : Eq (LinearMap.ker f) Bot.bot
    ⊢ Filter.Tendsto (Function.comp (⇑Complex.normSq) (Function.comp ⇑f fun p => F …
  -/
  have hf' : IsClosedEmbedding f := f.isClosedEmbedding_of_injective hf
  have h₂ : Tendsto (fun p : Fin 2 → ℤ => ((↑) : ℤ → ℝ) ∘ p) cofinite (cocompact _) := by
    convert Tendsto.pi_map_coprodᵢ fun _ => Int.tendsto_coe_cofinite
    · rw [coprodᵢ_cofinite]
    · rw [coprodᵢ_cocompact]
  /-
    z : UpperHalfPlane
    this✝ : Module Real (Fin 2 → Real) := NormedSpace.toModule
    π₀ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 0
    π₁ : LinearMap (RingHom.id Real) (Fin 2 → Real) Real := LinearMap.proj 1
    f : LinearMap (RingHom.id Real) (Fin 2 → Real) Complex := HAdd.hAdd (π₀.smulRi …
    f_def : Eq ⇑f fun p => HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1)
    this : Eq (fun p => Complex.normSq (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1))) ( …
    hf : Eq (LinearMap.ker f) Bot.bot
    hf' : Topology.IsClosedEmbedding ⇑f
    h₂ : Filter.Tendsto (fun p => Function.comp Int.cast p) Filter.cofinite (Filte …
    ⊢ Filter.Tendsto (Function.comp (⇑Complex.normSq) (Function.comp ⇑f fun p => F …
  -/
  exact tendsto_normSq_cocompact_atTop.comp (hf'.tendsto_cocompact.comp h₂)
  /-
    🎉 no goals
  -/


/-- Given `coprime_pair` `p=(c,d)`, the matrix `[[a,b],[*,*]]` is sent to `a*c+b*d`.
  This is the linear map version of this operation.
-/
def lcRow0 (p : Fin 2 → ℤ) : Matrix (Fin 2) (Fin 2) ℝ →ₗ[ℝ] ℝ :=
  ((p 0 : ℝ) • LinearMap.proj (0 : Fin 2) +
      (p 1 : ℝ) • LinearMap.proj (1 : Fin 2) : (Fin 2 → ℝ) →ₗ[ℝ] ℝ).comp
    (LinearMap.proj 0)


@[simp]
theorem lcRow0_apply (p : Fin 2 → ℤ) (g : Matrix (Fin 2) (Fin 2) ℝ) :
    lcRow0 p g = p 0 * g 0 0 + p 1 * g 0 1 :=
  rfl


/-- Linear map sending the matrix [a, b; c, d] to the matrix [ac₀ + bd₀, - ad₀ + bc₀; c, d], for
some fixed `(c₀, d₀)`. -/
@[simps!]
def lcRow0Extend {cd : Fin 2 → ℤ} (hcd : IsCoprime (cd 0) (cd 1)) :
    Matrix (Fin 2) (Fin 2) ℝ ≃ₗ[ℝ] Matrix (Fin 2) (Fin 2) ℝ :=
  LinearEquiv.piCongrRight
    ![by
      refine
        LinearMap.GeneralLinearGroup.generalLinearEquiv ℝ (Fin 2 → ℝ)
          (GeneralLinearGroup.toLin (planeConformalMatrix (cd 0 : ℝ) (-(cd 1 : ℝ)) ?_))
      /-
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        z : UpperHalfPlane
        cd : Fin 2 → Int
        hcd : IsCoprime (cd 0) (cd 1)
        ⊢ Ne (HAdd.hAdd (HPow.hPow (↑(cd 0)) 2) (HPow.hPow (Neg.neg ↑(cd 1)) 2)) 0
      -/
      norm_cast
      /-
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        z : UpperHalfPlane
        cd : Fin 2 → Int
        hcd : IsCoprime (cd 0) (cd 1)
        ⊢ Not (Eq (HAdd.hAdd (HPow.hPow (cd 0) 2) (HPow.hPow (Neg.neg (cd 1)) 2)) 0)
      -/
      rw [neg_sq]
      /-
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        z : UpperHalfPlane
        cd : Fin 2 → Int
        hcd : IsCoprime (cd 0) (cd 1)
        ⊢ Not (Eq (HAdd.hAdd (HPow.hPow (cd 0) 2) (HPow.hPow (cd 1) 2)) 0)
      -/
      exact hcd.sq_add_sq_ne_zero, LinearEquiv.refl ℝ (Fin 2 → ℝ)]
      /-
        🎉 no goals
      -/


/-- The map `lcRow0` is proper, that is, preimages of cocompact sets are finite in
`[[* , *], [c, d]]`. -/
theorem tendsto_lcRow0 {cd : Fin 2 → ℤ} (hcd : IsCoprime (cd 0) (cd 1)) :
    Tendsto (fun g : { g : SL(2, ℤ) // g 1 = cd } => lcRow0 cd ↑(↑g : SL(2, ℝ))) cofinite
      (cocompact ℝ) := by
  /-
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    ⊢ Filter.Tendsto (fun g => (ModularGroup.lcRow0 cd) ↑((Matrix.SpecialLinearGro …
  -/
  let mB : ℝ → Matrix (Fin 2) (Fin 2) ℝ := fun t => of ![![t, (-(1 : ℤ) : ℝ)], (↑) ∘ cd]
  have hmB : Continuous mB := by
    refine continuous_matrix ?_
    simp only [mB, Fin.forall_fin_two, continuous_const, continuous_id', of_apply, cons_val_zero,
      cons_val_one, and_self_iff]
  /-
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
    hmB : Continuous mB
    ⊢ Filter.Tendsto (fun g => (ModularGroup.lcRow0 cd) ↑((Matrix.SpecialLinearGro …
  -/
  refine Filter.Tendsto.of_tendsto_comp ?_ (comap_cocompact_le hmB)
  let f₁ : SL(2, ℤ) → Matrix (Fin 2) (Fin 2) ℝ := fun g =>
    Matrix.map (↑g : Matrix _ _ ℤ) ((↑) : ℤ → ℝ)
  have cocompact_ℝ_to_cofinite_ℤ_matrix :
    Tendsto (fun m : Matrix (Fin 2) (Fin 2) ℤ => Matrix.map m ((↑) : ℤ → ℝ)) cofinite
      (cocompact _) := by
    simpa only [coprodᵢ_cofinite, coprodᵢ_cocompact] using
      Tendsto.pi_map_coprodᵢ fun _ : Fin 2 =>
        Tendsto.pi_map_coprodᵢ fun _ : Fin 2 => Int.tendsto_coe_cofinite
  have hf₁ : Tendsto f₁ cofinite (cocompact _) :=
    cocompact_ℝ_to_cofinite_ℤ_matrix.comp Subtype.coe_injective.tendsto_cofinite
  have hf₂ : IsClosedEmbedding (lcRow0Extend hcd) :=
    (lcRow0Extend hcd).toContinuousLinearEquiv.toHomeomorph.isClosedEmbedding
  /-
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
    hmB : Continuous mB
    f₁ : Matrix.SpecialLinearGroup (Fin 2) Int → Matrix (Fin 2) (Fin 2) Real := fu …
    cocompact_ℝ_to_cofinite_ℤ_matrix : Filter.Tendsto (fun m => m.map Int.cast) Fi …
    hf₁ : Filter.Tendsto f₁ Filter.cofinite (Filter.cocompact (Matrix (Fin 2) (Fin …
    hf₂ : Topology.IsClosedEmbedding ⇑(ModularGroup.lcRow0Extend hcd)
    ⊢ Filter.Tendsto (Function.comp mB fun g => (ModularGroup.lcRow0 cd) ↑((Matrix …
  -/
  convert hf₂.tendsto_cocompact.comp (hf₁.comp Subtype.coe_injective.tendsto_cofinite) using 1
  /-
    case h.e'_3.h
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
    hmB : Continuous mB
    f₁ : Matrix.SpecialLinearGroup (Fin 2) Int → Matrix (Fin 2) (Fin 2) Real := fu …
    cocompact_ℝ_to_cofinite_ℤ_matrix : Filter.Tendsto (fun m => m.map Int.cast) Fi …
    hf₁ : Filter.Tendsto f₁ Filter.cofinite (Filter.cocompact (Matrix (Fin 2) (Fin …
    hf₂ : Topology.IsClosedEmbedding ⇑(ModularGroup.lcRow0Extend hcd)
    ⊢ Eq (Function.comp mB fun g => (ModularGroup.lcRow0 cd) ↑((Matrix.SpecialLine …
  -/
  ext ⟨g, rfl⟩ i j : 3
  /-
    case h.e'_3.h.h.mk.a
    f₁ : Matrix.SpecialLinearGroup (Fin 2) Int → Matrix (Fin 2) (Fin 2) Real := fu …
    cocompact_ℝ_to_cofinite_ℤ_matrix : Filter.Tendsto (fun m => m.map Int.cast) Fi …
    hf₁ : Filter.Tendsto f₁ Filter.cofinite (Filter.cocompact (Matrix (Fin 2) (Fin …
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hcd : IsCoprime (↑g 1 0) (↑g 1 1)
    mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
    hmB : Continuous mB
    hf₂ : Topology.IsClosedEmbedding ⇑(ModularGroup.lcRow0Extend hcd)
    i j : Fin 2
    ⊢ Eq (Function.comp mB (fun g_1 => (ModularGroup.lcRow0 (↑g 1)) ↑((Matrix.Spec …
  -/
  fin_cases i <;> [fin_cases j; skip]
  -- the following are proved by `simp`, but it is replaced by `simp only` to avoid timeouts.
  · simp only [Fin.isValue, Int.cast_one, map_apply_coe, RingHom.mapMatrix_apply,
      Int.coe_castRingHom, lcRow0_apply, map_apply, Fin.zero_eta, id_eq, Function.comp_apply,
      of_apply, cons_val', cons_val_zero, empty_val', cons_val_fin_one, lcRow0Extend_apply,
      LinearMap.GeneralLinearGroup.coeFn_generalLinearEquiv, GeneralLinearGroup.coe_toLin,
      val_planeConformalMatrix, neg_neg, mulVecLin_apply, mulVec, dotProduct, Fin.sum_univ_two,
      cons_val_one, head_cons, mB, f₁]
    /-
      case h.e'_3.h.h.mk.a.«0».«1»
      f₁ : Matrix.SpecialLinearGroup (Fin 2) Int → Matrix (Fin 2) (Fin 2) Real := fu …
      cocompact_ℝ_to_cofinite_ℤ_matrix : Filter.Tendsto (fun m => m.map Int.cast) Fi …
      hf₁ : Filter.Tendsto f₁ Filter.cofinite (Filter.cocompact (Matrix (Fin 2) (Fin …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hcd : IsCoprime (↑g 1 0) (↑g 1 1)
      mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
      hmB : Continuous mB
      hf₂ : Topology.IsClosedEmbedding ⇑(ModularGroup.lcRow0Extend hcd)
      ⊢ Eq (Function.comp mB (fun g_1 => (ModularGroup.lcRow0 (↑g 1)) ↑((Matrix.Spec …
    -/
  · convert congr_arg (fun n : ℤ => (-n : ℝ)) g.det_coe.symm using 1
    simp only [Fin.zero_eta, id_eq, Function.comp_apply, lcRow0Extend_apply, cons_val_zero,
      LinearMap.GeneralLinearGroup.coeFn_generalLinearEquiv, GeneralLinearGroup.coe_toLin,
      mulVecLin_apply, mulVec, dotProduct, det_fin_two, f₁]
    simp only [Fin.isValue, Fin.mk_one, val_planeConformalMatrix, neg_neg, of_apply, cons_val',
      empty_val', cons_val_fin_one, cons_val_one, head_fin_const, map_apply, Fin.sum_univ_two,
      cons_val_zero, neg_mul, head_cons, Int.cast_sub, Int.cast_mul, neg_sub]
    /-
      case h.e'_3
      f₁ : Matrix.SpecialLinearGroup (Fin 2) Int → Matrix (Fin 2) (Fin 2) Real := fu …
      cocompact_ℝ_to_cofinite_ℤ_matrix : Filter.Tendsto (fun m => m.map Int.cast) Fi …
      hf₁ : Filter.Tendsto f₁ Filter.cofinite (Filter.cocompact (Matrix (Fin 2) (Fin …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hcd : IsCoprime (↑g 1 0) (↑g 1 1)
      mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
      hmB : Continuous mB
      hf₂ : Topology.IsClosedEmbedding ⇑(ModularGroup.lcRow0Extend hcd)
      ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul ↑(↑g 1 1) ↑(↑g 0 0))) (HMul.hMul ↑(↑g 1 0) …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.h.mk.a.«1»
      f₁ : Matrix.SpecialLinearGroup (Fin 2) Int → Matrix (Fin 2) (Fin 2) Real := fu …
      cocompact_ℝ_to_cofinite_ℤ_matrix : Filter.Tendsto (fun m => m.map Int.cast) Fi …
      hf₁ : Filter.Tendsto f₁ Filter.cofinite (Filter.cocompact (Matrix (Fin 2) (Fin …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hcd : IsCoprime (↑g 1 0) (↑g 1 1)
      mB : Real → Matrix (Fin 2) (Fin 2) Real := fun t => Matrix.of (Matrix.vecCons  …
      hmB : Continuous mB
      hf₂ : Topology.IsClosedEmbedding ⇑(ModularGroup.lcRow0Extend hcd)
      j : Fin 2
      ⊢ Eq (Function.comp mB (fun g_1 => (ModularGroup.lcRow0 (↑g 1)) ↑((Matrix.Spec …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- This replaces `(g•z).re = a/c + *` in the standard theory with the following novel identity:
  `g • z = (a c + b d) / (c^2 + d^2) + (d z - c) / ((c^2 + d^2) (c z + d))`
  which does not need to be decomposed depending on whether `c = 0`. -/
theorem smul_eq_lcRow0_add {p : Fin 2 → ℤ} (hp : IsCoprime (p 0) (p 1)) (hg : g 1 = p) :
    ↑(g • z) =
      (lcRow0 p ↑(g : SL(2, ℝ)) : ℂ) / ((p 0 : ℂ) ^ 2 + (p 1 : ℂ) ^ 2) +
        ((p 1 : ℂ) * z - p 0) / (((p 0 : ℂ) ^ 2 + (p 1 : ℂ) ^ 2) * (p 0 * z + p 1)) := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    hg : Eq (↑g 1) p
    ⊢ Eq (↑(HSMul.hSMul g z)) (HAdd.hAdd (HDiv.hDiv (↑((ModularGroup.lcRow0 p) ↑(( …
  -/
  have nonZ1 : (p 0 : ℂ) ^ 2 + (p 1 : ℂ) ^ 2 ≠ 0 := mod_cast hp.sq_add_sq_ne_zero
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    hg : Eq (↑g 1) p
    nonZ1 : Ne (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2)) 0
    ⊢ Eq (↑(HSMul.hSMul g z)) (HAdd.hAdd (HDiv.hDiv (↑((ModularGroup.lcRow0 p) ↑(( …
  -/
  have : ((↑) : ℤ → ℝ) ∘ p ≠ 0 := fun h => hp.ne_zero (by ext i; simpa using congr_fun h i)
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    hg : Eq (↑g 1) p
    nonZ1 : Ne (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2)) 0
    this : Ne (Function.comp Int.cast p) 0
    ⊢ Eq (↑(HSMul.hSMul g z)) (HAdd.hAdd (HDiv.hDiv (↑((ModularGroup.lcRow0 p) ↑(( …
  -/
  have nonZ2 : (p 0 : ℂ) * z + p 1 ≠ 0 := by simpa using linear_ne_zero _ z this
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    hg : Eq (↑g 1) p
    nonZ1 : Ne (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2)) 0
    this : Ne (Function.comp Int.cast p) 0
    nonZ2 : Ne (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1)) 0
    ⊢ Eq (↑(HSMul.hSMul g z)) (HAdd.hAdd (HDiv.hDiv (↑((ModularGroup.lcRow0 p) ↑(( …
  -/
  field_simp [nonZ1, nonZ2, denom_ne_zero, num]
  rw [(by simp :
    (p 1 : ℂ) * z - p 0 = (p 1 * z - p 0) * ↑(Matrix.det (↑g : Matrix (Fin 2) (Fin 2) ℤ)))]
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    hg : Eq (↑g 1) p
    nonZ1 : Ne (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2)) 0
    this : Ne (Function.comp Int.cast p) 0
    nonZ2 : Ne (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1)) 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul ↑(↑g 0 0) ↑z) ↑(↑g 0 1)) (HMul.hMul (HAd …
  -/
  rw [← hg, det_fin_two]
  simp only [Int.coe_castRingHom, coe_matrix_coe, Int.cast_mul, ofReal_intCast, map_apply, denom,
    Int.cast_sub, coe_GLPos_coe_GL_coe_matrix, coe_apply_complex]
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    hg : Eq (↑g 1) p
    nonZ1 : Ne (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2)) 0
    this : Ne (Function.comp Int.cast p) 0
    nonZ2 : Ne (HAdd.hAdd (HMul.hMul ↑(p 0) ↑z) ↑(p 1)) 0
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul ↑(↑g 0 0) ↑z) ↑(↑g 0 1)) (HMul.hMul (HAd …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem tendsto_abs_re_smul {p : Fin 2 → ℤ} (hp : IsCoprime (p 0) (p 1)) :
    Tendsto
      (fun g : { g : SL(2, ℤ) // g 1 = p } => |((g : SL(2, ℤ)) • z).re|) cofinite atTop := by
  suffices
    Tendsto (fun g : (fun g : SL(2, ℤ) => g 1) ⁻¹' {p} => ((g : SL(2, ℤ)) • z).re) cofinite
      (cocompact ℝ)
    by exact tendsto_norm_cocompact_atTop.comp this
  have : ((p 0 : ℝ) ^ 2 + (p 1 : ℝ) ^ 2)⁻¹ ≠ 0 := by
    apply inv_ne_zero
    exact mod_cast hp.sq_add_sq_ne_zero
  /-
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    this : Ne (Inv.inv (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2))) 0
    ⊢ Filter.Tendsto (fun g => (HSMul.hSMul (↑g) z).re) Filter.cofinite (Filter.co …
  -/
  let f := Homeomorph.mulRight₀ _ this
  let ff := Homeomorph.addRight
    (((p 1 : ℂ) * z - p 0) / (((p 0 : ℂ) ^ 2 + (p 1 : ℂ) ^ 2) * (p 0 * z + p 1))).re
  /-
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    this : Ne (Inv.inv (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2))) 0
    f : Homeomorph Real Real := Homeomorph.mulRight₀ (Inv.inv (HAdd.hAdd (HPow.hPo …
    ff : Homeomorph Real Real := Homeomorph.addRight (HDiv.hDiv (HSub.hSub (HMul.h …
    ⊢ Filter.Tendsto (fun g => (HSMul.hSMul (↑g) z).re) Filter.cofinite (Filter.co …
  -/
  convert (f.trans ff).isClosedEmbedding.tendsto_cocompact.comp (tendsto_lcRow0 hp) with _ _ g
  change
    ((g : SL(2, ℤ)) • z).re =
      lcRow0 p ↑(↑g : SL(2, ℝ)) / ((p 0 : ℝ) ^ 2 + (p 1 : ℝ) ^ 2) +
        Complex.re (((p 1 : ℂ) * z - p 0) / (((p 0 : ℂ) ^ 2 + (p 1 : ℂ) ^ 2) * (p 0 * z + p 1)))
  /-
    case h.e'_3.h.h
    z : UpperHalfPlane
    p : Fin 2 → Int
    hp : IsCoprime (p 0) (p 1)
    this : Ne (Inv.inv (HAdd.hAdd (HPow.hPow (↑(p 0)) 2) (HPow.hPow (↑(p 1)) 2))) 0
    f : Homeomorph Real Real := Homeomorph.mulRight₀ (Inv.inv (HAdd.hAdd (HPow.hPo …
    ff : Homeomorph Real Real := Homeomorph.addRight (HDiv.hDiv (HSub.hSub (HMul.h …
    e_1✝ : Eq (↑(Set.preimage (fun g => ↑g 1) (Singleton.singleton p))) (Subtype f …
    g : ↑(Set.preimage (fun g => ↑g 1) (Singleton.singleton p))
    ⊢ Eq (HSMul.hSMul (↑g) z).re (HAdd.hAdd (HDiv.hDiv ((ModularGroup.lcRow0 p) ↑( …
  -/
  exact mod_cast congr_arg Complex.re (smul_eq_lcRow0_add z hp g.2)
  /-
    🎉 no goals
  -/


/-- For `z : ℍ`, there is a `g : SL(2,ℤ)` maximizing `(g•z).im` -/
theorem exists_max_im : ∃ g : SL(2, ℤ), ∀ g' : SL(2, ℤ), (g' • z).im ≤ (g • z).im := by
  classical
  let s : Set (Fin 2 → ℤ) := {cd | IsCoprime (cd 0) (cd 1)}
  have hs : s.Nonempty := ⟨![1, 1], isCoprime_one_left⟩
  obtain ⟨p, hp_coprime, hp⟩ :=
    Filter.Tendsto.exists_within_forall_le hs (tendsto_normSq_coprime_pair z)
  obtain ⟨g, -, hg⟩ := bottom_row_surj hp_coprime
  refine ⟨g, fun g' => ?_⟩
  rw [ModularGroup.im_smul_eq_div_normSq, ModularGroup.im_smul_eq_div_normSq,
    div_le_div_iff_of_pos_left]
  · simpa [← hg] using hp (g' 1) (bottom_row_coprime g')
  · exact z.im_pos
  · exact normSq_denom_pos g' z
  · exact normSq_denom_pos g z


/-- Given `z : ℍ` and a bottom row `(c,d)`, among the `g : SL(2,ℤ)` with this bottom row, minimize
  `|(g•z).re|`. -/
theorem exists_row_one_eq_and_min_re {cd : Fin 2 → ℤ} (hcd : IsCoprime (cd 0) (cd 1)) :
    ∃ g : SL(2, ℤ), g 1 = cd ∧ ∀ g' : SL(2, ℤ), g 1 = g' 1 →
      |(g • z).re| ≤ |(g' • z).re| := by
  haveI : Nonempty { g : SL(2, ℤ) // g 1 = cd } :=
    let ⟨x, hx⟩ := bottom_row_surj hcd
    ⟨⟨x, hx.2⟩⟩
  /-
    z : UpperHalfPlane
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    this : Nonempty (Subtype fun g => Eq (↑g 1) cd)
    ⊢ Exists fun g => And (Eq (↑g 1) cd) (∀ (g' : Matrix.SpecialLinearGroup (Fin 2 …
  -/
  obtain ⟨g, hg⟩ := Filter.Tendsto.exists_forall_le (tendsto_abs_re_smul z hcd)
  /-
    case intro
    z : UpperHalfPlane
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    this : Nonempty (Subtype fun g => Eq (↑g 1) cd)
    g : Subtype fun g => Eq (↑g 1) cd
    hg : ∀ (a : Subtype fun g => Eq (↑g 1) cd), LE.le (abs (HSMul.hSMul (↑g) z).re …
    ⊢ Exists fun g => And (Eq (↑g 1) cd) (∀ (g' : Matrix.SpecialLinearGroup (Fin 2 …
  -/
  refine ⟨g, g.2, ?_⟩
  /-
    case intro
    z : UpperHalfPlane
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    this : Nonempty (Subtype fun g => Eq (↑g 1) cd)
    g : Subtype fun g => Eq (↑g 1) cd
    hg : ∀ (a : Subtype fun g => Eq (↑g 1) cd), LE.le (abs (HSMul.hSMul (↑g) z).re …
    ⊢ ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑↑g 1) (↑g' 1) → LE.le ( …
  -/
  intro g1 hg1
  have : g1 ∈ (fun g : SL(2, ℤ) => g 1) ⁻¹' {cd} := by
    rw [Set.mem_preimage, Set.mem_singleton_iff]
    exact Eq.trans hg1.symm (Set.mem_singleton_iff.mp (Set.mem_preimage.mp g.2))
  /-
    case intro
    z : UpperHalfPlane
    cd : Fin 2 → Int
    hcd : IsCoprime (cd 0) (cd 1)
    this✝ : Nonempty (Subtype fun g => Eq (↑g 1) cd)
    g : Subtype fun g => Eq (↑g 1) cd
    hg : ∀ (a : Subtype fun g => Eq (↑g 1) cd), LE.le (abs (HSMul.hSMul (↑g) z).re …
    g1 : Matrix.SpecialLinearGroup (Fin 2) Int
    hg1 : Eq (↑↑g 1) (↑g1 1)
    this : Membership.mem (Set.preimage (fun g => ↑g 1) (Singleton.singleton cd)) g1
    ⊢ LE.le (abs (HSMul.hSMul (↑g) z).re) (abs (HSMul.hSMul g1 z).re)
  -/
  exact hg ⟨g1, this⟩
  /-
    🎉 no goals
  -/


theorem coe_T_zpow_smul_eq {n : ℤ} : (↑(T ^ n • z) : ℂ) = z + n := by
  /-
    z : UpperHalfPlane
    n : Int
    ⊢ Eq (↑(HSMul.hSMul (HPow.hPow ModularGroup.T n) z)) (HAdd.hAdd ↑z ↑n)
  -/
  rw [sl_moeb, UpperHalfPlane.coe_smul]
  /-
    z : UpperHalfPlane
    n : Int
    ⊢ Eq (HDiv.hDiv (UpperHalfPlane.num (↑(HPow.hPow ModularGroup.T n)) z) (UpperH …
  -/
  simp [coe_T_zpow, denom, num, -map_zpow]
  /-
    🎉 no goals
  -/


theorem re_T_zpow_smul (n : ℤ) : (T ^ n • z).re = z.re + n := by
  /-
    z : UpperHalfPlane
    n : Int
    ⊢ Eq (HSMul.hSMul (HPow.hPow ModularGroup.T n) z).re (HAdd.hAdd z.re ↑n)
  -/
  rw [← coe_re, coe_T_zpow_smul_eq, add_re, intCast_re, coe_re]
  /-
    🎉 no goals
  -/


theorem im_T_zpow_smul (n : ℤ) : (T ^ n • z).im = z.im := by
  /-
    z : UpperHalfPlane
    n : Int
    ⊢ Eq (HSMul.hSMul (HPow.hPow ModularGroup.T n) z).im z.im
  -/
  rw [← coe_im, coe_T_zpow_smul_eq, add_im, intCast_im, add_zero, coe_im]
  /-
    🎉 no goals
  -/


                                                /-
                                                  z : UpperHalfPlane
                                                  ⊢ Eq (HSMul.hSMul ModularGroup.T z).re (HAdd.hAdd z.re 1)
                                                -/
theorem re_T_smul : (T • z).re = z.re + 1 := by simpa using re_T_zpow_smul z 1
                                                /-
                                                  🎉 no goals
                                                -/


                                            /-
                                              z : UpperHalfPlane
                                              ⊢ Eq (HSMul.hSMul ModularGroup.T z).im z.im
                                            -/
theorem im_T_smul : (T • z).im = z.im := by simpa using im_T_zpow_smul z 1
                                            /-
                                              🎉 no goals
                                            -/


                                                      /-
                                                        z : UpperHalfPlane
                                                        ⊢ Eq (HSMul.hSMul (Inv.inv ModularGroup.T) z).re (HSub.hSub z.re 1)
                                                      -/
theorem re_T_inv_smul : (T⁻¹ • z).re = z.re - 1 := by simpa using re_T_zpow_smul z (-1)
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                  /-
                                                    z : UpperHalfPlane
                                                    ⊢ Eq (HSMul.hSMul (Inv.inv ModularGroup.T) z).im z.im
                                                  -/
theorem im_T_inv_smul : (T⁻¹ • z).im = z.im := by simpa using im_T_zpow_smul z (-1)
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem exists_eq_T_zpow_of_c_eq_zero (hc : g 1 0 = 0) :
    ∃ n : ℤ, ∀ z : ℍ, g • z = T ^ n • z := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 0
    ⊢ Exists fun n => ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (H …
  -/
  have had := g.det_coe
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 0
    had : Eq (↑g).det 1
    ⊢ Exists fun n => ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (H …
  -/
  replace had : g 0 0 * g 1 1 = 1 := by rw [det_fin_two, hc] at had; omega
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 0
    had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
    ⊢ Exists fun n => ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (H …
  -/
  rcases Int.eq_one_or_neg_one_of_mul_eq_one' had with (⟨ha, hd⟩ | ⟨ha, hd⟩)
    /-
      case inl.intro
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hc : Eq (↑g 1 0) 0
      had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
      ha : Eq (↑g 0 0) 1
      hd : Eq (↑g 1 1) 1
      ⊢ Exists fun n => ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (H …
    -/
  · use g 0 1
    /-
      case h
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hc : Eq (↑g 1 0) 0
      had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
      ha : Eq (↑g 0 0) 1
      hd : Eq (↑g 1 1) 1
      ⊢ ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (HPow.hPow Modular …
    -/
    suffices g = T ^ g 0 1 by intro z; conv_lhs => rw [this]
    /-
      case h
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hc : Eq (↑g 1 0) 0
      had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
      ha : Eq (↑g 0 0) 1
      hd : Eq (↑g 1 1) 1
      ⊢ Eq g (HPow.hPow ModularGroup.T (↑g 0 1))
    -/
    ext i j; fin_cases i <;> fin_cases j <;>
      /-
        case h.a.«0».«0»
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hc : Eq (↑g 1 0) 0
        had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
        ha : Eq (↑g 0 0) 1
        hd : Eq (↑g 1 1) 1
        ⊢ Eq (↑g ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)) (↑(HPow.hPow ModularGrou …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp [ha, hc, hd, coe_T_zpow, show (1 : Fin (0 + 2)) = (1 : Fin 2) from rfl]
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hc : Eq (↑g 1 0) 0
      had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
      ha : Eq (↑g 0 0) (-1)
      hd : Eq (↑g 1 1) (-1)
      ⊢ Exists fun n => ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (H …
    -/
  · use -(g 0 1)
    /-
      case h
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hc : Eq (↑g 1 0) 0
      had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
      ha : Eq (↑g 0 0) (-1)
      hd : Eq (↑g 1 1) (-1)
      ⊢ ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (HPow.hPow Modular …
    -/
    suffices g = -T ^ (-(g 0 1)) by intro z; conv_lhs => rw [this, SL_neg_smul]
    /-
      case h
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hc : Eq (↑g 1 0) 0
      had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
      ha : Eq (↑g 0 0) (-1)
      hd : Eq (↑g 1 1) (-1)
      ⊢ Eq g (Neg.neg (HPow.hPow ModularGroup.T (Neg.neg (↑g 0 1))))
    -/
    ext i j; fin_cases i <;> fin_cases j <;>
      /-
        case h.a.«0».«0»
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hc : Eq (↑g 1 0) 0
        had : Eq (HMul.hMul (↑g 0 0) (↑g 1 1)) 1
        ha : Eq (↑g 0 0) (-1)
        hd : Eq (↑g 1 1) (-1)
        ⊢ Eq (↑g ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)) (↑(Neg.neg (HPow.hPow Mo …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp [ha, hc, hd, coe_T_zpow, show (1 : Fin (0 + 2)) = (1 : Fin 2) from rfl]
      /-
        🎉 no goals
      -/

-- If `c = 1`, then `g` factorises into a product terms involving only `T` and `S`.

theorem g_eq_of_c_eq_one (hc : g 1 0 = 1) : g = T ^ g 0 0 * S * T ^ g 1 1 := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 1
    ⊢ Eq g (HMul.hMul (HMul.hMul (HPow.hPow ModularGroup.T (↑g 0 0)) ModularGroup. …
  -/
  have hg := g.det_coe.symm
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 1
    hg : Eq 1 (↑g).det
    ⊢ Eq g (HMul.hMul (HMul.hMul (HPow.hPow ModularGroup.T (↑g 0 0)) ModularGroup. …
  -/
  replace hg : g 0 1 = g 0 0 * g 1 1 - 1 := by rw [det_fin_two, hc] at hg; omega
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 1
    hg : Eq (↑g 0 1) (HSub.hSub (HMul.hMul (↑g 0 0) (↑g 1 1)) 1)
    ⊢ Eq g (HMul.hMul (HMul.hMul (HPow.hPow ModularGroup.T (↑g 0 0)) ModularGroup. …
  -/
  refine Subtype.ext ?_
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hc : Eq (↑g 1 0) 1
    hg : Eq (↑g 0 1) (HSub.hSub (HMul.hMul (↑g 0 0) (↑g 1 1)) 1)
    ⊢ Eq ↑g ↑(HMul.hMul (HMul.hMul (HPow.hPow ModularGroup.T (↑g 0 0)) ModularGrou …
  -/
  conv_lhs => rw [(g : Matrix _ _ ℤ).eta_fin_two]
  simp only [hg, sub_eq_add_neg, hc, coe_mul, coe_T_zpow, coe_S, mul_fin_two, mul_zero, mul_one,
    zero_add, one_mul, add_zero, zero_mul]


/-- If `1 < |z|`, then `|S • z| < 1`. -/
theorem normSq_S_smul_lt_one (h : 1 < normSq z) : normSq ↑(S • z) < 1 := by
  /-
    z : UpperHalfPlane
    h : LT.lt 1 (Complex.normSq ↑z)
    ⊢ LT.lt (Complex.normSq ↑(HSMul.hSMul ModularGroup.S z)) 1
  -/
  simpa [coe_S, num, denom] using (inv_lt_inv₀ z.normSq_pos zero_lt_one).mpr h
  /-
    🎉 no goals
  -/


/-- If `|z| < 1`, then applying `S` strictly decreases `im`. -/
theorem im_lt_im_S_smul (h : normSq z < 1) : z.im < (S • z).im := by
  have : z.im < z.im / normSq (z : ℂ) := by
    have imz : 0 < z.im := im_pos z
    apply (lt_div_iff₀ z.normSq_pos).mpr
    nlinarith
  /-
    z : UpperHalfPlane
    h : LT.lt (Complex.normSq ↑z) 1
    this : LT.lt z.im (HDiv.hDiv z.im (Complex.normSq ↑z))
    ⊢ LT.lt z.im (HSMul.hSMul ModularGroup.S z).im
  -/
  convert this
  /-
    case h.e'_4
    z : UpperHalfPlane
    h : LT.lt (Complex.normSq ↑z) 1
    this : LT.lt z.im (HDiv.hDiv z.im (Complex.normSq ↑z))
    ⊢ Eq (HSMul.hSMul ModularGroup.S z).im (HDiv.hDiv z.im (Complex.normSq ↑z))
  -/
  simp only [ModularGroup.im_smul_eq_div_normSq]
  /-
    case h.e'_4
    z : UpperHalfPlane
    h : LT.lt (Complex.normSq ↑z) 1
    this : LT.lt z.im (HDiv.hDiv z.im (Complex.normSq ↑z))
    ⊢ Eq (HDiv.hDiv z.im (Complex.normSq (UpperHalfPlane.denom (↑ModularGroup.S) z …
  -/
  simp [denom, coe_S]
  /-
    🎉 no goals
  -/


/-- The standard (closed) fundamental domain of the action of `SL(2,ℤ)` on `ℍ`. -/
def fd : Set ℍ :=
  {z | 1 ≤ normSq (z : ℂ) ∧ |z.re| ≤ (1 : ℝ) / 2}


/-- The standard open fundamental domain of the action of `SL(2,ℤ)` on `ℍ`. -/
def fdo : Set ℍ :=
  {z | 1 < normSq (z : ℂ) ∧ |z.re| < (1 : ℝ) / 2}


@[inherit_doc ModularGroup.fd]
scoped[Modular] notation "𝒟" => ModularGroup.fd


@[inherit_doc ModularGroup.fdo]
scoped[Modular] notation "𝒟ᵒ" => ModularGroup.fdo


theorem abs_two_mul_re_lt_one_of_mem_fdo (h : z ∈ 𝒟ᵒ) : |2 * z.re| < 1 := by
  /-
    z : UpperHalfPlane
    h : Membership.mem ModularGroup.fdo z
    ⊢ LT.lt (abs (HMul.hMul 2 z.re)) 1
  -/
  rw [abs_mul, abs_two, ← lt_div_iff₀' (zero_lt_two' ℝ)]
  /-
    z : UpperHalfPlane
    h : Membership.mem ModularGroup.fdo z
    ⊢ LT.lt (abs z.re) (1 / 2)
  -/
  exact h.2
  /-
    🎉 no goals
  -/


theorem three_lt_four_mul_im_sq_of_mem_fdo (h : z ∈ 𝒟ᵒ) : 3 < 4 * z.im ^ 2 := by
  /-
    z : UpperHalfPlane
    h : Membership.mem ModularGroup.fdo z
    ⊢ LT.lt 3 (HMul.hMul 4 (HPow.hPow z.im 2))
  -/
  have : 1 < z.re * z.re + z.im * z.im := by simpa [Complex.normSq_apply] using h.1
  /-
    z : UpperHalfPlane
    h : Membership.mem ModularGroup.fdo z
    this : LT.lt 1 (HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im))
    ⊢ LT.lt 3 (HMul.hMul 4 (HPow.hPow z.im 2))
  -/
  have := h.2
  /-
    z : UpperHalfPlane
    h : Membership.mem ModularGroup.fdo z
    this✝ : LT.lt 1 (HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im))
    this : LT.lt (abs z.re) (1 / 2)
    ⊢ LT.lt 3 (HMul.hMul 4 (HPow.hPow z.im 2))
  -/
                           /-
                             🎉 no goals
                           -/
  cases abs_cases z.re <;> nlinarith
                           /-
                             🎉 no goals
                           -/


/-- non-strict variant of `ModularGroup.three_le_four_mul_im_sq_of_mem_fdo` -/
theorem three_le_four_mul_im_sq_of_mem_fd {τ : ℍ} (h : τ ∈ 𝒟) : 3 ≤ 4 * τ.im ^ 2 := by
  /-
    τ : UpperHalfPlane
    h : Membership.mem ModularGroup.fd τ
    ⊢ LE.le 3 (HMul.hMul 4 (HPow.hPow τ.im 2))
  -/
  have : 1 ≤ τ.re * τ.re + τ.im * τ.im := by simpa [Complex.normSq_apply] using h.1
  /-
    τ : UpperHalfPlane
    h : Membership.mem ModularGroup.fd τ
    this : LE.le 1 (HAdd.hAdd (HMul.hMul τ.re τ.re) (HMul.hMul τ.im τ.im))
    ⊢ LE.le 3 (HMul.hMul 4 (HPow.hPow τ.im 2))
  -/
                           /-
                             🎉 no goals
                           -/
  cases abs_cases τ.re <;> nlinarith [h.2]
                           /-
                             🎉 no goals
                           -/


/-- If `z ∈ 𝒟ᵒ`, and `n : ℤ`, then `|z + n| > 1`. -/
theorem one_lt_normSq_T_zpow_smul (hz : z ∈ 𝒟ᵒ) (n : ℤ) : 1 < normSq (T ^ n • z : ℍ) := by
  /-
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    n : Int
    ⊢ LT.lt 1 (Complex.normSq ↑(HSMul.hSMul (HPow.hPow ModularGroup.T n) z))
  -/
  have hz₁ : 1 < z.re * z.re + z.im * z.im := hz.1
  /-
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    n : Int
    hz₁ : LT.lt 1 (HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im))
    ⊢ LT.lt 1 (Complex.normSq ↑(HSMul.hSMul (HPow.hPow ModularGroup.T n) z))
  -/
  have hzn := Int.nneg_mul_add_sq_of_abs_le_one n (abs_two_mul_re_lt_one_of_mem_fdo hz).le
  /-
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    n : Int
    hz₁ : LT.lt 1 (HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im))
    hzn : LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul 2 z.re)) (HMul.hMul ↑n ↑n))
    ⊢ LT.lt 1 (Complex.normSq ↑(HSMul.hSMul (HPow.hPow ModularGroup.T n) z))
  -/
  have : 1 < (z.re + ↑n) * (z.re + ↑n) + z.im * z.im := by linarith
  /-
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    n : Int
    hz₁ : LT.lt 1 (HAdd.hAdd (HMul.hMul z.re z.re) (HMul.hMul z.im z.im))
    hzn : LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul 2 z.re)) (HMul.hMul ↑n ↑n))
    this : LT.lt 1 (HAdd.hAdd (HMul.hMul (HAdd.hAdd z.re ↑n) (HAdd.hAdd z.re ↑n))  …
    ⊢ LT.lt 1 (Complex.normSq ↑(HSMul.hSMul (HPow.hPow ModularGroup.T n) z))
  -/
  simpa [coe_T_zpow, normSq, num, denom, -map_zpow]
  /-
    🎉 no goals
  -/


theorem eq_zero_of_mem_fdo_of_T_zpow_mem_fdo {n : ℤ} (hz : z ∈ 𝒟ᵒ) (hg : T ^ n • z ∈ 𝒟ᵒ) :
    n = 0 := by
  suffices |(n : ℝ)| < 1 by
    rwa [← Int.cast_abs, ← Int.cast_one, Int.cast_lt, Int.abs_lt_one_iff] at this
  /-
    z : UpperHalfPlane
    n : Int
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul (HPow.hPow ModularGroup.T n) …
    ⊢ LT.lt (abs ↑n) 1
  -/
  have h₁ := hz.2
  /-
    z : UpperHalfPlane
    n : Int
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul (HPow.hPow ModularGroup.T n) …
    h₁ : LT.lt (abs z.re) (1 / 2)
    ⊢ LT.lt (abs ↑n) 1
  -/
  have h₂ := hg.2
  /-
    z : UpperHalfPlane
    n : Int
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul (HPow.hPow ModularGroup.T n) …
    h₁ : LT.lt (abs z.re) (1 / 2)
    h₂ : LT.lt (abs (HSMul.hSMul (HPow.hPow ModularGroup.T n) z).re) (1 / 2)
    ⊢ LT.lt (abs ↑n) 1
  -/
  rw [re_T_zpow_smul] at h₂
  calc
    |(n : ℝ)| ≤ |z.re| + |z.re + (n : ℝ)| := abs_add' (n : ℝ) z.re
    _ < 1 / 2 + 1 / 2 := add_lt_add h₁ h₂
    _ = 1 := add_halves 1


/-- First Fundamental Domain Lemma: Any `z : ℍ` can be moved to `𝒟` by an element of
`SL(2,ℤ)` -/
theorem exists_smul_mem_fd (z : ℍ) : ∃ g : SL(2, ℤ), g • z ∈ 𝒟 := by
  -- obtain a g₀ which maximizes im (g • z),
  /-
    z : UpperHalfPlane
    ⊢ Exists fun g => Membership.mem ModularGroup.fd (HSMul.hSMul g z)
  -/
  obtain ⟨g₀, hg₀⟩ := exists_max_im z
  -- then among those, minimize re
  /-
    case intro
    z : UpperHalfPlane
    g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
    hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
    ⊢ Exists fun g => Membership.mem ModularGroup.fd (HSMul.hSMul g z)
  -/
  obtain ⟨g, hg, hg'⟩ := exists_row_one_eq_and_min_re z (bottom_row_coprime g₀)
  /-
    case intro.intro.intro
    z : UpperHalfPlane
    g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
    hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hg : Eq (↑g 1) (↑g₀ 1)
    hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
    ⊢ Exists fun g => Membership.mem ModularGroup.fd (HSMul.hSMul g z)
  -/
  refine ⟨g, ?_⟩
  -- `g` has same max im property as `g₀`
  have hg₀' : ∀ g' : SL(2, ℤ), (g' • z).im ≤ (g • z).im := by
    have hg'' : (g • z).im = (g₀ • z).im := by
      rw [ModularGroup.im_smul_eq_div_normSq, ModularGroup.im_smul_eq_div_normSq,
        denom_apply, denom_apply, hg]
    simpa only [hg''] using hg₀
  /-
    case intro.intro.intro
    z : UpperHalfPlane
    g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
    hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    hg : Eq (↑g 1) (↑g₀ 1)
    hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
    hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
    ⊢ Membership.mem ModularGroup.fd (HSMul.hSMul g z)
  -/
  constructor
  · -- Claim: `1 ≤ ⇑norm_sq ↑(g • z)`. If not, then `S•g•z` has larger imaginary part
    /-
      case intro.intro.intro.left
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
      ⊢ LE.le 1 (Complex.normSq ↑(HSMul.hSMul g z))
    -/
    contrapose! hg₀'
    /-
      case intro.intro.intro.left
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : LT.lt (Complex.normSq ↑(HSMul.hSMul g z)) 1
      ⊢ Exists fun g' => LT.lt (HSMul.hSMul g z).im (HSMul.hSMul g' z).im
    -/
    refine ⟨S * g, ?_⟩
    /-
      case intro.intro.intro.left
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : LT.lt (Complex.normSq ↑(HSMul.hSMul g z)) 1
      ⊢ LT.lt (HSMul.hSMul g z).im (HSMul.hSMul (HMul.hMul ModularGroup.S g) z).im
    -/
    rw [mul_smul]
    /-
      case intro.intro.intro.left
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : LT.lt (Complex.normSq ↑(HSMul.hSMul g z)) 1
      ⊢ LT.lt (HSMul.hSMul g z).im (HSMul.hSMul ModularGroup.S (HSMul.hSMul g z)).im
    -/
    exact im_lt_im_S_smul hg₀'
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.right
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
      ⊢ LE.le (abs (HSMul.hSMul g z).re) (1 / 2)
    -/
  · show |(g • z).re| ≤ 1 / 2
    -- if not, then either `T` or `T'` decrease |Re|.
    /-
      case intro.intro.intro.right
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
      ⊢ LE.le (abs (HSMul.hSMul g z).re) (1 / 2)
    -/
    rw [abs_le]
    /-
      case intro.intro.intro.right
      z : UpperHalfPlane
      g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
      hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
      g : Matrix.SpecialLinearGroup (Fin 2) Int
      hg : Eq (↑g 1) (↑g₀ 1)
      hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
      hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
      ⊢ And (LE.le (Neg.neg (1 / 2)) (HSMul.hSMul g z).re) (LE.le (HSMul.hSMul g z). …
    -/
    constructor
      /-
        case intro.intro.intro.right.left
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        ⊢ LE.le (Neg.neg (1 / 2)) (HSMul.hSMul g z).re
      -/
    · contrapose! hg'
      /-
        case intro.intro.intro.right.left
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        hg' : LT.lt (HSMul.hSMul g z).re (Neg.neg (1 / 2))
        ⊢ Exists fun g' => And (Eq (↑g 1) (↑g' 1)) (LT.lt (abs (HSMul.hSMul g' z).re)  …
      -/
      refine ⟨T * g, (T_mul_apply_one _).symm, ?_⟩
      /-
        case intro.intro.intro.right.left
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        hg' : LT.lt (HSMul.hSMul g z).re (Neg.neg (1 / 2))
        ⊢ LT.lt (abs (HSMul.hSMul (HMul.hMul ModularGroup.T g) z).re) (abs (HSMul.hSMu …
      -/
      rw [mul_smul, re_T_smul]
      /-
        case intro.intro.intro.right.left
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        hg' : LT.lt (HSMul.hSMul g z).re (Neg.neg (1 / 2))
        ⊢ LT.lt (abs (HAdd.hAdd (HSMul.hSMul g z).re 1)) (abs (HSMul.hSMul g z).re)
      -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      cases abs_cases ((g • z).re + 1) <;> cases abs_cases (g • z).re <;> linarith
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      /-
        case intro.intro.intro.right.right
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), Eq (↑g 1) (↑g' 1) → LE.l …
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        ⊢ LE.le (HSMul.hSMul g z).re (1 / 2)
      -/
    · contrapose! hg'
      /-
        case intro.intro.intro.right.right
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        hg' : LT.lt (1 / 2) (HSMul.hSMul g z).re
        ⊢ Exists fun g' => And (Eq (↑g 1) (↑g' 1)) (LT.lt (abs (HSMul.hSMul g' z).re)  …
      -/
      refine ⟨T⁻¹ * g, (T_inv_mul_apply_one _).symm, ?_⟩
      /-
        case intro.intro.intro.right.right
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        hg' : LT.lt (1 / 2) (HSMul.hSMul g z).re
        ⊢ LT.lt (abs (HSMul.hSMul (HMul.hMul (Inv.inv ModularGroup.T) g) z).re) (abs ( …
      -/
      rw [mul_smul, re_T_inv_smul]
      /-
        case intro.intro.intro.right.right
        z : UpperHalfPlane
        g₀ : Matrix.SpecialLinearGroup (Fin 2) Int
        hg₀ : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z) …
        g : Matrix.SpecialLinearGroup (Fin 2) Int
        hg : Eq (↑g 1) (↑g₀ 1)
        hg₀' : ∀ (g' : Matrix.SpecialLinearGroup (Fin 2) Int), LE.le (HSMul.hSMul g' z …
        hg' : LT.lt (1 / 2) (HSMul.hSMul g z).re
        ⊢ LT.lt (abs (HSub.hSub (HSMul.hSMul g z).re 1)) (abs (HSMul.hSMul g z).re)
      -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      cases abs_cases ((g • z).re - 1) <;> cases abs_cases (g • z).re <;> linarith
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- An auxiliary result en route to `ModularGroup.c_eq_zero`. -/
theorem abs_c_le_one (hz : z ∈ 𝒟ᵒ) (hg : g • z ∈ 𝒟ᵒ) : |g 1 0| ≤ 1 := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    ⊢ LE.le (abs (↑g 1 0)) 1
  -/
  let c' : ℤ := g 1 0
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    c' : Int := ↑g 1 0
    ⊢ LE.le (abs (↑g 1 0)) 1
  -/
  let c := (c' : ℝ)
  suffices 3 * c ^ 2 < 4 by
    rw [← Int.cast_pow, ← Int.cast_three, ← Int.cast_four, ← Int.cast_mul, Int.cast_lt] at this
    replace this : c' ^ 2 ≤ 1 ^ 2 := by omega
    rwa [sq_le_sq, abs_one] at this
  suffices c ≠ 0 → 9 * c ^ 4 < 16 by
    rcases eq_or_ne c 0 with (hc | hc)
    · rw [hc]; norm_num
    · refine (abs_lt_of_sq_lt_sq' ?_ (by norm_num)).2
      specialize this hc
      linarith
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    c' : Int := ↑g 1 0
    c : Real := ↑c'
    ⊢ Ne c 0 → LT.lt (HMul.hMul 9 (HPow.hPow c 4)) 16
  -/
  intro hc
  have h₁ : 3 * 3 * c ^ 4 < 4 * (g • z).im ^ 2 * (4 * z.im ^ 2) * c ^ 4 := by
    gcongr <;> apply three_lt_four_mul_im_sq_of_mem_fdo <;> assumption
  have h₂ : (c * z.im) ^ 4 / normSq (denom (↑g) z) ^ 2 ≤ 1 :=
    div_le_one_of_le₀
      (pow_four_le_pow_two_of_pow_two_le (z.c_mul_im_sq_le_normSq_denom g))
      (sq_nonneg _)
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    c' : Int := ↑g 1 0
    c : Real := ↑c'
    hc : Ne c 0
    h₁ : LT.lt (HMul.hMul (HMul.hMul 3 3) (HPow.hPow c 4)) (HMul.hMul (HMul.hMul ( …
    h₂ : LE.le (HDiv.hDiv (HPow.hPow (HMul.hMul c z.im) 4) (HPow.hPow (Complex.nor …
    ⊢ LT.lt (HMul.hMul 9 (HPow.hPow c 4)) 16
  -/
  let nsq := normSq (denom g z)
  calc
    9 * c ^ 4 < c ^ 4 * z.im ^ 2 * (g • z).im ^ 2 * 16 := by linarith
    _ = c ^ 4 * z.im ^ 4 / nsq ^ 2 * 16 := by
      rw [im_smul_eq_div_normSq, div_pow]
      ring
    _ ≤ 16 := by rw [← mul_pow]; linarith


/-- An auxiliary result en route to `ModularGroup.eq_smul_self_of_mem_fdo_mem_fdo`. -/
theorem c_eq_zero (hz : z ∈ 𝒟ᵒ) (hg : g • z ∈ 𝒟ᵒ) : g 1 0 = 0 := by
  have hp : ∀ {g' : SL(2, ℤ)}, g' • z ∈ 𝒟ᵒ → g' 1 0 ≠ 1 := by
    intro g' hg'
    by_contra hc
    let a := g' 0 0
    let d := g' 1 1
    have had : T ^ (-a) * g' = S * T ^ d := by rw [g_eq_of_c_eq_one hc]; group
    let w := T ^ (-a) • g' • z
    have h₁ : w = S • T ^ d • z := by simp only [w, ← mul_smul, had]
    replace h₁ : normSq w < 1 := h₁.symm ▸ normSq_S_smul_lt_one (one_lt_normSq_T_zpow_smul hz d)
    have h₂ : 1 < normSq w := one_lt_normSq_T_zpow_smul hg' (-a)
    linarith
  have hn : g 1 0 ≠ -1 := by
    intro hc
    replace hc : (-g) 1 0 = 1 := by simp [← neg_eq_iff_eq_neg.mpr hc]
    replace hg : -g • z ∈ 𝒟ᵒ := (SL_neg_smul g z).symm ▸ hg
    exact hp hg hc
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    hp : ∀ {g' : Matrix.SpecialLinearGroup (Fin 2) Int}, Membership.mem ModularGro …
    hn : Ne (↑g 1 0) (-1)
    ⊢ Eq (↑g 1 0) 0
  -/
  specialize hp hg
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    hn : Ne (↑g 1 0) (-1)
    hp : Ne (↑g 1 0) 1
    ⊢ Eq (↑g 1 0) 0
  -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  rcases Int.abs_le_one_iff.mp <| abs_c_le_one hz hg with ⟨⟩ <;> tauto
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Second Fundamental Domain Lemma: if both `z` and `g • z` are in the open domain `𝒟ᵒ`,
where `z : ℍ` and `g : SL(2,ℤ)`, then `z = g • z`. -/
theorem eq_smul_self_of_mem_fdo_mem_fdo (hz : z ∈ 𝒟ᵒ) (hg : g • z ∈ 𝒟ᵒ) : z = g • z := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    ⊢ Eq z (HSMul.hSMul g z)
  -/
  obtain ⟨n, hn⟩ := exists_eq_T_zpow_of_c_eq_zero (c_eq_zero hz hg)
  /-
    case intro
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul g z)
    n : Int
    hn : ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (HPow.hPow Modu …
    ⊢ Eq z (HSMul.hSMul g z)
  -/
  rw [hn] at hg ⊢
  /-
    case intro
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    z : UpperHalfPlane
    hz : Membership.mem ModularGroup.fdo z
    n : Int
    hg : Membership.mem ModularGroup.fdo (HSMul.hSMul (HPow.hPow ModularGroup.T n) …
    hn : ∀ (z : UpperHalfPlane), Eq (HSMul.hSMul g z) (HSMul.hSMul (HPow.hPow Modu …
    ⊢ Eq z (HSMul.hSMul (HPow.hPow ModularGroup.T n) z)
  -/
  simp [eq_zero_of_mem_fdo_of_T_zpow_mem_fdo hz hg, one_smul]
  /-
    🎉 no goals
  -/


lemma exists_one_half_le_im_smul (τ : ℍ) : ∃ γ : SL(2, ℤ), 1 / 2 ≤ im (γ • τ) := by
  /-
    τ : UpperHalfPlane
    ⊢ Exists fun γ => LE.le (1 / 2) (HSMul.hSMul γ τ).im
  -/
  obtain ⟨γ, hγ⟩ := exists_smul_mem_fd τ
  /-
    case intro
    τ : UpperHalfPlane
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    hγ : Membership.mem ModularGroup.fd (HSMul.hSMul γ τ)
    ⊢ Exists fun γ => LE.le (1 / 2) (HSMul.hSMul γ τ).im
  -/
  use γ
  /-
    case h
    τ : UpperHalfPlane
    γ : Matrix.SpecialLinearGroup (Fin 2) Int
    hγ : Membership.mem ModularGroup.fd (HSMul.hSMul γ τ)
    ⊢ LE.le (1 / 2) (HSMul.hSMul γ τ).im
  -/
  nlinarith [three_le_four_mul_im_sq_of_mem_fd hγ, im_pos (γ • τ)]
  /-
    🎉 no goals
  -/


/-- For every `τ : ℍ` there is some `γ ∈ SL(2, ℤ)` that sends it to an element whose
imaginary part is at least `1/2` and such that `denom γ τ` has norm at most 1. -/
lemma exists_one_half_le_im_smul_and_norm_denom_le (τ : ℍ) :
    ∃ γ : SL(2, ℤ), 1 / 2 ≤ im (γ • τ) ∧ ‖denom γ τ‖ ≤ 1 := by
  /-
    τ : UpperHalfPlane
    ⊢ Exists fun γ => And (LE.le (1 / 2) (HSMul.hSMul γ τ).im) (LE.le (Norm.norm ( …
  -/
  rcases le_total (1 / 2) τ.im with h | h
    /-
      case inl
      τ : UpperHalfPlane
      h : LE.le (1 / 2) τ.im
      ⊢ Exists fun γ => And (LE.le (1 / 2) (HSMul.hSMul γ τ).im) (LE.le (Norm.norm ( …
    -/
  · exact ⟨1, (one_smul SL(2, ℤ) τ).symm ▸ h, by simp only [coe_one, denom_one, norm_one, le_refl]⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      τ : UpperHalfPlane
      h : LE.le τ.im (1 / 2)
      ⊢ Exists fun γ => And (LE.le (1 / 2) (HSMul.hSMul γ τ).im) (LE.le (Norm.norm ( …
    -/
  · refine (exists_one_half_le_im_smul τ).imp (fun γ hγ ↦ ⟨hγ, ?_⟩)
    /-
      case inr
      τ : UpperHalfPlane
      h : LE.le τ.im (1 / 2)
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      hγ : LE.le (1 / 2) (HSMul.hSMul γ τ).im
      ⊢ LE.le (Norm.norm (UpperHalfPlane.denom (↑γ) τ)) 1
    -/
    have h1 : τ.im ≤ (γ • τ).im := h.trans hγ
    /-
      case inr
      τ : UpperHalfPlane
      h : LE.le τ.im (1 / 2)
      γ : Matrix.SpecialLinearGroup (Fin 2) Int
      hγ : LE.le (1 / 2) (HSMul.hSMul γ τ).im
      h1 : LE.le τ.im (HSMul.hSMul γ τ).im
      ⊢ LE.le (Norm.norm (UpperHalfPlane.denom (↑γ) τ)) 1
    -/
    rw [im_smul_eq_div_normSq, le_div_iff₀ (normSq_denom_pos (↑γ) τ), normSq_eq_norm_sq] at h1
    simpa only [norm_eq_abs, sq_le_one_iff_abs_le_one, Complex.abs_abs] using
      (mul_le_iff_le_one_right τ.2).mp h1


