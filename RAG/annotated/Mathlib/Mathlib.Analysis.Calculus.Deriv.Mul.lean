theorem hasDerivWithinAt_of_bilinear
    (hu : HasDerivWithinAt u u' s x) (hv : HasDerivWithinAt v v' s x) :
    HasDerivWithinAt (fun x ↦ B (u x) (v x)) (B (u x) v' + B u' (v x)) s x := by
  simpa using (B.hasFDerivWithinAt_of_bilinear
    hu.hasFDerivWithinAt hv.hasFDerivWithinAt).hasDerivWithinAt


theorem hasDerivAt_of_bilinear (hu : HasDerivAt u u' x) (hv : HasDerivAt v v' x) :
    HasDerivAt (fun x ↦ B (u x) (v x)) (B (u x) v' + B u' (v x)) x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    G : Type u_1
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    x : 𝕜
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    u : 𝕜 → E
    v : 𝕜 → F
    u' : E
    v' : F
    hu : HasDerivAt u u' x
    hv : HasDerivAt v v' x
    ⊢ HasDerivAt (fun x => (B (u x)) (v x)) (HAdd.hAdd ((B (u x)) v') ((B u') (v x …
  -/
  simpa using (B.hasFDerivAt_of_bilinear hu.hasFDerivAt hv.hasFDerivAt).hasDerivAt
  /-
    🎉 no goals
  -/


theorem hasStrictDerivAt_of_bilinear (hu : HasStrictDerivAt u u' x) (hv : HasStrictDerivAt v v' x) :
    HasStrictDerivAt (fun x ↦ B (u x) (v x)) (B (u x) v' + B u' (v x)) x := by
  simpa using
    (B.hasStrictFDerivAt_of_bilinear hu.hasStrictFDerivAt hv.hasStrictFDerivAt).hasStrictDerivAt


theorem derivWithin_of_bilinear (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hu : DifferentiableWithinAt 𝕜 u s x) (hv : DifferentiableWithinAt 𝕜 v s x) :
    derivWithin (fun y => B (u y) (v y)) s x =
      B (u x) (derivWithin v s x) + B (derivWithin u s x) (v x) :=
  (B.hasDerivWithinAt_of_bilinear hu.hasDerivWithinAt hv.hasDerivWithinAt).derivWithin hxs


theorem deriv_of_bilinear (hu : DifferentiableAt 𝕜 u x) (hv : DifferentiableAt 𝕜 v x) :
    deriv (fun y => B (u y) (v y)) x = B (u x) (deriv v x) + B (deriv u x) (v x) :=
  (B.hasDerivAt_of_bilinear hu.hasDerivAt hv.hasDerivAt).deriv


theorem HasDerivWithinAt.smul (hc : HasDerivWithinAt c c' s x) (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun y => c y • f y) (c x • f' + c' • f x) s x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivWithinAt c c' s x
    hf : HasDerivWithinAt f f' s x
    ⊢ HasDerivWithinAt (fun y => HSMul.hSMul (c y) (f y)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  simpa using (HasFDerivWithinAt.smul hc hf).hasDerivWithinAt
  /-
    🎉 no goals
  -/


theorem HasDerivAt.smul (hc : HasDerivAt c c' x) (hf : HasDerivAt f f' x) :
    HasDerivAt (fun y => c y • f y) (c x • f' + c' • f x) x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivAt c c' x
    hf : HasDerivAt f f' x
    ⊢ HasDerivAt (fun y => HSMul.hSMul (c y) (f y)) (HAdd.hAdd (HSMul.hSMul (c x)  …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivWithinAt c c' Set.univ x
    hf : HasDerivWithinAt f f' Set.univ x
    ⊢ HasDerivWithinAt (fun y => HSMul.hSMul (c y) (f y)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  exact hc.smul hf
  /-
    🎉 no goals
  -/


nonrec theorem HasStrictDerivAt.smul (hc : HasStrictDerivAt c c' x) (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun y => c y • f y) (c x • f' + c' • f x) x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasStrictDerivAt c c' x
    hf : HasStrictDerivAt f f' x
    ⊢ HasStrictDerivAt (fun y => HSMul.hSMul (c y) (f y)) (HAdd.hAdd (HSMul.hSMul  …
  -/
  simpa using (hc.smul hf).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem derivWithin_smul (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (hf : DifferentiableWithinAt 𝕜 f s x) :
    derivWithin (fun y => c y • f y) s x = c x • derivWithin f s x + derivWithin c s x • f x :=
  (hc.hasDerivWithinAt.smul hf.hasDerivWithinAt).derivWithin hxs


theorem deriv_smul (hc : DifferentiableAt 𝕜 c x) (hf : DifferentiableAt 𝕜 f x) :
    deriv (fun y => c y • f y) x = c x • deriv f x + deriv c x • f x :=
  (hc.hasDerivAt.smul hf.hasDerivAt).deriv


theorem HasStrictDerivAt.smul_const (hc : HasStrictDerivAt c c' x) (f : F) :
    HasStrictDerivAt (fun y => c y • f) (c' • f) x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasStrictDerivAt c c' x
    f : F
    ⊢ HasStrictDerivAt (fun y => HSMul.hSMul (c y) f) (HSMul.hSMul c' f) x
  -/
  have := hc.smul (hasStrictDerivAt_const x f)
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasStrictDerivAt c c' x
    f : F
    this : HasStrictDerivAt (fun y => HSMul.hSMul (c y) f) (HAdd.hAdd (HSMul.hSMul …
    ⊢ HasStrictDerivAt (fun y => HSMul.hSMul (c y) f) (HSMul.hSMul c' f) x
  -/
  rwa [smul_zero, zero_add] at this
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.smul_const (hc : HasDerivWithinAt c c' s x) (f : F) :
    HasDerivWithinAt (fun y => c y • f) (c' • f) s x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivWithinAt c c' s x
    f : F
    ⊢ HasDerivWithinAt (fun y => HSMul.hSMul (c y) f) (HSMul.hSMul c' f) s x
  -/
  have := hc.smul (hasDerivWithinAt_const x s f)
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivWithinAt c c' s x
    f : F
    this : HasDerivWithinAt (fun y => HSMul.hSMul (c y) f) (HAdd.hAdd (HSMul.hSMul …
    ⊢ HasDerivWithinAt (fun y => HSMul.hSMul (c y) f) (HSMul.hSMul c' f) s x
  -/
  rwa [smul_zero, zero_add] at this
  /-
    🎉 no goals
  -/


theorem HasDerivAt.smul_const (hc : HasDerivAt c c' x) (f : F) :
    HasDerivAt (fun y => c y • f) (c' • f) x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivAt c c' x
    f : F
    ⊢ HasDerivAt (fun y => HSMul.hSMul (c y) f) (HSMul.hSMul c' f) x
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    x : 𝕜
    𝕜' : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜'
    inst✝² : NormedAlgebra 𝕜 𝕜'
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivWithinAt c c' Set.univ x
    f : F
    ⊢ HasDerivWithinAt (fun y => HSMul.hSMul (c y) f) (HSMul.hSMul c' f) Set.univ x
  -/
  exact hc.smul_const f
  /-
    🎉 no goals
  -/


theorem derivWithin_smul_const (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hc : DifferentiableWithinAt 𝕜 c s x) (f : F) :
    derivWithin (fun y => c y • f) s x = derivWithin c s x • f :=
  (hc.hasDerivWithinAt.smul_const f).derivWithin hxs


theorem deriv_smul_const (hc : DifferentiableAt 𝕜 c x) (f : F) :
    deriv (fun y => c y • f) x = deriv c x • f :=
  (hc.hasDerivAt.smul_const f).deriv


nonrec theorem HasStrictDerivAt.const_smul (c : R) (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun y => c • f y) (c • f') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : Module R F
    inst✝¹ : SMulCommClass 𝕜 R F
    inst✝ : ContinuousConstSMul R F
    c : R
    hf : HasStrictDerivAt f f' x
    ⊢ HasStrictDerivAt (fun y => HSMul.hSMul c (f y)) (HSMul.hSMul c f') x
  -/
  simpa using (hf.const_smul c).hasStrictDerivAt
  /-
    🎉 no goals
  -/


nonrec theorem HasDerivAtFilter.const_smul (c : R) (hf : HasDerivAtFilter f f' x L) :
    HasDerivAtFilter (fun y => c • f y) (c • f') x L := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    L : Filter 𝕜
    R : Type u_2
    inst✝³ : Semiring R
    inst✝² : Module R F
    inst✝¹ : SMulCommClass 𝕜 R F
    inst✝ : ContinuousConstSMul R F
    c : R
    hf : HasDerivAtFilter f f' x L
    ⊢ HasDerivAtFilter (fun y => HSMul.hSMul c (f y)) (HSMul.hSMul c f') x L
  -/
  simpa using (hf.const_smul c).hasDerivAtFilter
  /-
    🎉 no goals
  -/


nonrec theorem HasDerivWithinAt.const_smul (c : R) (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun y => c • f y) (c • f') s x :=
  hf.const_smul c


nonrec theorem HasDerivAt.const_smul (c : R) (hf : HasDerivAt f f' x) :
    HasDerivAt (fun y => c • f y) (c • f') x :=
  hf.const_smul c


theorem derivWithin_const_smul (hxs : UniqueDiffWithinAt 𝕜 s x) (c : R)
    (hf : DifferentiableWithinAt 𝕜 f s x) :
    derivWithin (fun y => c • f y) s x = c • derivWithin f s x :=
  (hf.hasDerivWithinAt.const_smul c).derivWithin hxs


theorem deriv_const_smul (c : R) (hf : DifferentiableAt 𝕜 f x) :
    deriv (fun y => c • f y) x = c • deriv f x :=
  (hf.hasDerivAt.const_smul c).deriv


/-- A variant of `deriv_const_smul` without differentiability assumption when the scalar
multiplication is by field elements. -/
lemma deriv_const_smul' {f : 𝕜 → F} {x : 𝕜} {R : Type*} [Field R] [Module R F] [SMulCommClass 𝕜 R F]
    [ContinuousConstSMul R F] (c : R) :
    deriv (fun y ↦ c • f y) x = c • deriv f x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    R : Type u_3
    inst✝³ : Field R
    inst✝² : Module R F
    inst✝¹ : SMulCommClass 𝕜 R F
    inst✝ : ContinuousConstSMul R F
    c : R
    ⊢ Eq (deriv (fun y => HSMul.hSMul c (f y)) x) (HSMul.hSMul c (deriv f x))
  -/
  by_cases hf : DifferentiableAt 𝕜 f x
    /-
      case pos
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      f : 𝕜 → F
      x : 𝕜
      R : Type u_3
      inst✝³ : Field R
      inst✝² : Module R F
      inst✝¹ : SMulCommClass 𝕜 R F
      inst✝ : ContinuousConstSMul R F
      c : R
      hf : DifferentiableAt 𝕜 f x
      ⊢ Eq (deriv (fun y => HSMul.hSMul c (f y)) x) (HSMul.hSMul c (deriv f x))
    -/
  · exact deriv_const_smul c hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      f : 𝕜 → F
      x : 𝕜
      R : Type u_3
      inst✝³ : Field R
      inst✝² : Module R F
      inst✝¹ : SMulCommClass 𝕜 R F
      inst✝ : ContinuousConstSMul R F
      c : R
      hf : Not (DifferentiableAt 𝕜 f x)
      ⊢ Eq (deriv (fun y => HSMul.hSMul c (f y)) x) (HSMul.hSMul c (deriv f x))
    -/
  · rcases eq_or_ne c 0 with rfl | hc
      /-
        case neg.inl
        𝕜 : Type u
        inst✝⁶ : NontriviallyNormedField 𝕜
        F : Type v
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        f : 𝕜 → F
        x : 𝕜
        R : Type u_3
        inst✝³ : Field R
        inst✝² : Module R F
        inst✝¹ : SMulCommClass 𝕜 R F
        inst✝ : ContinuousConstSMul R F
        hf : Not (DifferentiableAt 𝕜 f x)
        ⊢ Eq (deriv (fun y => HSMul.hSMul 0 (f y)) x) (HSMul.hSMul 0 (deriv f x))
      -/
    · simp only [zero_smul, deriv_const']
      /-
        🎉 no goals
      -/
    · have H : ¬DifferentiableAt 𝕜 (fun y ↦ c • f y) x := by
        contrapose! hf
        conv => enter [2, y]; rw [← inv_smul_smul₀ hc (f y)]
        exact DifferentiableAt.const_smul hf c⁻¹
      /-
        case neg.inr
        𝕜 : Type u
        inst✝⁶ : NontriviallyNormedField 𝕜
        F : Type v
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        f : 𝕜 → F
        x : 𝕜
        R : Type u_3
        inst✝³ : Field R
        inst✝² : Module R F
        inst✝¹ : SMulCommClass 𝕜 R F
        inst✝ : ContinuousConstSMul R F
        c : R
        hf : Not (DifferentiableAt 𝕜 f x)
        hc : Ne c 0
        H : Not (DifferentiableAt 𝕜 (fun y => HSMul.hSMul c (f y)) x)
        ⊢ Eq (deriv (fun y => HSMul.hSMul c (f y)) x) (HSMul.hSMul c (deriv f x))
      -/
      rw [deriv_zero_of_not_differentiableAt hf, deriv_zero_of_not_differentiableAt H, smul_zero]
      /-
        🎉 no goals
      -/


theorem HasDerivWithinAt.mul (hc : HasDerivWithinAt c c' s x) (hd : HasDerivWithinAt d d' s x) :
    HasDerivWithinAt (fun y => c y * d y) (c' * d x + c x * d') s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c d : 𝕜 → 𝔸
    c' d' : 𝔸
    hc : HasDerivWithinAt c c' s x
    hd : HasDerivWithinAt d d' s x
    ⊢ HasDerivWithinAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HMul.hMul c' ( …
  -/
  have := (HasFDerivWithinAt.mul' hc hd).hasDerivWithinAt
  rwa [ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.smulRight_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply, one_smul, one_smul,
    add_comm] at this


theorem HasDerivAt.mul (hc : HasDerivAt c c' x) (hd : HasDerivAt d d' x) :
    HasDerivAt (fun y => c y * d y) (c' * d x + c x * d') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c d : 𝕜 → 𝔸
    c' d' : 𝔸
    hc : HasDerivAt c c' x
    hd : HasDerivAt d d' x
    ⊢ HasDerivAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HMul.hMul c' (d x))  …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c d : 𝕜 → 𝔸
    c' d' : 𝔸
    hc : HasDerivWithinAt c c' Set.univ x
    hd : HasDerivWithinAt d d' Set.univ x
    ⊢ HasDerivWithinAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HMul.hMul c' ( …
  -/
  exact hc.mul hd
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.mul (hc : HasStrictDerivAt c c' x) (hd : HasStrictDerivAt d d' x) :
    HasStrictDerivAt (fun y => c y * d y) (c' * d x + c x * d') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c d : 𝕜 → 𝔸
    c' d' : 𝔸
    hc : HasStrictDerivAt c c' x
    hd : HasStrictDerivAt d d' x
    ⊢ HasStrictDerivAt (fun y => HMul.hMul (c y) (d y)) (HAdd.hAdd (HMul.hMul c' ( …
  -/
  have := (HasStrictFDerivAt.mul' hc hd).hasStrictDerivAt
  rwa [ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.smulRight_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply, one_smul, one_smul,
    add_comm] at this


theorem derivWithin_mul (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (hd : DifferentiableWithinAt 𝕜 d s x) :
    derivWithin (fun y => c y * d y) s x = derivWithin c s x * d x + c x * derivWithin d s x :=
  (hc.hasDerivWithinAt.mul hd.hasDerivWithinAt).derivWithin hxs


@[simp]
theorem deriv_mul (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x) :
    deriv (fun y => c y * d y) x = deriv c x * d x + c x * deriv d x :=
  (hc.hasDerivAt.mul hd.hasDerivAt).deriv


theorem HasDerivWithinAt.mul_const (hc : HasDerivWithinAt c c' s x) (d : 𝔸) :
    HasDerivWithinAt (fun y => c y * d) (c' * d) s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c : 𝕜 → 𝔸
    c' : 𝔸
    hc : HasDerivWithinAt c c' s x
    d : 𝔸
    ⊢ HasDerivWithinAt (fun y => HMul.hMul (c y) d) (HMul.hMul c' d) s x
  -/
  convert hc.mul (hasDerivWithinAt_const x s d) using 1
  /-
    case h.e'_9
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c : 𝕜 → 𝔸
    c' : 𝔸
    hc : HasDerivWithinAt c c' s x
    d : 𝔸
    ⊢ Eq (HMul.hMul c' d) (HAdd.hAdd (HMul.hMul c' d) (HMul.hMul (c x) 0))
  -/
  rw [mul_zero, add_zero]
  /-
    🎉 no goals
  -/


theorem HasDerivAt.mul_const (hc : HasDerivAt c c' x) (d : 𝔸) :
    HasDerivAt (fun y => c y * d) (c' * d) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c : 𝕜 → 𝔸
    c' : 𝔸
    hc : HasDerivAt c c' x
    d : 𝔸
    ⊢ HasDerivAt (fun y => HMul.hMul (c y) d) (HMul.hMul c' d) x
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c : 𝕜 → 𝔸
    c' : 𝔸
    hc : HasDerivWithinAt c c' Set.univ x
    d : 𝔸
    ⊢ HasDerivWithinAt (fun y => HMul.hMul (c y) d) (HMul.hMul c' d) Set.univ x
  -/
  exact hc.mul_const d
  /-
    🎉 no goals
  -/


theorem hasDerivAt_mul_const (c : 𝕜) : HasDerivAt (fun x => x * c) c x := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    x c : 𝕜
    ⊢ HasDerivAt (fun x => HMul.hMul x c) c x
  -/
  simpa only [one_mul] using (hasDerivAt_id' x).mul_const c
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.mul_const (hc : HasStrictDerivAt c c' x) (d : 𝔸) :
    HasStrictDerivAt (fun y => c y * d) (c' * d) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c : 𝕜 → 𝔸
    c' : 𝔸
    hc : HasStrictDerivAt c c' x
    d : 𝔸
    ⊢ HasStrictDerivAt (fun y => HMul.hMul (c y) d) (HMul.hMul c' d) x
  -/
  convert hc.mul (hasStrictDerivAt_const x d) using 1
  /-
    case h.e'_9
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    c : 𝕜 → 𝔸
    c' : 𝔸
    hc : HasStrictDerivAt c c' x
    d : 𝔸
    ⊢ Eq (HMul.hMul c' d) (HAdd.hAdd (HMul.hMul c' d) (HMul.hMul (c x) 0))
  -/
  rw [mul_zero, add_zero]
  /-
    🎉 no goals
  -/


theorem derivWithin_mul_const (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (d : 𝔸) : derivWithin (fun y => c y * d) s x = derivWithin c s x * d :=
  (hc.hasDerivWithinAt.mul_const d).derivWithin hxs


theorem deriv_mul_const (hc : DifferentiableAt 𝕜 c x) (d : 𝔸) :
    deriv (fun y => c y * d) x = deriv c x * d :=
  (hc.hasDerivAt.mul_const d).deriv


theorem deriv_mul_const_field (v : 𝕜') : deriv (fun y => u y * v) x = deriv u x * v := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_2
    inst✝¹ : NormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    u : 𝕜 → 𝕜'
    v : 𝕜'
    ⊢ Eq (deriv (fun y => HMul.hMul (u y) v) x) (HMul.hMul (deriv u x) v)
  -/
  by_cases hu : DifferentiableAt 𝕜 u x
    /-
      case pos
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      𝕜' : Type u_2
      inst✝¹ : NormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      u : 𝕜 → 𝕜'
      v : 𝕜'
      hu : DifferentiableAt 𝕜 u x
      ⊢ Eq (deriv (fun y => HMul.hMul (u y) v) x) (HMul.hMul (deriv u x) v)
    -/
  · exact deriv_mul_const hu v
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      𝕜' : Type u_2
      inst✝¹ : NormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      u : 𝕜 → 𝕜'
      v : 𝕜'
      hu : Not (DifferentiableAt 𝕜 u x)
      ⊢ Eq (deriv (fun y => HMul.hMul (u y) v) x) (HMul.hMul (deriv u x) v)
    -/
  · rw [deriv_zero_of_not_differentiableAt hu, zero_mul]
    /-
      case neg
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      x : 𝕜
      𝕜' : Type u_2
      inst✝¹ : NormedField 𝕜'
      inst✝ : NormedAlgebra 𝕜 𝕜'
      u : 𝕜 → 𝕜'
      v : 𝕜'
      hu : Not (DifferentiableAt 𝕜 u x)
      ⊢ Eq (deriv (fun y => HMul.hMul (u y) v) x) 0
    -/
    rcases eq_or_ne v 0 with (rfl | hd)
      /-
        case neg.inl
        𝕜 : Type u
        inst✝² : NontriviallyNormedField 𝕜
        x : 𝕜
        𝕜' : Type u_2
        inst✝¹ : NormedField 𝕜'
        inst✝ : NormedAlgebra 𝕜 𝕜'
        u : 𝕜 → 𝕜'
        hu : Not (DifferentiableAt 𝕜 u x)
        ⊢ Eq (deriv (fun y => HMul.hMul (u y) 0) x) 0
      -/
    · simp only [mul_zero, deriv_const]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        𝕜 : Type u
        inst✝² : NontriviallyNormedField 𝕜
        x : 𝕜
        𝕜' : Type u_2
        inst✝¹ : NormedField 𝕜'
        inst✝ : NormedAlgebra 𝕜 𝕜'
        u : 𝕜 → 𝕜'
        v : 𝕜'
        hu : Not (DifferentiableAt 𝕜 u x)
        hd : Ne v 0
        ⊢ Eq (deriv (fun y => HMul.hMul (u y) v) x) 0
      -/
    · refine deriv_zero_of_not_differentiableAt (mt (fun H => ?_) hu)
      /-
        case neg.inr
        𝕜 : Type u
        inst✝² : NontriviallyNormedField 𝕜
        x : 𝕜
        𝕜' : Type u_2
        inst✝¹ : NormedField 𝕜'
        inst✝ : NormedAlgebra 𝕜 𝕜'
        u : 𝕜 → 𝕜'
        v : 𝕜'
        hu : Not (DifferentiableAt 𝕜 u x)
        hd : Ne v 0
        H : DifferentiableAt 𝕜 (fun y => HMul.hMul (u y) v) x
        ⊢ DifferentiableAt 𝕜 u x
      -/
      simpa only [mul_inv_cancel_right₀ hd] using H.mul_const v⁻¹
      /-
        🎉 no goals
      -/


@[simp]
theorem deriv_mul_const_field' (v : 𝕜') : (deriv fun x => u x * v) = fun x => deriv u x * v :=
  funext fun _ => deriv_mul_const_field v


theorem HasDerivWithinAt.const_mul (c : 𝔸) (hd : HasDerivWithinAt d d' s x) :
    HasDerivWithinAt (fun y => c * d y) (c * d') s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    d : 𝕜 → 𝔸
    d' c : 𝔸
    hd : HasDerivWithinAt d d' s x
    ⊢ HasDerivWithinAt (fun y => HMul.hMul c (d y)) (HMul.hMul c d') s x
  -/
  convert (hasDerivWithinAt_const x s c).mul hd using 1
  /-
    case h.e'_9
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    d : 𝕜 → 𝔸
    d' c : 𝔸
    hd : HasDerivWithinAt d d' s x
    ⊢ Eq (HMul.hMul c d') (HAdd.hAdd (HMul.hMul 0 (d x)) (HMul.hMul c d'))
  -/
  rw [zero_mul, zero_add]
  /-
    🎉 no goals
  -/


theorem HasDerivAt.const_mul (c : 𝔸) (hd : HasDerivAt d d' x) :
    HasDerivAt (fun y => c * d y) (c * d') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    d : 𝕜 → 𝔸
    d' c : 𝔸
    hd : HasDerivAt d d' x
    ⊢ HasDerivAt (fun y => HMul.hMul c (d y)) (HMul.hMul c d') x
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    d : 𝕜 → 𝔸
    d' c : 𝔸
    hd : HasDerivWithinAt d d' Set.univ x
    ⊢ HasDerivWithinAt (fun y => HMul.hMul c (d y)) (HMul.hMul c d') Set.univ x
  -/
  exact hd.const_mul c
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.const_mul (c : 𝔸) (hd : HasStrictDerivAt d d' x) :
    HasStrictDerivAt (fun y => c * d y) (c * d') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    d : 𝕜 → 𝔸
    d' c : 𝔸
    hd : HasStrictDerivAt d d' x
    ⊢ HasStrictDerivAt (fun y => HMul.hMul c (d y)) (HMul.hMul c d') x
  -/
  convert (hasStrictDerivAt_const _ _).mul hd using 1
  /-
    case h.e'_9
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝔸 : Type u_3
    inst✝¹ : NormedRing 𝔸
    inst✝ : NormedAlgebra 𝕜 𝔸
    d : 𝕜 → 𝔸
    d' c : 𝔸
    hd : HasStrictDerivAt d d' x
    ⊢ Eq (HMul.hMul c d') (HAdd.hAdd (HMul.hMul 0 (d x)) (HMul.hMul c d'))
  -/
  rw [zero_mul, zero_add]
  /-
    🎉 no goals
  -/


theorem derivWithin_const_mul (hxs : UniqueDiffWithinAt 𝕜 s x) (c : 𝔸)
    (hd : DifferentiableWithinAt 𝕜 d s x) :
    derivWithin (fun y => c * d y) s x = c * derivWithin d s x :=
  (hd.hasDerivWithinAt.const_mul c).derivWithin hxs


theorem deriv_const_mul (c : 𝔸) (hd : DifferentiableAt 𝕜 d x) :
    deriv (fun y => c * d y) x = c * deriv d x :=
  (hd.hasDerivAt.const_mul c).deriv


theorem deriv_const_mul_field (u : 𝕜') : deriv (fun y => u * v y) x = u * deriv v x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_2
    inst✝¹ : NormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    v : 𝕜 → 𝕜'
    u : 𝕜'
    ⊢ Eq (deriv (fun y => HMul.hMul u (v y)) x) (HMul.hMul u (deriv v x))
  -/
  simp only [mul_comm u, deriv_mul_const_field]
  /-
    🎉 no goals
  -/


@[simp]
theorem deriv_const_mul_field' (u : 𝕜') : (deriv fun x => u * v x) = fun x => u * deriv v x :=
  funext fun _ => deriv_const_mul_field u


theorem HasDerivAt.finset_prod (hf : ∀ i ∈ u, HasDerivAt (f i) (f' i) x) :
    HasDerivAt (∏ i ∈ u, f i ·) (∑ i ∈ u, (∏ j ∈ u.erase i, f j x) • f' i) x := by
  simpa [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply] using
    (HasFDerivAt.finset_prod (fun i hi ↦ (hf i hi).hasFDerivAt)).hasDerivAt


theorem HasDerivWithinAt.finset_prod (hf : ∀ i ∈ u, HasDerivWithinAt (f i) (f' i) s x) :
    HasDerivWithinAt (∏ i ∈ u, f i ·) (∑ i ∈ u, (∏ j ∈ u.erase i, f j x) • f' i) s x := by
  simpa [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply] using
    (HasFDerivWithinAt.finset_prod (fun i hi ↦ (hf i hi).hasFDerivWithinAt)).hasDerivWithinAt


theorem HasStrictDerivAt.finset_prod (hf : ∀ i ∈ u, HasStrictDerivAt (f i) (f' i) x) :
    HasStrictDerivAt (∏ i ∈ u, f i ·) (∑ i ∈ u, (∏ j ∈ u.erase i, f j x) • f' i) x := by
  simpa [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply] using
    (HasStrictFDerivAt.finset_prod (fun i hi ↦ (hf i hi).hasStrictFDerivAt)).hasStrictDerivAt


theorem deriv_finset_prod (hf : ∀ i ∈ u, DifferentiableAt 𝕜 (f i) x) :
    deriv (∏ i ∈ u, f i ·) x = ∑ i ∈ u, (∏ j ∈ u.erase i, f j x) • deriv (f i) x :=
  (HasDerivAt.finset_prod fun i hi ↦ (hf i hi).hasDerivAt).deriv


theorem derivWithin_finset_prod (hxs : UniqueDiffWithinAt 𝕜 s x)
    (hf : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (f i) s x) :
    derivWithin (∏ i ∈ u, f i ·) s x =
      ∑ i ∈ u, (∏ j ∈ u.erase i, f j x) • derivWithin (f i) s x :=
  (HasDerivWithinAt.finset_prod fun i hi ↦ (hf i hi).hasDerivWithinAt).derivWithin hxs


@[fun_prop]
theorem DifferentiableAt.finset_prod (hd : ∀ i ∈ u, DifferentiableAt 𝕜 (f i) x) :
    DifferentiableAt 𝕜 (∏ i ∈ u, f i ·) x := by
  classical
  exact
    (HasDerivAt.finset_prod (fun i hi ↦ DifferentiableAt.hasDerivAt (hd i hi))).differentiableAt


@[fun_prop]
theorem DifferentiableWithinAt.finset_prod (hd : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (f i) s x) :
    DifferentiableWithinAt 𝕜 (∏ i ∈ u, f i ·) s x := by
  classical
  exact (HasDerivWithinAt.finset_prod (fun i hi ↦
    DifferentiableWithinAt.hasDerivWithinAt (hd i hi))).differentiableWithinAt


@[fun_prop]
theorem DifferentiableOn.finset_prod (hd : ∀ i ∈ u, DifferentiableOn 𝕜 (f i) s) :
    DifferentiableOn 𝕜 (∏ i ∈ u, f i ·) s :=
  fun x hx ↦ .finset_prod (fun i hi ↦ hd i hi x hx)


@[fun_prop]
theorem Differentiable.finset_prod (hd : ∀ i ∈ u, Differentiable 𝕜 (f i)) :
    Differentiable 𝕜 (∏ i ∈ u, f i ·) :=
  fun x ↦ .finset_prod (fun i hi ↦ hd i hi x)


theorem HasDerivAt.div_const (hc : HasDerivAt c c' x) (d : 𝕜') :
    HasDerivAt (fun x => c x / d) (c' / d) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_2
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivAt c c' x
    d : 𝕜'
    ⊢ HasDerivAt (fun x => HDiv.hDiv (c x) d) (HDiv.hDiv c' d) x
  -/
  simpa only [div_eq_mul_inv] using hc.mul_const d⁻¹
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.div_const (hc : HasDerivWithinAt c c' s x) (d : 𝕜') :
    HasDerivWithinAt (fun x => c x / d) (c' / d) s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_2
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasDerivWithinAt c c' s x
    d : 𝕜'
    ⊢ HasDerivWithinAt (fun x => HDiv.hDiv (c x) d) (HDiv.hDiv c' d) s x
  -/
  simpa only [div_eq_mul_inv] using hc.mul_const d⁻¹
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.div_const (hc : HasStrictDerivAt c c' x) (d : 𝕜') :
    HasStrictDerivAt (fun x => c x / d) (c' / d) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_2
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c : 𝕜 → 𝕜'
    c' : 𝕜'
    hc : HasStrictDerivAt c c' x
    d : 𝕜'
    ⊢ HasStrictDerivAt (fun x => HDiv.hDiv (c x) d) (HDiv.hDiv c' d) x
  -/
  simpa only [div_eq_mul_inv] using hc.mul_const d⁻¹
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem DifferentiableWithinAt.div_const (hc : DifferentiableWithinAt 𝕜 c s x) (d : 𝕜') :
    DifferentiableWithinAt 𝕜 (fun x => c x / d) s x :=
  (hc.hasDerivWithinAt.div_const _).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.div_const (hc : DifferentiableAt 𝕜 c x) (d : 𝕜') :
    DifferentiableAt 𝕜 (fun x => c x / d) x :=
  (hc.hasDerivAt.div_const _).differentiableAt


@[fun_prop]
theorem DifferentiableOn.div_const (hc : DifferentiableOn 𝕜 c s) (d : 𝕜') :
    DifferentiableOn 𝕜 (fun x => c x / d) s := fun x hx => (hc x hx).div_const d


@[simp, fun_prop]
theorem Differentiable.div_const (hc : Differentiable 𝕜 c) (d : 𝕜') :
    Differentiable 𝕜 fun x => c x / d := fun x => (hc x).div_const d


theorem derivWithin_div_const (hc : DifferentiableWithinAt 𝕜 c s x)
    (d : 𝕜') (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun x => c x / d) s x = derivWithin c s x / d := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    s : Set 𝕜
    𝕜' : Type u_2
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c : 𝕜 → 𝕜'
    hc : DifferentiableWithinAt 𝕜 c s x
    d : 𝕜'
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (derivWithin (fun x => HDiv.hDiv (c x) d) s x) (HDiv.hDiv (derivWithin c  …
  -/
  simp [div_eq_inv_mul, derivWithin_const_mul, hc, hxs]
  /-
    🎉 no goals
  -/


@[simp]
theorem deriv_div_const (d : 𝕜') : deriv (fun x => c x / d) x = deriv c x / d := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    x : 𝕜
    𝕜' : Type u_2
    inst✝¹ : NontriviallyNormedField 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    c : 𝕜 → 𝕜'
    d : 𝕜'
    ⊢ Eq (deriv (fun x => HDiv.hDiv (c x) d) x) (HDiv.hDiv (deriv c x) d)
  -/
  simp only [div_eq_mul_inv, deriv_mul_const_field]
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.clm_comp (hc : HasStrictDerivAt c c' x) (hd : HasStrictDerivAt d d' x) :
    HasStrictDerivAt (fun y => (c y).comp (d y)) (c'.comp (d x) + (c x).comp d') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    d : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) E F
    d' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hc : HasStrictDerivAt c c' x
    hd : HasStrictDerivAt d d' x
    ⊢ HasStrictDerivAt (fun y => (c y).comp (d y)) (HAdd.hAdd (c'.comp (d x)) ((c  …
  -/
  have := (hc.hasStrictFDerivAt.clm_comp hd.hasStrictFDerivAt).hasStrictDerivAt
  rwa [add_apply, comp_apply, comp_apply, smulRight_apply, smulRight_apply, one_apply, one_smul,
    one_smul, add_comm] at this


theorem HasDerivWithinAt.clm_comp (hc : HasDerivWithinAt c c' s x)
    (hd : HasDerivWithinAt d d' s x) :
    HasDerivWithinAt (fun y => (c y).comp (d y)) (c'.comp (d x) + (c x).comp d') s x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : 𝕜
    s : Set 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    d : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) E F
    d' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hc : HasDerivWithinAt c c' s x
    hd : HasDerivWithinAt d d' s x
    ⊢ HasDerivWithinAt (fun y => (c y).comp (d y)) (HAdd.hAdd (c'.comp (d x)) ((c  …
  -/
  have := (hc.hasFDerivWithinAt.clm_comp hd.hasFDerivWithinAt).hasDerivWithinAt
  rwa [add_apply, comp_apply, comp_apply, smulRight_apply, smulRight_apply, one_apply, one_smul,
    one_smul, add_comm] at this


theorem HasDerivAt.clm_comp (hc : HasDerivAt c c' x) (hd : HasDerivAt d d' x) :
    HasDerivAt (fun y => (c y).comp (d y)) (c'.comp (d x) + (c x).comp d') x := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    d : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) E F
    d' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hc : HasDerivAt c c' x
    hd : HasDerivAt d d' x
    ⊢ HasDerivAt (fun y => (c y).comp (d y)) (HAdd.hAdd (c'.comp (d x)) ((c x).com …
  -/
  rw [← hasDerivWithinAt_univ] at *
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    E : Type w
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    x : 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    d : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) E F
    d' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hc : HasDerivWithinAt c c' Set.univ x
    hd : HasDerivWithinAt d d' Set.univ x
    ⊢ HasDerivWithinAt (fun y => (c y).comp (d y)) (HAdd.hAdd (c'.comp (d x)) ((c  …
  -/
  exact hc.clm_comp hd
  /-
    🎉 no goals
  -/


theorem derivWithin_clm_comp (hc : DifferentiableWithinAt 𝕜 c s x)
    (hd : DifferentiableWithinAt 𝕜 d s x) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun y => (c y).comp (d y)) s x =
      (derivWithin c s x).comp (d x) + (c x).comp (derivWithin d s x) :=
  (hc.hasDerivWithinAt.clm_comp hd.hasDerivWithinAt).derivWithin hxs


theorem deriv_clm_comp (hc : DifferentiableAt 𝕜 c x) (hd : DifferentiableAt 𝕜 d x) :
    deriv (fun y => (c y).comp (d y)) x = (deriv c x).comp (d x) + (c x).comp (deriv d x) :=
  (hc.hasDerivAt.clm_comp hd.hasDerivAt).deriv


theorem HasStrictDerivAt.clm_apply (hc : HasStrictDerivAt c c' x) (hu : HasStrictDerivAt u u' x) :
    HasStrictDerivAt (fun y => (c y) (u y)) (c' (u x) + c x u') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    x : 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    u : 𝕜 → F
    u' : F
    hc : HasStrictDerivAt c c' x
    hu : HasStrictDerivAt u u' x
    ⊢ HasStrictDerivAt (fun y => (c y) (u y)) (HAdd.hAdd (c' (u x)) ((c x) u')) x
  -/
  have := (hc.hasStrictFDerivAt.clm_apply hu.hasStrictFDerivAt).hasStrictDerivAt
  rwa [add_apply, comp_apply, flip_apply, smulRight_apply, smulRight_apply, one_apply, one_smul,
    one_smul, add_comm] at this


theorem HasDerivWithinAt.clm_apply (hc : HasDerivWithinAt c c' s x)
    (hu : HasDerivWithinAt u u' s x) :
    HasDerivWithinAt (fun y => (c y) (u y)) (c' (u x) + c x u') s x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    u : 𝕜 → F
    u' : F
    hc : HasDerivWithinAt c c' s x
    hu : HasDerivWithinAt u u' s x
    ⊢ HasDerivWithinAt (fun y => (c y) (u y)) (HAdd.hAdd (c' (u x)) ((c x) u')) s x
  -/
  have := (hc.hasFDerivWithinAt.clm_apply hu.hasFDerivWithinAt).hasDerivWithinAt
  rwa [add_apply, comp_apply, flip_apply, smulRight_apply, smulRight_apply, one_apply, one_smul,
    one_smul, add_comm] at this


theorem HasDerivAt.clm_apply (hc : HasDerivAt c c' x) (hu : HasDerivAt u u' x) :
    HasDerivAt (fun y => (c y) (u y)) (c' (u x) + c x u') x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    x : 𝕜
    G : Type u_2
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    c : 𝕜 → ContinuousLinearMap (RingHom.id 𝕜) F G
    c' : ContinuousLinearMap (RingHom.id 𝕜) F G
    u : 𝕜 → F
    u' : F
    hc : HasDerivAt c c' x
    hu : HasDerivAt u u' x
    ⊢ HasDerivAt (fun y => (c y) (u y)) (HAdd.hAdd (c' (u x)) ((c x) u')) x
  -/
  have := (hc.hasFDerivAt.clm_apply hu.hasFDerivAt).hasDerivAt
  rwa [add_apply, comp_apply, flip_apply, smulRight_apply, smulRight_apply, one_apply, one_smul,
    one_smul, add_comm] at this


theorem derivWithin_clm_apply (hxs : UniqueDiffWithinAt 𝕜 s x) (hc : DifferentiableWithinAt 𝕜 c s x)
    (hu : DifferentiableWithinAt 𝕜 u s x) :
    derivWithin (fun y => (c y) (u y)) s x = derivWithin c s x (u x) + c x (derivWithin u s x) :=
  (hc.hasDerivWithinAt.clm_apply hu.hasDerivWithinAt).derivWithin hxs


theorem deriv_clm_apply (hc : DifferentiableAt 𝕜 c x) (hu : DifferentiableAt 𝕜 u x) :
    deriv (fun y => (c y) (u y)) x = deriv c x (u x) + c x (deriv u x) :=
  (hc.hasDerivAt.clm_apply hu.hasDerivAt).deriv


