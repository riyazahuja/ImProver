theorem isConformalMap_conj : IsConformalMap (conjLIE : ℂ →L[ℝ] ℂ) :=
  conjLIE.toLinearIsometry.isConformalMap


theorem isConformalMap_complex_linear {map : ℂ →L[ℂ] E} (nonzero : map ≠ 0) :
    IsConformalMap (map.restrictScalars ℝ) := by
  have minor₁ : ‖map 1‖ ≠ 0 := by
    simpa only [ContinuousLinearMap.ext_ring_iff, Ne, norm_eq_zero] using nonzero
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : NormedSpace Complex E
    map : ContinuousLinearMap (RingHom.id Complex) Complex E
    nonzero : Ne map 0
    minor₁ : Ne (Norm.norm (map 1)) 0
    ⊢ IsConformalMap (ContinuousLinearMap.restrictScalars Real map)
  -/
  refine ⟨‖map 1‖, minor₁, ⟨‖map 1‖⁻¹ • ((map : ℂ →ₗ[ℂ] E) : ℂ →ₗ[ℝ] E), ?_⟩, ?_⟩
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      ⊢ ∀ (x : Complex), Eq (Norm.norm ((HSMul.hSMul (Inv.inv (Norm.norm (map 1))) ( …
    -/
  · intro x
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x : Complex
      ⊢ Eq (Norm.norm ((HSMul.hSMul (Inv.inv (Norm.norm (map 1))) (↑Real ↑map)) x))  …
    -/
    simp only [LinearMap.smul_apply]
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x : Complex
      ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (map 1))) ((↑Real ↑map) x)))  …
    -/
    have : x = x • (1 : ℂ) := by rw [smul_eq_mul, mul_one]
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x : Complex
      this : Eq x (HSMul.hSMul x 1)
      ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (map 1))) ((↑Real ↑map) x)))  …
    -/
    nth_rw 1 [this]
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x : Complex
      this : Eq x (HSMul.hSMul x 1)
      ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (map 1))) ((↑Real ↑map) (HSMu …
    -/
    rw [LinearMap.coe_restrictScalars]
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x : Complex
      this : Eq x (HSMul.hSMul x 1)
      ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv (Norm.norm (map 1))) (↑map (HSMul.hSMul  …
    -/
    simp only [map.coe_coe, map.map_smul, norm_smul, norm_inv, norm_norm]
    /-
      case refine_1
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x : Complex
      this : Eq x (HSMul.hSMul x 1)
      ⊢ Eq (HMul.hMul (Inv.inv (Norm.norm (map 1))) (HMul.hMul (Norm.norm x) (Norm.n …
    -/
    field_simp only [one_mul]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      ⊢ Eq (ContinuousLinearMap.restrictScalars Real map) (HSMul.hSMul (Norm.norm (m …
    -/
  · ext1
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp`; explicitly supplied simp lemma
    /-
      case refine_2.h
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : NormedSpace Complex E
      map : ContinuousLinearMap (RingHom.id Complex) Complex E
      nonzero : Ne map 0
      minor₁ : Ne (Norm.norm (map 1)) 0
      x✝ : Complex
      ⊢ Eq ((ContinuousLinearMap.restrictScalars Real map) x✝) ((HSMul.hSMul (Norm.n …
    -/
    simp [smul_inv_smul₀ minor₁]
    /-
      🎉 no goals
    -/


theorem isConformalMap_complex_linear_conj {map : ℂ →L[ℂ] E} (nonzero : map ≠ 0) :
    IsConformalMap ((map.restrictScalars ℝ).comp (conjCLE : ℂ →L[ℝ] ℂ)) :=
  (isConformalMap_complex_linear nonzero).comp isConformalMap_conj


theorem IsConformalMap.is_complex_or_conj_linear (h : IsConformalMap g) :
    (∃ map : ℂ →L[ℂ] ℂ, map.restrictScalars ℝ = g) ∨
      ∃ map : ℂ →L[ℂ] ℂ, map.restrictScalars ℝ = g ∘L ↑conjCLE := by
  /-
    g : ContinuousLinearMap (RingHom.id Real) Complex Complex
    h : IsConformalMap g
    ⊢ Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) g) ( …
  -/
  rcases h with ⟨c, -, li, rfl⟩
  obtain ⟨li, rfl⟩ : ∃ li' : ℂ ≃ₗᵢ[ℝ] ℂ, li'.toLinearIsometry = li :=
    ⟨li.toLinearIsometryEquiv rfl, by ext1; rfl⟩
  /-
    case intro.intro.intro.intro
    c : Real
    li : LinearIsometryEquiv (RingHom.id Real) Complex Complex
    ⊢ Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) (HSM …
  -/
  rcases linear_isometry_complex li with ⟨a, rfl | rfl⟩
  -- let rot := c • (a : ℂ) • ContinuousLinearMap.id ℂ ℂ,
    /-
      case intro.intro.intro.intro.intro.inl
      c : Real
      a : Circle
      ⊢ Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) (HSM …
    -/
  · refine Or.inl ⟨c • (a : ℂ) • ContinuousLinearMap.id ℂ ℂ, ?_⟩
    /-
      case intro.intro.intro.intro.intro.inl
      c : Real
      a : Circle
      ⊢ Eq (ContinuousLinearMap.restrictScalars Real (HSMul.hSMul c (HSMul.hSMul (↑a …
    -/
    ext1
    /-
      case intro.intro.intro.intro.intro.inl.h
      c : Real
      a : Circle
      x✝ : Complex
      ⊢ Eq ((ContinuousLinearMap.restrictScalars Real (HSMul.hSMul c (HSMul.hSMul (↑ …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.inr
      c : Real
      a : Circle
      ⊢ Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) (HSM …
    -/
  · refine Or.inr ⟨c • (a : ℂ) • ContinuousLinearMap.id ℂ ℂ, ?_⟩
    /-
      case intro.intro.intro.intro.intro.inr
      c : Real
      a : Circle
      ⊢ Eq (ContinuousLinearMap.restrictScalars Real (HSMul.hSMul c (HSMul.hSMul (↑a …
    -/
    ext1
    /-
      case intro.intro.intro.intro.intro.inr.h
      c : Real
      a : Circle
      x✝ : Complex
      ⊢ Eq ((ContinuousLinearMap.restrictScalars Real (HSMul.hSMul c (HSMul.hSMul (↑ …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- A real continuous linear map on the complex plane is conformal if and only if the map or its
    conjugate is complex linear, and the map is nonvanishing. -/
theorem isConformalMap_iff_is_complex_or_conj_linear :
    IsConformalMap g ↔
      ((∃ map : ℂ →L[ℂ] ℂ, map.restrictScalars ℝ = g) ∨
          ∃ map : ℂ →L[ℂ] ℂ, map.restrictScalars ℝ = g ∘L ↑conjCLE) ∧
        g ≠ 0 := by
  /-
    g : ContinuousLinearMap (RingHom.id Real) Complex Complex
    ⊢ Iff (IsConformalMap g) (And (Or (Exists fun map => Eq (ContinuousLinearMap.r …
  -/
  constructor
    /-
      case mp
      g : ContinuousLinearMap (RingHom.id Real) Complex Complex
      ⊢ IsConformalMap g → And (Or (Exists fun map => Eq (ContinuousLinearMap.restri …
    -/
  · exact fun h => ⟨h.is_complex_or_conj_linear, h.ne_zero⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      g : ContinuousLinearMap (RingHom.id Real) Complex Complex
      ⊢ And (Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) …
    -/
  · rintro ⟨⟨map, rfl⟩ | ⟨map, hmap⟩, h₂⟩
      /-
        case mpr.intro.inl.intro
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        h₂ : Ne (ContinuousLinearMap.restrictScalars Real map) 0
        ⊢ IsConformalMap (ContinuousLinearMap.restrictScalars Real map)
      -/
    · refine isConformalMap_complex_linear ?_
      /-
        case mpr.intro.inl.intro
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        h₂ : Ne (ContinuousLinearMap.restrictScalars Real map) 0
        ⊢ Ne map 0
      -/
      contrapose! h₂ with w
      /-
        case mpr.intro.inl.intro
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        w : Eq map 0
        ⊢ Eq (ContinuousLinearMap.restrictScalars Real map) 0
      -/
      simp only [w, restrictScalars_zero]
      /-
        🎉 no goals
      -/
    · have minor₁ : g = map.restrictScalars ℝ ∘L ↑conjCLE := by
        ext1
        simp only [hmap, coe_comp', ContinuousLinearEquiv.coe_coe, Function.comp_apply,
          conjCLE_apply, starRingEnd_self_apply]
      /-
        case mpr.intro.inr.intro
        g : ContinuousLinearMap (RingHom.id Real) Complex Complex
        h₂ : Ne g 0
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        hmap : Eq (ContinuousLinearMap.restrictScalars Real map) (g.comp ↑Complex.conj …
        minor₁ : Eq g ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.co …
        ⊢ IsConformalMap g
      -/
      rw [minor₁] at h₂ ⊢
      /-
        case mpr.intro.inr.intro
        g : ContinuousLinearMap (RingHom.id Real) Complex Complex
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        h₂ : Ne ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.conjCLE) 0
        hmap : Eq (ContinuousLinearMap.restrictScalars Real map) (g.comp ↑Complex.conj …
        minor₁ : Eq g ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.co …
        ⊢ IsConformalMap ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex …
      -/
      refine isConformalMap_complex_linear_conj ?_
      /-
        case mpr.intro.inr.intro
        g : ContinuousLinearMap (RingHom.id Real) Complex Complex
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        h₂ : Ne ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.conjCLE) 0
        hmap : Eq (ContinuousLinearMap.restrictScalars Real map) (g.comp ↑Complex.conj …
        minor₁ : Eq g ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.co …
        ⊢ Ne map 0
      -/
      contrapose! h₂ with w
      /-
        case mpr.intro.inr.intro
        g : ContinuousLinearMap (RingHom.id Real) Complex Complex
        map : ContinuousLinearMap (RingHom.id Complex) Complex Complex
        hmap : Eq (ContinuousLinearMap.restrictScalars Real map) (g.comp ↑Complex.conj …
        minor₁ : Eq g ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.co …
        w : Eq map 0
        ⊢ Eq ((ContinuousLinearMap.restrictScalars Real map).comp ↑Complex.conjCLE) 0
      -/
      simp only [w, restrictScalars_zero, zero_comp]
      /-
        🎉 no goals
      -/


/-- A real differentiable function of the complex plane into some complex normed space `E` is
conformal at a point `z` if it is holomorphic at that point with a nonvanishing differential.
This is a version of the Cauchy-Riemann equations. -/
theorem DifferentiableAt.conformalAt (h : DifferentiableAt ℂ f z) (hf' : deriv f z ≠ 0) :
    ConformalAt f z := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    f : Complex → E
    h : DifferentiableAt Complex f z
    hf' : Ne (deriv f z) 0
    ⊢ ConformalAt f z
  -/
  rw [conformalAt_iff_isConformalMap_fderiv, (h.hasFDerivAt.restrictScalars ℝ).fderiv]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    f : Complex → E
    h : DifferentiableAt Complex f z
    hf' : Ne (deriv f z) 0
    ⊢ IsConformalMap (ContinuousLinearMap.restrictScalars Real (fderiv Complex f z))
  -/
  apply isConformalMap_complex_linear
  /-
    case nonzero
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    z : Complex
    f : Complex → E
    h : DifferentiableAt Complex f z
    hf' : Ne (deriv f z) 0
    ⊢ Ne (fderiv Complex f z) 0
  -/
  simpa only [Ne, ContinuousLinearMap.ext_ring_iff]
  /-
    🎉 no goals
  -/


/-- A complex function is conformal if and only if the function is holomorphic or antiholomorphic
with a nonvanishing differential. -/
theorem conformalAt_iff_differentiableAt_or_differentiableAt_comp_conj {f : ℂ → ℂ} {z : ℂ} :
    ConformalAt f z ↔
      (DifferentiableAt ℂ f z ∨ DifferentiableAt ℂ (f ∘ conj) (conj z)) ∧ fderiv ℝ f z ≠ 0 := by
  /-
    f : Complex → Complex
    z : Complex
    ⊢ Iff (ConformalAt f z) (And (Or (DifferentiableAt Complex f z) (Differentiabl …
  -/
  rw [conformalAt_iff_isConformalMap_fderiv]
  /-
    f : Complex → Complex
    z : Complex
    ⊢ Iff (IsConformalMap (fderiv Real f z)) (And (Or (DifferentiableAt Complex f  …
  -/
  rw [isConformalMap_iff_is_complex_or_conj_linear]
  /-
    f : Complex → Complex
    z : Complex
    ⊢ Iff (And (Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real …
  -/
  apply and_congr_left
  /-
    case h
    f : Complex → Complex
    z : Complex
    ⊢ Ne (fderiv Real f z) 0 → Iff (Or (Exists fun map => Eq (ContinuousLinearMap. …
  -/
  intro h
  /-
    case h
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    ⊢ Iff (Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) …
  -/
  have h_diff := h.imp_symm fderiv_zero_of_not_differentiableAt
  /-
    case h
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    h_diff : DifferentiableAt Real f z
    ⊢ Iff (Or (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) …
  -/
  apply or_congr
    /-
      case h.h₁
      f : Complex → Complex
      z : Complex
      h : Ne (fderiv Real f z) 0
      h_diff : DifferentiableAt Real f z
      ⊢ Iff (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) (fd …
    -/
  · rw [differentiableAt_iff_restrictScalars ℝ h_diff]
    /-
      🎉 no goals
    -/
  /-
    case h.h₂
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    h_diff : DifferentiableAt Real f z
    ⊢ Iff (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) ((f …
  -/
  rw [← conj_conj z] at h_diff
  /-
    case h.h₂
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    h_diff : DifferentiableAt Real f ((starRingEnd Complex) ((starRingEnd Complex) …
    ⊢ Iff (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) ((f …
  -/
  rw [differentiableAt_iff_restrictScalars ℝ (h_diff.comp _ conjCLE.differentiableAt)]
  /-
    case h.h₂
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    h_diff : DifferentiableAt Real f ((starRingEnd Complex) ((starRingEnd Complex) …
    ⊢ Iff (Exists fun map => Eq (ContinuousLinearMap.restrictScalars Real map) ((f …
  -/
  refine exists_congr fun g => rfl.congr ?_
  /-
    case h.h₂
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    h_diff : DifferentiableAt Real f ((starRingEnd Complex) ((starRingEnd Complex) …
    g : ContinuousLinearMap (RingHom.id Complex) Complex Complex
    ⊢ Eq ((fderiv Real f z).comp ↑Complex.conjCLE) (fderiv Real (Function.comp f ⇑ …
  -/
  have : fderiv ℝ conj (conj z) = _ := conjCLE.fderiv
  /-
    case h.h₂
    f : Complex → Complex
    z : Complex
    h : Ne (fderiv Real f z) 0
    h_diff : DifferentiableAt Real f ((starRingEnd Complex) ((starRingEnd Complex) …
    g : ContinuousLinearMap (RingHom.id Complex) Complex Complex
    this : Eq (fderiv Real (⇑(starRingEnd Complex)) ((starRingEnd Complex) z)) ↑Co …
    ⊢ Eq ((fderiv Real f z).comp ↑Complex.conjCLE) (fderiv Real (Function.comp f ⇑ …
  -/
  simp [fderiv_comp _ h_diff conjCLE.differentiableAt, this, conj_conj]
  /-
    🎉 no goals
  -/


