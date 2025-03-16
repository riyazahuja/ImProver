/-- In a real vector space, a function `f` that approximates a linear equivalence on a subset `s`
can be extended to a homeomorphism of the whole space. -/
theorem exists_homeomorph_extension {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F] {s : Set E}
    {f : E → F} {f' : E ≃L[ℝ] F} {c : ℝ≥0} (hf : ApproximatesLinearOn f (f' : E →L[ℝ] F) s c)
    (hc : Subsingleton E ∨ lipschitzExtensionConstant F * c < ‖(f'.symm : F →L[ℝ] E)‖₊⁻¹) :
    ∃ g : E ≃ₜ F, EqOn f g s := by
  -- the difference `f - f'` is Lipschitz on `s`. It can be extended to a Lipschitz function `u`
  -- on the whole space, with a slightly worse Lipschitz constant. Then `f' + u` will be the
  -- desired homeomorphism.
  obtain ⟨u, hu, uf⟩ :
    ∃ u : E → F, LipschitzWith (lipschitzExtensionConstant F * c) u ∧ EqOn (f - ⇑f') u s :=
    hf.lipschitzOnWith.extend_finite_dimension
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : FiniteDimensional Real F
    s : Set E
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id Real) E F
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt (HMul.hMul (lipschitzExtensionConstant F) c) ( …
    u : E → F
    hu : LipschitzWith (HMul.hMul (lipschitzExtensionConstant F) c) u
    uf : Set.EqOn (HSub.hSub f ⇑f') u s
    ⊢ Exists fun g => Set.EqOn f (⇑g) s
  -/
  let g : E → F := fun x => f' x + u x
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : FiniteDimensional Real F
    s : Set E
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id Real) E F
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt (HMul.hMul (lipschitzExtensionConstant F) c) ( …
    u : E → F
    hu : LipschitzWith (HMul.hMul (lipschitzExtensionConstant F) c) u
    uf : Set.EqOn (HSub.hSub f ⇑f') u s
    g : E → F := fun x => HAdd.hAdd (f' x) (u x)
    ⊢ Exists fun g => Set.EqOn f (⇑g) s
  -/
  have fg : EqOn f g s := fun x hx => by simp_rw [g, ← uf hx, Pi.sub_apply, add_sub_cancel]
  have hg : ApproximatesLinearOn g (f' : E →L[ℝ] F) univ (lipschitzExtensionConstant F * c) := by
    apply LipschitzOnWith.approximatesLinearOn
    rw [lipschitzOnWith_univ]
    convert hu
    ext x
    simp only [g, add_sub_cancel_left, ContinuousLinearEquiv.coe_coe, Pi.sub_apply]
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : FiniteDimensional Real F
    s : Set E
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id Real) E F
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt (HMul.hMul (lipschitzExtensionConstant F) c) ( …
    u : E → F
    hu : LipschitzWith (HMul.hMul (lipschitzExtensionConstant F) c) u
    uf : Set.EqOn (HSub.hSub f ⇑f') u s
    g : E → F := fun x => HAdd.hAdd (f' x) (u x)
    fg : Set.EqOn f g s
    hg : ApproximatesLinearOn g (↑f') Set.univ (HMul.hMul (lipschitzExtensionConst …
    ⊢ Exists fun g => Set.EqOn f (⇑g) s
  -/
  haveI : FiniteDimensional ℝ E := f'.symm.finiteDimensional
  /-
    case intro.intro
    E : Type u_1
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    F : Type u_2
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Real F
    inst✝ : FiniteDimensional Real F
    s : Set E
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id Real) E F
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt (HMul.hMul (lipschitzExtensionConstant F) c) ( …
    u : E → F
    hu : LipschitzWith (HMul.hMul (lipschitzExtensionConstant F) c) u
    uf : Set.EqOn (HSub.hSub f ⇑f') u s
    g : E → F := fun x => HAdd.hAdd (f' x) (u x)
    fg : Set.EqOn f g s
    hg : ApproximatesLinearOn g (↑f') Set.univ (HMul.hMul (lipschitzExtensionConst …
    this : FiniteDimensional Real E
    ⊢ Exists fun g => Set.EqOn f (⇑g) s
  -/
  exact ⟨hg.toHomeomorph g hc, fg⟩
  /-
    🎉 no goals
  -/


