variable (𝕜) in
/-- If `E` is a locally convex space over `𝕜` (with `RCLike 𝕜`), and `s : Set E` is `ℝ`-convex, then
the closure of `s` and the weak closure of `s` coincide. More precisely, the topological closure
commutes with `toWeakSpace 𝕜 E`.

This holds more generally for any linear equivalence `e : E ≃ₗ[𝕜] F` between locally convex spaces
such that precomposition with `e` and `e.symm` preserves continuity of linear functionals. See
`LinearEquiv.image_closure_of_convex`. -/
theorem Convex.toWeakSpace_closure {s : Set E} (hs : Convex ℝ s) :
    (toWeakSpace 𝕜 E) '' (closure s) = closure (toWeakSpace 𝕜 E '' s) := by
  refine le_antisymm (map_continuous <| toWeakSpaceCLM 𝕜 E).continuousOn.image_closure
    (Set.compl_subset_compl.mp fun x hx ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module Real E
    inst✝⁴ : IsScalarTower Real 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Convex Real s
    x : WeakSpace 𝕜 E
    hx : Membership.mem (HasCompl.compl (Set.image (⇑(toWeakSpace 𝕜 E)) (closure s …
    ⊢ Membership.mem (HasCompl.compl (closure (Set.image (⇑(toWeakSpace 𝕜 E)) s))) x
  -/
  obtain ⟨x, -, rfl⟩ := (toWeakSpace 𝕜 E).toEquiv.image_compl (closure s) |>.symm.subset hx
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module Real E
    inst✝⁴ : IsScalarTower Real 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Convex Real s
    x : E
    hx : Membership.mem (HasCompl.compl (Set.image (⇑(toWeakSpace 𝕜 E)) (closure s …
    ⊢ Membership.mem (HasCompl.compl (closure (Set.image (⇑(toWeakSpace 𝕜 E)) s))) …
  -/
  have : ContinuousSMul ℝ E := IsScalarTower.continuousSMul 𝕜
  obtain ⟨f, u, hus, hux⟩ := RCLike.geometric_hahn_banach_closed_point (𝕜 := 𝕜)
    hs.closure isClosed_closure (by simpa using hx)
  let f' : WeakSpace 𝕜 E →L[𝕜] 𝕜 :=
    { toLinearMap := (f : E →ₗ[𝕜] 𝕜).comp ((toWeakSpace 𝕜 E).symm : WeakSpace 𝕜 E →ₗ[𝕜] E)
      cont := WeakBilin.eval_continuous (topDualPairing 𝕜 E).flip _ }
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module Real E
    inst✝⁴ : IsScalarTower Real 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Convex Real s
    x : E
    hx : Membership.mem (HasCompl.compl (Set.image (⇑(toWeakSpace 𝕜 E)) (closure s …
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    u : Real
    hus : ∀ (a : E), Membership.mem (closure s) a → LT.lt (RCLike.re (f a)) u
    hux : LT.lt u (RCLike.re (f x))
    f' : ContinuousLinearMap (RingHom.id 𝕜) (WeakSpace 𝕜 E) 𝕜 := { toLinearMap :=  …
    ⊢ Membership.mem (HasCompl.compl (closure (Set.image (⇑(toWeakSpace 𝕜 E)) s))) …
  -/
  have hux' : u < RCLike.reCLM.comp (f'.restrictScalars ℝ) (toWeakSpace 𝕜 E x) := by simpa [f']
  have hus' : closure (toWeakSpace 𝕜 E '' s) ⊆
      {y | RCLike.reCLM.comp (f'.restrictScalars ℝ) y ≤ u} := by
    refine closure_minimal ?_ <| isClosed_le (by fun_prop) (by fun_prop)
    rintro - ⟨y, hy, rfl⟩
    simpa [f'] using (hus y <| subset_closure hy).le
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module Real E
    inst✝⁴ : IsScalarTower Real 𝕜 E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    s : Set E
    hs : Convex Real s
    x : E
    hx : Membership.mem (HasCompl.compl (Set.image (⇑(toWeakSpace 𝕜 E)) (closure s …
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    u : Real
    hus : ∀ (a : E), Membership.mem (closure s) a → LT.lt (RCLike.re (f a)) u
    hux : LT.lt u (RCLike.re (f x))
    f' : ContinuousLinearMap (RingHom.id 𝕜) (WeakSpace 𝕜 E) 𝕜 := { toLinearMap :=  …
    hux' : LT.lt u ((RCLike.reCLM.comp (ContinuousLinearMap.restrictScalars Real f …
    hus' : HasSubset.Subset (closure (Set.image (⇑(toWeakSpace 𝕜 E)) s)) (setOf fu …
    ⊢ Membership.mem (HasCompl.compl (closure (Set.image (⇑(toWeakSpace 𝕜 E)) s))) …
  -/
  exact (hux'.not_le <| hus' ·)
  /-
    🎉 no goals
  -/


/-- If `e : E →ₗ[𝕜] F` is a linear map between locally convex spaces, and `f ∘ e` is continuous
for every continuous linear functional `f : F →L[𝕜] 𝕜`, then `e` commutes with the closure on
convex sets. -/
theorem LinearMap.image_closure_of_convex {s : Set E} (hs : Convex ℝ s) (e : E →ₗ[𝕜] F)
    (he : ∀ f : F →L[𝕜] 𝕜, Continuous (e.dualMap f)) :
    e '' (closure s) ⊆ closure (e '' s) := by
  suffices he' : Continuous (toWeakSpace 𝕜 F <| e <| (toWeakSpace 𝕜 E).symm ·) by
    have h_convex : Convex ℝ (e '' s) := hs.linear_image (F := F) e
    rw [← Set.image_subset_image_iff (toWeakSpace 𝕜 F).injective, h_convex.toWeakSpace_closure 𝕜]
    simpa only [Set.image_image, ← hs.toWeakSpace_closure 𝕜, LinearEquiv.symm_apply_apply]
      using he'.continuousOn.image_closure (s := toWeakSpace 𝕜 E '' s)
  exact WeakBilin.continuous_of_continuous_eval _ fun f ↦
    WeakBilin.eval_continuous _ { toLinearMap := e.dualMap f : E →L[𝕜] 𝕜 }


/-- If `e` is a linear isomorphism between two locally convex spaces, and `e` induces (via
precomposition) an isomorphism between their continuous duals, then `e` commutes with the closure
on convex sets.

The hypotheses hold automatically for `e := toWeakSpace 𝕜 E`, see `Convex.toWeakSpace_closure`. -/
theorem LinearEquiv.image_closure_of_convex {s : Set E} (hs : Convex ℝ s) (e : E ≃ₗ[𝕜] F)
    (he₁ : ∀ f : F →L[𝕜] 𝕜, Continuous (e.dualMap f))
    (he₂ : ∀ f : E →L[𝕜] 𝕜, Continuous (e.symm.dualMap f)) :
    e '' (closure s) = closure (e '' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁶ : RCLike 𝕜
    inst✝¹⁵ : AddCommGroup E
    inst✝¹⁴ : Module 𝕜 E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : Module 𝕜 F
    inst✝¹¹ : Module Real E
    inst✝¹⁰ : IsScalarTower Real 𝕜 E
    inst✝⁹ : Module Real F
    inst✝⁸ : IsScalarTower Real 𝕜 F
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul 𝕜 E
    inst✝⁴ : LocallyConvexSpace Real E
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousSMul 𝕜 F
    inst✝ : LocallyConvexSpace Real F
    s : Set E
    hs : Convex Real s
    e : LinearEquiv (RingHom.id 𝕜) E F
    he₁ : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Continuous ⇑(e.dualMap ↑f)
    he₂ : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑(e.symm.dual …
    ⊢ Eq (Set.image (⇑e) (closure s)) (closure (Set.image (⇑e) s))
  -/
  refine le_antisymm ((e : E →ₗ[𝕜] F).image_closure_of_convex hs he₁) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁶ : RCLike 𝕜
    inst✝¹⁵ : AddCommGroup E
    inst✝¹⁴ : Module 𝕜 E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : Module 𝕜 F
    inst✝¹¹ : Module Real E
    inst✝¹⁰ : IsScalarTower Real 𝕜 E
    inst✝⁹ : Module Real F
    inst✝⁸ : IsScalarTower Real 𝕜 F
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul 𝕜 E
    inst✝⁴ : LocallyConvexSpace Real E
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousSMul 𝕜 F
    inst✝ : LocallyConvexSpace Real F
    s : Set E
    hs : Convex Real s
    e : LinearEquiv (RingHom.id 𝕜) E F
    he₁ : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Continuous ⇑(e.dualMap ↑f)
    he₂ : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑(e.symm.dual …
    ⊢ LE.le (closure (Set.image (⇑e) s)) (Set.image (⇑e) (closure s))
  -/
  simp only [Set.le_eq_subset, ← Set.image_subset_image_iff e.symm.injective]
  simpa [Set.image_image]
    using (e.symm : F →ₗ[𝕜] E).image_closure_of_convex (hs.linear_image (e : E →ₗ[𝕜] F)) he₂


/-- If `e` is a linear isomorphism between two locally convex spaces, and `e` induces (via
precomposition) an isomorphism between their continuous duals, then `e` commutes with the closure
on convex sets.

The hypotheses hold automatically for `e := toWeakSpace 𝕜 E`, see `Convex.toWeakSpace_closure`. -/
theorem LinearEquiv.image_closure_of_convex' {s : Set E} (hs : Convex ℝ s) (e : E ≃ₗ[𝕜] F)
    (e_dual : (F →L[𝕜] 𝕜) ≃ (E →L[𝕜] 𝕜))
    (he : ∀ f : F →L[𝕜] 𝕜, (e_dual f : E →ₗ[𝕜] 𝕜) = e.dualMap f) :
    e '' (closure s) = closure (e '' s) := by
  have he' (f : E →L[𝕜] 𝕜) : (e_dual.symm f : F →ₗ[𝕜] 𝕜) = e.symm.dualMap f := by
    simp only [DFunLike.ext'_iff, ContinuousLinearMap.coe_coe] at he ⊢
    have (g : E →L[𝕜] 𝕜) : ⇑g = e_dual.symm g ∘ e := by
      have := he _ ▸ congr(⇑$(e_dual.apply_symm_apply g)).symm
      simpa
    ext x
    conv_rhs => rw [LinearEquiv.dualMap_apply, ContinuousLinearMap.coe_coe, this]
    simp
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁶ : RCLike 𝕜
    inst✝¹⁵ : AddCommGroup E
    inst✝¹⁴ : Module 𝕜 E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : Module 𝕜 F
    inst✝¹¹ : Module Real E
    inst✝¹⁰ : IsScalarTower Real 𝕜 E
    inst✝⁹ : Module Real F
    inst✝⁸ : IsScalarTower Real 𝕜 F
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : TopologicalAddGroup E
    inst✝⁵ : ContinuousSMul 𝕜 E
    inst✝⁴ : LocallyConvexSpace Real E
    inst✝³ : TopologicalSpace F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousSMul 𝕜 F
    inst✝ : LocallyConvexSpace Real F
    s : Set E
    hs : Convex Real s
    e : LinearEquiv (RingHom.id 𝕜) E F
    e_dual : Equiv (ContinuousLinearMap (RingHom.id 𝕜) F 𝕜) (ContinuousLinearMap ( …
    he : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (↑(e_dual f)) (e.dualM …
    he' : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Eq (↑(e_dual.symm f)) (e …
    ⊢ Eq (Set.image (⇑e) (closure s)) (closure (Set.image (⇑e) s))
  -/
  refine e.image_closure_of_convex hs ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁶ : RCLike 𝕜
      inst✝¹⁵ : AddCommGroup E
      inst✝¹⁴ : Module 𝕜 E
      inst✝¹³ : AddCommGroup F
      inst✝¹² : Module 𝕜 F
      inst✝¹¹ : Module Real E
      inst✝¹⁰ : IsScalarTower Real 𝕜 E
      inst✝⁹ : Module Real F
      inst✝⁸ : IsScalarTower Real 𝕜 F
      inst✝⁷ : TopologicalSpace E
      inst✝⁶ : TopologicalAddGroup E
      inst✝⁵ : ContinuousSMul 𝕜 E
      inst✝⁴ : LocallyConvexSpace Real E
      inst✝³ : TopologicalSpace F
      inst✝² : TopologicalAddGroup F
      inst✝¹ : ContinuousSMul 𝕜 F
      inst✝ : LocallyConvexSpace Real F
      s : Set E
      hs : Convex Real s
      e : LinearEquiv (RingHom.id 𝕜) E F
      e_dual : Equiv (ContinuousLinearMap (RingHom.id 𝕜) F 𝕜) (ContinuousLinearMap ( …
      he : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (↑(e_dual f)) (e.dualM …
      he' : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Eq (↑(e_dual.symm f)) (e …
      ⊢ ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Continuous ⇑(e.dualMap ↑f)
    -/
  · simpa [← he] using fun f ↦ map_continuous (e_dual f)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁶ : RCLike 𝕜
      inst✝¹⁵ : AddCommGroup E
      inst✝¹⁴ : Module 𝕜 E
      inst✝¹³ : AddCommGroup F
      inst✝¹² : Module 𝕜 F
      inst✝¹¹ : Module Real E
      inst✝¹⁰ : IsScalarTower Real 𝕜 E
      inst✝⁹ : Module Real F
      inst✝⁸ : IsScalarTower Real 𝕜 F
      inst✝⁷ : TopologicalSpace E
      inst✝⁶ : TopologicalAddGroup E
      inst✝⁵ : ContinuousSMul 𝕜 E
      inst✝⁴ : LocallyConvexSpace Real E
      inst✝³ : TopologicalSpace F
      inst✝² : TopologicalAddGroup F
      inst✝¹ : ContinuousSMul 𝕜 F
      inst✝ : LocallyConvexSpace Real F
      s : Set E
      hs : Convex Real s
      e : LinearEquiv (RingHom.id 𝕜) E F
      e_dual : Equiv (ContinuousLinearMap (RingHom.id 𝕜) F 𝕜) (ContinuousLinearMap ( …
      he : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (↑(e_dual f)) (e.dualM …
      he' : ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Eq (↑(e_dual.symm f)) (e …
      ⊢ ∀ (f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑(e.symm.dualMap  …
    -/
  · simpa [← he'] using fun f ↦ map_continuous (e_dual.symm f)
    /-
      🎉 no goals
    -/

