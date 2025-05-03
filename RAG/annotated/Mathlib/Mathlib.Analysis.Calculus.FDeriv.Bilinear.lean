@[fun_prop]
theorem IsBoundedBilinearMap.hasStrictFDerivAt (h : IsBoundedBilinearMap 𝕜 b) (p : E × F) :
    HasStrictFDerivAt b (h.deriv p) p := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    b : Prod E F → G
    h : IsBoundedBilinearMap 𝕜 b
    p : Prod E F
    ⊢ HasStrictFDerivAt b (h.deriv p) p
  -/
  simp only [hasStrictFDerivAt_iff_isLittleO]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    b : Prod E F → G
    h : IsBoundedBilinearMap 𝕜 b
    p : Prod E F
    ⊢ Asymptotics.IsLittleO (nhds { fst := p, snd := p }) (fun p_1 => HSub.hSub (H …
  -/
  simp only [← map_add_left_nhds_zero (p, p), isLittleO_map]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    b : Prod E F → G
    h : IsBoundedBilinearMap 𝕜 b
    p : Prod E F
    ⊢ Asymptotics.IsLittleO (nhds 0) (Function.comp (fun p_1 => HSub.hSub (HSub.hS …
  -/
  set T := (E × F) × E × F
  calc
    _ = fun x ↦ h.deriv (x.1 - x.2) (x.2.1, x.1.2) := by
      ext ⟨⟨x₁, y₁⟩, ⟨x₂, y₂⟩⟩
      rcases p with ⟨x, y⟩
      simp only [map_sub, deriv_apply, Function.comp_apply, Prod.mk_add_mk, h.add_right, h.add_left,
        Prod.mk_sub_mk, h.map_sub_left, h.map_sub_right, sub_add_sub_cancel]
      abel
    -- _ =O[𝓝 (0 : T)] fun x ↦ ‖x.1 - x.2‖ * ‖(x.2.1, x.1.2)‖ :=
    --     h.toContinuousLinearMap.deriv₂.isBoundedBilinearMap.isBigO_comp
    -- _ = o[𝓝 0] fun x ↦ ‖x.1 - x.2‖ * 1 := _
    _ =o[𝓝 (0 : T)] fun x ↦ x.1 - x.2 := by
      -- TODO : add 2 `calc` steps instead of the next 3 lines
      refine h.toContinuousLinearMap.deriv₂.isBoundedBilinearMap.isBigO_comp.trans_isLittleO ?_
      suffices (fun x : T ↦ ‖x.1 - x.2‖ * ‖(x.2.1, x.1.2)‖) =o[𝓝 0] fun x ↦ ‖x.1 - x.2‖ * 1 by
        simpa only [mul_one, isLittleO_norm_right] using this
      refine (isBigO_refl _ _).mul_isLittleO ((isLittleO_one_iff _).2 ?_)
      -- TODO: `continuity` fails
      exact (continuous_snd.fst.prod_mk continuous_fst.snd).norm.tendsto' _ _ (by simp)
    _ = _ := by simp [T, Function.comp_def]


@[fun_prop]
theorem IsBoundedBilinearMap.hasFDerivAt (h : IsBoundedBilinearMap 𝕜 b) (p : E × F) :
    HasFDerivAt b (h.deriv p) p :=
  (h.hasStrictFDerivAt p).hasFDerivAt


@[fun_prop]
theorem IsBoundedBilinearMap.hasFDerivWithinAt (h : IsBoundedBilinearMap 𝕜 b) (p : E × F) :
    HasFDerivWithinAt b (h.deriv p) u p :=
  (h.hasFDerivAt p).hasFDerivWithinAt


@[fun_prop]
theorem IsBoundedBilinearMap.differentiableAt (h : IsBoundedBilinearMap 𝕜 b) (p : E × F) :
    DifferentiableAt 𝕜 b p :=
  (h.hasFDerivAt p).differentiableAt


@[fun_prop]
theorem IsBoundedBilinearMap.differentiableWithinAt (h : IsBoundedBilinearMap 𝕜 b) (p : E × F) :
    DifferentiableWithinAt 𝕜 b u p :=
  (h.differentiableAt p).differentiableWithinAt


protected theorem IsBoundedBilinearMap.fderiv (h : IsBoundedBilinearMap 𝕜 b) (p : E × F) :
    fderiv 𝕜 b p = h.deriv p :=
  HasFDerivAt.fderiv (h.hasFDerivAt p)


protected theorem IsBoundedBilinearMap.fderivWithin (h : IsBoundedBilinearMap 𝕜 b) (p : E × F)
    (hxs : UniqueDiffWithinAt 𝕜 u p) : fderivWithin 𝕜 b u p = h.deriv p := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    b : Prod E F → G
    u : Set (Prod E F)
    h : IsBoundedBilinearMap 𝕜 b
    p : Prod E F
    hxs : UniqueDiffWithinAt 𝕜 u p
    ⊢ Eq (fderivWithin 𝕜 b u p) (h.deriv p)
  -/
  rw [DifferentiableAt.fderivWithin (h.differentiableAt p) hxs]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    b : Prod E F → G
    u : Set (Prod E F)
    h : IsBoundedBilinearMap 𝕜 b
    p : Prod E F
    hxs : UniqueDiffWithinAt 𝕜 u p
    ⊢ Eq (fderiv 𝕜 b p) (h.deriv p)
  -/
  exact h.fderiv p
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem IsBoundedBilinearMap.differentiable (h : IsBoundedBilinearMap 𝕜 b) : Differentiable 𝕜 b :=
  fun x => h.differentiableAt x


@[fun_prop]
theorem IsBoundedBilinearMap.differentiableOn (h : IsBoundedBilinearMap 𝕜 b) :
    DifferentiableOn 𝕜 b u :=
  h.differentiable.differentiableOn


@[fun_prop]
theorem ContinuousLinearMap.hasFDerivWithinAt_of_bilinear {f : G' → E} {g : G' → F}
    {f' : G' →L[𝕜] E} {g' : G' →L[𝕜] F} {x : G'} {s : Set G'} (hf : HasFDerivWithinAt f f' s x)
    (hg : HasFDerivWithinAt g g' s x) :
    HasFDerivWithinAt (fun y => B (f y) (g y))
      (B.precompR G' (f x) g' + B.precompL G' f' (g x)) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : G' → E
    g : G' → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) G' E
    g' : ContinuousLinearMap (RingHom.id 𝕜) G' F
    x : G'
    s : Set G'
    hf : HasFDerivWithinAt f f' s x
    hg : HasFDerivWithinAt g g' s x
    ⊢ HasFDerivWithinAt (fun y => (B (f y)) (g y)) (HAdd.hAdd (((ContinuousLinearM …
  -/
  exact (B.isBoundedBilinearMap.hasFDerivAt (f x, g x)).comp_hasFDerivWithinAt x (hf.prod hg)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem ContinuousLinearMap.hasFDerivAt_of_bilinear {f : G' → E} {g : G' → F} {f' : G' →L[𝕜] E}
    {g' : G' →L[𝕜] F} {x : G'} (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x) :
    HasFDerivAt (fun y => B (f y) (g y)) (B.precompR G' (f x) g' + B.precompL G' f' (g x)) x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    G' : Type u_5
    inst✝¹ : NormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    B : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    f : G' → E
    g : G' → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) G' E
    g' : ContinuousLinearMap (RingHom.id 𝕜) G' F
    x : G'
    hf : HasFDerivAt f f' x
    hg : HasFDerivAt g g' x
    ⊢ HasFDerivAt (fun y => (B (f y)) (g y)) (HAdd.hAdd (((ContinuousLinearMap.pre …
  -/
  exact (B.isBoundedBilinearMap.hasFDerivAt (f x, g x)).comp x (hf.prod hg)
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem ContinuousLinearMap.hasStrictFDerivAt_of_bilinear
    {f : G' → E} {g : G' → F} {f' : G' →L[𝕜] E}
    {g' : G' →L[𝕜] F} {x : G'} (hf : HasStrictFDerivAt f f' x) (hg : HasStrictFDerivAt g g' x) :
    HasStrictFDerivAt (fun y => B (f y) (g y))
      (B.precompR G' (f x) g' + B.precompL G' f' (g x)) x :=
  (B.isBoundedBilinearMap.hasStrictFDerivAt (f x, g x)).comp x (hf.prod hg)


theorem ContinuousLinearMap.fderivWithin_of_bilinear {f : G' → E} {g : G' → F} {x : G'} {s : Set G'}
    (hf : DifferentiableWithinAt 𝕜 f s x) (hg : DifferentiableWithinAt 𝕜 g s x)
    (hs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun y => B (f y) (g y)) s x =
      B.precompR G' (f x) (fderivWithin 𝕜 g s x) + B.precompL G' (fderivWithin 𝕜 f s x) (g x) :=
  (B.hasFDerivWithinAt_of_bilinear hf.hasFDerivWithinAt hg.hasFDerivWithinAt).fderivWithin hs


theorem ContinuousLinearMap.fderiv_of_bilinear {f : G' → E} {g : G' → F} {x : G'}
    (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    fderiv 𝕜 (fun y => B (f y) (g y)) x =
      B.precompR G' (f x) (fderiv 𝕜 g x) + B.precompL G' (fderiv 𝕜 f x) (g x) :=
  (B.hasFDerivAt_of_bilinear hf.hasFDerivAt hg.hasFDerivAt).fderiv


