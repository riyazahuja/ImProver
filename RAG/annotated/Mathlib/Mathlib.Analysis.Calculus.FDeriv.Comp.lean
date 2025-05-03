theorem HasFDerivAtFilter.comp {g : F → G} {g' : F →L[𝕜] G} {L' : Filter F}
    (hg : HasFDerivAtFilter g g' (f x) L') (hf : HasFDerivAtFilter f f' x L) (hL : Tendsto f L L') :
    HasFDerivAtFilter (g ∘ f) (g'.comp f') x L := by
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
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    g : F → G
    g' : ContinuousLinearMap (RingHom.id 𝕜) F G
    L' : Filter F
    hg : HasFDerivAtFilter g g' (f x) L'
    hf : HasFDerivAtFilter f f' x L
    hL : Filter.Tendsto f L L'
    ⊢ HasFDerivAtFilter (Function.comp g f) (g'.comp f') x L
  -/
  let eq₁ := (g'.isBigO_comp _ _).trans_isLittleO hf.isLittleO
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
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    g : F → G
    g' : ContinuousLinearMap (RingHom.id 𝕜) F G
    L' : Filter F
    hg : HasFDerivAtFilter g g' (f x) L'
    hf : HasFDerivAtFilter f f' x L
    hL : Filter.Tendsto f L L'
    eq₁ : Asymptotics.IsLittleO L (fun x' => g' (HSub.hSub (HSub.hSub (f x') (f x) …
    ⊢ HasFDerivAtFilter (Function.comp g f) (g'.comp f') x L
  -/
  let eq₂ := (hg.isLittleO.comp_tendsto hL).trans_isBigO hf.isBigO_sub
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
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    g : F → G
    g' : ContinuousLinearMap (RingHom.id 𝕜) F G
    L' : Filter F
    hg : HasFDerivAtFilter g g' (f x) L'
    hf : HasFDerivAtFilter f f' x L
    hL : Filter.Tendsto f L L'
    eq₁ : Asymptotics.IsLittleO L (fun x' => g' (HSub.hSub (HSub.hSub (f x') (f x) …
    eq₂ : Asymptotics.IsLittleO L (Function.comp (fun x' => HSub.hSub (HSub.hSub ( …
    ⊢ HasFDerivAtFilter (Function.comp g f) (g'.comp f') x L
  -/
  refine .of_isLittleO <| eq₂.triangle <| eq₁.congr_left fun x' => ?_
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
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    L : Filter E
    g : F → G
    g' : ContinuousLinearMap (RingHom.id 𝕜) F G
    L' : Filter F
    hg : HasFDerivAtFilter g g' (f x) L'
    hf : HasFDerivAtFilter f f' x L
    hL : Filter.Tendsto f L L'
    eq₁ : Asymptotics.IsLittleO L (fun x' => g' (HSub.hSub (HSub.hSub (f x') (f x) …
    eq₂ : Asymptotics.IsLittleO L (Function.comp (fun x' => HSub.hSub (HSub.hSub ( …
    x' : E
    ⊢ Eq (g' (HSub.hSub (HSub.hSub (f x') (f x)) (f' (HSub.hSub x' x)))) (HSub.hSu …
  -/
  simp
  /-
    🎉 no goals
  -/

/- A readable version of the previous theorem, a general form of the chain rule. -/

@[fun_prop]
theorem HasFDerivWithinAt.comp {g : F → G} {g' : F →L[𝕜] G} {t : Set F}
    (hg : HasFDerivWithinAt g g' t (f x)) (hf : HasFDerivWithinAt f f' s x) (hst : MapsTo f s t) :
    HasFDerivWithinAt (g ∘ f) (g'.comp f') s x :=
  HasFDerivAtFilter.comp x hg hf <| hf.continuousWithinAt.tendsto_nhdsWithin hst


@[fun_prop]
theorem HasFDerivAt.comp_hasFDerivWithinAt {g : F → G} {g' : F →L[𝕜] G}
    (hg : HasFDerivAt g g' (f x)) (hf : HasFDerivWithinAt f f' s x) :
    HasFDerivWithinAt (g ∘ f) (g'.comp f') s x :=
  hg.comp x hf hf.continuousWithinAt


@[fun_prop]
theorem HasFDerivWithinAt.comp_of_tendsto {g : F → G} {g' : F →L[𝕜] G} {t : Set F}
    (hg : HasFDerivWithinAt g g' t (f x)) (hf : HasFDerivWithinAt f f' s x)
    (hst : Tendsto f (𝓝[s] x) (𝓝[t] f x)) : HasFDerivWithinAt (g ∘ f) (g'.comp f') s x :=
  HasFDerivAtFilter.comp x hg hf hst


@[deprecated (since := "2024-10-18")]
alias HasFDerivWithinAt.comp_of_mem := HasFDerivWithinAt.comp_of_tendsto


/-- The chain rule. -/
@[fun_prop]
theorem HasFDerivAt.comp {g : F → G} {g' : F →L[𝕜] G} (hg : HasFDerivAt g g' (f x))
    (hf : HasFDerivAt f f' x) : HasFDerivAt (g ∘ f) (g'.comp f') x :=
  HasFDerivAtFilter.comp x hg hf hf.continuousAt


@[fun_prop]
theorem DifferentiableWithinAt.comp {g : F → G} {t : Set F}
    (hg : DifferentiableWithinAt 𝕜 g t (f x)) (hf : DifferentiableWithinAt 𝕜 f s x)
    (h : MapsTo f s t) : DifferentiableWithinAt 𝕜 (g ∘ f) s x :=
  (hg.hasFDerivWithinAt.comp x hf.hasFDerivWithinAt h).differentiableWithinAt


@[fun_prop]
theorem DifferentiableWithinAt.comp' {g : F → G} {t : Set F}
    (hg : DifferentiableWithinAt 𝕜 g t (f x)) (hf : DifferentiableWithinAt 𝕜 f s x) :
    DifferentiableWithinAt 𝕜 (g ∘ f) (s ∩ f ⁻¹' t) x :=
  hg.comp x (hf.mono inter_subset_left) inter_subset_right


@[fun_prop]
theorem DifferentiableAt.comp {g : F → G} (hg : DifferentiableAt 𝕜 g (f x))
    (hf : DifferentiableAt 𝕜 f x) : DifferentiableAt 𝕜 (g ∘ f) x :=
  (hg.hasFDerivAt.comp x hf.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableAt.comp_differentiableWithinAt {g : F → G} (hg : DifferentiableAt 𝕜 g (f x))
    (hf : DifferentiableWithinAt 𝕜 f s x) : DifferentiableWithinAt 𝕜 (g ∘ f) s x :=
  hg.differentiableWithinAt.comp x hf (mapsTo_univ _ _)


theorem fderivWithin_comp {g : F → G} {t : Set F} (hg : DifferentiableWithinAt 𝕜 g t (f x))
    (hf : DifferentiableWithinAt 𝕜 f s x) (h : MapsTo f s t) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (g ∘ f) s x = (fderivWithin 𝕜 g t (f x)).comp (fderivWithin 𝕜 f s x) :=
  (hg.hasFDerivWithinAt.comp x hf.hasFDerivWithinAt h).fderivWithin hxs


@[deprecated (since := "2024-10-31")] alias fderivWithin.comp := fderivWithin_comp


theorem fderivWithin_comp_of_eq {g : F → G} {t : Set F} {y : F}
    (hg : DifferentiableWithinAt 𝕜 g t y) (hf : DifferentiableWithinAt 𝕜 f s x) (h : MapsTo f s t)
    (hxs : UniqueDiffWithinAt 𝕜 s x) (hy : f x = y) :
    fderivWithin 𝕜 (g ∘ f) s x = (fderivWithin 𝕜 g t (f x)).comp (fderivWithin 𝕜 f s x) := by
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
    f : E → F
    x : E
    s : Set E
    g : F → G
    t : Set F
    y : F
    hg : DifferentiableWithinAt 𝕜 g t y
    hf : DifferentiableWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    hxs : UniqueDiffWithinAt 𝕜 s x
    hy : Eq (f x) y
    ⊢ Eq (fderivWithin 𝕜 (Function.comp g f) s x) ((fderivWithin 𝕜 g t (f x)).comp …
  -/
  subst hy; exact fderivWithin_comp _ hg hf h hxs
            /-
              🎉 no goals
            -/


/-- A variant for the derivative of a composition, written without `∘`. -/
theorem fderivWithin_comp' {g : F → G} {t : Set F} (hg : DifferentiableWithinAt 𝕜 g t (f x))
    (hf : DifferentiableWithinAt 𝕜 f s x) (h : MapsTo f s t) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun y ↦ g (f y)) s x
      = (fderivWithin 𝕜 g t (f x)).comp (fderivWithin 𝕜 f s x) :=
  fderivWithin_comp _ hg hf h hxs


/-- A variant for the derivative of a composition, written without `∘`. -/
theorem fderivWithin_comp_of_eq' {g : F → G} {t : Set F} {y : F}
    (hg : DifferentiableWithinAt 𝕜 g t y) (hf : DifferentiableWithinAt 𝕜 f s x) (h : MapsTo f s t)
    (hxs : UniqueDiffWithinAt 𝕜 s x) (hy : f x = y) :
    fderivWithin 𝕜 (fun y ↦ g (f y)) s x
      = (fderivWithin 𝕜 g t (f x)).comp (fderivWithin 𝕜 f s x) := by
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
    f : E → F
    x : E
    s : Set E
    g : F → G
    t : Set F
    y : F
    hg : DifferentiableWithinAt 𝕜 g t y
    hf : DifferentiableWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    hxs : UniqueDiffWithinAt 𝕜 s x
    hy : Eq (f x) y
    ⊢ Eq (fderivWithin 𝕜 (fun y => g (f y)) s x) ((fderivWithin 𝕜 g t (f x)).comp  …
  -/
  subst hy; exact fderivWithin_comp _ hg hf h hxs
            /-
              🎉 no goals
            -/


/-- A version of `fderivWithin_comp` that is useful to rewrite the composition of two derivatives
  into a single derivative. This version always applies, but creates a new side-goal `f x = y`. -/
theorem fderivWithin_fderivWithin {g : F → G} {f : E → F} {x : E} {y : F} {s : Set E} {t : Set F}
    (hg : DifferentiableWithinAt 𝕜 g t y) (hf : DifferentiableWithinAt 𝕜 f s x) (h : MapsTo f s t)
    (hxs : UniqueDiffWithinAt 𝕜 s x) (hy : f x = y) (v : E) :
    fderivWithin 𝕜 g t y (fderivWithin 𝕜 f s x v) = fderivWithin 𝕜 (g ∘ f) s x v := by
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
    g : F → G
    f : E → F
    x : E
    y : F
    s : Set E
    t : Set F
    hg : DifferentiableWithinAt 𝕜 g t y
    hf : DifferentiableWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    hxs : UniqueDiffWithinAt 𝕜 s x
    hy : Eq (f x) y
    v : E
    ⊢ Eq ((fderivWithin 𝕜 g t y) ((fderivWithin 𝕜 f s x) v)) ((fderivWithin 𝕜 (Fun …
  -/
  subst y
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
    g : F → G
    f : E → F
    x : E
    s : Set E
    t : Set F
    hf : DifferentiableWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    hxs : UniqueDiffWithinAt 𝕜 s x
    v : E
    hg : DifferentiableWithinAt 𝕜 g t (f x)
    ⊢ Eq ((fderivWithin 𝕜 g t (f x)) ((fderivWithin 𝕜 f s x) v)) ((fderivWithin 𝕜  …
  -/
  rw [fderivWithin_comp x hg hf h hxs, coe_comp', Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- Ternary version of `fderivWithin_comp`, with equality assumptions of basepoints added, in
  order to apply more easily as a rewrite from right-to-left. -/
theorem fderivWithin_comp₃ {g' : G → G'} {g : F → G} {t : Set F} {u : Set G} {y : F} {y' : G}
    (hg' : DifferentiableWithinAt 𝕜 g' u y') (hg : DifferentiableWithinAt 𝕜 g t y)
    (hf : DifferentiableWithinAt 𝕜 f s x) (h2g : MapsTo g t u) (h2f : MapsTo f s t) (h3g : g y = y')
    (h3f : f x = y) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (g' ∘ g ∘ f) s x =
      (fderivWithin 𝕜 g' u y').comp ((fderivWithin 𝕜 g t y).comp (fderivWithin 𝕜 f s x)) := by
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
    f : E → F
    x : E
    s : Set E
    g' : G → G'
    g : F → G
    t : Set F
    u : Set G
    y : F
    y' : G
    hg' : DifferentiableWithinAt 𝕜 g' u y'
    hg : DifferentiableWithinAt 𝕜 g t y
    hf : DifferentiableWithinAt 𝕜 f s x
    h2g : Set.MapsTo g t u
    h2f : Set.MapsTo f s t
    h3g : Eq (g y) y'
    h3f : Eq (f x) y
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (fderivWithin 𝕜 (Function.comp g' (Function.comp g f)) s x) ((fderivWithi …
  -/
  substs h3g h3f
  exact (hg'.hasFDerivWithinAt.comp x (hg.hasFDerivWithinAt.comp x hf.hasFDerivWithinAt h2f) <|
    h2g.comp h2f).fderivWithin hxs


@[deprecated (since := "2024-10-31")] alias fderivWithin.comp₃ := fderivWithin_comp₃


theorem fderiv_comp {g : F → G} (hg : DifferentiableAt 𝕜 g (f x)) (hf : DifferentiableAt 𝕜 f x) :
    fderiv 𝕜 (g ∘ f) x = (fderiv 𝕜 g (f x)).comp (fderiv 𝕜 f x) :=
  (hg.hasFDerivAt.comp x hf.hasFDerivAt).fderiv


@[deprecated (since := "2024-10-31")] alias fderiv.comp := fderiv_comp


/-- A variant for the derivative of a composition, written without `∘`. -/
theorem fderiv_comp' {g : F → G} (hg : DifferentiableAt 𝕜 g (f x)) (hf : DifferentiableAt 𝕜 f x) :
    fderiv 𝕜 (fun y ↦ g (f y)) x = (fderiv 𝕜 g (f x)).comp (fderiv 𝕜 f x) :=
  fderiv_comp x hg hf


theorem fderiv_comp_fderivWithin {g : F → G} (hg : DifferentiableAt 𝕜 g (f x))
    (hf : DifferentiableWithinAt 𝕜 f s x) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (g ∘ f) s x = (fderiv 𝕜 g (f x)).comp (fderivWithin 𝕜 f s x) :=
  (hg.hasFDerivAt.comp_hasFDerivWithinAt x hf.hasFDerivWithinAt).fderivWithin hxs


@[deprecated (since := "2024-10-31")] alias fderiv.comp_fderivWithin := fderiv_comp_fderivWithin


@[fun_prop]
theorem DifferentiableOn.comp {g : F → G} {t : Set F} (hg : DifferentiableOn 𝕜 g t)
    (hf : DifferentiableOn 𝕜 f s) (st : MapsTo f s t) : DifferentiableOn 𝕜 (g ∘ f) s :=
  fun x hx => DifferentiableWithinAt.comp x (hg (f x) (st hx)) (hf x hx) st


@[fun_prop]
theorem Differentiable.comp {g : F → G} (hg : Differentiable 𝕜 g) (hf : Differentiable 𝕜 f) :
    Differentiable 𝕜 (g ∘ f) :=
  fun x => DifferentiableAt.comp x (hg (f x)) (hf x)


@[fun_prop]
theorem Differentiable.comp_differentiableOn {g : F → G} (hg : Differentiable 𝕜 g)
    (hf : DifferentiableOn 𝕜 f s) : DifferentiableOn 𝕜 (g ∘ f) s :=
  hg.differentiableOn.comp hf (mapsTo_univ _ _)


/-- The chain rule for derivatives in the sense of strict differentiability. -/
@[fun_prop]
protected theorem HasStrictFDerivAt.comp {g : F → G} {g' : F →L[𝕜] G}
    (hg : HasStrictFDerivAt g g' (f x)) (hf : HasStrictFDerivAt f f' x) :
    HasStrictFDerivAt (fun x => g (f x)) (g'.comp f') x :=
  .of_isLittleO <|
    ((hg.isLittleO.comp_tendsto (hf.continuousAt.prodMap' hf.continuousAt)).trans_isBigO
        hf.isBigO_sub).triangle <| by
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
        f : E → F
        f' : ContinuousLinearMap (RingHom.id 𝕜) E F
        x : E
        g : F → G
        g' : ContinuousLinearMap (RingHom.id 𝕜) F G
        hg : HasStrictFDerivAt g g' (f x)
        hf : HasStrictFDerivAt f f' x
        ⊢ Asymptotics.IsLittleO (nhds { fst := x, snd := x }) (fun x => HSub.hSub (g'  …
      -/
      simpa only [g'.map_sub, f'.coe_comp'] using (g'.isBigO_comp _ _).trans_isLittleO hf.isLittleO
      /-
        🎉 no goals
      -/


@[fun_prop]
protected theorem Differentiable.iterate {f : E → E} (hf : Differentiable 𝕜 f) (n : ℕ) :
    Differentiable 𝕜 f^[n] :=
  Nat.recOn n differentiable_id fun _ ihn => ihn.comp hf


@[fun_prop]
protected theorem DifferentiableOn.iterate {f : E → E} (hf : DifferentiableOn 𝕜 f s)
    (hs : MapsTo f s s) (n : ℕ) : DifferentiableOn 𝕜 f^[n] s :=
  Nat.recOn n differentiableOn_id fun _ ihn => ihn.comp hf hs


protected theorem HasFDerivAtFilter.iterate {f : E → E} {f' : E →L[𝕜] E}
    (hf : HasFDerivAtFilter f f' x L) (hL : Tendsto f L L) (hx : f x = x) (n : ℕ) :
    HasFDerivAtFilter f^[n] (f' ^ n) x L := by
  induction n with
  | zero => exact hasFDerivAtFilter_id x L
  | succ n ihn =>
    rw [Function.iterate_succ, pow_succ]
    rw [← hx] at ihn
    exact ihn.comp x hf hL


@[fun_prop]
protected theorem HasFDerivAt.iterate {f : E → E} {f' : E →L[𝕜] E} (hf : HasFDerivAt f f' x)
    (hx : f x = x) (n : ℕ) : HasFDerivAt f^[n] (f' ^ n) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivAt f f' x
    hx : Eq (f x) x
    n : Nat
    ⊢ HasFDerivAt (Nat.iterate f n) (HPow.hPow f' n) x
  -/
  refine HasFDerivAtFilter.iterate hf ?_ hx n
  -- Porting note: was `convert hf.continuousAt`
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivAt f f' x
    hx : Eq (f x) x
    n : Nat
    ⊢ Filter.Tendsto f (nhds x) (nhds x)
  -/
  convert hf.continuousAt.tendsto
  /-
    case h.e'_5.h.e'_3
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivAt f f' x
    hx : Eq (f x) x
    n : Nat
    ⊢ Eq x (f x)
  -/
  exact hx.symm
  /-
    🎉 no goals
  -/


@[fun_prop]
protected theorem HasFDerivWithinAt.iterate {f : E → E} {f' : E →L[𝕜] E}
    (hf : HasFDerivWithinAt f f' s x) (hx : f x = x) (hs : MapsTo f s s) (n : ℕ) :
    HasFDerivWithinAt f^[n] (f' ^ n) s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivWithinAt f f' s x
    hx : Eq (f x) x
    hs : Set.MapsTo f s s
    n : Nat
    ⊢ HasFDerivWithinAt (Nat.iterate f n) (HPow.hPow f' n) s x
  -/
  refine HasFDerivAtFilter.iterate hf ?_ hx n
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivWithinAt f f' s x
    hx : Eq (f x) x
    hs : Set.MapsTo f s s
    n : Nat
    ⊢ Filter.Tendsto f (nhdsWithin x s) (nhdsWithin x s)
  -/
  rw [_root_.nhdsWithin] -- Porting note: Added `rw` to get rid of an error
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivWithinAt f f' s x
    hx : Eq (f x) x
    hs : Set.MapsTo f s s
    n : Nat
    ⊢ Filter.Tendsto f (Min.min (nhds x) (Filter.principal s)) (Min.min (nhds x) ( …
  -/
  convert tendsto_inf.2 ⟨hf.continuousWithinAt, _⟩
  /-
    case h.e'_5.h.e'_3.h.e'_3
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    s : Set E
    f : E → E
    f' : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : HasFDerivWithinAt f f' s x
    hx : Eq (f x) x
    hs : Set.MapsTo f s s
    n : Nat
    ⊢ Eq x (f x)
  -/
  exacts [hx.symm, (tendsto_principal_principal.2 hs).mono_left inf_le_right]
  /-
    🎉 no goals
  -/


@[fun_prop]
protected theorem HasStrictFDerivAt.iterate {f : E → E} {f' : E →L[𝕜] E}
    (hf : HasStrictFDerivAt f f' x) (hx : f x = x) (n : ℕ) :
    HasStrictFDerivAt f^[n] (f' ^ n) x := by
  induction n with
  | zero => exact hasStrictFDerivAt_id x
  | succ n ihn =>
    rw [Function.iterate_succ, pow_succ]
    rw [← hx] at ihn
    exact ihn.comp x hf


@[fun_prop]
protected theorem DifferentiableAt.iterate {f : E → E} (hf : DifferentiableAt 𝕜 f x) (hx : f x = x)
    (n : ℕ) : DifferentiableAt 𝕜 f^[n] x :=
  (hf.hasFDerivAt.iterate hx n).differentiableAt


@[fun_prop]
protected theorem DifferentiableWithinAt.iterate {f : E → E} (hf : DifferentiableWithinAt 𝕜 f s x)
    (hx : f x = x) (hs : MapsTo f s s) (n : ℕ) : DifferentiableWithinAt 𝕜 f^[n] s x :=
  (hf.hasFDerivWithinAt.iterate hx hs n).differentiableWithinAt


