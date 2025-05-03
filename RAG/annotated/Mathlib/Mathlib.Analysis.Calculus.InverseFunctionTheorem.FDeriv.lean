/-- If `f` has derivative `f'` at `a` in the strict sense and `c > 0`, then `f` approximates `f'`
with constant `c` on some neighborhood of `a`. -/
theorem approximates_deriv_on_nhds {f : E → F} {f' : E →L[𝕜] F} {a : E}
    (hf : HasStrictFDerivAt f f' a) {c : ℝ≥0} (hc : Subsingleton E ∨ 0 < c) :
    ∃ s ∈ 𝓝 a, ApproximatesLinearOn f f' s c := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    c : NNReal
    hc : Or (Subsingleton E) (LT.lt 0 c)
    ⊢ Exists fun s => And (Membership.mem (nhds a) s) (ApproximatesLinearOn f f' s …
  -/
  cases' hc with hE hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      a : E
      hf : HasStrictFDerivAt f f' a
      c : NNReal
      hE : Subsingleton E
      ⊢ Exists fun s => And (Membership.mem (nhds a) s) (ApproximatesLinearOn f f' s …
    -/
  · refine ⟨univ, IsOpen.mem_nhds isOpen_univ trivial, fun x _ y _ => ?_⟩
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      a : E
      hf : HasStrictFDerivAt f f' a
      c : NNReal
      hE : Subsingleton E
      x : E
      x✝¹ : Membership.mem Set.univ x
      y : E
      x✝ : Membership.mem Set.univ y
      ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f x) (f y)) (f' (HSub.hSub x y)))) ( …
    -/
    simp [@Subsingleton.elim E hE x y]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    c : NNReal
    hc : LT.lt 0 c
    ⊢ Exists fun s => And (Membership.mem (nhds a) s) (ApproximatesLinearOn f f' s …
  -/
  have := hf.isLittleO.def hc
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    c : NNReal
    hc : LT.lt 0 c
    this : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (HSub.hSub (f x …
    ⊢ Exists fun s => And (Membership.mem (nhds a) s) (ApproximatesLinearOn f f' s …
  -/
  rw [nhds_prod_eq, Filter.Eventually, mem_prod_same_iff] at this
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    c : NNReal
    hc : LT.lt 0 c
    this : Exists fun t => And (Membership.mem (nhds a) t) (HasSubset.Subset (SPro …
    ⊢ Exists fun s => And (Membership.mem (nhds a) s) (ApproximatesLinearOn f f' s …
  -/
  rcases this with ⟨s, has, hs⟩
  /-
    case inr.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    c : NNReal
    hc : LT.lt 0 c
    s : Set E
    has : Membership.mem (nhds a) s
    hs : HasSubset.Subset (SProd.sprod s s) (setOf fun x => LE.le (Norm.norm (HSub …
    ⊢ Exists fun s => And (Membership.mem (nhds a) s) (ApproximatesLinearOn f f' s …
  -/
  exact ⟨s, has, fun x hx y hy => hs (mk_mem_prod hx hy)⟩
  /-
    🎉 no goals
  -/


theorem map_nhds_eq_of_surj [CompleteSpace E] [CompleteSpace F] {f : E → F} {f' : E →L[𝕜] F} {a : E}
    (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) (h : LinearMap.range f' = ⊤) :
    map f (𝓝 a) = 𝓝 (f a) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    h : Eq (LinearMap.range f') Top.top
    ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
  -/
  let f'symm := f'.nonlinearRightInverseOfSurjective h
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    h : Eq (LinearMap.range f') Top.top
    f'symm : f'.NonlinearRightInverse := f'.nonlinearRightInverseOfSurjective h
    ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
  -/
  set c : ℝ≥0 := f'symm.nnnorm⁻¹ / 2 with hc
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    h : Eq (LinearMap.range f') Top.top
    f'symm : f'.NonlinearRightInverse := f'.nonlinearRightInverseOfSurjective h
    c : NNReal := HDiv.hDiv (Inv.inv f'symm.nnnorm) 2
    hc : Eq c (HDiv.hDiv (Inv.inv f'symm.nnnorm) 2)
    ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
  -/
  have f'symm_pos : 0 < f'symm.nnnorm := f'.nonlinearRightInverseOfSurjective_nnnorm_pos h
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    h : Eq (LinearMap.range f') Top.top
    f'symm : f'.NonlinearRightInverse := f'.nonlinearRightInverseOfSurjective h
    c : NNReal := HDiv.hDiv (Inv.inv f'symm.nnnorm) 2
    hc : Eq c (HDiv.hDiv (Inv.inv f'symm.nnnorm) 2)
    f'symm_pos : LT.lt 0 f'symm.nnnorm
    ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
  -/
  have cpos : 0 < c := by simp [hc, half_pos, inv_pos, f'symm_pos]
  obtain ⟨s, s_nhds, hs⟩ : ∃ s ∈ 𝓝 a, ApproximatesLinearOn f f' s c :=
    hf.approximates_deriv_on_nhds (Or.inr cpos)
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    h : Eq (LinearMap.range f') Top.top
    f'symm : f'.NonlinearRightInverse := f'.nonlinearRightInverseOfSurjective h
    c : NNReal := HDiv.hDiv (Inv.inv f'symm.nnnorm) 2
    hc : Eq c (HDiv.hDiv (Inv.inv f'symm.nnnorm) 2)
    f'symm_pos : LT.lt 0 f'symm.nnnorm
    cpos : LT.lt 0 c
    s : Set E
    s_nhds : Membership.mem (nhds a) s
    hs : ApproximatesLinearOn f f' s c
    ⊢ Eq (Filter.map f (nhds a)) (nhds (f a))
  -/
  apply hs.map_nhds_eq f'symm s_nhds (Or.inr (NNReal.half_lt_self _))
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    h : Eq (LinearMap.range f') Top.top
    f'symm : f'.NonlinearRightInverse := f'.nonlinearRightInverseOfSurjective h
    c : NNReal := HDiv.hDiv (Inv.inv f'symm.nnnorm) 2
    hc : Eq c (HDiv.hDiv (Inv.inv f'symm.nnnorm) 2)
    f'symm_pos : LT.lt 0 f'symm.nnnorm
    cpos : LT.lt 0 c
    s : Set E
    s_nhds : Membership.mem (nhds a) s
    hs : ApproximatesLinearOn f f' s c
    ⊢ Ne (Inv.inv f'symm.nnnorm) 0
  -/
  simp [ne_of_gt f'symm_pos]
  /-
    🎉 no goals
  -/


theorem approximates_deriv_on_open_nhds (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    ∃ s : Set E, a ∈ s ∧ IsOpen s ∧
      ApproximatesLinearOn f (f' : E →L[𝕜] F) s (‖(f'.symm : F →L[𝕜] E)‖₊⁻¹ / 2) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f (↑f') a
    ⊢ Exists fun s => And (Membership.mem s a) (And (IsOpen s) (ApproximatesLinear …
  -/
  simp only [← and_assoc]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f (↑f') a
    ⊢ Exists fun s => And (And (Membership.mem s a) (IsOpen s)) (ApproximatesLinea …
  -/
  refine ((nhds_basis_opens a).exists_iff fun s t => ApproximatesLinearOn.mono_set).1 ?_
  exact
    hf.approximates_deriv_on_nhds <|
      f'.subsingleton_or_nnnorm_symm_pos.imp id fun hf' => half_pos <| inv_pos.2 hf'


/-- Given a function with an invertible strict derivative at `a`, returns a `PartialHomeomorph`
with `to_fun = f` and `a ∈ source`. This is a part of the inverse function theorem.
The other part `HasStrictFDerivAt.to_localInverse` states that the inverse function
of this `PartialHomeomorph` has derivative `f'.symm`. -/
def toPartialHomeomorph (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) : PartialHomeomorph E F :=
  ApproximatesLinearOn.toPartialHomeomorph f (Classical.choose hf.approximates_deriv_on_open_nhds)
    (Classical.choose_spec hf.approximates_deriv_on_open_nhds).2.2
    (f'.subsingleton_or_nnnorm_symm_pos.imp id fun hf' =>
      NNReal.half_lt_self <| ne_of_gt <| inv_pos.2 hf')
    (Classical.choose_spec hf.approximates_deriv_on_open_nhds).2.1


@[simp]
theorem toPartialHomeomorph_coe (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    (hf.toPartialHomeomorph f : E → F) = f :=
  rfl


theorem mem_toPartialHomeomorph_source (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    a ∈ (hf.toPartialHomeomorph f).source :=
  (Classical.choose_spec hf.approximates_deriv_on_open_nhds).1


theorem image_mem_toPartialHomeomorph_target (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    f a ∈ (hf.toPartialHomeomorph f).target :=
  (hf.toPartialHomeomorph f).map_source hf.mem_toPartialHomeomorph_source


theorem map_nhds_eq_of_equiv (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    map f (𝓝 a) = 𝓝 (f a) :=
  (hf.toPartialHomeomorph f).map_nhds_eq hf.mem_toPartialHomeomorph_source


/-- Given a function `f` with an invertible derivative, returns a function that is locally inverse
to `f`. -/
def localInverse (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) : F → E :=
  (hf.toPartialHomeomorph f).symm


theorem localInverse_def (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    hf.localInverse f _ _ = (hf.toPartialHomeomorph f).symm :=
  rfl


theorem eventually_left_inverse (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    ∀ᶠ x in 𝓝 a, hf.localInverse f f' a (f x) = x :=
  (hf.toPartialHomeomorph f).eventually_left_inverse hf.mem_toPartialHomeomorph_source


@[simp]
theorem localInverse_apply_image (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    hf.localInverse f f' a (f a) = a :=
  hf.eventually_left_inverse.self_of_nhds


theorem eventually_right_inverse (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    ∀ᶠ y in 𝓝 (f a), f (hf.localInverse f f' a y) = y :=
  (hf.toPartialHomeomorph f).eventually_right_inverse' hf.mem_toPartialHomeomorph_source


theorem localInverse_continuousAt (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    ContinuousAt (hf.localInverse f f' a) (f a) :=
  (hf.toPartialHomeomorph f).continuousAt_symm hf.image_mem_toPartialHomeomorph_target


theorem localInverse_tendsto (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    Tendsto (hf.localInverse f f' a) (𝓝 <| f a) (𝓝 a) :=
  (hf.toPartialHomeomorph f).tendsto_symm hf.mem_toPartialHomeomorph_source


theorem localInverse_unique (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) {g : F → E}
    (hg : ∀ᶠ x in 𝓝 a, g (f x) = x) : ∀ᶠ y in 𝓝 (f a), g y = localInverse f f' a hf y :=
  eventuallyEq_of_left_inv_of_right_inv hg hf.eventually_right_inverse <|
    (hf.toPartialHomeomorph f).tendsto_symm hf.mem_toPartialHomeomorph_source


/-- If `f` has an invertible derivative `f'` at `a` in the sense of strict differentiability `(hf)`,
then the inverse function `hf.localInverse f` has derivative `f'.symm` at `f a`. -/
theorem to_localInverse (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) :
    HasStrictFDerivAt (hf.localInverse f f' a) (f'.symm : F →L[𝕜] E) (f a) :=
  (hf.toPartialHomeomorph f).hasStrictFDerivAt_symm hf.image_mem_toPartialHomeomorph_target <| by
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      a : E
      inst✝ : CompleteSpace E
      hf : HasStrictFDerivAt f (↑f') a
      ⊢ HasStrictFDerivAt (↑(HasStrictFDerivAt.toPartialHomeomorph f hf)) (↑f') (↑(H …
    -/
    simpa [← localInverse_def] using hf
    /-
      🎉 no goals
    -/


/-- If `f : E → F` has an invertible derivative `f'` at `a` in the sense of strict differentiability
and `g (f x) = x` in a neighborhood of `a`, then `g` has derivative `f'.symm` at `f a`.

For a version assuming `f (g y) = y` and continuity of `g` at `f a` but not `[CompleteSpace E]`
see `of_local_left_inverse`. -/
theorem to_local_left_inverse (hf : HasStrictFDerivAt f (f' : E →L[𝕜] F) a) {g : F → E}
    (hg : ∀ᶠ x in 𝓝 a, g (f x) = x) : HasStrictFDerivAt g (f'.symm : F →L[𝕜] E) (f a) :=
  hf.to_localInverse.congr_of_eventuallyEq <| (hf.localInverse_unique hg).mono fun _ => Eq.symm


/-- If a function has an invertible strict derivative at all points, then it is an open map. -/
theorem isOpenMap_of_hasStrictFDerivAt_equiv [CompleteSpace E] {f : E → F} {f' : E → E ≃L[𝕜] F}
    (hf : ∀ x, HasStrictFDerivAt f (f' x : E →L[𝕜] F) x) : IsOpenMap f :=
  isOpenMap_iff_nhds_le.2 fun x => (hf x).map_nhds_eq_of_equiv.ge

@[deprecated (since := "2024-03-23")]
alias open_map_of_strict_fderiv_equiv := isOpenMap_of_hasStrictFDerivAt_equiv

