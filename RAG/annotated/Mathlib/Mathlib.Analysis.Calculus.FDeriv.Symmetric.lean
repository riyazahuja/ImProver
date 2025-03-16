variable (𝕜) in
/-- Definition recording that a function has a symmetric second derivative within a set at
a point. This is automatic in most cases of interest (open sets over real or complex vector fields,
or general case for analytic functions), but we can express theorems of calculus using this
as a general assumption, and then specialize to these situations. -/
def IsSymmSndFDerivWithinAt (f : E → F) (s : Set E) (x : E) : Prop :=
  ∀ v w, fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x v w = fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x w v


variable (𝕜) in
/-- Definition recording that a function has a symmetric second derivative at
a point. This is automatic in most cases of interest (open sets over real or complex vector fields,
or general case for analytic functions), but we can express theorems of calculus using this
as a general assumption, and then specialize to these situations. -/
def IsSymmSndFDerivAt (f : E → F) (x : E) : Prop :=
  ∀ v w, fderiv 𝕜 (fderiv 𝕜 f) x v w = fderiv 𝕜 (fderiv 𝕜 f) x w v


protected lemma IsSymmSndFDerivWithinAt.eq (h : IsSymmSndFDerivWithinAt 𝕜 f s x) (v w : E) :
    fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x v w = fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x w v :=
  h v w


protected lemma IsSymmSndFDerivAt.eq
    (h : IsSymmSndFDerivAt 𝕜 f x) (v w : E) :
    fderiv 𝕜 (fderiv 𝕜 f) x v w = fderiv 𝕜 (fderiv 𝕜 f) x w v :=
  h v w


lemma fderivWithin_fderivWithin_eq_of_mem_nhdsWithin (h : t ∈ 𝓝[s] x)
    (hf : ContDiffWithinAt 𝕜 2 f t x) (hs : UniqueDiffOn 𝕜 s) (ht : UniqueDiffOn 𝕜 t) (hx : x ∈ s) :
    fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x = fderivWithin 𝕜 (fderivWithin 𝕜 f t) t x := by
  have A : ∀ᶠ y in 𝓝[s] x, fderivWithin 𝕜 f s y = fderivWithin 𝕜 f t y := by
    have : ∀ᶠ y in 𝓝[s] x, ContDiffWithinAt 𝕜 2 f t y :=
      nhdsWithin_le_iff.2 h (nhdsWithin_mono _ (subset_insert x t) (hf.eventually (by simp)))
    filter_upwards [self_mem_nhdsWithin, this, eventually_eventually_nhdsWithin.2 h]
      with y hy h'y h''y
    exact fderivWithin_of_mem_nhdsWithin h''y (hs y hy) (h'y.differentiableWithinAt one_le_two)
  have : fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x = fderivWithin 𝕜 (fderivWithin 𝕜 f t) s x := by
    apply Filter.EventuallyEq.fderivWithin_eq A
    exact fderivWithin_of_mem_nhdsWithin h (hs x hx) (hf.differentiableWithinAt one_le_two)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : Membership.mem (nhdsWithin x s) t
    hf : ContDiffWithinAt 𝕜 2 f t x
    hs : UniqueDiffOn 𝕜 s
    ht : UniqueDiffOn 𝕜 t
    hx : Membership.mem s x
    A : Filter.Eventually (fun y => Eq (fderivWithin 𝕜 f s y) (fderivWithin 𝕜 f t  …
    this : Eq (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) (fderivWithin 𝕜 (fderivWit …
    ⊢ Eq (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) (fderivWithin 𝕜 (fderivWithin 𝕜 …
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : Membership.mem (nhdsWithin x s) t
    hf : ContDiffWithinAt 𝕜 2 f t x
    hs : UniqueDiffOn 𝕜 s
    ht : UniqueDiffOn 𝕜 t
    hx : Membership.mem s x
    A : Filter.Eventually (fun y => Eq (fderivWithin 𝕜 f s y) (fderivWithin 𝕜 f t  …
    this : Eq (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) (fderivWithin 𝕜 (fderivWit …
    ⊢ Eq (fderivWithin 𝕜 (fderivWithin 𝕜 f t) s x) (fderivWithin 𝕜 (fderivWithin 𝕜 …
  -/
  apply fderivWithin_of_mem_nhdsWithin h (hs x hx)
  exact (hf.fderivWithin_right (m := 1) ht le_rfl
    (mem_of_mem_nhdsWithin hx h)).differentiableWithinAt le_rfl


lemma fderivWithin_fderivWithin_eq_of_eventuallyEq (h : s =ᶠ[𝓝 x] t) :
    fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x = fderivWithin 𝕜 (fderivWithin 𝕜 f t) t x := calc
  fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x
    = fderivWithin 𝕜 (fderivWithin 𝕜 f t) s x :=
      (fderivWithin_eventually_congr_set h).fderivWithin_eq_nhds
  _ = fderivWithin 𝕜 (fderivWithin 𝕜 f t) t x := fderivWithin_congr_set h


lemma fderivWithin_fderivWithin_eq_of_mem_nhds {f : E → F} {x : E} {s : Set E}
    (h : s ∈ 𝓝 x) :
    fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x = fderiv 𝕜 (fderiv 𝕜 f) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    ⊢ Eq (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) (fderiv 𝕜 (fderiv 𝕜 f) x)
  -/
  simp only [← fderivWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    ⊢ Eq (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) (fderivWithin 𝕜 (fderivWithin 𝕜 …
  -/
  apply fderivWithin_fderivWithin_eq_of_eventuallyEq
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : Membership.mem (nhds x) s
    ⊢ (nhds x).EventuallyEq s Set.univ
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[simp] lemma isSymmSndFDerivWithinAt_univ :
    IsSymmSndFDerivWithinAt 𝕜 f univ x ↔ IsSymmSndFDerivAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    ⊢ Iff (IsSymmSndFDerivWithinAt 𝕜 f Set.univ x) (IsSymmSndFDerivAt 𝕜 f x)
  -/
  simp [IsSymmSndFDerivWithinAt, IsSymmSndFDerivAt]
  /-
    🎉 no goals
  -/


theorem IsSymmSndFDerivWithinAt.mono_of_mem_nhdsWithin (h : IsSymmSndFDerivWithinAt 𝕜 f t x)
    (hst : t ∈ 𝓝[s] x) (hf : ContDiffWithinAt 𝕜 2 f t x)
    (hs : UniqueDiffOn 𝕜 s) (ht : UniqueDiffOn 𝕜 t) (hx : x ∈ s) :
    IsSymmSndFDerivWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivWithinAt 𝕜 f t x
    hst : Membership.mem (nhdsWithin x s) t
    hf : ContDiffWithinAt 𝕜 2 f t x
    hs : UniqueDiffOn 𝕜 s
    ht : UniqueDiffOn 𝕜 t
    hx : Membership.mem s x
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  intro v w
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivWithinAt 𝕜 f t x
    hst : Membership.mem (nhdsWithin x s) t
    hf : ContDiffWithinAt 𝕜 2 f t x
    hs : UniqueDiffOn 𝕜 s
    ht : UniqueDiffOn 𝕜 t
    hx : Membership.mem s x
    v w : E
    ⊢ Eq (((fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) v) w) (((fderivWithin 𝕜 (fder …
  -/
  rw [fderivWithin_fderivWithin_eq_of_mem_nhdsWithin hst hf hs ht hx]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivWithinAt 𝕜 f t x
    hst : Membership.mem (nhdsWithin x s) t
    hf : ContDiffWithinAt 𝕜 2 f t x
    hs : UniqueDiffOn 𝕜 s
    ht : UniqueDiffOn 𝕜 t
    hx : Membership.mem s x
    v w : E
    ⊢ Eq (((fderivWithin 𝕜 (fderivWithin 𝕜 f t) t x) v) w) (((fderivWithin 𝕜 (fder …
  -/
  exact h v w
  /-
    🎉 no goals
  -/


theorem IsSymmSndFDerivWithinAt.congr_set (h : IsSymmSndFDerivWithinAt 𝕜 f s x)
    (hst : s =ᶠ[𝓝 x] t) : IsSymmSndFDerivWithinAt 𝕜 f t x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivWithinAt 𝕜 f s x
    hst : (nhds x).EventuallyEq s t
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f t x
  -/
  intro v w
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivWithinAt 𝕜 f s x
    hst : (nhds x).EventuallyEq s t
    v w : E
    ⊢ Eq (((fderivWithin 𝕜 (fderivWithin 𝕜 f t) t x) v) w) (((fderivWithin 𝕜 (fder …
  -/
  rw [fderivWithin_fderivWithin_eq_of_eventuallyEq hst.symm]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s t : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivWithinAt 𝕜 f s x
    hst : (nhds x).EventuallyEq s t
    v w : E
    ⊢ Eq (((fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) v) w) (((fderivWithin 𝕜 (fder …
  -/
  exact h v w
  /-
    🎉 no goals
  -/


theorem isSymmSndFDerivWithinAt_congr_set (hst : s =ᶠ[𝓝 x] t) :
    IsSymmSndFDerivWithinAt 𝕜 f s x ↔ IsSymmSndFDerivWithinAt 𝕜 f t x :=
  ⟨fun h ↦ h.congr_set hst, fun h ↦ h.congr_set hst.symm⟩


theorem IsSymmSndFDerivAt.isSymmSndFDerivWithinAt (h : IsSymmSndFDerivAt 𝕜 f x)
    (hf : ContDiffAt 𝕜 2 f x) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    IsSymmSndFDerivWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    h : IsSymmSndFDerivAt 𝕜 f x
    hf : ContDiffAt 𝕜 2 f x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  simp only [← isSymmSndFDerivWithinAt_univ, ← contDiffWithinAt_univ] at h hf
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    h : IsSymmSndFDerivWithinAt 𝕜 f Set.univ x
    hf : ContDiffWithinAt 𝕜 2 f Set.univ x
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  exact h.mono_of_mem_nhdsWithin univ_mem hf hs uniqueDiffOn_univ hx
  /-
    🎉 no goals
  -/


/-- If a function is analytic within a set at a point, then its second derivative is symmetric. -/
theorem ContDiffWithinAt.isSymmSndFDerivWithinAt_of_omega (hf : ContDiffWithinAt 𝕜 ω f s x)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    IsSymmSndFDerivWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    hf : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  intro v w
  rw [← iteratedFDerivWithin_two_apply' f hs hx, ← iteratedFDerivWithin_two_apply' f hs hx,
    ← hf.iteratedFDerivWithin_comp_perm hs hx _ (Equiv.swap 0 1)]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    hf : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    v w : E
    ⊢ Eq ((iteratedFDerivWithin 𝕜 2 f s x) (Function.comp (Matrix.vecCons v (Matri …
  -/
  congr
  /-
    case h.e_6.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    hf : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    v w : E
    ⊢ Eq (Function.comp (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty)) ⇑(Eq …
  -/
  ext i
  /-
    case h.e_6.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    hf : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    v w : E
    i : Fin 2
    ⊢ Eq (Function.comp (Matrix.vecCons v (Matrix.vecCons w Matrix.vecEmpty)) (⇑(E …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases i <;> rfl
                  /-
                    🎉 no goals
                  -/


/-- If a function is analytic at a point, then its second derivative is symmetric. -/
theorem ContDiffAt.isSymmSndFDerivAt_of_omega (hf : ContDiffAt 𝕜 ω f x) :
    IsSymmSndFDerivAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    hf : ContDiffAt 𝕜 Top.top f x
    ⊢ IsSymmSndFDerivAt 𝕜 f x
  -/
  simp only [← isSymmSndFDerivWithinAt_univ, ← contDiffWithinAt_univ] at hf ⊢
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    hf : ContDiffWithinAt 𝕜 Top.top f Set.univ x
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f Set.univ x
  -/
  exact hf.isSymmSndFDerivWithinAt_of_omega uniqueDiffOn_univ (mem_univ _)
  /-
    🎉 no goals
  -/


/-- Assume that `f` is differentiable inside a convex set `s`, and that its derivative `f'` is
differentiable at a point `x`. Then, given two vectors `v` and `w` pointing inside `s`, one can
Taylor-expand to order two the function `f` on the segment `[x + h v, x + h (v + w)]`, giving a
bilinear estimate for `f (x + hv + hw) - f (x + hv)` in terms of `f' w` and of `f'' ⬝ w`, up to
`o(h^2)`.

This is a technical statement used to show that the second derivative is symmetric. -/
theorem Convex.taylor_approx_two_segment {v w : E} (hv : x + v ∈ interior s)
    (hw : x + v + w ∈ interior s) :
    (fun h : ℝ => f (x + h • v + h • w)
        - f (x + h • v) - h • f' x w - h ^ 2 • f'' v w - (h ^ 2 / 2) • f'' w w) =o[𝓝[>] 0]
      fun h => h ^ 2 := by
  -- it suffices to check that the expression is bounded by `ε * ((‖v‖ + ‖w‖) * ‖w‖) * h^2` for
  -- small enough `h`, for any positive `ε`.
  refine IsLittleO.trans_isBigO
    (isLittleO_iff.2 fun ε εpos => ?_) (isBigO_const_mul_self ((‖v‖ + ‖w‖) * ‖w‖) _ _)
  -- consider a ball of radius `δ` around `x` in which the Taylor approximation for `f''` is
  -- good up to `δ`.
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    hv : Membership.mem (interior s) (HAdd.hAdd x v)
    hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x_1 => LE.le (Norm.norm (HSub.hSub (HSub.hSub (HSub.h …
  -/
  rw [HasFDerivWithinAt, hasFDerivAtFilter_iff_isLittleO, isLittleO_iff] at hx
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
    v w : E
    hv : Membership.mem (interior s) (HAdd.hAdd x v)
    hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x_1 => LE.le (Norm.norm (HSub.hSub (HSub.hSub (HSub.h …
  -/
  rcases Metric.mem_nhdsWithin_iff.1 (hx εpos) with ⟨δ, δpos, sδ⟩
  have E1 : ∀ᶠ h in 𝓝[>] (0 : ℝ), h * (‖v‖ + ‖w‖) < δ := by
    have : Filter.Tendsto (fun h => h * (‖v‖ + ‖w‖)) (𝓝[>] (0 : ℝ)) (𝓝 (0 * (‖v‖ + ‖w‖))) :=
      (continuous_id.mul continuous_const).continuousWithinAt
    apply (tendsto_order.1 this).2 δ
    simpa only [zero_mul] using δpos
  have E2 : ∀ᶠ h in 𝓝[>] (0 : ℝ), (h : ℝ) < 1 :=
    mem_nhdsWithin_of_mem_nhds <| Iio_mem_nhds zero_lt_one
  /-
    case intro.intro
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
    v w : E
    hv : Membership.mem (interior s) (HAdd.hAdd x v)
    hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : GT.gt δ 0
    sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
    E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
    E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
    ⊢ Filter.Eventually (fun x_1 => LE.le (Norm.norm (HSub.hSub (HSub.hSub (HSub.h …
  -/
  filter_upwards [E1, E2, self_mem_nhdsWithin] with h hδ h_lt_1 hpos
  -- we consider `h` small enough that all points under consideration belong to this ball,
  -- and also with `0 < h < 1`.
  /-
    case h
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
    v w : E
    hv : Membership.mem (interior s) (HAdd.hAdd x v)
    hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : GT.gt δ 0
    sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
    E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
    E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
    h : Real
    hδ : LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) (Norm.norm w))) δ
    h_lt_1 : LT.lt h 1
    hpos : Membership.mem (Set.Ioi 0) h
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd ( …
  -/
  replace hpos : 0 < h := hpos
  have xt_mem : ∀ t ∈ Icc (0 : ℝ) 1, x + h • v + (t * h) • w ∈ interior s := by
    intro t ht
    have : x + h • v ∈ interior s := s_conv.add_smul_mem_interior xs hv ⟨hpos, h_lt_1.le⟩
    rw [← smul_smul]
    apply s_conv.interior.add_smul_mem this _ ht
    rw [add_assoc] at hw
    convert s_conv.add_smul_mem_interior xs hw ⟨hpos, h_lt_1.le⟩ using 1
    module
  -- define a function `g` on `[0,1]` (identified with `[v, v + w]`) such that `g 1 - g 0` is the
  -- quantity to be estimated. We will check that its derivative is given by an explicit
  -- expression `g'`, that we can bound. Then the desired bound for `g 1 - g 0` follows from the
  -- mean value inequality.
  let g t :=
    f (x + h • v + (t * h) • w) - (t * h) • f' x w - (t * h ^ 2) • f'' v w -
      ((t * h) ^ 2 / 2) • f'' w w
  set g' := fun t =>
    f' (x + h • v + (t * h) • w) (h • w) - h • f' x w - h ^ 2 • f'' v w - (t * h ^ 2) • f'' w w
    with hg'
  -- check that `g'` is the derivative of `g`, by a straightforward computation
  have g_deriv : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivWithinAt g (g' t) (Icc 0 1) t := by
    intro t ht
    apply_rules [HasDerivWithinAt.sub, HasDerivWithinAt.add]
    · refine (hf _ ?_).comp_hasDerivWithinAt _ ?_
      · exact xt_mem t ht
      apply_rules [HasDerivAt.hasDerivWithinAt, HasDerivAt.const_add, HasDerivAt.smul_const,
        hasDerivAt_mul_const]
    · apply_rules [HasDerivAt.hasDerivWithinAt, HasDerivAt.smul_const, hasDerivAt_mul_const]
    · apply_rules [HasDerivAt.hasDerivWithinAt, HasDerivAt.smul_const, hasDerivAt_mul_const]
    · suffices H : HasDerivWithinAt (fun u => ((u * h) ^ 2 / 2) • f'' w w)
          ((((2 : ℕ) : ℝ) * (t * h) ^ (2 - 1) * (1 * h) / 2) • f'' w w) (Icc 0 1) t by
        convert H using 2
        ring
      apply_rules [HasDerivAt.hasDerivWithinAt, HasDerivAt.smul_const, hasDerivAt_id',
        HasDerivAt.pow, HasDerivAt.mul_const]
  -- check that `g'` is uniformly bounded, with a suitable bound `ε * ((‖v‖ + ‖w‖) * ‖w‖) * h^2`.
  have g'_bound : ∀ t ∈ Ico (0 : ℝ) 1, ‖g' t‖ ≤ ε * ((‖v‖ + ‖w‖) * ‖w‖) * h ^ 2 := by
    intro t ht
    have I : ‖h • v + (t * h) • w‖ ≤ h * (‖v‖ + ‖w‖) :=
      calc
        ‖h • v + (t * h) • w‖ ≤ ‖h • v‖ + ‖(t * h) • w‖ := norm_add_le _ _
        _ = h * ‖v‖ + t * (h * ‖w‖) := by
          simp only [norm_smul, Real.norm_eq_abs, hpos.le, abs_of_nonneg, abs_mul, ht.left,
            mul_assoc]
        _ ≤ h * ‖v‖ + 1 * (h * ‖w‖) := by gcongr; exact ht.2.le
        _ = h * (‖v‖ + ‖w‖) := by ring
    calc
      ‖g' t‖ = ‖(f' (x + h • v + (t * h) • w) - f' x - f'' (h • v + (t * h) • w)) (h • w)‖ := by
        rw [hg']
        congrm ‖?_‖
        simp only [ContinuousLinearMap.sub_apply, ContinuousLinearMap.add_apply,
          ContinuousLinearMap.smul_apply, map_add, map_smul]
        module
      _ ≤ ‖f' (x + h • v + (t * h) • w) - f' x - f'' (h • v + (t * h) • w)‖ * ‖h • w‖ :=
        (ContinuousLinearMap.le_opNorm _ _)
      _ ≤ ε * ‖h • v + (t * h) • w‖ * ‖h • w‖ := by
        gcongr
        have H : x + h • v + (t * h) • w ∈ Metric.ball x δ ∩ interior s := by
          refine ⟨?_, xt_mem t ⟨ht.1, ht.2.le⟩⟩
          rw [add_assoc, add_mem_ball_iff_norm]
          exact I.trans_lt hδ
        simpa only [mem_setOf_eq, add_assoc x, add_sub_cancel_left] using sδ H
      _ ≤ ε * (‖h • v‖ + ‖h • w‖) * ‖h • w‖ := by
        gcongr
        apply (norm_add_le _ _).trans
        gcongr
        simp only [norm_smul, Real.norm_eq_abs, abs_mul, abs_of_nonneg, ht.1, hpos.le, mul_assoc]
        exact mul_le_of_le_one_left (by positivity) ht.2.le
      _ = ε * ((‖v‖ + ‖w‖) * ‖w‖) * h ^ 2 := by
        simp only [norm_smul, Real.norm_eq_abs, abs_mul, abs_of_nonneg, hpos.le]; ring
  -- conclude using the mean value inequality
  have I : ‖g 1 - g 0‖ ≤ ε * ((‖v‖ + ‖w‖) * ‖w‖) * h ^ 2 := by
    simpa only [mul_one, sub_zero] using
      norm_image_sub_le_of_norm_deriv_le_segment' g_deriv g'_bound 1 (right_mem_Icc.2 zero_le_one)
  /-
    case h
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
    v w : E
    hv : Membership.mem (interior s) (HAdd.hAdd x v)
    hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : GT.gt δ 0
    sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
    E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
    E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
    h : Real
    hδ : LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) (Norm.norm w))) δ
    h_lt_1 : LT.lt h 1
    hpos : LT.lt 0 h
    xt_mem : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → Membership.mem (interi …
    g : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAdd.h …
    g' : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAd …
    hg' : Eq g' fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAdd.hAd …
    g_deriv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivWithinAt g (g …
    g'_bound : ∀ (t : Real), Membership.mem (Set.Ico 0 1) t → LE.le (Norm.norm (g' …
    I : LE.le (Norm.norm (HSub.hSub (g 1) (g 0))) (HMul.hMul (HMul.hMul ε (HMul.hM …
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd ( …
  -/
  convert I using 1
    /-
      case h.e'_3
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      s : Set E
      s_conv : Convex Real s
      f : E → F
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
      hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
      x : E
      xs : Membership.mem s x
      hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
      v w : E
      hv : Membership.mem (interior s) (HAdd.hAdd x v)
      hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
      ε : Real
      εpos : LT.lt 0 ε
      δ : Real
      δpos : GT.gt δ 0
      sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
      E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
      E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
      h : Real
      hδ : LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) (Norm.norm w))) δ
      h_lt_1 : LT.lt h 1
      hpos : LT.lt 0 h
      xt_mem : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → Membership.mem (interi …
      g : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAdd.h …
      g' : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAd …
      hg' : Eq g' fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAdd.hAd …
      g_deriv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivWithinAt g (g …
      g'_bound : ∀ (t : Real), Membership.mem (Set.Ico 0 1) t → LE.le (Norm.norm (g' …
      I : LE.le (Norm.norm (HSub.hSub (g 1) (g 0))) (HMul.hMul (HMul.hMul ε (HMul.hM …
      ⊢ Eq (Norm.norm (HSub.hSub (HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAd …
    -/
  · congr 1
    simp only [g, Nat.one_ne_zero, add_zero, one_mul, zero_div, zero_mul, sub_zero,
      zero_smul, Ne, not_false_iff, zero_pow, reduceCtorEq]
    /-
      case h.e'_3.e_a
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      s : Set E
      s_conv : Convex Real s
      f : E → F
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
      hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
      x : E
      xs : Membership.mem s x
      hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
      v w : E
      hv : Membership.mem (interior s) (HAdd.hAdd x v)
      hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
      ε : Real
      εpos : LT.lt 0 ε
      δ : Real
      δpos : GT.gt δ 0
      sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
      E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
      E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
      h : Real
      hδ : LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) (Norm.norm w))) δ
      h_lt_1 : LT.lt h 1
      hpos : LT.lt 0 h
      xt_mem : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → Membership.mem (interi …
      g : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAdd.h …
      g' : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAd …
      hg' : Eq g' fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAdd.hAd …
      g_deriv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivWithinAt g (g …
      g'_bound : ∀ (t : Real), Membership.mem (Set.Ico 0 1) t → LE.le (Norm.norm (g' …
      I : LE.le (Norm.norm (HSub.hSub (g 1) (g 0))) (HMul.hMul (HMul.hMul ε (HMul.hM …
      ⊢ Eq (HSub.hSub (HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAdd.hAdd x (H …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      s : Set E
      s_conv : Convex Real s
      f : E → F
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
      hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
      x : E
      xs : Membership.mem s x
      hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
      v w : E
      hv : Membership.mem (interior s) (HAdd.hAdd x v)
      hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
      ε : Real
      εpos : LT.lt 0 ε
      δ : Real
      δpos : GT.gt δ 0
      sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
      E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
      E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
      h : Real
      hδ : LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) (Norm.norm w))) δ
      h_lt_1 : LT.lt h 1
      hpos : LT.lt 0 h
      xt_mem : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → Membership.mem (interi …
      g : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAdd.h …
      g' : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAd …
      hg' : Eq g' fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAdd.hAd …
      g_deriv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivWithinAt g (g …
      g'_bound : ∀ (t : Real), Membership.mem (Set.Ico 0 1) t → LE.le (Norm.norm (g' …
      I : LE.le (Norm.norm (HSub.hSub (g 1) (g 0))) (HMul.hMul (HMul.hMul ε (HMul.hM …
      ⊢ Eq (HMul.hMul ε (Norm.norm (HMul.hMul (HMul.hMul (HAdd.hAdd (Norm.norm v) (N …
    -/
  · simp (discharger := positivity) only [Real.norm_eq_abs, abs_mul, abs_of_nonneg, abs_pow]
    /-
      case h.e'_4
      E : Type u_1
      F : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      s : Set E
      s_conv : Convex Real s
      f : E → F
      f' : E → ContinuousLinearMap (RingHom.id Real) E F
      f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
      hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
      x : E
      xs : Membership.mem s x
      hx : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm  …
      v w : E
      hv : Membership.mem (interior s) (HAdd.hAdd x v)
      hw : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x v) w)
      ε : Real
      εpos : LT.lt 0 ε
      δ : Real
      δpos : GT.gt δ 0
      sδ : HasSubset.Subset (Inter.inter (Metric.ball x δ) (interior s)) (setOf fun  …
      E1 : Filter.Eventually (fun h => LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) ( …
      E2 : Filter.Eventually (fun h => LT.lt h 1) (nhdsWithin 0 (Set.Ioi 0))
      h : Real
      hδ : LT.lt (HMul.hMul h (HAdd.hAdd (Norm.norm v) (Norm.norm w))) δ
      h_lt_1 : LT.lt h 1
      hpos : LT.lt 0 h
      xt_mem : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → Membership.mem (interi …
      g : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub (f (HAdd.hAdd (HAdd.h …
      g' : Real → F := fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAd …
      hg' : Eq g' fun t => HSub.hSub (HSub.hSub (HSub.hSub ((f' (HAdd.hAdd (HAdd.hAd …
      g_deriv : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivWithinAt g (g …
      g'_bound : ∀ (t : Real), Membership.mem (Set.Ico 0 1) t → LE.le (Norm.norm (g' …
      I : LE.le (Norm.norm (HSub.hSub (g 1) (g 0))) (HMul.hMul (HMul.hMul ε (HMul.hM …
      ⊢ Eq (HMul.hMul ε (HMul.hMul (HMul.hMul (HAdd.hAdd (Norm.norm v) (Norm.norm w) …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- One can get `f'' v w` as the limit of `h ^ (-2)` times the alternate sum of the values of `f`
along the vertices of a quadrilateral with sides `h v` and `h w` based at `x`.
In a setting where `f` is not guaranteed to be continuous at `f`, we can still
get this if we use a quadrilateral based at `h v + h w`. -/
theorem Convex.isLittleO_alternate_sum_square {v w : E} (h4v : x + (4 : ℝ) • v ∈ interior s)
    (h4w : x + (4 : ℝ) • w ∈ interior s) :
    (fun h : ℝ => f (x + h • (2 • v + 2 • w)) + f (x + h • (v + w))
        - f (x + h • (2 • v + w)) - f (x + h • (v + 2 • w)) - h ^ 2 • f'' v w) =o[𝓝[>] 0]
      fun h => h ^ 2 := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HSub.h …
  -/
  have A : (1 : ℝ) / 2 ∈ Ioc (0 : ℝ) 1 := ⟨by norm_num, by norm_num⟩
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Membership.mem (Set.Ioc 0 1) (1 / 2)
    ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HSub.h …
  -/
  have B : (1 : ℝ) / 2 ∈ Icc (0 : ℝ) 1 := ⟨by norm_num, by norm_num⟩
  have h2v2w : x + (2 : ℝ) • v + (2 : ℝ) • w ∈ interior s := by
    convert s_conv.interior.add_smul_sub_mem h4v h4w B using 1
    module
  have h2vww : x + (2 • v + w) + w ∈ interior s := by
    convert h2v2w using 1
    module
  have h2v : x + (2 : ℝ) • v ∈ interior s := by
    convert s_conv.add_smul_sub_mem_interior xs h4v A using 1
    module
  have h2w : x + (2 : ℝ) • w ∈ interior s := by
    convert s_conv.add_smul_sub_mem_interior xs h4w A using 1
    module
  have hvw : x + (v + w) ∈ interior s := by
    convert s_conv.add_smul_sub_mem_interior xs h2v2w A using 1
    module
  have h2vw : x + (2 • v + w) ∈ interior s := by
    convert s_conv.interior.add_smul_sub_mem h2v h2v2w B using 1
    module
  have hvww : x + (v + w) + w ∈ interior s := by
    convert s_conv.interior.add_smul_sub_mem h2w h2v2w B using 1
    module
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Membership.mem (Set.Ioc 0 1) (1 / 2)
    B : Membership.mem (Set.Icc 0 1) (1 / 2)
    h2v2w : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HSMul.hSMul 2 v)) …
    h2vww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd (HSMul. …
    h2v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 v))
    h2w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 w))
    hvw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd v w))
    h2vw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd (HSMul.hSMul 2 v) w))
    hvww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd v w)) w)
    ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HSub.h …
  -/
  have TA1 := s_conv.taylor_approx_two_segment hf xs hx h2vw h2vww
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Membership.mem (Set.Ioc 0 1) (1 / 2)
    B : Membership.mem (Set.Icc 0 1) (1 / 2)
    h2v2w : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HSMul.hSMul 2 v)) …
    h2vww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd (HSMul. …
    h2v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 v))
    h2w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 w))
    hvw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd v w))
    h2vw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd (HSMul.hSMul 2 v) w))
    hvww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd v w)) w)
    TA1 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HSub.h …
  -/
  have TA2 := s_conv.taylor_approx_two_segment hf xs hx hvw hvww
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Membership.mem (Set.Ioc 0 1) (1 / 2)
    B : Membership.mem (Set.Icc 0 1) (1 / 2)
    h2v2w : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HSMul.hSMul 2 v)) …
    h2vww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd (HSMul. …
    h2v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 v))
    h2w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 w))
    hvw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd v w))
    h2vw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd (HSMul.hSMul 2 v) w))
    hvww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd v w)) w)
    TA1 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    TA2 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    ⊢ Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HSub.h …
  -/
  convert TA1.sub TA2 using 1
  /-
    case h.e'_7
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Membership.mem (Set.Ioc 0 1) (1 / 2)
    B : Membership.mem (Set.Icc 0 1) (1 / 2)
    h2v2w : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HSMul.hSMul 2 v)) …
    h2vww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd (HSMul. …
    h2v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 v))
    h2w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 w))
    hvw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd v w))
    h2vw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd (HSMul.hSMul 2 v) w))
    hvww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd v w)) w)
    TA1 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    TA2 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    ⊢ Eq (fun h => HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (f (HAdd.hAdd x (HSM …
  -/
  ext h
  simp only [two_smul, smul_add, ← add_assoc, ContinuousLinearMap.map_add,
    ContinuousLinearMap.add_apply, Pi.smul_apply, ContinuousLinearMap.coe_smul',
    ContinuousLinearMap.map_smul]
  /-
    case h.e'_7.h
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Membership.mem (Set.Ioc 0 1) (1 / 2)
    B : Membership.mem (Set.Icc 0 1) (1 / 2)
    h2v2w : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HSMul.hSMul 2 v)) …
    h2vww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd (HSMul. …
    h2v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 v))
    h2w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 2 w))
    hvw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd v w))
    h2vw : Membership.mem (interior s) (HAdd.hAdd x (HAdd.hAdd (HSMul.hSMul 2 v) w))
    hvww : Membership.mem (interior s) (HAdd.hAdd (HAdd.hAdd x (HAdd.hAdd v w)) w)
    TA1 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    TA2 : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSub.hSub (HS …
    h : Real
    ⊢ Eq (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (f (HAdd.hAdd (HAdd.hAdd (HAd …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- Assume that `f` is differentiable inside a convex set `s`, and that its derivative `f'` is
differentiable at a point `x`. Then, given two vectors `v` and `w` pointing inside `s`, one
has `f'' v w = f'' w v`. Superseded by `Convex.second_derivative_within_at_symmetric`, which
removes the assumption that `v` and `w` point inside `s`. -/
theorem Convex.second_derivative_within_at_symmetric_of_mem_interior {v w : E}
    (h4v : x + (4 : ℝ) • v ∈ interior s) (h4w : x + (4 : ℝ) • w ∈ interior s) :
    f'' w v = f'' v w := by
  have A : (fun h : ℝ => h ^ 2 • (f'' w v - f'' v w)) =o[𝓝[>] 0] fun h => h ^ 2 := by
    convert (s_conv.isLittleO_alternate_sum_square hf xs hx h4v h4w).sub
      (s_conv.isLittleO_alternate_sum_square hf xs hx h4w h4v) using 1
    ext h
    simp only [add_comm, smul_add, smul_sub]
    abel
  have B : (fun _ : ℝ => f'' w v - f'' v w) =o[𝓝[>] 0] fun _ => (1 : ℝ) := by
    have : (fun h : ℝ => 1 / h ^ 2) =O[𝓝[>] 0] fun h => 1 / h ^ 2 := isBigO_refl _ _
    have C := this.smul_isLittleO A
    apply C.congr' _ _
    · filter_upwards [self_mem_nhdsWithin]
      intro h (hpos : 0 < h)
      match_scalars <;> field_simp
    · filter_upwards [self_mem_nhdsWithin] with h (hpos : 0 < h)
      field_simp
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    h4v : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 v))
    h4w : Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 w))
    A : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun h => HSMul.hSMul (HP …
    B : Asymptotics.IsLittleO (nhdsWithin 0 (Set.Ioi 0)) (fun x => HSub.hSub ((f'' …
    ⊢ Eq ((f'' w) v) ((f'' v) w)
  -/
  simpa only [sub_eq_zero] using isLittleO_const_const_iff.1 B
  /-
    🎉 no goals
  -/


/-- If a function is differentiable inside a convex set with nonempty interior, and has a second
derivative at a point of this convex set, then this second derivative is symmetric. -/
theorem Convex.second_derivative_within_at_symmetric {s : Set E} (s_conv : Convex ℝ s)
    (hne : (interior s).Nonempty) {f : E → F} {f' : E → E →L[ℝ] F} {f'' : E →L[ℝ] E →L[ℝ] F}
    (hf : ∀ x ∈ interior s, HasFDerivAt f (f' x) x) {x : E} (xs : x ∈ s)
    (hx : HasFDerivWithinAt f' f'' (interior s) x) (v w : E) : f'' v w = f'' w v := by
  /- we work around a point `x + 4 z` in the interior of `s`. For any vector `m`,
    then `x + 4 (z + t m)` also belongs to the interior of `s` for small enough `t`. This means that
    we will be able to apply `second_derivative_within_at_symmetric_of_mem_interior` to show
    that `f''` is symmetric, after cancelling all the contributions due to `z`. -/
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    hne : (interior s).Nonempty
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w : E
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  rcases hne with ⟨y, hy⟩
  /-
    case intro
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w y : E
    hy : Membership.mem (interior s) y
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  obtain ⟨z, hz⟩ : ∃ z, z = ((1 : ℝ) / 4) • (y - x) := ⟨((1 : ℝ) / 4) • (y - x), rfl⟩
  have A : ∀ m : E, Filter.Tendsto (fun t : ℝ => x + (4 : ℝ) • (z + t • m)) (𝓝 0) (𝓝 y) := by
    intro m
    have : x + (4 : ℝ) • (z + (0 : ℝ) • m) = y := by simp [hz]
    rw [← this]
    refine tendsto_const_nhds.add <| tendsto_const_nhds.smul <| tendsto_const_nhds.add ?_
    exact continuousAt_id.smul continuousAt_const
  have B : ∀ m : E, ∀ᶠ t in 𝓝[>] (0 : ℝ), x + (4 : ℝ) • (z + t • m) ∈ interior s := by
    intro m
    apply nhdsWithin_le_nhds
    apply A m
    rw [mem_interior_iff_mem_nhds] at hy
    exact interior_mem_nhds.2 hy
  -- we choose `t m > 0` such that `x + 4 (z + (t m) m)` belongs to the interior of `s`, for any
  -- vector `m`.
  /-
    case intro.intro
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w y : E
    hy : Membership.mem (interior s) y
    z : E
    hz : Eq z (HSMul.hSMul (1 / 4) (HSub.hSub y x))
    A : ∀ (m : E), Filter.Tendsto (fun t => HAdd.hAdd x (HSMul.hSMul 4 (HAdd.hAdd  …
    B : ∀ (m : E), Filter.Eventually (fun t => Membership.mem (interior s) (HAdd.h …
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  choose t ts tpos using fun m => ((B m).and self_mem_nhdsWithin).exists
  -- applying `second_derivative_within_at_symmetric_of_mem_interior` to the vectors `z`
  -- and `z + (t m) m`, we deduce that `f'' m z = f'' z m` for all `m`.
  have C : ∀ m : E, f'' m z = f'' z m := by
    intro m
    have : f'' (z + t m • m) (z + t 0 • (0 : E)) = f'' (z + t 0 • (0 : E)) (z + t m • m) :=
      s_conv.second_derivative_within_at_symmetric_of_mem_interior hf xs hx (ts 0) (ts m)
    simp only [ContinuousLinearMap.map_add, ContinuousLinearMap.map_smul, add_right_inj,
      ContinuousLinearMap.add_apply, Pi.smul_apply, ContinuousLinearMap.coe_smul', add_zero,
      ContinuousLinearMap.zero_apply, smul_zero, ContinuousLinearMap.map_zero] at this
    exact smul_right_injective F (tpos m).ne' this
  -- applying `second_derivative_within_at_symmetric_of_mem_interior` to the vectors `z + (t v) v`
  -- and `z + (t w) w`, we deduce that `f'' v w = f'' w v`. Cross terms involving `z` can be
  -- eliminated thanks to the fact proved above that `f'' m z = f'' z m`.
  have : f'' (z + t v • v) (z + t w • w) = f'' (z + t w • w) (z + t v • v) :=
    s_conv.second_derivative_within_at_symmetric_of_mem_interior hf xs hx (ts w) (ts v)
  simp only [ContinuousLinearMap.map_add, ContinuousLinearMap.map_smul, smul_add, smul_smul,
    ContinuousLinearMap.add_apply, Pi.smul_apply, ContinuousLinearMap.coe_smul', C] at this
  have : (t v * t w) • (f'' v) w = (t v * t w) • (f'' w) v := by
    linear_combination (norm := module) this
  /-
    case intro.intro
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w y : E
    hy : Membership.mem (interior s) y
    z : E
    hz : Eq z (HSMul.hSMul (1 / 4) (HSub.hSub y x))
    A : ∀ (m : E), Filter.Tendsto (fun t => HAdd.hAdd x (HSMul.hSMul 4 (HAdd.hAdd  …
    B : ∀ (m : E), Filter.Eventually (fun t => Membership.mem (interior s) (HAdd.h …
    t : E → Real
    ts : ∀ (m : E), Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 (HAdd. …
    tpos : ∀ (m : E), LT.lt 0 (t m)
    C : ∀ (m : E), Eq ((f'' m) z) ((f'' z) m)
    this✝ : Eq (HAdd.hAdd (HAdd.hAdd ((f'' z) z) (HSMul.hSMul (t v) ((f'' z) v)))  …
    this : Eq (HSMul.hSMul (HMul.hMul (t v) (t w)) ((f'' v) w)) (HSMul.hSMul (HMul …
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  apply smul_right_injective F _ this
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    s : Set E
    s_conv : Convex Real s
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : ∀ (x : E), Membership.mem (interior s) x → HasFDerivAt f (f' x) x
    x : E
    xs : Membership.mem s x
    hx : HasFDerivWithinAt f' f'' (interior s) x
    v w y : E
    hy : Membership.mem (interior s) y
    z : E
    hz : Eq z (HSMul.hSMul (1 / 4) (HSub.hSub y x))
    A : ∀ (m : E), Filter.Tendsto (fun t => HAdd.hAdd x (HSMul.hSMul 4 (HAdd.hAdd  …
    B : ∀ (m : E), Filter.Eventually (fun t => Membership.mem (interior s) (HAdd.h …
    t : E → Real
    ts : ∀ (m : E), Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul 4 (HAdd. …
    tpos : ∀ (m : E), LT.lt 0 (t m)
    C : ∀ (m : E), Eq ((f'' m) z) ((f'' z) m)
    this✝ : Eq (HAdd.hAdd (HAdd.hAdd ((f'' z) z) (HSMul.hSMul (t v) ((f'' z) v)))  …
    this : Eq (HSMul.hSMul (HMul.hMul (t v) (t w)) ((f'' v) w)) (HSMul.hSMul (HMul …
    ⊢ Ne (HMul.hMul (t v) (t w)) 0
  -/
  simp [(tpos v).ne', (tpos w).ne']
  /-
    🎉 no goals
  -/


/-- If a function is differentiable around `x`, and has two derivatives at `x`, then the second
derivative is symmetric. Version over `ℝ`. See `second_derivative_symmetric_of_eventually` for a
version over `ℝ` or `ℂ`. -/
theorem second_derivative_symmetric_of_eventually_of_real {f : E → F} {f' : E → E →L[ℝ] F}
    {f'' : E →L[ℝ] E →L[ℝ] F} (hf : ∀ᶠ y in 𝓝 x, HasFDerivAt f (f' y) y) (hx : HasFDerivAt f' f'' x)
    (v w : E) : f'' v w = f'' w v := by
  /-
    E : Type u_1
    F : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : E
    f : E → F
    f' : E → ContinuousLinearMap (RingHom.id Real) E F
    f'' : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.id …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  rcases Metric.mem_nhds_iff.1 hf with ⟨ε, εpos, hε⟩
  have A : (interior (Metric.ball x ε)).Nonempty := by
    rwa [Metric.isOpen_ball.interior_eq, Metric.nonempty_ball]
  exact
    Convex.second_derivative_within_at_symmetric (convex_ball x ε) A
      (fun y hy => hε (interior_subset hy)) (Metric.mem_ball_self εpos) hx.hasFDerivWithinAt v w


theorem second_derivative_symmetric_of_eventually [IsRCLikeNormedField 𝕜]
    {f' : E → E →L[𝕜] F} {x : E}
    {f'' : E →L[𝕜] E →L[𝕜] F} (hf : ∀ᶠ y in 𝓝 x, HasFDerivAt f (f' y) y)
    (hx : HasFDerivAt f' f'' x) (v w : E) : f'' v w = f'' w v := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  let _ := IsRCLikeNormedField.rclike 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  let _ : NormedSpace ℝ E := NormedSpace.restrictScalars ℝ 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    x✝ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  let _ : NormedSpace ℝ F := NormedSpace.restrictScalars ℝ 𝕜 F
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    x✝¹ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    x✝ : NormedSpace Real F := NormedSpace.restrictScalars Real 𝕜 F
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  let _ : LinearMap.CompatibleSMul E F ℝ 𝕜 := LinearMap.IsScalarTower.compatibleSMul
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝³ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    x✝² : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    x✝¹ : NormedSpace Real F := NormedSpace.restrictScalars Real 𝕜 F
    x✝ : LinearMap.CompatibleSMul E F Real 𝕜 := LinearMap.IsScalarTower.compatible …
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  let _ : LinearMap.CompatibleSMul E (E →L[𝕜] F) ℝ 𝕜 := LinearMap.IsScalarTower.compatibleSMul
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝⁴ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    x✝³ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    x✝² : NormedSpace Real F := NormedSpace.restrictScalars Real 𝕜 F
    x✝¹ : LinearMap.CompatibleSMul E F Real 𝕜 := LinearMap.IsScalarTower.compatibl …
    x✝ : LinearMap.CompatibleSMul E (ContinuousLinearMap (RingHom.id 𝕜) E F) Real  …
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  let f'R : E → E →L[ℝ] F := fun x ↦ (f' x).restrictScalars ℝ
  have hfR : ∀ᶠ y in 𝓝 x, HasFDerivAt f (f'R y) y := by
    filter_upwards [hf] with y hy using HasFDerivAt.restrictScalars ℝ hy
  let f''Rl : E →ₗ[ℝ] E →ₗ[ℝ] F :=
  { toFun := fun x ↦
      { toFun := fun y ↦ f'' x y
        map_add' := by simp
        map_smul' := by simp }
    map_add' := by intros; ext; simp
    map_smul' := by intros; ext; simp }
  let f''R : E →L[ℝ] E →L[ℝ] F := by
    refine LinearMap.mkContinuous₂ f''Rl (‖f''‖) (fun x y ↦ ?_)
    simp only [LinearMap.coe_mk, AddHom.coe_mk, f''Rl]
    exact ContinuousLinearMap.le_opNorm₂ f'' x y
  have : HasFDerivAt f'R f''R x := by
    simp only [hasFDerivAt_iff_tendsto] at hx ⊢
    exact hx
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝⁴ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    x✝³ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    x✝² : NormedSpace Real F := NormedSpace.restrictScalars Real 𝕜 F
    x✝¹ : LinearMap.CompatibleSMul E F Real 𝕜 := LinearMap.IsScalarTower.compatibl …
    x✝ : LinearMap.CompatibleSMul E (ContinuousLinearMap (RingHom.id 𝕜) E F) Real  …
    f'R : E → ContinuousLinearMap (RingHom.id Real) E F := fun x => ContinuousLine …
    hfR : Filter.Eventually (fun y => HasFDerivAt f (f'R y) y) (nhds x)
    f''Rl : LinearMap (RingHom.id Real) E (LinearMap (RingHom.id Real) E F) := { t …
    f''R : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.i …
    this : HasFDerivAt f'R f''R x
    ⊢ Eq ((f'' v) w) ((f'' w) v)
  -/
  change f''R v w = f''R w v
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : IsRCLikeNormedField 𝕜
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    x : E
    f'' : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) …
    hf : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hx : HasFDerivAt f' f'' x
    v w : E
    x✝⁴ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    x✝³ : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    x✝² : NormedSpace Real F := NormedSpace.restrictScalars Real 𝕜 F
    x✝¹ : LinearMap.CompatibleSMul E F Real 𝕜 := LinearMap.IsScalarTower.compatibl …
    x✝ : LinearMap.CompatibleSMul E (ContinuousLinearMap (RingHom.id 𝕜) E F) Real  …
    f'R : E → ContinuousLinearMap (RingHom.id Real) E F := fun x => ContinuousLine …
    hfR : Filter.Eventually (fun y => HasFDerivAt f (f'R y) y) (nhds x)
    f''Rl : LinearMap (RingHom.id Real) E (LinearMap (RingHom.id Real) E F) := { t …
    f''R : ContinuousLinearMap (RingHom.id Real) E (ContinuousLinearMap (RingHom.i …
    this : HasFDerivAt f'R f''R x
    ⊢ Eq ((f''R v) w) ((f''R w) v)
  -/
  exact second_derivative_symmetric_of_eventually_of_real hfR this v w
  /-
    🎉 no goals
  -/


/-- If a function is differentiable, and has two derivatives at `x`, then the second
derivative is symmetric. -/
theorem second_derivative_symmetric [IsRCLikeNormedField 𝕜]
    {f' : E → E →L[𝕜] F} {f'' : E →L[𝕜] E →L[𝕜] F} {x : E}
    (hf : ∀ y, HasFDerivAt f (f' y) y) (hx : HasFDerivAt f' f'' x) (v w : E) : f'' v w = f'' w v :=
  second_derivative_symmetric_of_eventually (Filter.Eventually.of_forall hf) hx v w


open scoped Classical in
variable (𝕜) in
/-- `minSmoothness 𝕜 n` is the minimal smoothness exponent larger than or equal to `n` for which
one can do serious calculus in `𝕜`. If `𝕜` is `ℝ` or `ℂ`, this is just `n`. Otherwise,
this is `ω` as only analytic functions are well behaved on `ℚₚ`, say. -/
noncomputable irreducible_def minSmoothness (n : WithTop ℕ∞) :=
  if IsRCLikeNormedField 𝕜 then n else ω


@[simp] lemma minSmoothness_of_isRCLikeNormedField [h : IsRCLikeNormedField 𝕜] {n : WithTop ℕ∞} :
    minSmoothness 𝕜 n = n := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    h : IsRCLikeNormedField 𝕜
    n : WithTop ENat
    ⊢ Eq (minSmoothness 𝕜 n) n
  -/
  simp [minSmoothness, h]
  /-
    🎉 no goals
  -/


lemma le_minSmoothness {n : WithTop ℕ∞} : n ≤ minSmoothness 𝕜 n := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    n : WithTop ENat
    ⊢ LE.le n (minSmoothness 𝕜 n)
  -/
  simp only [minSmoothness]
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    n : WithTop ENat
    ⊢ LE.le n (ite (IsRCLikeNormedField 𝕜) n Top.top)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


/-- If `minSmoothness 𝕜 m ≤ n` for some (finite) integer `m`, then one can
find `n' ∈ [minSmoothness 𝕜 m, n]` which is not `∞`: over `ℝ` or `ℂ`, just take `m`, and otherwise
just take `ω`. The interest of this technical lemma is that, if a function is `C^{n'}` at a point
for `n' ≠ ∞`, then it is `C^{n'}` on a neighborhood of the point (this property fails only
in `C^∞` smoothness, see `ContDiffWithinAt.contDiffOn`). -/
lemma exist_minSmoothness_le_ne_infty {n : WithTop ℕ∞} {m : ℕ} (hm : minSmoothness 𝕜 m ≤ n) :
    ∃ n', minSmoothness 𝕜 m ≤ n' ∧ n' ≤ n ∧ n' ≠ ∞ := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    n : WithTop ENat
    m : Nat
    hm : LE.le (minSmoothness 𝕜 ↑m) n
    ⊢ Exists fun n' => And (LE.le (minSmoothness 𝕜 ↑m) n') (And (LE.le n' n) (Ne n …
  -/
  simp only [minSmoothness] at hm ⊢
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    n : WithTop ENat
    m : Nat
    hm : LE.le (ite (IsRCLikeNormedField 𝕜) (↑m) Top.top) n
    ⊢ Exists fun n' => And (LE.le (ite (IsRCLikeNormedField 𝕜) (↑m) Top.top) n') ( …
  -/
  split_ifs with h
    /-
      case pos
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      n : WithTop ENat
      m : Nat
      hm : LE.le (ite (IsRCLikeNormedField 𝕜) (↑m) Top.top) n
      h : IsRCLikeNormedField 𝕜
      ⊢ Exists fun n' => And (LE.le (↑m) n') (And (LE.le n' n) (Ne n' ↑Top.top))
    -/
  · simp only [h, ↓reduceIte] at hm
    /-
      case pos
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      n : WithTop ENat
      m : Nat
      h : IsRCLikeNormedField 𝕜
      hm : LE.le (↑m) n
      ⊢ Exists fun n' => And (LE.le (↑m) n') (And (LE.le n' n) (Ne n' ↑Top.top))
    -/
    exact ⟨m, le_rfl, hm, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      n : WithTop ENat
      m : Nat
      hm : LE.le (ite (IsRCLikeNormedField 𝕜) (↑m) Top.top) n
      h : Not (IsRCLikeNormedField 𝕜)
      ⊢ Exists fun n' => And (LE.le Top.top n') (And (LE.le n' n) (Ne n' ↑Top.top))
    -/
  · simp only [h, ↓reduceIte, top_le_iff] at hm
    /-
      case neg
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      n : WithTop ENat
      m : Nat
      h : Not (IsRCLikeNormedField 𝕜)
      hm : Eq n Top.top
      ⊢ Exists fun n' => And (LE.le Top.top n') (And (LE.le n' n) (Ne n' ↑Top.top))
    -/
    refine ⟨ω, le_rfl, by simp [hm], by simp⟩
    /-
      🎉 no goals
    -/


/-- If a function is `C^2` at a point, then its second derivative there is symmetric. Over a field
different from `ℝ` or `ℂ`, we should require that the function is analytic. -/
theorem ContDiffAt.isSymmSndFDerivAt {n : WithTop ℕ∞}
    (hf : ContDiffAt 𝕜 n f x) (hn : minSmoothness 𝕜 2 ≤ n) : IsSymmSndFDerivAt 𝕜 f x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffAt 𝕜 n f x
    hn : LE.le (minSmoothness 𝕜 2) n
    ⊢ IsSymmSndFDerivAt 𝕜 f x
  -/
  by_cases h : IsRCLikeNormedField 𝕜
  -- First deal with the `ℝ` or `ℂ` case, where `C^2` is enough.
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x : E
      n : WithTop ENat
      hf : ContDiffAt 𝕜 n f x
      hn : LE.le (minSmoothness 𝕜 2) n
      h : IsRCLikeNormedField 𝕜
      ⊢ IsSymmSndFDerivAt 𝕜 f x
    -/
  · intro v w
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x : E
      n : WithTop ENat
      hf : ContDiffAt 𝕜 n f x
      hn : LE.le (minSmoothness 𝕜 2) n
      h : IsRCLikeNormedField 𝕜
      v w : E
      ⊢ Eq (((fderiv 𝕜 (fderiv 𝕜 f) x) v) w) (((fderiv 𝕜 (fderiv 𝕜 f) x) w) v)
    -/
    apply second_derivative_symmetric_of_eventually (f := f) (f' := fderiv 𝕜 f) (x := x)
    · obtain ⟨u, hu, h'u⟩ : ∃ u ∈ 𝓝 x, ContDiffOn 𝕜 2 f u :=
        (hf.of_le hn).contDiffOn (m := 2) le_minSmoothness (by simp)
      /-
        case pos.hf.intro.intro
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        x : E
        n : WithTop ENat
        hf : ContDiffAt 𝕜 n f x
        hn : LE.le (minSmoothness 𝕜 2) n
        h : IsRCLikeNormedField 𝕜
        v w : E
        u : Set E
        hu : Membership.mem (nhds x) u
        h'u : ContDiffOn 𝕜 2 f u
        ⊢ Filter.Eventually (fun y => HasFDerivAt f (fderiv 𝕜 f y) y) (nhds x)
      -/
      rcases mem_nhds_iff.1 hu with ⟨v, vu, v_open, xv⟩
      /-
        case pos.hf.intro.intro.intro.intro.intro
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        x : E
        n : WithTop ENat
        hf : ContDiffAt 𝕜 n f x
        hn : LE.le (minSmoothness 𝕜 2) n
        h : IsRCLikeNormedField 𝕜
        v✝ w : E
        u : Set E
        hu : Membership.mem (nhds x) u
        h'u : ContDiffOn 𝕜 2 f u
        v : Set E
        vu : HasSubset.Subset v u
        v_open : IsOpen v
        xv : Membership.mem v x
        ⊢ Filter.Eventually (fun y => HasFDerivAt f (fderiv 𝕜 f y) y) (nhds x)
      -/
      filter_upwards [v_open.mem_nhds xv] with y hy
      have : DifferentiableAt 𝕜 f y := by
        have := (h'u.mono vu y hy).contDiffAt (v_open.mem_nhds hy)
        exact this.differentiableAt one_le_two
      /-
        case h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        x : E
        n : WithTop ENat
        hf : ContDiffAt 𝕜 n f x
        hn : LE.le (minSmoothness 𝕜 2) n
        h : IsRCLikeNormedField 𝕜
        v✝ w : E
        u : Set E
        hu : Membership.mem (nhds x) u
        h'u : ContDiffOn 𝕜 2 f u
        v : Set E
        vu : HasSubset.Subset v u
        v_open : IsOpen v
        xv : Membership.mem v x
        y : E
        hy : Membership.mem v y
        this : DifferentiableAt 𝕜 f y
        ⊢ HasFDerivAt f (fderiv 𝕜 f y) y
      -/
      exact DifferentiableAt.hasFDerivAt this
      /-
        🎉 no goals
      -/
    · have : DifferentiableAt 𝕜 (fderiv 𝕜 f) x := by
        apply ContDiffAt.differentiableAt _ le_rfl
        exact hf.fderiv_right (le_minSmoothness.trans hn)
      /-
        case pos.hx
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        x : E
        n : WithTop ENat
        hf : ContDiffAt 𝕜 n f x
        hn : LE.le (minSmoothness 𝕜 2) n
        h : IsRCLikeNormedField 𝕜
        v w : E
        this : DifferentiableAt 𝕜 (fderiv 𝕜 f) x
        ⊢ HasFDerivAt (fderiv 𝕜 f) (fderiv 𝕜 (fderiv 𝕜 f) x) x
      -/
      exact DifferentiableAt.hasFDerivAt this
      /-
        🎉 no goals
      -/
  -- then deal with the case of an arbitrary field, with analytic functions.
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x : E
      n : WithTop ENat
      hf : ContDiffAt 𝕜 n f x
      hn : LE.le (minSmoothness 𝕜 2) n
      h : Not (IsRCLikeNormedField 𝕜)
      ⊢ IsSymmSndFDerivAt 𝕜 f x
    -/
  · simp only [minSmoothness, h, ↓reduceIte, top_le_iff] at hn
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x : E
      n : WithTop ENat
      hf : ContDiffAt 𝕜 n f x
      h : Not (IsRCLikeNormedField 𝕜)
      hn : Eq n Top.top
      ⊢ IsSymmSndFDerivAt 𝕜 f x
    -/
    apply ContDiffAt.isSymmSndFDerivAt_of_omega
    /-
      case neg.hf
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x : E
      n : WithTop ENat
      hf : ContDiffAt 𝕜 n f x
      h : Not (IsRCLikeNormedField 𝕜)
      hn : Eq n Top.top
      ⊢ ContDiffAt 𝕜 Top.top f x
    -/
    simpa [hn] using hf
    /-
      🎉 no goals
    -/


/-- If a function is `C^2` within a set at a point, and accumulated by points in the interior
of the set, then its second derivative there is symmetric. Over a field
different from `ℝ` or `ℂ`, we should require that the function is analytic. -/
theorem ContDiffWithinAt.isSymmSndFDerivWithinAt {n : WithTop ℕ∞}
    (hf : ContDiffWithinAt 𝕜 n f s x) (hn : minSmoothness 𝕜 2 ≤ n)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ closure (interior s)) (h'x : x ∈ s) :
    IsSymmSndFDerivWithinAt 𝕜 f s x := by
  /- We argue that, at interior points, the second derivative is symmetric, and moreover by
  continuity it converges to the second derivative at `x`. Therefore, the latter is also
  symmetric. -/
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  obtain ⟨m, hm, hmn, m_ne⟩ := exist_minSmoothness_le_ne_infty hn
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    m : WithTop ENat
    hm : LE.le (minSmoothness 𝕜 ↑2) m
    hmn : LE.le m n
    m_ne : Ne m ↑Top.top
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  rcases (hf.of_le hmn).contDiffOn' le_rfl (by simp [m_ne]) with ⟨u, u_open, xu, hu⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    m : WithTop ENat
    hm : LE.le (minSmoothness 𝕜 ↑2) m
    hmn : LE.le m n
    m_ne : Ne m ↑Top.top
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 m f (Inter.inter (Insert.insert x s) u)
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  simp only [insert_eq_of_mem h'x] at hu
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    m : WithTop ENat
    hm : LE.le (minSmoothness 𝕜 ↑2) m
    hmn : LE.le m n
    m_ne : Ne m ↑Top.top
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 m f (Inter.inter s u)
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  have h'u : UniqueDiffOn 𝕜 (s ∩ u) := hs.inter u_open
  obtain ⟨y, hy, y_lim⟩ : ∃ y, (∀ (n : ℕ), y n ∈ interior s) ∧ Tendsto y atTop (𝓝 x) :=
    mem_closure_iff_seq_limit.1 hx
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    m : WithTop ENat
    hm : LE.le (minSmoothness 𝕜 ↑2) m
    hmn : LE.le m n
    m_ne : Ne m ↑Top.top
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 m f (Inter.inter s u)
    h'u : UniqueDiffOn 𝕜 (Inter.inter s u)
    y : Nat → E
    hy : ∀ (n : Nat), Membership.mem (interior s) (y n)
    y_lim : Filter.Tendsto y Filter.atTop (nhds x)
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  have L : ∀ᶠ k in atTop, y k ∈ u := y_lim (u_open.mem_nhds xu)
  have I : ∀ᶠ k in atTop, IsSymmSndFDerivWithinAt 𝕜 f s (y k) := by
    filter_upwards [L] with k hk
    have s_mem : s ∈ 𝓝 (y k) := by
      apply mem_of_superset (isOpen_interior.mem_nhds (hy k))
      exact interior_subset
    have : IsSymmSndFDerivAt 𝕜 f (y k) := by
      apply ContDiffAt.isSymmSndFDerivAt _ (n := m) hm
      apply (hu (y k) ⟨(interior_subset (hy k)), hk⟩).contDiffAt
      exact inter_mem s_mem (u_open.mem_nhds hk)
    intro v w
    rw [fderivWithin_fderivWithin_eq_of_mem_nhds s_mem]
    exact this v w
  have A : ContinuousOn (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s) (s ∩ u) := by
    have : ContinuousOn (fderivWithin 𝕜 (fderivWithin 𝕜 f (s ∩ u)) (s ∩ u)) (s ∩ u) :=
      ((hu.fderivWithin h'u (m := 1) (le_minSmoothness.trans hm)).fderivWithin h'u
      (m := 0) le_rfl).continuousOn
    apply this.congr
    intro y hy
    apply fderivWithin_fderivWithin_eq_of_eventuallyEq
    filter_upwards [u_open.mem_nhds hy.2] with z hz
    change (z ∈ s) = (z ∈ s ∩ u)
    aesop
  have B : Tendsto (fun k ↦ fderivWithin 𝕜 (fderivWithin 𝕜 f s) s (y k)) atTop
      (𝓝 (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x)) := by
    have : Tendsto y atTop (𝓝[s ∩ u] x) := by
      apply tendsto_nhdsWithin_iff.2 ⟨y_lim, ?_⟩
      filter_upwards [L] with k hk using ⟨interior_subset (hy k), hk⟩
    exact (A x ⟨h'x, xu⟩ ).tendsto.comp this
  have C (v w : E) : Tendsto (fun k ↦ fderivWithin 𝕜 (fderivWithin 𝕜 f s) s (y k) v w) atTop
      (𝓝 (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x v w)) := by
    have : Continuous (fun (A : E →L[𝕜] E →L[𝕜] F) ↦ A v w) := by fun_prop
    exact (this.tendsto _).comp B
  have C' (v w : E) : Tendsto (fun k ↦ fderivWithin 𝕜 (fderivWithin 𝕜 f s) s (y k) w v) atTop
      (𝓝 (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x v w)) := by
    apply (C v w).congr'
    filter_upwards [I] with k hk using hk v w
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    m : WithTop ENat
    hm : LE.le (minSmoothness 𝕜 ↑2) m
    hmn : LE.le m n
    m_ne : Ne m ↑Top.top
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 m f (Inter.inter s u)
    h'u : UniqueDiffOn 𝕜 (Inter.inter s u)
    y : Nat → E
    hy : ∀ (n : Nat), Membership.mem (interior s) (y n)
    y_lim : Filter.Tendsto y Filter.atTop (nhds x)
    L : Filter.Eventually (fun k => Membership.mem u (y k)) Filter.atTop
    I : Filter.Eventually (fun k => IsSymmSndFDerivWithinAt 𝕜 f s (y k)) Filter.at …
    A : ContinuousOn (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s) (Inter.inter s u)
    B : Filter.Tendsto (fun k => fderivWithin 𝕜 (fderivWithin 𝕜 f s) s (y k)) Filt …
    C : ∀ (v w : E), Filter.Tendsto (fun k => ((fderivWithin 𝕜 (fderivWithin 𝕜 f s …
    C' : ∀ (v w : E), Filter.Tendsto (fun k => ((fderivWithin 𝕜 (fderivWithin 𝕜 f  …
    ⊢ IsSymmSndFDerivWithinAt 𝕜 f s x
  -/
  intro v w
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    hf : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le (minSmoothness 𝕜 2) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem s x
    m : WithTop ENat
    hm : LE.le (minSmoothness 𝕜 ↑2) m
    hmn : LE.le m n
    m_ne : Ne m ↑Top.top
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 m f (Inter.inter s u)
    h'u : UniqueDiffOn 𝕜 (Inter.inter s u)
    y : Nat → E
    hy : ∀ (n : Nat), Membership.mem (interior s) (y n)
    y_lim : Filter.Tendsto y Filter.atTop (nhds x)
    L : Filter.Eventually (fun k => Membership.mem u (y k)) Filter.atTop
    I : Filter.Eventually (fun k => IsSymmSndFDerivWithinAt 𝕜 f s (y k)) Filter.at …
    A : ContinuousOn (fderivWithin 𝕜 (fderivWithin 𝕜 f s) s) (Inter.inter s u)
    B : Filter.Tendsto (fun k => fderivWithin 𝕜 (fderivWithin 𝕜 f s) s (y k)) Filt …
    C : ∀ (v w : E), Filter.Tendsto (fun k => ((fderivWithin 𝕜 (fderivWithin 𝕜 f s …
    C' : ∀ (v w : E), Filter.Tendsto (fun k => ((fderivWithin 𝕜 (fderivWithin 𝕜 f  …
    v w : E
    ⊢ Eq (((fderivWithin 𝕜 (fderivWithin 𝕜 f s) s x) v) w) (((fderivWithin 𝕜 (fder …
  -/
  exact tendsto_nhds_unique (C v w) (C' w v)
  /-
    🎉 no goals
  -/


