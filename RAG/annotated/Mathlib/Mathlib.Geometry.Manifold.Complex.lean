/-- **Maximum modulus principle**: if `f : M → F` is complex differentiable in a neighborhood of `c`
and the norm `‖f z‖` has a local maximum at `c`, then `‖f z‖` is locally constant in a neighborhood
of `c`. This is a manifold version of `Complex.norm_eventually_eq_of_isLocalMax`. -/
theorem Complex.norm_eventually_eq_of_mdifferentiableAt_of_isLocalMax {f : M → F} {c : M}
    (hd : ∀ᶠ z in 𝓝 c, MDifferentiableAt I 𝓘(ℂ, F) f z) (hc : IsLocalMax (norm ∘ f) c) :
    ∀ᶠ y in 𝓝 c, ‖f y‖ = ‖f c‖ := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    c : M
    hd : Filter.Eventually (fun z => MDifferentiableAt I (modelWithCornersSelf Com …
    hc : IsLocalMax (Function.comp Norm.norm f) c
    ⊢ Filter.Eventually (fun y => Eq (Norm.norm (f y)) (Norm.norm (f c))) (nhds c)
  -/
  set e := extChartAt I c
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    c : M
    hd : Filter.Eventually (fun z => MDifferentiableAt I (modelWithCornersSelf Com …
    hc : IsLocalMax (Function.comp Norm.norm f) c
    e : PartialEquiv M E := extChartAt I c
    ⊢ Filter.Eventually (fun y => Eq (Norm.norm (f y)) (Norm.norm (f c))) (nhds c)
  -/
  have hI : range I = univ := ModelWithCorners.Boundaryless.range_eq_univ
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    c : M
    hd : Filter.Eventually (fun z => MDifferentiableAt I (modelWithCornersSelf Com …
    hc : IsLocalMax (Function.comp Norm.norm f) c
    e : PartialEquiv M E := extChartAt I c
    hI : Eq (Set.range ↑I) Set.univ
    ⊢ Filter.Eventually (fun y => Eq (Norm.norm (f y)) (Norm.norm (f c))) (nhds c)
  -/
  have H₁ : 𝓝[range I] (e c) = 𝓝 (e c) := by rw [hI, nhdsWithin_univ]
  have H₂ : map e.symm (𝓝 (e c)) = 𝓝 c := by
    rw [← map_extChartAt_symm_nhdsWithin_range (I := I) c, H₁]
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    c : M
    hd : Filter.Eventually (fun z => MDifferentiableAt I (modelWithCornersSelf Com …
    hc : IsLocalMax (Function.comp Norm.norm f) c
    e : PartialEquiv M E := extChartAt I c
    hI : Eq (Set.range ↑I) Set.univ
    H₁ : Eq (nhdsWithin (↑e c) (Set.range ↑I)) (nhds (↑e c))
    H₂ : Eq (Filter.map (↑e.symm) (nhds (↑e c))) (nhds c)
    ⊢ Filter.Eventually (fun y => Eq (Norm.norm (f y)) (Norm.norm (f c))) (nhds c)
  -/
  rw [← H₂, eventually_map]
  replace hd : ∀ᶠ y in 𝓝 (e c), DifferentiableAt ℂ (f ∘ e.symm) y := by
    have : e.target ∈ 𝓝 (e c) := H₁ ▸ extChartAt_target_mem_nhdsWithin c
    filter_upwards [this, Tendsto.eventually H₂.le hd] with y hyt hy₂
    have hys : e.symm y ∈ (chartAt H c).source := by
      rw [← extChartAt_source I c]
      exact (extChartAt I c).map_target hyt
    have hfy : f (e.symm y) ∈ (chartAt F (0 : F)).source := mem_univ _
    rw [mdifferentiableAt_iff_of_mem_source hys hfy, hI, differentiableWithinAt_univ,
      e.right_inv hyt] at hy₂
    exact hy₂.2
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    c : M
    hc : IsLocalMax (Function.comp Norm.norm f) c
    e : PartialEquiv M E := extChartAt I c
    hI : Eq (Set.range ↑I) Set.univ
    H₁ : Eq (nhdsWithin (↑e c) (Set.range ↑I)) (nhds (↑e c))
    H₂ : Eq (Filter.map (↑e.symm) (nhds (↑e c))) (nhds c)
    hd : Filter.Eventually (fun y => DifferentiableAt Complex (Function.comp f ↑e. …
    ⊢ Filter.Eventually (fun a => Eq (Norm.norm (f (↑e.symm a))) (Norm.norm (f c)) …
  -/
  convert norm_eventually_eq_of_isLocalMax hd _
    /-
      case h.e'_2.h.h.e'_3.h.e'_3
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      F : Type u_2
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace Complex F
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Complex E H
      inst✝³ : I.Boundaryless
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      f : M → F
      c : M
      hc : IsLocalMax (Function.comp Norm.norm f) c
      e : PartialEquiv M E := extChartAt I c
      hI : Eq (Set.range ↑I) Set.univ
      H₁ : Eq (nhdsWithin (↑e c) (Set.range ↑I)) (nhds (↑e c))
      H₂ : Eq (Filter.map (↑e.symm) (nhds (↑e c))) (nhds c)
      hd : Filter.Eventually (fun y => DifferentiableAt Complex (Function.comp f ↑e. …
      x✝ : E
      ⊢ Eq (f c) (Function.comp f (↑e.symm) (↑e c))
    -/
  · exact congr_arg f (extChartAt_to_inv _).symm
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      F : Type u_2
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace Complex F
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Complex E H
      inst✝³ : I.Boundaryless
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      f : M → F
      c : M
      hc : IsLocalMax (Function.comp Norm.norm f) c
      e : PartialEquiv M E := extChartAt I c
      hI : Eq (Set.range ↑I) Set.univ
      H₁ : Eq (nhdsWithin (↑e c) (Set.range ↑I)) (nhds (↑e c))
      H₂ : Eq (Filter.map (↑e.symm) (nhds (↑e c))) (nhds c)
      hd : Filter.Eventually (fun y => DifferentiableAt Complex (Function.comp f ↑e. …
      ⊢ IsLocalMax (Function.comp Norm.norm (Function.comp f ↑e.symm)) (↑e c)
    -/
  · simpa only [e, IsLocalMax, IsMaxFilter, ← H₂, (· ∘ ·), extChartAt_to_inv] using hc
    /-
      🎉 no goals
    -/


/-- **Maximum modulus principle** on a connected set. Let `U` be a (pre)connected open set in a
complex normed space. Let `f : E → F` be a function that is complex differentiable on `U`. Suppose
that `‖f x‖` takes its maximum value on `U` at `c ∈ U`. Then `‖f x‖ = ‖f c‖` for all `x ∈ U`. -/
theorem norm_eqOn_of_isPreconnected_of_isMaxOn {f : M → F} {U : Set M} {c : M}
    (hd : MDifferentiableOn I 𝓘(ℂ, F) f U) (hc : IsPreconnected U) (ho : IsOpen U)
    (hcU : c ∈ U) (hm : IsMaxOn (norm ∘ f) U c) : EqOn (norm ∘ f) (const M ‖f c‖) U := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    ⊢ Set.EqOn (Function.comp Norm.norm f) (Function.const M (Norm.norm (f c))) U
  -/
  set V := {z ∈ U | ‖f z‖ = ‖f c‖}
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    ⊢ Set.EqOn (Function.comp Norm.norm f) (Function.const M (Norm.norm (f c))) U
  -/
  suffices U ⊆ V from fun x hx ↦ (this hx).2
  have hVo : IsOpen V := by
    refine isOpen_iff_mem_nhds.2 fun x hx ↦ inter_mem (ho.mem_nhds hx.1) ?_
    replace hm : IsLocalMax (‖f ·‖) x :=
      mem_of_superset (ho.mem_nhds hx.1) fun z hz ↦ (hm hz).out.trans_eq hx.2.symm
    replace hd : ∀ᶠ y in 𝓝 x, MDifferentiableAt I 𝓘(ℂ, F) f y :=
      (eventually_mem_nhds_iff.2 (ho.mem_nhds hx.1)).mono fun z ↦ hd.mdifferentiableAt
    exact (Complex.norm_eventually_eq_of_mdifferentiableAt_of_isLocalMax hd hm).mono fun _ ↦
      (Eq.trans · hx.2)
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    hVo : IsOpen V
    ⊢ HasSubset.Subset U V
  -/
  have hVne : (U ∩ V).Nonempty := ⟨c, hcU, hcU, rfl⟩
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    ⊢ HasSubset.Subset U V
  -/
  set W := U ∩ {z | ‖f z‖ = ‖f c‖}ᶜ
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set M := Inter.inter U (HasCompl.compl (setOf fun z => Eq (Norm.norm (f z) …
    ⊢ HasSubset.Subset U V
  -/
  have hWo : IsOpen W := hd.continuousOn.norm.isOpen_inter_preimage ho isOpen_ne
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set M := Inter.inter U (HasCompl.compl (setOf fun z => Eq (Norm.norm (f z) …
    hWo : IsOpen W
    ⊢ HasSubset.Subset U V
  -/
  have hdVW : Disjoint V W := disjoint_compl_right.mono inf_le_right inf_le_right
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set M := Inter.inter U (HasCompl.compl (setOf fun z => Eq (Norm.norm (f z) …
    hWo : IsOpen W
    hdVW : Disjoint V W
    ⊢ HasSubset.Subset U V
  -/
  have hUVW : U ⊆ V ∪ W := fun x hx => (eq_or_ne ‖f x‖ ‖f c‖).imp (.intro hx) (.intro hx)
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    c : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hc : IsPreconnected U
    ho : IsOpen U
    hcU : Membership.mem U c
    hm : IsMaxOn (Function.comp Norm.norm f) U c
    V : Set M := setOf fun z => And (Membership.mem U z) (Eq (Norm.norm (f z)) (No …
    hVo : IsOpen V
    hVne : (Inter.inter U V).Nonempty
    W : Set M := Inter.inter U (HasCompl.compl (setOf fun z => Eq (Norm.norm (f z) …
    hWo : IsOpen W
    hdVW : Disjoint V W
    hUVW : HasSubset.Subset U (Union.union V W)
    ⊢ HasSubset.Subset U V
  -/
  exact hc.subset_left_of_subset_union hVo hWo hdVW hUVW hVne
  /-
    🎉 no goals
  -/


/-- **Maximum modulus principle** on a connected set. Let `U` be a (pre)connected open set in a
complex normed space.  Let `f : E → F` be a function that is complex differentiable on `U`. Suppose
that `‖f x‖` takes its maximum value on `U` at `c ∈ U`. Then `f x = f c` for all `x ∈ U`.

TODO: change assumption from `IsMaxOn` to `IsLocalMax`. -/
theorem eqOn_of_isPreconnected_of_isMaxOn_norm [StrictConvexSpace ℝ F] {f : M → F} {U : Set M}
    {c : M} (hd : MDifferentiableOn I 𝓘(ℂ, F) f U) (hc : IsPreconnected U) (ho : IsOpen U)
    (hcU : c ∈ U) (hm : IsMaxOn (norm ∘ f) U c) : EqOn f (const M (f c)) U := fun x hx =>
  have H₁ : ‖f x‖ = ‖f c‖ := hd.norm_eqOn_of_isPreconnected_of_isMaxOn hc ho hcU hm hx
  -- TODO: Add `MDifferentiableOn.add` etc; does it mean importing `Manifold.Algebra.Monoid`?
  have hd' : MDifferentiableOn I 𝓘(ℂ, F) (f · + f c) U := fun x hx ↦
    ⟨(hd x hx).1.add continuousWithinAt_const, (hd x hx).2.add_const _⟩
  have H₂ : ‖f x + f c‖ = ‖f c + f c‖ :=
    hd'.norm_eqOn_of_isPreconnected_of_isMaxOn hc ho hcU hm.norm_add_self hx
                                        /-
                                          E : Type u_1
                                          inst✝⁹ : NormedAddCommGroup E
                                          inst✝⁸ : NormedSpace Complex E
                                          F : Type u_2
                                          inst✝⁷ : NormedAddCommGroup F
                                          inst✝⁶ : NormedSpace Complex F
                                          H : Type u_3
                                          inst✝⁵ : TopologicalSpace H
                                          I : ModelWithCorners Complex E H
                                          inst✝⁴ : I.Boundaryless
                                          M : Type u_4
                                          inst✝³ : TopologicalSpace M
                                          inst✝² : ChartedSpace H M
                                          inst✝¹ : SmoothManifoldWithCorners I M
                                          inst✝ : StrictConvexSpace Real F
                                          f : M → F
                                          U : Set M
                                          c : M
                                          hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
                                          hc : IsPreconnected U
                                          ho : IsOpen U
                                          hcU : Membership.mem U c
                                          hm : IsMaxOn (Function.comp Norm.norm f) U c
                                          x : M
                                          hx : Membership.mem U x
                                          H₁ : Eq (Norm.norm (f x)) (Norm.norm (f c))
                                          hd' : MDifferentiableOn I (modelWithCornersSelf Complex F) (fun x => HAdd.hAdd …
                                          H₂ : Eq (Norm.norm (HAdd.hAdd (f x) (f c))) (Norm.norm (HAdd.hAdd (f c) (f c)))
                                          ⊢ Eq (Norm.norm (HAdd.hAdd (f x) (Function.const M (f c) x))) (HAdd.hAdd (Norm …
                                        -/
  eq_of_norm_eq_of_norm_add_eq H₁ <| by simp only [H₂, SameRay.rfl.norm_add, H₁, Function.const]
                                        /-
                                          🎉 no goals
                                        -/


/-- If a function `f : M → F` from a complex manifold to a complex normed space is holomorphic on a
(pre)connected compact open set, then it is a constant on this set. -/
theorem apply_eq_of_isPreconnected_isCompact_isOpen {f : M → F} {U : Set M} {a b : M}
    (hd : MDifferentiableOn I 𝓘(ℂ, F) f U) (hpc : IsPreconnected U) (hc : IsCompact U)
    (ho : IsOpen U) (ha : a ∈ U) (hb : b ∈ U) : f a = f b := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    a b : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hpc : IsPreconnected U
    hc : IsCompact U
    ho : IsOpen U
    ha : Membership.mem U a
    hb : Membership.mem U b
    ⊢ Eq (f a) (f b)
  -/
  refine ?_
  -- Subtract `f b` to avoid the assumption `[StrictConvexSpace ℝ F]`
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    f : M → F
    U : Set M
    a b : M
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hpc : IsPreconnected U
    hc : IsCompact U
    ho : IsOpen U
    ha : Membership.mem U a
    hb : Membership.mem U b
    ⊢ Eq (f a) (f b)
  -/
  wlog hb₀ : f b = 0 generalizing f
  · have hd' : MDifferentiableOn I 𝓘(ℂ, F) (f · - f b) U := fun x hx ↦
      ⟨(hd x hx).1.sub continuousWithinAt_const, (hd x hx).2.sub_const _⟩
    /-
      case inr
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      F : Type u_2
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace Complex F
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners Complex E H
      inst✝³ : I.Boundaryless
      M : Type u_4
      inst✝² : TopologicalSpace M
      inst✝¹ : ChartedSpace H M
      inst✝ : SmoothManifoldWithCorners I M
      f : M → F
      U : Set M
      a b : M
      hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
      hpc : IsPreconnected U
      hc : IsCompact U
      ho : IsOpen U
      ha : Membership.mem U a
      hb : Membership.mem U b
      this : ∀ {f : M → F}, MDifferentiableOn I (modelWithCornersSelf Complex F) f U …
      hb₀ : Not (Eq (f b) 0)
      hd' : MDifferentiableOn I (modelWithCornersSelf Complex F) (fun x => HSub.hSub …
      ⊢ Eq (f a) (f b)
    -/
    simpa [sub_eq_zero] using this hd' (sub_self _)
    /-
      🎉 no goals
    -/
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    U : Set M
    a b : M
    hpc : IsPreconnected U
    hc : IsCompact U
    ho : IsOpen U
    ha : Membership.mem U a
    hb : Membership.mem U b
    f : M → F
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hb₀ : Eq (f b) 0
    ⊢ Eq (f a) (f b)
  -/
  rcases hc.exists_isMaxOn ⟨a, ha⟩ hd.continuousOn.norm with ⟨c, hcU, hc⟩
  have : ∀ x ∈ U, ‖f x‖ = ‖f c‖ :=
    norm_eqOn_of_isPreconnected_of_isMaxOn hd hpc ho hcU hc
  /-
    case intro.intro
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Complex F
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners Complex E H
    inst✝³ : I.Boundaryless
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : SmoothManifoldWithCorners I M
    U : Set M
    a b : M
    hpc : IsPreconnected U
    hc✝ : IsCompact U
    ho : IsOpen U
    ha : Membership.mem U a
    hb : Membership.mem U b
    f : M → F
    hd : MDifferentiableOn I (modelWithCornersSelf Complex F) f U
    hb₀ : Eq (f b) 0
    c : M
    hcU : Membership.mem U c
    hc : IsMaxOn (fun x => Norm.norm (f x)) U c
    this : ∀ (x : M), Membership.mem U x → Eq (Norm.norm (f x)) (Norm.norm (f c))
    ⊢ Eq (f a) (f b)
  -/
  rw [hb₀, ← norm_eq_zero, this a ha, ← this b hb, hb₀, norm_zero]
  /-
    🎉 no goals
  -/


/-- A holomorphic function on a compact complex manifold is locally constant. -/
protected theorem isLocallyConstant {f : M → F} (hf : MDifferentiable I 𝓘(ℂ, F) f) :
    IsLocallyConstant f :=
  haveI : LocallyConnectedSpace H := I.toHomeomorph.locallyConnectedSpace
  haveI : LocallyConnectedSpace M := ChartedSpace.locallyConnectedSpace H M
  IsLocallyConstant.of_constant_on_preconnected_clopens fun _ hpc hclo _a ha _b hb ↦
    hf.mdifferentiableOn.apply_eq_of_isPreconnected_isCompact_isOpen hpc
      hclo.isClosed.isCompact hclo.isOpen hb ha


/-- A holomorphic function on a compact connected complex manifold is constant. -/
theorem apply_eq_of_compactSpace [PreconnectedSpace M] {f : M → F}
    (hf : MDifferentiable I 𝓘(ℂ, F) f) (a b : M) : f a = f b :=
  hf.isLocallyConstant.apply_eq_of_preconnectedSpace _ _


/-- A holomorphic function on a compact connected complex manifold is the constant function `f ≡ v`,
for some value `v`. -/
theorem exists_eq_const_of_compactSpace [PreconnectedSpace M] {f : M → F}
    (hf : MDifferentiable I 𝓘(ℂ, F) f) : ∃ v : F, f = Function.const M v :=
  hf.isLocallyConstant.exists_eq_const


