/-- For `B` a topological space and `F` a `𝕜`-normed space, a map from `U : Set B` to `F ≃L[𝕜] F`
determines a partial homeomorphism from `B × F` to itself by its action fiberwise. -/
def partialHomeomorph (φ : B → F ≃L[𝕜] F) (hU : IsOpen U)
    (hφ : ContinuousOn (fun x => φ x : B → F →L[𝕜] F) U)
    (h2φ : ContinuousOn (fun x => (φ x).symm : B → F →L[𝕜] F) U) :
    PartialHomeomorph (B × F) (B × F) where
  toFun x := (x.1, φ x.1 x.2)
  invFun x := (x.1, (φ x.1).symm x.2)
  source := U ×ˢ univ
  target := U ×ˢ univ
  map_source' _x hx := mk_mem_prod hx.1 (mem_univ _)
  map_target' _x hx := mk_mem_prod hx.1 (mem_univ _)
  left_inv' _ _ := Prod.ext rfl (ContinuousLinearEquiv.symm_apply_apply _ _)
  right_inv' _ _ := Prod.ext rfl (ContinuousLinearEquiv.apply_symm_apply _ _)
  open_source := hU.prod isOpen_univ
  open_target := hU.prod isOpen_univ
  continuousOn_toFun :=
    have : ContinuousOn (fun p : B × F => ((φ p.1 : F →L[𝕜] F), p.2)) (U ×ˢ univ) :=
      hφ.prod_map continuousOn_id
    continuousOn_fst.prod (isBoundedBilinearMap_apply.continuous.comp_continuousOn this)
  continuousOn_invFun :=
    haveI : ContinuousOn (fun p : B × F => (((φ p.1).symm : F →L[𝕜] F), p.2)) (U ×ˢ univ) :=
      h2φ.prod_map continuousOn_id
    continuousOn_fst.prod (isBoundedBilinearMap_apply.continuous.comp_continuousOn this)


/-- Compute the composition of two partial homeomorphisms induced by fiberwise linear
equivalences. -/
theorem trans_partialHomeomorph_apply (hU : IsOpen U)
    (hφ : ContinuousOn (fun x => φ x : B → F →L[𝕜] F) U)
    (h2φ : ContinuousOn (fun x => (φ x).symm : B → F →L[𝕜] F) U) (hU' : IsOpen U')
    (hφ' : ContinuousOn (fun x => φ' x : B → F →L[𝕜] F) U')
    (h2φ' : ContinuousOn (fun x => (φ' x).symm : B → F →L[𝕜] F) U') (b : B) (v : F) :
    (FiberwiseLinear.partialHomeomorph φ hU hφ h2φ ≫ₕ
      FiberwiseLinear.partialHomeomorph φ' hU' hφ' h2φ')
        ⟨b, v⟩ =
      ⟨b, φ' b (φ b v)⟩ :=
  rfl


/-- Compute the source of the composition of two partial homeomorphisms induced by fiberwise linear
equivalences. -/
theorem source_trans_partialHomeomorph (hU : IsOpen U)
    (hφ : ContinuousOn (fun x => φ x : B → F →L[𝕜] F) U)
    (h2φ : ContinuousOn (fun x => (φ x).symm : B → F →L[𝕜] F) U) (hU' : IsOpen U')
    (hφ' : ContinuousOn (fun x => φ' x : B → F →L[𝕜] F) U')
    (h2φ' : ContinuousOn (fun x => (φ' x).symm : B → F →L[𝕜] F) U') :
    (FiberwiseLinear.partialHomeomorph φ hU hφ h2φ ≫ₕ
          FiberwiseLinear.partialHomeomorph φ' hU' hφ' h2φ').source =
      (U ∩ U') ×ˢ univ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    φ φ' : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
    U U' : Set B
    hU : IsOpen U
    hφ : ContinuousOn (fun x => ↑(φ x)) U
    h2φ : ContinuousOn (fun x => ↑(φ x).symm) U
    hU' : IsOpen U'
    hφ' : ContinuousOn (fun x => ↑(φ' x)) U'
    h2φ' : ContinuousOn (fun x => ↑(φ' x).symm) U'
    ⊢ Eq ((FiberwiseLinear.partialHomeomorph φ hU hφ h2φ).trans (FiberwiseLinear.p …
  -/
  dsimp only [FiberwiseLinear.partialHomeomorph]; mfld_set_tac
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Compute the target of the composition of two partial homeomorphisms induced by fiberwise linear
equivalences. -/
theorem target_trans_partialHomeomorph (hU : IsOpen U)
    (hφ : ContinuousOn (fun x => φ x : B → F →L[𝕜] F) U)
    (h2φ : ContinuousOn (fun x => (φ x).symm : B → F →L[𝕜] F) U) (hU' : IsOpen U')
    (hφ' : ContinuousOn (fun x => φ' x : B → F →L[𝕜] F) U')
    (h2φ' : ContinuousOn (fun x => (φ' x).symm : B → F →L[𝕜] F) U') :
    (FiberwiseLinear.partialHomeomorph φ hU hφ h2φ ≫ₕ
          FiberwiseLinear.partialHomeomorph φ' hU' hφ' h2φ').target =
      (U ∩ U') ×ˢ univ := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    φ φ' : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
    U U' : Set B
    hU : IsOpen U
    hφ : ContinuousOn (fun x => ↑(φ x)) U
    h2φ : ContinuousOn (fun x => ↑(φ x).symm) U
    hU' : IsOpen U'
    hφ' : ContinuousOn (fun x => ↑(φ' x)) U'
    h2φ' : ContinuousOn (fun x => ↑(φ' x).symm) U'
    ⊢ Eq ((FiberwiseLinear.partialHomeomorph φ hU hφ h2φ).trans (FiberwiseLinear.p …
  -/
  dsimp only [FiberwiseLinear.partialHomeomorph]; mfld_set_tac
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Let `e` be a partial homeomorphism of `B × F`.  Suppose that at every point `p` in the source of
`e`, there is some neighbourhood `s` of `p` on which `e` is equal to a bi-smooth fiberwise linear
partial homeomorphism.
Then the source of `e` is of the form `U ×ˢ univ`, for some set `U` in `B`, and, at any point `x` in
`U`, admits a neighbourhood `u` of `x` such that `e` is equal on `u ×ˢ univ` to some bi-smooth
fiberwise linear partial homeomorphism. -/
theorem SmoothFiberwiseLinear.locality_aux₁ (e : PartialHomeomorph (B × F) (B × F))
    (h : ∀ p ∈ e.source, ∃ s : Set (B × F), IsOpen s ∧ p ∈ s ∧
      ∃ (φ : B → F ≃L[𝕜] F) (u : Set B) (hu : IsOpen u)
        (hφ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x : F →L[𝕜] F)) u)
        (h2φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => ((φ x).symm : F →L[𝕜] F)) u),
          (e.restr s).EqOnSource
            (FiberwiseLinear.partialHomeomorph φ hu hφ.continuousOn h2φ.continuousOn)) :
    ∃ U : Set B, e.source = U ×ˢ univ ∧ ∀ x ∈ U,
        ∃ (φ : B → F ≃L[𝕜] F) (u : Set B) (hu : IsOpen u) (_huU : u ⊆ U) (_hux : x ∈ u),
          ∃ (hφ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x : F →L[𝕜] F)) u)
            (h2φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => ((φ x).symm : F →L[𝕜] F)) u),
            (e.restr (u ×ˢ univ)).EqOnSource
              (FiberwiseLinear.partialHomeomorph φ hu hφ.continuousOn h2φ.continuousOn) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    EB : Type u_4
    inst✝³ : NormedAddCommGroup EB
    inst✝² : NormedSpace 𝕜 EB
    HB : Type u_5
    inst✝¹ : TopologicalSpace HB
    inst✝ : ChartedSpace HB B
    IB : ModelWithCorners 𝕜 EB HB
    e : PartialHomeomorph (Prod B F) (Prod B F)
    h : ∀ (p : Prod B F), Membership.mem e.source p → Exists fun s => And (IsOpen  …
    ⊢ Exists fun U => And (Eq e.source (SProd.sprod U Set.univ)) (∀ (x : B), Membe …
  -/
  rw [SetCoe.forall'] at h
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    EB : Type u_4
    inst✝³ : NormedAddCommGroup EB
    inst✝² : NormedSpace 𝕜 EB
    HB : Type u_5
    inst✝¹ : TopologicalSpace HB
    inst✝ : ChartedSpace HB B
    IB : ModelWithCorners 𝕜 EB HB
    e : PartialHomeomorph (Prod B F) (Prod B F)
    h : ∀ (x : ↑e.source), Exists fun s => And (IsOpen s) (And (Membership.mem s ↑ …
    ⊢ Exists fun U => And (Eq e.source (SProd.sprod U Set.univ)) (∀ (x : B), Membe …
  -/
  choose s hs hsp φ u hu hφ h2φ heφ using h
  have hesu : ∀ p : e.source, e.source ∩ s p = u p ×ˢ univ := by
    intro p
    rw [← e.restr_source' (s _) (hs _)]
    exact (heφ p).1
  have hu' : ∀ p : e.source, (p : B × F).fst ∈ u p := by
    intro p
    have : (p : B × F) ∈ e.source ∩ s p := ⟨p.prop, hsp p⟩
    simpa only [hesu, mem_prod, mem_univ, and_true] using this
  have heu : ∀ p : e.source, ∀ q : B × F, q.fst ∈ u p → q ∈ e.source := by
    intro p q hq
    have : q ∈ u p ×ˢ (univ : Set F) := ⟨hq, trivial⟩
    rw [← hesu p] at this
    exact this.1
  have he : e.source = (Prod.fst '' e.source) ×ˢ (univ : Set F) := by
    apply HasSubset.Subset.antisymm
    · intro p hp
      exact ⟨⟨p, hp, rfl⟩, trivial⟩
    · rintro ⟨x, v⟩ ⟨⟨p, hp, rfl : p.fst = x⟩, -⟩
      exact heu ⟨p, hp⟩ (p.fst, v) (hu' ⟨p, hp⟩)
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    EB : Type u_4
    inst✝³ : NormedAddCommGroup EB
    inst✝² : NormedSpace 𝕜 EB
    HB : Type u_5
    inst✝¹ : TopologicalSpace HB
    inst✝ : ChartedSpace HB B
    IB : ModelWithCorners 𝕜 EB HB
    e : PartialHomeomorph (Prod B F) (Prod B F)
    s : ↑e.source → Set (Prod B F)
    hs : ∀ (x : ↑e.source), IsOpen (s x)
    hsp : ∀ (x : ↑e.source), Membership.mem (s x) ↑x
    φ : ↑e.source → B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
    u : ↑e.source → Set B
    hu : ∀ (x : ↑e.source), IsOpen (u x)
    hφ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLine …
    h2φ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLin …
    heφ : ∀ (x : ↑e.source), (e.restr (s x)).EqOnSource (FiberwiseLinear.partialHo …
    hesu : ∀ (p : ↑e.source), Eq (Inter.inter e.source (s p)) (SProd.sprod (u p) S …
    hu' : ∀ (p : ↑e.source), Membership.mem (u p) (↑p).1
    heu : ∀ (p : ↑e.source) (q : Prod B F), Membership.mem (u p) q.1 → Membership. …
    he : Eq e.source (SProd.sprod (Set.image Prod.fst e.source) Set.univ)
    ⊢ Exists fun U => And (Eq e.source (SProd.sprod U Set.univ)) (∀ (x : B), Membe …
  -/
  refine ⟨Prod.fst '' e.source, he, ?_⟩
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    EB : Type u_4
    inst✝³ : NormedAddCommGroup EB
    inst✝² : NormedSpace 𝕜 EB
    HB : Type u_5
    inst✝¹ : TopologicalSpace HB
    inst✝ : ChartedSpace HB B
    IB : ModelWithCorners 𝕜 EB HB
    e : PartialHomeomorph (Prod B F) (Prod B F)
    s : ↑e.source → Set (Prod B F)
    hs : ∀ (x : ↑e.source), IsOpen (s x)
    hsp : ∀ (x : ↑e.source), Membership.mem (s x) ↑x
    φ : ↑e.source → B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
    u : ↑e.source → Set B
    hu : ∀ (x : ↑e.source), IsOpen (u x)
    hφ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLine …
    h2φ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLin …
    heφ : ∀ (x : ↑e.source), (e.restr (s x)).EqOnSource (FiberwiseLinear.partialHo …
    hesu : ∀ (p : ↑e.source), Eq (Inter.inter e.source (s p)) (SProd.sprod (u p) S …
    hu' : ∀ (p : ↑e.source), Membership.mem (u p) (↑p).1
    heu : ∀ (p : ↑e.source) (q : Prod B F), Membership.mem (u p) q.1 → Membership. …
    he : Eq e.source (SProd.sprod (Set.image Prod.fst e.source) Set.univ)
    ⊢ ∀ (x : B), Membership.mem (Set.image Prod.fst e.source) x → Exists fun φ =>  …
  -/
  rintro x ⟨p, hp, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    EB : Type u_4
    inst✝³ : NormedAddCommGroup EB
    inst✝² : NormedSpace 𝕜 EB
    HB : Type u_5
    inst✝¹ : TopologicalSpace HB
    inst✝ : ChartedSpace HB B
    IB : ModelWithCorners 𝕜 EB HB
    e : PartialHomeomorph (Prod B F) (Prod B F)
    s : ↑e.source → Set (Prod B F)
    hs : ∀ (x : ↑e.source), IsOpen (s x)
    hsp : ∀ (x : ↑e.source), Membership.mem (s x) ↑x
    φ : ↑e.source → B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
    u : ↑e.source → Set B
    hu : ∀ (x : ↑e.source), IsOpen (u x)
    hφ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLine …
    h2φ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLin …
    heφ : ∀ (x : ↑e.source), (e.restr (s x)).EqOnSource (FiberwiseLinear.partialHo …
    hesu : ∀ (p : ↑e.source), Eq (Inter.inter e.source (s p)) (SProd.sprod (u p) S …
    hu' : ∀ (p : ↑e.source), Membership.mem (u p) (↑p).1
    heu : ∀ (p : ↑e.source) (q : Prod B F), Membership.mem (u p) q.1 → Membership. …
    he : Eq e.source (SProd.sprod (Set.image Prod.fst e.source) Set.univ)
    p : Prod B F
    hp : Membership.mem e.source p
    ⊢ Exists fun φ => Exists fun u => Exists fun hu => Exists fun _huU => Exists f …
  -/
  refine ⟨φ ⟨p, hp⟩, u ⟨p, hp⟩, hu ⟨p, hp⟩, ?_, hu' _, hφ ⟨p, hp⟩, h2φ ⟨p, hp⟩, ?_⟩
    /-
      case intro.intro.refine_1
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      s : ↑e.source → Set (Prod B F)
      hs : ∀ (x : ↑e.source), IsOpen (s x)
      hsp : ∀ (x : ↑e.source), Membership.mem (s x) ↑x
      φ : ↑e.source → B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      u : ↑e.source → Set B
      hu : ∀ (x : ↑e.source), IsOpen (u x)
      hφ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLine …
      h2φ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLin …
      heφ : ∀ (x : ↑e.source), (e.restr (s x)).EqOnSource (FiberwiseLinear.partialHo …
      hesu : ∀ (p : ↑e.source), Eq (Inter.inter e.source (s p)) (SProd.sprod (u p) S …
      hu' : ∀ (p : ↑e.source), Membership.mem (u p) (↑p).1
      heu : ∀ (p : ↑e.source) (q : Prod B F), Membership.mem (u p) q.1 → Membership. …
      he : Eq e.source (SProd.sprod (Set.image Prod.fst e.source) Set.univ)
      p : Prod B F
      hp : Membership.mem e.source p
      ⊢ HasSubset.Subset (u ⟨p, hp⟩) (Set.image Prod.fst e.source)
    -/
  · intro y hy; exact ⟨(y, 0), heu ⟨p, hp⟩ ⟨_, _⟩ hy, rfl⟩
                /-
                  🎉 no goals
                -/
    /-
      case intro.intro.refine_2
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      s : ↑e.source → Set (Prod B F)
      hs : ∀ (x : ↑e.source), IsOpen (s x)
      hsp : ∀ (x : ↑e.source), Membership.mem (s x) ↑x
      φ : ↑e.source → B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      u : ↑e.source → Set B
      hu : ∀ (x : ↑e.source), IsOpen (u x)
      hφ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLine …
      h2φ : ∀ (x : ↑e.source), ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLin …
      heφ : ∀ (x : ↑e.source), (e.restr (s x)).EqOnSource (FiberwiseLinear.partialHo …
      hesu : ∀ (p : ↑e.source), Eq (Inter.inter e.source (s p)) (SProd.sprod (u p) S …
      hu' : ∀ (p : ↑e.source), Membership.mem (u p) (↑p).1
      heu : ∀ (p : ↑e.source) (q : Prod B F), Membership.mem (u p) q.1 → Membership. …
      he : Eq e.source (SProd.sprod (Set.image Prod.fst e.source) Set.univ)
      p : Prod B F
      hp : Membership.mem e.source p
      ⊢ (e.restr (SProd.sprod (u ⟨p, hp⟩) Set.univ)).EqOnSource (FiberwiseLinear.par …
    -/
  · rw [← hesu, e.restr_source_inter]; exact heφ ⟨p, hp⟩
                                       /-
                                         🎉 no goals
                                       -/


/-- Let `e` be a partial homeomorphism of `B × F` whose source is `U ×ˢ univ`, for some set `U` in
`B`, and which, at any point `x` in `U`, admits a neighbourhood `u` of `x` such that `e` is equal
on `u ×ˢ univ` to some bi-smooth fiberwise linear partial homeomorphism.  Then `e` itself
is equal to some bi-smooth fiberwise linear partial homeomorphism.

This is the key mathematical point of the `locality` condition in the construction of the
`StructureGroupoid` of bi-smooth fiberwise linear partial homeomorphisms.  The proof is by gluing
together the various bi-smooth fiberwise linear partial homeomorphism which exist locally.

The `U` in the conclusion is the same `U` as in the hypothesis. We state it like this, because this
is exactly what we need for `smoothFiberwiseLinear`. -/
theorem SmoothFiberwiseLinear.locality_aux₂ (e : PartialHomeomorph (B × F) (B × F)) (U : Set B)
    (hU : e.source = U ×ˢ univ)
    (h : ∀ x ∈ U,
      ∃ (φ : B → F ≃L[𝕜] F) (u : Set B) (hu : IsOpen u) (_hUu : u ⊆ U) (_hux : x ∈ u)
        (hφ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x : F →L[𝕜] F)) u)
        (h2φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => ((φ x).symm : F →L[𝕜] F)) u),
          (e.restr (u ×ˢ univ)).EqOnSource
            (FiberwiseLinear.partialHomeomorph φ hu hφ.continuousOn h2φ.continuousOn)) :
    ∃ (Φ : B → F ≃L[𝕜] F) (U : Set B) (hU₀ : IsOpen U) (hΦ :
      ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (Φ x : F →L[𝕜] F)) U) (h2Φ :
      ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => ((Φ x).symm : F →L[𝕜] F)) U),
      e.EqOnSource (FiberwiseLinear.partialHomeomorph Φ hU₀ hΦ.continuousOn h2Φ.continuousOn) := by
  classical
  rw [SetCoe.forall'] at h
  choose! φ u hu hUu hux hφ h2φ heφ using h
  have heuφ : ∀ x : U, EqOn e (fun q => (q.1, φ x q.1 q.2)) (u x ×ˢ univ) := fun x p hp ↦ by
    refine (heφ x).2 ?_
    rw [(heφ x).1]
    exact hp
  have huφ : ∀ (x x' : U) (y : B), y ∈ u x → y ∈ u x' → φ x y = φ x' y := fun p p' y hyp hyp' ↦ by
    ext v
    have h1 : e (y, v) = (y, φ p y v) := heuφ _ ⟨(id hyp : (y, v).fst ∈ u p), trivial⟩
    have h2 : e (y, v) = (y, φ p' y v) := heuφ _ ⟨(id hyp' : (y, v).fst ∈ u p'), trivial⟩
    exact congr_arg Prod.snd (h1.symm.trans h2)
  have hUu' : U = ⋃ i, u i := by
    ext x
    rw [mem_iUnion]
    refine ⟨fun h => ⟨⟨x, h⟩, hux _⟩, ?_⟩
    rintro ⟨x, hx⟩
    exact hUu x hx
  have hU' : IsOpen U := by
    rw [hUu']
    apply isOpen_iUnion hu
  let Φ₀ : U → F ≃L[𝕜] F := iUnionLift u (fun x => φ x ∘ (↑)) huφ U hUu'.le
  let Φ : B → F ≃L[𝕜] F := fun y =>
    if hy : y ∈ U then Φ₀ ⟨y, hy⟩ else ContinuousLinearEquiv.refl 𝕜 F
  have hΦ : ∀ (y) (hy : y ∈ U), Φ y = Φ₀ ⟨y, hy⟩ := fun y hy => dif_pos hy
  have hΦφ : ∀ x : U, ∀ y ∈ u x, Φ y = φ x y := by
    intro x y hyu
    refine (hΦ y (hUu x hyu)).trans ?_
    exact iUnionLift_mk ⟨y, hyu⟩ _
  have hΦ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun y => (Φ y : F →L[𝕜] F)) U := by
    apply contMDiffOn_of_locally_contMDiffOn
    intro x hx
    refine ⟨u ⟨x, hx⟩, hu ⟨x, hx⟩, hux _, ?_⟩
    refine (ContMDiffOn.congr (hφ ⟨x, hx⟩) ?_).mono inter_subset_right
    intro y hy
    rw [hΦφ ⟨x, hx⟩ y hy]
  have h2Φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun y => ((Φ y).symm : F →L[𝕜] F)) U := by
    apply contMDiffOn_of_locally_contMDiffOn
    intro x hx
    refine ⟨u ⟨x, hx⟩, hu ⟨x, hx⟩, hux _, ?_⟩
    refine (ContMDiffOn.congr (h2φ ⟨x, hx⟩) ?_).mono inter_subset_right
    intro y hy
    rw [hΦφ ⟨x, hx⟩ y hy]
  refine ⟨Φ, U, hU', hΦ, h2Φ, hU, fun p hp => ?_⟩
  rw [hU] at hp
  rw [heuφ ⟨p.fst, hp.1⟩ ⟨hux _, hp.2⟩]
  congrm (_, ?_)
  rw [hΦφ]
  apply hux


variable {F B IB} in
-- Having this private lemma speeds up `simp` calls below a lot.
-- TODO: understand why and fix the underlying issue (relatedly, the `simp` calls
-- in `smoothFiberwiseLinear` are quite slow, even with this change)
private theorem mem_aux {e : PartialHomeomorph (B × F) (B × F)} :
    (e ∈ ⋃ (φ : B → F ≃L[𝕜] F) (U : Set B) (hU : IsOpen U)
      (hφ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => φ x : B → F →L[𝕜] F) U)
      (h2φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x).symm : B → F →L[𝕜] F) U),
        {e | e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU hφ.continuousOn
          h2φ.continuousOn)}) ↔
      ∃ (φ : B → F ≃L[𝕜] F) (U : Set B) (hU : IsOpen U)
        (hφ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => φ x : B → F →L[𝕜] F) U)
        (h2φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x).symm : B → F →L[𝕜] F) U),
          e.EqOnSource
            (FiberwiseLinear.partialHomeomorph φ hU hφ.continuousOn h2φ.continuousOn) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    EB : Type u_4
    inst✝³ : NormedAddCommGroup EB
    inst✝² : NormedSpace 𝕜 EB
    HB : Type u_5
    inst✝¹ : TopologicalSpace HB
    inst✝ : ChartedSpace HB B
    IB : ModelWithCorners 𝕜 EB HB
    e : PartialHomeomorph (Prod B F) (Prod B F)
    ⊢ Iff (Membership.mem (Set.iUnion fun φ => Set.iUnion fun U => Set.iUnion fun  …
  -/
  simp only [mem_iUnion, mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- For `B` a manifold and `F` a normed space, the groupoid on `B × F` consisting of local
homeomorphisms which are bi-smooth and fiberwise linear, and induce the identity on `B`.
When a (topological) vector bundle is smooth, then the composition of charts associated
to the vector bundle belong to this groupoid. -/
def smoothFiberwiseLinear : StructureGroupoid (B × F) where
  members :=
    ⋃ (φ : B → F ≃L[𝕜] F) (U : Set B) (hU : IsOpen U)
      (hφ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => φ x : B → F →L[𝕜] F) U)
      (h2φ : ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x).symm : B → F →L[𝕜] F) U),
        {e | e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU hφ.continuousOn h2φ.continuousOn)}
  trans' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ ∀ (e e' : PartialHomeomorph (Prod B F) (Prod B F)), Membership.mem (Set.iUni …
    -/
    simp only [mem_aux]
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ ∀ (e e' : PartialHomeomorph (Prod B F) (Prod B F)), (Exists fun φ => Exists  …
    -/
    rintro e e' ⟨φ, U, hU, hφ, h2φ, heφ⟩ ⟨φ', U', hU', hφ', h2φ', heφ'⟩
    refine ⟨fun b => (φ b).trans (φ' b), _, hU.inter hU', ?_, ?_,
      Setoid.trans (PartialHomeomorph.EqOnSource.trans' heφ heφ') ⟨?_, ?_⟩⟩
    · show
        ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤
          (fun x : B => (φ' x).toContinuousLinearMap ∘L (φ x).toContinuousLinearMap) (U ∩ U')
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_3
        inst✝⁷ : TopologicalSpace B
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        EB : Type u_4
        inst✝³ : NormedAddCommGroup EB
        inst✝² : NormedSpace 𝕜 EB
        HB : Type u_5
        inst✝¹ : TopologicalSpace HB
        inst✝ : ChartedSpace HB B
        IB : ModelWithCorners 𝕜 EB HB
        e e' : PartialHomeomorph (Prod B F) (Prod B F)
        φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U : Set B
        hU : IsOpen U
        hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
        h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
        φ' : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U' : Set B
        hU' : IsOpen U'
        hφ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        h2φ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id …
        heφ' : e'.EqOnSource (FiberwiseLinear.partialHomeomorph φ' hU' ⋯ ⋯)
        ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
      -/
      exact (hφ'.mono inter_subset_right).clm_comp (hφ.mono inter_subset_left)
      /-
        🎉 no goals
      -/
    · show
        ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤
          (fun x : B => (φ x).symm.toContinuousLinearMap ∘L (φ' x).symm.toContinuousLinearMap)
          (U ∩ U')
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_3
        inst✝⁷ : TopologicalSpace B
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        EB : Type u_4
        inst✝³ : NormedAddCommGroup EB
        inst✝² : NormedSpace 𝕜 EB
        HB : Type u_5
        inst✝¹ : TopologicalSpace HB
        inst✝ : ChartedSpace HB B
        IB : ModelWithCorners 𝕜 EB HB
        e e' : PartialHomeomorph (Prod B F) (Prod B F)
        φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U : Set B
        hU : IsOpen U
        hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
        h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
        φ' : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U' : Set B
        hU' : IsOpen U'
        hφ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        h2φ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id …
        heφ' : e'.EqOnSource (FiberwiseLinear.partialHomeomorph φ' hU' ⋯ ⋯)
        ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
      -/
      exact (h2φ.mono inter_subset_left).clm_comp (h2φ'.mono inter_subset_right)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_3
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_3
        inst✝⁷ : TopologicalSpace B
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        EB : Type u_4
        inst✝³ : NormedAddCommGroup EB
        inst✝² : NormedSpace 𝕜 EB
        HB : Type u_5
        inst✝¹ : TopologicalSpace HB
        inst✝ : ChartedSpace HB B
        IB : ModelWithCorners 𝕜 EB HB
        e e' : PartialHomeomorph (Prod B F) (Prod B F)
        φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U : Set B
        hU : IsOpen U
        hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
        h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
        φ' : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U' : Set B
        hU' : IsOpen U'
        hφ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        h2φ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id …
        heφ' : e'.EqOnSource (FiberwiseLinear.partialHomeomorph φ' hU' ⋯ ⋯)
        ⊢ Eq ((FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯).trans (FiberwiseLinear.part …
      -/
    · apply FiberwiseLinear.source_trans_partialHomeomorph
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_4
        𝕜 : Type u_1
        B : Type u_2
        F : Type u_3
        inst✝⁷ : TopologicalSpace B
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        EB : Type u_4
        inst✝³ : NormedAddCommGroup EB
        inst✝² : NormedSpace 𝕜 EB
        HB : Type u_5
        inst✝¹ : TopologicalSpace HB
        inst✝ : ChartedSpace HB B
        IB : ModelWithCorners 𝕜 EB HB
        e e' : PartialHomeomorph (Prod B F) (Prod B F)
        φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U : Set B
        hU : IsOpen U
        hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
        h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
        φ' : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
        U' : Set B
        hU' : IsOpen U'
        hφ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
        h2φ' : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id …
        heφ' : e'.EqOnSource (FiberwiseLinear.partialHomeomorph φ' hU' ⋯ ⋯)
        ⊢ Set.EqOn (↑((FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯).trans (FiberwiseLin …
      -/
    · rintro ⟨b, v⟩ -; apply FiberwiseLinear.trans_partialHomeomorph_apply
                       /-
                         🎉 no goals
                       -/
  -- Porting note: without introducing `e` first, the first `simp only` fails
  symm' := fun e ↦ by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      ⊢ Membership.mem (Set.iUnion fun φ => Set.iUnion fun U => Set.iUnion fun hU => …
    -/
    simp only [mem_aux]
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      ⊢ (Exists fun φ => Exists fun U => Exists fun hU => Exists fun hφ => Exists fu …
    -/
    rintro ⟨φ, U, hU, hφ, h2φ, heφ⟩
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ Exists fun φ => Exists fun U => Exists fun hU => Exists fun hφ => Exists fun …
    -/
    refine ⟨fun b => (φ b).symm, U, hU, h2φ, ?_, PartialHomeomorph.EqOnSource.symm' heφ⟩
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    simp_rw [ContinuousLinearEquiv.symm_symm]
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      ⊢ ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) F …
    -/
    exact hφ
    /-
      🎉 no goals
    -/
  id_mem' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ Membership.mem (Set.iUnion fun φ => Set.iUnion fun U => Set.iUnion fun hU => …
    -/
    simp_rw [mem_aux]
    refine ⟨fun _ ↦ ContinuousLinearEquiv.refl 𝕜 F, univ, isOpen_univ, contMDiffOn_const,
      contMDiffOn_const, ⟨?_, fun b _hb ↦ rfl⟩⟩
    simp only [FiberwiseLinear.partialHomeomorph, PartialHomeomorph.refl_partialEquiv,
      PartialEquiv.refl_source, univ_prod_univ]
  locality' := by
    -- the hard work has been extracted to `locality_aux₁` and `locality_aux₂`
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ ∀ (e : PartialHomeomorph (Prod B F) (Prod B F)), (∀ (x : Prod B F), Membersh …
    -/
    simp only [mem_aux]
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ ∀ (e : PartialHomeomorph (Prod B F) (Prod B F)), (∀ (x : Prod B F), Membersh …
    -/
    intro e he
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      he : ∀ (x : Prod B F), Membership.mem e.source x → Exists fun s => And (IsOpen …
      ⊢ Exists fun φ => Exists fun U => Exists fun hU => Exists fun hφ => Exists fun …
    -/
    obtain ⟨U, hU, h⟩ := SmoothFiberwiseLinear.locality_aux₁ e he
    /-
      case intro.intro
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e : PartialHomeomorph (Prod B F) (Prod B F)
      he : ∀ (x : Prod B F), Membership.mem e.source x → Exists fun s => And (IsOpen …
      U : Set B
      hU : Eq e.source (SProd.sprod U Set.univ)
      h : ∀ (x : B), Membership.mem U x → Exists fun φ => Exists fun u => Exists fun …
      ⊢ Exists fun φ => Exists fun U => Exists fun hU => Exists fun hφ => Exists fun …
    -/
    exact SmoothFiberwiseLinear.locality_aux₂ e U hU h
    /-
      🎉 no goals
    -/
  mem_of_eqOnSource' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ ∀ (e e' : PartialHomeomorph (Prod B F) (Prod B F)), Membership.mem (Set.iUni …
    -/
    simp only [mem_aux]
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      ⊢ ∀ (e e' : PartialHomeomorph (Prod B F) (Prod B F)), (Exists fun φ => Exists  …
    -/
    rintro e e' ⟨φ, U, hU, hφ, h2φ, heφ⟩ hee'
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      EB : Type u_4
      inst✝³ : NormedAddCommGroup EB
      inst✝² : NormedSpace 𝕜 EB
      HB : Type u_5
      inst✝¹ : TopologicalSpace HB
      inst✝ : ChartedSpace HB B
      IB : ModelWithCorners 𝕜 EB HB
      e e' : PartialHomeomorph (Prod B F) (Prod B F)
      φ : B → ContinuousLinearEquiv (RingHom.id 𝕜) F F
      U : Set B
      hU : IsOpen U
      hφ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜 …
      h2φ : ContMDiffOn IB (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id  …
      heφ : e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU ⋯ ⋯)
      hee' : HasEquiv.Equiv e' e
      ⊢ Exists fun φ => Exists fun U => Exists fun hU => Exists fun hφ => Exists fun …
    -/
    exact ⟨φ, U, hU, hφ, h2φ, Setoid.trans hee' heφ⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_smoothFiberwiseLinear_iff (e : PartialHomeomorph (B × F) (B × F)) :
    e ∈ smoothFiberwiseLinear B F IB ↔
      ∃ (φ : B → F ≃L[𝕜] F) (U : Set B) (hU : IsOpen U) (hφ :
        ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => φ x : B → F →L[𝕜] F) U) (h2φ :
        ContMDiffOn IB 𝓘(𝕜, F →L[𝕜] F) ⊤ (fun x => (φ x).symm : B → F →L[𝕜] F) U),
        e.EqOnSource (FiberwiseLinear.partialHomeomorph φ hU hφ.continuousOn h2φ.continuousOn) :=
  mem_aux

