/-- Proper group action in the sense of Bourbaki:
the map `G × X → X × X` is a proper map (see `IsProperMap`). -/
class ProperVAdd (G X : Type*) [TopologicalSpace G] [TopologicalSpace X] [AddGroup G]
    [AddAction G X] : Prop where
  /-- Proper group action in the sense of Bourbaki:
  the map `G × X → X × X` is a proper map (see `IsProperMap`). -/
  isProperMap_vadd_pair : IsProperMap (fun gx ↦ (gx.1 +ᵥ gx.2, gx.2) : G × X → X × X)


/-- Proper group action in the sense of Bourbaki:
the map `G × X → X × X` is a proper map (see `IsProperMap`). -/
@[to_additive existing (attr := mk_iff)]
class ProperSMul (G X : Type*) [TopologicalSpace G] [TopologicalSpace X] [Group G]
    [MulAction G X] : Prop where
  /-- Proper group action in the sense of Bourbaki:
  the map `G × X → X × X` is a proper map (see `IsProperMap`). -/
  isProperMap_smul_pair : IsProperMap (fun gx ↦ (gx.1 • gx.2, gx.2) : G × X → X × X)


/-- If a group acts properly then in particular it acts continuously. -/
@[to_additive "If a group acts properly then in particular it acts continuously."]
-- See note [lower instance property]
instance (priority := 100) ProperSMul.toContinuousSMul [ProperSMul G X] : ContinuousSMul G X where
  continuous_smul := isProperMap_smul_pair.continuous.fst


/-- A group `G` acts properly on a topological space `X` if and only if for all ultrafilters
`𝒰` on `X × G`, if `𝒰` converges to `(x₁, x₂)` along the map `(g, x) ↦ (g • x, x)`,
then there exists `g : G` such that `g • x₂ = x₁` and `𝒰.fst` converges to `g`. -/
@[to_additive "A group `G` acts properly on a topological space `X` if and only if
for all ultrafilters `𝒰` on `X`, if `𝒰` converges to `(x₁, x₂)`
along the map `(g, x) ↦ (g • x, x)`, then there exists `g : G` such that `g • x₂ = x₁`
and `𝒰.fst` converges to `g`."]
theorem properSMul_iff_continuousSMul_ultrafilter_tendsto :
    ProperSMul G X ↔ ContinuousSMul G X ∧
      (∀ 𝒰 : Ultrafilter (G × X), ∀ x₁ x₂ : X,
        Tendsto (fun gx : G × X ↦ (gx.1 • gx.2, gx.2)) 𝒰 (𝓝 (x₁, x₂)) →
      ∃ g : G, g • x₂ = x₁ ∧ Tendsto (Prod.fst : G × X → G) 𝒰 (𝓝 g)) := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝³ : Group G
    inst✝² : MulAction G X
    inst✝¹ : TopologicalSpace G
    inst✝ : TopologicalSpace X
    ⊢ Iff (ProperSMul G X) (And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X …
  -/
  refine ⟨fun h ↦ ⟨inferInstance, fun 𝒰 x₁ x₂ h' ↦ ?_⟩, fun ⟨cont, h⟩ ↦ ?_⟩
    /-
      case refine_1
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      h : ProperSMul G X
      𝒰 : Ultrafilter (Prod G X)
      x₁ x₂ : X
      h' : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 })  …
      ⊢ Exists fun g => And (Eq (HSMul.hSMul g x₂) x₁) (Filter.Tendsto Prod.fst (↑𝒰) …
    -/
  · rw [properSMul_iff, isProperMap_iff_ultrafilter] at h
    /-
      case refine_1
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      h : And (Continuous fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) ( …
      𝒰 : Ultrafilter (Prod G X)
      x₁ x₂ : X
      h' : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 })  …
      ⊢ Exists fun g => And (Eq (HSMul.hSMul g x₂) x₁) (Filter.Tendsto Prod.fst (↑𝒰) …
    -/
    rcases h.2 h' with ⟨gx, hgx1, hgx2⟩
    /-
      case refine_1.intro.intro
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      h : And (Continuous fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) ( …
      𝒰 : Ultrafilter (Prod G X)
      x₁ x₂ : X
      h' : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 })  …
      gx : Prod G X
      hgx1 : Eq { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 } { fst := x₁, snd := x₂ }
      hgx2 : LE.le (↑𝒰) (nhds gx)
      ⊢ Exists fun g => And (Eq (HSMul.hSMul g x₂) x₁) (Filter.Tendsto Prod.fst (↑𝒰) …
    -/
    refine ⟨gx.1, ?_, (continuous_fst.tendsto gx).mono_left hgx2⟩
    /-
      case refine_1.intro.intro
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      h : And (Continuous fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) ( …
      𝒰 : Ultrafilter (Prod G X)
      x₁ x₂ : X
      h' : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 })  …
      gx : Prod G X
      hgx1 : Eq { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 } { fst := x₁, snd := x₂ }
      hgx2 : LE.le (↑𝒰) (nhds gx)
      ⊢ Eq (HSMul.hSMul gx.1 x₂) x₁
    -/
    simp only [Prod.mk.injEq] at hgx1
    /-
      case refine_1.intro.intro
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      h : And (Continuous fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) ( …
      𝒰 : Ultrafilter (Prod G X)
      x₁ x₂ : X
      h' : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 })  …
      gx : Prod G X
      hgx2 : LE.le (↑𝒰) (nhds gx)
      hgx1 : And (Eq (HSMul.hSMul gx.1 gx.2) x₁) (Eq gx.2 x₂)
      ⊢ Eq (HSMul.hSMul gx.1 x₂) x₁
    -/
    rw [← hgx1.2, hgx1.1]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      x✝ : And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Fil …
      cont : ContinuousSMul G X
      h : ∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { fs …
      ⊢ ProperSMul G X
    -/
  · rw [properSMul_iff, isProperMap_iff_ultrafilter]
    /-
      case refine_2
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      x✝ : And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Fil …
      cont : ContinuousSMul G X
      h : ∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { fs …
      ⊢ And (Continuous fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) (∀  …
    -/
    refine ⟨by fun_prop, fun 𝒰 (x₁, x₂) hxx ↦ ?_⟩
    /-
      case refine_2
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      x✝¹ : And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Fi …
      cont : ContinuousSMul G X
      h : ∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { fs …
      𝒰 : Ultrafilter (Prod G X)
      x✝ : Prod X X
      x₁ x₂ : X
      hxx : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) …
      ⊢ Exists fun x => And (Eq { fst := HSMul.hSMul x.1 x.2, snd := x.2 } { fst :=  …
    -/
    rcases h 𝒰 x₁ x₂ hxx with ⟨g, hg1, hg2⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      x✝¹ : And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Fi …
      cont : ContinuousSMul G X
      h : ∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { fs …
      𝒰 : Ultrafilter (Prod G X)
      x✝ : Prod X X
      x₁ x₂ : X
      hxx : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) …
      g : G
      hg1 : Eq (HSMul.hSMul g x₂) x₁
      hg2 : Filter.Tendsto Prod.fst (↑𝒰) (nhds g)
      ⊢ Exists fun x => And (Eq { fst := HSMul.hSMul x.1 x.2, snd := x.2 } { fst :=  …
    -/
    refine ⟨(g, x₂), by simp_rw [hg1], ?_⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      x✝¹ : And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Fi …
      cont : ContinuousSMul G X
      h : ∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { fs …
      𝒰 : Ultrafilter (Prod G X)
      x✝ : Prod X X
      x₁ x₂ : X
      hxx : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) …
      g : G
      hg1 : Eq (HSMul.hSMul g x₂) x₁
      hg2 : Filter.Tendsto Prod.fst (↑𝒰) (nhds g)
      ⊢ LE.le (↑𝒰) (nhds { fst := g, snd := x₂ })
    -/
    rw [nhds_prod_eq, 𝒰.le_prod]
    /-
      case refine_2.intro.intro
      G : Type u_1
      X : Type u_2
      inst✝³ : Group G
      inst✝² : MulAction G X
      inst✝¹ : TopologicalSpace G
      inst✝ : TopologicalSpace X
      x✝¹ : And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Fi …
      cont : ContinuousSMul G X
      h : ∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { fs …
      𝒰 : Ultrafilter (Prod G X)
      x✝ : Prod X X
      x₁ x₂ : X
      hxx : Filter.Tendsto (fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }) …
      g : G
      hg1 : Eq (HSMul.hSMul g x₂) x₁
      hg2 : Filter.Tendsto Prod.fst (↑𝒰) (nhds g)
      ⊢ And (Filter.Tendsto Prod.fst (↑𝒰) (nhds g)) (Filter.Tendsto Prod.snd (↑𝒰) (n …
    -/
    exact ⟨hg2, (continuous_snd.tendsto _).comp hxx⟩
    /-
      🎉 no goals
    -/


/-- A group `G` acts properly on a T2 topological space `X` if and only if for all ultrafilters
`𝒰` on `X × G`, if `𝒰` converges to `(x₁, x₂)` along the map `(g, x) ↦ (g • x, x)`,
then there exists `g : G` such that `𝒰.fst` converges to `g`. -/
theorem properSMul_iff_continuousSMul_ultrafilter_tendsto_t2 [T2Space X] :
    ProperSMul G X ↔ ContinuousSMul G X ∧
      (∀ 𝒰 : Ultrafilter (G × X), ∀ x₁ x₂ : X,
        Tendsto (fun gx : G × X ↦ (gx.1 • gx.2, gx.2)) 𝒰 (𝓝 (x₁, x₂)) →
     ∃ g : G, Tendsto (Prod.fst : G × X → G) 𝒰 (𝓝 g)) := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    ⊢ Iff (ProperSMul G X) (And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X …
  -/
  rw [properSMul_iff_continuousSMul_ultrafilter_tendsto]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    ⊢ Iff (And (ContinuousSMul G X) (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), F …
  -/
  refine and_congr_right fun hc ↦ ?_
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    hc : ContinuousSMul G X
    ⊢ Iff (∀ (𝒰 : Ultrafilter (Prod G X)) (x₁ x₂ : X), Filter.Tendsto (fun gx => { …
  -/
  congrm ∀ 𝒰 x₁ x₂ hxx, ∃ g, ?_
  exact and_iff_right_of_imp fun hg ↦ tendsto_nhds_unique
    (hg.smul ((continuous_snd.tendsto _).comp hxx)) ((continuous_fst.tendsto _).comp hxx)


/-- If `G` acts properly on `X`, then the quotient space is Hausdorff (T2). -/
@[to_additive "If `G` acts properly on `X`, then the quotient space is Hausdorff (T2)."]
theorem t2Space_quotient_mulAction_of_properSMul [ProperSMul G X] :
    T2Space (Quotient (MulAction.orbitRel G X)) := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : ProperSMul G X
    ⊢ T2Space (Quotient (MulAction.orbitRel G X))
  -/
  rw [t2_iff_isClosed_diagonal]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : ProperSMul G X
    ⊢ IsClosed (Set.diagonal (Quotient (MulAction.orbitRel G X)))
  -/
  set R := MulAction.orbitRel G X
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : ProperSMul G X
    R : Setoid X := MulAction.orbitRel G X
    ⊢ IsClosed (Set.diagonal (Quotient R))
  -/
  let π : X → Quotient R := Quotient.mk'
  have : IsOpenQuotientMap (Prod.map π π) :=
    MulAction.isOpenQuotientMap_quotientMk.prodMap MulAction.isOpenQuotientMap_quotientMk
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : ProperSMul G X
    R : Setoid X := MulAction.orbitRel G X
    π : X → Quotient R := Quotient.mk'
    this : IsOpenQuotientMap (Prod.map π π)
    ⊢ IsClosed (Set.diagonal (Quotient R))
  -/
  rw [← this.isQuotientMap.isClosed_preimage]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : ProperSMul G X
    R : Setoid X := MulAction.orbitRel G X
    π : X → Quotient R := Quotient.mk'
    this : IsOpenQuotientMap (Prod.map π π)
    ⊢ IsClosed (Set.preimage (Prod.map π π) (Set.diagonal (Quotient R)))
  -/
  convert ProperSMul.isProperMap_smul_pair.isClosedMap.isClosed_range
    /-
      case h.e'_3
      G : Type u_1
      X : Type u_2
      inst✝⁴ : Group G
      inst✝³ : MulAction G X
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalSpace X
      inst✝ : ProperSMul G X
      R : Setoid X := MulAction.orbitRel G X
      π : X → Quotient R := Quotient.mk'
      this : IsOpenQuotientMap (Prod.map π π)
      ⊢ Eq (Set.preimage (Prod.map π π) (Set.diagonal (Quotient R))) (Set.range fun  …
    -/
  · ext ⟨x₁, x₂⟩
    simp only [mem_preimage, map_apply, mem_diagonal_iff, mem_range, Prod.mk.injEq, Prod.exists,
      exists_eq_right]
    /-
      case h.e'_3.h.mk
      G : Type u_1
      X : Type u_2
      inst✝⁴ : Group G
      inst✝³ : MulAction G X
      inst✝² : TopologicalSpace G
      inst✝¹ : TopologicalSpace X
      inst✝ : ProperSMul G X
      R : Setoid X := MulAction.orbitRel G X
      π : X → Quotient R := Quotient.mk'
      this : IsOpenQuotientMap (Prod.map π π)
      x₁ x₂ : X
      ⊢ Iff (Eq (π x₁) (π x₂)) (Exists fun a => Eq (HSMul.hSMul a x₂) x₁)
    -/
    rw [Quotient.eq', MulAction.orbitRel_apply, MulAction.mem_orbit_iff]
    /-
      🎉 no goals
    -/
  /-
    case convert_3
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    inst✝ : ProperSMul G X
    R : Setoid X := MulAction.orbitRel G X
    π : X → Quotient R := Quotient.mk'
    this : IsOpenQuotientMap (Prod.map π π)
    ⊢ TopologicalSpace G
  -/
  all_goals infer_instance
  /-
    🎉 no goals
  -/


/-- If a T2 group acts properly on a topological space, then this topological space is T2. -/
@[to_additive "If a T2 group acts properly on a topological space,
then this topological space is T2."]
theorem t2Space_of_properSMul_of_t2Group [h_proper : ProperSMul G X] [T2Space G] : T2Space X := by
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    ⊢ T2Space X
  -/
  let f := fun x : X ↦ ((1 : G), x)
  have proper_f : IsProperMap f := by
    refine IsClosedEmbedding.isProperMap ⟨?_, ?_⟩
    · let g := fun gx : G × X ↦ gx.2
      have : Function.LeftInverse g f := fun x ↦ by simp [f, g]
      exact this.isEmbedding (by fun_prop) (by fun_prop)
    · have : range f = ({1} ×ˢ univ) := by simp [f]
      rw [this]
      exact isClosed_singleton.prod isClosed_univ
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    ⊢ T2Space X
  -/
  rw [t2_iff_isClosed_diagonal]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    ⊢ IsClosed (Set.diagonal X)
  -/
  let g := fun gx : G × X ↦ (gx.1 • gx.2, gx.2)
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    g : Prod G X → Prod X X := fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx …
    ⊢ IsClosed (Set.diagonal X)
  -/
  have proper_g : IsProperMap g := (properSMul_iff G X).1 h_proper
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    g : Prod G X → Prod X X := fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx …
    proper_g : IsProperMap g
    ⊢ IsClosed (Set.diagonal X)
  -/
  have : g ∘ f = fun x ↦ (x, x) := by ext x <;> simp [f, g]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    g : Prod G X → Prod X X := fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx …
    proper_g : IsProperMap g
    this : Eq (Function.comp g f) fun x => { fst := x, snd := x }
    ⊢ IsClosed (Set.diagonal X)
  -/
  have range_gf : range (g ∘ f) = diagonal X := by simp [this]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    g : Prod G X → Prod X X := fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx …
    proper_g : IsProperMap g
    this : Eq (Function.comp g f) fun x => { fst := x, snd := x }
    range_gf : Eq (Set.range (Function.comp g f)) (Set.diagonal X)
    ⊢ IsClosed (Set.diagonal X)
  -/
  rw [← range_gf]
  /-
    G : Type u_1
    X : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G X
    inst✝² : TopologicalSpace G
    inst✝¹ : TopologicalSpace X
    h_proper : ProperSMul G X
    inst✝ : T2Space G
    f : X → Prod G X := fun x => { fst := 1, snd := x }
    proper_f : IsProperMap f
    g : Prod G X → Prod X X := fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx …
    proper_g : IsProperMap g
    this : Eq (Function.comp g f) fun x => { fst := x, snd := x }
    range_gf : Eq (Set.range (Function.comp g f)) (Set.diagonal X)
    ⊢ IsClosed (Set.range (Function.comp g f))
  -/
  exact (proper_f.comp proper_g).isClosed_range
  /-
    🎉 no goals
  -/


/-- If two groups `H` and `G` act on a topological space `X` such that `G` acts properly and
there exists a group homomorphims `H → G` which is a closed embedding compatible with the actions,
then `H` also acts properly on `X`. -/
@[to_additive "If two groups `H` and `G` act on a topological space `X` such that `G` acts properly
and there exists a group homomorphims `H → G` which is a closed embedding compatible with the
actions, then `H` also acts properly on `X`."]
theorem properSMul_of_isClosedEmbedding {H : Type*} [Group H] [MulAction H X] [TopologicalSpace H]
    [ProperSMul G X] (f : H →* G) (f_clemb : IsClosedEmbedding f)
    (f_compat : ∀ (h : H) (x : X), f h • x = h • x) : ProperSMul H X where
  isProperMap_smul_pair := by
    /-
      G : Type u_1
      X : Type u_2
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G X
      inst✝⁵ : TopologicalSpace G
      inst✝⁴ : TopologicalSpace X
      H : Type u_3
      inst✝³ : Group H
      inst✝² : MulAction H X
      inst✝¹ : TopologicalSpace H
      inst✝ : ProperSMul G X
      f : MonoidHom H G
      f_clemb : Topology.IsClosedEmbedding ⇑f
      f_compat : ∀ (h : H) (x : X), Eq (HSMul.hSMul (f h) x) (HSMul.hSMul h x)
      ⊢ IsProperMap fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }
    -/
    have h : IsProperMap (Prod.map f (fun x : X ↦ x)) := f_clemb.isProperMap.prodMap isProperMap_id
    have : (fun hx : H × X ↦ (hx.1 • hx.2, hx.2)) = (fun hx ↦ (f hx.1 • hx.2, hx.2)) := by
      simp [f_compat]
    /-
      G : Type u_1
      X : Type u_2
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G X
      inst✝⁵ : TopologicalSpace G
      inst✝⁴ : TopologicalSpace X
      H : Type u_3
      inst✝³ : Group H
      inst✝² : MulAction H X
      inst✝¹ : TopologicalSpace H
      inst✝ : ProperSMul G X
      f : MonoidHom H G
      f_clemb : Topology.IsClosedEmbedding ⇑f
      f_compat : ∀ (h : H) (x : X), Eq (HSMul.hSMul (f h) x) (HSMul.hSMul h x)
      h : IsProperMap (Prod.map ⇑f fun x => x)
      this : Eq (fun hx => { fst := HSMul.hSMul hx.1 hx.2, snd := hx.2 }) fun hx =>  …
      ⊢ IsProperMap fun gx => { fst := HSMul.hSMul gx.1 gx.2, snd := gx.2 }
    -/
    rw [this]
    /-
      G : Type u_1
      X : Type u_2
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G X
      inst✝⁵ : TopologicalSpace G
      inst✝⁴ : TopologicalSpace X
      H : Type u_3
      inst✝³ : Group H
      inst✝² : MulAction H X
      inst✝¹ : TopologicalSpace H
      inst✝ : ProperSMul G X
      f : MonoidHom H G
      f_clemb : Topology.IsClosedEmbedding ⇑f
      f_compat : ∀ (h : H) (x : X), Eq (HSMul.hSMul (f h) x) (HSMul.hSMul h x)
      h : IsProperMap (Prod.map ⇑f fun x => x)
      this : Eq (fun hx => { fst := HSMul.hSMul hx.1 hx.2, snd := hx.2 }) fun hx =>  …
      ⊢ IsProperMap fun hx => { fst := HSMul.hSMul (f hx.1) hx.2, snd := hx.2 }
    -/
    exact h.comp <| ProperSMul.isProperMap_smul_pair
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias properSMul_of_closedEmbedding := properSMul_of_isClosedEmbedding


/-- If `H` is a closed subgroup of `G` and `G` acts properly on X then so does `H`. -/
@[to_additive "If `H` is a closed subgroup of `G` and `G` acts properly on X then so does `H`."]
instance {H : Subgroup G} [ProperSMul G X] [H_closed : IsClosed (H : Set G)] : ProperSMul H X :=
  properSMul_of_isClosedEmbedding H.subtype H_closed.isClosedEmbedding_subtypeVal fun _ _ ↦ rfl

