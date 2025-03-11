/-- For a smooth vector bundle `E` over a manifold `B` and a smooth map `f : B' → B`, the pullback
vector bundle `f *ᵖ E` is a smooth vector bundle. -/
instance SmoothVectorBundle.pullback : SmoothVectorBundle F (f *ᵖ E) IB' where
  contMDiffOn_coordChangeL := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      E : B → Type u_5
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      inst✝¹⁸ : (x : B) → AddCommMonoid (E x)
      inst✝¹⁷ : (x : B) → Module 𝕜 (E x)
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedSpace 𝕜 F
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹³ : (x : B) → TopologicalSpace (E x)
      EB : Type u_6
      inst✝¹² : NormedAddCommGroup EB
      inst✝¹¹ : NormedSpace 𝕜 EB
      HB : Type u_7
      inst✝¹⁰ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁹ : TopologicalSpace B
      inst✝⁸ : ChartedSpace HB B
      EB' : Type u_8
      inst✝⁷ : NormedAddCommGroup EB'
      inst✝⁶ : NormedSpace 𝕜 EB'
      HB' : Type u_9
      inst✝⁵ : TopologicalSpace HB'
      IB' : ModelWithCorners 𝕜 EB' HB'
      inst✝⁴ : TopologicalSpace B'
      inst✝³ : ChartedSpace HB' B'
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      f : ContMDiffMap IB' IB B' B Top.top
      ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
    -/
    rintro _ _ ⟨e, he, rfl⟩ ⟨e', he', rfl⟩
    /-
      case mk.intro.intro.mk.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      E : B → Type u_5
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      inst✝¹⁸ : (x : B) → AddCommMonoid (E x)
      inst✝¹⁷ : (x : B) → Module 𝕜 (E x)
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedSpace 𝕜 F
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹³ : (x : B) → TopologicalSpace (E x)
      EB : Type u_6
      inst✝¹² : NormedAddCommGroup EB
      inst✝¹¹ : NormedSpace 𝕜 EB
      HB : Type u_7
      inst✝¹⁰ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁹ : TopologicalSpace B
      inst✝⁸ : ChartedSpace HB B
      EB' : Type u_8
      inst✝⁷ : NormedAddCommGroup EB'
      inst✝⁶ : NormedSpace 𝕜 EB'
      HB' : Type u_9
      inst✝⁵ : TopologicalSpace HB'
      IB' : ModelWithCorners 𝕜 EB' HB'
      inst✝⁴ : TopologicalSpace B'
      inst✝³ : ChartedSpace HB' B'
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      f : ContMDiffMap IB' IB B' B Top.top
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      ⊢ ContMDiffOn IB' (modelWithCornersSelf 𝕜 (ContinuousLinearMap (RingHom.id 𝕜)  …
    -/
    refine ((contMDiffOn_coordChangeL e e').comp f.contMDiff.contMDiffOn fun b hb => hb).congr ?_
    /-
      case mk.intro.intro.mk.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      E : B → Type u_5
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      inst✝¹⁸ : (x : B) → AddCommMonoid (E x)
      inst✝¹⁷ : (x : B) → Module 𝕜 (E x)
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedSpace 𝕜 F
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹³ : (x : B) → TopologicalSpace (E x)
      EB : Type u_6
      inst✝¹² : NormedAddCommGroup EB
      inst✝¹¹ : NormedSpace 𝕜 EB
      HB : Type u_7
      inst✝¹⁰ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁹ : TopologicalSpace B
      inst✝⁸ : ChartedSpace HB B
      EB' : Type u_8
      inst✝⁷ : NormedAddCommGroup EB'
      inst✝⁶ : NormedSpace 𝕜 EB'
      HB' : Type u_9
      inst✝⁵ : TopologicalSpace HB'
      IB' : ModelWithCorners 𝕜 EB' HB'
      inst✝⁴ : TopologicalSpace B'
      inst✝³ : ChartedSpace HB' B'
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      f : ContMDiffMap IB' IB B' B Top.top
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      ⊢ ∀ (y : B'), Membership.mem (Set.preimage (⇑f) (Inter.inter e.baseSet e'.base …
    -/
    rintro b (hb : f b ∈ e.baseSet ∩ e'.baseSet); ext v
    /-
      case mk.intro.intro.mk.intro.intro.h
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      E : B → Type u_5
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      inst✝¹⁸ : (x : B) → AddCommMonoid (E x)
      inst✝¹⁷ : (x : B) → Module 𝕜 (E x)
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedSpace 𝕜 F
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹³ : (x : B) → TopologicalSpace (E x)
      EB : Type u_6
      inst✝¹² : NormedAddCommGroup EB
      inst✝¹¹ : NormedSpace 𝕜 EB
      HB : Type u_7
      inst✝¹⁰ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁹ : TopologicalSpace B
      inst✝⁸ : ChartedSpace HB B
      EB' : Type u_8
      inst✝⁷ : NormedAddCommGroup EB'
      inst✝⁶ : NormedSpace 𝕜 EB'
      HB' : Type u_9
      inst✝⁵ : TopologicalSpace HB'
      IB' : ModelWithCorners 𝕜 EB' HB'
      inst✝⁴ : TopologicalSpace B'
      inst✝³ : ChartedSpace HB' B'
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      f : ContMDiffMap IB' IB B' B Top.top
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      b : B'
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) (f b)
      v : F
      ⊢ Eq (↑(Trivialization.coordChangeL 𝕜 (e.pullback f) (e'.pullback f) b) v) ((F …
    -/
    show ((e.pullback f).coordChangeL 𝕜 (e'.pullback f) b) v = (e.coordChangeL 𝕜 e' (f b)) v
    /-
      case mk.intro.intro.mk.intro.intro.h
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      E : B → Type u_5
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      inst✝¹⁸ : (x : B) → AddCommMonoid (E x)
      inst✝¹⁷ : (x : B) → Module 𝕜 (E x)
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedSpace 𝕜 F
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹³ : (x : B) → TopologicalSpace (E x)
      EB : Type u_6
      inst✝¹² : NormedAddCommGroup EB
      inst✝¹¹ : NormedSpace 𝕜 EB
      HB : Type u_7
      inst✝¹⁰ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁹ : TopologicalSpace B
      inst✝⁸ : ChartedSpace HB B
      EB' : Type u_8
      inst✝⁷ : NormedAddCommGroup EB'
      inst✝⁶ : NormedSpace 𝕜 EB'
      HB' : Type u_9
      inst✝⁵ : TopologicalSpace HB'
      IB' : ModelWithCorners 𝕜 EB' HB'
      inst✝⁴ : TopologicalSpace B'
      inst✝³ : ChartedSpace HB' B'
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      f : ContMDiffMap IB' IB B' B Top.top
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      b : B'
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) (f b)
      v : F
      ⊢ Eq ((Trivialization.coordChangeL 𝕜 (e.pullback f) (e'.pullback f) b) v) ((Tr …
    -/
    rw [e.coordChangeL_apply e' hb, (e.pullback f).coordChangeL_apply' _]
    /-
      case mk.intro.intro.mk.intro.intro.h
      𝕜 : Type u_1
      B : Type u_2
      B' : Type u_3
      F : Type u_4
      E : B → Type u_5
      inst✝¹⁹ : NontriviallyNormedField 𝕜
      inst✝¹⁸ : (x : B) → AddCommMonoid (E x)
      inst✝¹⁷ : (x : B) → Module 𝕜 (E x)
      inst✝¹⁶ : NormedAddCommGroup F
      inst✝¹⁵ : NormedSpace 𝕜 F
      inst✝¹⁴ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹³ : (x : B) → TopologicalSpace (E x)
      EB : Type u_6
      inst✝¹² : NormedAddCommGroup EB
      inst✝¹¹ : NormedSpace 𝕜 EB
      HB : Type u_7
      inst✝¹⁰ : TopologicalSpace HB
      IB : ModelWithCorners 𝕜 EB HB
      inst✝⁹ : TopologicalSpace B
      inst✝⁸ : ChartedSpace HB B
      EB' : Type u_8
      inst✝⁷ : NormedAddCommGroup EB'
      inst✝⁶ : NormedSpace 𝕜 EB'
      HB' : Type u_9
      inst✝⁵ : TopologicalSpace HB'
      IB' : ModelWithCorners 𝕜 EB' HB'
      inst✝⁴ : TopologicalSpace B'
      inst✝³ : ChartedSpace HB' B'
      inst✝² : FiberBundle F E
      inst✝¹ : VectorBundle 𝕜 F E
      inst✝ : SmoothVectorBundle F E IB
      f : ContMDiffMap IB' IB B' B Top.top
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      b : B'
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) (f b)
      v : F
      ⊢ Eq (↑(e'.pullback f) (↑(e.pullback f).symm { fst := b, snd := v })).2 (↑e' { …
    -/
    exacts [rfl, hb]
    /-
      🎉 no goals
    -/

