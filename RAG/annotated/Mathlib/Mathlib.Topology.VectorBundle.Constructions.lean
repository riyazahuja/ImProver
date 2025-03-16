instance trivialization.isLinear : (trivialization B F).IsLinear 𝕜 where
  linear _ _ := ⟨fun _ _ => rfl, fun _ _ => rfl⟩


theorem trivialization.coordChangeL (b : B) :
    (trivialization B F).coordChangeL 𝕜 (trivialization B F) b =
      ContinuousLinearEquiv.refl 𝕜 F := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace B
    b : B
    ⊢ Eq (Trivialization.coordChangeL 𝕜 (Bundle.Trivial.trivialization B F) (Bundl …
  -/
  ext v
  /-
    case h.h
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace B
    b : B
    v : F
    ⊢ Eq ((Trivialization.coordChangeL 𝕜 (Bundle.Trivial.trivialization B F) (Bund …
  -/
  rw [Trivialization.coordChangeL_apply']
  /-
    case h.h
    𝕜 : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace B
    b : B
    v : F
    ⊢ Eq (↑(Bundle.Trivial.trivialization B F) (↑(Bundle.Trivial.trivialization B  …
  -/
  exacts [rfl, ⟨mem_univ _, mem_univ _⟩]
  /-
    🎉 no goals
  -/


instance vectorBundle : VectorBundle 𝕜 F (Bundle.Trivial B F) where
  trivialization_linear' e he := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : TopologicalSpace B
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      ⊢ Trivialization.IsLinear 𝕜 e
    -/
    rw [eq_trivialization B F e]
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : TopologicalSpace B
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      ⊢ Trivialization.IsLinear 𝕜 (Bundle.Trivial.trivialization B F)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  continuousOn_coordChange' e e' he he' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : TopologicalSpace B
      e e' : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      he' : MemTrivializationAtlas e'
      ⊢ ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜 e e' b)) (Inter.inter …
    -/
    obtain rfl := eq_trivialization B F e
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : TopologicalSpace B
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      he : MemTrivializationAtlas (Bundle.Trivial.trivialization B F)
      ⊢ ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜 (Bundle.Trivial.trivi …
    -/
    obtain rfl := eq_trivialization B F e'
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : TopologicalSpace B
      he he' : MemTrivializationAtlas (Bundle.Trivial.trivialization B F)
      ⊢ ContinuousOn (fun b => ↑(Trivialization.coordChangeL 𝕜 (Bundle.Trivial.trivi …
    -/
    simp only [trivialization.coordChangeL]
    /-
      𝕜 : Type u_1
      B : Type u_2
      F : Type u_3
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : TopologicalSpace B
      he he' : MemTrivializationAtlas (Bundle.Trivial.trivialization B F)
      ⊢ ContinuousOn (fun b => ↑(ContinuousLinearEquiv.refl 𝕜 F)) (Inter.inter (Bund …
    -/
    exact continuous_const.continuousOn
    /-
      🎉 no goals
    -/


instance prod.isLinear [e₁.IsLinear 𝕜] [e₂.IsLinear 𝕜] : (e₁.prod e₂).IsLinear 𝕜 where
  linear := fun _ ⟨h₁, h₂⟩ =>
    (((e₁.linear 𝕜 h₁).mk' _).prodMap ((e₂.linear 𝕜 h₂).mk' _)).isLinear


@[simp]
theorem coordChangeL_prod [e₁.IsLinear 𝕜] [e₁'.IsLinear 𝕜] [e₂.IsLinear 𝕜] [e₂'.IsLinear 𝕜] ⦃b⦄
    (hb : b ∈ (e₁.prod e₂).baseSet ∩ (e₁'.prod e₂').baseSet) :
    ((e₁.prod e₂).coordChangeL 𝕜 (e₁'.prod e₂') b : F₁ × F₂ →L[𝕜] F₁ × F₂) =
      (e₁.coordChangeL 𝕜 e₁' b : F₁ →L[𝕜] F₁).prodMap (e₂.coordChangeL 𝕜 e₂' b) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹³ : NormedAddCommGroup F₁
    inst✝¹² : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹⁰ : NormedAddCommGroup F₂
    inst✝⁹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝⁸ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁷ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁵ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁴ : (x : B) → Module 𝕜 (E₂ x)
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝³ : Trivialization.IsLinear 𝕜 e₁
    inst✝² : Trivialization.IsLinear 𝕜 e₁'
    inst✝¹ : Trivialization.IsLinear 𝕜 e₂
    inst✝ : Trivialization.IsLinear 𝕜 e₂'
    b : B
    hb : Membership.mem (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) b
    ⊢ Eq (↑(Trivialization.coordChangeL 𝕜 (e₁.prod e₂) (e₁'.prod e₂') b)) ((↑(Triv …
  -/
  rw [ContinuousLinearMap.ext_iff, ContinuousLinearMap.coe_prodMap']
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹³ : NormedAddCommGroup F₁
    inst✝¹² : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹⁰ : NormedAddCommGroup F₂
    inst✝⁹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝⁸ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁷ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁵ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁴ : (x : B) → Module 𝕜 (E₂ x)
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝³ : Trivialization.IsLinear 𝕜 e₁
    inst✝² : Trivialization.IsLinear 𝕜 e₁'
    inst✝¹ : Trivialization.IsLinear 𝕜 e₂
    inst✝ : Trivialization.IsLinear 𝕜 e₂'
    b : B
    hb : Membership.mem (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) b
    ⊢ ∀ (x : Prod F₁ F₂), Eq (↑(Trivialization.coordChangeL 𝕜 (e₁.prod e₂) (e₁'.pr …
  -/
  rintro ⟨v₁, v₂⟩
  show
    (e₁.prod e₂).coordChangeL 𝕜 (e₁'.prod e₂') b (v₁, v₂) =
      (e₁.coordChangeL 𝕜 e₁' b v₁, e₂.coordChangeL 𝕜 e₂' b v₂)
  /-
    case mk
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹³ : NormedAddCommGroup F₁
    inst✝¹² : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹⁰ : NormedAddCommGroup F₂
    inst✝⁹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝⁸ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁷ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁵ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁴ : (x : B) → Module 𝕜 (E₂ x)
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝³ : Trivialization.IsLinear 𝕜 e₁
    inst✝² : Trivialization.IsLinear 𝕜 e₁'
    inst✝¹ : Trivialization.IsLinear 𝕜 e₂
    inst✝ : Trivialization.IsLinear 𝕜 e₂'
    b : B
    hb : Membership.mem (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) b
    v₁ : F₁
    v₂ : F₂
    ⊢ Eq ((Trivialization.coordChangeL 𝕜 (e₁.prod e₂) (e₁'.prod e₂') b) { fst := v …
  -/
  rw [e₁.coordChangeL_apply e₁', e₂.coordChangeL_apply e₂', (e₁.prod e₂).coordChangeL_apply']
  /-
    case mk
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    inst✝¹⁴ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹³ : NormedAddCommGroup F₁
    inst✝¹² : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹⁰ : NormedAddCommGroup F₂
    inst✝⁹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝⁸ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁷ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁵ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁴ : (x : B) → Module 𝕜 (E₂ x)
    e₁ e₁' : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ e₂' : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝³ : Trivialization.IsLinear 𝕜 e₁
    inst✝² : Trivialization.IsLinear 𝕜 e₁'
    inst✝¹ : Trivialization.IsLinear 𝕜 e₂
    inst✝ : Trivialization.IsLinear 𝕜 e₂'
    b : B
    hb : Membership.mem (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) b
    v₁ : F₁
    v₂ : F₂
    ⊢ Eq (↑(e₁'.prod e₂') (↑(e₁.prod e₂).symm { fst := b, snd := { fst := v₁, snd  …
  -/
  exacts [rfl, hb, ⟨hb.1.2, hb.2.2⟩, ⟨hb.1.1, hb.2.1⟩]
  /-
    🎉 no goals
  -/


theorem prod_apply [e₁.IsLinear 𝕜] [e₂.IsLinear 𝕜] {x : B} (hx₁ : x ∈ e₁.baseSet)
    (hx₂ : x ∈ e₂.baseSet) (v₁ : E₁ x) (v₂ : E₂ x) :
    prod e₁ e₂ ⟨x, (v₁, v₂)⟩ =
      ⟨x, e₁.continuousLinearEquivAt 𝕜 x hx₁ v₁, e₂.continuousLinearEquivAt 𝕜 x hx₂ v₂⟩ :=
  rfl


/-- The product of two vector bundles is a vector bundle. -/
instance VectorBundle.prod [VectorBundle 𝕜 F₁ E₁] [VectorBundle 𝕜 F₂ E₂] :
    VectorBundle 𝕜 (F₁ × F₂) (E₁ ×ᵇ E₂) where
  trivialization_linear' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹⁷ : NontriviallyNormedField 𝕜
      inst✝¹⁶ : TopologicalSpace B
      F₁ : Type u_3
      inst✝¹⁵ : NormedAddCommGroup F₁
      inst✝¹⁴ : NormedSpace 𝕜 F₁
      E₁ : B → Type u_4
      inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_5
      inst✝¹² : NormedAddCommGroup F₂
      inst✝¹¹ : NormedSpace 𝕜 F₂
      E₂ : B → Type u_6
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
      inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
      inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
      inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
      inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
      inst✝³ : FiberBundle F₁ E₁
      inst✝² : FiberBundle F₂ E₂
      inst✝¹ : VectorBundle 𝕜 F₁ E₁
      inst✝ : VectorBundle 𝕜 F₂ E₂
      ⊢ ∀ (e : Trivialization (Prod F₁ F₂) Bundle.TotalSpace.proj) [inst : MemTrivia …
    -/
    rintro _ ⟨e₁, e₂, he₁, he₂, rfl⟩
    /-
      case mk.intro.intro.intro.intro
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹⁷ : NontriviallyNormedField 𝕜
      inst✝¹⁶ : TopologicalSpace B
      F₁ : Type u_3
      inst✝¹⁵ : NormedAddCommGroup F₁
      inst✝¹⁴ : NormedSpace 𝕜 F₁
      E₁ : B → Type u_4
      inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_5
      inst✝¹² : NormedAddCommGroup F₂
      inst✝¹¹ : NormedSpace 𝕜 F₂
      E₂ : B → Type u_6
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
      inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
      inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
      inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
      inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
      inst✝³ : FiberBundle F₁ E₁
      inst✝² : FiberBundle F₂ E₂
      inst✝¹ : VectorBundle 𝕜 F₁ E₁
      inst✝ : VectorBundle 𝕜 F₂ E₂
      e₁ : Trivialization F₁ Bundle.TotalSpace.proj
      e₂ : Trivialization F₂ Bundle.TotalSpace.proj
      he₁ : MemTrivializationAtlas e₁
      he₂ : MemTrivializationAtlas e₂
      ⊢ Trivialization.IsLinear 𝕜 (e₁.prod e₂)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  continuousOn_coordChange' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹⁷ : NontriviallyNormedField 𝕜
      inst✝¹⁶ : TopologicalSpace B
      F₁ : Type u_3
      inst✝¹⁵ : NormedAddCommGroup F₁
      inst✝¹⁴ : NormedSpace 𝕜 F₁
      E₁ : B → Type u_4
      inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
      F₂ : Type u_5
      inst✝¹² : NormedAddCommGroup F₂
      inst✝¹¹ : NormedSpace 𝕜 F₂
      E₂ : B → Type u_6
      inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
      inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
      inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
      inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
      inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
      inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
      inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
      inst✝³ : FiberBundle F₁ E₁
      inst✝² : FiberBundle F₂ E₂
      inst✝¹ : VectorBundle 𝕜 F₁ E₁
      inst✝ : VectorBundle 𝕜 F₂ E₂
      ⊢ ∀ (e e' : Trivialization (Prod F₁ F₂) Bundle.TotalSpace.proj) [inst : MemTri …
    -/
    rintro _ _ ⟨e₁, e₂, he₁, he₂, rfl⟩ ⟨e₁', e₂', he₁', he₂', rfl⟩
    refine (((continuousOn_coordChange 𝕜 e₁ e₁').mono ?_).prod_mapL 𝕜
      ((continuousOn_coordChange 𝕜 e₂ e₂').mono ?_)).congr ?_ <;>
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_1
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (e₁.prod e₂).baseSet (e₁'.prod e₂').baseSet) ( …
      -/
      dsimp only [baseSet_prod, mfld_simps]
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_1
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.int …
      -/
    · mfld_set_tac
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_2
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        ⊢ HasSubset.Subset (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.int …
      -/
    · mfld_set_tac
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_3
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        ⊢ Set.EqOn (fun b => ↑(Trivialization.coordChangeL 𝕜 (e₁.prod e₂) (e₁'.prod e₂ …
      -/
    · rintro b hb
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_3
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        b : B
        hb : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
        ⊢ Eq ((fun b => ↑(Trivialization.coordChangeL 𝕜 (e₁.prod e₂) (e₁'.prod e₂') b) …
      -/
      rw [ContinuousLinearMap.ext_iff]
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_3
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        b : B
        hb : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
        ⊢ ∀ (x : Prod F₁ F₂), Eq (((fun b => ↑(Trivialization.coordChangeL 𝕜 (e₁.prod  …
      -/
      rintro ⟨v₁, v₂⟩
      show (e₁.prod e₂).coordChangeL 𝕜 (e₁'.prod e₂') b (v₁, v₂) =
        (e₁.coordChangeL 𝕜 e₁' b v₁, e₂.coordChangeL 𝕜 e₂' b v₂)
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_3.mk
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        b : B
        hb : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
        v₁ : F₁
        v₂ : F₂
        ⊢ Eq ((Trivialization.coordChangeL 𝕜 (e₁.prod e₂) (e₁'.prod e₂') b) { fst := v …
      -/
      rw [e₁.coordChangeL_apply e₁', e₂.coordChangeL_apply e₂', (e₁.prod e₂).coordChangeL_apply']
      /-
        case mk.intro.intro.intro.intro.mk.intro.intro.intro.intro.refine_3.mk
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹⁷ : NontriviallyNormedField 𝕜
        inst✝¹⁶ : TopologicalSpace B
        F₁ : Type u_3
        inst✝¹⁵ : NormedAddCommGroup F₁
        inst✝¹⁴ : NormedSpace 𝕜 F₁
        E₁ : B → Type u_4
        inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
        F₂ : Type u_5
        inst✝¹² : NormedAddCommGroup F₂
        inst✝¹¹ : NormedSpace 𝕜 F₂
        E₂ : B → Type u_6
        inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
        inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
        inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
        inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
        inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
        inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
        inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
        inst✝³ : FiberBundle F₁ E₁
        inst✝² : FiberBundle F₂ E₂
        inst✝¹ : VectorBundle 𝕜 F₁ E₁
        inst✝ : VectorBundle 𝕜 F₂ E₂
        e₁ : Trivialization F₁ Bundle.TotalSpace.proj
        e₂ : Trivialization F₂ Bundle.TotalSpace.proj
        he₁ : MemTrivializationAtlas e₁
        he₂ : MemTrivializationAtlas e₂
        e₁' : Trivialization F₁ Bundle.TotalSpace.proj
        e₂' : Trivialization F₂ Bundle.TotalSpace.proj
        he₁' : MemTrivializationAtlas e₁'
        he₂' : MemTrivializationAtlas e₂'
        b : B
        hb : Membership.mem (Inter.inter (Inter.inter e₁.baseSet e₂.baseSet) (Inter.in …
        v₁ : F₁
        v₂ : F₂
        ⊢ Eq (↑(e₁'.prod e₂') (↑(e₁.prod e₂).symm { fst := b, snd := { fst := v₁, snd  …
      -/
      exacts [rfl, hb, ⟨hb.1.2, hb.2.2⟩, ⟨hb.1.1, hb.2.1⟩]
      /-
        🎉 no goals
      -/


@[simp]
theorem Trivialization.continuousLinearEquivAt_prod {e₁ : Trivialization F₁ (π F₁ E₁)}
    {e₂ : Trivialization F₂ (π F₂ E₂)} [e₁.IsLinear 𝕜] [e₂.IsLinear 𝕜] {x : B}
    (hx : x ∈ (e₁.prod e₂).baseSet) :
    (e₁.prod e₂).continuousLinearEquivAt 𝕜 x hx =
      (e₁.continuousLinearEquivAt 𝕜 x hx.1).prod (e₂.continuousLinearEquivAt 𝕜 x hx.2) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁷ : NontriviallyNormedField 𝕜
    inst✝¹⁶ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹⁵ : NormedAddCommGroup F₁
    inst✝¹⁴ : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹² : NormedAddCommGroup F₂
    inst✝¹¹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
    inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : FiberBundle F₂ E₂
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear 𝕜 e₁
    inst✝ : Trivialization.IsLinear 𝕜 e₂
    x : B
    hx : Membership.mem (e₁.prod e₂).baseSet x
    ⊢ Eq (Trivialization.continuousLinearEquivAt 𝕜 (e₁.prod e₂) x hx) ((Trivializa …
  -/
  ext v : 2
  /-
    case h.h
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁷ : NontriviallyNormedField 𝕜
    inst✝¹⁶ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹⁵ : NormedAddCommGroup F₁
    inst✝¹⁴ : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹² : NormedAddCommGroup F₂
    inst✝¹¹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
    inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : FiberBundle F₂ E₂
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear 𝕜 e₁
    inst✝ : Trivialization.IsLinear 𝕜 e₂
    x : B
    hx : Membership.mem (e₁.prod e₂).baseSet x
    v : Prod (E₁ x) (E₂ x)
    ⊢ Eq ((Trivialization.continuousLinearEquivAt 𝕜 (e₁.prod e₂) x hx) v) (((Trivi …
  -/
  obtain ⟨v₁, v₂⟩ := v
  /-
    case h.h.mk
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁷ : NontriviallyNormedField 𝕜
    inst✝¹⁶ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹⁵ : NormedAddCommGroup F₁
    inst✝¹⁴ : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹² : NormedAddCommGroup F₂
    inst✝¹¹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
    inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : FiberBundle F₂ E₂
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear 𝕜 e₁
    inst✝ : Trivialization.IsLinear 𝕜 e₂
    x : B
    hx : Membership.mem (e₁.prod e₂).baseSet x
    v₁ : E₁ x
    v₂ : E₂ x
    ⊢ Eq ((Trivialization.continuousLinearEquivAt 𝕜 (e₁.prod e₂) x hx) { fst := v₁ …
  -/
  rw [(e₁.prod e₂).continuousLinearEquivAt_apply 𝕜, Trivialization.prod]
  /-
    case h.h.mk
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹⁷ : NontriviallyNormedField 𝕜
    inst✝¹⁶ : TopologicalSpace B
    F₁ : Type u_3
    inst✝¹⁵ : NormedAddCommGroup F₁
    inst✝¹⁴ : NormedSpace 𝕜 F₁
    E₁ : B → Type u_4
    inst✝¹³ : TopologicalSpace (Bundle.TotalSpace F₁ E₁)
    F₂ : Type u_5
    inst✝¹² : NormedAddCommGroup F₂
    inst✝¹¹ : NormedSpace 𝕜 F₂
    E₂ : B → Type u_6
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F₂ E₂)
    inst✝⁹ : (x : B) → AddCommMonoid (E₁ x)
    inst✝⁸ : (x : B) → Module 𝕜 (E₁ x)
    inst✝⁷ : (x : B) → AddCommMonoid (E₂ x)
    inst✝⁶ : (x : B) → Module 𝕜 (E₂ x)
    inst✝⁵ : (x : B) → TopologicalSpace (E₁ x)
    inst✝⁴ : (x : B) → TopologicalSpace (E₂ x)
    inst✝³ : FiberBundle F₁ E₁
    inst✝² : FiberBundle F₂ E₂
    e₁ : Trivialization F₁ Bundle.TotalSpace.proj
    e₂ : Trivialization F₂ Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear 𝕜 e₁
    inst✝ : Trivialization.IsLinear 𝕜 e₂
    x : B
    hx : Membership.mem (e₁.prod e₂).baseSet x
    v₁ : E₁ x
    v₂ : E₂ x
    ⊢ Eq ((fun y => (↑{ toFun := Trivialization.Prod.toFun' e₁ e₂, invFun := Trivi …
  -/
  exact (congr_arg Prod.snd (prod_apply 𝕜 hx.1 hx.2 v₁ v₂) : _)
  /-
    🎉 no goals
  -/


instance [i : ∀ x : B, AddCommMonoid (E x)] (x : B') : AddCommMonoid ((f *ᵖ E) x) := i _


instance [Semiring R] [∀ x : B, AddCommMonoid (E x)] [i : ∀ x, Module R (E x)] (x : B') :
    Module R ((f *ᵖ E) x) := i _


instance Trivialization.pullback_linear (e : Trivialization F (π F E)) [e.IsLinear 𝕜] (f : K) :
    (Trivialization.pullback (B' := B') e f).IsLinear 𝕜 where
  linear _ h := e.linear 𝕜 h


instance VectorBundle.pullback [∀ x, TopologicalSpace (E x)] [FiberBundle F E] [VectorBundle 𝕜 F E]
    (f : K) : VectorBundle 𝕜 F ((f : B' → B) *ᵖ E) where
  trivialization_linear' := by
    /-
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
      ⊢ ∀ (e : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivializationAtl …
    -/
    rintro _ ⟨e, he, rfl⟩
    /-
      case mk.intro.intro
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      ⊢ Trivialization.IsLinear 𝕜 (e.pullback f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  continuousOn_coordChange' := by
    /-
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
      ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
    -/
    rintro _ _ ⟨e, he, rfl⟩ ⟨e', he', rfl⟩
    refine ((continuousOn_coordChange 𝕜 e e').comp
      (map_continuous f).continuousOn fun b hb => hb).congr ?_
    /-
      case mk.intro.intro.mk.intro.intro
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      ⊢ Set.EqOn (fun b => ↑(Trivialization.coordChangeL 𝕜 (e.pullback f) (e'.pullba …
    -/
    rintro b (hb : f b ∈ e.baseSet ∩ e'.baseSet); ext v
    /-
      case mk.intro.intro.mk.intro.intro.h
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
      e : Trivialization F Bundle.TotalSpace.proj
      he : MemTrivializationAtlas e
      e' : Trivialization F Bundle.TotalSpace.proj
      he' : MemTrivializationAtlas e'
      b : B'
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) (f b)
      v : F
      ⊢ Eq (((fun b => ↑(Trivialization.coordChangeL 𝕜 (e.pullback f) (e'.pullback f …
    -/
    show ((e.pullback f).coordChangeL 𝕜 (e'.pullback f) b) v = (e.coordChangeL 𝕜 e' (f b)) v
    /-
      case mk.intro.intro.mk.intro.intro.h
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
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
      R : Type u_1
      𝕜 : Type u_2
      B : Type u_3
      F : Type u_4
      E : B → Type u_5
      B' : Type u_6
      f✝ : B' → B
      inst✝¹² : TopologicalSpace B'
      inst✝¹¹ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace 𝕜 F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : (x : B) → AddCommMonoid (E x)
      inst✝⁵ : (x : B) → Module 𝕜 (E x)
      K : Type u_7
      inst✝⁴ : FunLike K B' B
      inst✝³ : ContinuousMapClass K B' B
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      inst✝ : VectorBundle 𝕜 F E
      f : K
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


