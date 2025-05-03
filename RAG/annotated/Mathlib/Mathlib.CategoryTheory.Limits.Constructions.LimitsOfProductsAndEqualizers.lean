/--
(Implementation) Given the appropriate product and equalizer cones, build the cone for `F` which is
limiting if the given cones are also.
-/
@[simps]
def buildLimit
    (hs : ∀ f : Σp : J × J, p.1 ⟶ p.2, s ≫ c₂.π.app ⟨f⟩ = c₁.π.app ⟨f.1.1⟩ ≫ F.map f.2)
    (ht : ∀ f : Σp : J × J, p.1 ⟶ p.2, t ≫ c₂.π.app ⟨f⟩ = c₁.π.app ⟨f.1.2⟩)
    (i : Fork s t) : Cone F where
  pt := i.pt
  π :=
    { app := fun _ => i.ι ≫ c₁.π.app ⟨_⟩
      naturality := fun j₁ j₂ f => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J C
          c₁ : CategoryTheory.Limits.Fan F.obj
          c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
          s t : Quiver.Hom c₁.pt c₂.pt
          hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          i : CategoryTheory.Limits.Fork s t
          j₁ j₂ : J
          f : Quiver.Hom j₁ j₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J C
          c₁ : CategoryTheory.Limits.Fan F.obj
          c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
          s t : Quiver.Hom c₁.pt c₂.pt
          hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          i : CategoryTheory.Limits.Fork s t
          j₁ j₂ : J
          f : Quiver.Hom j₁ j₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id i.p …
        -/
        rw [Category.id_comp, Category.assoc, ← hs ⟨⟨_, _⟩, f⟩, i.condition_assoc, ht] }
        /-
          🎉 no goals
        -/


/--
(Implementation) Show the cone constructed in `buildLimit` is limiting, provided the cones used in
its construction are.
-/
def buildIsLimit (t₁ : IsLimit c₁) (t₂ : IsLimit c₂) (hi : IsLimit i) :
    IsLimit (buildLimit s t hs ht i) where
  lift q := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J C
      c₁ : CategoryTheory.Limits.Fan F.obj
      c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
      s t : Quiver.Hom c₁.pt c₂.pt
      hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
      ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
      i : CategoryTheory.Limits.Fork s t
      t₁ : CategoryTheory.Limits.IsLimit c₁
      t₂ : CategoryTheory.Limits.IsLimit c₂
      hi : CategoryTheory.Limits.IsLimit i
      q : CategoryTheory.Limits.Cone F
      ⊢ Quiver.Hom q.pt (CategoryTheory.Limits.HasLimitOfHasProductsOfHasEqualizers. …
    -/
    refine hi.lift (Fork.ofι ?_ ?_)
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Fan F.obj
        c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Fork s t
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        hi : CategoryTheory.Limits.IsLimit i
        q : CategoryTheory.Limits.Cone F
        ⊢ Quiver.Hom q.1 c₁.pt
      -/
    · refine t₁.lift (Fan.mk _ fun j => ?_)
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Fan F.obj
        c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Fork s t
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        hi : CategoryTheory.Limits.IsLimit i
        q : CategoryTheory.Limits.Cone F
        j : J
        ⊢ Quiver.Hom q.1 (F.obj j)
      -/
      apply q.π.app j
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Fan F.obj
        c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Fork s t
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        hi : CategoryTheory.Limits.IsLimit i
        q : CategoryTheory.Limits.Cone F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t₁.lift (CategoryTheory.Limits.Fan.m …
      -/
    · apply t₂.hom_ext
      /-
        case refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Fan F.obj
        c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Fork s t
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        hi : CategoryTheory.Limits.IsLimit i
        q : CategoryTheory.Limits.Cone F
        ⊢ ∀ (j : CategoryTheory.Discrete (Sigma fun p => Quiver.Hom p.1 p.2)), Eq (Cat …
      -/
      intro ⟨j⟩
      /-
        case refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Fan F.obj
        c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Fork s t
        t₁ : CategoryTheory.Limits.IsLimit c₁
        t₂ : CategoryTheory.Limits.IsLimit c₂
        hi : CategoryTheory.Limits.IsLimit i
        q : CategoryTheory.Limits.Cone F
        j : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp [hs, ht]
      /-
        🎉 no goals
      -/
  uniq q m w :=
    hi.hom_ext
      (i.equalizer_ext
        (t₁.hom_ext fun j => by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            J : Type w
            inst✝ : CategoryTheory.SmallCategory J
            F : CategoryTheory.Functor J C
            c₁ : CategoryTheory.Limits.Fan F.obj
            c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
            s t : Quiver.Hom c₁.pt c₂.pt
            hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            i : CategoryTheory.Limits.Fork s t
            t₁ : CategoryTheory.Limits.IsLimit c₁
            t₂ : CategoryTheory.Limits.IsLimit c₂
            hi : CategoryTheory.Limits.IsLimit i
            q : CategoryTheory.Limits.Cone F
            m : Quiver.Hom q.pt (CategoryTheory.Limits.HasLimitOfHasProductsOfHasEqualizer …
            w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
            j : CategoryTheory.Discrete J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
          -/
          cases' j with j
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  J : Type w
                  inst✝ : CategoryTheory.SmallCategory J
                  F : CategoryTheory.Functor J C
                  c₁ : CategoryTheory.Limits.Fan F.obj
                  c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
                  s✝ t : Quiver.Hom c₁.pt c₂.pt
                  hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
                  ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
                  i : CategoryTheory.Limits.Fork s✝ t
                  t₁ : CategoryTheory.Limits.IsLimit c₁
                  t₂ : CategoryTheory.Limits.IsLimit c₂
                  hi : CategoryTheory.Limits.IsLimit i
                  s : CategoryTheory.Limits.Cone F
                  j : J
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun q => hi.lift (CategoryTheory.Li …
                -/
          /-
            case mk
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            J : Type w
            inst✝ : CategoryTheory.SmallCategory J
            F : CategoryTheory.Functor J C
            c₁ : CategoryTheory.Limits.Fan F.obj
            c₂ : CategoryTheory.Limits.Fan fun f => F.obj f.fst.2
            s t : Quiver.Hom c₁.pt c₂.pt
            hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            i : CategoryTheory.Limits.Fork s t
            t₁ : CategoryTheory.Limits.IsLimit c₁
            t₂ : CategoryTheory.Limits.IsLimit c₂
            hi : CategoryTheory.Limits.IsLimit i
            q : CategoryTheory.Limits.Cone F
            m : Quiver.Hom q.pt (CategoryTheory.Limits.HasLimitOfHasProductsOfHasEqualizer …
            w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
          -/
                /-
                  🎉 no goals
                -/
          simpa using w j))
          /-
            🎉 no goals
          -/
  fac s j := by simp


/-- Given the existence of the appropriate (possibly finite) products and equalizers,
we can construct a limit cone for `F`.
(This assumes the existence of all equalizers, which is technically stronger than needed.)
-/
noncomputable def limitConeOfEqualizerAndProduct (F : J ⥤ C) [HasLimit (Discrete.functor F.obj)]
    [HasLimit (Discrete.functor fun f : Σp : J × J, p.1 ⟶ p.2 => F.obj f.1.2)] [HasEqualizers C] :
    LimitCone F where
  cone := _
  isLimit :=
    buildIsLimit (Pi.lift fun f => limit.π (Discrete.functor F.obj) ⟨_⟩ ≫ F.map f.2)
                                                                      /-
                                                                        C : Type u
                                                                        inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                        J : Type w
                                                                        inst✝³ : CategoryTheory.SmallCategory J
                                                                        F : CategoryTheory.Functor J C
                                                                        inst✝² : CategoryTheory.Limits.HasLimit (CategoryTheory.Discrete.functor F.obj)
                                                                        inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Discrete.functor fun f …
                                                                        inst✝ : CategoryTheory.Limits.HasEqualizers C
                                                                        ⊢ ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStruct …
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
      (Pi.lift fun f => limit.π (Discrete.functor F.obj) ⟨f.1.2⟩) (by simp) (by simp)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
      (limit.isLimit _) (limit.isLimit _) (limit.isLimit _)


/--
Given the existence of the appropriate (possibly finite) products and equalizers, we know a limit of
`F` exists.
(This assumes the existence of all equalizers, which is technically stronger than needed.)
-/
theorem hasLimit_of_equalizer_and_product (F : J ⥤ C) [HasLimit (Discrete.functor F.obj)]
    [HasLimit (Discrete.functor fun f : Σp : J × J, p.1 ⟶ p.2 => F.obj f.1.2)] [HasEqualizers C] :
    HasLimit F :=
  HasLimit.mk (limitConeOfEqualizerAndProduct F)


/-- A limit can be realised as a subobject of a product. -/
noncomputable def limitSubobjectProduct [HasLimitsOfSize.{w, w} C] (F : J ⥤ C) :
    limit F ⟶ ∏ᶜ fun j => F.obj j :=
  have := hasFiniteLimits_of_hasLimitsOfSize C
  (limit.isoLimitCone (limitConeOfEqualizerAndProduct F)).hom ≫ equalizer.ι _ _


instance limitSubobjectProduct_mono [HasLimitsOfSize.{w, w} C] (F : J ⥤ C) :
    Mono (limitSubobjectProduct F) :=
  mono_comp _ _


/-- Any category with products and equalizers has all limits.

See <https://stacks.math.columbia.edu/tag/002N>.
-/
theorem has_limits_of_hasEqualizers_and_products [HasProducts.{w} C] [HasEqualizers C] :
    HasLimitsOfSize.{w, w} C :=
  { has_limits_of_shape :=
    fun _ _ => { has_limit := fun F => hasLimit_of_equalizer_and_product F } }


/-- Any category with finite products and equalizers has all finite limits.

See <https://stacks.math.columbia.edu/tag/002O>.
-/
theorem hasFiniteLimits_of_hasEqualizers_and_finite_products [HasFiniteProducts C]
    [HasEqualizers C] : HasFiniteLimits C where
  out _ := { has_limit := fun F => hasLimit_of_equalizer_and_product F }


/-- If a functor preserves equalizers and the appropriate products, it preserves limits. -/
lemma preservesLimit_of_preservesEqualizers_and_product :
    PreservesLimitsOfShape J G where
  preservesLimit {K} := by
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      ⊢ CategoryTheory.Limits.PreservesLimit K G
    -/
    let P := ∏ᶜ K.obj
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.piObj K.obj
      ⊢ CategoryTheory.Limits.PreservesLimit K G
    -/
    let Q := ∏ᶜ fun f : Σp : J × J, p.fst ⟶ p.snd => K.obj f.1.2
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.piObj K.obj
      Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
      ⊢ CategoryTheory.Limits.PreservesLimit K G
    -/
    let s : P ⟶ Q := Pi.lift fun f => limit.π (Discrete.functor K.obj) ⟨_⟩ ≫ K.map f.2
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.piObj K.obj
      Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
      s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
      ⊢ CategoryTheory.Limits.PreservesLimit K G
    -/
    let t : P ⟶ Q := Pi.lift fun f => limit.π (Discrete.functor K.obj) ⟨f.1.2⟩
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.piObj K.obj
      Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
      s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
      t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
      ⊢ CategoryTheory.Limits.PreservesLimit K G
    -/
    let I := equalizer s t
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.piObj K.obj
      Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
      s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
      t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
      I : C := CategoryTheory.Limits.equalizer s t
      ⊢ CategoryTheory.Limits.PreservesLimit K G
    -/
    let i : I ⟶ P := equalizer.ι s t
    apply preservesLimit_of_preserves_limit_cone
        (buildIsLimit s t (by simp [P, s]) (by simp [P, t]) (limit.isLimit _)
          (limit.isLimit _) (limit.isLimit _))
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.piObj K.obj
      Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
      s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
      t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
      I : C := CategoryTheory.Limits.equalizer s t
      i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.HasLimitOfHa …
    -/
    apply IsLimit.ofIsoLimit (buildIsLimit _ _ _ _ _ _ _) _
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.Fan (K.comp G).obj
      -/
    · exact Fan.mk _ fun j => G.map (Pi.π _ j)
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.Fan fun f => (K.comp G).obj f.fst.2
      -/
    · exact Fan.mk (G.obj Q) fun f => G.map (Pi.π _ f)
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ Quiver.Hom (CategoryTheory.Limits.Fan.mk (G.obj (CategoryTheory.Limits.piObj …
      -/
    · apply G.map s
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ Quiver.Hom (CategoryTheory.Limits.Fan.mk (G.obj (CategoryTheory.Limits.piObj …
      -/
    · apply G.map t
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStruct …
      -/
    · intro f
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map s) ((CategoryTheory.Limits.Fan …
      -/
      dsimp [P, Q, s, Fan.mk]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.Pi.lift …
      -/
      simp only [← G.map_comp, limit.lift_π]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (G.map ((CategoryTheory.Limits.Fan.mk (CategoryTheory.Limits.piObj K.obj) …
      -/
      congr
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStruct …
      -/
    · intro f
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map t) ((CategoryTheory.Limits.Fan …
      -/
      dsimp [P, Q, t, Fan.mk]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.Pi.lift …
      -/
      simp only [← G.map_comp, limit.lift_π]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (G.map ((CategoryTheory.Limits.Fan.mk (CategoryTheory.Limits.piObj K.obj) …
      -/
      apply congrArg G.map
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq ((CategoryTheory.Limits.Fan.mk (CategoryTheory.Limits.piObj K.obj) fun f  …
      -/
      dsimp
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.Fork (G.map s) (G.map t)
      -/
    · apply Fork.ofι (G.map i)
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map i) (G.map s)) (CategoryTheory. …
      -/
      rw [← G.map_comp, ← G.map_comp]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp i s)) (G.map (CategoryTheory.C …
      -/
      apply congrArg G.map
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i s) (CategoryTheory.CategoryStruct.c …
      -/
      exact equalizer.condition s t
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fan.mk (G.obj (Category …
      -/
    · apply isLimitOfHasProductOfPreservesLimit
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fan.mk (G.obj Q) fun f  …
      -/
    · apply isLimitOfHasProductOfPreservesLimit
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (G.map i) ⋯)
      -/
    · apply isLimitForkMapOfIsLimit
      /-
        case l
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι i ?w)
      -/
      apply equalizerIsEqualizer
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ CategoryTheory.Iso (CategoryTheory.Limits.HasLimitOfHasProductsOfHasEqualize …
      -/
    · refine Cones.ext (Iso.refl _) ?_
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigm …
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
        inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.piObj K.obj
        Q : C := CategoryTheory.Limits.piObj fun f => K.obj f.fst.2
        s : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Ca …
        t : Quiver.Hom P Q := CategoryTheory.Limits.Pi.lift fun f => CategoryTheory.Li …
        I : C := CategoryTheory.Limits.equalizer s t
        i : Quiver.Hom I P := CategoryTheory.Limits.equalizer.ι s t
        ⊢ ∀ (j : J), Eq ((CategoryTheory.Limits.HasLimitOfHasProductsOfHasEqualizers.b …
      -/
      intro j; dsimp [P, Q, I, i]; simp
                                   /-
                                     🎉 no goals
                                   -/
-- See note [dsimp, simp].


/-- If G preserves equalizers and finite products, it preserves finite limits. -/
lemma preservesFiniteLimits_of_preservesEqualizers_and_finiteProducts [HasEqualizers C]
    [HasFiniteProducts C] (G : C ⥤ D) [PreservesLimitsOfShape WalkingParallelPair G]
    [PreservesFiniteProducts G] : PreservesFiniteLimits G where
  preservesFiniteLimits := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      ⊢ ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryTheor …
    -/
    intro J sJ fJ
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J G
    -/
    haveI : Fintype J := inferInstance
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this : Fintype J
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J G
    -/
    haveI : Fintype ((p : J × J) × (p.fst ⟶ p.snd)) := inferInstance
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this✝ : Fintype J
      this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J G
    -/
    apply @preservesLimit_of_preservesEqualizers_and_product _ _ _ sJ _ _ ?_ ?_ _ G _ ?_ ?_
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete J) C
      -/
    · apply hasLimitsOfShape_discrete _ _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete (Sigma fun p …
      -/
    · apply hasLimitsOfShape_discrete _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete J) G
      -/
    · apply PreservesFiniteProducts.preserves _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasEqualizers C
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
        inst✝ : CategoryTheory.Limits.PreservesFiniteProducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete (Sigma …
      -/
    · apply PreservesFiniteProducts.preserves _
      /-
        🎉 no goals
      -/


/-- If G preserves equalizers and products, it preserves all limits. -/
lemma preservesLimits_of_preservesEqualizers_and_products [HasEqualizers C]
    [HasProducts.{w} C] (G : C ⥤ D) [PreservesLimitsOfShape WalkingParallelPair G]
    [∀ J, PreservesLimitsOfShape (Discrete.{w} J) G] : PreservesLimitsOfSize.{w, w} G where
  preservesLimitsOfShape := preservesLimit_of_preservesEqualizers_and_product G


theorem hasFiniteLimits_of_hasTerminal_and_pullbacks [HasTerminal C] [HasPullbacks C] :
    HasFiniteLimits C :=
  @hasFiniteLimits_of_hasEqualizers_and_finite_products C _
    (@hasFiniteProducts_of_has_binary_and_terminal C _
      (hasBinaryProducts_of_hasTerminal_and_pullbacks C) inferInstance)
    (@hasEqualizers_of_hasPullbacks_and_binary_products C _
      (hasBinaryProducts_of_hasTerminal_and_pullbacks C) inferInstance)


/-- If G preserves terminal objects and pullbacks, it preserves all finite limits. -/
lemma preservesFiniteLimits_of_preservesTerminal_and_pullbacks [HasTerminal C]
    [HasPullbacks C] (G : C ⥤ D) [PreservesLimitsOfShape (Discrete.{0} PEmpty) G]
    [PreservesLimitsOfShape WalkingCospan G] : PreservesFiniteLimits G := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasTerminal C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits G
  -/
  haveI : HasFiniteLimits C := hasFiniteLimits_of_hasTerminal_and_pullbacks
  haveI : PreservesLimitsOfShape (Discrete WalkingPair) G :=
    preservesBinaryProducts_of_preservesTerminal_and_pullbacks G
  haveI : PreservesLimitsOfShape WalkingParallelPair G :=
      preservesEqualizers_of_preservesPullbacks_and_binaryProducts G
  apply
    @preservesFiniteLimits_of_preservesEqualizers_and_finiteProducts _ _ _ _ _ _ G _ ?_
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasTerminal C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    this✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    this : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Walk …
    ⊢ CategoryTheory.Limits.PreservesFiniteProducts G
  -/
  apply PreservesFiniteProducts.mk
  /-
    case preserves
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasTerminal C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    this✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    this : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Walk …
    ⊢ ∀ (J : Type) [inst : Fintype J], CategoryTheory.Limits.PreservesLimitsOfShap …
  -/
  apply preservesFiniteProducts_of_preserves_binary_and_terminal G
  /-
    🎉 no goals
  -/


/-- (Implementation) Given the appropriate coproduct and coequalizer cocones,
build the cocone for `F` which is colimiting if the given cocones are also.
-/
@[simps]
def buildColimit
    (hs : ∀ f : Σp : J × J, p.1 ⟶ p.2, c₁.ι.app ⟨f⟩ ≫ s = F.map f.2 ≫ c₂.ι.app ⟨f.1.2⟩)
    (ht : ∀ f : Σp : J × J, p.1 ⟶ p.2, c₁.ι.app ⟨f⟩ ≫ t = c₂.ι.app ⟨f.1.1⟩)
    (i : Cofork s t) : Cocone F where
  pt := i.pt
  ι :=
    { app := fun _ => c₂.ι.app ⟨_⟩ ≫ i.π
      naturality := fun j₁ j₂ f => by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝¹ : CategoryTheory.SmallCategory J
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor J C
          c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
          c₂ : CategoryTheory.Limits.Cofan F.obj
          s t : Quiver.Hom c₁.pt c₂.pt
          hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          i : CategoryTheory.Limits.Cofork s t
          j₁ j₂ : J
          f : Quiver.Hom j₁ j₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun x => CategoryTheory.C …
        -/
        dsimp
        have reassoced (f : (p : J × J) × (p.fst ⟶ p.snd)) {W : C} {h : _ ⟶ W} :
          c₁.ι.app ⟨f⟩ ≫ s ≫ h = F.map f.snd ≫ c₂.ι.app ⟨f.fst.snd⟩ ≫ h := by
            simp only [← Category.assoc, eq_whisker (hs f)]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          J : Type w
          inst✝¹ : CategoryTheory.SmallCategory J
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor J C
          c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
          c₂ : CategoryTheory.Limits.Cofan F.obj
          s t : Quiver.Hom c₁.pt c₂.pt
          hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
          i : CategoryTheory.Limits.Cofork s t
          j₁ j₂ : J
          f : Quiver.Hom j₁ j₂
          reassoced : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2) {W : C} {h : Quiver.Hom  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.CategoryStr …
        -/
        rw [Category.comp_id, ← reassoced ⟨⟨_, _⟩, f⟩, i.condition, ← Category.assoc, ht] }
        /-
          🎉 no goals
        -/


/-- (Implementation) Show the cocone constructed in `buildColimit` is colimiting,
provided the cocones used in its construction are.
-/
def buildIsColimit (t₁ : IsColimit c₁) (t₂ : IsColimit c₂) (hi : IsColimit i) :
    IsColimit (buildColimit s t hs ht i) where
  desc q := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝¹ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor J C
      c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
      c₂ : CategoryTheory.Limits.Cofan F.obj
      s t : Quiver.Hom c₁.pt c₂.pt
      hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
      ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
      i : CategoryTheory.Limits.Cofork s t
      t₁ : CategoryTheory.Limits.IsColimit c₁
      t₂ : CategoryTheory.Limits.IsColimit c₂
      hi : CategoryTheory.Limits.IsColimit i
      q : CategoryTheory.Limits.Cocone F
      ⊢ Quiver.Hom (CategoryTheory.Limits.HasColimitOfHasCoproductsOfHasCoequalizers …
    -/
    refine hi.desc (Cofork.ofπ ?_ ?_)
      /-
        case refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝¹ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
        c₂ : CategoryTheory.Limits.Cofan F.obj
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Cofork s t
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        hi : CategoryTheory.Limits.IsColimit i
        q : CategoryTheory.Limits.Cocone F
        ⊢ Quiver.Hom c₂.pt q.1
      -/
    · refine t₂.desc (Cofan.mk _ fun j => ?_)
      /-
        case refine_1
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝¹ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
        c₂ : CategoryTheory.Limits.Cofan F.obj
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Cofork s t
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        hi : CategoryTheory.Limits.IsColimit i
        q : CategoryTheory.Limits.Cocone F
        j : J
        ⊢ Quiver.Hom (F.obj j) q.1
      -/
      apply q.ι.app j
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝¹ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
        c₂ : CategoryTheory.Limits.Cofan F.obj
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Cofork s t
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        hi : CategoryTheory.Limits.IsColimit i
        q : CategoryTheory.Limits.Cocone F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s (t₂.desc (CategoryTheory.Limits.Cof …
      -/
    · apply t₁.hom_ext
      /-
        case refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝¹ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
        c₂ : CategoryTheory.Limits.Cofan F.obj
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Cofork s t
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        hi : CategoryTheory.Limits.IsColimit i
        q : CategoryTheory.Limits.Cocone F
        ⊢ ∀ (j : CategoryTheory.Discrete (Sigma fun p => Quiver.Hom p.1 p.2)), Eq (Cat …
      -/
      intro j
      /-
        case refine_2
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝¹ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
        c₂ : CategoryTheory.Limits.Cofan F.obj
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Cofork s t
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        hi : CategoryTheory.Limits.IsColimit i
        q : CategoryTheory.Limits.Cocone F
        j : CategoryTheory.Discrete (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.ι.app j) (CategoryTheory.Category …
      -/
      cases' j with j
      have reassoced_s (f : (p : J × J) × (p.fst ⟶ p.snd)) {W : C} (h : _ ⟶ W) :
        c₁.ι.app ⟨f⟩ ≫ s ≫ h = F.map f.snd ≫ c₂.ι.app ⟨f.fst.snd⟩ ≫ h := by
          simp only [← Category.assoc]
          apply eq_whisker (hs f)
      have reassoced_t (f : (p : J × J) × (p.fst ⟶ p.snd)) {W : C} (h : _ ⟶ W) :
        c₁.ι.app ⟨f⟩ ≫ t ≫ h = c₂.ι.app ⟨f.fst.fst⟩ ≫ h := by
          simp only [← Category.assoc]
          apply eq_whisker (ht f)
      /-
        case refine_2.mk
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝¹ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor J C
        c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
        c₂ : CategoryTheory.Limits.Cofan F.obj
        s t : Quiver.Hom c₁.pt c₂.pt
        hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
        i : CategoryTheory.Limits.Cofork s t
        t₁ : CategoryTheory.Limits.IsColimit c₁
        t₂ : CategoryTheory.Limits.IsColimit c₂
        hi : CategoryTheory.Limits.IsColimit i
        q : CategoryTheory.Limits.Cocone F
        j : Sigma fun p => Quiver.Hom p.1 p.2
        reassoced_s : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2) {W : C} (h : Quiver.Ho …
        reassoced_t : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2) {W : C} (h : Quiver.Ho …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.ι.app { as := j }) (CategoryTheor …
      -/
      simp [reassoced_s, reassoced_t]
      /-
        🎉 no goals
      -/
  uniq q m w :=
    hi.hom_ext
      (i.coequalizer_ext
        (t₂.hom_ext fun j => by
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            J : Type w
            inst✝¹ : CategoryTheory.SmallCategory J
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor J C
            c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
            c₂ : CategoryTheory.Limits.Cofan F.obj
            s t : Quiver.Hom c₁.pt c₂.pt
            hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            i : CategoryTheory.Limits.Cofork s t
            t₁ : CategoryTheory.Limits.IsColimit c₁
            t₂ : CategoryTheory.Limits.IsColimit c₂
            hi : CategoryTheory.Limits.IsColimit i
            q : CategoryTheory.Limits.Cocone F
            m : Quiver.Hom (CategoryTheory.Limits.HasColimitOfHasCoproductsOfHasCoequalize …
            w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
            j : CategoryTheory.Discrete J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₂.ι.app j) (CategoryTheory.Category …
          -/
          cases' j with j
                /-
                  C : Type u
                  inst✝² : CategoryTheory.Category.{v, u} C
                  J : Type w
                  inst✝¹ : CategoryTheory.SmallCategory J
                  D : Type u₂
                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                  F : CategoryTheory.Functor J C
                  c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
                  c₂ : CategoryTheory.Limits.Cofan F.obj
                  s✝ t : Quiver.Hom c₁.pt c₂.pt
                  hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
                  ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
                  i : CategoryTheory.Limits.Cofork s✝ t
                  t₁ : CategoryTheory.Limits.IsColimit c₁
                  t₂ : CategoryTheory.Limits.IsColimit c₂
                  hi : CategoryTheory.Limits.IsColimit i
                  s : CategoryTheory.Limits.Cocone F
                  j : J
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.HasColimitOfH …
                -/
          /-
            case mk
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            J : Type w
            inst✝¹ : CategoryTheory.SmallCategory J
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor J C
            c₁ : CategoryTheory.Limits.Cofan fun f => F.obj f.fst.1
            c₂ : CategoryTheory.Limits.Cofan F.obj
            s t : Quiver.Hom c₁.pt c₂.pt
            hs : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            ht : ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStr …
            i : CategoryTheory.Limits.Cofork s t
            t₁ : CategoryTheory.Limits.IsColimit c₁
            t₂ : CategoryTheory.Limits.IsColimit c₂
            hi : CategoryTheory.Limits.IsColimit i
            q : CategoryTheory.Limits.Cocone F
            m : Quiver.Hom (CategoryTheory.Limits.HasColimitOfHasCoproductsOfHasCoequalize …
            w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₂.ι.app { as := j }) (CategoryTheor …
          -/
                /-
                  🎉 no goals
                -/
          simpa using w j))
          /-
            🎉 no goals
          -/
  fac s j := by simp


/-- Given the existence of the appropriate (possibly finite) coproducts and coequalizers,
we can construct a colimit cocone for `F`.
(This assumes the existence of all coequalizers, which is technically stronger than needed.)
-/
noncomputable def colimitCoconeOfCoequalizerAndCoproduct (F : J ⥤ C)
    [HasColimit (Discrete.functor F.obj)]
    [HasColimit (Discrete.functor fun f : Σp : J × J, p.1 ⟶ p.2 => F.obj f.1.1)]
    [HasCoequalizers C] : ColimitCocone F where
  cocone := _
  isColimit :=
    buildIsColimit (Sigma.desc fun f => F.map f.2 ≫ colimit.ι (Discrete.functor F.obj) ⟨f.1.2⟩)
                                                                           /-
                                                                             C : Type u
                                                                             inst✝⁵ : CategoryTheory.Category.{v, u} C
                                                                             J : Type w
                                                                             inst✝⁴ : CategoryTheory.SmallCategory J
                                                                             D : Type u₂
                                                                             inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                                             F : CategoryTheory.Functor J C
                                                                             inst✝² : CategoryTheory.Limits.HasColimit (CategoryTheory.Discrete.functor F.o …
                                                                             inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Discrete.functor fun …
                                                                             inst✝ : CategoryTheory.Limits.HasCoequalizers C
                                                                             ⊢ ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStruct …
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
      (Sigma.desc fun f => colimit.ι (Discrete.functor F.obj) ⟨f.1.1⟩) (by simp) (by simp)
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
      (colimit.isColimit _) (colimit.isColimit _) (colimit.isColimit _)


/-- Given the existence of the appropriate (possibly finite) coproducts and coequalizers,
we know a colimit of `F` exists.
(This assumes the existence of all coequalizers, which is technically stronger than needed.)
-/
theorem hasColimit_of_coequalizer_and_coproduct (F : J ⥤ C) [HasColimit (Discrete.functor F.obj)]
    [HasColimit (Discrete.functor fun f : Σp : J × J, p.1 ⟶ p.2 => F.obj f.1.1)]
    [HasCoequalizers C] : HasColimit F :=
  HasColimit.mk (colimitCoconeOfCoequalizerAndCoproduct F)


/-- A colimit can be realised as a quotient of a coproduct. -/
noncomputable def colimitQuotientCoproduct [HasColimitsOfSize.{w, w} C] (F : J ⥤ C) :
    ∐ (fun j => F.obj j) ⟶ colimit F :=
  have := hasFiniteColimits_of_hasColimitsOfSize C
  coequalizer.π _ _ ≫ (colimit.isoColimitCocone (colimitCoconeOfCoequalizerAndCoproduct F)).inv


instance colimitQuotientCoproduct_epi [HasColimitsOfSize.{w, w} C] (F : J ⥤ C) :
    Epi (colimitQuotientCoproduct F) :=
  epi_comp _ _


/-- Any category with coproducts and coequalizers has all colimits.

See <https://stacks.math.columbia.edu/tag/002P>.
-/
theorem has_colimits_of_hasCoequalizers_and_coproducts [HasCoproducts.{w} C] [HasCoequalizers C] :
    HasColimitsOfSize.{w, w} C where
  has_colimits_of_shape := fun _ _ =>
      { has_colimit := fun F => hasColimit_of_coequalizer_and_coproduct F }


/-- Any category with finite coproducts and coequalizers has all finite colimits.

See <https://stacks.math.columbia.edu/tag/002Q>.
-/
theorem hasFiniteColimits_of_hasCoequalizers_and_finite_coproducts [HasFiniteCoproducts C]
    [HasCoequalizers C] : HasFiniteColimits C where
  out _ := { has_colimit := fun F => hasColimit_of_coequalizer_and_coproduct F }

-- Porting note: removed and added individually
-- noncomputable section

/-- If a functor preserves coequalizers and the appropriate coproducts, it preserves colimits. -/
lemma preservesColimit_of_preservesCoequalizers_and_coproduct :
    PreservesColimitsOfShape J G where
  preservesColimit {K} := by
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      ⊢ CategoryTheory.Limits.PreservesColimit K G
    -/
    let P := ∐ K.obj
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      ⊢ CategoryTheory.Limits.PreservesColimit K G
    -/
    let Q := ∐ fun f : Σp : J × J, p.fst ⟶ p.snd => K.obj f.1.1
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      ⊢ CategoryTheory.Limits.PreservesColimit K G
    -/
    let s : Q ⟶ P := Sigma.desc fun f => K.map f.2 ≫ colimit.ι (Discrete.functor K.obj) ⟨_⟩
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      ⊢ CategoryTheory.Limits.PreservesColimit K G
    -/
    let t : Q ⟶ P := Sigma.desc fun f => colimit.ι (Discrete.functor K.obj) ⟨f.1.1⟩
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      ⊢ CategoryTheory.Limits.PreservesColimit K G
    -/
    let I := coequalizer s t
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      I : C := CategoryTheory.Limits.coequalizer s t
      ⊢ CategoryTheory.Limits.PreservesColimit K G
    -/
    let i : P ⟶ I := coequalizer.π s t
    apply preservesColimit_of_preserves_colimit_cocone
        (buildIsColimit s t (by simp [P, s]) (by simp [P, t]) (colimit.isColimit _)
          (colimit.isColimit _) (colimit.isColimit _))
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      I : C := CategoryTheory.Limits.coequalizer s t
      i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.HasColim …
    -/
    apply IsColimit.ofIsoColimit (buildIsColimit _ _ _ _ _ _ _) _
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.Cofan fun f => (K.comp G).obj f.fst.1
      -/
    · refine Cofan.mk (G.obj Q) fun j => G.map ?_
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        j : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Quiver.Hom (K.obj j.fst.1) Q
      -/
      apply Sigma.ι _ j
      /-
        🎉 no goals
      -/
    -- fun j => G.map (Sigma.ι _ j)
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.Cofan (K.comp G).obj
      -/
    · exact Cofan.mk _ fun f => G.map (Sigma.ι _ f)
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ Quiver.Hom (CategoryTheory.Limits.Cofan.mk (G.obj Q) fun j => G.map (Categor …
      -/
    · apply G.map s
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ Quiver.Hom (CategoryTheory.Limits.Cofan.mk (G.obj Q) fun j => G.map (Categor …
      -/
    · apply G.map t
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStruct …
      -/
    · intro f
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofan.mk (G.o …
      -/
      dsimp [P, Q, s, Cofan.mk]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.Sigma.ι …
      -/
      simp only [← G.map_comp, colimit.ι_desc]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (G.map ((CategoryTheory.Limits.Cofan.mk (CategoryTheory.Limits.sigmaObj K …
      -/
      congr
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ ∀ (f : Sigma fun p => Quiver.Hom p.1 p.2), Eq (CategoryTheory.CategoryStruct …
      -/
    · intro f
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofan.mk (G.o …
      -/
      dsimp [P, Q, t, Cofan.mk]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.Sigma.ι …
      -/
      simp only [← G.map_comp, colimit.ι_desc]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        f : Sigma fun p => Quiver.Hom p.1 p.2
        ⊢ Eq (G.map ((CategoryTheory.Limits.Cofan.mk (CategoryTheory.Limits.sigmaObj K …
      -/
      dsimp
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.Cofork (G.map s) (G.map t)
      -/
    · refine Cofork.ofπ (G.map i) ?_
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map s) (G.map i)) (CategoryTheory. …
      -/
      rw [← G.map_comp, ← G.map_comp]
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp s i)) (G.map (CategoryTheory.C …
      -/
      apply congrArg G.map
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s i) (CategoryTheory.CategoryStruct.c …
      -/
      apply coequalizer.condition
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (G.obj Q) fu …
      -/
    · apply isColimitOfHasCoproductOfPreservesColimit
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk (G.obj (Cate …
      -/
    · apply isColimitOfHasCoproductOfPreservesColimit
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ (G.map i) ⋯)
      -/
    · apply isColimitCoforkMapOfIsColimit
      /-
        case l
        C : Type u
        inst✝⁸ : CategoryTheory.Category.{v, u} C
        J : Type w
        inst✝⁷ : CategoryTheory.SmallCategory J
        D : Type u₂
        inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
        inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
        inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        G : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
        inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
        K : CategoryTheory.Functor J C
        P : C := CategoryTheory.Limits.sigmaObj K.obj
        Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
        s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
        I : C := CategoryTheory.Limits.coequalizer s t
        i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
        ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ i ?w)
      -/
      apply coequalizerIsCoequalizer
      /-
        🎉 no goals
      -/
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      I : C := CategoryTheory.Limits.coequalizer s t
      i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.HasColimitOfHasCoproductsOfHasCoeq …
    -/
    refine Cocones.ext (Iso.refl _) ?_
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      I : C := CategoryTheory.Limits.coequalizer s t
      i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Ha …
    -/
    intro j
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      I : C := CategoryTheory.Limits.coequalizer s t
      i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.HasColimitOfH …
    -/
    dsimp [P, Q, I, i]
    /-
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝⁷ : CategoryTheory.SmallCategory J
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
      inst✝⁵ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      inst✝⁴ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Si …
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      G : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
      K : CategoryTheory.Functor J C
      P : C := CategoryTheory.Limits.sigmaObj K.obj
      Q : C := CategoryTheory.Limits.sigmaObj fun f => K.obj f.fst.1
      s : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      t : Quiver.Hom Q P := CategoryTheory.Limits.Sigma.desc fun f => CategoryTheory …
      I : C := CategoryTheory.Limits.coequalizer s t
      i : Quiver.Hom P I := CategoryTheory.Limits.coequalizer.π s t
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/
-- See note [dsimp, simp].


/-- If G preserves coequalizers and finite coproducts, it preserves finite colimits. -/
lemma preservesFiniteColimits_of_preservesCoequalizers_and_finiteCoproducts
    [HasCoequalizers C] [HasFiniteCoproducts C] (G : C ⥤ D)
    [PreservesColimitsOfShape WalkingParallelPair G]
    [PreservesFiniteCoproducts G] : PreservesFiniteColimits G where
  preservesFiniteColimits := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
      ⊢ ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryTheor …
    -/
    intro J sJ fJ
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J G
    -/
    haveI : Fintype J := inferInstance
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this : Fintype J
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J G
    -/
    haveI : Fintype ((p : J × J) × (p.fst ⟶ p.snd)) := inferInstance
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasCoequalizers C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
      inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this✝ : Fintype J
      this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J G
    -/
    apply @preservesColimit_of_preservesCoequalizers_and_coproduct _ _ _ sJ _ _ ?_ ?_ _ G _ ?_ ?_
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete J) C
      -/
    · apply hasColimitsOfShape_discrete _ _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Sigma fun …
      -/
    · apply hasColimitsOfShape_discrete _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete J) G
      -/
    · apply PreservesFiniteCoproducts.preserves _
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝³ : CategoryTheory.Limits.HasCoequalizers C
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
        inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts G
        J : Type
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        this✝ : Fintype J
        this : Fintype (Sigma fun p => Quiver.Hom p.1 p.2)
        ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete (Sig …
      -/
    · apply PreservesFiniteCoproducts.preserves _
      /-
        🎉 no goals
      -/


/-- If G preserves coequalizers and coproducts, it preserves all colimits. -/
lemma preservesColimits_of_preservesCoequalizers_and_coproducts [HasCoequalizers C]
    [HasCoproducts.{w} C] (G : C ⥤ D) [PreservesColimitsOfShape WalkingParallelPair G]
    [∀ J, PreservesColimitsOfShape (Discrete.{w} J) G] : PreservesColimitsOfSize.{w, w} G where
  preservesColimitsOfShape := preservesColimit_of_preservesCoequalizers_and_coproduct G


theorem hasFiniteColimits_of_hasInitial_and_pushouts [HasInitial C] [HasPushouts C] :
    HasFiniteColimits C :=
  @hasFiniteColimits_of_hasCoequalizers_and_finite_coproducts C _
    (@hasFiniteCoproducts_of_has_binary_and_initial C _
      (hasBinaryCoproducts_of_hasInitial_and_pushouts C) inferInstance)
    (@hasCoequalizers_of_hasPushouts_and_binary_coproducts C _
      (hasBinaryCoproducts_of_hasInitial_and_pushouts C) inferInstance)


/-- If G preserves initial objects and pushouts, it preserves all finite colimits. -/
lemma preservesFiniteColimits_of_preservesInitial_and_pushouts [HasInitial C]
    [HasPushouts C] (G : C ⥤ D) [PreservesColimitsOfShape (Discrete.{0} PEmpty) G]
    [PreservesColimitsOfShape WalkingSpan G] : PreservesFiniteColimits G := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasInitial C
    inst✝² : CategoryTheory.Limits.HasPushouts C
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits G
  -/
  haveI : HasFiniteColimits C := hasFiniteColimits_of_hasInitial_and_pushouts
  haveI : PreservesColimitsOfShape (Discrete WalkingPair) G :=
    preservesBinaryCoproducts_of_preservesInitial_and_pushouts G
  haveI : PreservesColimitsOfShape (WalkingParallelPair) G :=
      (preservesCoequalizers_of_preservesPushouts_and_binaryCoproducts G)
  refine
    @preservesFiniteColimits_of_preservesCoequalizers_and_finiteCoproducts _ _ _ _ _ _ G _ ?_
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasInitial C
    inst✝² : CategoryTheory.Limits.HasPushouts C
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
    this✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    this✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
    this : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.Wa …
    ⊢ CategoryTheory.Limits.PreservesFiniteCoproducts G
  -/
  apply PreservesFiniteCoproducts.mk
  /-
    case preserves
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasInitial C
    inst✝² : CategoryTheory.Limits.HasPushouts C
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
    this✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    this✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
    this : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.Wa …
    ⊢ ∀ (J : Type) [inst : Fintype J], CategoryTheory.Limits.PreservesColimitsOfSh …
  -/
  apply preservesFiniteCoproductsOfPreservesBinaryAndInitial G
  /-
    🎉 no goals
  -/


