/-- If the product `X ⨯ Y` and the equalizer of `π₁ ≫ f` and `π₂ ≫ g` exist, then the
    pullback of `f` and `g` exists: It is given by composing the equalizer with the projections. -/
theorem hasLimit_cospan_of_hasLimit_pair_of_hasLimit_parallelPair {C : Type u} [𝒞 : Category.{v} C]
    {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasLimit (pair X Y)]
    [HasLimit (parallelPair (prod.fst ≫ f) (prod.snd ≫ g))] : HasLimit (cospan f g) :=
  let π₁ : X ⨯ Y ⟶ X := prod.fst
  let π₂ : X ⨯ Y ⟶ Y := prod.snd
  let e := equalizer.ι (π₁ ≫ f) (π₂ ≫ g)
  HasLimit.mk
    { cone :=
        PullbackCone.mk (e ≫ π₁) (e ≫ π₂) <| by
          /-
            C : Type u
            𝒞 : CategoryTheory.Category.{v, u} C
            X Y Z : C
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
            inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
            π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
            π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
            e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
          -/
          rw [Category.assoc, equalizer.condition]
          /-
            C : Type u
            𝒞 : CategoryTheory.Category.{v, u} C
            X Y Z : C
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
            inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
            π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
            π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
            e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι (C …
          -/
          simp [e]
          /-
            🎉 no goals
          -/
      isLimit :=
        PullbackCone.IsLimit.mk _ (fun s => equalizer.lift
          (prod.lift (s.π.app WalkingCospan.left) (s.π.app WalkingCospan.right)) <| by
            /-
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
              inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
              π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
              π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
              e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
              s : CategoryTheory.Limits.PullbackCone f g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (s.π …
            -/
            rw [← Category.assoc, limit.lift_π, ← Category.assoc, limit.lift_π]
            /-
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
              inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
              π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
              π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
              e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
              s : CategoryTheory.Limits.PullbackCone f g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.BinaryFan.mk  …
            -/
            exact PullbackCone.condition _)
            /-
              🎉 no goals
            -/
              /-
                C : Type u
                𝒞 : CategoryTheory.Category.{v, u} C
                X Y Z : C
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
                inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
                π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
                π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
                e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
                ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory.CategoryS …
              -/
              /-
                🎉 no goals
              -/
          (by simp [π₁, e]) (by simp [π₂, e]) fun s m h₁ h₂ => by
                                /-
                                  🎉 no goals
                                -/
          /-
            C : Type u
            𝒞 : CategoryTheory.Category.{v, u} C
            X Y Z : C
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
            inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
            π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
            π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
            e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
            s : CategoryTheory.Limits.PullbackCone f g
            m : Quiver.Hom s.pt (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryS …
            h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
            h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
            ⊢ Eq m ((fun s => CategoryTheory.Limits.equalizer.lift (CategoryTheory.Limits. …
          -/
          ext
            /-
              case h.h₁
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
              inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
              π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
              π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
              e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
              s : CategoryTheory.Limits.PullbackCone f g
              m : Quiver.Hom s.pt (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryS …
              h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
              h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
            -/
          · dsimp; simpa using h₁
                   /-
                     🎉 no goals
                   -/
            /-
              case h.h₂
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair X Y)
              inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair (Ca …
              π₁ : Quiver.Hom (CategoryTheory.Limits.prod X Y) X := CategoryTheory.Limits.pr …
              π₂ : Quiver.Hom (CategoryTheory.Limits.prod X Y) Y := CategoryTheory.Limits.pr …
              e : Quiver.Hom (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryStruct …
              s : CategoryTheory.Limits.PullbackCone f g
              m : Quiver.Hom s.pt (CategoryTheory.Limits.equalizer (CategoryTheory.CategoryS …
              h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
              h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
            -/
          · simpa using h₂ }
            /-
              🎉 no goals
            -/


/-- If a category has all binary products and all equalizers, then it also has all pullbacks.
    As usual, this is not an instance, since there may be a more direct way to construct
    pullbacks. -/
theorem hasPullbacks_of_hasBinaryProducts_of_hasEqualizers (C : Type u) [Category.{v} C]
    [HasBinaryProducts C] [HasEqualizers C] : HasPullbacks C :=
  { has_limit := fun F => hasLimitOfIso (diagramIsoCospan F).symm }


/-- If the coproduct `Y ⨿ Z` and the coequalizer of `f ≫ ι₁` and `g ≫ ι₂` exist, then the
    pushout of `f` and `g` exists: It is given by composing the inclusions with the coequalizer. -/
theorem hasColimit_span_of_hasColimit_pair_of_hasColimit_parallelPair {C : Type u}
    [𝒞 : Category.{v} C] {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasColimit (pair Y Z)]
    [HasColimit (parallelPair (f ≫ coprod.inl) (g ≫ coprod.inr))] : HasColimit (span f g) :=
  let ι₁ : Y ⟶ Y ⨿ Z := coprod.inl
  let ι₂ : Z ⟶ Y ⨿ Z := coprod.inr
  let c := coequalizer.π (f ≫ ι₁) (g ≫ ι₂)
  HasColimit.mk
    { cocone :=
        PushoutCocone.mk (ι₁ ≫ c) (ι₂ ≫ c) <| by
          /-
            C : Type u
            𝒞 : CategoryTheory.Category.{v, u} C
            X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
            inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
            ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
            ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
            c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
          -/
          rw [← Category.assoc, ← Category.assoc, coequalizer.condition]
          /-
            🎉 no goals
          -/
      isColimit :=
        PushoutCocone.IsColimit.mk _
          (fun s => coequalizer.desc
              (coprod.desc (s.ι.app WalkingSpan.left) (s.ι.app WalkingSpan.right)) <| by
            /-
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom X Z
              inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
              inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
              ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
              s : CategoryTheory.Limits.PushoutCocone f g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
            -/
            rw [Category.assoc, colimit.ι_desc, Category.assoc, colimit.ι_desc]
            /-
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom X Z
              inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
              inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
              ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
              s : CategoryTheory.Limits.PushoutCocone f g
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limits.BinaryCofan …
            -/
            exact PushoutCocone.condition _)
            /-
              🎉 no goals
            -/
              /-
                C : Type u
                𝒞 : CategoryTheory.Category.{v, u} C
                X Y Z : C
                f : Quiver.Hom X Y
                g : Quiver.Hom X Z
                inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
                inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
                ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
                ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
                c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
                ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone f g), Eq (CategoryTheory.Category …
              -/
              /-
                🎉 no goals
              -/
          (by simp [ι₁, c]) (by simp [ι₂, c]) fun s m h₁ h₂ => by
                                /-
                                  🎉 no goals
                                -/
          /-
            C : Type u
            𝒞 : CategoryTheory.Category.{v, u} C
            X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom X Z
            inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
            inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
            ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
            ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
            c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
            s : CategoryTheory.Limits.PushoutCocone f g
            m : Quiver.Hom (CategoryTheory.Limits.coequalizer (CategoryTheory.CategoryStru …
            h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
            ⊢ Eq m ((fun s => CategoryTheory.Limits.coequalizer.desc (CategoryTheory.Limit …
          -/
          ext
            /-
              case h.h₁
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom X Z
              inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
              inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
              ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
              s : CategoryTheory.Limits.PushoutCocone f g
              m : Quiver.Hom (CategoryTheory.Limits.coequalizer (CategoryTheory.CategoryStru …
              h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
              h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
            -/
          · simpa using h₁
            /-
              🎉 no goals
            -/
            /-
              case h.h₂
              C : Type u
              𝒞 : CategoryTheory.Category.{v, u} C
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom X Z
              inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pair Y Z)
              inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair ( …
              ι₁ : Quiver.Hom Y (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              ι₂ : Quiver.Hom Z (CategoryTheory.Limits.coprod Y Z) := CategoryTheory.Limits. …
              c : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) (CategoryTheory.Limits.coequ …
              s : CategoryTheory.Limits.PushoutCocone f g
              m : Quiver.Hom (CategoryTheory.Limits.coequalizer (CategoryTheory.CategoryStru …
              h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
              h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
            -/
          · simpa using h₂ }
            /-
              🎉 no goals
            -/


/-- If a category has all binary coproducts and all coequalizers, then it also has all pushouts.
    As usual, this is not an instance, since there may be a more direct way to construct
    pushouts. -/
theorem hasPushouts_of_hasBinaryCoproducts_of_hasCoequalizers (C : Type u) [Category.{v} C]
    [HasBinaryCoproducts C] [HasCoequalizers C] : HasPushouts C :=
  hasPushouts_of_hasColimit_span C


