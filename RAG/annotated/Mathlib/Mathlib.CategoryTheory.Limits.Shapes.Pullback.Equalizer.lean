/-- The equalizer of `f g : X ⟶ Y` is the pullback of the diagonal map `Y ⟶ Y × Y`
along the map `(f, g) : X ⟶ Y × Y`. -/
lemma isPullback_equalizer_prod [HasEqualizer f g] [HasBinaryProduct Y Y] :
    IsPullback (equalizer.ι f g) (equalizer.ι f g ≫ f) (prod.lift f g) (prod.lift (𝟙 _) (𝟙 _)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
    inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
    ⊢ CategoryTheory.IsPullback (CategoryTheory.Limits.equalizer.ι f g) (CategoryT …
  -/
  refine ⟨⟨by ext <;> simp [equalizer.condition f g]⟩, ⟨PullbackCone.IsLimit.mk _ ?_ ?_ ?_ ?_⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      ⊢ (s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f g …
    -/
  · refine fun s ↦ equalizer.lift s.fst ?_
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f g) ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s.fst f) (CategoryTheory.CategoryStru …
    -/
    have H₁ : s.fst ≫ f = s.snd := by simpa using congr($s.condition ≫ prod.fst)
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f g) ( …
      H₁ : Eq (CategoryTheory.CategoryStruct.comp s.fst f) s.snd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s.fst f) (CategoryTheory.CategoryStru …
    -/
    have H₂ : s.fst ≫ g = s.snd := by simpa using congr($s.condition ≫ prod.snd)
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f g) ( …
      H₁ : Eq (CategoryTheory.CategoryStruct.comp s.fst f) s.snd
      H₂ : Eq (CategoryTheory.CategoryStruct.comp s.fst g) s.snd
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s.fst f) (CategoryTheory.CategoryStru …
    -/
    exact H₁.trans H₂.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f …
    -/
  · exact fun s ↦ by simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f …
    -/
  · exact fun s ↦ by simpa using congr($s.condition ≫ prod.fst)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasEqualizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
      ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone (CategoryTheory.Limits.prod.lift f …
    -/
  · exact fun s m hm _ ↦ by ext; simp [*]
    /-
      🎉 no goals
    -/


/-- The coequalizer of `f g : X ⟶ Y` is the pushout of the diagonal map `X ⨿ X ⟶ X`
along the map `(f, g) : X ⨿ X ⟶ Y`. -/
lemma isPushout_coequalizer_coprod [HasCoequalizer f g] [HasBinaryCoproduct X X] :
    IsPushout (coprod.desc f g) (coprod.desc (𝟙 _) (𝟙 _))
      (coequalizer.π f g) (f ≫ coequalizer.π f g) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
    ⊢ CategoryTheory.IsPushout (CategoryTheory.Limits.coprod.desc f g) (CategoryTh …
  -/
  refine ⟨⟨by ext <;> simp [coequalizer.condition f g]⟩, ⟨PushoutCocone.IsColimit.mk _ ?_ ?_ ?_ ?_⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      ⊢ (s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.desc  …
    -/
  · refine fun s ↦ coequalizer.desc s.inl ?_
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.desc f g …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.inl) (CategoryTheory.CategoryStru …
    -/
    have H₁ : f ≫ s.inl = s.inr := by simpa using congr(coprod.inl ≫ $s.condition)
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.desc f g …
      H₁ : Eq (CategoryTheory.CategoryStruct.comp f s.inl) s.inr
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.inl) (CategoryTheory.CategoryStru …
    -/
    have H₂ : g ≫ s.inl = s.inr := by simpa using congr(coprod.inr ≫ $s.condition)
    /-
      case refine_1
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.desc f g …
      H₁ : Eq (CategoryTheory.CategoryStruct.comp f s.inl) s.inr
      H₂ : Eq (CategoryTheory.CategoryStruct.comp g s.inl) s.inr
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.inl) (CategoryTheory.CategoryStru …
    -/
    exact H₁.trans H₂.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.des …
    -/
  · exact fun s ↦ by simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.des …
    -/
  · exact fun s ↦ by simpa using congr(coprod.inl ≫ $s.condition)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCoequalizer f g
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
      ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone (CategoryTheory.Limits.coprod.des …
    -/
  · exact fun s m hm _ ↦ by ext; simp [*]
    /-
      🎉 no goals
    -/


