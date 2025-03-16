/-- If `f : X ⟶ Z` is iso, then `X ×[Z] Y ≅ Y`. This is the explicit limit cone. -/
def pullbackConeOfLeftIso : PullbackCone f g :=
                                          /-
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            X Y Z : C
                                            f : Quiver.Hom X Z
                                            g : Quiver.Hom Y Z
                                            inst✝ : CategoryTheory.IsIso f
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
                                          -/
  PullbackCone.mk (g ≫ inv f) (𝟙 _) <| by simp
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem pullbackConeOfLeftIso_x : (pullbackConeOfLeftIso f g).pt = Y := rfl


@[simp]
theorem pullbackConeOfLeftIso_fst : (pullbackConeOfLeftIso f g).fst = g ≫ inv f := rfl


@[simp]
theorem pullbackConeOfLeftIso_snd : (pullbackConeOfLeftIso f g).snd = 𝟙 _ := rfl


                                                                                            /-
                                                                                              C : Type u
                                                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                              X Y Z : C
                                                                                              f : Quiver.Hom X Z
                                                                                              g : Quiver.Hom Y Z
                                                                                              inst✝ : CategoryTheory.IsIso f
                                                                                              ⊢ Eq ((CategoryTheory.Limits.pullbackConeOfLeftIso f g).π.app Option.none) g
                                                                                            -/
theorem pullbackConeOfLeftIso_π_app_none : (pullbackConeOfLeftIso f g).π.app none = g := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
theorem pullbackConeOfLeftIso_π_app_left : (pullbackConeOfLeftIso f g).π.app left = g ≫ inv f :=
  rfl


@[simp]
theorem pullbackConeOfLeftIso_π_app_right : (pullbackConeOfLeftIso f g).π.app right = 𝟙 _ := rfl


/-- Verify that the constructed limit cone is indeed a limit. -/
def pullbackConeOfLeftIsoIsLimit : IsLimit (pullbackConeOfLeftIso f g) :=
                                                 /-
                                                   C : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                   X Y Z : C
                                                   f : Quiver.Hom X Z
                                                   g : Quiver.Hom Y Z
                                                   inst✝ : CategoryTheory.IsIso f
                                                   s : CategoryTheory.Limits.PullbackCone f g
                                                   ⊢ And (Eq (CategoryTheory.CategoryStruct.comp s.snd (CategoryTheory.Limits.pul …
                                                 -/
  PullbackCone.isLimitAux' _ fun s => ⟨s.snd, by simp [← s.condition_assoc]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem hasPullback_of_left_iso : HasPullback f g :=
  ⟨⟨⟨_, pullbackConeOfLeftIsoIsLimit f g⟩⟩⟩


instance pullback_snd_iso_of_left_iso : IsIso (pullback.snd f g) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.snd f g)
  -/
  refine ⟨⟨pullback.lift (g ≫ inv f) (𝟙 _) (by simp), ?_, by simp⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd f …
  -/
  ext
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [← pullback.condition_assoc]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [pullback.condition_assoc]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pullback_inv_snd_fst_of_left_isIso :
    inv (pullback.snd f g) ≫ pullback.fst f g = g ≫ inv f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
  -/
  rw [IsIso.inv_comp_eq, ← pullback.condition_assoc, IsIso.hom_inv_id, Category.comp_id]
  /-
    🎉 no goals
  -/


/-- If `g : Y ⟶ Z` is iso, then `X ×[Z] Y ≅ X`. This is the explicit limit cone. -/
def pullbackConeOfRightIso : PullbackCone f g :=
                                          /-
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            X Y Z : C
                                            f : Quiver.Hom X Z
                                            g : Quiver.Hom Y Z
                                            inst✝ : CategoryTheory.IsIso g
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                          -/
  PullbackCone.mk (𝟙 _) (f ≫ inv g) <| by simp
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem pullbackConeOfRightIso_x : (pullbackConeOfRightIso f g).pt = X := rfl


@[simp]
theorem pullbackConeOfRightIso_fst : (pullbackConeOfRightIso f g).fst = 𝟙 _ := rfl


@[simp]
theorem pullbackConeOfRightIso_snd : (pullbackConeOfRightIso f g).snd = f ≫ inv g := rfl


                                                                                              /-
                                                                                                C : Type u
                                                                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                                X Y Z : C
                                                                                                f : Quiver.Hom X Z
                                                                                                g : Quiver.Hom Y Z
                                                                                                inst✝ : CategoryTheory.IsIso g
                                                                                                ⊢ Eq ((CategoryTheory.Limits.pullbackConeOfRightIso f g).π.app Option.none) f
                                                                                              -/
theorem pullbackConeOfRightIso_π_app_none : (pullbackConeOfRightIso f g).π.app none = f := by simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[simp]
theorem pullbackConeOfRightIso_π_app_left : (pullbackConeOfRightIso f g).π.app left = 𝟙 _ :=
  rfl


@[simp]
theorem pullbackConeOfRightIso_π_app_right : (pullbackConeOfRightIso f g).π.app right = f ≫ inv g :=
  rfl


/-- Verify that the constructed limit cone is indeed a limit. -/
def pullbackConeOfRightIsoIsLimit : IsLimit (pullbackConeOfRightIso f g) :=
                                                 /-
                                                   C : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                   X Y Z : C
                                                   f : Quiver.Hom X Z
                                                   g : Quiver.Hom Y Z
                                                   inst✝ : CategoryTheory.IsIso g
                                                   s : CategoryTheory.Limits.PullbackCone f g
                                                   ⊢ And (Eq (CategoryTheory.CategoryStruct.comp s.fst (CategoryTheory.Limits.pul …
                                                 -/
  PullbackCone.isLimitAux' _ fun s => ⟨s.fst, by simp [s.condition_assoc]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem hasPullback_of_right_iso : HasPullback f g :=
  ⟨⟨⟨_, pullbackConeOfRightIsoIsLimit f g⟩⟩⟩


instance pullback_snd_iso_of_right_iso : IsIso (pullback.fst f g) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.fst f g)
  -/
  refine ⟨⟨pullback.lift (𝟙 _) (f ≫ inv g) (by simp), ?_, by simp⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst f …
  -/
  ext
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [pullback.condition_assoc]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pullback_inv_fst_snd_of_right_isIso :
    inv (pullback.fst f g) ≫ pullback.snd f g = f ≫ inv g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
  -/
  rw [IsIso.inv_comp_eq, pullback.condition_assoc, IsIso.hom_inv_id, Category.comp_id]
  /-
    🎉 no goals
  -/


/-- If `f : X ⟶ Y` is iso, then `Y ⨿[X] Z ≅ Z`. This is the explicit colimit cocone. -/
def pushoutCoconeOfLeftIso : PushoutCocone f g :=
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             X Y Z : C
                                             f : Quiver.Hom X Y
                                             g : Quiver.Hom X Z
                                             inst✝ : CategoryTheory.IsIso f
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                           -/
  PushoutCocone.mk (inv f ≫ g) (𝟙 _) <| by simp
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem pushoutCoconeOfLeftIso_x : (pushoutCoconeOfLeftIso f g).pt = Z := rfl


@[simp]
theorem pushoutCoconeOfLeftIso_inl : (pushoutCoconeOfLeftIso f g).inl = inv f ≫ g := rfl


@[simp]
theorem pushoutCoconeOfLeftIso_inr : (pushoutCoconeOfLeftIso f g).inr = 𝟙 _ := rfl


theorem pushoutCoconeOfLeftIso_ι_app_none : (pushoutCoconeOfLeftIso f g).ι.app none = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq ((CategoryTheory.Limits.pushoutCoconeOfLeftIso f g).ι.app Option.none) g
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem pushoutCoconeOfLeftIso_ι_app_left : (pushoutCoconeOfLeftIso f g).ι.app left = inv f ≫ g :=
  rfl


@[simp]
theorem pushoutCoconeOfLeftIso_ι_app_right : (pushoutCoconeOfLeftIso f g).ι.app right = 𝟙 _ := rfl


/-- Verify that the constructed cocone is indeed a colimit. -/
def pushoutCoconeOfLeftIsoIsLimit : IsColimit (pushoutCoconeOfLeftIso f g) :=
                                                    /-
                                                      C : Type u
                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                      X Y Z : C
                                                      f : Quiver.Hom X Y
                                                      g : Quiver.Hom X Z
                                                      inst✝ : CategoryTheory.IsIso f
                                                      s : CategoryTheory.Limits.PushoutCocone f g
                                                      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushoutCo …
                                                    -/
  PushoutCocone.isColimitAux' _ fun s => ⟨s.inr, by simp [← s.condition]⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem hasPushout_of_left_iso : HasPushout f g :=
  ⟨⟨⟨_, pushoutCoconeOfLeftIsoIsLimit f g⟩⟩⟩


instance pushout_inr_iso_of_left_iso : IsIso (pushout.inr f g) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pushout.inr f g)
  -/
  refine ⟨⟨pushout.desc (inv f ≫ g) (𝟙 _) (by simp), by simp, ?_⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.desc ( …
  -/
  ext
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.IsIso f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
    -/
  · simp [← pushout.condition]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.IsIso f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
    -/
  · simp [pushout.condition_assoc]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pushout_inl_inv_inr_of_right_isIso :
    pushout.inl f g ≫ inv (pushout.inr f g) = inv f ≫ g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
  -/
  rw [IsIso.eq_inv_comp, pushout.condition_assoc, IsIso.hom_inv_id, Category.comp_id]
  /-
    🎉 no goals
  -/


/-- If `f : X ⟶ Z` is iso, then `Y ⨿[X] Z ≅ Y`. This is the explicit colimit cocone. -/
def pushoutCoconeOfRightIso : PushoutCocone f g :=
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             X Y Z : C
                                             f : Quiver.Hom X Y
                                             g : Quiver.Hom X Z
                                             inst✝ : CategoryTheory.IsIso g
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                                           -/
  PushoutCocone.mk (𝟙 _) (inv g ≫ f) <| by simp
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem pushoutCoconeOfRightIso_x : (pushoutCoconeOfRightIso f g).pt = Y := rfl


@[simp]
theorem pushoutCoconeOfRightIso_inl : (pushoutCoconeOfRightIso f g).inl = 𝟙 _ := rfl


@[simp]
theorem pushoutCoconeOfRightIso_inr : (pushoutCoconeOfRightIso f g).inr = inv g ≫ f := rfl


theorem pushoutCoconeOfRightIso_ι_app_none : (pushoutCoconeOfRightIso f g).ι.app none = f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq ((CategoryTheory.Limits.pushoutCoconeOfRightIso f g).ι.app Option.none) f
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem pushoutCoconeOfRightIso_ι_app_left : (pushoutCoconeOfRightIso f g).ι.app left = 𝟙 _ := rfl


@[simp]
theorem pushoutCoconeOfRightIso_ι_app_right :
    (pushoutCoconeOfRightIso f g).ι.app right = inv g ≫ f := rfl


/-- Verify that the constructed cocone is indeed a colimit. -/
def pushoutCoconeOfRightIsoIsLimit : IsColimit (pushoutCoconeOfRightIso f g) :=
                                                    /-
                                                      C : Type u
                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                      X Y Z : C
                                                      f : Quiver.Hom X Y
                                                      g : Quiver.Hom X Z
                                                      inst✝ : CategoryTheory.IsIso g
                                                      s : CategoryTheory.Limits.PushoutCocone f g
                                                      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushoutCo …
                                                    -/
  PushoutCocone.isColimitAux' _ fun s => ⟨s.inl, by simp [← s.condition]⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem hasPushout_of_right_iso : HasPushout f g :=
  ⟨⟨⟨_, pushoutCoconeOfRightIsoIsLimit f g⟩⟩⟩


instance pushout_inl_iso_of_right_iso : IsIso (pushout.inl _ _ : _ ⟶ pushout f g) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pushout.inl f g)
  -/
  refine ⟨⟨pushout.desc (𝟙 _) (inv g ≫ f) (by simp), by simp, ?_⟩⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.desc ( …
  -/
  ext
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.IsIso g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl f  …
    -/
  · simp [← pushout.condition]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.IsIso g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
    -/
  · simp [pushout.condition]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pushout_inr_inv_inl_of_right_isIso :
    pushout.inr f g ≫ inv (pushout.inl f g) = inv g ≫ f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr f  …
  -/
  rw [IsIso.eq_inv_comp, ← pushout.condition_assoc, IsIso.hom_inv_id, Category.comp_id]
  /-
    🎉 no goals
  -/


