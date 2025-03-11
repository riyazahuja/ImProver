/-- `HasPullback f g` represents a particular choice of limiting cone
for the pair of morphisms `f : X ⟶ Z` and `g : Y ⟶ Z`.
-/
abbrev HasPullback {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :=
  HasLimit (cospan f g)


/-- `HasPushout f g` represents a particular choice of colimiting cocone
for the pair of morphisms `f : X ⟶ Y` and `g : X ⟶ Z`.
-/
abbrev HasPushout {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) :=
  HasColimit (span f g)


/-- `pullback f g` computes the pullback of a pair of morphisms with the same target. -/
abbrev pullback {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] :=
  limit (cospan f g)


/-- The cone associated to the pullback of `f` and `g`-/
abbrev pullback.cone {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] : PullbackCone f g :=
  limit.cone (cospan f g)


/-- `pushout f g` computes the pushout of a pair of morphisms with the same source. -/
abbrev pushout {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] :=
  colimit (span f g)


/-- The cocone associated to the pullback of `f` and `g` -/
abbrev pushout.cocone {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] : PushoutCocone f g :=
  colimit.cocone (span f g)


/-- The first projection of the pullback of `f` and `g`. -/
abbrev pullback.fst {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] : pullback f g ⟶ X :=
  limit.π (cospan f g) WalkingCospan.left


/-- The second projection of the pullback of `f` and `g`. -/
abbrev pullback.snd {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] : pullback f g ⟶ Y :=
  limit.π (cospan f g) WalkingCospan.right


/-- The first inclusion into the pushout of `f` and `g`. -/
abbrev pushout.inl {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] : Y ⟶ pushout f g :=
  colimit.ι (span f g) WalkingSpan.left


/-- The second inclusion into the pushout of `f` and `g`. -/
abbrev pushout.inr {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] : Z ⟶ pushout f g :=
  colimit.ι (span f g) WalkingSpan.right


/-- A pair of morphisms `h : W ⟶ X` and `k : W ⟶ Y` satisfying `h ≫ f = k ≫ g` induces a morphism
    `pullback.lift : W ⟶ pullback f g`. -/
abbrev pullback.lift {W X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] (h : W ⟶ X)
    (k : W ⟶ Y) (w : h ≫ f = k ≫ g) : W ⟶ pullback f g :=
  limit.lift _ (PullbackCone.mk h k w)


/-- A pair of morphisms `h : Y ⟶ W` and `k : Z ⟶ W` satisfying `f ≫ h = g ≫ k` induces a morphism
    `pushout.desc : pushout f g ⟶ W`. -/
abbrev pushout.desc {W X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] (h : Y ⟶ W) (k : Z ⟶ W)
    (w : f ≫ h = g ≫ k) : pushout f g ⟶ W :=
  colimit.desc _ (PushoutCocone.mk h k w)


/-- The cone associated to a pullback is a limit cone. -/
abbrev pullback.isLimit {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] :
    IsLimit (pullback.cone f g) :=
  limit.isLimit (cospan f g)


/-- The cocone associated to a pushout is a colimit cone. -/
abbrev pushout.isColimit {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] :
    IsColimit (pushout.cocone f g) :=
  colimit.isColimit (span f g)


@[simp]
theorem PullbackCone.fst_limit_cone {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasLimit (cospan f g)] :
    PullbackCone.fst (limit.cone (cospan f g)) = pullback.fst f g := rfl


@[simp]
theorem PullbackCone.snd_limit_cone {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasLimit (cospan f g)] :
    PullbackCone.snd (limit.cone (cospan f g)) = pullback.snd f g := rfl


theorem PushoutCocone.inl_colimit_cocone {X Y Z : C} (f : Z ⟶ X) (g : Z ⟶ Y)
    [HasColimit (span f g)] : PushoutCocone.inl (colimit.cocone (span f g)) = pushout.inl _ _ := rfl


theorem PushoutCocone.inr_colimit_cocone {X Y Z : C} (f : Z ⟶ X) (g : Z ⟶ Y)
    [HasColimit (span f g)] : PushoutCocone.inr (colimit.cocone (span f g)) = pushout.inr _ _ := rfl


@[reassoc]
theorem pullback.lift_fst {W X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] (h : W ⟶ X)
    (k : W ⟶ Y) (w : h ≫ f = k ≫ g) : pullback.lift h k w ≫ pullback.fst f g = h :=
  limit.lift_π _ _


@[reassoc]
theorem pullback.lift_snd {W X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] (h : W ⟶ X)
    (k : W ⟶ Y) (w : h ≫ f = k ≫ g) : pullback.lift h k w ≫ pullback.snd f g = k :=
  limit.lift_π _ _


@[reassoc]
theorem pushout.inl_desc {W X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] (h : Y ⟶ W)
    (k : Z ⟶ W) (w : f ≫ h = g ≫ k) : pushout.inl _ _ ≫ pushout.desc h k w = h :=
  colimit.ι_desc _ _


@[reassoc]
theorem pushout.inr_desc {W X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] (h : Y ⟶ W)
    (k : Z ⟶ W) (w : f ≫ h = g ≫ k) : pushout.inr _ _ ≫ pushout.desc h k w = k :=
  colimit.ι_desc _ _


/-- A pair of morphisms `h : W ⟶ X` and `k : W ⟶ Y` satisfying `h ≫ f = k ≫ g` induces a morphism
    `l : W ⟶ pullback f g` such that `l ≫ pullback.fst = h` and `l ≫ pullback.snd = k`. -/
def pullback.lift' {W X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] (h : W ⟶ X) (k : W ⟶ Y)
    (w : h ≫ f = k ≫ g) :
      { l : W ⟶ pullback f g // l ≫ pullback.fst f g = h ∧ l ≫ pullback.snd f g = k } :=
  ⟨pullback.lift h k w, pullback.lift_fst _ _ _, pullback.lift_snd _ _ _⟩


/-- A pair of morphisms `h : Y ⟶ W` and `k : Z ⟶ W` satisfying `f ≫ h = g ≫ k` induces a morphism
    `l : pushout f g ⟶ W` such that `pushout.inl _ _ ≫ l = h` and `pushout.inr _ _ ≫ l = k`. -/
def pullback.desc' {W X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] (h : Y ⟶ W) (k : Z ⟶ W)
    (w : f ≫ h = g ≫ k) :
      { l : pushout f g ⟶ W // pushout.inl _ _ ≫ l = h ∧ pushout.inr _ _ ≫ l = k } :=
  ⟨pushout.desc h k w, pushout.inl_desc _ _ _, pushout.inr_desc _ _ _⟩


@[reassoc]
theorem pullback.condition {X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] :
    pullback.fst f g ≫ f = pullback.snd f g ≫ g :=
  PullbackCone.condition _


@[reassoc]
theorem pushout.condition {X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] :
    f ≫ (pushout.inl f g) = g ≫ pushout.inr _ _ :=
  PushoutCocone.condition _


/-- Two morphisms into a pullback are equal if their compositions with the pullback morphisms are
    equal -/
@[ext 1100]
theorem pullback.hom_ext {X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] {W : C}
    {k l : W ⟶ pullback f g} (h₀ : k ≫ pullback.fst f g = l ≫ pullback.fst f g)
    (h₁ : k ≫ pullback.snd f g = l ≫ pullback.snd f g) : k = l :=
  limit.hom_ext <| PullbackCone.equalizer_ext _ h₀ h₁


/-- The pullback cone built from the pullback projections is a pullback. -/
def pullbackIsPullback {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] :
    IsLimit (PullbackCone.mk (pullback.fst f g) (pullback.snd f g) pullback.condition) :=
  PullbackCone.mkSelfIsLimit <| pullback.isLimit f g


/-- Two morphisms out of a pushout are equal if their compositions with the pushout morphisms are
    equal -/
@[ext 1100]
theorem pushout.hom_ext {X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] {W : C}
    {k l : pushout f g ⟶ W} (h₀ : pushout.inl _ _ ≫ k = pushout.inl _ _ ≫ l)
    (h₁ : pushout.inr _ _ ≫ k = pushout.inr _ _ ≫ l) : k = l :=
  colimit.hom_ext <| PushoutCocone.coequalizer_ext _ h₀ h₁


/-- The pushout cocone built from the pushout coprojections is a pushout. -/
def pushoutIsPushout {X Y Z : C} (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] :
    IsColimit (PushoutCocone.mk (pushout.inl f g) (pushout.inr _ _) pushout.condition) :=
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                     W X✝ Y✝ Z✝ X Y Z : C
                                                                                     f : Quiver.Hom X Y
                                                                                     g : Quiver.Hom X Z
                                                                                     inst✝ : CategoryTheory.Limits.HasPushout f g
                                                                                     ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone f g), Eq (CategoryTheory.Category …
                                                                                   -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  PushoutCocone.IsColimit.mk _ (fun s => pushout.desc s.inl s.inr s.condition) (by simp) (by simp)
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          W X✝ Y✝ Z✝ X Y Z : C
          f : Quiver.Hom X Y
          g : Quiver.Hom X Z
          inst✝ : CategoryTheory.Limits.HasPushout f g
          ⊢ ∀ (s : CategoryTheory.Limits.PushoutCocone f g) (m : Quiver.Hom (CategoryThe …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- Given such a diagram, then there is a natural morphism `W ×ₛ X ⟶ Y ×ₜ Z`.

```
W ⟶ Y
  ↘   ↘
  S ⟶ T
  ↗   ↗
X ⟶ Z
```
-/
abbrev pullback.map {W X Y Z S T : C} (f₁ : W ⟶ S) (f₂ : X ⟶ S) [HasPullback f₁ f₂] (g₁ : Y ⟶ T)
    (g₂ : Z ⟶ T) [HasPullback g₁ g₂] (i₁ : W ⟶ Y) (i₂ : X ⟶ Z) (i₃ : S ⟶ T)
    (eq₁ : f₁ ≫ i₃ = i₁ ≫ g₁) (eq₂ : f₂ ≫ i₃ = i₂ ≫ g₂) : pullback f₁ f₂ ⟶ pullback g₁ g₂ :=
  pullback.lift (pullback.fst f₁ f₂ ≫ i₁) (pullback.snd f₁ f₂ ≫ i₂)
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          W✝ X✝ Y✝ Z✝ W X Y Z S T : C
          f₁ : Quiver.Hom W S
          f₂ : Quiver.Hom X S
          inst✝¹ : CategoryTheory.Limits.HasPullback f₁ f₂
          g₁ : Quiver.Hom Y T
          g₂ : Quiver.Hom Z T
          inst✝ : CategoryTheory.Limits.HasPullback g₁ g₂
          i₁ : Quiver.Hom W Y
          i₂ : Quiver.Hom X Z
          i₃ : Quiver.Hom S T
          eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
          eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
    (by simp only [Category.assoc, ← eq₁, ← eq₂, pullback.condition_assoc])
        /-
          🎉 no goals
        -/


/-- The canonical map `X ×ₛ Y ⟶ X ×ₜ Y` given `S ⟶ T`. -/
abbrev pullback.mapDesc {X Y S T : C} (f : X ⟶ S) (g : Y ⟶ S) (i : S ⟶ T) [HasPullback f g]
    [HasPullback (f ≫ i) (g ≫ i)] : pullback f g ⟶ pullback (f ≫ i) (g ≫ i) :=
  pullback.map f g (f ≫ i) (g ≫ i) (𝟙 _) (𝟙 _) i (Category.id_comp _).symm (Category.id_comp _).symm


@[reassoc]
lemma pullback.map_comp {X Y Z X' Y' Z' X'' Y'' Z'' : C}
    {f : X ⟶ Z} {g : Y ⟶ Z} {f' : X' ⟶ Z'} {g' : Y' ⟶ Z'} {f'' : X'' ⟶ Z''} {g'' : Y'' ⟶ Z''}
    (i₁ : X ⟶ X') (j₁ : X' ⟶ X'') (i₂ : Y ⟶ Y') (j₂ : Y' ⟶ Y'') (i₃ : Z ⟶ Z') (j₃ : Z' ⟶ Z'')
    [HasPullback f g] [HasPullback f' g'] [HasPullback f'' g'']
    (e₁ e₂ e₃ e₄) :
    pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂ ≫ pullback.map f' g' f'' g'' j₁ j₂ j₃ e₃ e₄ =
      pullback.map f g f'' g'' (i₁ ≫ j₁) (i₂ ≫ j₂) (i₃ ≫ j₃)
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              W X✝ Y✝ Z✝ X Y Z X' Y' Z' X'' Y'' Z'' : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              f' : Quiver.Hom X' Z'
              g' : Quiver.Hom Y' Z'
              f'' : Quiver.Hom X'' Z''
              g'' : Quiver.Hom Y'' Z''
              i₁ : Quiver.Hom X X'
              j₁ : Quiver.Hom X' X''
              i₂ : Quiver.Hom Y Y'
              j₂ : Quiver.Hom Y' Y''
              i₃ : Quiver.Hom Z Z'
              j₃ : Quiver.Hom Z' Z''
              inst✝² : CategoryTheory.Limits.HasPullback f g
              inst✝¹ : CategoryTheory.Limits.HasPullback f' g'
              inst✝ : CategoryTheory.Limits.HasPullback f'' g''
              e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
              e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
              e₃ : Eq (CategoryTheory.CategoryStruct.comp f' j₃) (CategoryTheory.CategoryStr …
              e₄ : Eq (CategoryTheory.CategoryStruct.comp g' j₃) (CategoryTheory.CategoryStr …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
            -/
        (by rw [reassoc_of% e₁, e₃, Category.assoc])
            /-
              🎉 no goals
            -/
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              W X✝ Y✝ Z✝ X Y Z X' Y' Z' X'' Y'' Z'' : C
              f : Quiver.Hom X Z
              g : Quiver.Hom Y Z
              f' : Quiver.Hom X' Z'
              g' : Quiver.Hom Y' Z'
              f'' : Quiver.Hom X'' Z''
              g'' : Quiver.Hom Y'' Z''
              i₁ : Quiver.Hom X X'
              j₁ : Quiver.Hom X' X''
              i₂ : Quiver.Hom Y Y'
              j₂ : Quiver.Hom Y' Y''
              i₃ : Quiver.Hom Z Z'
              j₃ : Quiver.Hom Z' Z''
              inst✝² : CategoryTheory.Limits.HasPullback f g
              inst✝¹ : CategoryTheory.Limits.HasPullback f' g'
              inst✝ : CategoryTheory.Limits.HasPullback f'' g''
              e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
              e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
              e₃ : Eq (CategoryTheory.CategoryStruct.comp f' j₃) (CategoryTheory.CategoryStr …
              e₄ : Eq (CategoryTheory.CategoryStruct.comp g' j₃) (CategoryTheory.CategoryStr …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
            -/
            /-
              🎉 no goals
            -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
        (by rw [reassoc_of% e₂, e₄, Category.assoc]) := by ext <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
lemma pullback.map_id {X Y Z : C}
    {f : X ⟶ Z} {g : Y ⟶ Z} [HasPullback f g] :
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 W X✝ Y✝ Z✝ X Y Z : C
                                                 f : Quiver.Hom X Z
                                                 g : Quiver.Hom Y Z
                                                 inst✝ : CategoryTheory.Limits.HasPullback f g
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Z …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    pullback.map f g f g (𝟙 _) (𝟙 _) (𝟙 _) (by simp) (by simp) = 𝟙 _ := by ext <;> simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- Given such a diagram, then there is a natural morphism `W ⨿ₛ X ⟶ Y ⨿ₜ Z`.

```
  W ⟶ Y
 ↗   ↗
S ⟶ T
 ↘   ↘
  X ⟶ Z
```
-/
abbrev pushout.map {W X Y Z S T : C} (f₁ : S ⟶ W) (f₂ : S ⟶ X) [HasPushout f₁ f₂] (g₁ : T ⟶ Y)
    (g₂ : T ⟶ Z) [HasPushout g₁ g₂] (i₁ : W ⟶ Y) (i₂ : X ⟶ Z) (i₃ : S ⟶ T) (eq₁ : f₁ ≫ i₁ = i₃ ≫ g₁)
    (eq₂ : f₂ ≫ i₂ = i₃ ≫ g₂) : pushout f₁ f₂ ⟶ pushout g₁ g₂ :=
  pushout.desc (i₁ ≫ pushout.inl _ _) (i₂ ≫ pushout.inr _ _)
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          W✝ X✝ Y✝ Z✝ W X Y Z S T : C
          f₁ : Quiver.Hom S W
          f₂ : Quiver.Hom S X
          inst✝¹ : CategoryTheory.Limits.HasPushout f₁ f₂
          g₁ : Quiver.Hom T Y
          g₂ : Quiver.Hom T Z
          inst✝ : CategoryTheory.Limits.HasPushout g₁ g₂
          i₁ : Quiver.Hom W Y
          i₂ : Quiver.Hom X Z
          i₃ : Quiver.Hom S T
          eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₁) (CategoryTheory.CategorySt …
          eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₂) (CategoryTheory.CategorySt …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruct.com …
        -/
    (by simp only [reassoc_of% eq₁, reassoc_of% eq₂, condition])
        /-
          🎉 no goals
        -/


/-- The canonical map `X ⨿ₛ Y ⟶ X ⨿ₜ Y` given `S ⟶ T`. -/
abbrev pushout.mapLift {X Y S T : C} (f : T ⟶ X) (g : T ⟶ Y) (i : S ⟶ T) [HasPushout f g]
    [HasPushout (i ≫ f) (i ≫ g)] : pushout (i ≫ f) (i ≫ g) ⟶ pushout f g :=
  pushout.map (i ≫ f) (i ≫ g) f g (𝟙 _) (𝟙 _) i (Category.comp_id _) (Category.comp_id _)


@[reassoc]
lemma pushout.map_comp {X Y Z X' Y' Z' X'' Y'' Z'' : C}
    {f : X ⟶ Y} {g : X ⟶ Z} {f' : X' ⟶ Y'} {g' : X' ⟶ Z'} {f'' : X'' ⟶ Y''} {g'' : X'' ⟶ Z''}
    (i₁ : X ⟶ X') (j₁ : X' ⟶ X'') (i₂ : Y ⟶ Y') (j₂ : Y' ⟶ Y'') (i₃ : Z ⟶ Z') (j₃ : Z' ⟶ Z'')
    [HasPushout f g] [HasPushout f' g'] [HasPushout f'' g'']
    (e₁ e₂ e₃ e₄) :
    pushout.map f g f' g' i₂ i₃ i₁ e₁ e₂ ≫ pushout.map f' g' f'' g'' j₂ j₃ j₁ e₃ e₄ =
      pushout.map f g f'' g'' (i₂ ≫ j₂) (i₃ ≫ j₃) (i₁ ≫ j₁)
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              W X✝ Y✝ Z✝ X Y Z X' Y' Z' X'' Y'' Z'' : C
              f : Quiver.Hom X Y
              g : Quiver.Hom X Z
              f' : Quiver.Hom X' Y'
              g' : Quiver.Hom X' Z'
              f'' : Quiver.Hom X'' Y''
              g'' : Quiver.Hom X'' Z''
              i₁ : Quiver.Hom X X'
              j₁ : Quiver.Hom X' X''
              i₂ : Quiver.Hom Y Y'
              j₂ : Quiver.Hom Y' Y''
              i₃ : Quiver.Hom Z Z'
              j₃ : Quiver.Hom Z' Z''
              inst✝² : CategoryTheory.Limits.HasPushout f g
              inst✝¹ : CategoryTheory.Limits.HasPushout f' g'
              inst✝ : CategoryTheory.Limits.HasPushout f'' g''
              e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₂) (CategoryTheory.CategoryStru …
              e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
              e₃ : Eq (CategoryTheory.CategoryStruct.comp f' j₂) (CategoryTheory.CategoryStr …
              e₄ : Eq (CategoryTheory.CategoryStruct.comp g' j₃) (CategoryTheory.CategoryStr …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
            -/
        (by rw [reassoc_of% e₁, e₃, Category.assoc])
            /-
              🎉 no goals
            -/
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              W X✝ Y✝ Z✝ X Y Z X' Y' Z' X'' Y'' Z'' : C
              f : Quiver.Hom X Y
              g : Quiver.Hom X Z
              f' : Quiver.Hom X' Y'
              g' : Quiver.Hom X' Z'
              f'' : Quiver.Hom X'' Y''
              g'' : Quiver.Hom X'' Z''
              i₁ : Quiver.Hom X X'
              j₁ : Quiver.Hom X' X''
              i₂ : Quiver.Hom Y Y'
              j₂ : Quiver.Hom Y' Y''
              i₃ : Quiver.Hom Z Z'
              j₃ : Quiver.Hom Z' Z''
              inst✝² : CategoryTheory.Limits.HasPushout f g
              inst✝¹ : CategoryTheory.Limits.HasPushout f' g'
              inst✝ : CategoryTheory.Limits.HasPushout f'' g''
              e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₂) (CategoryTheory.CategoryStru …
              e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
              e₃ : Eq (CategoryTheory.CategoryStruct.comp f' j₂) (CategoryTheory.CategoryStr …
              e₄ : Eq (CategoryTheory.CategoryStruct.comp g' j₃) (CategoryTheory.CategoryStr …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
            -/
            /-
              🎉 no goals
            -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
        (by rw [reassoc_of% e₂, e₄, Category.assoc]) := by ext <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
lemma pushout.map_id {X Y Z : C}
    {f : X ⟶ Y} {g : X ⟶ Z} [HasPushout f g] :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                W X✝ Y✝ Z✝ X Y Z : C
                                                f : Quiver.Hom X Y
                                                g : Quiver.Hom X Z
                                                inst✝ : CategoryTheory.Limits.HasPushout f g
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
    pushout.map f g f g (𝟙 _) (𝟙 _) (𝟙 _) (by simp) (by simp) = 𝟙 _ := by ext <;> simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance pullback.map_isIso {W X Y Z S T : C} (f₁ : W ⟶ S) (f₂ : X ⟶ S) [HasPullback f₁ f₂]
    (g₁ : Y ⟶ T) (g₂ : Z ⟶ T) [HasPullback g₁ g₂] (i₁ : W ⟶ Y) (i₂ : X ⟶ Z) (i₃ : S ⟶ T)
    (eq₁ : f₁ ≫ i₃ = i₁ ≫ g₁) (eq₂ : f₂ ≫ i₃ = i₂ ≫ g₂) [IsIso i₁] [IsIso i₂] [IsIso i₃] :
    IsIso (pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁ eq₂) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ W X Y Z S T : C
    f₁ : Quiver.Hom W S
    f₂ : Quiver.Hom X S
    inst✝⁴ : CategoryTheory.Limits.HasPullback f₁ f₂
    g₁ : Quiver.Hom Y T
    g₂ : Quiver.Hom Z T
    inst✝³ : CategoryTheory.Limits.HasPullback g₁ g₂
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
    inst✝² : CategoryTheory.IsIso i₁
    inst✝¹ : CategoryTheory.IsIso i₂
    inst✝ : CategoryTheory.IsIso i₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.map f₁ f₂ g₁ g₂ i₁ i₂ i …
  -/
  refine ⟨⟨pullback.map _ _ _ _ (inv i₁) (inv i₂) (inv i₃) ?_ ?_, ?_, ?_⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      inst✝⁴ : CategoryTheory.Limits.HasPullback f₁ f₂
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      inst✝³ : CategoryTheory.Limits.HasPullback g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.inv i₃)) (Category …
    -/
  · rw [IsIso.comp_inv_eq, Category.assoc, eq₁, IsIso.inv_hom_id_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      inst✝⁴ : CategoryTheory.Limits.HasPullback f₁ f₂
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      inst✝³ : CategoryTheory.Limits.HasPullback g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₂ (CategoryTheory.inv i₃)) (Category …
    -/
  · rw [IsIso.comp_inv_eq, Category.assoc, eq₂, IsIso.inv_hom_id_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      inst✝⁴ : CategoryTheory.Limits.HasPullback f₁ f₂
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      inst✝³ : CategoryTheory.Limits.HasPullback g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map f …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom W S
      f₂ : Quiver.Hom X S
      inst✝⁴ : CategoryTheory.Limits.HasPullback f₁ f₂
      g₁ : Quiver.Hom Y T
      g₂ : Quiver.Hom Z T
      inst✝³ : CategoryTheory.Limits.HasPullback g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₃) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₃) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map g …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/


/-- If `f₁ = f₂` and `g₁ = g₂`, we may construct a canonical
isomorphism `pullback f₁ g₁ ≅ pullback f₂ g₂` -/
@[simps! hom]
def pullback.congrHom {X Y Z : C} {f₁ f₂ : X ⟶ Z} {g₁ g₂ : Y ⟶ Z} (h₁ : f₁ = f₂) (h₂ : g₁ = g₂)
    [HasPullback f₁ g₁] [HasPullback f₂ g₂] : pullback f₁ g₁ ≅ pullback f₂ g₂ :=
                                                      /-
                                                        C : Type u
                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                        W X✝ Y✝ Z✝ X Y Z : C
                                                        f₁ f₂ : Quiver.Hom X Z
                                                        g₁ g₂ : Quiver.Hom Y Z
                                                        h₁ : Eq f₁ f₂
                                                        h₂ : Eq g₁ g₂
                                                        inst✝¹ : CategoryTheory.Limits.HasPullback f₁ g₁
                                                        inst✝ : CategoryTheory.Limits.HasPullback f₂ g₂
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruct.id  …
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  asIso <| pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) (by simp [h₁]) (by simp [h₂])
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem pullback.congrHom_inv {X Y Z : C} {f₁ f₂ : X ⟶ Z} {g₁ g₂ : Y ⟶ Z} (h₁ : f₁ = f₂)
    (h₂ : g₁ = g₂) [HasPullback f₁ g₁] [HasPullback f₂ g₂] :
    (pullback.congrHom h₁ h₂).inv =
                                                 /-
                                                   C : Type u
                                                   inst✝² : CategoryTheory.Category.{v, u} C
                                                   W X✝ Y✝ Z✝ X Y Z : C
                                                   f₁ f₂ : Quiver.Hom X Z
                                                   g₁ g₂ : Quiver.Hom Y Z
                                                   h₁ : Eq f₁ f₂
                                                   h₂ : Eq g₁ g₂
                                                   inst✝¹ : CategoryTheory.Limits.HasPullback f₁ g₁
                                                   inst✝ : CategoryTheory.Limits.HasPullback f₂ g₂
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f₂ (CategoryTheory.CategoryStruct.id  …
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
      pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) (by simp [h₁]) (by simp [h₂]) := by
                                                                /-
                                                                  🎉 no goals
                                                                -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f₁ f₂ : Quiver.Hom X Z
    g₁ g₂ : Quiver.Hom Y Z
    h₁ : Eq f₁ f₂
    h₂ : Eq g₁ g₂
    inst✝¹ : CategoryTheory.Limits.HasPullback f₁ g₁
    inst✝ : CategoryTheory.Limits.HasPullback f₂ g₂
    ⊢ Eq (CategoryTheory.Limits.pullback.congrHom h₁ h₂).inv (CategoryTheory.Limit …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [Iso.inv_comp_eq]
          /-
            🎉 no goals
          -/


instance pushout.map_isIso {W X Y Z S T : C} (f₁ : S ⟶ W) (f₂ : S ⟶ X) [HasPushout f₁ f₂]
    (g₁ : T ⟶ Y) (g₂ : T ⟶ Z) [HasPushout g₁ g₂] (i₁ : W ⟶ Y) (i₂ : X ⟶ Z) (i₃ : S ⟶ T)
    (eq₁ : f₁ ≫ i₁ = i₃ ≫ g₁) (eq₂ : f₂ ≫ i₂ = i₃ ≫ g₂) [IsIso i₁] [IsIso i₂] [IsIso i₃] :
    IsIso (pushout.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ eq₁ eq₂) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    W✝ X✝ Y✝ Z✝ W X Y Z S T : C
    f₁ : Quiver.Hom S W
    f₂ : Quiver.Hom S X
    inst✝⁴ : CategoryTheory.Limits.HasPushout f₁ f₂
    g₁ : Quiver.Hom T Y
    g₂ : Quiver.Hom T Z
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    i₁ : Quiver.Hom W Y
    i₂ : Quiver.Hom X Z
    i₃ : Quiver.Hom S T
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₁) (CategoryTheory.CategorySt …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₂) (CategoryTheory.CategorySt …
    inst✝² : CategoryTheory.IsIso i₁
    inst✝¹ : CategoryTheory.IsIso i₂
    inst✝ : CategoryTheory.IsIso i₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pushout.map f₁ f₂ g₁ g₂ i₁ i₂ i₃ …
  -/
  refine ⟨⟨pushout.map _ _ _ _ (inv i₁) (inv i₂) (inv i₃) ?_ ?_, ?_, ?_⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom S W
      f₂ : Quiver.Hom S X
      inst✝⁴ : CategoryTheory.Limits.HasPushout f₁ f₂
      g₁ : Quiver.Hom T Y
      g₂ : Quiver.Hom T Z
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₁) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₂) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.inv i₁)) (Category …
    -/
  · rw [IsIso.comp_inv_eq, Category.assoc, eq₁, IsIso.inv_hom_id_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom S W
      f₂ : Quiver.Hom S X
      inst✝⁴ : CategoryTheory.Limits.HasPushout f₁ f₂
      g₁ : Quiver.Hom T Y
      g₂ : Quiver.Hom T Z
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₁) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₂) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₂ (CategoryTheory.inv i₂)) (Category …
    -/
  · rw [IsIso.comp_inv_eq, Category.assoc, eq₂, IsIso.inv_hom_id_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom S W
      f₂ : Quiver.Hom S X
      inst✝⁴ : CategoryTheory.Limits.HasPushout f₁ f₂
      g₁ : Quiver.Hom T Y
      g₂ : Quiver.Hom T Z
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₁) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₂) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.map f₁ …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      W✝ X✝ Y✝ Z✝ W X Y Z S T : C
      f₁ : Quiver.Hom S W
      f₂ : Quiver.Hom S X
      inst✝⁴ : CategoryTheory.Limits.HasPushout f₁ f₂
      g₁ : Quiver.Hom T Y
      g₂ : Quiver.Hom T Z
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      i₁ : Quiver.Hom W Y
      i₂ : Quiver.Hom X Z
      i₃ : Quiver.Hom S T
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ i₁) (CategoryTheory.CategorySt …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ i₂) (CategoryTheory.CategorySt …
      inst✝² : CategoryTheory.IsIso i₁
      inst✝¹ : CategoryTheory.IsIso i₂
      inst✝ : CategoryTheory.IsIso i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.map g₁ …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/


theorem pullback.mapDesc_comp {X Y S T S' : C} (f : X ⟶ T) (g : Y ⟶ T) (i : T ⟶ S) (i' : S ⟶ S')
    [HasPullback f g] [HasPullback (f ≫ i) (g ≫ i)] [HasPullback (f ≫ i ≫ i') (g ≫ i ≫ i')]
    [HasPullback ((f ≫ i) ≫ i') ((g ≫ i) ≫ i')] :
    pullback.mapDesc f g (i ≫ i') = pullback.mapDesc f g i ≫ pullback.mapDesc _ _ i' ≫
    (pullback.congrHom (Category.assoc _ _ _) (Category.assoc _ _ _)).hom := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y S T S' : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    i' : Quiver.Hom S S'
    inst✝³ : CategoryTheory.Limits.HasPullback f g
    inst✝² : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    ⊢ Eq (CategoryTheory.Limits.pullback.mapDesc f g (CategoryTheory.CategoryStruc …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- If `f₁ = f₂` and `g₁ = g₂`, we may construct a canonical
isomorphism `pushout f₁ g₁ ≅ pullback f₂ g₂` -/
@[simps! hom]
def pushout.congrHom {X Y Z : C} {f₁ f₂ : X ⟶ Y} {g₁ g₂ : X ⟶ Z} (h₁ : f₁ = f₂) (h₂ : g₁ = g₂)
    [HasPushout f₁ g₁] [HasPushout f₂ g₂] : pushout f₁ g₁ ≅ pushout f₂ g₂ :=
                                                     /-
                                                       C : Type u
                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                       W X✝ Y✝ Z✝ X Y Z : C
                                                       f₁ f₂ : Quiver.Hom X Y
                                                       g₁ g₂ : Quiver.Hom X Z
                                                       h₁ : Eq f₁ f₂
                                                       h₂ : Eq g₁ g₂
                                                       inst✝¹ : CategoryTheory.Limits.HasPushout f₁ g₁
                                                       inst✝ : CategoryTheory.Limits.HasPushout f₂ g₂
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ (CategoryTheory.CategoryStruct.id  …
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  asIso <| pushout.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) (by simp [h₁]) (by simp [h₂])
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem pushout.congrHom_inv {X Y Z : C} {f₁ f₂ : X ⟶ Y} {g₁ g₂ : X ⟶ Z} (h₁ : f₁ = f₂)
    (h₂ : g₁ = g₂) [HasPushout f₁ g₁] [HasPushout f₂ g₂] :
    (pushout.congrHom h₁ h₂).inv =
                                                /-
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  W X✝ Y✝ Z✝ X Y Z : C
                                                  f₁ f₂ : Quiver.Hom X Y
                                                  g₁ g₂ : Quiver.Hom X Z
                                                  h₁ : Eq f₁ f₂
                                                  h₂ : Eq g₁ g₂
                                                  inst✝¹ : CategoryTheory.Limits.HasPushout f₁ g₁
                                                  inst✝ : CategoryTheory.Limits.HasPushout f₂ g₂
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f₂ (CategoryTheory.CategoryStruct.id  …
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
      pushout.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) (by simp [h₁]) (by simp [h₂]) := by
                                                               /-
                                                                 🎉 no goals
                                                               -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f₁ f₂ : Quiver.Hom X Y
    g₁ g₂ : Quiver.Hom X Z
    h₁ : Eq f₁ f₂
    h₂ : Eq g₁ g₂
    inst✝¹ : CategoryTheory.Limits.HasPushout f₁ g₁
    inst✝ : CategoryTheory.Limits.HasPushout f₂ g₂
    ⊢ Eq (CategoryTheory.Limits.pushout.congrHom h₁ h₂).inv (CategoryTheory.Limits …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [Iso.comp_inv_eq]
          /-
            🎉 no goals
          -/


theorem pushout.mapLift_comp {X Y S T S' : C} (f : T ⟶ X) (g : T ⟶ Y) (i : S ⟶ T) (i' : S' ⟶ S)
    [HasPushout f g] [HasPushout (i ≫ f) (i ≫ g)] [HasPushout (i' ≫ i ≫ f) (i' ≫ i ≫ g)]
    [HasPushout ((i' ≫ i) ≫ f) ((i' ≫ i) ≫ g)] :
    pushout.mapLift f g (i' ≫ i) =
      (pushout.congrHom (Category.assoc _ _ _) (Category.assoc _ _ _)).hom ≫
        pushout.mapLift _ _ i' ≫ pushout.mapLift f g i := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y S T S' : C
    f : Quiver.Hom T X
    g : Quiver.Hom T Y
    i : Quiver.Hom S T
    i' : Quiver.Hom S' S
    inst✝³ : CategoryTheory.Limits.HasPushout f g
    inst✝² : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp ( …
    ⊢ Eq (CategoryTheory.Limits.pushout.mapLift f g (CategoryTheory.CategoryStruct …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- The comparison morphism for the pullback of `f,g`.
This is an isomorphism iff `G` preserves the pullback of `f,g`; see
`Mathlib/CategoryTheory/Limits/Preserves/Shapes/Pullbacks.lean`
-/
def pullbackComparison (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g] [HasPullback (G.map f) (G.map g)] :
    G.obj (pullback f g) ⟶ pullback (G.map f) (G.map g) :=
  pullback.lift (G.map (pullback.fst f g)) (G.map (pullback.snd f g))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝¹ : CategoryTheory.Limits.HasPullback f g
          inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.pullbac …
        -/
    (by simp only [← G.map_comp, pullback.condition])
        /-
          🎉 no goals
        -/


@[reassoc (attr := simp)]
theorem pullbackComparison_comp_fst (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g]
    [HasPullback (G.map f) (G.map g)] :
    pullbackComparison G f g ≫ pullback.fst _ _ = G.map (pullback.fst f g) :=
  pullback.lift_fst _ _ _


@[reassoc (attr := simp)]
theorem pullbackComparison_comp_snd (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g]
    [HasPullback (G.map f) (G.map g)] :
    pullbackComparison G f g ≫ pullback.snd _ _ = G.map (pullback.snd f g):=
  pullback.lift_snd _ _ _


@[reassoc (attr := simp)]
theorem map_lift_pullbackComparison (f : X ⟶ Z) (g : Y ⟶ Z) [HasPullback f g]
    [HasPullback (G.map f) (G.map g)] {W : C} {h : W ⟶ X} {k : W ⟶ Y} (w : h ≫ f = k ≫ g) :
    G.map (pullback.lift _ _ w) ≫ pullbackComparison G f g =
                                            /-
                                              C : Type u
                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                              W✝ X Y Z : C
                                              D : Type u₂
                                              inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                              G : CategoryTheory.Functor C D
                                              f : Quiver.Hom X Z
                                              g : Quiver.Hom Y Z
                                              inst✝¹ : CategoryTheory.Limits.HasPullback f g
                                              inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
                                              W : C
                                              h : Quiver.Hom W X
                                              k : Quiver.Hom W Y
                                              w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) (CategoryTheory. …
                                            -/
      pullback.lift (G.map h) (G.map k) (by simp only [← G.map_comp, w]) := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasPullback f g
    inst✝ : CategoryTheory.Limits.HasPullback (G.map f) (G.map g)
    W : C
    h : Quiver.Hom W X
    k : Quiver.Hom W Y
    w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.pullbac …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [← G.map_comp]
          /-
            🎉 no goals
          -/


/-- The comparison morphism for the pushout of `f,g`.
This is an isomorphism iff `G` preserves the pushout of `f,g`; see
`Mathlib/CategoryTheory/Limits/Preserves/Shapes/Pullbacks.lean`
-/
def pushoutComparison (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g] [HasPushout (G.map f) (G.map g)] :
    pushout (G.map f) (G.map g) ⟶ G.obj (pushout f g) :=
  pushout.desc (G.map (pushout.inl _ _)) (G.map (pushout.inr _ _))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          W X Y Z : C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          G : CategoryTheory.Functor C D
          f : Quiver.Hom X Y
          g : Quiver.Hom X Z
          inst✝¹ : CategoryTheory.Limits.HasPushout f g
          inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
        -/
    (by simp only [← G.map_comp, pushout.condition])
        /-
          🎉 no goals
        -/


@[reassoc (attr := simp)]
theorem inl_comp_pushoutComparison (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g]
    [HasPushout (G.map f) (G.map g)] : pushout.inl _ _ ≫ pushoutComparison G f g =
      G.map (pushout.inl _ _) :=
  pushout.inl_desc _ _ _


@[reassoc (attr := simp)]
theorem inr_comp_pushoutComparison (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g]
    [HasPushout (G.map f) (G.map g)] : pushout.inr _ _ ≫ pushoutComparison G f g =
      G.map (pushout.inr _ _) :=
  pushout.inr_desc _ _ _


@[reassoc (attr := simp)]
theorem pushoutComparison_map_desc (f : X ⟶ Y) (g : X ⟶ Z) [HasPushout f g]
    [HasPushout (G.map f) (G.map g)] {W : C} {h : Y ⟶ W} {k : Z ⟶ W} (w : f ≫ h = g ≫ k) :
    pushoutComparison G f g ≫ G.map (pushout.desc _ _ w) =
                                           /-
                                             C : Type u
                                             inst✝³ : CategoryTheory.Category.{v, u} C
                                             W✝ X Y Z : C
                                             D : Type u₂
                                             inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                             G : CategoryTheory.Functor C D
                                             f : Quiver.Hom X Y
                                             g : Quiver.Hom X Z
                                             inst✝¹ : CategoryTheory.Limits.HasPushout f g
                                             inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
                                             W : C
                                             h : Quiver.Hom Y W
                                             k : Quiver.Hom Z W
                                             w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) (CategoryTheory. …
                                           -/
      pushout.desc (G.map h) (G.map k) (by simp only [← G.map_comp, w]) := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    inst✝¹ : CategoryTheory.Limits.HasPushout f g
    inst✝ : CategoryTheory.Limits.HasPushout (G.map f) (G.map g)
    W : C
    h : Quiver.Hom Y W
    k : Quiver.Hom Z W
    w : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushoutCompari …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [← G.map_comp]
          /-
            🎉 no goals
          -/


/-- Making this a global instance would make the typeclass search go in an infinite loop. -/
theorem hasPullback_symmetry [HasPullback f g] : HasPullback g f :=
  ⟨⟨⟨_, PullbackCone.flipIsLimit (pullbackIsPullback f g)⟩⟩⟩


/-- The isomorphism `X ×[Z] Y ≅ Y ×[Z] X`. -/
def pullbackSymmetry [HasPullback f g] : pullback f g ≅ pullback g f :=
  IsLimit.conePointUniqueUpToIso
    (PullbackCone.flipIsLimit (pullbackIsPullback f g)) (limit.isLimit _)


@[reassoc (attr := simp)]
theorem pullbackSymmetry_hom_comp_fst [HasPullback f g] :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             X Y Z : C
                                                                             f : Quiver.Hom X Z
                                                                             g : Quiver.Hom Y Z
                                                                             inst✝ : CategoryTheory.Limits.HasPullback f g
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
                                                                           -/
    (pullbackSymmetry f g).hom ≫ pullback.fst g f = pullback.snd f g := by simp [pullbackSymmetry]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[reassoc (attr := simp)]
theorem pullbackSymmetry_hom_comp_snd [HasPullback f g] :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             X Y Z : C
                                                                             f : Quiver.Hom X Z
                                                                             g : Quiver.Hom Y Z
                                                                             inst✝ : CategoryTheory.Limits.HasPullback f g
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
                                                                           -/
    (pullbackSymmetry f g).hom ≫ pullback.snd g f = pullback.fst f g := by simp [pullbackSymmetry]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[reassoc (attr := simp)]
theorem pullbackSymmetry_inv_comp_fst [HasPullback f g] :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             X Y Z : C
                                                                             f : Quiver.Hom X Z
                                                                             g : Quiver.Hom Y Z
                                                                             inst✝ : CategoryTheory.Limits.HasPullback f g
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
                                                                           -/
    (pullbackSymmetry f g).inv ≫ pullback.fst f g = pullback.snd g f := by simp [Iso.inv_comp_eq]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[reassoc (attr := simp)]
theorem pullbackSymmetry_inv_comp_snd [HasPullback f g] :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             X Y Z : C
                                                                             f : Quiver.Hom X Z
                                                                             g : Quiver.Hom Y Z
                                                                             inst✝ : CategoryTheory.Limits.HasPullback f g
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackSymmet …
                                                                           -/
    (pullbackSymmetry f g).inv ≫ pullback.snd f g = pullback.fst g f := by simp [Iso.inv_comp_eq]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Making this a global instance would make the typeclass search go in an infinite loop. -/
theorem hasPushout_symmetry [HasPushout f g] : HasPushout g f :=
  ⟨⟨⟨_, PushoutCocone.flipIsColimit (pushoutIsPushout f g)⟩⟩⟩


/-- The isomorphism `Y ⨿[X] Z ≅ Z ⨿[X] Y`. -/
def pushoutSymmetry [HasPushout f g] : pushout f g ≅ pushout g f :=
  IsColimit.coconePointUniqueUpToIso
    (PushoutCocone.flipIsColimit (pushoutIsPushout f g)) (colimit.isColimit _)


@[reassoc (attr := simp)]
theorem inl_comp_pushoutSymmetry_hom [HasPushout f g] :
    pushout.inl _ _ ≫ (pushoutSymmetry f g).hom = pushout.inr _ _ :=
  (colimit.isColimit (span f g)).comp_coconePointUniqueUpToIso_hom
    (PushoutCocone.flipIsColimit (pushoutIsPushout g f)) _


@[reassoc (attr := simp)]
theorem inr_comp_pushoutSymmetry_hom [HasPushout f g] :
    pushout.inr _ _ ≫ (pushoutSymmetry f g).hom = pushout.inl _ _ :=
  (colimit.isColimit (span f g)).comp_coconePointUniqueUpToIso_hom
    (PushoutCocone.flipIsColimit (pushoutIsPushout g f)) _


@[reassoc (attr := simp)]
theorem inl_comp_pushoutSymmetry_inv [HasPushout f g] :
                                                                        /-
                                                                          C : Type u
                                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                          X Y Z : C
                                                                          f : Quiver.Hom X Y
                                                                          g : Quiver.Hom X Z
                                                                          inst✝ : CategoryTheory.Limits.HasPushout f g
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl g  …
                                                                        -/
    pushout.inl _ _ ≫ (pushoutSymmetry f g).inv = pushout.inr _ _ := by simp [Iso.comp_inv_eq]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[reassoc (attr := simp)]
theorem inr_comp_pushoutSymmetry_inv [HasPushout f g] :
                                                                        /-
                                                                          C : Type u
                                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                          X Y Z : C
                                                                          f : Quiver.Hom X Y
                                                                          g : Quiver.Hom X Z
                                                                          inst✝ : CategoryTheory.Limits.HasPushout f g
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g  …
                                                                        -/
    pushout.inr _ _ ≫ (pushoutSymmetry f g).inv = pushout.inl _ _ := by simp [Iso.comp_inv_eq]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- `HasPullbacks` represents a choice of pullback for every pair of morphisms

See <https://stacks.math.columbia.edu/tag/001W>
-/
abbrev HasPullbacks :=
  HasLimitsOfShape WalkingCospan C


/-- `HasPushouts` represents a choice of pushout for every pair of morphisms -/
abbrev HasPushouts :=
  HasColimitsOfShape WalkingSpan C


/-- If `C` has all limits of diagrams `cospan f g`, then it has all pullbacks -/
theorem hasPullbacks_of_hasLimit_cospan
    [∀ {X Y Z : C} {f : X ⟶ Z} {g : Y ⟶ Z}, HasLimit (cospan f g)] : HasPullbacks C :=
  { has_limit := fun F => hasLimitOfIso (diagramIsoCospan F).symm }


/-- If `C` has all colimits of diagrams `span f g`, then it has all pushouts -/
theorem hasPushouts_of_hasColimit_span
    [∀ {X Y Z : C} {f : X ⟶ Y} {g : X ⟶ Z}, HasColimit (span f g)] : HasPushouts C :=
  { has_colimit := fun F => hasColimitOfIso (diagramIsoSpan F) }


/-- The duality equivalence `WalkingSpanᵒᵖ ≌ WalkingCospan` -/
@[simps!]
def walkingSpanOpEquiv : WalkingSpanᵒᵖ ≌ WalkingCospan :=
  widePushoutShapeOpEquiv _


/-- The duality equivalence `WalkingCospanᵒᵖ ≌ WalkingSpan` -/
@[simps!]
def walkingCospanOpEquiv : WalkingCospanᵒᵖ ≌ WalkingSpan :=
  widePullbackShapeOpEquiv _

-- see Note [lower instance priority]

/-- Having wide pullback at any universe level implies having binary pullbacks. -/
instance (priority := 100) hasPullbacks_of_hasWidePullbacks (D : Type u) [Category.{v} D]
    [HasWidePullbacks.{w} D] : HasPullbacks.{v,u} D :=
  hasWidePullbacks_shrink WalkingPair

-- see Note [lower instance priority]

/-- Having wide pushout at any universe level implies having binary pushouts. -/
instance (priority := 100) hasPushouts_of_hasWidePushouts (D : Type u) [Category.{v} D]
    [HasWidePushouts.{w} D] : HasPushouts.{v,u} D :=
  hasWidePushouts_shrink WalkingPair


