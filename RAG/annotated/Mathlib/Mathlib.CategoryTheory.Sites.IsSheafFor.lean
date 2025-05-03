/-- A family of elements for a presheaf `P` given a collection of arrows `R` with fixed codomain `X`
consists of an element of `P Y` for every `f : Y ⟶ X` in `R`.
A presheaf is a sheaf (resp, separated) if every *compatible* family of elements has exactly one
(resp, at most one) amalgamation.

This data is referred to as a `family` in [MM92], Chapter III, Section 4. It is also a concrete
version of the elements of the middle object in https://stacks.math.columbia.edu/tag/00VM which is
more useful for direct calculations. It is also used implicitly in Definition C2.1.2 in [Elephant].
-/
def FamilyOfElements (P : Cᵒᵖ ⥤ Type w) (R : Presieve X) :=
  ∀ ⦃Y : C⦄ (f : Y ⟶ X), R f → P.obj (op Y)


instance : Inhabited (FamilyOfElements P (⊥ : Presieve X)) :=
  ⟨fun _ _ => False.elim⟩


/-- A family of elements for a presheaf on the presieve `R₂` can be restricted to a smaller presieve
`R₁`.
-/
def FamilyOfElements.restrict {R₁ R₂ : Presieve X} (h : R₁ ≤ R₂) :
    FamilyOfElements P R₂ → FamilyOfElements P R₁ := fun x _ f hf => x f (h _ hf)


/-- The image of a family of elements by a morphism of presheaves. -/
def FamilyOfElements.map (p : FamilyOfElements P R) (φ : P ⟶ Q) :
    FamilyOfElements Q R :=
  fun _ f hf => φ.app _ (p f hf)


@[simp]
lemma FamilyOfElements.map_apply
    (p : FamilyOfElements P R) (φ : P ⟶ Q) {Y : C} (f : Y ⟶ X) (hf : R f) :
    p.map φ f hf = φ.app _ (p f hf) := rfl


lemma FamilyOfElements.restrict_map
    (p : FamilyOfElements P R) (φ : P ⟶ Q) {R' : Presieve X} (h : R' ≤ R) :
    (p.restrict h).map φ = (p.map φ).restrict h := rfl


/-- A family of elements for the arrow set `R` is *compatible* if for any `f₁ : Y₁ ⟶ X` and
`f₂ : Y₂ ⟶ X` in `R`, and any `g₁ : Z ⟶ Y₁` and `g₂ : Z ⟶ Y₂`, if the square `g₁ ≫ f₁ = g₂ ≫ f₂`
commutes then the elements of `P Z` obtained by restricting the element of `P Y₁` along `g₁` and
restricting the element of `P Y₂` along `g₂` are the same.

In special cases, this condition can be simplified, see `pullbackCompatible_iff` and
`compatible_iff_sieveCompatible`.

This is referred to as a "compatible family" in Definition C2.1.2 of [Elephant], and on nlab:
https://ncatlab.org/nlab/show/sheaf#GeneralDefinitionInComponents

For a more explicit version in the case where `R` is of the form `Presieve.ofArrows`, see
`CategoryTheory.Presieve.Arrows.Compatible`.
-/
def FamilyOfElements.Compatible (x : FamilyOfElements P R) : Prop :=
  ∀ ⦃Y₁ Y₂ Z⦄ (g₁ : Z ⟶ Y₁) (g₂ : Z ⟶ Y₂) ⦃f₁ : Y₁ ⟶ X⦄ ⦃f₂ : Y₂ ⟶ X⦄ (h₁ : R f₁) (h₂ : R f₂),
    g₁ ≫ f₁ = g₂ ≫ f₂ → P.map g₁.op (x f₁ h₁) = P.map g₂.op (x f₂ h₂)


/--
If the category `C` has pullbacks, this is an alternative condition for a family of elements to be
compatible: For any `f : Y ⟶ X` and `g : Z ⟶ X` in the presieve `R`, the restriction of the
given elements for `f` and `g` to the pullback agree.
This is equivalent to being compatible (provided `C` has pullbacks), shown in
`pullbackCompatible_iff`.

This is the definition for a "matching" family given in [MM92], Chapter III, Section 4,
Equation (5). Viewing the type `FamilyOfElements` as the middle object of the fork in
https://stacks.math.columbia.edu/tag/00VM, this condition expresses that `pr₀* (x) = pr₁* (x)`,
using the notation defined there.

For a more explicit version in the case where `R` is of the form `Presieve.ofArrows`, see
`CategoryTheory.Presieve.Arrows.PullbackCompatible`.
-/
def FamilyOfElements.PullbackCompatible (x : FamilyOfElements P R) [R.hasPullbacks] : Prop :=
  ∀ ⦃Y₁ Y₂⦄ ⦃f₁ : Y₁ ⟶ X⦄ ⦃f₂ : Y₂ ⟶ X⦄ (h₁ : R f₁) (h₂ : R f₂),
    haveI := hasPullbacks.has_pullbacks h₁ h₂
    P.map (pullback.fst f₁ f₂).op (x f₁ h₁) = P.map (pullback.snd f₁ f₂).op (x f₂ h₂)


theorem pullbackCompatible_iff (x : FamilyOfElements P R) [R.hasPullbacks] :
    x.Compatible ↔ x.PullbackCompatible := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    inst✝ : R.hasPullbacks
    ⊢ Iff x.Compatible x.PullbackCompatible
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      inst✝ : R.hasPullbacks
      ⊢ x.Compatible → x.PullbackCompatible
    -/
  · intro t Y₁ Y₂ f₁ f₂ hf₁ hf₂
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      inst✝ : R.hasPullbacks
      t : x.Compatible
      Y₁ Y₂ : C
      f₁ : Quiver.Hom Y₁ X
      f₂ : Quiver.Hom Y₂ X
      hf₁ : R f₁
      hf₂ : R f₂
      ⊢ Eq (P.map (CategoryTheory.Limits.pullback.fst f₁ f₂).op (x f₁ hf₁)) (P.map ( …
    -/
    apply t
    /-
      case mp.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      inst✝ : R.hasPullbacks
      t : x.Compatible
      Y₁ Y₂ : C
      f₁ : Quiver.Hom Y₁ X
      f₂ : Quiver.Hom Y₂ X
      hf₁ : R f₁
      hf₂ : R f₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst f …
    -/
    haveI := hasPullbacks.has_pullbacks hf₁ hf₂
    /-
      case mp.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      inst✝ : R.hasPullbacks
      t : x.Compatible
      Y₁ Y₂ : C
      f₁ : Quiver.Hom Y₁ X
      f₂ : Quiver.Hom Y₂ X
      hf₁ : R f₁
      hf₂ : R f₂
      this : CategoryTheory.Limits.HasPullback f₁ f₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst f …
    -/
    apply pullback.condition
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      inst✝ : R.hasPullbacks
      ⊢ x.PullbackCompatible → x.Compatible
    -/
  · intro t Y₁ Y₂ Z g₁ g₂ f₁ f₂ hf₁ hf₂ comm
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      inst✝ : R.hasPullbacks
      t : x.PullbackCompatible
      Y₁ Y₂ Z : C
      g₁ : Quiver.Hom Z Y₁
      g₂ : Quiver.Hom Z Y₂
      f₁ : Quiver.Hom Y₁ X
      f₂ : Quiver.Hom Y₂ X
      hf₁ : R f₁
      hf₂ : R f₂
      comm : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryS …
      ⊢ Eq (P.map g₁.op (x f₁ hf₁)) (P.map g₂.op (x f₂ hf₂))
    -/
    haveI := hasPullbacks.has_pullbacks hf₁ hf₂
    rw [← pullback.lift_fst _ _ comm, op_comp, FunctorToTypes.map_comp_apply, t hf₁ hf₂,
      ← FunctorToTypes.map_comp_apply, ← op_comp, pullback.lift_snd]


/-- The restriction of a compatible family is compatible. -/
theorem FamilyOfElements.Compatible.restrict {R₁ R₂ : Presieve X} (h : R₁ ≤ R₂)
    {x : FamilyOfElements P R₂} : x.Compatible → (x.restrict h).Compatible :=
  fun q _ _ _ g₁ g₂ _ _ h₁ h₂ comm => q g₁ g₂ (h _ h₁) (h _ h₂) comm


/-- Extend a family of elements to the sieve generated by an arrow set.
This is the construction described as "easy" in Lemma C2.1.3 of [Elephant].
-/
noncomputable def FamilyOfElements.sieveExtend (x : FamilyOfElements P R) :
    FamilyOfElements P (generate R : Presieve X) := fun _ _ hf =>
  P.map hf.choose_spec.choose.op (x _ hf.choose_spec.choose_spec.choose_spec.1)


/-- The extension of a compatible family to the generated sieve is compatible. -/
theorem FamilyOfElements.Compatible.sieveExtend {x : FamilyOfElements P R} (hx : x.Compatible) :
    x.sieveExtend.Compatible := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    hx : x.Compatible
    ⊢ x.sieveExtend.Compatible
  -/
  intro _ _ _ _ _ _ _ h₁ h₂ comm
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    hx : x.Compatible
    Y₁✝ Y₂✝ Z✝ : C
    g₁✝ : Quiver.Hom Z✝ Y₁✝
    g₂✝ : Quiver.Hom Z✝ Y₂✝
    f₁✝ : Quiver.Hom Y₁✝ X
    f₂✝ : Quiver.Hom Y₂✝ X
    h₁ : (CategoryTheory.Sieve.generate R).arrows f₁✝
    h₂ : (CategoryTheory.Sieve.generate R).arrows f₂✝
    comm : Eq (CategoryTheory.CategoryStruct.comp g₁✝ f₁✝) (CategoryTheory.Categor …
    ⊢ Eq (P.map g₁✝.op (x.sieveExtend f₁✝ h₁)) (P.map g₂✝.op (x.sieveExtend f₂✝ h₂))
  -/
  iterate 2 erw [← FunctorToTypes.map_comp_apply]; rw [← op_comp]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    hx : x.Compatible
    Y₁✝ Y₂✝ Z✝ : C
    g₁✝ : Quiver.Hom Z✝ Y₁✝
    g₂✝ : Quiver.Hom Z✝ Y₂✝
    f₁✝ : Quiver.Hom Y₁✝ X
    f₂✝ : Quiver.Hom Y₂✝ X
    h₁ : (CategoryTheory.Sieve.generate R).arrows f₁✝
    h₂ : (CategoryTheory.Sieve.generate R).arrows f₂✝
    comm : Eq (CategoryTheory.CategoryStruct.comp g₁✝ f₁✝) (CategoryTheory.Categor …
    ⊢ Eq (P.map (CategoryTheory.CategoryStruct.comp g₁✝ ⋯.choose).op (x ⋯.choose ⋯ …
  -/
  apply hx
  /-
    case a
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    hx : x.Compatible
    Y₁✝ Y₂✝ Z✝ : C
    g₁✝ : Quiver.Hom Z✝ Y₁✝
    g₂✝ : Quiver.Hom Z✝ Y₂✝
    f₁✝ : Quiver.Hom Y₁✝ X
    f₂✝ : Quiver.Hom Y₂✝ X
    h₁ : (CategoryTheory.Sieve.generate R).arrows f₁✝
    h₂ : (CategoryTheory.Sieve.generate R).arrows f₂✝
    comm : Eq (CategoryTheory.CategoryStruct.comp g₁✝ f₁✝) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
  -/
  simp [comm, h₁.choose_spec.choose_spec.choose_spec.2, h₂.choose_spec.choose_spec.choose_spec.2]
  /-
    🎉 no goals
  -/


/-- The extension of a family agrees with the original family. -/
theorem extend_agrees {x : FamilyOfElements P R} (t : x.Compatible) {f : Y ⟶ X} (hf : R f) :
    x.sieveExtend f (le_generate R Y hf) = x f hf := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : x.Compatible
    f : Quiver.Hom Y X
    hf : R f
    ⊢ Eq (x.sieveExtend f ⋯) (x f hf)
  -/
  have h := (le_generate R Y hf).choose_spec
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : x.Compatible
    f : Quiver.Hom Y X
    hf : R f
    h : Exists fun h => Exists fun g => And (R g) (Eq (CategoryTheory.CategoryStru …
    ⊢ Eq (x.sieveExtend f ⋯) (x f hf)
  -/
  unfold FamilyOfElements.sieveExtend
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : x.Compatible
    f : Quiver.Hom Y X
    hf : R f
    h : Exists fun h => Exists fun g => And (R g) (Eq (CategoryTheory.CategoryStru …
    ⊢ Eq (P.map ⋯.choose.op (x ⋯.choose ⋯)) (x f hf)
  -/
  rw [t h.choose (𝟙 _) _ hf _]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      t : x.Compatible
      f : Quiver.Hom Y X
      hf : R f
      h : Exists fun h => Exists fun g => And (R g) (Eq (CategoryTheory.CategoryStru …
      ⊢ Eq (P.map (CategoryTheory.CategoryStruct.id Y).op (x f hf)) (x f hf)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      t : x.Compatible
      f : Quiver.Hom Y X
      hf : R f
      h : Exists fun h => Exists fun g => And (R g) (Eq (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.choose ⋯.choose) (CategoryTheory.Ca …
    -/
  · rw [id_comp]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      t : x.Compatible
      f : Quiver.Hom Y X
      hf : R f
      h : Exists fun h => Exists fun g => And (R g) (Eq (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.choose ⋯.choose) f
    -/
    exact h.choose_spec.choose_spec.2
    /-
      🎉 no goals
    -/


/-- The restriction of an extension is the original. -/
@[simp]
theorem restrict_extend {x : FamilyOfElements P R} (t : x.Compatible) :
    x.sieveExtend.restrict (le_generate R) = x := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : x.Compatible
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x.sieveExtend) x
  -/
  funext Y f hf
  /-
    case h.h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : x.Compatible
    Y : C
    f : Quiver.Hom Y X
    hf : R f
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x.sieveExtend f hf)  …
  -/
  exact extend_agrees t hf
  /-
    🎉 no goals
  -/


/--
If the arrow set for a family of elements is actually a sieve (i.e. it is downward closed) then the
consistency condition can be simplified.
This is an equivalent condition, see `compatible_iff_sieveCompatible`.

This is the notion of "matching" given for families on sieves given in [MM92], Chapter III,
Section 4, Equation 1, and nlab: https://ncatlab.org/nlab/show/matching+family.
See also the discussion before Lemma C2.1.4 of [Elephant].
-/
def FamilyOfElements.SieveCompatible (x : FamilyOfElements P (S : Presieve X)) : Prop :=
  ∀ ⦃Y Z⦄ (f : Y ⟶ X) (g : Z ⟶ Y) (hf), x (g ≫ f) (S.downward_closed hf g) = P.map g.op (x f hf)


theorem compatible_iff_sieveCompatible (x : FamilyOfElements P (S : Presieve X)) :
    x.Compatible ↔ x.SieveCompatible := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    ⊢ Iff x.Compatible x.SieveCompatible
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      ⊢ x.Compatible → x.SieveCompatible
    -/
  · intro h Y Z f g hf
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      h : x.Compatible
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      hf : S.arrows f
      ⊢ Eq (x (CategoryTheory.CategoryStruct.comp g f) ⋯) (P.map g.op (x f hf))
    -/
    simpa using h (𝟙 _) g (S.downward_closed hf g) hf (id_comp _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      ⊢ x.SieveCompatible → x.Compatible
    -/
  · intro h Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ k
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      h : x.SieveCompatible
      Y₁ Y₂ Z : C
      g₁ : Quiver.Hom Z Y₁
      g₂ : Quiver.Hom Z Y₂
      f₁ : Quiver.Hom Y₁ X
      f₂ : Quiver.Hom Y₂ X
      h₁ : S.arrows f₁
      h₂ : S.arrows f₂
      k : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
      ⊢ Eq (P.map g₁.op (x f₁ h₁)) (P.map g₂.op (x f₂ h₂))
    -/
    simp_rw [← h f₁ g₁ h₁, ← h f₂ g₂ h₂]
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      h : x.SieveCompatible
      Y₁ Y₂ Z : C
      g₁ : Quiver.Hom Z Y₁
      g₂ : Quiver.Hom Z Y₂
      f₁ : Quiver.Hom Y₁ X
      f₂ : Quiver.Hom Y₂ X
      h₁ : S.arrows f₁
      h₂ : S.arrows f₂
      k : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
      ⊢ Eq (x (CategoryTheory.CategoryStruct.comp g₁ f₁) ⋯) (x (CategoryTheory.Categ …
    -/
    congr
    /-
      🎉 no goals
    -/


theorem FamilyOfElements.Compatible.to_sieveCompatible {x : FamilyOfElements P (S : Presieve X)}
    (t : x.Compatible) : x.SieveCompatible :=
  (compatible_iff_sieveCompatible x).1 t


/--
Given a family of elements `x` for the sieve `S` generated by a presieve `R`, if `x` is restricted
to `R` and then extended back up to `S`, the resulting extension equals `x`.
-/
@[simp]
theorem extend_restrict {x : FamilyOfElements P (generate R).arrows} (t : x.Compatible) :
    (x.restrict (le_generate R)).sieveExtend = x := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
    t : x.Compatible
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x).sieveExtend x
  -/
  rw [compatible_iff_sieveCompatible] at t
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
    t : x.SieveCompatible
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x).sieveExtend x
  -/
  funext _ _ h
  /-
    case h.h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
    t : x.SieveCompatible
    x✝¹ : C
    x✝ : Quiver.Hom x✝¹ X
    h : (CategoryTheory.Sieve.generate R).arrows x✝
    ⊢ Eq ((CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x).sieveExtend x✝ h …
  -/
  apply (t _ _ _).symm.trans
  /-
    case h.h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
    t : x.SieveCompatible
    x✝¹ : C
    x✝ : Quiver.Hom x✝¹ X
    h : (CategoryTheory.Sieve.generate R).arrows x✝
    ⊢ Eq (x (CategoryTheory.CategoryStruct.comp ⋯.choose ⋯.choose) ⋯) (x x✝ h)
  -/
  congr
  /-
    case h.h.h.e_f
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
    t : x.SieveCompatible
    x✝¹ : C
    x✝ : Quiver.Hom x✝¹ X
    h : (CategoryTheory.Sieve.generate R).arrows x✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ⋯.choose ⋯.choose) x✝
  -/
  exact h.choose_spec.choose_spec.choose_spec.2
  /-
    🎉 no goals
  -/


/--
Two compatible families on the sieve generated by a presieve `R` are equal if and only if they are
equal when restricted to `R`.
-/
theorem restrict_inj {x₁ x₂ : FamilyOfElements P (generate R).arrows} (t₁ : x₁.Compatible)
    (t₂ : x₂.Compatible) : x₁.restrict (le_generate R) = x₂.restrict (le_generate R) → x₁ = x₂ :=
  fun h => by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x₁ x₂ : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.gener …
    t₁ : x₁.Compatible
    t₂ : x₂.Compatible
    h : Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x₁) (CategoryTheor …
    ⊢ Eq x₁ x₂
  -/
  rw [← extend_restrict t₁, ← extend_restrict t₂]
  -- Porting note: congr fails to make progress
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x₁ x₂ : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.gener …
    t₁ : x₁.Compatible
    t₂ : x₂.Compatible
    h : Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x₁) (CategoryTheor …
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x₁).sieveExtend (Cat …
  -/
  apply congr_arg
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x₁ x₂ : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.gener …
    t₁ : x₁.Compatible
    t₂ : x₂.Compatible
    h : Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x₁) (CategoryTheor …
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x₁) (CategoryTheory. …
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- Compatible families of elements for a presheaf of types `P` and a presieve `R`
    are in 1-1 correspondence with compatible families for the same presheaf and
    the sieve generated by `R`, through extension and restriction. -/
@[simps]
noncomputable def compatibleEquivGenerateSieveCompatible :
    { x : FamilyOfElements P R // x.Compatible } ≃
      { x : FamilyOfElements P (generate R : Presieve X) // x.Compatible } where
  toFun x := ⟨x.1.sieveExtend, x.2.sieveExtend⟩
  invFun x := ⟨x.1.restrict (le_generate R), x.2.restrict _⟩
  left_inv x := Subtype.ext (restrict_extend x.2)
  right_inv x := Subtype.ext (extend_restrict x.2)


theorem FamilyOfElements.comp_of_compatible (S : Sieve X) {x : FamilyOfElements P S}
    (t : x.Compatible) {f : Y ⟶ X} (hf : S f) {Z} (g : Z ⟶ Y) :
    x (g ≫ f) (S.downward_closed hf g) = P.map g.op (x f hf) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    t : x.Compatible
    f : Quiver.Hom Y X
    hf : S.arrows f
    Z : C
    g : Quiver.Hom Z Y
    ⊢ Eq (x (CategoryTheory.CategoryStruct.comp g f) ⋯) (P.map g.op (x f hf))
  -/
  simpa using t (𝟙 _) g (S.downward_closed hf g) hf (id_comp _)
  /-
    🎉 no goals
  -/


/--
Given a family of elements of a sieve `S` on `F(X)`, we can realize it as a family of elements of
`S.functorPullback F`.
-/
def FamilyOfElements.functorPullback (x : FamilyOfElements P T) :
    FamilyOfElements (F.op ⋙ P) (T.functorPullback F) := fun _ f hf => x (F.map f) hf


theorem FamilyOfElements.Compatible.functorPullback (h : x.Compatible) :
    (x.functorPullback F).Compatible := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    Z : D
    T : CategoryTheory.Presieve (F.obj Z)
    x : CategoryTheory.Presieve.FamilyOfElements P T
    h : x.Compatible
    ⊢ (CategoryTheory.Presieve.FamilyOfElements.functorPullback F x).Compatible
  -/
  intro Z₁ Z₂ W g₁ g₂ f₁ f₂ h₁ h₂ eq
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    Z : D
    T : CategoryTheory.Presieve (F.obj Z)
    x : CategoryTheory.Presieve.FamilyOfElements P T
    h : x.Compatible
    Z₁ Z₂ W : D
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    f₁ : Quiver.Hom Z₁ Z
    f₂ : Quiver.Hom Z₂ Z
    h₁ : CategoryTheory.Presieve.functorPullback F T f₁
    h₂ : CategoryTheory.Presieve.functorPullback F T f₂
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStr …
    ⊢ Eq ((F.op.comp P).map g₁.op (CategoryTheory.Presieve.FamilyOfElements.functo …
  -/
  exact h (F.map g₁) (F.map g₂) h₁ h₂ (by simp only [← F.map_comp, eq])
  /-
    🎉 no goals
  -/


/-- Given a family of elements of a sieve `S` on `X` whose values factors through `F`, we can
realize it as a family of elements of `S.functorPushforward F`. Since the preimage is obtained by
choice, this is not well-defined generally.
-/
noncomputable def FamilyOfElements.functorPushforward {D : Type u₂} [Category.{v₂} D] (F : D ⥤ C)
    {X : D} {T : Presieve X} (x : FamilyOfElements (F.op ⋙ P) T) :
    FamilyOfElements P (T.functorPushforward F) := fun Y f h => by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P Q U : CategoryTheory.Functor (Opposite C) (Type w)
    X✝ Y✝ : C
    S : CategoryTheory.Sieve X✝
    R : CategoryTheory.Presieve X✝
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    X : D
    T : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp P) T
    Y : C
    f : Quiver.Hom Y (F.obj X)
    h : CategoryTheory.Presieve.functorPushforward F T f
    ⊢ P.obj { unop := Y }
  -/
  obtain ⟨Z, g, h, h₁, _⟩ := getFunctorPushforwardStructure h
  /-
    case mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P Q U : CategoryTheory.Functor (Opposite C) (Type w)
    X✝ Y✝ : C
    S : CategoryTheory.Sieve X✝
    R : CategoryTheory.Presieve X✝
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor D C
    X : D
    T : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp P) T
    Y : C
    f : Quiver.Hom Y (F.obj X)
    h✝ : CategoryTheory.Presieve.functorPushforward F T f
    Z : D
    g : Quiver.Hom Z X
    h : Quiver.Hom Y (F.obj Z)
    h₁ : T g
    fac✝ : Eq f (CategoryTheory.CategoryStruct.comp h (F.map g))
    ⊢ P.obj { unop := Y }
  -/
  exact P.map h.op (x g h₁)
  /-
    🎉 no goals
  -/


/-- Given a family of elements of a sieve `S` on `X`, and a map `Y ⟶ X`, we can obtain a
family of elements of `S.pullback f` by taking the same elements.
-/
def FamilyOfElements.pullback (f : Y ⟶ X) (x : FamilyOfElements P (S : Presieve X)) :
    FamilyOfElements P (S.pullback f : Presieve Y) := fun _ g hg => x (g ≫ f) hg


theorem FamilyOfElements.Compatible.pullback (f : Y ⟶ X) {x : FamilyOfElements P S.arrows}
    (h : x.Compatible) : (x.pullback f).Compatible := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    h : x.Compatible
    ⊢ (CategoryTheory.Presieve.FamilyOfElements.pullback f x).Compatible
  -/
  simp only [compatible_iff_sieveCompatible] at h ⊢
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    h : x.SieveCompatible
    ⊢ (CategoryTheory.Presieve.FamilyOfElements.pullback f x).SieveCompatible
  -/
  intro W Z f₁ f₂ hf
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    h : x.SieveCompatible
    W Z : C
    f₁ : Quiver.Hom W Y
    f₂ : Quiver.Hom Z W
    hf : (CategoryTheory.Sieve.pullback f S).arrows f₁
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.pullback f x (CategoryTheory.Ca …
  -/
  unfold FamilyOfElements.pullback
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    h : x.SieveCompatible
    W Z : C
    f₁ : Quiver.Hom W Y
    f₂ : Quiver.Hom Z W
    hf : (CategoryTheory.Sieve.pullback f S).arrows f₁
    ⊢ Eq (x (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
  -/
  rw [← h (f₁ ≫ f) f₂ hf]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    h : x.SieveCompatible
    W Z : C
    f₁ : Quiver.Hom W Y
    f₂ : Quiver.Hom Z W
    hf : (CategoryTheory.Sieve.pullback f S).arrows f₁
    ⊢ Eq (x (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
  -/
  congr 1
  /-
    case e_f
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X Y : C
    S : CategoryTheory.Sieve X
    f : Quiver.Hom Y X
    x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
    h : x.SieveCompatible
    W Z : C
    f₁ : Quiver.Hom W Y
    f₂ : Quiver.Hom Z W
    hf : (CategoryTheory.Sieve.pullback f S).arrows f₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  simp only [assoc]
  /-
    🎉 no goals
  -/


/-- Given a morphism of presheaves `f : P ⟶ Q`, we can take a family of elements valued in `P` to a
family of elements valued in `Q` by composing with `f`.
-/
def FamilyOfElements.compPresheafMap (f : P ⟶ Q) (x : FamilyOfElements P R) :
    FamilyOfElements Q R := fun Y g hg => f.app (op Y) (x g hg)


@[simp]
theorem FamilyOfElements.compPresheafMap_id (x : FamilyOfElements P R) :
    x.compPresheafMap (𝟙 P) = x :=
  rfl


@[simp]
theorem FamilyOfElements.compPresheafMap_comp (x : FamilyOfElements P R) (f : P ⟶ Q)
    (g : Q ⟶ U) : (x.compPresheafMap f).compPresheafMap g = x.compPresheafMap (f ≫ g) :=
  rfl


theorem FamilyOfElements.Compatible.compPresheafMap (f : P ⟶ Q) {x : FamilyOfElements P R}
    (h : x.Compatible) : (x.compPresheafMap f).Compatible := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    f : Quiver.Hom P Q
    x : CategoryTheory.Presieve.FamilyOfElements P R
    h : x.Compatible
    ⊢ (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f x).Compatible
  -/
  intro Z₁ Z₂ W g₁ g₂ f₁ f₂ h₁ h₂ eq
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    f : Quiver.Hom P Q
    x : CategoryTheory.Presieve.FamilyOfElements P R
    h : x.Compatible
    Z₁ Z₂ W : C
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    f₁ : Quiver.Hom Z₁ X
    f₂ : Quiver.Hom Z₂ X
    h₁ : R f₁
    h₂ : R f₂
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStr …
    ⊢ Eq (Q.map g₁.op (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f  …
  -/
  unfold FamilyOfElements.compPresheafMap
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    f : Quiver.Hom P Q
    x : CategoryTheory.Presieve.FamilyOfElements P R
    h : x.Compatible
    Z₁ Z₂ W : C
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    f₁ : Quiver.Hom Z₁ X
    f₂ : Quiver.Hom Z₂ X
    h₁ : R f₁
    h₂ : R f₂
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStr …
    ⊢ Eq (Q.map g₁.op (f.app { unop := Z₁ } (x f₁ h₁))) (Q.map g₂.op (f.app { unop …
  -/
  rwa [← FunctorToTypes.naturality, ← FunctorToTypes.naturality, h]
  /-
    🎉 no goals
  -/


/--
The given element `t` of `P.obj (op X)` is an *amalgamation* for the family of elements `x` if every
restriction `P.map f.op t = x_f` for every arrow `f` in the presieve `R`.

This is the definition given in https://ncatlab.org/nlab/show/sheaf#GeneralDefinitionInComponents,
and https://ncatlab.org/nlab/show/matching+family, as well as [MM92], Chapter III, Section 4,
equation (2).
-/
def FamilyOfElements.IsAmalgamation (x : FamilyOfElements P R) (t : P.obj (op X)) : Prop :=
  ∀ ⦃Y : C⦄ (f : Y ⟶ X) (h : R f), P.map f.op t = x f h


theorem FamilyOfElements.IsAmalgamation.compPresheafMap {x : FamilyOfElements P R} {t} (f : P ⟶ Q)
    (h : x.IsAmalgamation t) : (x.compPresheafMap f).IsAmalgamation (f.app (op X) t) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    f : Quiver.Hom P Q
    h : x.IsAmalgamation t
    ⊢ (CategoryTheory.Presieve.FamilyOfElements.compPresheafMap f x).IsAmalgamatio …
  -/
  intro Y g hg
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    f : Quiver.Hom P Q
    h : x.IsAmalgamation t
    Y : C
    g : Quiver.Hom Y X
    hg : R g
    ⊢ Eq (Q.map g.op (f.app { unop := X } t)) (CategoryTheory.Presieve.FamilyOfEle …
  -/
  dsimp [FamilyOfElements.compPresheafMap]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    f : Quiver.Hom P Q
    h : x.IsAmalgamation t
    Y : C
    g : Quiver.Hom Y X
    hg : R g
    ⊢ Eq (Q.map g.op (f.app { unop := X } t)) (f.app { unop := Y } (x g hg))
  -/
  change (f.app _ ≫ Q.map _) _ = _
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P Q : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    f : Quiver.Hom P Q
    h : x.IsAmalgamation t
    Y : C
    g : Quiver.Hom Y X
    hg : R g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := X }) (Q.map g.op) t) …
  -/
  rw [← f.naturality, types_comp_apply, h g hg]
  /-
    🎉 no goals
  -/


theorem is_compatible_of_exists_amalgamation (x : FamilyOfElements P R)
    (h : ∃ t, x.IsAmalgamation t) : x.Compatible := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    h : Exists fun t => x.IsAmalgamation t
    ⊢ x.Compatible
  -/
  cases' h with t ht
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    ht : x.IsAmalgamation t
    ⊢ x.Compatible
  -/
  intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ comm
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    ht : x.IsAmalgamation t
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : R f₁
    h₂ : R f₂
    comm : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryS …
    ⊢ Eq (P.map g₁.op (x f₁ h₁)) (P.map g₂.op (x f₂ h₂))
  -/
  rw [← ht _ h₁, ← ht _ h₂, ← FunctorToTypes.map_comp_apply, ← op_comp, comm]
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    ht : x.IsAmalgamation t
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : R f₁
    h₂ : R f₂
    comm : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryS …
    ⊢ Eq (P.map (CategoryTheory.CategoryStruct.comp g₂ f₂).op t) (P.map g₂.op (P.m …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isAmalgamation_restrict {R₁ R₂ : Presieve X} (h : R₁ ≤ R₂) (x : FamilyOfElements P R₂)
    (t : P.obj (op X)) (ht : x.IsAmalgamation t) : (x.restrict h).IsAmalgamation t := fun Y f hf =>
  ht f (h Y hf)


theorem isAmalgamation_sieveExtend {R : Presieve X} (x : FamilyOfElements P R) (t : P.obj (op X))
    (ht : x.IsAmalgamation t) : x.sieveExtend.IsAmalgamation t := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    ht : x.IsAmalgamation t
    ⊢ x.sieveExtend.IsAmalgamation t
  -/
  intro Y f hf
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    ht : x.IsAmalgamation t
    Y : C
    f : Quiver.Hom Y X
    hf : (CategoryTheory.Sieve.generate R).arrows f
    ⊢ Eq (P.map f.op t) (x.sieveExtend f hf)
  -/
  dsimp [FamilyOfElements.sieveExtend]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    t : P.obj { unop := X }
    ht : x.IsAmalgamation t
    Y : C
    f : Quiver.Hom Y X
    hf : (CategoryTheory.Sieve.generate R).arrows f
    ⊢ Eq (P.map f.op t) (P.map ⋯.choose.op (x ⋯.choose ⋯))
  -/
  rw [← ht _, ← FunctorToTypes.map_comp_apply, ← op_comp, hf.choose_spec.choose_spec.choose_spec.2]
  /-
    🎉 no goals
  -/


/-- A presheaf is separated for a presieve if there is at most one amalgamation. -/
def IsSeparatedFor (P : Cᵒᵖ ⥤ Type w) (R : Presieve X) : Prop :=
  ∀ (x : FamilyOfElements P R) (t₁ t₂), x.IsAmalgamation t₁ → x.IsAmalgamation t₂ → t₁ = t₂


theorem IsSeparatedFor.ext {R : Presieve X} (hR : IsSeparatedFor P R) {t₁ t₂ : P.obj (op X)}
    (h : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄ (_ : R f), P.map f.op t₁ = P.map f.op t₂) : t₁ = t₂ :=
  hR (fun _ f _ => P.map f.op t₂) t₁ t₂ (fun _ _ hf => h hf) fun _ _ _ => rfl


theorem isSeparatedFor_iff_generate :
    IsSeparatedFor P R ↔ IsSeparatedFor P (generate R : Presieve X) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (CategoryTheory.Presieve.IsSeparatedFor P R) (CategoryTheory.Presieve.Is …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P R → CategoryTheory.Presieve.IsSepar …
    -/
  · intro h x t₁ t₂ ht₁ ht₂
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      h : CategoryTheory.Presieve.IsSeparatedFor P R
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
      t₁ t₂ : P.obj { unop := X }
      ht₁ : x.IsAmalgamation t₁
      ht₂ : x.IsAmalgamation t₂
      ⊢ Eq t₁ t₂
    -/
    apply h (x.restrict (le_generate R)) t₁ t₂ _ _
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        X : C
        R : CategoryTheory.Presieve X
        h : CategoryTheory.Presieve.IsSeparatedFor P R
        x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
        t₁ t₂ : P.obj { unop := X }
        ht₁ : x.IsAmalgamation t₁
        ht₂ : x.IsAmalgamation t₂
        ⊢ (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x).IsAmalgamation t₁
      -/
    · exact isAmalgamation_restrict _ x t₁ ht₁
      /-
        🎉 no goals
      -/
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        X : C
        R : CategoryTheory.Presieve X
        h : CategoryTheory.Presieve.IsSeparatedFor P R
        x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
        t₁ t₂ : P.obj { unop := X }
        ht₁ : x.IsAmalgamation t₁
        ht₂ : x.IsAmalgamation t₂
        ⊢ (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x).IsAmalgamation t₂
      -/
    · exact isAmalgamation_restrict _ x t₂ ht₂
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.generate R).a …
    -/
  · intro h x t₁ t₂ ht₁ ht₂
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      h : CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.generate R) …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      t₁ t₂ : P.obj { unop := X }
      ht₁ : x.IsAmalgamation t₁
      ht₂ : x.IsAmalgamation t₂
      ⊢ Eq t₁ t₂
    -/
    apply h x.sieveExtend
      /-
        case mpr.a
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        X : C
        R : CategoryTheory.Presieve X
        h : CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.generate R) …
        x : CategoryTheory.Presieve.FamilyOfElements P R
        t₁ t₂ : P.obj { unop := X }
        ht₁ : x.IsAmalgamation t₁
        ht₂ : x.IsAmalgamation t₂
        ⊢ x.sieveExtend.IsAmalgamation t₁
      -/
    · exact isAmalgamation_sieveExtend x t₁ ht₁
      /-
        🎉 no goals
      -/
      /-
        case mpr.a
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        X : C
        R : CategoryTheory.Presieve X
        h : CategoryTheory.Presieve.IsSeparatedFor P (CategoryTheory.Sieve.generate R) …
        x : CategoryTheory.Presieve.FamilyOfElements P R
        t₁ t₂ : P.obj { unop := X }
        ht₁ : x.IsAmalgamation t₁
        ht₂ : x.IsAmalgamation t₂
        ⊢ x.sieveExtend.IsAmalgamation t₂
      -/
    · exact isAmalgamation_sieveExtend x t₂ ht₂
      /-
        🎉 no goals
      -/


theorem isSeparatedFor_top (P : Cᵒᵖ ⥤ Type w) : IsSeparatedFor P (⊤ : Presieve X) :=
  fun x t₁ t₂ h₁ h₂ => by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    x : CategoryTheory.Presieve.FamilyOfElements P Top.top
    t₁ t₂ : P.obj { unop := X }
    h₁ : x.IsAmalgamation t₁
    h₂ : x.IsAmalgamation t₂
    ⊢ Eq t₁ t₂
  -/
  have q₁ := h₁ (𝟙 X) (by tauto)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    x : CategoryTheory.Presieve.FamilyOfElements P Top.top
    t₁ t₂ : P.obj { unop := X }
    h₁ : x.IsAmalgamation t₁
    h₂ : x.IsAmalgamation t₂
    q₁ : Eq (P.map (CategoryTheory.CategoryStruct.id X).op t₁) (x (CategoryTheory. …
    ⊢ Eq t₁ t₂
  -/
  have q₂ := h₂ (𝟙 X) (by tauto)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    x : CategoryTheory.Presieve.FamilyOfElements P Top.top
    t₁ t₂ : P.obj { unop := X }
    h₁ : x.IsAmalgamation t₁
    h₂ : x.IsAmalgamation t₂
    q₁ : Eq (P.map (CategoryTheory.CategoryStruct.id X).op t₁) (x (CategoryTheory. …
    q₂ : Eq (P.map (CategoryTheory.CategoryStruct.id X).op t₂) (x (CategoryTheory. …
    ⊢ Eq t₁ t₂
  -/
  simp only [op_id, FunctorToTypes.map_id_apply] at q₁ q₂
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    x : CategoryTheory.Presieve.FamilyOfElements P Top.top
    t₁ t₂ : P.obj { unop := X }
    h₁ : x.IsAmalgamation t₁
    h₂ : x.IsAmalgamation t₂
    q₁ : Eq t₁ (x (CategoryTheory.CategoryStruct.id X) trivial)
    q₂ : Eq t₂ (x (CategoryTheory.CategoryStruct.id X) trivial)
    ⊢ Eq t₁ t₂
  -/
  rw [q₁, q₂]
  /-
    🎉 no goals
  -/


/-- We define `P` to be a sheaf for the presieve `R` if every compatible family has a unique
amalgamation.

This is the definition of a sheaf for the given presieve given in C2.1.2 of [Elephant], and
https://ncatlab.org/nlab/show/sheaf#GeneralDefinitionInComponents.
Using `compatible_iff_sieveCompatible`,
this is equivalent to the definition of a sheaf in [MM92], Chapter III, Section 4.
-/
def IsSheafFor (P : Cᵒᵖ ⥤ Type w) (R : Presieve X) : Prop :=
  ∀ x : FamilyOfElements P R, x.Compatible → ∃! t, x.IsAmalgamation t


/-- This is an equivalent condition to be a sheaf, which is useful for the abstraction to local
operators on elementary toposes. However this definition is defined only for sieves, not presieves.
The equivalence between this and `IsSheafFor` is given in `isSheafFor_iff_yonedaSheafCondition`.
This version is also useful to establish that being a sheaf is preserved under isomorphism of
presheaves.

See the discussion before Equation (3) of [MM92], Chapter III, Section 4. See also C2.1.4 of
[Elephant]. This is also a direct reformulation of <https://stacks.math.columbia.edu/tag/00Z8>.
-/
def YonedaSheafCondition (P : Cᵒᵖ ⥤ Type v₁) (S : Sieve X) : Prop :=
  ∀ f : S.functor ⟶ P, ∃! g, S.functorInclusion ≫ g = f

-- TODO: We can generalize the universe parameter v₁ above by composing with
-- appropriate `ulift_functor`s.

/-- (Implementation). This is a (primarily internal) equivalence between natural transformations
and compatible families.

Cf the discussion after Lemma 7.47.10 in <https://stacks.math.columbia.edu/tag/00YW>. See also
the proof of C2.1.4 of [Elephant], and the discussion in [MM92], Chapter III, Section 4.
-/
def natTransEquivCompatibleFamily {P : Cᵒᵖ ⥤ Type v₁} :
    (S.functor ⟶ P) ≃ { x : FamilyOfElements P (S : Presieve X) // x.Compatible } where
  toFun α := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      α : Quiver.Hom S.functor P
      ⊢ Subtype fun x => x.Compatible
    -/
    refine ⟨fun Y f hf => ?_, ?_⟩
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
        X Y✝ : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        P : CategoryTheory.Functor (Opposite C) (Type v₁)
        α : Quiver.Hom S.functor P
        Y : C
        f : Quiver.Hom Y X
        hf : S.arrows f
        ⊢ P.obj { unop := Y }
      -/
    · apply α.app (op Y) ⟨_, hf⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
        X Y : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        P : CategoryTheory.Functor (Opposite C) (Type v₁)
        α : Quiver.Hom S.functor P
        ⊢ CategoryTheory.Presieve.FamilyOfElements.Compatible fun Y f hf => α.app { un …
      -/
    · rw [compatible_iff_sieveCompatible]
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
        X Y : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        P : CategoryTheory.Functor (Opposite C) (Type v₁)
        α : Quiver.Hom S.functor P
        ⊢ CategoryTheory.Presieve.FamilyOfElements.SieveCompatible fun Y f hf => α.app …
      -/
      intro Y Z f g hf
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
        X Y✝ : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        P : CategoryTheory.Functor (Opposite C) (Type v₁)
        α : Quiver.Hom S.functor P
        Y Z : C
        f : Quiver.Hom Y X
        g : Quiver.Hom Z Y
        hf : S.arrows f
        ⊢ Eq ((fun Y f hf => α.app { unop := Y } ⟨f, hf⟩) Z (CategoryTheory.CategorySt …
      -/
      dsimp
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
        X Y✝ : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        P : CategoryTheory.Functor (Opposite C) (Type v₁)
        α : Quiver.Hom S.functor P
        Y Z : C
        f : Quiver.Hom Y X
        g : Quiver.Hom Z Y
        hf : S.arrows f
        ⊢ Eq (α.app { unop := Z } ⟨CategoryTheory.CategoryStruct.comp g f, ⋯⟩) (P.map  …
      -/
      rw [← FunctorToTypes.naturality _ _ α g.op]
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
        X Y✝ : C
        S : CategoryTheory.Sieve X
        R : CategoryTheory.Presieve X
        P : CategoryTheory.Functor (Opposite C) (Type v₁)
        α : Quiver.Hom S.functor P
        Y Z : C
        f : Quiver.Hom Y X
        g : Quiver.Hom Z Y
        hf : S.arrows f
        ⊢ Eq (α.app { unop := Z } ⟨CategoryTheory.CategoryStruct.comp g f, ⋯⟩) (α.app  …
      -/
      rfl
      /-
        🎉 no goals
      -/
  invFun t :=
    { app := fun _ f => t.1 _ f.2
      naturality := fun Y Z g => by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
          X Y✝ : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          t : Subtype fun x => x.Compatible
          Y Z : Opposite C
          g : Quiver.Hom Y Z
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.functor.map g) ((fun x f => ↑t ↑f  …
        -/
        ext ⟨f, hf⟩
        /-
          case h.mk
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
          X Y✝ : C
          S : CategoryTheory.Sieve X
          R : CategoryTheory.Presieve X
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          t : Subtype fun x => x.Compatible
          Y Z : Opposite C
          g : Quiver.Hom Y Z
          f : Quiver.Hom (Opposite.unop Y) X
          hf : S.arrows f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.functor.map g) ((fun x f => ↑t ↑f  …
        -/
        apply t.2.to_sieveCompatible _ }
        /-
          🎉 no goals
        -/
  left_inv α := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      α : Quiver.Hom S.functor P
      ⊢ Eq ((fun t => { app := fun x f => ↑t ↑f ⋯, naturality := ⋯ }) ((fun α => ⟨fu …
    -/
    ext X ⟨_, _⟩
    /-
      case w.h.h.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
      X✝ Y : C
      S : CategoryTheory.Sieve X✝
      R : CategoryTheory.Presieve X✝
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      α : Quiver.Hom S.functor P
      X : Opposite C
      val✝ : Quiver.Hom (Opposite.unop X) X✝
      property✝ : S.arrows val✝
      ⊢ Eq (((fun t => { app := fun x f => ↑t ↑f ⋯, naturality := ⋯ }) ((fun α => ⟨f …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      ⊢ Function.RightInverse (fun t => { app := fun x f => ↑t ↑f ⋯, naturality := ⋯ …
    -/
    rintro ⟨x, hx⟩
    /-
      case mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P✝ Q U : CategoryTheory.Functor (Opposite C) (Type w)
      X Y : C
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : CategoryTheory.Presieve.FamilyOfElements P S.arrows
      hx : x.Compatible
      ⊢ Eq ((fun α => ⟨fun Y f hf => α.app { unop := Y } ⟨f, hf⟩, ⋯⟩) ((fun t => { a …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- (Implementation). A lemma useful to prove `isSheafFor_iff_yonedaSheafCondition`. -/
theorem extension_iff_amalgamation {P : Cᵒᵖ ⥤ Type v₁} (x : S.functor ⟶ P) (g : yoneda.obj X ⟶ P) :
    S.functorInclusion ≫ g = x ↔
      (natTransEquivCompatibleFamily x).1.IsAmalgamation (yonedaEquiv g) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Quiver.Hom S.functor P
    g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp S.functorInclusion g) x) ((↑(Cat …
  -/
  change _ ↔ ∀ ⦃Y : C⦄ (f : Y ⟶ X) (h : S f), P.map f.op (yonedaEquiv g) = x.app (op Y) ⟨f, h⟩
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Quiver.Hom S.functor P
    g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp S.functorInclusion g) x) (∀ ⦃Y : …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      ⊢ Eq (CategoryTheory.CategoryStruct.comp S.functorInclusion g) x → ∀ ⦃Y : C⦄ ( …
    -/
  · rintro rfl Y f hf
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (P.map f.op (CategoryTheory.yonedaEquiv g)) ((CategoryTheory.CategoryStru …
    -/
    rw [yonedaEquiv_naturality]
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
    -/
    dsimp
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
    -/
    simp [yonedaEquiv_apply]
    /-
      🎉 no goals
    -/
  -- See note [dsimp, simp].
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      ⊢ (∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (h : S.arrows f), Eq (P.map f.op (CategoryTh …
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      h : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (h : S.arrows f), Eq (P.map f.op (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp S.functorInclusion g) x
    -/
    ext Y ⟨f, hf⟩
    /-
      case mpr.w.h.h.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      h : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (h : S.arrows f), Eq (P.map f.op (CategoryT …
      Y : Opposite C
      f : Quiver.Hom (Opposite.unop Y) X
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp S.functorInclusion g).app Y ⟨f, hf⟩) …
    -/
    convert h f hf
    /-
      case h.e'_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      h : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (h : S.arrows f), Eq (P.map f.op (CategoryT …
      Y : Opposite C
      f : Quiver.Hom (Opposite.unop Y) X
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp S.functorInclusion g).app Y ⟨f, hf⟩) …
    -/
    rw [yonedaEquiv_naturality]
    /-
      case h.e'_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      h : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (h : S.arrows f), Eq (P.map f.op (CategoryT …
      Y : Opposite C
      f : Quiver.Hom (Opposite.unop Y) X
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp S.functorInclusion g).app Y ⟨f, hf⟩) …
    -/
    dsimp [yonedaEquiv]
    /-
      case h.e'_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      S : CategoryTheory.Sieve X
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      x : Quiver.Hom S.functor P
      g : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      h : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (h : S.arrows f), Eq (P.map f.op (CategoryT …
      Y : Opposite C
      f : Quiver.Hom (Opposite.unop Y) X
      hf : S.arrows f
      ⊢ Eq (g.app Y f) (g.app Y (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The yoneda version of the sheaf condition is equivalent to the sheaf condition.

C2.1.4 of [Elephant].
-/
theorem isSheafFor_iff_yonedaSheafCondition {P : Cᵒᵖ ⥤ Type v₁} :
    IsSheafFor P (S : Presieve X) ↔ YonedaSheafCondition P S := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P S.arrows) (CategoryTheory.Presieve …
  -/
  rw [IsSheafFor, YonedaSheafCondition]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Iff (∀ (x : CategoryTheory.Presieve.FamilyOfElements P S.arrows), x.Compatib …
  -/
  simp_rw [extension_iff_amalgamation]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Iff (∀ (x : CategoryTheory.Presieve.FamilyOfElements P S.arrows), x.Compatib …
  -/
  rw [Equiv.forall_congr_left natTransEquivCompatibleFamily]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Iff (∀ (x : CategoryTheory.Presieve.FamilyOfElements P S.arrows), x.Compatib …
  -/
  rw [Subtype.forall]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    S : CategoryTheory.Sieve X
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Iff (∀ (x : CategoryTheory.Presieve.FamilyOfElements P S.arrows), x.Compatib …
  -/
  exact forall₂_congr fun x hx ↦ by simp [Equiv.existsUnique_congr_right]
  /-
    🎉 no goals
  -/


/--
If `P` is a sheaf for the sieve `S` on `X`, a natural transformation from `S` (viewed as a functor)
to `P` can be (uniquely) extended to all of `yoneda.obj X`.

      f
   S  →  P
   ↓  ↗
   yX

-/
noncomputable def IsSheafFor.extend {P : Cᵒᵖ ⥤ Type v₁} (h : IsSheafFor P (S : Presieve X))
    (f : S.functor ⟶ P) : yoneda.obj X ⟶ P :=
  (isSheafFor_iff_yonedaSheafCondition.1 h f).exists.choose


/--
Show that the extension of `f : S.functor ⟶ P` to all of `yoneda.obj X` is in fact an extension, ie
that the triangle below commutes, provided `P` is a sheaf for `S`

      f
   S  →  P
   ↓  ↗
   yX

-/
@[reassoc (attr := simp)]
theorem IsSheafFor.functorInclusion_comp_extend {P : Cᵒᵖ ⥤ Type v₁} (h : IsSheafFor P S.arrows)
    (f : S.functor ⟶ P) : S.functorInclusion ≫ h.extend f = f :=
  (isSheafFor_iff_yonedaSheafCondition.1 h f).exists.choose_spec


/-- The extension of `f` to `yoneda.obj X` is unique. -/
theorem IsSheafFor.unique_extend {P : Cᵒᵖ ⥤ Type v₁} (h : IsSheafFor P S.arrows) {f : S.functor ⟶ P}
    (t : yoneda.obj X ⟶ P) (ht : S.functorInclusion ≫ t = f) : t = h.extend f :=
  (isSheafFor_iff_yonedaSheafCondition.1 h f).unique ht (h.functorInclusion_comp_extend f)


/--
If `P` is a sheaf for the sieve `S` on `X`, then if two natural transformations from `yoneda.obj X`
to `P` agree when restricted to the subfunctor given by `S`, they are equal.
-/
theorem IsSheafFor.hom_ext {P : Cᵒᵖ ⥤ Type v₁} (h : IsSheafFor P (S : Presieve X))
    (t₁ t₂ : yoneda.obj X ⟶ P) (ht : S.functorInclusion ≫ t₁ = S.functorInclusion ≫ t₂) :
    t₁ = t₂ :=
  (h.unique_extend t₁ ht).trans (h.unique_extend t₂ rfl).symm


/-- `P` is a sheaf for `R` iff it is separated for `R` and there exists an amalgamation. -/
theorem isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor :
    (IsSeparatedFor P R ∧ ∀ x : FamilyOfElements P R, x.Compatible → ∃ t, x.IsAmalgamation t) ↔
      IsSheafFor P R := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (And (CategoryTheory.Presieve.IsSeparatedFor P R) (∀ (x : CategoryTheory …
  -/
  rw [IsSeparatedFor, ← forall_and]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), And (∀ (t₁ t₂ : P …
  -/
  apply forall_congr'
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ ∀ (a : CategoryTheory.Presieve.FamilyOfElements P R), Iff (And (∀ (t₁ t₂ : P …
  -/
  intro x
  /-
    case h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    x : CategoryTheory.Presieve.FamilyOfElements P R
    ⊢ Iff (And (∀ (t₁ t₂ : P.obj { unop := X }), x.IsAmalgamation t₁ → x.IsAmalgam …
  -/
  constructor
    /-
      case h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      ⊢ And (∀ (t₁ t₂ : P.obj { unop := X }), x.IsAmalgamation t₁ → x.IsAmalgamation …
    -/
  · intro z hx
    /-
      case h.mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      z : And (∀ (t₁ t₂ : P.obj { unop := X }), x.IsAmalgamation t₁ → x.IsAmalgamati …
      hx : x.Compatible
      ⊢ ExistsUnique fun t => x.IsAmalgamation t
    -/
    exact existsUnique_of_exists_of_unique (z.2 hx) z.1
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      ⊢ (x.Compatible → ExistsUnique fun t => x.IsAmalgamation t) → And (∀ (t₁ t₂ :  …
    -/
  · intro h
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      h : x.Compatible → ExistsUnique fun t => x.IsAmalgamation t
      ⊢ And (∀ (t₁ t₂ : P.obj { unop := X }), x.IsAmalgamation t₁ → x.IsAmalgamation …
    -/
    refine ⟨?_, ExistsUnique.exists ∘ h⟩
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      h : x.Compatible → ExistsUnique fun t => x.IsAmalgamation t
      ⊢ ∀ (t₁ t₂ : P.obj { unop := X }), x.IsAmalgamation t₁ → x.IsAmalgamation t₂ → …
    -/
    intro t₁ t₂ ht₁ ht₂
    /-
      case h.mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      h : x.Compatible → ExistsUnique fun t => x.IsAmalgamation t
      t₁ t₂ : P.obj { unop := X }
      ht₁ : x.IsAmalgamation t₁
      ht₂ : x.IsAmalgamation t₂
      ⊢ Eq t₁ t₂
    -/
    apply (h _).unique ht₁ ht₂
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      x : CategoryTheory.Presieve.FamilyOfElements P R
      h : x.Compatible → ExistsUnique fun t => x.IsAmalgamation t
      t₁ t₂ : P.obj { unop := X }
      ht₁ : x.IsAmalgamation t₁
      ht₂ : x.IsAmalgamation t₂
      ⊢ x.Compatible
    -/
    exact is_compatible_of_exists_amalgamation x ⟨_, ht₂⟩
    /-
      🎉 no goals
    -/


/-- If `P` is separated for `R` and every family has an amalgamation, then `P` is a sheaf for `R`.
-/
theorem IsSeparatedFor.isSheafFor (t : IsSeparatedFor P R) :
    (∀ x : FamilyOfElements P R, x.Compatible → ∃ t, x.IsAmalgamation t) → IsSheafFor P R := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    t : CategoryTheory.Presieve.IsSeparatedFor P R
    ⊢ (∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exists …
  -/
  rw [← isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    t : CategoryTheory.Presieve.IsSeparatedFor P R
    ⊢ (∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exists …
  -/
  exact And.intro t
  /-
    🎉 no goals
  -/


/-- If `P` is a sheaf for `R`, it is separated for `R`. -/
theorem IsSheafFor.isSeparatedFor : IsSheafFor P R → IsSeparatedFor P R := fun q =>
  (isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor.2 q).1


/-- Get the amalgamation of the given compatible family, provided we have a sheaf. -/
noncomputable def IsSheafFor.amalgamate (t : IsSheafFor P R) (x : FamilyOfElements P R)
    (hx : x.Compatible) : P.obj (op X) :=
  (t x hx).exists.choose


theorem IsSheafFor.isAmalgamation (t : IsSheafFor P R) {x : FamilyOfElements P R}
    (hx : x.Compatible) : x.IsAmalgamation (t.amalgamate x hx) :=
  (t x hx).exists.choose_spec


@[simp]
theorem IsSheafFor.valid_glue (t : IsSheafFor P R) {x : FamilyOfElements P R} (hx : x.Compatible)
    (f : Y ⟶ X) (Hf : R f) : P.map f.op (t.amalgamate x hx) = x f Hf :=
  t.isAmalgamation hx f Hf


/-- C2.1.3 in [Elephant] -/
theorem isSheafFor_iff_generate (R : Presieve X) :
    IsSheafFor P R ↔ IsSheafFor P (generate R : Presieve X) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P R) (CategoryTheory.Presieve.IsShea …
  -/
  rw [← isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (And (CategoryTheory.Presieve.IsSeparatedFor P R) (∀ (x : CategoryTheory …
  -/
  rw [← isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (And (CategoryTheory.Presieve.IsSeparatedFor P R) (∀ (x : CategoryTheory …
  -/
  rw [← isSeparatedFor_iff_generate]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (And (CategoryTheory.Presieve.IsSeparatedFor P R) (∀ (x : CategoryTheory …
  -/
  apply and_congr (Iff.refl _)
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    X : C
    R : CategoryTheory.Presieve X
    ⊢ Iff (∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Ex …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      ⊢ (∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exists …
    -/
  · intro q x hx
    /-
      case mp
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      q : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exist …
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
      hx : x.Compatible
      ⊢ Exists fun t => x.IsAmalgamation t
    -/
    apply Exists.imp _ (q _ (hx.restrict (le_generate R)))
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      q : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exist …
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
      hx : x.Compatible
      ⊢ ∀ (a : P.obj { unop := X }), (CategoryTheory.Presieve.FamilyOfElements.restr …
    -/
    intro t ht
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      q : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exist …
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.generate  …
      hx : x.Compatible
      t : P.obj { unop := X }
      ht : (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x).IsAmalgamation t
      ⊢ x.IsAmalgamation t
    -/
    simpa [hx] using isAmalgamation_sieveExtend _ _ ht
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      ⊢ (∀ (x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.gen …
    -/
  · intro q x hx
    /-
      case mpr
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      q : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.ge …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      ⊢ Exists fun t => x.IsAmalgamation t
    -/
    apply Exists.imp _ (q _ hx.sieveExtend)
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      q : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.ge …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      ⊢ ∀ (a : P.obj { unop := X }), x.sieveExtend.IsAmalgamation a → x.IsAmalgamati …
    -/
    intro t ht
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      X : C
      R : CategoryTheory.Presieve X
      q : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Sieve.ge …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      t : P.obj { unop := X }
      ht : x.sieveExtend.IsAmalgamation t
      ⊢ x.IsAmalgamation t
    -/
    simpa [hx] using isAmalgamation_restrict (le_generate R) _ _ ht
    /-
      🎉 no goals
    -/


/-- Every presheaf is a sheaf for the family {𝟙 X}.

[Elephant] C2.1.5(i)
-/
theorem isSheafFor_singleton_iso (P : Cᵒᵖ ⥤ Type w) : IsSheafFor P (Presieve.singleton (𝟙 X)) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.singleton (Cat …
  -/
  intro x _
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Presieve.single …
    a✝ : x.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  refine ⟨x _ (Presieve.singleton_self _), ?_, ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Presieve.single …
      a✝ : x.Compatible
      ⊢ (fun t => x.IsAmalgamation t) (x (CategoryTheory.CategoryStruct.id X) ⋯)
    -/
  · rintro _ _ ⟨rfl, rfl⟩
    /-
      case refine_1.mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Presieve.single …
      a✝ : x.Compatible
      Y : C
      ⊢ Eq (P.map (CategoryTheory.CategoryStruct.id X).op (x (CategoryTheory.Categor …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Presieve.single …
      a✝ : x.Compatible
      ⊢ ∀ (y : P.obj { unop := X }), (fun t => x.IsAmalgamation t) y → Eq y (x (Cate …
    -/
  · intro t ht
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Presieve.single …
      a✝ : x.Compatible
      t : P.obj { unop := X }
      ht : x.IsAmalgamation t
      ⊢ Eq t (x (CategoryTheory.CategoryStruct.id X) ⋯)
    -/
    simpa using ht _ (Presieve.singleton_self _)
    /-
      🎉 no goals
    -/


/-- Every presheaf is a sheaf for the maximal sieve.

[Elephant] C2.1.5(ii)
-/
theorem isSheafFor_top_sieve (P : Cᵒᵖ ⥤ Type w) : IsSheafFor P ((⊤ : Sieve X) : Presieve X) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ CategoryTheory.Presieve.IsSheafFor P Top.top.arrows
  -/
  rw [← generate_of_singleton_isSplitEpi (𝟙 X)]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.generate (Categor …
  -/
  rw [← isSheafFor_iff_generate]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.singleton (Cat …
  -/
  apply isSheafFor_singleton_iso
  /-
    🎉 no goals
  -/


/-- If `P₁ : Cᵒᵖ ⥤ Type w` and `P₂  : Cᵒᵖ ⥤ Type w` are two naturally equivalent
presheaves, and `P₁` is a sheaf for a presieve `R`, then `P₂` is also a sheaf for `R`. -/
lemma isSheafFor_of_nat_equiv {P₁ : Cᵒᵖ ⥤ Type w} {P₂ : Cᵒᵖ ⥤ Type w'}
    (e : ∀ ⦃X : C⦄, P₁.obj (op X) ≃ P₂.obj (op X))
    (he : ∀ ⦃X Y : C⦄ (f : X ⟶ Y) (x : P₁.obj (op Y)),
      e (P₁.map f.op x) = P₂.map f.op (e x))
    {X : C} {R : Presieve X} (hP₁ : IsSheafFor P₁ R) :
    IsSheafFor P₂ R := fun x₂ hx₂ ↦ by
  have he' : ∀ ⦃X Y : C⦄ (f : X ⟶ Y) (x : P₂.obj (op Y)),
    e.symm (P₂.map f.op x) = P₁.map f.op (e.symm x) := fun X Y f x ↦
      e.injective (by simp only [Equiv.apply_symm_apply, he])
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P₁ : CategoryTheory.Functor (Opposite C) (Type w)
    P₂ : CategoryTheory.Functor (Opposite C) (Type w')
    e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
    he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
    X : C
    R : CategoryTheory.Presieve X
    hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
    x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
    hx₂ : x₂.Compatible
    he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
    ⊢ ExistsUnique fun t => x₂.IsAmalgamation t
  -/
  let x₁ : FamilyOfElements P₁ R := fun Y f hf ↦ e.symm (x₂ f hf)
  have hx₁ : x₁.Compatible := fun Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ fac ↦ e.injective
    (by simp only [he, Equiv.apply_symm_apply, hx₂ g₁ g₂ h₁ h₂ fac, x₁])
  have : ∀ (t₂ : P₂.obj (op X)),
      x₂.IsAmalgamation t₂ ↔ x₁.IsAmalgamation (e.symm t₂) := fun t₂ ↦ by
    simp only [FamilyOfElements.IsAmalgamation, x₁,
      ← he', EmbeddingLike.apply_eq_iff_eq]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P₁ : CategoryTheory.Functor (Opposite C) (Type w)
    P₂ : CategoryTheory.Functor (Opposite C) (Type w')
    e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
    he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
    X : C
    R : CategoryTheory.Presieve X
    hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
    x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
    hx₂ : x₂.Compatible
    he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
    x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
    hx₁ : x₁.Compatible
    this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
    ⊢ ExistsUnique fun t => x₂.IsAmalgamation t
  -/
  refine ⟨e (hP₁.amalgamate x₁ hx₁), ?_, ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P₁ : CategoryTheory.Functor (Opposite C) (Type w)
      P₂ : CategoryTheory.Functor (Opposite C) (Type w')
      e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
      he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
      X : C
      R : CategoryTheory.Presieve X
      hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
      x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
      hx₂ : x₂.Compatible
      he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
      x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
      hx₁ : x₁.Compatible
      this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
      ⊢ (fun t => x₂.IsAmalgamation t) (e (hP₁.amalgamate x₁ hx₁))
    -/
  · dsimp
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P₁ : CategoryTheory.Functor (Opposite C) (Type w)
      P₂ : CategoryTheory.Functor (Opposite C) (Type w')
      e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
      he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
      X : C
      R : CategoryTheory.Presieve X
      hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
      x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
      hx₂ : x₂.Compatible
      he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
      x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
      hx₁ : x₁.Compatible
      this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
      ⊢ x₂.IsAmalgamation (e (hP₁.amalgamate x₁ hx₁))
    -/
    simp only [this, Equiv.symm_apply_apply]
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P₁ : CategoryTheory.Functor (Opposite C) (Type w)
      P₂ : CategoryTheory.Functor (Opposite C) (Type w')
      e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
      he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
      X : C
      R : CategoryTheory.Presieve X
      hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
      x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
      hx₂ : x₂.Compatible
      he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
      x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
      hx₁ : x₁.Compatible
      this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
      ⊢ x₁.IsAmalgamation (hP₁.amalgamate x₁ hx₁)
    -/
    exact IsSheafFor.isAmalgamation hP₁ hx₁
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P₁ : CategoryTheory.Functor (Opposite C) (Type w)
      P₂ : CategoryTheory.Functor (Opposite C) (Type w')
      e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
      he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
      X : C
      R : CategoryTheory.Presieve X
      hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
      x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
      hx₂ : x₂.Compatible
      he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
      x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
      hx₁ : x₁.Compatible
      this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
      ⊢ ∀ (y : P₂.obj { unop := X }), (fun t => x₂.IsAmalgamation t) y → Eq y (e (hP …
    -/
  · intro t₂ ht₂
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P₁ : CategoryTheory.Functor (Opposite C) (Type w)
      P₂ : CategoryTheory.Functor (Opposite C) (Type w')
      e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
      he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
      X : C
      R : CategoryTheory.Presieve X
      hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
      x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
      hx₂ : x₂.Compatible
      he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
      x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
      hx₁ : x₁.Compatible
      this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
      t₂ : P₂.obj { unop := X }
      ht₂ : x₂.IsAmalgamation t₂
      ⊢ Eq t₂ (e (hP₁.amalgamate x₁ hx₁))
    -/
    refine e.symm.injective ?_
    /-
      case refine_2
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P₁ : CategoryTheory.Functor (Opposite C) (Type w)
      P₂ : CategoryTheory.Functor (Opposite C) (Type w')
      e : ⦃X : C⦄ → Equiv (P₁.obj { unop := X }) (P₂.obj { unop := X })
      he : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₁.obj { unop := Y }), Eq (e (P₁.ma …
      X : C
      R : CategoryTheory.Presieve X
      hP₁ : CategoryTheory.Presieve.IsSheafFor P₁ R
      x₂ : CategoryTheory.Presieve.FamilyOfElements P₂ R
      hx₂ : x₂.Compatible
      he' : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y) (x : P₂.obj { unop := Y }), Eq (e.symm  …
      x₁ : CategoryTheory.Presieve.FamilyOfElements P₁ R := fun Y f hf => e.symm (x₂ …
      hx₁ : x₁.Compatible
      this : ∀ (t₂ : P₂.obj { unop := X }), Iff (x₂.IsAmalgamation t₂) (x₁.IsAmalgam …
      t₂ : P₂.obj { unop := X }
      ht₂ : x₂.IsAmalgamation t₂
      ⊢ Eq (e.symm t₂) (e.symm (e (hP₁.amalgamate x₁ hx₁)))
    -/
    simp only [Equiv.symm_apply_apply]
    exact hP₁.isSeparatedFor x₁ _ _ (by simpa only [this] using ht₂)
      (IsSheafFor.isAmalgamation hP₁ hx₁)


/-- If `P` is a sheaf for `S`, and it is iso to `P'`, then `P'` is a sheaf for `S`. This shows that
"being a sheaf for a presieve" is a mathematical or hygienic property.
-/
theorem isSheafFor_iso {P' : Cᵒᵖ ⥤ Type w} (i : P ≅ P') (hP : IsSheafFor P R) :
    IsSheafFor P' R :=
  isSheafFor_of_nat_equiv (fun X ↦ (i.app (op X)).toEquiv)
    (fun _ _ f x ↦ congr_fun (i.hom.naturality f.op) x) hP


/-- If a presieve `R` on `X` has a subsieve `S` such that:

* `P` is a sheaf for `S`.
* For every `f` in `R`, `P` is separated for the pullback of `S` along `f`,

then `P` is a sheaf for `R`.

This is closely related to [Elephant] C2.1.6(i).
-/
theorem isSheafFor_subsieve_aux (P : Cᵒᵖ ⥤ Type w) {S : Sieve X} {R : Presieve X}
    (h : (S : Presieve X) ≤ R) (hS : IsSheafFor P (S : Presieve X))
    (trans : ∀ ⦃Y⦄ ⦃f : Y ⟶ X⦄, R f → IsSeparatedFor P (S.pullback f : Presieve Y)) :
    IsSheafFor P R := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    S : CategoryTheory.Sieve X
    R : CategoryTheory.Presieve X
    h : LE.le S.arrows R
    hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
    trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
    ⊢ CategoryTheory.Presieve.IsSheafFor P R
  -/
  rw [← isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    S : CategoryTheory.Sieve X
    R : CategoryTheory.Presieve X
    h : LE.le S.arrows R
    hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
    trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
    ⊢ And (CategoryTheory.Presieve.IsSeparatedFor P R) (∀ (x : CategoryTheory.Pres …
  -/
  constructor
    /-
      case left
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P R
    -/
  · intro x t₁ t₂ ht₁ ht₂
    exact
      hS.isSeparatedFor _ _ _ (isAmalgamation_restrict h x t₁ ht₁)
        (isAmalgamation_restrict h x t₂ ht₂)
    /-
      case right
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      ⊢ ∀ (x : CategoryTheory.Presieve.FamilyOfElements P R), x.Compatible → Exists  …
    -/
  · intro x hx
    /-
      case right
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      ⊢ Exists fun t => x.IsAmalgamation t
    -/
    use hS.amalgamate _ (hx.restrict h)
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      ⊢ x.IsAmalgamation (hS.amalgamate (CategoryTheory.Presieve.FamilyOfElements.re …
    -/
    intro W j hj
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      W : C
      j : Quiver.Hom W X
      hj : R j
      ⊢ Eq (P.map j.op (hS.amalgamate (CategoryTheory.Presieve.FamilyOfElements.rest …
    -/
    apply (trans hj).ext
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      W : C
      j : Quiver.Hom W X
      hj : R j
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y W⦄, (CategoryTheory.Sieve.pullback j S).arrows f …
    -/
    intro Y f hf
    rw [← FunctorToTypes.map_comp_apply, ← op_comp, hS.valid_glue (hx.restrict h) _ hf,
      FamilyOfElements.restrict, ← hx (𝟙 _) f (h _ hf) _ (id_comp _)]
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      S : CategoryTheory.Sieve X
      R : CategoryTheory.Presieve X
      h : LE.le S.arrows R
      hS : CategoryTheory.Presieve.IsSheafFor P S.arrows
      trans : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, R f → CategoryTheory.Presieve.IsSepara …
      x : CategoryTheory.Presieve.FamilyOfElements P R
      hx : x.Compatible
      W : C
      j : Quiver.Hom W X
      hj : R j
      Y : C
      f : Quiver.Hom Y W
      hf : (CategoryTheory.Sieve.pullback j S).arrows f
      ⊢ Eq (x (CategoryTheory.CategoryStruct.comp f j) ⋯) (P.map (CategoryTheory.Cat …
    -/
    simp
    /-
      🎉 no goals
    -/


/--
If `P` is a sheaf for every pullback of the sieve `S`, then `P` is a sheaf for any presieve which
contains `S`.
This is closely related to [Elephant] C2.1.6.
-/
theorem isSheafFor_subsieve (P : Cᵒᵖ ⥤ Type w) {S : Sieve X} {R : Presieve X}
    (h : (S : Presieve X) ≤ R) (trans : ∀ ⦃Y⦄ (f : Y ⟶ X),
      IsSheafFor P (S.pullback f : Presieve Y)) :
    IsSheafFor P R :=
                                  /-
                                    C : Type u₁
                                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                    X : C
                                    P : CategoryTheory.Functor (Opposite C) (Type w)
                                    S : CategoryTheory.Sieve X
                                    R : CategoryTheory.Presieve X
                                    h : LE.le S.arrows R
                                    trans : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X), CategoryTheory.Presieve.IsSheafFor P ( …
                                    ⊢ CategoryTheory.Presieve.IsSheafFor P S.arrows
                                  -/
  isSheafFor_subsieve_aux P h (by simpa using trans (𝟙 _)) fun _ f _ => (trans f).isSeparatedFor
                                  /-
                                    🎉 no goals
                                  -/


/--
A more explicit version of `FamilyOfElements.Compatible` for a `Presieve.ofArrows`.
-/
def Arrows.Compatible (x : (i : I) → P.obj (op (X i))) : Prop :=
  ∀ i j Z (gi : Z ⟶ X i) (gj : Z ⟶ X j), gi ≫ π i = gj ≫ π j →
    P.map gi.op (x i) = P.map gj.op (x j)


lemma FamilyOfElements.isAmalgamation_iff_ofArrows (x : FamilyOfElements P (ofArrows X π))
    (t : P.obj (op B)) :
    x.IsAmalgamation t ↔ ∀ (i : I), P.map (π i).op t = x _ (ofArrows.mk i) :=
  ⟨fun h i ↦ h _ (ofArrows.mk i), fun h _ f ⟨i⟩ ↦ h i⟩


theorem exists_familyOfElements (hx : Compatible P π x) :
    ∃ (x' : FamilyOfElements P (ofArrows X π)), ∀ (i : I), x' _ (ofArrows.mk i) = x i := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type u_1
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    x : (i : I) → P.obj { unop := X i }
    hx : CategoryTheory.Presieve.Arrows.Compatible P π x
    ⊢ Exists fun x' => ∀ (i : I), Eq (x' (π i) ⋯) (x i)
  -/
  choose i h h' using @ofArrows_surj _ _ _ _ _ π
  exact ⟨fun Y f hf ↦ P.map (eqToHom (h f hf).symm).op (x _),
    fun j ↦ (hx _ j (X j) _ (𝟙 _) <| by rw [← h', id_comp]).trans <| by simp⟩


/--
A `FamilyOfElements` associated to an explicit family of elements.
-/
noncomputable
def familyOfElements : FamilyOfElements P (ofArrows X π) :=
  (exists_familyOfElements hx).choose


@[simp]
theorem familyOfElements_ofArrows_mk (i : I) :
    hx.familyOfElements _ (ofArrows.mk i) = x i :=
  (exists_familyOfElements hx).choose_spec _


theorem familyOfElements_compatible : hx.familyOfElements.Compatible := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type u_1
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    x : (i : I) → P.obj { unop := X i }
    hx : CategoryTheory.Presieve.Arrows.Compatible P π x
    ⊢ hx.familyOfElements.Compatible
  -/
  rintro Y₁ Y₂ Z g₁ g₂ f₁ f₂ ⟨i⟩ ⟨j⟩ hgf
  /-
    case mk.mk
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type u_1
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    x : (i : I) → P.obj { unop := X i }
    hx : CategoryTheory.Presieve.Arrows.Compatible P π x
    Z Y✝ : C
    i : I
    g₁ : Quiver.Hom Z (X i)
    Y : C
    j : I
    g₂ : Quiver.Hom Z (X j)
    hgf : Eq (CategoryTheory.CategoryStruct.comp g₁ (π i)) (CategoryTheory.Categor …
    ⊢ Eq (P.map g₁.op (hx.familyOfElements (π i) ⋯)) (P.map g₂.op (hx.familyOfElem …
  -/
  simp [hx i j Z g₁ g₂ hgf]
  /-
    🎉 no goals
  -/


theorem isSheafFor_arrows_iff : (ofArrows X π).IsSheafFor P ↔
    (∀ (x : (i : I) → P.obj (op (X i))), Arrows.Compatible P π x →
    ∃! t, ∀ i, P.map (π i).op t = x i) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type u_1
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows  …
  -/
  refine ⟨fun h x hx ↦ ?_, fun h x hx ↦ ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type u_1
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      h : CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
      x : (i : I) → P.obj { unop := X i }
      hx : CategoryTheory.Presieve.Arrows.Compatible P π x
      ⊢ ExistsUnique fun t => ∀ (i : I), Eq (P.map (π i).op t) (x i)
    -/
  · obtain ⟨t, ht₁, ht₂⟩ := h _ hx.familyOfElements_compatible
    /-
      case refine_1.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type u_1
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      h : CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
      x : (i : I) → P.obj { unop := X i }
      hx : CategoryTheory.Presieve.Arrows.Compatible P π x
      t : P.obj { unop := B }
      ht₁ : hx.familyOfElements.IsAmalgamation t
      ht₂ : ∀ (y : P.obj { unop := B }), (fun t => hx.familyOfElements.IsAmalgamatio …
      ⊢ ExistsUnique fun t => ∀ (i : I), Eq (P.map (π i).op t) (x i)
    -/
    refine ⟨t, fun i ↦ ?_, fun t' ht' ↦ ht₂ _ fun _ _ ⟨i⟩ ↦ ?_⟩
      /-
        case refine_1.intro.intro.refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        B : C
        I : Type u_1
        X : I → C
        π : (i : I) → Quiver.Hom (X i) B
        h : CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
        x : (i : I) → P.obj { unop := X i }
        hx : CategoryTheory.Presieve.Arrows.Compatible P π x
        t : P.obj { unop := B }
        ht₁ : hx.familyOfElements.IsAmalgamation t
        ht₂ : ∀ (y : P.obj { unop := B }), (fun t => hx.familyOfElements.IsAmalgamatio …
        i : I
        ⊢ Eq (P.map (π i).op t) (x i)
      -/
    · rw [ht₁ _ (ofArrows.mk i), hx.familyOfElements_ofArrows_mk]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        P : CategoryTheory.Functor (Opposite C) (Type w)
        B : C
        I : Type u_1
        X : I → C
        π : (i : I) → Quiver.Hom (X i) B
        h : CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
        x : (i : I) → P.obj { unop := X i }
        hx : CategoryTheory.Presieve.Arrows.Compatible P π x
        t : P.obj { unop := B }
        ht₁ : hx.familyOfElements.IsAmalgamation t
        ht₂ : ∀ (y : P.obj { unop := B }), (fun t => hx.familyOfElements.IsAmalgamatio …
        t' : P.obj { unop := B }
        ht' : (fun t => ∀ (i : I), Eq (P.map (π i).op t) (x i)) t'
        x✝² : C
        x✝¹ : Quiver.Hom x✝² B
        x✝ : CategoryTheory.Presieve.ofArrows X π x✝¹
        i : I
        ⊢ Eq (P.map (π i).op t') (hx.familyOfElements (π i) ⋯)
      -/
    · rw [ht', hx.familyOfElements_ofArrows_mk]
      /-
        🎉 no goals
      -/
  · obtain ⟨t, hA, ht⟩ := h (fun i ↦ x (π i) (ofArrows.mk _))
      (fun i j Z gi gj ↦ hx gi gj (ofArrows.mk _) (ofArrows.mk _))
    /-
      case refine_2.intro.intro
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type u_1
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      h : ∀ (x : (i : I) → P.obj { unop := X i }), CategoryTheory.Presieve.Arrows.Co …
      x : CategoryTheory.Presieve.FamilyOfElements P (CategoryTheory.Presieve.ofArro …
      hx : x.Compatible
      t : P.obj { unop := B }
      hA : ∀ (i : I), Eq (P.map (π i).op t) (x (π i) ⋯)
      ht : ∀ (y : P.obj { unop := B }), (fun t => ∀ (i : I), Eq (P.map (π i).op t) ( …
      ⊢ ExistsUnique fun t => x.IsAmalgamation t
    -/
    exact ⟨t, fun Y f ⟨i⟩ ↦ hA i, fun y hy ↦ ht y (fun i ↦ hy (π i) (ofArrows.mk _))⟩
    /-
      🎉 no goals
    -/


/--
A more explicit version of `FamilyOfElements.PullbackCompatible` for a `Presieve.ofArrows`.
-/
def Arrows.PullbackCompatible (x : (i : I) → P.obj (op (X i))) : Prop :=
  ∀ i j, P.map (pullback.fst (π i) (π j)).op (x i) =
    P.map (pullback.snd (π i) (π j)).op (x j)


theorem Arrows.pullbackCompatible_iff (x : (i : I) → P.obj (op (X i))) :
    Compatible P π x ↔ PullbackCompatible P π x := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type u_1
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    x : (i : I) → P.obj { unop := X i }
    ⊢ Iff (CategoryTheory.Presieve.Arrows.Compatible P π x) (CategoryTheory.Presie …
  -/
  refine ⟨fun t i j ↦ ?_, fun t i j Z gi gj comm ↦ ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type u_1
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      t : CategoryTheory.Presieve.Arrows.Compatible P π x
      i j : I
      ⊢ Eq (P.map (CategoryTheory.Limits.pullback.fst (π i) (π j)).op (x i)) (P.map  …
    -/
  · apply t
    /-
      case refine_1.a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      I : Type u_1
      X : I → C
      π : (i : I) → Quiver.Hom (X i) B
      inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
      x : (i : I) → P.obj { unop := X i }
      t : CategoryTheory.Presieve.Arrows.Compatible P π x
      i j : I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    exact pullback.condition
    /-
      🎉 no goals
    -/
  · rw [← pullback.lift_fst _ _ comm, op_comp, FunctorToTypes.map_comp_apply, t i j,
      ← FunctorToTypes.map_comp_apply, ← op_comp, pullback.lift_snd]


theorem isSheafFor_arrows_iff_pullbacks : (ofArrows X π).IsSheafFor P ↔
    (∀ (x : (i : I) → P.obj (op (X i))), Arrows.PullbackCompatible P π x →
    ∃! t, ∀ i, P.map (π i).op t = x i) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    B : C
    I : Type u_1
    X : I → C
    π : (i : I) → Quiver.Hom (X i) B
    inst✝ : (CategoryTheory.Presieve.ofArrows X π).hasPullbacks
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows  …
  -/
  simp_rw [← Arrows.pullbackCompatible_iff, isSheafFor_arrows_iff]
  /-
    🎉 no goals
  -/


