/-- The type of Compact Hausdorff topological spaces satisfying an additional property `P`. -/
structure CompHausLike where
  /-- The underlying topological space of an object of `CompHausLike P`. -/
  toTop : TopCat
  /-- The underlying topological space is compact. -/
  [is_compact : CompactSpace toTop]
  /-- The underlying topological space is T2. -/
  [is_hausdorff : T2Space toTop]
  /-- The underlying topological space satisfies P. -/
  prop : P toTop


instance : CoeSort (CompHausLike P) (Type u) :=
  ⟨fun X => X.toTop⟩


instance category : Category (CompHausLike P) :=
  InducedCategory.category toTop


instance concreteCategory : ConcreteCategory (CompHausLike P) :=
  InducedCategory.concreteCategory _


instance hasForget₂ : HasForget₂ (CompHausLike P) TopCat :=
  InducedCategory.hasForget₂ _


/-- This wraps the predicate `P : TopCat → Prop` in a typeclass. -/
class HasProp : Prop where
  hasProp : P (TopCat.of X)


/-- A constructor for objects of the category `CompHausLike P`,
taking a type, and bundling the compact Hausdorff topology
found by typeclass inference. -/
def of : CompHausLike P where
  toTop := TopCat.of X
  is_compact := ‹_›
  is_hausdorff := ‹_›
  prop := HasProp.hasProp


@[simp]
theorem coe_of : (CompHausLike.of P X : Type _) = X :=
  rfl


@[simp]
theorem coe_id (X : CompHausLike P) : (𝟙 ((forget (CompHausLike P)).obj X)) = id :=
  rfl


@[simp]
theorem coe_comp {X Y Z : CompHausLike P} (f : X ⟶ Y) (g : Y ⟶ Z) :
    ((forget (CompHausLike P)).map f ≫ (forget (CompHausLike P)).map g) = g ∘ f :=
  rfl

-- Note (https://github.com/leanprover-community/mathlib4/issues/10754): Lean does not see through the forgetful functor here

instance (X : CompHausLike.{u} P) : TopologicalSpace ((forget (CompHausLike P)).obj X) :=
  inferInstanceAs (TopologicalSpace X.toTop)

-- Note (https://github.com/leanprover-community/mathlib4/issues/10754): Lean does not see through the forgetful functor here

instance (X : CompHausLike.{u} P) : CompactSpace ((forget (CompHausLike P)).obj X) :=
  inferInstanceAs (CompactSpace X.toTop)

-- Note (https://github.com/leanprover-community/mathlib4/issues/10754): Lean does not see through the forgetful functor here

instance (X : CompHausLike.{u} P) : T2Space ((forget (CompHausLike P)).obj X) :=
  inferInstanceAs (T2Space X.toTop)


/-- If `P` imples `P'`, then there is a functor from `CompHausLike P` to `CompHausLike P'`. -/
@[simps]
def toCompHausLike {P P' : TopCat → Prop} (h : ∀ (X : CompHausLike P), P X.toTop → P' X.toTop) :
    CompHausLike P ⥤ CompHausLike P' where
  obj X :=
    have : HasProp P' X := ⟨(h _ X.prop)⟩
    CompHausLike.of _ X
  map f := f


/-- If `P` imples `P'`, then the functor from `CompHausLike P` to `CompHausLike P'` is fully
faithful. -/
def fullyFaithfulToCompHausLike : (toCompHausLike h).FullyFaithful :=
  fullyFaithfulInducedFunctor _


instance : (toCompHausLike h).Full := (fullyFaithfulToCompHausLike h).full


instance : (toCompHausLike h).Faithful := (fullyFaithfulToCompHausLike h).faithful


/-- The fully faithful embedding of `CompHausLike P` in `TopCat`. -/
@[simps!]
def compHausLikeToTop : CompHausLike.{u} P ⥤ TopCat.{u} :=
  inducedFunctor _ -- deriving Full, Faithful -- Porting note: deriving fails, adding manually.


/-- The functor from `CompHausLike P` to `TopCat` is fully faithful. -/
def fullyFaithfulCompHausLikeToTop : (compHausLikeToTop P).FullyFaithful :=
  fullyFaithfulInducedFunctor _


instance : (compHausLikeToTop P).Full  :=
  inferInstanceAs (inducedFunctor _).Full


instance : (compHausLikeToTop P).Faithful :=
  inferInstanceAs (inducedFunctor _).Faithful


instance (X : CompHausLike P) : CompactSpace ((compHausLikeToTop P).obj X) :=
  inferInstanceAs (CompactSpace X.toTop)


instance (X : CompHausLike P) : T2Space ((compHausLikeToTop P).obj X) :=
  inferInstanceAs (T2Space X.toTop)


theorem epi_of_surjective {X Y : CompHausLike.{u} P} (f : X ⟶ Y) (hf : Function.Surjective f) :
    Epi f := by
  /-
    P : TopCat → Prop
    X Y : CompHausLike P
    f : Quiver.Hom X Y
    hf : Function.Surjective ⇑f
    ⊢ CategoryTheory.Epi f
  -/
  rw [← CategoryTheory.epi_iff_surjective] at hf
  /-
    P : TopCat → Prop
    X Y : CompHausLike P
    f : Quiver.Hom X Y
    hf : CategoryTheory.Epi ⇑f
    ⊢ CategoryTheory.Epi f
  -/
  exact (forget (CompHausLike P)).epi_of_epi_map hf
  /-
    🎉 no goals
  -/


theorem mono_iff_injective {X Y : CompHausLike.{u} P} (f : X ⟶ Y) :
    Mono f ↔ Function.Injective f := by
  /-
    P : TopCat → Prop
    X Y : CompHausLike P
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective ⇑f)
  -/
  constructor
    /-
      case mp
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono f → Function.Injective ⇑f
    -/
  · intro hf x₁ x₂ h
    /-
      case mp
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      hf : CategoryTheory.Mono f
      x₁ x₂ : (CategoryTheory.forget (CompHausLike P)).obj X
      h : Eq (f x₁) (f x₂)
      ⊢ Eq x₁ x₂
    -/
    let g₁ : X ⟶ X := ⟨fun _ => x₁, continuous_const⟩
    /-
      case mp
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      hf : CategoryTheory.Mono f
      x₁ x₂ : (CategoryTheory.forget (CompHausLike P)).obj X
      h : Eq (f x₁) (f x₂)
      g₁ : Quiver.Hom X X := { toFun := fun x => x₁, continuous_toFun := ⋯ }
      ⊢ Eq x₁ x₂
    -/
    let g₂ : X ⟶ X := ⟨fun _ => x₂, continuous_const⟩
    /-
      case mp
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      hf : CategoryTheory.Mono f
      x₁ x₂ : (CategoryTheory.forget (CompHausLike P)).obj X
      h : Eq (f x₁) (f x₂)
      g₁ : Quiver.Hom X X := { toFun := fun x => x₁, continuous_toFun := ⋯ }
      g₂ : Quiver.Hom X X := { toFun := fun x => x₂, continuous_toFun := ⋯ }
      ⊢ Eq x₁ x₂
    -/
    have : g₁ ≫ f = g₂ ≫ f := by ext; exact h
    /-
      case mp
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      hf : CategoryTheory.Mono f
      x₁ x₂ : (CategoryTheory.forget (CompHausLike P)).obj X
      h : Eq (f x₁) (f x₂)
      g₁ : Quiver.Hom X X := { toFun := fun x => x₁, continuous_toFun := ⋯ }
      g₂ : Quiver.Hom X X := { toFun := fun x => x₂, continuous_toFun := ⋯ }
      this : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategorySt …
      ⊢ Eq x₁ x₂
    -/
    exact ContinuousMap.congr_fun ((cancel_mono _).mp this) x₁
    /-
      🎉 no goals
    -/
    /-
      case mpr
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      ⊢ Function.Injective ⇑f → CategoryTheory.Mono f
    -/
  · rw [← CategoryTheory.mono_iff_injective]
    /-
      case mpr
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono ⇑f → CategoryTheory.Mono f
    -/
    apply (forget (CompHausLike P)).mono_of_mono_map
    /-
      🎉 no goals
    -/


/-- Any continuous function on compact Hausdorff spaces is a closed map. -/
theorem isClosedMap {X Y : CompHausLike.{u} P} (f : X ⟶ Y) : IsClosedMap f := fun _ hC =>
  (hC.isCompact.image f.continuous).isClosed


/-- Any continuous bijection of compact Hausdorff spaces is an isomorphism. -/
theorem isIso_of_bijective {X Y : CompHausLike.{u} P} (f : X ⟶ Y) (bij : Function.Bijective f) :
    IsIso f := by
  /-
    P : TopCat → Prop
    X Y : CompHausLike P
    f : Quiver.Hom X Y
    bij : Function.Bijective ⇑f
    ⊢ CategoryTheory.IsIso f
  -/
  let E := Equiv.ofBijective _ bij
  have hE : Continuous E.symm := by
    rw [continuous_iff_isClosed]
    intro S hS
    rw [← E.image_eq_preimage]
    exact isClosedMap f S hS
  /-
    P : TopCat → Prop
    X Y : CompHausLike P
    f : Quiver.Hom X Y
    bij : Function.Bijective ⇑f
    E : Equiv ((CategoryTheory.forget (CompHausLike P)).obj X) ((CategoryTheory.fo …
    hE : Continuous ⇑E.symm
    ⊢ CategoryTheory.IsIso f
  -/
  refine ⟨⟨⟨E.symm, hE⟩, ?_, ?_⟩⟩
    /-
      case refine_1
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      bij : Function.Bijective ⇑f
      E : Equiv ((CategoryTheory.forget (CompHausLike P)).obj X) ((CategoryTheory.fo …
      hE : Continuous ⇑E.symm
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f { toFun := ⇑E.symm, continuous_toFu …
    -/
  · ext x
    /-
      case refine_1.w
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      bij : Function.Bijective ⇑f
      E : Equiv ((CategoryTheory.forget (CompHausLike P)).obj X) ((CategoryTheory.fo …
      hE : Continuous ⇑E.symm
      x : (CategoryTheory.forget (CompHausLike P)).obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp f { toFun := ⇑E.symm, continuous_toF …
    -/
    apply E.symm_apply_apply
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      bij : Function.Bijective ⇑f
      E : Equiv ((CategoryTheory.forget (CompHausLike P)).obj X) ((CategoryTheory.fo …
      hE : Continuous ⇑E.symm
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := ⇑E.symm, continuous_toFun  …
    -/
  · ext x
    /-
      case refine_2.w
      P : TopCat → Prop
      X Y : CompHausLike P
      f : Quiver.Hom X Y
      bij : Function.Bijective ⇑f
      E : Equiv ((CategoryTheory.forget (CompHausLike P)).obj X) ((CategoryTheory.fo …
      hE : Continuous ⇑E.symm
      x : (CategoryTheory.forget (CompHausLike P)).obj Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := ⇑E.symm, continuous_toFun …
    -/
    apply E.apply_symm_apply
    /-
      🎉 no goals
    -/


instance forget_reflectsIsomorphisms :
    (forget (CompHausLike.{u} P)).ReflectsIsomorphisms :=
      /-
        P : TopCat → Prop
        X : Type u
        inst✝³ : TopologicalSpace X
        inst✝² : CompactSpace X
        inst✝¹ : T2Space X
        inst✝ : CompHausLike.HasProp P X
        ⊢ ∀ {A B : CompHausLike P} (f : Quiver.Hom A B) [inst : CategoryTheory.IsIso ( …
      -/
  ⟨by intro A B f hf; rw [isIso_iff_bijective] at hf; exact isIso_of_bijective _ hf⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Any continuous bijection of compact Hausdorff spaces induces an isomorphism. -/
noncomputable def isoOfBijective {X Y : CompHausLike.{u} P} (f : X ⟶ Y)
    (bij : Function.Bijective f) : X ≅ Y :=
  letI := isIso_of_bijective _ bij
  asIso f


/-- Construct an isomorphism from a homeomorphism. -/
@[simps!]
def isoOfHomeo {X Y : CompHausLike.{u} P} (f : X ≃ₜ Y) : X ≅ Y :=
  (fullyFaithfulCompHausLikeToTop P).preimageIso (TopCat.isoOfHomeo f)


/-- Construct a homeomorphism from an isomorphism. -/
@[simps!]
def homeoOfIso {X Y : CompHausLike.{u} P} (f : X ≅ Y) : X ≃ₜ Y :=
  TopCat.homeoOfIso <| (compHausLikeToTop P).mapIso f


/-- The equivalence between isomorphisms in `CompHaus` and homeomorphisms
of topological spaces. -/
@[simps]
def isoEquivHomeo {X Y : CompHausLike.{u} P} : (X ≅ Y) ≃ (X ≃ₜ Y) where
  toFun := homeoOfIso
  invFun := isoOfHomeo
  left_inv _ := rfl
  right_inv _ := rfl


/-- A constant map as a morphism in `CompHausLike` -/
def const {P : TopCat.{u} → Prop}
    (T : CompHausLike.{u} P) {S : CompHausLike.{u} P} (s : S) : T ⟶ S :=
  ContinuousMap.const _ s


lemma const_comp {P : TopCat.{u} → Prop} {S T U : CompHausLike.{u} P}
    (s : S) (g : S ⟶ U) : T.const s ≫ g = T.const (g s) :=
  rfl


