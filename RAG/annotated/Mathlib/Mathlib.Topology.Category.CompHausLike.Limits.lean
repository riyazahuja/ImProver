/--
A typeclass describing the property that forming the disjoint union is stable under the
property `P`.
-/
abbrev HasExplicitFiniteCoproduct := HasProp P (Σ (a : α), X a)


/--
The coproduct of a finite family of objects in `CompHaus`, constructed as the disjoint
union with its usual topology.
-/
def finiteCoproduct : CompHausLike P := CompHausLike.of P (Σ (a : α), X a)


/--
The inclusion of one of the factors into the explicit finite coproduct.
-/
def finiteCoproduct.ι (a : α) : X a ⟶ finiteCoproduct X where
  toFun := fun x ↦ ⟨a, x⟩
  continuous_toFun := continuous_sigmaMk (σ := fun a ↦ X a)


/--
To construct a morphism from the explicit finite coproduct, it suffices to
specify a morphism from each of its factors.
This is essentially the universal property of the coproduct.
-/
def finiteCoproduct.desc {B : CompHausLike P} (e : (a : α) → (X a ⟶ B)) :
    finiteCoproduct X ⟶ B where
  toFun := fun ⟨a, x⟩ ↦ e a x
  continuous_toFun := by
    /-
      P : TopCat → Prop
      α : Type w
      inst✝¹ : Finite α
      X : α → CompHausLike P
      inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
      B : CompHausLike P
      e : (a : α) → Quiver.Hom (X a) B
      ⊢ Continuous fun x => CompHausLike.finiteCoproduct.desc.match_1 X (fun x => ↑B …
    -/
    apply continuous_sigma
    /-
      case hf
      P : TopCat → Prop
      α : Type w
      inst✝¹ : Finite α
      X : α → CompHausLike P
      inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
      B : CompHausLike P
      e : (a : α) → Quiver.Hom (X a) B
      ⊢ ∀ (i : α), Continuous fun a => CompHausLike.finiteCoproduct.desc.match_1 X ( …
    -/
    intro a; exact (e a).continuous
             /-
               🎉 no goals
             -/


@[reassoc (attr := simp)]
lemma finiteCoproduct.ι_desc {B : CompHausLike P} (e : (a : α) → (X a ⟶ B)) (a : α) :
    finiteCoproduct.ι X a ≫ finiteCoproduct.desc X e = e a := rfl


lemma finiteCoproduct.hom_ext {B : CompHausLike P} (f g : finiteCoproduct X ⟶ B)
    (h : ∀ a : α, finiteCoproduct.ι X a ≫ f = finiteCoproduct.ι X a ≫ g) : f = g := by
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    f g : Quiver.Hom (CompHausLike.finiteCoproduct X) B
    h : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (CompHausLike.finiteCopr …
    ⊢ Eq f g
  -/
  ext ⟨a, x⟩
  /-
    case w.mk
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    f g : Quiver.Hom (CompHausLike.finiteCoproduct X) B
    h : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (CompHausLike.finiteCopr …
    a : α
    x : ↑(X a).toTop
    ⊢ Eq (f ⟨a, x⟩) (g ⟨a, x⟩)
  -/
  specialize h a
  /-
    case w.mk
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    f g : Quiver.Hom (CompHausLike.finiteCoproduct X) B
    a : α
    x : ↑(X a).toTop
    h : Eq (CategoryTheory.CategoryStruct.comp (CompHausLike.finiteCoproduct.ι X a …
    ⊢ Eq (f ⟨a, x⟩) (g ⟨a, x⟩)
  -/
  apply_fun (fun q ↦ q x) at h
  /-
    case w.mk
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    f g : Quiver.Hom (CompHausLike.finiteCoproduct X) B
    a : α
    x : ↑(X a).toTop
    h : Eq ((CategoryTheory.CategoryStruct.comp (CompHausLike.finiteCoproduct.ι X  …
    ⊢ Eq (f ⟨a, x⟩) (g ⟨a, x⟩)
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- The coproduct cocone associated to the explicit finite coproduct. -/
abbrev finiteCoproduct.cofan : Limits.Cofan X :=
  Cofan.mk (finiteCoproduct X) (finiteCoproduct.ι X)


/-- The explicit finite coproduct cocone is a colimit cocone. -/
def finiteCoproduct.isColimit : Limits.IsColimit (finiteCoproduct.cofan X) :=
  mkCofanColimit _
    (fun s ↦ desc _ fun a ↦ s.inj a)
    (fun _ _ ↦ ι_desc _ _ _)
    fun _ _ hm ↦ finiteCoproduct.hom_ext _ _ _ fun a ↦
      (DFunLike.ext _ _ fun t ↦ congrFun (congrArg DFunLike.coe (hm a)) t)


lemma finiteCoproduct.ι_injective (a : α) : Function.Injective (finiteCoproduct.ι X a) := by
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    a : α
    ⊢ Function.Injective ⇑(CompHausLike.finiteCoproduct.ι X a)
  -/
  intro x y hxy
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    a : α
    x y : (CategoryTheory.forget (CompHausLike P)).obj (X a)
    hxy : Eq ((CompHausLike.finiteCoproduct.ι X a) x) ((CompHausLike.finiteCoprodu …
    ⊢ Eq x y
  -/
  exact eq_of_heq (Sigma.ext_iff.mp hxy).2
  /-
    🎉 no goals
  -/


lemma finiteCoproduct.ι_jointly_surjective (R : finiteCoproduct X) :
    ∃ (a : α) (r : X a), R = finiteCoproduct.ι X a r := ⟨R.fst, R.snd, rfl⟩


lemma finiteCoproduct.ι_desc_apply {B : CompHausLike P} {π : (a : α) → X a ⟶ B} (a : α) :
    ∀ x, finiteCoproduct.desc X π (finiteCoproduct.ι X a x) = π a x := by
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    π : (a : α) → Quiver.Hom (X a) B
    a : α
    ⊢ ∀ (x : (CategoryTheory.forget (CompHausLike P)).obj (X a)), Eq ((CompHausLik …
  -/
  intro x
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    π : (a : α) → Quiver.Hom (X a) B
    a : α
    x : (CategoryTheory.forget (CompHausLike P)).obj (X a)
    ⊢ Eq ((CompHausLike.finiteCoproduct.desc X π) ((CompHausLike.finiteCoproduct.ι …
  -/
  change (ι X a ≫ desc X π) _ = _
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    B : CompHausLike P
    π : (a : α) → Quiver.Hom (X a) B
    a : α
    x : (CategoryTheory.forget (CompHausLike P)).obj (X a)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CompHausLike.finiteCoproduct.ι X a) …
  -/
  simp only [ι_desc]
  /-
    🎉 no goals
  -/


instance : HasCoproduct X where
  exists_colimit := ⟨finiteCoproduct.cofan X, finiteCoproduct.isColimit X⟩


variable (P) in
/--
A typeclass describing the property that forming all finite disjoint unions is stable under the
property `P`.
-/
class HasExplicitFiniteCoproducts : Prop where
  hasProp {α : Type w} [Finite α] (X : α → CompHausLike.{max u w} P) : HasExplicitFiniteCoproduct X

/-
This linter complains that the universes `u` and `w` only occur together, but `w` appears by itself
in the indexing type of the coproduct. In almost all cases, `w` will be either `0` or `u`, but we
want to allow both possibilities.
-/

instance [HasExplicitFiniteCoproducts.{w} P] (α : Type w) [Finite α] :
    HasColimitsOfShape (Discrete α) (CompHausLike P) where
  has_colimit _ := hasColimitOfIso Discrete.natIsoFunctor


instance [HasExplicitFiniteCoproducts.{w} P] : HasFiniteCoproducts (CompHausLike.{max u w} P) where
  out n := by
    /-
      P : TopCat → Prop
      α : Type w
      inst✝² : Finite α
      X : α → CompHausLike P
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproduct X
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      n : Nat
      ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Fin n)) ( …
    -/
    let α := ULift.{w} (Fin n)
    /-
      P : TopCat → Prop
      α✝ : Type w
      inst✝² : Finite α✝
      X : α✝ → CompHausLike P
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproduct X
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      n : Nat
      α : Type w := ULift.{w, 0} (Fin n)
      ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Fin n)) ( …
    -/
    let e : Discrete α ≌ Discrete (Fin n) := Discrete.equivalence Equiv.ulift
    /-
      P : TopCat → Prop
      α✝ : Type w
      inst✝² : Finite α✝
      X : α✝ → CompHausLike P
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproduct X
      inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
      n : Nat
      α : Type w := ULift.{w, 0} (Fin n)
      e : CategoryTheory.Equivalence (CategoryTheory.Discrete α) (CategoryTheory.Dis …
      ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete (Fin n)) ( …
    -/
    exact hasColimitsOfShape_of_equivalence e
    /-
      🎉 no goals
    -/


/-- The inclusion maps into the explicit finite coproduct are open embeddings. -/
lemma finiteCoproduct.isOpenEmbedding_ι (a : α) :
    IsOpenEmbedding (finiteCoproduct.ι X a) :=
  .sigmaMk (σ := fun a ↦ X a)


@[deprecated (since := "2024-10-18")]
alias finiteCoproduct.openEmbedding_ι := finiteCoproduct.isOpenEmbedding_ι


/-- The inclusion maps into the abstract finite coproduct are open embeddings. -/
lemma Sigma.isOpenEmbedding_ι (a : α) :
    IsOpenEmbedding (Sigma.ι X a) := by
  refine IsOpenEmbedding.of_comp _ (homeoOfIso ((colimit.isColimit _).coconePointUniqueUpToIso
    (finiteCoproduct.isColimit X))).isOpenEmbedding ?_
  /-
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    a : α
    ⊢ Topology.IsOpenEmbedding (Function.comp ⇑(CompHausLike.homeoOfIso ((Category …
  -/
  convert finiteCoproduct.isOpenEmbedding_ι X a
  /-
    case h.e'_5.h
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    a : α
    e_2✝ : Eq (↑(CompHausLike.finiteCoproduct.cofan X).pt.toTop) ((CategoryTheory. …
    ⊢ Eq (Function.comp ⇑(CompHausLike.homeoOfIso ((CategoryTheory.Limits.colimit. …
  -/
  ext x
  /-
    case h.e'_5.h.h
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    a : α
    e_2✝ : Eq (↑(CompHausLike.finiteCoproduct.cofan X).pt.toTop) ((CategoryTheory. …
    x : (CategoryTheory.forget (CompHausLike P)).obj (X a)
    ⊢ Eq (Function.comp (⇑(CompHausLike.homeoOfIso ((CategoryTheory.Limits.colimit …
  -/
  change (Sigma.ι X a ≫ _) x = _
  /-
    case h.e'_5.h.h
    P : TopCat → Prop
    α : Type w
    inst✝¹ : Finite α
    X : α → CompHausLike P
    inst✝ : CompHausLike.HasExplicitFiniteCoproduct X
    a : α
    e_2✝ : Eq (↑(CompHausLike.finiteCoproduct.cofan X).pt.toTop) ((CategoryTheory. …
    x : (CategoryTheory.forget (CompHausLike P)).obj (X a)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι X a)  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias Sigma.openEmbedding_ι := Sigma.isOpenEmbedding_ι


/-- The functor to `TopCat` preserves finite coproducts if they exist. -/
instance (P) [HasExplicitFiniteCoproducts.{0} P] :
    PreservesFiniteCoproducts (compHausLikeToTop P) := by
  /-
    P✝¹ : TopCat → Prop
    α : Type w
    inst✝³ : Finite α
    X : α → CompHausLike P✝¹
    inst✝² : CompHausLike.HasExplicitFiniteCoproduct X
    P✝ : TopCat → Prop
    inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P✝
    P : TopCat → Prop
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    ⊢ CategoryTheory.Limits.PreservesFiniteCoproducts (CompHausLike.compHausLikeTo …
  -/
  refine ⟨fun J hJ ↦ ⟨fun {F} ↦ ?_⟩⟩
  suffices PreservesColimit (Discrete.functor (F.obj ∘ Discrete.mk)) (compHausLikeToTop P) from
    preservesColimit_of_iso_diagram _ Discrete.natIsoFunctor.symm
  /-
    P✝¹ : TopCat → Prop
    α : Type w
    inst✝³ : Finite α
    X : α → CompHausLike P✝¹
    inst✝² : CompHausLike.HasExplicitFiniteCoproduct X
    P✝ : TopCat → Prop
    inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P✝
    P : TopCat → Prop
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    J : Type
    hJ : Fintype J
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CompHausLike P)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functor (Fun …
  -/
  apply preservesColimit_of_preserves_colimit_cocone (CompHausLike.finiteCoproduct.isColimit _)
  /-
    P✝¹ : TopCat → Prop
    α : Type w
    inst✝³ : Finite α
    X : α → CompHausLike P✝¹
    inst✝² : CompHausLike.HasExplicitFiniteCoproduct X
    P✝ : TopCat → Prop
    inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P✝
    P : TopCat → Prop
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    J : Type
    hJ : Fintype J
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CompHausLike P)
    ⊢ CategoryTheory.Limits.IsColimit ((CompHausLike.compHausLikeToTop P).mapCocon …
  -/
  exact TopCat.sigmaCofanIsColimit _
  /-
    🎉 no goals
  -/


/-- The functor to another `CompHausLike` preserves finite coproducts if they exist. -/
noncomputable instance {P' : TopCat.{u} → Prop}
    (h : ∀ (X : CompHausLike P), P X.toTop → P' X.toTop) :
    PreservesFiniteCoproducts (toCompHausLike h) := by
  have : PreservesFiniteCoproducts (toCompHausLike h ⋙ compHausLikeToTop P') :=
    inferInstanceAs (PreservesFiniteCoproducts (compHausLikeToTop _))
  /-
    P✝ : TopCat → Prop
    α : Type w
    inst✝² : Finite α
    X : α → CompHausLike P✝
    inst✝¹ : CompHausLike.HasExplicitFiniteCoproduct X
    P : TopCat → Prop
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    P' : TopCat → Prop
    h : ∀ (X : CompHausLike P), P X.toTop → P' X.toTop
    this : CategoryTheory.Limits.PreservesFiniteCoproducts ((CompHausLike.toCompHa …
    ⊢ CategoryTheory.Limits.PreservesFiniteCoproducts (CompHausLike.toCompHausLike …
  -/
  exact preservesFiniteCoproducts_of_reflects_of_preserves (toCompHausLike h) (compHausLikeToTop P')
  /-
    🎉 no goals
  -/


/--
A typeclass describing the property that an explicit pullback is stable under the property `P`.
-/
abbrev HasExplicitPullback := HasProp P { xy : X × Y | f xy.fst = g xy.snd }


/--
The pullback of two morphisms `f,g` in `CompHaus`, constructed explicitly as the set of
pairs `(x,y)` such that `f x = g y`, with the topology induced by the product.
-/
def pullback : CompHausLike P :=
  letI set := { xy : X × Y | f xy.fst = g xy.snd }
  haveI : CompactSpace set :=
    isCompact_iff_compactSpace.mp (isClosed_eq (f.continuous.comp continuous_fst)
      (g.continuous.comp continuous_snd)).isCompact
  CompHausLike.of P set


/--
The projection from the pullback to the first component.
-/
def pullback.fst : pullback f g ⟶ X where
  toFun := fun ⟨⟨x, _⟩, _⟩ ↦ x
  continuous_toFun := Continuous.comp continuous_fst continuous_subtype_val


/--
The projection from the pullback to the second component.
-/
def pullback.snd : pullback f g ⟶ Y where
  toFun := fun ⟨⟨_,y⟩,_⟩ ↦ y
  continuous_toFun := Continuous.comp continuous_snd continuous_subtype_val


@[reassoc]
lemma pullback.condition : pullback.fst f g ≫ f = pullback.snd f g ≫ g := by
  /-
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CompHausLike.pullback.fst f g) f) (C …
  -/
  ext ⟨_,h⟩; exact h
             /-
               🎉 no goals
             -/


/--
Construct a morphism to the explicit pullback given morphisms to the factors
which are compatible with the maps to the base.
This is essentially the universal property of the pullback.
-/
def pullback.lift {Z : CompHausLike P} (a : Z ⟶ X) (b : Z ⟶ Y) (w : a ≫ f = b ≫ g) :
    Z ⟶ pullback f g where
                                   /-
                                     P : TopCat → Prop
                                     X Y B : CompHausLike P
                                     f : Quiver.Hom X B
                                     g : Quiver.Hom Y B
                                     inst✝ : CompHausLike.HasExplicitPullback f g
                                     Z : CompHausLike P
                                     a : Quiver.Hom Z X
                                     b : Quiver.Hom Z Y
                                     w : Eq (CategoryTheory.CategoryStruct.comp a f) (CategoryTheory.CategoryStruct …
                                     z : ↑Z.toTop
                                     ⊢ Membership.mem (setOf fun xy => Eq (f xy.1) (g xy.2)) { fst := a z, snd := b …
                                   -/
  toFun := fun z ↦ ⟨⟨a z, b z⟩, by apply_fun (fun q ↦ q z) at w; exact w⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  continuous_toFun := by
    /-
      P : TopCat → Prop
      X Y B : CompHausLike P
      f : Quiver.Hom X B
      g : Quiver.Hom Y B
      inst✝ : CompHausLike.HasExplicitPullback f g
      Z : CompHausLike P
      a : Quiver.Hom Z X
      b : Quiver.Hom Z Y
      w : Eq (CategoryTheory.CategoryStruct.comp a f) (CategoryTheory.CategoryStruct …
      ⊢ Continuous fun z => ⟨{ fst := a z, snd := b z }, ⋯⟩
    -/
    apply Continuous.subtype_mk
    /-
      case h
      P : TopCat → Prop
      X Y B : CompHausLike P
      f : Quiver.Hom X B
      g : Quiver.Hom Y B
      inst✝ : CompHausLike.HasExplicitPullback f g
      Z : CompHausLike P
      a : Quiver.Hom Z X
      b : Quiver.Hom Z Y
      w : Eq (CategoryTheory.CategoryStruct.comp a f) (CategoryTheory.CategoryStruct …
      ⊢ Continuous fun x => { fst := a x, snd := b x }
    -/
    rw [continuous_prod_mk]
    /-
      case h
      P : TopCat → Prop
      X Y B : CompHausLike P
      f : Quiver.Hom X B
      g : Quiver.Hom Y B
      inst✝ : CompHausLike.HasExplicitPullback f g
      Z : CompHausLike P
      a : Quiver.Hom Z X
      b : Quiver.Hom Z Y
      w : Eq (CategoryTheory.CategoryStruct.comp a f) (CategoryTheory.CategoryStruct …
      ⊢ And (Continuous ⇑a) (Continuous ⇑b)
    -/
    exact ⟨a.continuous, b.continuous⟩
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pullback.lift_fst {Z : CompHausLike P} (a : Z ⟶ X) (b : Z ⟶ Y) (w : a ≫ f = b ≫ g) :
    pullback.lift f g a b w ≫ pullback.fst f g = a := rfl


@[reassoc (attr := simp)]
lemma pullback.lift_snd {Z : CompHausLike P} (a : Z ⟶ X) (b : Z ⟶ Y) (w : a ≫ f = b ≫ g) :
    pullback.lift f g a b w ≫ pullback.snd f g = b := rfl


lemma pullback.hom_ext {Z : CompHausLike P} (a b : Z ⟶ pullback f g)
    (hfst : a ≫ pullback.fst f g = b ≫ pullback.fst f g)
    (hsnd : a ≫ pullback.snd f g = b ≫ pullback.snd f g) : a = b := by
  /-
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    Z : CompHausLike P
    a b : Quiver.Hom Z (CompHausLike.pullback f g)
    hfst : Eq (CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.fst f g …
    hsnd : Eq (CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.snd f g …
    ⊢ Eq a b
  -/
  ext z
  /-
    case w
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    Z : CompHausLike P
    a b : Quiver.Hom Z (CompHausLike.pullback f g)
    hfst : Eq (CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.fst f g …
    hsnd : Eq (CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.snd f g …
    z : (CategoryTheory.forget (CompHausLike P)).obj Z
    ⊢ Eq (a z) (b z)
  -/
  apply_fun (fun q ↦ q z) at hfst hsnd
  /-
    case w
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    Z : CompHausLike P
    a b : Quiver.Hom Z (CompHausLike.pullback f g)
    z : (CategoryTheory.forget (CompHausLike P)).obj Z
    hfst : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.fst f  …
    hsnd : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.snd f  …
    ⊢ Eq (a z) (b z)
  -/
  apply Subtype.ext
  /-
    case w.a
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    Z : CompHausLike P
    a b : Quiver.Hom Z (CompHausLike.pullback f g)
    z : (CategoryTheory.forget (CompHausLike P)).obj Z
    hfst : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.fst f  …
    hsnd : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.snd f  …
    ⊢ Eq ↑(a z) ↑(b z)
  -/
  apply Prod.ext
    /-
      case w.a.fst
      P : TopCat → Prop
      X Y B : CompHausLike P
      f : Quiver.Hom X B
      g : Quiver.Hom Y B
      inst✝ : CompHausLike.HasExplicitPullback f g
      Z : CompHausLike P
      a b : Quiver.Hom Z (CompHausLike.pullback f g)
      z : (CategoryTheory.forget (CompHausLike P)).obj Z
      hfst : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.fst f  …
      hsnd : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.snd f  …
      ⊢ Eq (↑(a z)).1 (↑(b z)).1
    -/
  · exact hfst
    /-
      🎉 no goals
    -/
    /-
      case w.a.snd
      P : TopCat → Prop
      X Y B : CompHausLike P
      f : Quiver.Hom X B
      g : Quiver.Hom Y B
      inst✝ : CompHausLike.HasExplicitPullback f g
      Z : CompHausLike P
      a b : Quiver.Hom Z (CompHausLike.pullback f g)
      z : (CategoryTheory.forget (CompHausLike P)).obj Z
      hfst : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.fst f  …
      hsnd : Eq ((CategoryTheory.CategoryStruct.comp a (CompHausLike.pullback.snd f  …
      ⊢ Eq (↑(a z)).2 (↑(b z)).2
    -/
  · exact hsnd
    /-
      🎉 no goals
    -/


/--
The pullback cone whose cone point is the explicit pullback.
-/
@[simps! pt π]
def pullback.cone : Limits.PullbackCone f g :=
  Limits.PullbackCone.mk (pullback.fst f g) (pullback.snd f g) (pullback.condition f g)


/--
The explicit pullback cone is a limit cone.
-/
@[simps! lift]
def pullback.isLimit : Limits.IsLimit (pullback.cone f g) :=
  Limits.PullbackCone.isLimitAux _
    (fun s ↦ pullback.lift f g s.fst s.snd s.condition)
    (fun _ ↦ pullback.lift_fst _ _ _ _ _)
    (fun _ ↦ pullback.lift_snd _ _ _ _ _)
    (fun _ _ hm ↦ pullback.hom_ext _ _ _ _ (hm .left) (hm .right))


instance : HasLimit (cospan f g) where
  exists_limit := ⟨⟨pullback.cone f g, pullback.isLimit f g⟩⟩


/-- The functor to `TopCat` creates pullbacks if they exist. -/
noncomputable instance : CreatesLimit (cospan f g) (compHausLikeToTop P) := by
  refine createsLimitOfFullyFaithfulOfIso (pullback f g)
    (((TopCat.pullbackConeIsLimit f g).conePointUniqueUpToIso
        (limit.isLimit _)) ≪≫ Limits.lim.mapIso (?_ ≪≫ (diagramIsoCospan _).symm))
  /-
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.cospan f g) (CategoryTheory.Limits …
  -/
  exact Iso.refl _
  /-
    🎉 no goals
  -/


/-- The functor to `TopCat` preserves pullbacks. -/
noncomputable instance : PreservesLimit (cospan f g) (compHausLikeToTop P) :=
  preservesLimit_of_createsLimit_and_hasLimit _ _


/-- The functor to another `CompHausLike` preserves pullbacks. -/
noncomputable instance {P' : TopCat → Prop}
    (h : ∀ (X : CompHausLike P), P X.toTop → P' X.toTop) :
    PreservesLimit (cospan f g) (toCompHausLike h) := by
  have : PreservesLimit (cospan f g) (toCompHausLike h ⋙ compHausLikeToTop P') :=
    inferInstanceAs (PreservesLimit _ (compHausLikeToTop _))
  /-
    P : TopCat → Prop
    X Y B : CompHausLike P
    f : Quiver.Hom X B
    g : Quiver.Hom Y B
    inst✝ : CompHausLike.HasExplicitPullback f g
    P' : TopCat → Prop
    h : ∀ (X : CompHausLike P), P X.toTop → P' X.toTop
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) …
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan f g) (Com …
  -/
  exact preservesLimit_of_reflects_of_preserves (toCompHausLike h) (compHausLikeToTop P')
  /-
    🎉 no goals
  -/


variable (P) in
/--
A typeclass describing the property that forming all explicit pullbacks is stable under the
property `P`.
-/
class HasExplicitPullbacks : Prop where
  hasProp {X Y B : CompHausLike P} (f : X ⟶ B) (g : Y ⟶ B) : HasExplicitPullback f g


instance [HasExplicitPullbacks P] : HasPullbacks (CompHausLike P) where
  has_limit F := hasLimitOfIso (diagramIsoCospan F).symm


variable (P) in
/--
A typeclass describing the property that explicit pullbacks along inclusion maps into disjoint
unions is stable under the property `P`.
-/
class HasExplicitPullbacksOfInclusions [HasExplicitFiniteCoproducts.{0} P] : Prop where
  hasProp : ∀ {X Y Z : CompHausLike P} (f : Z ⟶ X ⨿ Y), HasExplicitPullback coprod.inl f


instance [HasExplicitPullbacks P] [HasExplicitFiniteCoproducts.{0} P] :
    HasExplicitPullbacksOfInclusions P where
  hasProp _ := inferInstance


instance [HasExplicitPullbacksOfInclusions P] : HasPullbacksOfInclusions (CompHausLike P) where
  hasPullbackInl _ := inferInstance


theorem hasPullbacksOfInclusions
    (hP' : ∀ ⦃X Y B : CompHausLike.{u} P⦄ (f : X ⟶ B) (g : Y ⟶ B)
      (_ : IsOpenEmbedding f), HasExplicitPullback f g) :
    HasExplicitPullbacksOfInclusions P :=
  { hasProp := by
      /-
        P : TopCat → Prop
        inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
        hP' : ∀ ⦃X Y B : CompHausLike P⦄ (f : Quiver.Hom X B) (g : Quiver.Hom Y B), To …
        ⊢ ∀ {X Y Z : CompHausLike P} (f : Quiver.Hom Z (CategoryTheory.Limits.coprod X …
      -/
      intro _ _ _ f
      /-
        P : TopCat → Prop
        inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
        hP' : ∀ ⦃X Y B : CompHausLike P⦄ (f : Quiver.Hom X B) (g : Quiver.Hom Y B), To …
        X✝ Y✝ Z✝ : CompHausLike P
        f : Quiver.Hom Z✝ (CategoryTheory.Limits.coprod X✝ Y✝)
        ⊢ CompHausLike.HasExplicitPullback CategoryTheory.Limits.coprod.inl f
      -/
      apply hP'
      /-
        case x
        P : TopCat → Prop
        inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
        hP' : ∀ ⦃X Y B : CompHausLike P⦄ (f : Quiver.Hom X B) (g : Quiver.Hom Y B), To …
        X✝ Y✝ Z✝ : CompHausLike P
        f : Quiver.Hom Z✝ (CategoryTheory.Limits.coprod X✝ Y✝)
        ⊢ Topology.IsOpenEmbedding ⇑CategoryTheory.Limits.coprod.inl
      -/
      exact Sigma.isOpenEmbedding_ι _ _ }
      /-
        🎉 no goals
      -/


/-- The functor to `TopCat` preserves pullbacks of inclusions if they exist. -/
noncomputable instance [HasExplicitPullbacksOfInclusions P] :
    PreservesPullbacksOfInclusions (compHausLikeToTop P) :=
  { preservesPullbackInl := by
      /-
        P : TopCat → Prop
        inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
        inst✝ : CompHausLike.HasExplicitPullbacksOfInclusions P
        ⊢ ∀ {X Y Z : CompHausLike P} (f : Quiver.Hom Z (CategoryTheory.Limits.coprod X …
      -/
      intros X Y Z f
      /-
        P : TopCat → Prop
        inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
        inst✝ : CompHausLike.HasExplicitPullbacksOfInclusions P
        X Y Z : CompHausLike P
        f : Quiver.Hom Z (CategoryTheory.Limits.coprod X Y)
        ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan CategoryT …
      -/
      infer_instance }
      /-
        🎉 no goals
      -/


instance [HasExplicitPullbacksOfInclusions P] : FinitaryExtensive (CompHausLike P) :=
  finitaryExtensive_of_preserves_and_reflects (compHausLikeToTop P)


theorem finitaryExtensive (hP' : ∀ ⦃X Y B : CompHausLike.{u} P⦄ (f : X ⟶ B) (g : Y ⟶ B)
    (_ : IsOpenEmbedding f), HasExplicitPullback f g) :
      FinitaryExtensive (CompHausLike P) :=
  have := hasPullbacksOfInclusions hP'
  finitaryExtensive_of_preserves_and_reflects (compHausLikeToTop P)


/-- A one-element space is terminal in `CompHaus` -/
def isTerminalPUnit [HasProp P PUnit.{u+1}] :
    IsTerminal (CompHausLike.of P PUnit.{u + 1}) :=
  haveI : ∀ X, Unique (X ⟶ CompHausLike.of P PUnit.{u + 1}) := fun _ ↦
    ⟨⟨⟨fun _ ↦ PUnit.unit, continuous_const⟩⟩, fun _ ↦ rfl⟩
  Limits.IsTerminal.ofUnique _


