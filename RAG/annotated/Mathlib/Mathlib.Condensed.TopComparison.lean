/--
An auxiliary lemma to that allows us to use `IsQuotientMap.lift` in the proof of
`equalizerCondition_yonedaPresheaf`.
-/
theorem factorsThrough_of_pullbackCondition {Z B : C} {π : Z ⟶ B} [HasPullback π π]
    [PreservesLimit (cospan π π) G]
    {a : C(G.obj Z, X)}
    (ha : a ∘ (G.map (pullback.fst _ _)) = a ∘ (G.map (pullback.snd π π))) :
    Function.FactorsThrough a (G.map π) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝² : TopologicalSpace X
    Z B : C
    π : Quiver.Hom Z B
    inst✝¹ : CategoryTheory.Limits.HasPullback π π
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan π π …
    a : ContinuousMap (↑(G.obj Z)) X
    ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
    ⊢ Function.FactorsThrough ⇑a ⇑(G.map π)
  -/
  intro x y hxy
  let xy : G.obj (pullback π π) := (PreservesPullback.iso G π π).inv <|
    (TopCat.pullbackIsoProdSubtype (G.map π) (G.map π)).inv ⟨(x, y), hxy⟩
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝² : TopologicalSpace X
    Z B : C
    π : Quiver.Hom Z B
    inst✝¹ : CategoryTheory.Limits.HasPullback π π
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan π π …
    a : ContinuousMap (↑(G.obj Z)) X
    ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
    x y : ↑(G.obj Z)
    hxy : Eq ((G.map π) x) ((G.map π) y)
    xy : ↑(G.obj (CategoryTheory.Limits.pullback π π)) := (CategoryTheory.Limits.P …
    ⊢ Eq (a x) (a y)
  -/
  have ha' := congr_fun ha xy
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝² : TopologicalSpace X
    Z B : C
    π : Quiver.Hom Z B
    inst✝¹ : CategoryTheory.Limits.HasPullback π π
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan π π …
    a : ContinuousMap (↑(G.obj Z)) X
    ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
    x y : ↑(G.obj Z)
    hxy : Eq ((G.map π) x) ((G.map π) y)
    xy : ↑(G.obj (CategoryTheory.Limits.pullback π π)) := (CategoryTheory.Limits.P …
    ha' : Eq (Function.comp (⇑a) (⇑(G.map (CategoryTheory.Limits.pullback.fst π π) …
    ⊢ Eq (a x) (a y)
  -/
  dsimp at ha'
  have h₁ : ∀ y, G.map (pullback.fst _ _) ((PreservesPullback.iso G π π).inv y) =
      pullback.fst (G.map π) (G.map π) y := by
    simp only [← PreservesPullback.iso_inv_fst]; intro y; rfl
  have h₂ : ∀ y, G.map (pullback.snd _ _) ((PreservesPullback.iso G π π).inv y) =
      pullback.snd (G.map π) (G.map π) y := by
    simp only [← PreservesPullback.iso_inv_snd]; intro y; rfl
  rw [h₁, h₂, TopCat.pullbackIsoProdSubtype_inv_fst_apply,
    TopCat.pullbackIsoProdSubtype_inv_snd_apply] at ha'
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝² : TopologicalSpace X
    Z B : C
    π : Quiver.Hom Z B
    inst✝¹ : CategoryTheory.Limits.HasPullback π π
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.cospan π π …
    a : ContinuousMap (↑(G.obj Z)) X
    ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
    x y : ↑(G.obj Z)
    hxy : Eq ((G.map π) x) ((G.map π) y)
    xy : ↑(G.obj (CategoryTheory.Limits.pullback π π)) := (CategoryTheory.Limits.P …
    ha' : Eq (a (↑⟨{ fst := x, snd := y }, hxy⟩).1) (a (↑⟨{ fst := x, snd := y },  …
    h₁ : ∀ (y : ↑(CategoryTheory.Limits.pullback (G.map π) (G.map π))), Eq ((G.map …
    h₂ : ∀ (y : ↑(CategoryTheory.Limits.pullback (G.map π) (G.map π))), Eq ((G.map …
    ⊢ Eq (a x) (a y)
  -/
  simpa using ha'
  /-
    🎉 no goals
  -/


/--
If `G` preserves the relevant pullbacks and every effective epi in `C` is a quotient map (which is
the case when `C` is `CompHaus` or `Profinite`), then `yonedaPresheaf` satisfies the equalizer
condition which is required to be a sheaf for the regular topology.
-/
theorem equalizerCondition_yonedaPresheaf
    [∀ (Z B : C) (π : Z ⟶ B) [EffectiveEpi π], PreservesLimit (cospan π π) G]
    (hq : ∀ (Z B : C) (π : Z ⟶ B) [EffectiveEpi π], IsQuotientMap (G.map π)) :
      EqualizerCondition (yonedaPresheaf G X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝¹ : TopologicalSpace X
    inst✝ : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π …
    hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
    ⊢ CategoryTheory.regularTopology.EqualizerCondition (ContinuousMap.yonedaPresh …
  -/
  apply EqualizerCondition.mk
  /-
    case hP
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝¹ : TopologicalSpace X
    inst✝ : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π …
    hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
    ⊢ ∀ (X_1 B : C) (π : Quiver.Hom X_1 B) [inst : CategoryTheory.EffectiveEpi π]  …
  -/
  intro Z B π _ _
  /-
    case hP
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor C TopCat
    X : Type w'
    inst✝³ : TopologicalSpace X
    inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
    hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
    Z B : C
    π : Quiver.Hom Z B
    inst✝¹ : CategoryTheory.EffectiveEpi π
    inst✝ : CategoryTheory.Limits.HasPullback π π
    ⊢ Function.Bijective (CategoryTheory.regularTopology.MapToEqualizer (Continuou …
  -/
  refine ⟨fun a b h ↦ ?_, fun ⟨a, ha⟩ ↦ ?_⟩
  · simp only [yonedaPresheaf, unop_op, Quiver.Hom.unop_op, Set.coe_setOf, MapToEqualizer,
      Set.mem_setOf_eq, Subtype.mk.injEq, comp, ContinuousMap.mk.injEq] at h
    /-
      case hP.refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      a b : (ContinuousMap.yonedaPresheaf G X).obj { unop := B }
      h : Eq (Function.comp ⇑a ⇑(G.map π)) (Function.comp ⇑b ⇑(G.map π))
      ⊢ Eq a b
    -/
    simp only [yonedaPresheaf, unop_op]
    /-
      case hP.refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      a b : (ContinuousMap.yonedaPresheaf G X).obj { unop := B }
      h : Eq (Function.comp ⇑a ⇑(G.map π)) (Function.comp ⇑b ⇑(G.map π))
      ⊢ Eq a b
    -/
    ext x
    /-
      case hP.refine_1.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      a b : (ContinuousMap.yonedaPresheaf G X).obj { unop := B }
      h : Eq (Function.comp ⇑a ⇑(G.map π)) (Function.comp ⇑b ⇑(G.map π))
      x : ↑(G.obj (Opposite.unop { unop := B }))
      ⊢ Eq (a x) (b x)
    -/
    obtain ⟨y, hy⟩ := (hq Z B π).surjective x
    /-
      case hP.refine_1.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      a b : (ContinuousMap.yonedaPresheaf G X).obj { unop := B }
      h : Eq (Function.comp ⇑a ⇑(G.map π)) (Function.comp ⇑b ⇑(G.map π))
      x : ↑(G.obj (Opposite.unop { unop := B }))
      y : ↑(G.obj Z)
      hy : Eq ((G.map π) y) x
      ⊢ Eq (a x) (b x)
    -/
    rw [← hy]
    /-
      case hP.refine_1.h.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      a b : (ContinuousMap.yonedaPresheaf G X).obj { unop := B }
      h : Eq (Function.comp ⇑a ⇑(G.map π)) (Function.comp ⇑b ⇑(G.map π))
      x : ↑(G.obj (Opposite.unop { unop := B }))
      y : ↑(G.obj Z)
      hy : Eq ((G.map π) y) x
      ⊢ Eq (a ((G.map π) y)) (b ((G.map π) y))
    -/
    exact congr_fun h y
    /-
      🎉 no goals
    -/
  · simp only [yonedaPresheaf, comp, unop_op, Quiver.Hom.unop_op, Set.mem_setOf_eq,
      ContinuousMap.mk.injEq] at ha
    simp only [yonedaPresheaf, comp, unop_op, Quiver.Hom.unop_op, Set.coe_setOf,
      MapToEqualizer, Set.mem_setOf_eq, Subtype.mk.injEq]
    /-
      case hP.refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      x✝ : ↑(setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).map (CategoryTheo …
      a : (ContinuousMap.yonedaPresheaf G X).obj { unop := Z }
      ha✝ : Membership.mem (setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).ma …
      ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
      ⊢ Exists fun a_1 => Eq { toFun := Function.comp ⇑a_1 ⇑(G.map π), continuous_to …
    -/
    simp only [yonedaPresheaf, unop_op] at a
    /-
      case hP.refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      x✝ : ↑(setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).map (CategoryTheo …
      a : ContinuousMap (↑(G.obj Z)) X
      ha✝ : Membership.mem (setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).ma …
      ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
      ⊢ Exists fun a_1 => Eq { toFun := Function.comp ⇑a_1 ⇑(G.map π), continuous_to …
    -/
    refine ⟨(hq Z B π).lift a (factorsThrough_of_pullbackCondition G X ha), ?_⟩
    /-
      case hP.refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      x✝ : ↑(setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).map (CategoryTheo …
      a : ContinuousMap (↑(G.obj Z)) X
      ha✝ : Membership.mem (setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).ma …
      ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
      ⊢ Eq { toFun := Function.comp ⇑(⋯.lift a ⋯) ⇑(G.map π), continuous_toFun := ⋯  …
    -/
    congr
    /-
      case hP.refine_2.e_toFun
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X : Type w'
      inst✝³ : TopologicalSpace X
      inst✝² : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi  …
      hq : ∀ (Z B : C) (π : Quiver.Hom Z B) [inst : CategoryTheory.EffectiveEpi π],  …
      Z B : C
      π : Quiver.Hom Z B
      inst✝¹ : CategoryTheory.EffectiveEpi π
      inst✝ : CategoryTheory.Limits.HasPullback π π
      x✝ : ↑(setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).map (CategoryTheo …
      a : ContinuousMap (↑(G.obj Z)) X
      ha✝ : Membership.mem (setOf fun x => Eq ((ContinuousMap.yonedaPresheaf G X).ma …
      ha : Eq (Function.comp ⇑a ⇑(G.map (CategoryTheory.Limits.pullback.fst π π))) ( …
      ⊢ Eq (Function.comp ⇑(⋯.lift a ⋯) ⇑(G.map π)) a.toFun
    -/
    exact DFunLike.ext'_iff.mp ((hq Z B π).lift_comp a (factorsThrough_of_pullbackCondition G X ha))
    /-
      🎉 no goals
    -/


/--
If `G` preserves finite coproducts (which is the case when `C` is `CompHaus`, `Profinite` or
`Stonean`), then `yonedaPresheaf` preserves finite products, which is required to be a sheaf for
the extensive topology.
-/
noncomputable instance [PreservesFiniteCoproducts G] :
    PreservesFiniteProducts (yonedaPresheaf G X) :=
  have := preservesFiniteProducts_op G
  ⟨fun _ ↦ comp_preservesLimitsOfShape G.op (yonedaPresheaf' X)⟩


/--
The sheaf on `CompHausLike P` of continuous maps to a topological space.
-/
@[simps! val_obj val_map]
def TopCat.toSheafCompHausLike :
    have := CompHausLike.preregular hs
    Sheaf (coherentTopology (CompHausLike.{u} P)) (Type (max u w)) where
  val := yonedaPresheaf.{u, max u w} (CompHausLike.compHausLikeToTop.{u} P) X
  cond := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X✝ : Type w'
      inst✝² : TopologicalSpace X✝
      P : TopCat → Prop
      X : TopCat
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology (CompHausLi …
    -/
    have := CompHausLike.preregular hs
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X✝ : Type w'
      inst✝² : TopologicalSpace X✝
      P : TopCat → Prop
      X : TopCat
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      this : CategoryTheory.Preregular (CompHausLike P)
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology (CompHausLi …
    -/
    rw [Presheaf.isSheaf_iff_preservesFiniteProducts_and_equalizerCondition]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X✝ : Type w'
      inst✝² : TopologicalSpace X✝
      P : TopCat → Prop
      X : TopCat
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      this : CategoryTheory.Preregular (CompHausLike P)
      ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts (ContinuousMap.yonedaPres …
    -/
    refine ⟨inferInstance, ?_⟩
    apply (config := { allowSynthFailures := true }) equalizerCondition_yonedaPresheaf
      (CompHausLike.compHausLikeToTop.{u} P) X
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X✝ : Type w'
      inst✝² : TopologicalSpace X✝
      P : TopCat → Prop
      X : TopCat
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      this : CategoryTheory.Preregular (CompHausLike P)
      ⊢ ∀ (Z B : CompHausLike P) (π : Quiver.Hom Z B) [inst : CategoryTheory.Effecti …
    -/
    intro Z B π he
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor C TopCat
      X✝ : Type w'
      inst✝² : TopologicalSpace X✝
      P : TopCat → Prop
      X : TopCat
      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      this : CategoryTheory.Preregular (CompHausLike P)
      Z B : CompHausLike P
      π : Quiver.Hom Z B
      he : CategoryTheory.EffectiveEpi π
      ⊢ Topology.IsQuotientMap ⇑((CompHausLike.compHausLikeToTop P).map π)
    -/
    apply IsQuotientMap.of_surjective_continuous (hs _ he) π.continuous
    /-
      🎉 no goals
    -/


/--
`TopCat.toSheafCompHausLike` yields a functor from `TopCat.{max u w}` to
`Sheaf (coherentTopology (CompHausLike.{u} P)) (Type (max u w))`.
-/
@[simps]
noncomputable def topCatToSheafCompHausLike :
    have := CompHausLike.preregular hs
    TopCat.{max u w} ⥤ Sheaf (coherentTopology (CompHausLike.{u} P)) (Type (max u w)) where
  obj X := X.toSheafCompHausLike P hs
                                    /-
                                      C : Type u
                                      inst✝³ : CategoryTheory.Category.{v, u} C
                                      G : CategoryTheory.Functor C TopCat
                                      X✝¹ : Type w'
                                      inst✝² : TopologicalSpace X✝¹
                                      P : TopCat → Prop
                                      X : TopCat
                                      inst✝¹ : CompHausLike.HasExplicitFiniteCoproducts P
                                      inst✝ : CompHausLike.HasExplicitPullbacks P
                                      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
                                      X✝ Y✝ : TopCat
                                      f : Quiver.Hom X✝ Y✝
                                      ⊢ ∀ ⦃X Y : Opposite (CompHausLike P)⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheo …
                                    -/
  map f := ⟨⟨fun _ g ↦ f.comp g, by aesop⟩⟩
                                    /-
                                      🎉 no goals
                                    -/


/--
Associate to a `(u+1)`-small topological space the corresponding condensed set, given by
`yonedaPresheaf`.
-/
noncomputable abbrev TopCat.toCondensedSet (X : TopCat.{u+1}) : CondensedSet.{u} :=
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                G : CategoryTheory.Functor C TopCat
                                                X✝ : Type w'
                                                inst✝ : TopologicalSpace X✝
                                                X : TopCat
                                                x✝² x✝¹ : CompHausLike fun x => True
                                                x✝ : Quiver.Hom x✝² x✝¹
                                                ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (?m.29690 x✝² x✝¹ x✝)) (List.con …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  toSheafCompHausLike.{u+1} _ X (fun _ _ _ ↦ ((CompHaus.effectiveEpi_tfae _).out 0 2).mp)
                                              /-
                                                🎉 no goals
                                              -/


/--
`TopCat.toCondensedSet` yields a functor from `TopCat.{u+1}` to `CondensedSet.{u}`.
-/
noncomputable abbrev topCatToCondensedSet : TopCat.{u+1} ⥤ CondensedSet.{u} :=
                                                  /-
                                                    C : Type u
                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                    G : CategoryTheory.Functor C TopCat
                                                    X : Type w'
                                                    inst✝ : TopologicalSpace X
                                                    x✝² x✝¹ : CompHausLike fun x => True
                                                    x✝ : Quiver.Hom x✝² x✝¹
                                                    ⊢ Eq ((List.cons (CategoryTheory.EffectiveEpi (?m.29942 x✝² x✝¹ x✝)) (List.con …
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  topCatToSheafCompHausLike.{u+1} _ (fun _ _ _ ↦ ((CompHaus.effectiveEpi_tfae _).out 0 2).mp)
                                                  /-
                                                    🎉 no goals
                                                  -/

