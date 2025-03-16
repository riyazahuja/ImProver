/-- A morphism of sheaves of types is locally bijective iff it is an isomorphism.
(This is generalized below as `isLocallyBijective_iff_isIso`.) -/
private lemma isLocallyBijective_iff_isIso' :
    IsLocallyInjective f ∧ IsLocallySurjective f ↔ IsIso f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F G : CategoryTheory.Sheaf J (Type w)
    f : Quiver.Hom F G
    ⊢ Iff (And (CategoryTheory.Sheaf.IsLocallyInjective f) (CategoryTheory.Sheaf.I …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      ⊢ And (CategoryTheory.Sheaf.IsLocallyInjective f) (CategoryTheory.Sheaf.IsLoca …
    -/
  · rintro ⟨h₁, _⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      h₁ : CategoryTheory.Sheaf.IsLocallyInjective f
      right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
      ⊢ CategoryTheory.IsIso f
    -/
    rw [isLocallyInjective_iff_injective] at h₁
    suffices ∀ (X : Cᵒᵖ), Function.Surjective (f.val.app X) by
      rw [← isIso_iff_of_reflects_iso _ (sheafToPresheaf _ _), NatTrans.isIso_iff_isIso_app]
      intro X
      rw [isIso_iff_bijective]
      exact ⟨h₁ X, this X⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      h₁ : ∀ (X : Opposite C), Function.Injective ⇑(f.val.app X)
      right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
      ⊢ ∀ (X : Opposite C), Function.Surjective (f.val.app X)
    -/
    intro X s
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      h₁ : ∀ (X : Opposite C), Function.Injective ⇑(f.val.app X)
      right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
      X : Opposite C
      s : G.val.obj X
      ⊢ Exists fun a => Eq (f.val.app X a) s
    -/
    have H := (isSheaf_iff_isSheaf_of_type J F.val).1 F.cond _ (Presheaf.imageSieve_mem J f.val s)
    let t : Presieve.FamilyOfElements F.val (Presheaf.imageSieve f.val s).arrows :=
      fun Y g hg => Presheaf.localPreimage f.val s g hg
    have ht : t.Compatible := by
      intro Y₁ Y₂ W g₁ g₂ f₁ f₂ hf₁ hf₂ w
      apply h₁
      have eq₁ := FunctorToTypes.naturality _ _ f.val g₁.op (t f₁ hf₁)
      have eq₂ := FunctorToTypes.naturality _ _ f.val g₂.op (t f₂ hf₂)
      have eq₃ := congr_arg (G.val.map g₁.op) (Presheaf.app_localPreimage f.val s _ hf₁)
      have eq₄ := congr_arg (G.val.map g₂.op) (Presheaf.app_localPreimage f.val s _ hf₂)
      refine eq₁.trans (eq₃.trans (Eq.trans ?_ (eq₄.symm.trans eq₂.symm)))
      erw [← FunctorToTypes.map_comp_apply, ← FunctorToTypes.map_comp_apply]
      simp only [← op_comp, w]
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      h₁ : ∀ (X : Opposite C), Function.Injective ⇑(f.val.app X)
      right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
      X : Opposite C
      s : G.val.obj X
      H : CategoryTheory.Presieve.IsSheafFor F.val (CategoryTheory.Presheaf.imageSie …
      t : CategoryTheory.Presieve.FamilyOfElements F.val (CategoryTheory.Presheaf.im …
      ht : t.Compatible
      ⊢ Exists fun a => Eq (f.val.app X a) s
    -/
    refine ⟨H.amalgamate t ht, ?_⟩
    · apply (Presieve.isSeparated_of_isSheaf _ _
        ((isSheaf_iff_isSheaf_of_type J G.val).1 G.cond) _
        (Presheaf.imageSieve_mem J f.val s)).ext
      /-
        case mp.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F G : CategoryTheory.Sheaf J (Type w)
        f : Quiver.Hom F G
        h₁ : ∀ (X : Opposite C), Function.Injective ⇑(f.val.app X)
        right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
        X : Opposite C
        s : G.val.obj X
        H : CategoryTheory.Presieve.IsSheafFor F.val (CategoryTheory.Presheaf.imageSie …
        t : CategoryTheory.Presieve.FamilyOfElements F.val (CategoryTheory.Presheaf.im …
        ht : t.Compatible
        ⊢ ∀ ⦃Y : C⦄ ⦃f_1 : Quiver.Hom Y (Opposite.unop X)⦄, (CategoryTheory.Presheaf.i …
      -/
      intro Y g hg
      /-
        case mp.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F G : CategoryTheory.Sheaf J (Type w)
        f : Quiver.Hom F G
        h₁ : ∀ (X : Opposite C), Function.Injective ⇑(f.val.app X)
        right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
        X : Opposite C
        s : G.val.obj X
        H : CategoryTheory.Presieve.IsSheafFor F.val (CategoryTheory.Presheaf.imageSie …
        t : CategoryTheory.Presieve.FamilyOfElements F.val (CategoryTheory.Presheaf.im …
        ht : t.Compatible
        Y : C
        g : Quiver.Hom Y (Opposite.unop X)
        hg : (CategoryTheory.Presheaf.imageSieve f.val s).arrows g
        ⊢ Eq (G.val.map g.op (f.val.app X (H.amalgamate t ht))) (G.val.map g.op s)
      -/
      rw [← FunctorToTypes.naturality, H.valid_glue ht]
      /-
        case mp.intro
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J : CategoryTheory.GrothendieckTopology C
        F G : CategoryTheory.Sheaf J (Type w)
        f : Quiver.Hom F G
        h₁ : ∀ (X : Opposite C), Function.Injective ⇑(f.val.app X)
        right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
        X : Opposite C
        s : G.val.obj X
        H : CategoryTheory.Presieve.IsSheafFor F.val (CategoryTheory.Presheaf.imageSie …
        t : CategoryTheory.Presieve.FamilyOfElements F.val (CategoryTheory.Presheaf.im …
        ht : t.Compatible
        Y : C
        g : Quiver.Hom Y (Opposite.unop X)
        hg : (CategoryTheory.Presheaf.imageSieve f.val s).arrows g
        ⊢ Eq (f.val.app { unop := Y } (t g ?mp.intro.Hf)) (G.val.map g.op s)
      -/
      exact Presheaf.app_localPreimage f.val s g hg
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      ⊢ CategoryTheory.IsIso f → And (CategoryTheory.Sheaf.IsLocallyInjective f) (Ca …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F G : CategoryTheory.Sheaf J (Type w)
      f : Quiver.Hom F G
      a✝ : CategoryTheory.IsIso f
      ⊢ And (CategoryTheory.Sheaf.IsLocallyInjective f) (CategoryTheory.Sheaf.IsLoca …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> infer_instance
                    /-
                      🎉 no goals
                    -/


lemma isLocallyBijective_iff_isIso :
    IsLocallyInjective f ∧ IsLocallySurjective f ↔ IsIso f := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} A
    inst✝² : CategoryTheory.ConcreteCategory A
    F G : CategoryTheory.Sheaf J A
    f : Quiver.Hom F G
    inst✝¹ : (CategoryTheory.forget A).ReflectsIsomorphisms
    inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
    ⊢ Iff (And (CategoryTheory.Sheaf.IsLocallyInjective f) (CategoryTheory.Sheaf.I …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Sheaf J A
      f : Quiver.Hom F G
      inst✝¹ : (CategoryTheory.forget A).ReflectsIsomorphisms
      inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
      ⊢ And (CategoryTheory.Sheaf.IsLocallyInjective f) (CategoryTheory.Sheaf.IsLoca …
    -/
  · rintro ⟨_, _⟩
    rw [← isIso_iff_of_reflects_iso f (sheafCompose J (forget A)),
      ← isLocallyBijective_iff_isIso']
    /-
      case mp.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Sheaf J A
      f : Quiver.Hom F G
      inst✝¹ : (CategoryTheory.forget A).ReflectsIsomorphisms
      inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
      left✝ : CategoryTheory.Sheaf.IsLocallyInjective f
      right✝ : CategoryTheory.Sheaf.IsLocallySurjective f
      ⊢ And (CategoryTheory.Sheaf.IsLocallyInjective ((CategoryTheory.sheafCompose J …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> infer_instance
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Sheaf J A
      f : Quiver.Hom F G
      inst✝¹ : (CategoryTheory.forget A).ReflectsIsomorphisms
      inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
      ⊢ CategoryTheory.IsIso f → And (CategoryTheory.Sheaf.IsLocallyInjective f) (Ca …
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝³ : CategoryTheory.Category.{v', u'} A
      inst✝² : CategoryTheory.ConcreteCategory A
      F G : CategoryTheory.Sheaf J A
      f : Quiver.Hom F G
      inst✝¹ : (CategoryTheory.forget A).ReflectsIsomorphisms
      inst✝ : J.HasSheafCompose (CategoryTheory.forget A)
      a✝ : CategoryTheory.IsIso f
      ⊢ And (CategoryTheory.Sheaf.IsLocallyInjective f) (CategoryTheory.Sheaf.IsLoca …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> infer_instance
                    /-
                      🎉 no goals
                    -/


/-- Given a category `C` equipped with a Grothendieck topology `J` and a concrete category `A`,
this property holds if a morphism in `Cᵒᵖ ⥤ A` satisfies `J.W` (i.e. becomes an iso after
sheafification) iff it is both locally injective and locally surjective. -/
class WEqualsLocallyBijective : Prop where
  iff {X Y : Cᵒᵖ ⥤ A} (f : X ⟶ Y) :
    J.W f ↔ Presheaf.IsLocallyInjective J f ∧ Presheaf.IsLocallySurjective J f


lemma W_iff_isLocallyBijective :
    J.W f ↔ Presheaf.IsLocallyInjective J f ∧ Presheaf.IsLocallySurjective J f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    inst✝ : J.WEqualsLocallyBijective A
    X Y : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom X Y
    ⊢ Iff (J.W f) (And (CategoryTheory.Presheaf.IsLocallyInjective J f) (CategoryT …
  -/
  apply WEqualsLocallyBijective.iff
  /-
    🎉 no goals
  -/


lemma W_of_isLocallyBijective [Presheaf.IsLocallyInjective J f]
    [Presheaf.IsLocallySurjective J f] : J.W f := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} A
    inst✝³ : CategoryTheory.ConcreteCategory A
    inst✝² : J.WEqualsLocallyBijective A
    X Y : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f
    ⊢ J.W f
  -/
  rw [W_iff_isLocallyBijective]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} A
    inst✝³ : CategoryTheory.ConcreteCategory A
    inst✝² : J.WEqualsLocallyBijective A
    X Y : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J f
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J f
    ⊢ And (CategoryTheory.Presheaf.IsLocallyInjective J f) (CategoryTheory.Preshea …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> infer_instance
                  /-
                    🎉 no goals
                  -/


lemma W.isLocallyInjective (hf : J.W f) : Presheaf.IsLocallyInjective J f :=
  ((J.W_iff_isLocallyBijective f).1 hf).1


lemma W.isLocallySurjective (hf : J.W f) : Presheaf.IsLocallySurjective J f :=
  ((J.W_iff_isLocallyBijective f).1 hf).2


instance : Presheaf.IsLocallyInjective J (CategoryTheory.toSheafify J P) :=
  (J.W_toSheafify P).isLocallyInjective


instance : Presheaf.IsLocallySurjective J (CategoryTheory.toSheafify J P) :=
  (J.W_toSheafify P).isLocallySurjective


lemma WEqualsLocallyBijective.mk' [HasWeakSheafify J A] [(forget A).ReflectsIsomorphisms]
    [J.HasSheafCompose (forget A)]
    [∀ (P : Cᵒᵖ ⥤ A), Presheaf.IsLocallyInjective J (CategoryTheory.toSheafify J P)]
    [∀ (P : Cᵒᵖ ⥤ A), Presheaf.IsLocallySurjective J (CategoryTheory.toSheafify J P)] :
    J.WEqualsLocallyBijective A where
  iff {P Q} f := by
    rw [W_iff, ← Sheaf.isLocallyBijective_iff_isIso,
      ← Presheaf.isLocallyInjective_comp_iff J f (CategoryTheory.toSheafify J Q),
      ← Presheaf.isLocallySurjective_comp_iff J f (CategoryTheory.toSheafify J Q),
      CategoryTheory.toSheafify_naturality, Presheaf.comp_isLocallyInjective_iff,
      Presheaf.comp_isLocallySurjective_iff]


instance {D : Type w} [Category.{w'} D] [ConcreteCategory.{max u v} D]
    [HasWeakSheafify J D] [J.HasSheafCompose (forget D)]
    [J.PreservesSheafification (forget D)] [(forget D).ReflectsIsomorphisms] :
    J.WEqualsLocallyBijective D := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝⁷ : CategoryTheory.Category.{v', u'} A
    inst✝⁶ : CategoryTheory.ConcreteCategory A
    D : Type w
    inst✝⁵ : CategoryTheory.Category.{w', w} D
    inst✝⁴ : CategoryTheory.ConcreteCategory D
    inst✝³ : CategoryTheory.HasWeakSheafify J D
    inst✝² : J.HasSheafCompose (CategoryTheory.forget D)
    inst✝¹ : J.PreservesSheafification (CategoryTheory.forget D)
    inst✝ : (CategoryTheory.forget D).ReflectsIsomorphisms
    ⊢ J.WEqualsLocallyBijective D
  -/
  apply WEqualsLocallyBijective.mk'
  /-
    🎉 no goals
  -/


instance : J.WEqualsLocallyBijective (Type (max u v)) := inferInstance


lemma isLocallyInjective_presheafToSheaf_map_iff :
    Sheaf.IsLocallyInjective ((presheafToSheaf J A).map φ) ↔ IsLocallyInjective J φ := by
  rw [← Sheaf.isLocallyInjective_sheafToPresheaf_map_iff,
    ← isLocallyInjective_comp_iff J _ (toSheafify J Q),
    ← comp_isLocallyInjective_iff J (toSheafify J P),
    toSheafify_naturality, sheafToPresheaf_map]


lemma isLocallySurjective_presheafToSheaf_map_iff :
    Sheaf.IsLocallySurjective ((presheafToSheaf J A).map φ) ↔ IsLocallySurjective J φ := by
  rw [← Sheaf.isLocallySurjective_sheafToPresheaf_map_iff,
    ← isLocallySurjective_comp_iff J _ (toSheafify J Q),
    ← comp_isLocallySurjective_iff J (toSheafify J P),
    toSheafify_naturality, sheafToPresheaf_map]


