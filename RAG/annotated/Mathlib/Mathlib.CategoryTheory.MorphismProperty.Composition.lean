/-- Typeclass expressing that a morphism property contain identities. -/
class ContainsIdentities (W : MorphismProperty C) : Prop where
  /-- for all `X : C`, the identity of `X` satisfies the morphism property -/
  id_mem : ∀ (X : C), W (𝟙 X)


lemma id_mem (W : MorphismProperty C) [W.ContainsIdentities] (X : C) :
    W (𝟙 X) := ContainsIdentities.id_mem X


instance op (W : MorphismProperty C) [W.ContainsIdentities] :
    W.op.ContainsIdentities := ⟨fun X => W.id_mem X.unop⟩


instance unop (W : MorphismProperty Cᵒᵖ) [W.ContainsIdentities] :
    W.unop.ContainsIdentities := ⟨fun X => W.id_mem (Opposite.op X)⟩


lemma of_op (W : MorphismProperty C) [W.op.ContainsIdentities] :
    W.ContainsIdentities := (inferInstance : W.op.unop.ContainsIdentities)


lemma of_unop (W : MorphismProperty Cᵒᵖ) [W.unop.ContainsIdentities] :
    W.ContainsIdentities := (inferInstance : W.unop.op.ContainsIdentities)


instance inverseImage {P : MorphismProperty D} [P.ContainsIdentities] (F : C ⥤ D) :
    (P.inverseImage F).ContainsIdentities where
                 /-
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   D : Type u'
                   inst✝¹ : CategoryTheory.Category.{v', u'} D
                   P : CategoryTheory.MorphismProperty D
                   inst✝ : P.ContainsIdentities
                   F : CategoryTheory.Functor C D
                   X : C
                   ⊢ P.inverseImage F (CategoryTheory.CategoryStruct.id X)
                 -/
  id_mem X := by simpa only [← F.map_id] using P.id_mem (F.obj X)
                 /-
                   🎉 no goals
                 -/


instance inf {P Q : MorphismProperty C} [P.ContainsIdentities] [Q.ContainsIdentities] :
    (P ⊓ Q).ContainsIdentities where
  id_mem X := ⟨P.id_mem X, Q.id_mem X⟩


instance Prod.containsIdentities {C₁ C₂ : Type*} [Category C₁] [Category C₂]
    (W₁ : MorphismProperty C₁) (W₂ : MorphismProperty C₂)
    [W₁.ContainsIdentities] [W₂.ContainsIdentities] : (prod W₁ W₂).ContainsIdentities :=
  ⟨fun _ => ⟨W₁.id_mem _, W₂.id_mem _⟩⟩


instance Pi.containsIdentities {J : Type w} {C : J → Type u}
  [∀ j, Category.{v} (C j)] (W : ∀ j, MorphismProperty (C j)) [∀ j, (W j).ContainsIdentities] :
    (pi W).ContainsIdentities :=
  ⟨fun _ _ => MorphismProperty.id_mem _ _⟩


lemma of_isIso (P : MorphismProperty C) [P.ContainsIdentities] [P.RespectsIso] {X Y : C} (f : X ⟶ Y)
    [IsIso f] : P f :=
  Category.id_comp f ▸ RespectsIso.postcomp P f (𝟙 X) (P.id_mem X)


lemma isomorphisms_le_of_containsIdentities (P : MorphismProperty C) [P.ContainsIdentities]
    [P.RespectsIso] :
    isomorphisms C ≤ P := fun _ _ f (_ : IsIso f) ↦ P.of_isIso f


/-- A morphism property satisfies `IsStableUnderComposition` if the composition of
two such morphisms still falls in the class. -/
class IsStableUnderComposition (P : MorphismProperty C) : Prop where
  comp_mem {X Y Z} (f : X ⟶ Y) (g : Y ⟶ Z) : P f → P g → P (f ≫ g)


lemma comp_mem (W : MorphismProperty C) [W.IsStableUnderComposition]
    {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (hf : W f) (hg : W g) : W (f ≫ g) :=
  IsStableUnderComposition.comp_mem f g hf hg


instance (priority := 900) (W : MorphismProperty C) [W.IsStableUnderComposition] :
    W.Respects W where
  precomp _ hi _ hf := W.comp_mem _ _ hi hf
  postcomp _ hi _ hf := W.comp_mem _ _ hf hi


instance IsStableUnderComposition.op {P : MorphismProperty C} [P.IsStableUnderComposition] :
    P.op.IsStableUnderComposition where
  comp_mem f g hf hg := P.comp_mem g.unop f.unop hg hf


instance IsStableUnderComposition.unop {P : MorphismProperty Cᵒᵖ} [P.IsStableUnderComposition] :
    P.unop.IsStableUnderComposition where
  comp_mem f g hf hg := P.comp_mem g.op f.op hg hf


instance IsStableUnderComposition.inf {P Q : MorphismProperty C} [P.IsStableUnderComposition]
    [Q.IsStableUnderComposition] :
    (P ⊓ Q).IsStableUnderComposition where
  comp_mem f g hf hg := ⟨P.comp_mem f g hf.left hg.left, Q.comp_mem f g hf.right hg.right⟩


/-- A morphism property is `StableUnderInverse` if the inverse of a morphism satisfying
the property still falls in the class. -/
def StableUnderInverse (P : MorphismProperty C) : Prop :=
  ∀ ⦃X Y⦄ (e : X ≅ Y), P e.hom → P e.inv


theorem StableUnderInverse.op {P : MorphismProperty C} (h : StableUnderInverse P) :
    StableUnderInverse P.op := fun _ _ e he => h e.unop he


theorem StableUnderInverse.unop {P : MorphismProperty Cᵒᵖ} (h : StableUnderInverse P) :
    StableUnderInverse P.unop := fun _ _ e he => h e.op he


theorem respectsIso_of_isStableUnderComposition {P : MorphismProperty C}
    [P.IsStableUnderComposition] (hP : isomorphisms C ≤ P) :
    RespectsIso P := RespectsIso.mk _
  (fun _ _ hf => P.comp_mem _ _ (hP _ (isomorphisms.infer_property _)) hf)
    (fun _ _ hf => P.comp_mem _ _ hf (hP _ (isomorphisms.infer_property _)))


instance IsStableUnderComposition.inverseImage {P : MorphismProperty D} [P.IsStableUnderComposition]
    (F : C ⥤ D) : (P.inverseImage F).IsStableUnderComposition where
                           /-
                             C : Type u
                             inst✝² : CategoryTheory.Category.{v, u} C
                             D : Type u'
                             inst✝¹ : CategoryTheory.Category.{v', u'} D
                             P : CategoryTheory.MorphismProperty D
                             inst✝ : P.IsStableUnderComposition
                             F : CategoryTheory.Functor C D
                             X✝ Y✝ Z✝ : C
                             f : Quiver.Hom X✝ Y✝
                             g : Quiver.Hom Y✝ Z✝
                             hf : P.inverseImage F f
                             hg : P.inverseImage F g
                             ⊢ P.inverseImage F (CategoryTheory.CategoryStruct.comp f g)
                           -/
  comp_mem f g hf hg := by simpa only [← F.map_comp] using P.comp_mem _ _ hf hg
                           /-
                             🎉 no goals
                           -/


/-- Given `app : Π X, F₁.obj X ⟶ F₂.obj X` where `F₁` and `F₂` are two functors,
this is the `morphism_property C` satisfied by the morphisms in `C` with respect
to whom `app` is natural. -/
@[simp]
def naturalityProperty {F₁ F₂ : C ⥤ D} (app : ∀ X, F₁.obj X ⟶ F₂.obj X) : MorphismProperty C :=
  fun X Y f => F₁.map f ≫ app Y = app X ≫ F₂.map f


instance isStableUnderComposition {F₁ F₂ : C ⥤ D} (app : ∀ X, F₁.obj X ⟶ F₂.obj X) :
    (naturalityProperty app).IsStableUnderComposition where
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F₁ F₂ : CategoryTheory.Functor C D
      app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.naturalityProperty app f
      hg : CategoryTheory.MorphismProperty.naturalityProperty app g
      ⊢ CategoryTheory.MorphismProperty.naturalityProperty app (CategoryTheory.Categ …
    -/
    simp only [naturalityProperty] at hf hg ⊢
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F₁ F₂ : CategoryTheory.Functor C D
      app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : Eq (CategoryTheory.CategoryStruct.comp (F₁.map f) (app Y✝)) (CategoryTheo …
      hg : Eq (CategoryTheory.CategoryStruct.comp (F₁.map g) (app Z✝)) (CategoryTheo …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.map (CategoryTheory.CategoryStruc …
    -/
    simp only [Functor.map_comp, Category.assoc, hg]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F₁ F₂ : CategoryTheory.Functor C D
      app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : Eq (CategoryTheory.CategoryStruct.comp (F₁.map f) (app Y✝)) (CategoryTheo …
      hg : Eq (CategoryTheory.CategoryStruct.comp (F₁.map g) (app Z✝)) (CategoryTheo …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.map f) (CategoryTheory.CategorySt …
    -/
    slice_lhs 1 2 => rw [hf]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F₁ F₂ : CategoryTheory.Functor C D
      app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : Eq (CategoryTheory.CategoryStruct.comp (F₁.map f) (app Y✝)) (CategoryTheo …
      hg : Eq (CategoryTheory.CategoryStruct.comp (F₁.map g) (app Z✝)) (CategoryTheo …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Category.assoc]
    /-
      🎉 no goals
    -/


theorem stableUnderInverse {F₁ F₂ : C ⥤ D} (app : ∀ X, F₁.obj X ⟶ F₂.obj X) :
    (naturalityProperty app).StableUnderInverse := fun X Y e he => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F₁ F₂ : CategoryTheory.Functor C D
    app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
    X Y : C
    e : CategoryTheory.Iso X Y
    he : CategoryTheory.MorphismProperty.naturalityProperty app e.hom
    ⊢ CategoryTheory.MorphismProperty.naturalityProperty app e.inv
  -/
  simp only [naturalityProperty] at he ⊢
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F₁ F₂ : CategoryTheory.Functor C D
    app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
    X Y : C
    e : CategoryTheory.Iso X Y
    he : Eq (CategoryTheory.CategoryStruct.comp (F₁.map e.hom) (app Y)) (CategoryT …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.map e.inv) (app X)) (CategoryTheo …
  -/
  rw [← cancel_epi (F₁.map e.hom)]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F₁ F₂ : CategoryTheory.Functor C D
    app : (X : C) → Quiver.Hom (F₁.obj X) (F₂.obj X)
    X Y : C
    e : CategoryTheory.Iso X Y
    he : Eq (CategoryTheory.CategoryStruct.comp (F₁.map e.hom) (app Y)) (CategoryT …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.map e.hom) (CategoryTheory.Catego …
  -/
  slice_rhs 1 2 => rw [he]
  simp only [Category.assoc, ← F₁.map_comp_assoc, ← F₂.map_comp, e.hom_inv_id, Functor.map_id,
    Category.id_comp, Category.comp_id]


/-- A morphism property is multiplicative if it contains identities and is stable by
composition. -/
class IsMultiplicative (W : MorphismProperty C)
    extends W.ContainsIdentities, W.IsStableUnderComposition : Prop


instance op (W : MorphismProperty C) [IsMultiplicative W] : IsMultiplicative W.op where
  comp_mem f g hf hg := W.comp_mem g.unop f.unop hg hf


instance unop (W : MorphismProperty Cᵒᵖ) [IsMultiplicative W] : IsMultiplicative W.unop where
  id_mem _ := W.id_mem _
  comp_mem f g hf hg := W.comp_mem g.op f.op hg hf


lemma of_op (W : MorphismProperty C) [IsMultiplicative W.op] : IsMultiplicative W :=
  (inferInstance : IsMultiplicative W.op.unop)


lemma of_unop (W : MorphismProperty Cᵒᵖ) [IsMultiplicative W.unop] : IsMultiplicative W :=
  (inferInstance : IsMultiplicative W.unop.op)


instance : MorphismProperty.IsMultiplicative (⊤ : MorphismProperty C) where
  comp_mem _ _ _ _ := trivial
  id_mem _ := trivial


instance : (isomorphisms C).IsMultiplicative where
  id_mem _ := isomorphisms.infer_property _
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.isomorphisms C f
      hg : CategoryTheory.MorphismProperty.isomorphisms C g
      ⊢ CategoryTheory.MorphismProperty.isomorphisms C (CategoryTheory.CategoryStruc …
    -/
    rw [isomorphisms.iff] at hf hg ⊢
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.IsIso f
      hg : CategoryTheory.IsIso g
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance : (monomorphisms C).IsMultiplicative where
  id_mem _ := monomorphisms.infer_property _
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.monomorphisms C f
      hg : CategoryTheory.MorphismProperty.monomorphisms C g
      ⊢ CategoryTheory.MorphismProperty.monomorphisms C (CategoryTheory.CategoryStru …
    -/
    rw [monomorphisms.iff] at hf hg ⊢
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.Mono f
      hg : CategoryTheory.Mono g
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp f g)
    -/
    apply mono_comp
    /-
      🎉 no goals
    -/


instance : (epimorphisms C).IsMultiplicative where
  id_mem _ := epimorphisms.infer_property _
  comp_mem f g hf hg := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.MorphismProperty.epimorphisms C f
      hg : CategoryTheory.MorphismProperty.epimorphisms C g
      ⊢ CategoryTheory.MorphismProperty.epimorphisms C (CategoryTheory.CategoryStruc …
    -/
    rw [epimorphisms.iff] at hf hg ⊢
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      hf : CategoryTheory.Epi f
      hg : CategoryTheory.Epi g
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp f g)
    -/
    apply epi_comp
    /-
      🎉 no goals
    -/


instance {P : MorphismProperty D} [P.IsMultiplicative] (F : C ⥤ D) :
    (P.inverseImage F).IsMultiplicative where


instance inf {P Q : MorphismProperty C} [P.IsMultiplicative] [Q.IsMultiplicative] :
    (P ⊓ Q).IsMultiplicative where


/-- A class of morphisms `W` has the of-postcomp property wrt. `W'` if whenever
`g` is in `W'` and `f ≫ g` is in `W`, also `f` is in `W`. -/
class HasOfPostcompProperty (W W' : MorphismProperty C) : Prop where
  of_postcomp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) : W' g → W (f ≫ g) → W f


/-- A class of morphisms `W` has the of-precomp property wrt. `W'` if whenever
`f` is in `W'` and `f ≫ g` is in `W`, also `g` is in `W`. -/
class HasOfPrecompProperty (W W' : MorphismProperty C) : Prop where
  of_precomp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) : W' f → W (f ≫ g) → W g


/-- A class of morphisms `W` has the two-out-of-three property if whenever two out
of three maps in `f`, `g`, `f ≫ g` are in `W`, then the third map is also in `W`. -/
class HasTwoOutOfThreeProperty (W : MorphismProperty C)
    extends W.IsStableUnderComposition, W.HasOfPostcompProperty W,
      W.HasOfPrecompProperty W : Prop where


lemma of_postcomp [W.HasOfPostcompProperty W'] {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (hg : W' g)
    (hfg : W (f ≫ g)) : W f :=
  HasOfPostcompProperty.of_postcomp f g hg hfg


lemma of_precomp [W.HasOfPrecompProperty W'] {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (hf : W' f)
    (hfg : W (f ≫ g)) : W g :=
  HasOfPrecompProperty.of_precomp f g hf hfg


lemma postcomp_iff [W.RespectsRight W'] [W.HasOfPostcompProperty W']
    {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (hg : W' g) : W (f ≫ g) ↔ W f :=
  ⟨W.of_postcomp f g hg, fun hf ↦ RespectsRight.postcomp _ hg _ hf⟩


lemma precomp_iff [W.RespectsLeft W'] [W.HasOfPrecompProperty W']
    {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (hf : W' f) :
    W (f ≫ g) ↔ W g :=
  ⟨W.of_precomp f g hf, fun hg ↦ RespectsLeft.precomp _ hf _ hg⟩


instance : (isomorphisms C).HasTwoOutOfThreeProperty where
  of_postcomp f g := fun (hg : IsIso g) (hfg : IsIso (f ≫ g)) =>
       /-
         C : Type u
         inst✝¹ : CategoryTheory.Category.{v, u} C
         D : Type u'
         inst✝ : CategoryTheory.Category.{v', u'} D
         X✝ Y✝ Z✝ : C
         f : Quiver.Hom X✝ Y✝
         g : Quiver.Hom Y✝ Z✝
         hg : CategoryTheory.IsIso g
         hfg : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
         ⊢ CategoryTheory.MorphismProperty.isomorphisms C f
       -/
    by simpa using (inferInstance : IsIso ((f ≫ g) ≫ inv g))
       /-
         🎉 no goals
       -/
  of_precomp f g := fun (hf : IsIso f) (hfg : IsIso (f ≫ g)) =>
       /-
         C : Type u
         inst✝¹ : CategoryTheory.Category.{v, u} C
         D : Type u'
         inst✝ : CategoryTheory.Category.{v', u'} D
         X✝ Y✝ Z✝ : C
         f : Quiver.Hom X✝ Y✝
         g : Quiver.Hom Y✝ Z✝
         hf : CategoryTheory.IsIso f
         hfg : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f g)
         ⊢ CategoryTheory.MorphismProperty.isomorphisms C g
       -/
    by simpa using (inferInstance : IsIso (inv f ≫ (f ≫ g)))
       /-
         🎉 no goals
       -/


instance (F : C ⥤ D) (W : MorphismProperty D) [W.HasTwoOutOfThreeProperty] :
    (W.inverseImage F).HasTwoOutOfThreeProperty where
                                                                     /-
                                                                       C : Type u
                                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                                       D : Type u'
                                                                       inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                                       F : CategoryTheory.Functor C D
                                                                       W : CategoryTheory.MorphismProperty D
                                                                       inst✝ : W.HasTwoOutOfThreeProperty
                                                                       X✝ Y✝ Z✝ : C
                                                                       f : Quiver.Hom X✝ Y✝
                                                                       g : Quiver.Hom Y✝ Z✝
                                                                       hg : W.inverseImage F g
                                                                       hfg : W.inverseImage F (CategoryTheory.CategoryStruct.comp f g)
                                                                       ⊢ W (CategoryTheory.CategoryStruct.comp (F.map f) (F.map g))
                                                                     -/
  of_postcomp f g hg hfg := W.of_postcomp (F.map f) (F.map g) hg (by simpa using hfg)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                                                   /-
                                                                     C : Type u
                                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                                     D : Type u'
                                                                     inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                                     F : CategoryTheory.Functor C D
                                                                     W : CategoryTheory.MorphismProperty D
                                                                     inst✝ : W.HasTwoOutOfThreeProperty
                                                                     X✝ Y✝ Z✝ : C
                                                                     f : Quiver.Hom X✝ Y✝
                                                                     g : Quiver.Hom Y✝ Z✝
                                                                     hf : W.inverseImage F f
                                                                     hfg : W.inverseImage F (CategoryTheory.CategoryStruct.comp f g)
                                                                     ⊢ W (CategoryTheory.CategoryStruct.comp (F.map f) (F.map g))
                                                                   -/
  of_precomp f g hf hfg := W.of_precomp (F.map f) (F.map g) hf (by simpa using hfg)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


