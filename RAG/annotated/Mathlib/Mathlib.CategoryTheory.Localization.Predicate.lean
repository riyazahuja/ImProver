/-- The predicate expressing that, up to equivalence, a functor `L : C ⥤ D`
identifies the category `D` with the localized category of `C` with respect
to `W : MorphismProperty C`. -/
class IsLocalization : Prop where
  /-- the functor inverts the given `MorphismProperty` -/
  inverts : W.IsInvertedBy L
  /-- the induced functor from the constructed localized category is an equivalence -/
  isEquivalence : IsEquivalence (Localization.Construction.lift L inverts)


instance q_isLocalization : W.Q.IsLocalization W where
  inverts := W.Q_inverts
  isEquivalence := by
    suffices Localization.Construction.lift W.Q W.Q_inverts = 𝟭 _ by
      rw [this]
      infer_instance
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.733, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.781, u_3} E
      ⊢ Eq (CategoryTheory.Localization.Construction.lift W.Q ⋯) (CategoryTheory.Fun …
    -/
    apply Localization.Construction.uniq
    /-
      case h
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.733, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.781, u_3} E
      ⊢ Eq (W.Q.comp (CategoryTheory.Localization.Construction.lift W.Q ⋯)) (W.Q.com …
    -/
    simp only [Localization.Construction.fac]
    /-
      case h
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.733, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.781, u_3} E
      ⊢ Eq W.Q (W.Q.comp (CategoryTheory.Functor.id W.Localization))
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- This universal property states that a functor `L : C ⥤ D` inverts morphisms
in `W` and the all functors `D ⥤ E` (for a fixed category `E`) uniquely factors
through `L`. -/
structure StrictUniversalPropertyFixedTarget where
  /-- the functor `L` inverts `W` -/
  inverts : W.IsInvertedBy L
  /-- any functor `C ⥤ E` which inverts `W` can be lifted as a functor `D ⥤ E`  -/
  lift : ∀ (F : C ⥤ E) (_ : W.IsInvertedBy F), D ⥤ E
  /-- there is a factorisation involving the lifted functor  -/
  fac : ∀ (F : C ⥤ E) (hF : W.IsInvertedBy F), L ⋙ lift F hF = F
  /-- uniqueness of the lifted functor -/
  uniq : ∀ (F₁ F₂ : D ⥤ E) (_ : L ⋙ F₁ = L ⋙ F₂), F₁ = F₂


/-- The localized category `W.Localization` that was constructed satisfies
the universal property of the localization. -/
@[simps]
def strictUniversalPropertyFixedTargetQ : StrictUniversalPropertyFixedTarget W.Q W E where
  inverts := W.Q_inverts
  lift := Construction.lift
  fac := Construction.fac
  uniq := Construction.uniq


instance : Inhabited (StrictUniversalPropertyFixedTarget W.Q W E) :=
  ⟨strictUniversalPropertyFixedTargetQ _ _⟩


/-- When `W` consists of isomorphisms, the identity satisfies the universal property
of the localization. -/
@[simps]
def strictUniversalPropertyFixedTargetId (hW : W ≤ MorphismProperty.isomorphisms C) :
    StrictUniversalPropertyFixedTarget (𝟭 C) W E where
  inverts _ _ f hf := hW f hf
  lift F _ := F
  fac F hF := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.4553, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.4557, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.4605, u_3} E
      hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
      F : CategoryTheory.Functor C E
      hF : W.IsInvertedBy F
      ⊢ Eq ((CategoryTheory.Functor.id C).comp ((fun F x => F) F hF)) F
    -/
    cases F
    /-
      case mk
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.4553, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.4557, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.4605, u_3} E
      hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
      toPrefunctor✝ : Prefunctor C E
      map_id✝ : ∀ (X : C), Eq (toPrefunctor✝.map (CategoryTheory.CategoryStruct.id X …
      map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq (toPre …
      hF : W.IsInvertedBy { toPrefunctor := toPrefunctor✝, map_id := map_id✝, map_co …
      ⊢ Eq ((CategoryTheory.Functor.id C).comp ((fun F x => F) { toPrefunctor := toP …
    -/
    rfl
    /-
      🎉 no goals
    -/
  uniq F₁ F₂ eq := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.4553, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.4557, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.4605, u_3} E
      hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
      F₁ F₂ : CategoryTheory.Functor C E
      eq : Eq ((CategoryTheory.Functor.id C).comp F₁) ((CategoryTheory.Functor.id C) …
      ⊢ Eq F₁ F₂
    -/
    cases F₁
    /-
      case mk
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.4553, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.4557, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.4605, u_3} E
      hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
      F₂ : CategoryTheory.Functor C E
      toPrefunctor✝ : Prefunctor C E
      map_id✝ : ∀ (X : C), Eq (toPrefunctor✝.map (CategoryTheory.CategoryStruct.id X …
      map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq (toPre …
      eq : Eq ((CategoryTheory.Functor.id C).comp { toPrefunctor := toPrefunctor✝, m …
      ⊢ Eq { toPrefunctor := toPrefunctor✝, map_id := map_id✝, map_comp := map_comp✝ …
    -/
    cases F₂
    /-
      case mk.mk
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.4553, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.4557, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.4605, u_3} E
      hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
      toPrefunctor✝¹ : Prefunctor C E
      map_id✝¹ : ∀ (X : C), Eq (toPrefunctor✝¹.map (CategoryTheory.CategoryStruct.id …
      map_comp✝¹ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq (toPr …
      toPrefunctor✝ : Prefunctor C E
      map_id✝ : ∀ (X : C), Eq (toPrefunctor✝.map (CategoryTheory.CategoryStruct.id X …
      map_comp✝ : ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq (toPre …
      eq : Eq ((CategoryTheory.Functor.id C).comp { toPrefunctor := toPrefunctor✝¹,  …
      ⊢ Eq { toPrefunctor := toPrefunctor✝¹, map_id := map_id✝¹, map_comp := map_com …
    -/
    exact eq
    /-
      🎉 no goals
    -/


theorem IsLocalization.mk' (h₁ : Localization.StrictUniversalPropertyFixedTarget L W D)
    (h₂ : Localization.StrictUniversalPropertyFixedTarget L W W.Localization) :
    IsLocalization L W :=
  { inverts := h₁.inverts
    isEquivalence := IsEquivalence.mk' (h₂.lift W.Q W.Q_inverts)
      (eqToIso (Localization.Construction.uniq _ _ (by
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
          inst✝ : CategoryTheory.Category.{u_5, u_2} D
          L : CategoryTheory.Functor C D
          W : CategoryTheory.MorphismProperty C
          h₁ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W D
          h₂ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W W.Loca …
          ⊢ Eq (W.Q.comp (CategoryTheory.Functor.id W.Localization)) (W.Q.comp ((Categor …
        -/
        simp only [← Functor.assoc, Localization.Construction.fac, h₂.fac, Functor.comp_id])))
        /-
          🎉 no goals
        -/
      (eqToIso (h₁.uniq _ _ (by
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
          inst✝ : CategoryTheory.Category.{u_5, u_2} D
          L : CategoryTheory.Functor C D
          W : CategoryTheory.MorphismProperty C
          h₁ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W D
          h₂ : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W W.Loca …
          ⊢ Eq (L.comp ((h₂.lift W.Q ⋯).comp (CategoryTheory.Localization.Construction.l …
        -/
        simp only [← Functor.assoc, h₂.fac, Localization.Construction.fac, Functor.comp_id]))) }
        /-
          🎉 no goals
        -/


theorem IsLocalization.for_id (hW : W ≤ MorphismProperty.isomorphisms C) : (𝟭 C).IsLocalization W :=
  IsLocalization.mk' _ _ (Localization.strictUniversalPropertyFixedTargetId W _ hW)
    (Localization.strictUniversalPropertyFixedTargetId W _ hW)


theorem inverts : W.IsInvertedBy L :=
  (inferInstance : L.IsLocalization W).inverts


/-- The isomorphism `L.obj X ≅ L.obj Y` that is deduced from a morphism `f : X ⟶ Y` which
belongs to `W`, when `L.IsLocalization W`. -/
@[simps! hom]
def isoOfHom {X Y : C} (f : X ⟶ Y) (hf : W f) : L.obj X ≅ L.obj Y :=
  haveI : IsIso (L.map f) := inverts L W f hf
  asIso (L.map f)


@[reassoc (attr := simp)]
lemma isoOfHom_hom_inv_id {X Y : C} (f : X ⟶ Y) (hf : W f) :
    L.map f ≫ (isoOfHom L W f hf).inv = 𝟙 _ :=
  (isoOfHom L W f hf).hom_inv_id


@[reassoc (attr := simp)]
lemma isoOfHom_inv_hom_id {X Y : C} (f : X ⟶ Y) (hf : W f) :
    (isoOfHom L W f hf).inv ≫ L.map f = 𝟙 _ :=
  (isoOfHom L W f hf).inv_hom_id


@[simp]
lemma isoOfHom_id_inv (X : C) (hX : W (𝟙 X)) :
    (isoOfHom L W (𝟙 X) hX).inv = 𝟙 _ := by
  rw [← cancel_mono (isoOfHom L W (𝟙 X) hX).hom, Iso.inv_hom_id, id_comp,
    isoOfHom_hom, Functor.map_id]


lemma Construction.wIso_eq_isoOfHom {X Y : C} (f : X ⟶ Y) (hf : W f) :
                                                       /-
                                                         C : Type u_1
                                                         inst✝ : CategoryTheory.Category.{u_4, u_1} C
                                                         W : CategoryTheory.MorphismProperty C
                                                         X Y : C
                                                         f : Quiver.Hom X Y
                                                         hf : W f
                                                         ⊢ Eq (CategoryTheory.Localization.Construction.wIso f hf) (CategoryTheory.Loca …
                                                       -/
    Construction.wIso f hf = isoOfHom W.Q W f hf := by ext; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma Construction.wInv_eq_isoOfHom_inv {X Y : C} (f : X ⟶ Y) (hf : W f) :
    Construction.wInv f hf = (isoOfHom W.Q W f hf).inv :=
  congr_arg Iso.inv (wIso_eq_isoOfHom f hf)


instance : (Localization.Construction.lift L (inverts L W)).IsEquivalence :=
  (inferInstance : L.IsLocalization W).isEquivalence


/-- A chosen equivalence of categories `W.Localization ≅ D` for a functor
`L : C ⥤ D` which satisfies `L.IsLocalization W`. This shall be used in
order to deduce properties of `L` from properties of `W.Q`. -/
def equivalenceFromModel : W.Localization ≌ D :=
  (Localization.Construction.lift L (inverts L W)).asEquivalence


/-- Via the equivalence of categories `equivalence_from_model L W : W.localization ≌ D`,
one may identify the functors `W.Q` and `L`. -/
def qCompEquivalenceFromModelFunctorIso : W.Q ⋙ (equivalenceFromModel L W).functor ≅ L :=
  eqToIso (Construction.fac _ _)


/-- Via the equivalence of categories `equivalence_from_model L W : W.localization ≌ D`,
one may identify the functors `L` and `W.Q`. -/
def compEquivalenceFromModelInverseIso : L ⋙ (equivalenceFromModel L W).inverse ≅ W.Q :=
  calc
    L ⋙ (equivalenceFromModel L W).inverse ≅ _ :=
      isoWhiskerRight (qCompEquivalenceFromModelFunctorIso L W).symm _
    _ ≅ W.Q ⋙ (equivalenceFromModel L W).functor ⋙ (equivalenceFromModel L W).inverse :=
      (Functor.associator _ _ _)
    _ ≅ W.Q ⋙ 𝟭 _ := isoWhiskerLeft _ (equivalenceFromModel L W).unitIso.symm
    _ ≅ W.Q := Functor.rightUnitor _


theorem essSurj (W) [L.IsLocalization W] : L.EssSurj :=
  ⟨fun X =>
    ⟨(Construction.objEquiv W).invFun ((equivalenceFromModel L W).inverse.obj X),
      Nonempty.intro
        ((qCompEquivalenceFromModelFunctorIso L W).symm.app _ ≪≫
          (equivalenceFromModel L W).counitIso.app X)⟩⟩


/-- The functor `(D ⥤ E) ⥤ W.functors_inverting E` induced by the composition
with a localization functor `L : C ⥤ D` with respect to `W : morphism_property C`. -/
def whiskeringLeftFunctor : (D ⥤ E) ⥤ W.FunctorsInverting E :=
  FullSubcategory.lift _ ((whiskeringLeft _ _ E).obj L)
    (MorphismProperty.IsInvertedBy.of_comp W L (inverts L W))


instance : (whiskeringLeftFunctor L W E).IsEquivalence := by
  let iso : (whiskeringLeft (MorphismProperty.Localization W) D E).obj
    (equivalenceFromModel L W).functor ⋙
      (Construction.whiskeringLeftEquivalence W E).functor ≅ whiskeringLeftFunctor L W E :=
    NatIso.ofComponents (fun F => eqToIso (by
      ext
      change (W.Q ⋙ Localization.Construction.lift L (inverts L W)) ⋙ F = L ⋙ F
      rw [Construction.fac])) (fun τ => by
        ext
        dsimp [Construction.whiskeringLeftEquivalence, equivalenceFromModel, whiskerLeft]
        erw [NatTrans.comp_app, NatTrans.comp_app, eqToHom_app, eqToHom_app, eqToHom_refl,
          eqToHom_refl, comp_id, id_comp]
        · rfl
        all_goals
          change (W.Q ⋙ Localization.Construction.lift L (inverts L W)) ⋙ _ = L ⋙ _
          rw [Construction.fac])
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    iso : CategoryTheory.Iso (((CategoryTheory.whiskeringLeft W.Localization D E). …
    ⊢ (CategoryTheory.Localization.whiskeringLeftFunctor L W E).IsEquivalence
  -/
  exact Functor.isEquivalence_of_iso iso
  /-
    🎉 no goals
  -/


/-- The equivalence of categories `(D ⥤ E) ≌ (W.FunctorsInverting E)` induced by
the composition with a localization functor `L : C ⥤ D` with respect to
`W : MorphismProperty C`. -/
def functorEquivalence : D ⥤ E ≌ W.FunctorsInverting E :=
  (whiskeringLeftFunctor L W E).asEquivalence


/-- The functor `(D ⥤ E) ⥤ (C ⥤ E)` given by the composition with a localization
functor `L : C ⥤ D` with respect to `W : MorphismProperty C`. -/
@[nolint unusedArguments]
def whiskeringLeftFunctor' [L.IsLocalization W] (E : Type*) [Category E] :
    (D ⥤ E) ⥤ C ⥤ E :=
  (whiskeringLeft C D E).obj L


theorem whiskeringLeftFunctor'_eq :
    whiskeringLeftFunctor' L W E = Localization.whiskeringLeftFunctor L W E ⋙ inducedFunctor _ :=
  rfl


variable {E} in
@[simp]
theorem whiskeringLeftFunctor'_obj (F : D ⥤ E) : (whiskeringLeftFunctor' L W E).obj F = L ⋙ F :=
  rfl


instance : (whiskeringLeftFunctor' L W E).Full := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    ⊢ (CategoryTheory.Localization.whiskeringLeftFunctor' L W E).Full
  -/
  rw [whiskeringLeftFunctor'_eq]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    ⊢ ((CategoryTheory.Localization.whiskeringLeftFunctor L W E).comp (CategoryThe …
  -/
  apply @Functor.Full.comp _ _ _ _ _ _ _ _ ?_ ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_6, u_1} C
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
      inst✝ : L.IsLocalization W
      ⊢ (CategoryTheory.Localization.whiskeringLeftFunctor L W E).Full
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    ⊢ (CategoryTheory.inducedFunctor CategoryTheory.FullSubcategory.obj).Full
  -/
  apply InducedCategory.full -- why is it not found automatically ???
  /-
    🎉 no goals
  -/


instance : (whiskeringLeftFunctor' L W E).Faithful := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    ⊢ (CategoryTheory.Localization.whiskeringLeftFunctor' L W E).Faithful
  -/
  rw [whiskeringLeftFunctor'_eq]
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    ⊢ ((CategoryTheory.Localization.whiskeringLeftFunctor L W E).comp (CategoryThe …
  -/
  apply @Functor.Faithful.comp _ _ _ _ _ _ _ _ ?_ ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_6, u_1} C
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
      inst✝ : L.IsLocalization W
      ⊢ (CategoryTheory.Localization.whiskeringLeftFunctor L W E).Faithful
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} E
    inst✝ : L.IsLocalization W
    ⊢ (CategoryTheory.inducedFunctor CategoryTheory.FullSubcategory.obj).Faithful
  -/
  apply InducedCategory.faithful -- why is it not found automatically ???
  /-
    🎉 no goals
  -/


lemma full_whiskeringLeft (L : C ⥤ D) (W) [L.IsLocalization W] (E : Type*) [Category E] :
    ((whiskeringLeft C D E).obj L).Full :=
  inferInstanceAs (whiskeringLeftFunctor' L W E).Full


lemma faithful_whiskeringLeft (L : C ⥤ D) (W) [L.IsLocalization W] (E : Type*) [Category E] :
    ((whiskeringLeft C D E).obj L).Faithful :=
  inferInstanceAs (whiskeringLeftFunctor' L W E).Faithful


theorem natTrans_ext (L : C ⥤ D) (W) [L.IsLocalization W] {F₁ F₂ : D ⥤ E} {τ τ' : F₁ ⟶ F₂}
    (h : ∀ X : C, τ.app (L.obj X) = τ'.app (L.obj X)) : τ = τ' := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    F₁ F₂ : CategoryTheory.Functor D E
    τ τ' : Quiver.Hom F₁ F₂
    h : ∀ (X : C), Eq (τ.app (L.obj X)) (τ'.app (L.obj X))
    ⊢ Eq τ τ'
  -/
  haveI := essSurj L W
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    F₁ F₂ : CategoryTheory.Functor D E
    τ τ' : Quiver.Hom F₁ F₂
    h : ∀ (X : C), Eq (τ.app (L.obj X)) (τ'.app (L.obj X))
    this : L.EssSurj
    ⊢ Eq τ τ'
  -/
  ext Y
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    E : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} E
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    F₁ F₂ : CategoryTheory.Functor D E
    τ τ' : Quiver.Hom F₁ F₂
    h : ∀ (X : C), Eq (τ.app (L.obj X)) (τ'.app (L.obj X))
    this : L.EssSurj
    Y : D
    ⊢ Eq (τ.app Y) (τ'.app Y)
  -/
  rw [← cancel_epi (F₁.map (L.objObjPreimageIso Y).hom), τ.naturality, τ'.naturality, h]
  /-
    🎉 no goals
  -/

-- Porting note: the field `iso` was renamed `Lifting.iso'` and it was redefined as
-- `Lifting.iso` with explicit parameters

/-- When `L : C ⥤ D` is a localization functor for `W : MorphismProperty C` and
`F : C ⥤ E` is a functor, we shall say that `F' : D ⥤ E` lifts `F` if the obvious diagram
is commutative up to an isomorphism. -/
class Lifting (W : MorphismProperty C) (F : C ⥤ E) (F' : D ⥤ E) where
  /-- the isomorphism relating the localization functor and the two other given functors -/
  iso' : L ⋙ F' ≅ F


/-- The distinguished isomorphism `L ⋙ F' ≅ F` given by `[Lifting L W F F']`. -/
def Lifting.iso (F : C ⥤ E) (F' : D ⥤ E) [Lifting L W F F'] :
    L ⋙ F' ≅ F :=
  Lifting.iso' W


/-- Given a localization functor `L : C ⥤ D` for `W : MorphismProperty C` and
a functor `F : C ⥤ E` which inverts `W`, this is a choice of functor
`D ⥤ E` which lifts `F`. -/
def lift (F : C ⥤ E) (hF : W.IsInvertedBy F) (L : C ⥤ D) [L.IsLocalization W] : D ⥤ E :=
  (functorEquivalence L W E).inverse.obj ⟨F, hF⟩


instance liftingLift (F : C ⥤ E) (hF : W.IsInvertedBy F) (L : C ⥤ D) [L.IsLocalization W] :
    Lifting L W F (lift F hF L) :=
  ⟨(inducedFunctor _).mapIso ((functorEquivalence L W E).counitIso.app ⟨F, hF⟩)⟩

-- Porting note: removed the unnecessary @[simps] attribute

/-- The canonical isomorphism `L ⋙ lift F hF L ≅ F` for any functor `F : C ⥤ E`
which inverts `W`, when `L : C ⥤ D` is a localization functor for `W`. -/
def fac (F : C ⥤ E) (hF : W.IsInvertedBy F) (L : C ⥤ D) [L.IsLocalization W] :
    L ⋙ lift F hF L ≅ F :=
  Lifting.iso L W F _


instance liftingConstructionLift (F : C ⥤ D) (hF : W.IsInvertedBy F) :
    Lifting W.Q W F (Construction.lift F hF) :=
  ⟨eqToIso (Construction.fac F hF)⟩


/-- Given a localization functor `L : C ⥤ D` for `W : MorphismProperty C`,
if `(F₁' F₂' : D ⥤ E)` are functors which lifts functors `(F₁ F₂ : C ⥤ E)`,
a natural transformation `τ : F₁ ⟶ F₂` uniquely lifts to a natural transformation `F₁' ⟶ F₂'`. -/
def liftNatTrans (F₁ F₂ : C ⥤ E) (F₁' F₂' : D ⥤ E) [Lifting L W F₁ F₁'] [Lifting L W F₂ F₂']
    (τ : F₁ ⟶ F₂) : F₁' ⟶ F₂' :=
  (whiskeringLeftFunctor' L W E).preimage
    ((Lifting.iso L W F₁ F₁').hom ≫ τ ≫ (Lifting.iso L W F₂ F₂').inv)


@[simp]
theorem liftNatTrans_app (F₁ F₂ : C ⥤ E) (F₁' F₂' : D ⥤ E) [Lifting L W F₁ F₁'] [Lifting L W F₂ F₂']
    (τ : F₁ ⟶ F₂) (X : C) :
    (liftNatTrans L W F₁ F₂ F₁' F₂' τ).app (L.obj X) =
      (Lifting.iso L W F₁ F₁').hom.app X ≫ τ.app X ≫ (Lifting.iso L W F₂ F₂').inv.app X :=
  congr_app (Functor.map_preimage (whiskeringLeftFunctor' L W E) _) X


@[reassoc (attr := simp)]
theorem comp_liftNatTrans (F₁ F₂ F₃ : C ⥤ E) (F₁' F₂' F₃' : D ⥤ E) [h₁ : Lifting L W F₁ F₁']
    [h₂ : Lifting L W F₂ F₂'] [h₃ : Lifting L W F₃ F₃'] (τ : F₁ ⟶ F₂) (τ' : F₂ ⟶ F₃) :
    liftNatTrans L W F₁ F₂ F₁' F₂' τ ≫ liftNatTrans L W F₂ F₃ F₂' F₃' τ' =
      liftNatTrans L W F₁ F₃ F₁' F₃' (τ ≫ τ') :=
  natTrans_ext L W fun X => by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_6, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_5, u_3} E
      inst✝ : L.IsLocalization W
      F₁ F₂ F₃ : CategoryTheory.Functor C E
      F₁' F₂' F₃' : CategoryTheory.Functor D E
      h₁ : CategoryTheory.Localization.Lifting L W F₁ F₁'
      h₂ : CategoryTheory.Localization.Lifting L W F₂ F₂'
      h₃ : CategoryTheory.Localization.Lifting L W F₃ F₃'
      τ : Quiver.Hom F₁ F₂
      τ' : Quiver.Hom F₂ F₃
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Localization.liftNat …
    -/
    simp only [NatTrans.comp_app, liftNatTrans_app, assoc, Iso.inv_hom_id_app_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem liftNatTrans_id (F : C ⥤ E) (F' : D ⥤ E) [h : Lifting L W F F'] :
    liftNatTrans L W F F F' F' (𝟙 F) = 𝟙 F' :=
  natTrans_ext L W fun X => by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_6, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_5, u_3} E
      inst✝ : L.IsLocalization W
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      h : CategoryTheory.Localization.Lifting L W F F'
      X : C
      ⊢ Eq ((CategoryTheory.Localization.liftNatTrans L W F F F' F' (CategoryTheory. …
    -/
    simp only [liftNatTrans_app, NatTrans.id_app, id_comp, Iso.hom_inv_id_app]
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      inst✝² : CategoryTheory.Category.{u_6, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_5, u_3} E
      inst✝ : L.IsLocalization W
      F : CategoryTheory.Functor C E
      F' : CategoryTheory.Functor D E
      h : CategoryTheory.Localization.Lifting L W F F'
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.id ((L.comp F').obj X)) (CategoryTheory.Ca …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Given a localization functor `L : C ⥤ D` for `W : MorphismProperty C`,
if `(F₁' F₂' : D ⥤ E)` are functors which lifts functors `(F₁ F₂ : C ⥤ E)`,
a natural isomorphism `τ : F₁ ⟶ F₂` lifts to a natural isomorphism `F₁' ⟶ F₂'`. -/
@[simps]
def liftNatIso (F₁ F₂ : C ⥤ E) (F₁' F₂' : D ⥤ E) [h₁ : Lifting L W F₁ F₁'] [h₂ : Lifting L W F₂ F₂']
    (e : F₁ ≅ F₂) : F₁' ≅ F₂' where
  hom := liftNatTrans L W F₁ F₂ F₁' F₂' e.hom
  inv := liftNatTrans L W F₂ F₁ F₂' F₁' e.inv


@[simps]
instance compRight {E' : Type*} [Category E'] (F : C ⥤ E) (F' : D ⥤ E) [Lifting L W F F']
    (G : E ⥤ E') : Lifting L W (F ⋙ G) (F' ⋙ G) :=
  ⟨isoWhiskerRight (iso L W F F') G⟩


@[simps]
instance id : Lifting L W L (𝟭 D) :=
  ⟨Functor.rightUnitor L⟩


@[simps]
instance compLeft (F : D ⥤ E) : Localization.Lifting L W (L ⋙ F) F := ⟨Iso.refl _⟩


@[simp]
lemma compLeft_iso (W) (F : D ⥤ E) : Localization.Lifting.iso L W (L ⋙ F) F = Iso.refl _ := rfl


/-- Given a localization functor `L : C ⥤ D` for `W : MorphismProperty C`,
if `F₁' : D ⥤ E` lifts a functor `F₁ : C ⥤ D`, then a functor `F₂'` which
is isomorphic to `F₁'` also lifts a functor `F₂` that is isomorphic to `F₁`. -/
@[simps]
def ofIsos {F₁ F₂ : C ⥤ E} {F₁' F₂' : D ⥤ E} (e : F₁ ≅ F₂) (e' : F₁' ≅ F₂') [Lifting L W F₁ F₁'] :
    Lifting L W F₂ F₂' :=
  ⟨isoWhiskerLeft L e'.symm ≪≫ iso L W F₁ F₁' ≪≫ e⟩


theorem of_iso {L₁ L₂ : C ⥤ D} (e : L₁ ≅ L₂) [L₁.IsLocalization W] : L₂.IsLocalization W := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    W : CategoryTheory.MorphismProperty C
    L₁ L₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso L₁ L₂
    inst✝ : L₁.IsLocalization W
    ⊢ L₂.IsLocalization W
  -/
  have h := Localization.inverts L₁ W
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    W : CategoryTheory.MorphismProperty C
    L₁ L₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso L₁ L₂
    inst✝ : L₁.IsLocalization W
    h : W.IsInvertedBy L₁
    ⊢ L₂.IsLocalization W
  -/
  rw [MorphismProperty.IsInvertedBy.iff_of_iso W e] at h
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    W : CategoryTheory.MorphismProperty C
    L₁ L₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso L₁ L₂
    inst✝ : L₁.IsLocalization W
    h : W.IsInvertedBy L₂
    ⊢ L₂.IsLocalization W
  -/
  let F₁ := Localization.Construction.lift L₁ (Localization.inverts L₁ W)
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    W : CategoryTheory.MorphismProperty C
    L₁ L₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso L₁ L₂
    inst✝ : L₁.IsLocalization W
    h : W.IsInvertedBy L₂
    F₁ : CategoryTheory.Functor W.Localization D := CategoryTheory.Localization.Co …
    ⊢ L₂.IsLocalization W
  -/
  let F₂ := Localization.Construction.lift L₂ h
  exact
    { inverts := h
      isEquivalence := Functor.isEquivalence_of_iso (liftNatIso W.Q W L₁ L₂ F₁ F₂ e) }


/-- If `L : C ⥤ D` is a localization for `W : MorphismProperty C`, then it is also
the case of a functor obtained by post-composing `L` with an equivalence of categories. -/
theorem of_equivalence_target {E : Type*} [Category E] (L' : C ⥤ E) (eq : D ≌ E)
    [L.IsLocalization W] (e : L ⋙ eq.functor ≅ L') : L'.IsLocalization W := by
  have h : W.IsInvertedBy L' := by
    rw [← MorphismProperty.IsInvertedBy.iff_of_iso W e]
    exact MorphismProperty.IsInvertedBy.of_comp W L (Localization.inverts L W) eq.functor
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_7, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} E
    L' : CategoryTheory.Functor C E
    eq : CategoryTheory.Equivalence D E
    inst✝ : L.IsLocalization W
    e : CategoryTheory.Iso (L.comp eq.functor) L'
    h : W.IsInvertedBy L'
    ⊢ L'.IsLocalization W
  -/
  let F₁ := Localization.Construction.lift L (Localization.inverts L W)
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_7, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} E
    L' : CategoryTheory.Functor C E
    eq : CategoryTheory.Equivalence D E
    inst✝ : L.IsLocalization W
    e : CategoryTheory.Iso (L.comp eq.functor) L'
    h : W.IsInvertedBy L'
    F₁ : CategoryTheory.Functor W.Localization D := CategoryTheory.Localization.Co …
    ⊢ L'.IsLocalization W
  -/
  let F₂ := Localization.Construction.lift L' h
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_7, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    E : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} E
    L' : CategoryTheory.Functor C E
    eq : CategoryTheory.Equivalence D E
    inst✝ : L.IsLocalization W
    e : CategoryTheory.Iso (L.comp eq.functor) L'
    h : W.IsInvertedBy L'
    F₁ : CategoryTheory.Functor W.Localization D := CategoryTheory.Localization.Co …
    F₂ : CategoryTheory.Functor W.Localization E := CategoryTheory.Localization.Co …
    ⊢ L'.IsLocalization W
  -/
  let e' : F₁ ⋙ eq.functor ≅ F₂ := liftNatIso W.Q W (L ⋙ eq.functor) L' _ _ e
  exact
    { inverts := h
      isEquivalence := Functor.isEquivalence_of_iso e' }


instance (F : D ⥤ E) [F.IsEquivalence] [L.IsLocalization W] :
    (L ⋙ F).IsLocalization W :=
  of_equivalence_target L W _ F.asEquivalence (Iso.refl _)


lemma of_isEquivalence (L : C ⥤ D) (W : MorphismProperty C)
    (hW : W ≤ MorphismProperty.isomorphisms C) [IsEquivalence L] :
    L.IsLocalization W := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
    inst✝ : L.IsEquivalence
    ⊢ L.IsLocalization W
  -/
  haveI : (𝟭 C).IsLocalization W := for_id W hW
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    hW : LE.le W (CategoryTheory.MorphismProperty.isomorphisms C)
    inst✝ : L.IsEquivalence
    this : (CategoryTheory.Functor.id C).IsLocalization W
    ⊢ L.IsLocalization W
  -/
  exact of_equivalence_target (𝟭 C) W L L.asEquivalence L.leftUnitor
  /-
    🎉 no goals
  -/


/-- If `L₁ : C ⥤ D₁` and `L₂ : C ⥤ D₂` are two localization functors for the
same `MorphismProperty C`, this is an equivalence of categories `D₁ ≌ D₂`. -/
def uniq : D₁ ≌ D₂ :=
  (equivalenceFromModel L₁ W').symm.trans (equivalenceFromModel L₂ W')


lemma uniq_symm : (uniq L₁ L₂ W').symm = uniq L₂ L₁ W' := rfl


/-- The functor of equivalence of localized categories given by `Localization.uniq` is
compatible with the localization functors. -/
def compUniqFunctor : L₁ ⋙ (uniq L₁ L₂ W').functor ≅ L₂ :=
  calc
    L₁ ⋙ (uniq L₁ L₂ W').functor ≅ (L₁ ⋙ (equivalenceFromModel L₁ W').inverse) ⋙
      (equivalenceFromModel L₂ W').functor := (Functor.associator _ _ _).symm
    _ ≅ W'.Q ⋙ (equivalenceFromModel L₂ W').functor :=
      isoWhiskerRight (compEquivalenceFromModelInverseIso L₁ W') _
    _ ≅ L₂ := qCompEquivalenceFromModelFunctorIso L₂ W'


/-- The inverse functor of equivalence of localized categories given by `Localization.uniq` is
compatible with the localization functors. -/
def compUniqInverse : L₂ ⋙ (uniq L₁ L₂ W').inverse ≅ L₁ := compUniqFunctor L₂ L₁ W'


instance : Lifting L₁ W' L₂ (uniq L₁ L₂ W').functor := ⟨compUniqFunctor L₁ L₂ W'⟩

instance : Lifting L₂ W' L₁ (uniq L₁ L₂ W').inverse := ⟨compUniqInverse L₁ L₂ W'⟩


/-- If `L₁ : C ⥤ D₁` and `L₂ : C ⥤ D₂` are two localization functors for the
same `MorphismProperty C`, any functor `F : D₁ ⥤ D₂` equipped with an isomorphism
`L₁ ⋙ F ≅ L₂` is isomorphic to the functor of the equivalence given by `uniq`. -/
def isoUniqFunctor (F : D₁ ⥤ D₂) (e : L₁ ⋙ F ≅ L₂) :
    F ≅ (uniq L₁ L₂ W').functor :=
  letI : Lifting L₁ W' L₂ F := ⟨e⟩
  liftNatIso L₁ W' L₂ L₂ F (uniq L₁ L₂ W').functor (Iso.refl L₂)


/-- The property that two morphisms become equal in the localized category. -/
def AreEqualizedByLocalization : Prop := W.Q.map f = W.Q.map g


lemma areEqualizedByLocalization_iff [L.IsLocalization W] :
    AreEqualizedByLocalization W f g ↔ L.map f = L.map g := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝ : L.IsLocalization W
    ⊢ Iff (CategoryTheory.AreEqualizedByLocalization W f g) (Eq (L.map f) (L.map g))
  -/
  dsimp [AreEqualizedByLocalization]
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝ : L.IsLocalization W
    ⊢ Iff (Eq (W.Q.map f) (W.Q.map g)) (Eq (L.map f) (L.map g))
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      ⊢ Eq (W.Q.map f) (W.Q.map g) → Eq (L.map f) (L.map g)
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (W.Q.map f) (W.Q.map g)
      ⊢ Eq (L.map f) (L.map g)
    -/
    let e := Localization.compUniqFunctor W.Q L W
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (W.Q.map f) (W.Q.map g)
      e : CategoryTheory.Iso (W.Q.comp (CategoryTheory.Localization.uniq W.Q L W).fu …
      ⊢ Eq (L.map f) (L.map g)
    -/
    rw [← NatIso.naturality_1 e f, ← NatIso.naturality_1 e g]
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (W.Q.map f) (W.Q.map g)
      e : CategoryTheory.Iso (W.Q.comp (CategoryTheory.Localization.uniq W.Q L W).fu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (CategoryTheory.Categor …
    -/
    dsimp
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (W.Q.map f) (W.Q.map g)
      e : CategoryTheory.Iso (W.Q.comp (CategoryTheory.Localization.uniq W.Q L W).fu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (CategoryTheory.Categor …
    -/
    rw [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      ⊢ Eq (L.map f) (L.map g) → Eq (W.Q.map f) (W.Q.map g)
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (L.map f) (L.map g)
      ⊢ Eq (W.Q.map f) (W.Q.map g)
    -/
    let e := Localization.compUniqFunctor L W.Q W
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (L.map f) (L.map g)
      e : CategoryTheory.Iso (L.comp (CategoryTheory.Localization.uniq L W.Q W).func …
      ⊢ Eq (W.Q.map f) (W.Q.map g)
    -/
    rw [← NatIso.naturality_1 e f, ← NatIso.naturality_1 e g]
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (L.map f) (L.map g)
      e : CategoryTheory.Iso (L.comp (CategoryTheory.Localization.uniq L W.Q W).func …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (CategoryTheory.Categor …
    -/
    dsimp
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      X Y : C
      f g : Quiver.Hom X Y
      inst✝ : L.IsLocalization W
      h : Eq (L.map f) (L.map g)
      e : CategoryTheory.Iso (L.comp (CategoryTheory.Localization.uniq L W.Q W).func …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.inv.app X) (CategoryTheory.Categor …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


lemma mk (L : C ⥤ D) [L.IsLocalization W] (h : L.map f = L.map g) :
    AreEqualizedByLocalization W f g :=
  (areEqualizedByLocalization_iff L W f g).2 h


lemma map_eq (h : AreEqualizedByLocalization W f g) (L : C ⥤ D) [L.IsLocalization W] :
    L.map f = L.map g :=
  (areEqualizedByLocalization_iff L W f g).1 h


