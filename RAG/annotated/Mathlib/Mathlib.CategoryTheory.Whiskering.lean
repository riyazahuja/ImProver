/-- If `α : G ⟶ H` then
`whiskerLeft F α : (F ⋙ G) ⟶ (F ⋙ H)` has components `α.app (F.obj X)`.
-/
@[simps]
def whiskerLeft (F : C ⥤ D) {G H : D ⥤ E} (α : G ⟶ H) :
    F ⋙ G ⟶ F ⋙ H where
  app X := α.app (F.obj X)
                         /-
                           C : Type u₁
                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                           D : Type u₂
                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                           E : Type u₃
                           inst✝ : CategoryTheory.Category.{v₃, u₃} E
                           F : CategoryTheory.Functor C D
                           G H : CategoryTheory.Functor D E
                           α : Quiver.Hom G H
                           X Y : C
                           f : Quiver.Hom X Y
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).map f) ((fun X => α.app ( …
                         -/
  naturality X Y f := by rw [Functor.comp_map, Functor.comp_map, α.naturality]
                         /-
                           🎉 no goals
                         -/


/-- If `α : G ⟶ H` then
`whisker_right α F : (G ⋙ F) ⟶ (G ⋙ F)` has components `F.map (α.app X)`.
-/
@[simps]
def whiskerRight {G H : C ⥤ D} (α : G ⟶ H) (F : D ⥤ E) :
    G ⋙ F ⟶ H ⋙ F where
  app X := F.map (α.app X)
  naturality X Y f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      G H : CategoryTheory.Functor C D
      α : Quiver.Hom G H
      F : CategoryTheory.Functor D E
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.comp F).map f) ((fun X => F.map ( …
    -/
    rw [Functor.comp_map, Functor.comp_map, ← F.map_comp, ← F.map_comp, α.naturality]
    /-
      🎉 no goals
    -/


/-- Left-composition gives a functor `(C ⥤ D) ⥤ ((D ⥤ E) ⥤ (C ⥤ E))`.

`(whiskeringLeft.obj F).obj G` is `F ⋙ G`, and
`(whiskeringLeft.obj F).map α` is `whiskerLeft F α`.
-/
@[simps]
def whiskeringLeft : (C ⥤ D) ⥤ (D ⥤ E) ⥤ C ⥤ E where
  obj F :=
    { obj := fun G => F ⋙ G
      map := fun α => whiskerLeft F α }
  map τ :=
    { app := fun H =>
        { app := fun c => H.map (τ.app c)
                                        /-
                                          C : Type u₁
                                          inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                          D : Type u₂
                                          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                          E : Type u₃
                                          inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                          X✝ Y✝ : CategoryTheory.Functor C D
                                          τ : Quiver.Hom X✝ Y✝
                                          H : CategoryTheory.Functor D E
                                          X Y : C
                                          f : Quiver.Hom X Y
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun F => { obj := fun G => F.comp …
                                        -/
          naturality := fun X Y f => by dsimp; rw [← H.map_comp, ← H.map_comp, ← τ.naturality] }
                                               /-
                                                 🎉 no goals
                                               -/
                                    /-
                                      C : Type u₁
                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                      E : Type u₃
                                      inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                      X✝ Y✝ : CategoryTheory.Functor C D
                                      τ : Quiver.Hom X✝ Y✝
                                      X Y : CategoryTheory.Functor D E
                                      f : Quiver.Hom X Y
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun F => { obj := fun G => F.comp  …
                                    -/
      naturality := fun X Y f => by ext; dsimp; rw [f.naturality] }
                                                /-
                                                  🎉 no goals
                                                -/


/-- Right-composition gives a functor `(D ⥤ E) ⥤ ((C ⥤ D) ⥤ (C ⥤ E))`.

`(whiskeringRight.obj H).obj F` is `F ⋙ H`, and
`(whiskeringRight.obj H).map α` is `whiskerRight α H`.
-/
@[simps]
def whiskeringRight : (D ⥤ E) ⥤ (C ⥤ D) ⥤ C ⥤ E where
  obj H :=
    { obj := fun F => F ⋙ H
      map := fun α => whiskerRight α H }
  map τ :=
    { app := fun F =>
        { app := fun c => τ.app (F.obj c)
                                        /-
                                          C : Type u₁
                                          inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                          D : Type u₂
                                          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                          E : Type u₃
                                          inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                          X✝ Y✝ : CategoryTheory.Functor D E
                                          τ : Quiver.Hom X✝ Y✝
                                          F : CategoryTheory.Functor C D
                                          X Y : C
                                          f : Quiver.Hom X Y
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun H => { obj := fun F => F.comp …
                                        -/
          naturality := fun X Y f => by dsimp; rw [τ.naturality] }
                                               /-
                                                 🎉 no goals
                                               -/
                                    /-
                                      C : Type u₁
                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                      E : Type u₃
                                      inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                      X✝ Y✝ : CategoryTheory.Functor D E
                                      τ : Quiver.Hom X✝ Y✝
                                      X Y : CategoryTheory.Functor C D
                                      f : Quiver.Hom X Y
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun H => { obj := fun F => F.comp  …
                                    -/
      naturality := fun X Y f => by ext; dsimp; rw [← NatTrans.naturality] }
                                                /-
                                                  🎉 no goals
                                                -/


instance faithful_whiskeringRight_obj {F : D ⥤ E} [F.Faithful] :
    ((whiskeringRight C D E).obj F).Faithful where
  map_injective hαβ := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.Faithful
      X✝ Y✝ : CategoryTheory.Functor C D
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      hαβ : Eq (((CategoryTheory.whiskeringRight C D E).obj F).map a₁✝) (((CategoryT …
      ⊢ Eq a₁✝ a₂✝
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.Faithful
      X✝ Y✝ : CategoryTheory.Functor C D
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      hαβ : Eq (((CategoryTheory.whiskeringRight C D E).obj F).map a₁✝) (((CategoryT …
      X : C
      ⊢ Eq (a₁✝.app X) (a₂✝.app X)
    -/
    exact F.map_injective <| congr_fun (congr_arg NatTrans.app hαβ) X
    /-
      🎉 no goals
    -/


/-- If `F : D ⥤ E` is fully faithful, then so is
`(whiskeringRight C D E).obj F : (C ⥤ D) ⥤ C ⥤ E`. -/
@[simps]
def Functor.FullyFaithful.whiskeringRight {F : D ⥤ E} (hF : F.FullyFaithful)
    (C : Type*) [Category C] :
    ((whiskeringRight C D E).obj F).FullyFaithful where
  preimage f :=
    { app := fun X => hF.preimage (f.app X)
      naturality := fun _ _ g => by
        /-
          C✝ : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          F : CategoryTheory.Functor D E
          hF : F.FullyFaithful
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.30057, u_1} C
          X✝ Y✝ : CategoryTheory.Functor C D
          f : Quiver.Hom (((CategoryTheory.whiskeringRight C D E).obj F).obj X✝) (((Cate …
          x✝¹ x✝ : C
          g : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.map g) ((fun X => hF.preimage (f. …
        -/
        apply hF.map_injective
        /-
          C✝ : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          F : CategoryTheory.Functor D E
          hF : F.FullyFaithful
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.30057, u_1} C
          X✝ Y✝ : CategoryTheory.Functor C D
          f : Quiver.Hom (((CategoryTheory.whiskeringRight C D E).obj F).obj X✝) (((Cate …
          x✝¹ x✝ : C
          g : Quiver.Hom x✝¹ x✝
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (X✝.map g) ((fun X => hF.preim …
        -/
        dsimp
        /-
          C✝ : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          F : CategoryTheory.Functor D E
          hF : F.FullyFaithful
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.30057, u_1} C
          X✝ Y✝ : CategoryTheory.Functor C D
          f : Quiver.Hom (((CategoryTheory.whiskeringRight C D E).obj F).obj X✝) (((Cate …
          x✝¹ x✝ : C
          g : Quiver.Hom x✝¹ x✝
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (X✝.map g) (hF.preimage (f.app …
        -/
        simp only [map_comp, map_preimage]
        /-
          C✝ : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          F : CategoryTheory.Functor D E
          hF : F.FullyFaithful
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.30057, u_1} C
          X✝ Y✝ : CategoryTheory.Functor C D
          f : Quiver.Hom (((CategoryTheory.whiskeringRight C D E).obj F).obj X✝) (((Cate …
          x✝¹ x✝ : C
          g : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (X✝.map g)) (f.app x✝)) (Categ …
        -/
        apply f.naturality }
        /-
          🎉 no goals
        -/


theorem whiskeringLeft_obj_id : (whiskeringLeft C C E).obj (𝟭 _) = 𝟭 _ :=
  rfl


/-- The isomorphism between left-whiskering on the identity functor and the identity of the functor
between the resulting functor categories. -/
def whiskeringLeftObjIdIso : (whiskeringLeft C C E).obj (𝟭 _) ≅ 𝟭 _ :=
  Iso.refl _


theorem whiskeringLeft_obj_comp {D' : Type u₄} [Category.{v₄} D'] (F : C ⥤ D) (G : D ⥤ D') :
    (whiskeringLeft C D' E).obj (F ⋙ G) =
    (whiskeringLeft D D' E).obj G ⋙ (whiskeringLeft C D E).obj F :=
  rfl


/-- The isomorphism between left-whiskering on the composition of functors and the composition
of two left-whiskering applications. -/
def whiskeringLeftObjCompIso {D' : Type u₄} [Category.{v₄} D'] (F : C ⥤ D) (G : D ⥤ D') :
    (whiskeringLeft C D' E).obj (F ⋙ G) ≅
    (whiskeringLeft D D' E).obj G ⋙ (whiskeringLeft C D E).obj F :=
  Iso.refl _


theorem whiskeringRight_obj_id : (whiskeringRight E C C).obj (𝟭 _) = 𝟭 _ :=
  rfl


/-- The isomorphism between right-whiskering on the identity functor and the identity of the functor
between the resulting functor categories. -/
def wiskeringRightObjIdIso : (whiskeringRight E C C).obj (𝟭 _) ≅ 𝟭 _ :=
  Iso.refl _


theorem whiskeringRight_obj_comp {D' : Type u₄} [Category.{v₄} D'] (F : C ⥤ D) (G : D ⥤ D') :
    (whiskeringRight E C D).obj F ⋙ (whiskeringRight E D D').obj G =
    (whiskeringRight E C D').obj (F ⋙ G) :=
  rfl


/-- The isomorphism between right-whiskering on the composition of functors and the composition
of two right-whiskering applications. -/
def whiskeringRightObjCompIso {D' : Type u₄} [Category.{v₄} D'] (F : C ⥤ D) (G : D ⥤ D') :
    (whiskeringRight E C D).obj F ⋙ (whiskeringRight E D D').obj G ≅
    (whiskeringRight E C D').obj (F ⋙ G) :=
  Iso.refl _


instance full_whiskeringRight_obj {F : D ⥤ E} [F.Faithful] [F.Full] :
    ((whiskeringRight C D E).obj F).Full :=
  ((Functor.FullyFaithful.ofFullyFaithful F).whiskeringRight C).full


@[simp]
theorem whiskerLeft_id (F : C ⥤ D) {G : D ⥤ E} :
    whiskerLeft F (NatTrans.id G) = NatTrans.id (F.comp G) :=
  rfl


@[simp]
theorem whiskerLeft_id' (F : C ⥤ D) {G : D ⥤ E} : whiskerLeft F (𝟙 G) = 𝟙 (F.comp G) :=
  rfl


@[simp]
theorem whiskerRight_id {G : C ⥤ D} (F : D ⥤ E) :
    whiskerRight (NatTrans.id G) F = NatTrans.id (G.comp F) :=
  ((whiskeringRight C D E).obj F).map_id _


@[simp]
theorem whiskerRight_id' {G : C ⥤ D} (F : D ⥤ E) : whiskerRight (𝟙 G) F = 𝟙 (G.comp F) :=
  ((whiskeringRight C D E).obj F).map_id _


@[simp, reassoc]
theorem whiskerLeft_comp (F : C ⥤ D) {G H K : D ⥤ E} (α : G ⟶ H) (β : H ⟶ K) :
    whiskerLeft F (α ≫ β) = whiskerLeft F α ≫ whiskerLeft F β :=
  rfl


@[simp, reassoc]
theorem whiskerRight_comp {G H K : C ⥤ D} (α : G ⟶ H) (β : H ⟶ K) (F : D ⥤ E) :
    whiskerRight (α ≫ β) F = whiskerRight α F ≫ whiskerRight β F :=
  ((whiskeringRight C D E).obj F).map_comp α β


/-- If `α : G ≅ H` is a natural isomorphism then
`iso_whisker_left F α : (F ⋙ G) ≅ (F ⋙ H)` has components `α.app (F.obj X)`.
-/
def isoWhiskerLeft (F : C ⥤ D) {G H : D ⥤ E} (α : G ≅ H) : F ⋙ G ≅ F ⋙ H :=
  ((whiskeringLeft C D E).obj F).mapIso α


@[simp]
theorem isoWhiskerLeft_hom (F : C ⥤ D) {G H : D ⥤ E} (α : G ≅ H) :
    (isoWhiskerLeft F α).hom = whiskerLeft F α.hom :=
  rfl


@[simp]
theorem isoWhiskerLeft_inv (F : C ⥤ D) {G H : D ⥤ E} (α : G ≅ H) :
    (isoWhiskerLeft F α).inv = whiskerLeft F α.inv :=
  rfl


/-- If `α : G ≅ H` then
`iso_whisker_right α F : (G ⋙ F) ≅ (H ⋙ F)` has components `F.map_iso (α.app X)`.
-/
def isoWhiskerRight {G H : C ⥤ D} (α : G ≅ H) (F : D ⥤ E) : G ⋙ F ≅ H ⋙ F :=
  ((whiskeringRight C D E).obj F).mapIso α


@[simp]
theorem isoWhiskerRight_hom {G H : C ⥤ D} (α : G ≅ H) (F : D ⥤ E) :
    (isoWhiskerRight α F).hom = whiskerRight α.hom F :=
  rfl


@[simp]
theorem isoWhiskerRight_inv {G H : C ⥤ D} (α : G ≅ H) (F : D ⥤ E) :
    (isoWhiskerRight α F).inv = whiskerRight α.inv F :=
  rfl


instance isIso_whiskerLeft (F : C ⥤ D) {G H : D ⥤ E} (α : G ⟶ H) [IsIso α] :
    IsIso (whiskerLeft F α) :=
  (isoWhiskerLeft F (asIso α)).isIso_hom


instance isIso_whiskerRight {G H : C ⥤ D} (α : G ⟶ H) (F : D ⥤ E) [IsIso α] :
    IsIso (whiskerRight α F) :=
  (isoWhiskerRight (asIso α) F).isIso_hom


@[simp]
theorem whiskerLeft_twice (F : B ⥤ C) (G : C ⥤ D) {H K : D ⥤ E} (α : H ⟶ K) :
    whiskerLeft F (whiskerLeft G α) = whiskerLeft (F ⋙ G) α :=
  rfl


@[simp]
theorem whiskerRight_twice {H K : B ⥤ C} (F : C ⥤ D) (G : D ⥤ E) (α : H ⟶ K) :
    whiskerRight (whiskerRight α F) G = whiskerRight α (F ⋙ G) :=
  rfl


theorem whiskerRight_left (F : B ⥤ C) {G H : C ⥤ D} (α : G ⟶ H) (K : D ⥤ E) :
    whiskerRight (whiskerLeft F α) K = whiskerLeft F (whiskerRight α K) :=
  rfl


/-- The left unitor, a natural isomorphism `((𝟭 _) ⋙ F) ≅ F`.
-/
@[simps]
def leftUnitor (F : A ⥤ B) :
    𝟭 A ⋙ F ≅ F where
  hom := { app := fun X => 𝟙 (F.obj X) }
  inv := { app := fun X => 𝟙 (F.obj X) }


/-- The right unitor, a natural isomorphism `(F ⋙ (𝟭 B)) ≅ F`.
-/
@[simps]
def rightUnitor (F : A ⥤ B) :
    F ⋙ 𝟭 B ≅ F where
  hom := { app := fun X => 𝟙 (F.obj X) }
  inv := { app := fun X => 𝟙 (F.obj X) }


/-- The associator for functors, a natural isomorphism `((F ⋙ G) ⋙ H) ≅ (F ⋙ (G ⋙ H))`.

(In fact, `iso.refl _` will work here, but it tends to make Lean slow later,
and it's usually best to insert explicit associators.)
-/
@[simps]
def associator (F : A ⥤ B) (G : B ⥤ C) (H : C ⥤ D) :
    (F ⋙ G) ⋙ H ≅ F ⋙ G ⋙ H where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


protected theorem assoc (F : A ⥤ B) (G : B ⥤ C) (H : C ⥤ D) : (F ⋙ G) ⋙ H = F ⋙ G ⋙ H :=
  rfl


theorem triangle (F : A ⥤ B) (G : B ⥤ C) :
    (associator F (𝟭 B) G).hom ≫ whiskerLeft F (leftUnitor G).hom =
                                               /-
                                                 A : Type u₁
                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} A
                                                 B : Type u₂
                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
                                                 C : Type u₃
                                                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                 F : CategoryTheory.Functor A B
                                                 G : CategoryTheory.Functor B C
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.associator (CategoryTheory.Functor …
                                               -/
      whiskerRight (rightUnitor F).hom G := by aesop_cat
                                               /-
                                                 🎉 no goals
                                               -/

-- See note [dsimp, simp].

theorem pentagon :
    whiskerRight (associator F G H).hom K ≫
        (associator F (G ⋙ H) K).hom ≫ whiskerLeft F (associator G H K).hom =
                                                                        /-
                                                                          A : Type u₁
                                                                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
                                                                          B : Type u₂
                                                                          inst✝³ : CategoryTheory.Category.{v₂, u₂} B
                                                                          C : Type u₃
                                                                          inst✝² : CategoryTheory.Category.{v₃, u₃} C
                                                                          D : Type u₄
                                                                          inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                                                                          E : Type u₅
                                                                          inst✝ : CategoryTheory.Category.{v₅, u₅} E
                                                                          F : CategoryTheory.Functor A B
                                                                          G : CategoryTheory.Functor B C
                                                                          H : CategoryTheory.Functor C D
                                                                          K : CategoryTheory.Functor D E
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (F.assoc …
                                                                        -/
      (associator (F ⋙ G) H K).hom ≫ (associator F G (H ⋙ K)).hom := by aesop_cat
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The obvious functor `(C₁ ⥤ D₁) ⥤ (C₂ ⥤ D₂) ⥤ (D₁ ⥤ D₂ ⥤ E) ⥤ (C₁ ⥤ C₂ ⥤ E)`. -/
@[simps!]
def whiskeringLeft₂ :
    (C₁ ⥤ D₁) ⥤ (C₂ ⥤ D₂) ⥤ (D₁ ⥤ D₂ ⥤ E) ⥤ (C₁ ⥤ C₂ ⥤ E) where
  obj F₁ :=
    { obj := fun F₂ ↦
        (whiskeringRight D₁ (D₂ ⥤ E) (C₂ ⥤ E)).obj ((whiskeringLeft C₂ D₂ E).obj F₂) ⋙
          (whiskeringLeft C₁ D₁ (C₂ ⥤ E)).obj F₁
      map := fun φ ↦ whiskerRight
        ((whiskeringRight D₁ (D₂ ⥤ E) (C₂ ⥤ E)).map ((whiskeringLeft C₂ D₂ E).map φ)) _ }
  map ψ :=
    { app := fun F₂ ↦ whiskerLeft _ ((whiskeringLeft C₁ D₁ (C₂ ⥤ E)).map ψ) }


/-- Auxiliary definition for `whiskeringLeft₃`. -/
@[simps!]
def whiskeringLeft₃ObjObjObj (F₁ : C₁ ⥤ D₁) (F₂ : C₂ ⥤ D₂) (F₃ : C₃ ⥤ D₃) :
    (D₁ ⥤ D₂ ⥤ D₃ ⥤ E) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ E :=
  (whiskeringRight _ _ _).obj (((whiskeringLeft₂ E).obj F₂).obj F₃) ⋙
    (whiskeringLeft C₁ D₁ _).obj F₁


/-- Auxiliary definition for `whiskeringLeft₃`. -/
@[simps]
def whiskeringLeft₃ObjObjMap (F₁ : C₁ ⥤ D₁) (F₂ : C₂ ⥤ D₂) {F₃ F₃' : C₃ ⥤ D₃} (τ₃ : F₃ ⟶ F₃') :
    whiskeringLeft₃ObjObjObj E F₁ F₂ F₃ ⟶
      whiskeringLeft₃ObjObjObj E F₁ F₂ F₃' where
  app F := whiskerLeft _ (whiskerLeft _ (((whiskeringLeft₂ E).obj F₂).map τ₃))


variable (C₃ D₃) in
/-- Auxiliary definition for `whiskeringLeft₃`. -/
@[simps]
def whiskeringLeft₃ObjObj (F₁ : C₁ ⥤ D₁) (F₂ : C₂ ⥤ D₂) :
    (C₃ ⥤ D₃) ⥤ (D₁ ⥤ D₂ ⥤ D₃ ⥤ E) ⥤ (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) where
  obj F₃ := whiskeringLeft₃ObjObjObj E F₁ F₂ F₃
  map τ₃ := whiskeringLeft₃ObjObjMap E F₁ F₂ τ₃


variable (C₃ D₃) in
/-- Auxiliary definition for `whiskeringLeft₃`. -/
@[simps]
def whiskeringLeft₃ObjMap (F₁ : C₁ ⥤ D₁) {F₂ F₂' : C₂ ⥤ D₂} (τ₂ : F₂ ⟶ F₂') :
    whiskeringLeft₃ObjObj C₃ D₃ E F₁ F₂ ⟶ whiskeringLeft₃ObjObj C₃ D₃ E F₁ F₂' where
  app F₃ := whiskerRight ((whiskeringRight _ _ _).map (((whiskeringLeft₂ E).map τ₂).app F₃)) _


variable (C₂ C₃ D₂ D₃) in
/-- Auxiliary definition for `whiskeringLeft₃`. -/
@[simps]
def whiskeringLeft₃Obj (F₁ : C₁ ⥤ D₁) :
    (C₂ ⥤ D₂) ⥤ (C₃ ⥤ D₃) ⥤ (D₁ ⥤ D₂ ⥤ D₃ ⥤ E) ⥤ (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) where
  obj F₂ := whiskeringLeft₃ObjObj C₃ D₃ E F₁ F₂
  map τ₂ := whiskeringLeft₃ObjMap C₃ D₃ E F₁ τ₂


variable (C₂ C₃ D₂ D₃) in
/-- Auxiliary definition for `whiskeringLeft₃`. -/
@[simps]
def whiskeringLeft₃Map {F₁ F₁' : C₁ ⥤ D₁} (τ₁ : F₁ ⟶ F₁') :
    whiskeringLeft₃Obj C₂ C₃ D₂ D₃ E F₁ ⟶ whiskeringLeft₃Obj C₂ C₃ D₂ D₃ E F₁' where
  app F₂ := { app F₃ := whiskerLeft _ ((whiskeringLeft _ _ _).map τ₁) }


/-- The obvious functor `(C₁ ⥤ D₁) ⥤ (C₂ ⥤ D₂) ⥤ (D₁ ⥤ D₂ ⥤ E) ⥤ (C₁ ⥤ C₂ ⥤ E)`. -/
@[simps!]
def whiskeringLeft₃ :
    (C₁ ⥤ D₁) ⥤ (C₂ ⥤ D₂) ⥤ (C₃ ⥤ D₃) ⥤ (D₁ ⥤ D₂ ⥤ D₃ ⥤ E) ⥤ (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) where
  obj F₁ := whiskeringLeft₃Obj C₂ C₃ D₂ D₃ E F₁
  map τ₁ := whiskeringLeft₃Map C₂ C₃ D₂ D₃ E τ₁


variable {E} in
/-- The "postcomposition" with a functor `E ⥤ E'` gives a functor
`(E ⥤ E') ⥤ (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ E'`. -/
@[simps!]
def Functor.postcompose₃ {E' : Type*} [Category E'] :
    (E ⥤ E') ⥤ (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ E' :=
  whiskeringRight C₃ _ _ ⋙ whiskeringRight C₂ _ _ ⋙ whiskeringRight C₁ _ _


