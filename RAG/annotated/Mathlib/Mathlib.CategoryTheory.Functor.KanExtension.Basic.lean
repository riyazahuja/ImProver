/-- Given two functors `L : C ⥤ D` and `F : C ⥤ H`, this is the category of functors
`F' : H ⥤ D` equipped with a natural transformation `L ⋙ F' ⟶ F`. -/
abbrev RightExtension (L : C ⥤ D) (F : C ⥤ H) :=
  CostructuredArrow ((whiskeringLeft C D H).obj L) F


/-- Given two functors `L : C ⥤ D` and `F : C ⥤ H`, this is the category of functors
`F' : H ⥤ D` equipped with a natural transformation `F ⟶ L ⋙ F'`. -/
abbrev LeftExtension (L : C ⥤ D) (F : C ⥤ H) :=
  StructuredArrow F ((whiskeringLeft C D H).obj L)


/-- Constructor for objects of the category `Functor.RightExtension L F`. -/
@[simps!]
def RightExtension.mk (F' : D ⥤ H) {L : C ⥤ D} {F : C ⥤ H} (α : L ⋙ F' ⟶ F) :
    RightExtension L F :=
  CostructuredArrow.mk α


/-- Constructor for objects of the category `Functor.LeftExtension L F`. -/
@[simps!]
def LeftExtension.mk (F' : D ⥤ H) {L : C ⥤ D} {F : C ⥤ H} (α : F ⟶ L ⋙ F') :
    LeftExtension L F :=
  StructuredArrow.mk α


/-- Given `α : L ⋙ F' ⟶ F`, the property `F'.IsRightKanExtension α` asserts that
`(F', α)` is a terminal object in the category `RightExtension L F`, i.e. that `(F', α)`
is a right Kan extension of `F` along `L`. -/
class IsRightKanExtension : Prop where
  nonempty_isUniversal : Nonempty (RightExtension.mk F' α).IsUniversal


/-- If `(F', α)` is a right Kan extension of `F` along `L`, then `(F', α)` is a terminal object
in the category `RightExtension L F`. -/
noncomputable def isUniversalOfIsRightKanExtension : (RightExtension.mk F' α).IsUniversal :=
  IsRightKanExtension.nonempty_isUniversal.some


/-- If `(F', α)` is a right Kan extension of `F` along `L` and `β : L ⋙ G ⟶ F` is
a natural transformation, this is the induced morphism `G ⟶ F'`. -/
noncomputable def liftOfIsRightKanExtension (G : D ⥤ H) (β : L ⋙ G ⟶ F) : G ⟶ F' :=
  (F'.isUniversalOfIsRightKanExtension α).lift (RightExtension.mk G β)


@[reassoc (attr := simp)]
lemma liftOfIsRightKanExtension_fac (G : D ⥤ H) (β : L ⋙ G ⟶ F) :
    whiskerLeft L (F'.liftOfIsRightKanExtension α G β) ≫ α = β :=
  (F'.isUniversalOfIsRightKanExtension α).fac (RightExtension.mk G β)


@[reassoc (attr := simp)]
lemma liftOfIsRightKanExtension_fac_app (G : D ⥤ H) (β : L ⋙ G ⟶ F) (X : C) :
    (F'.liftOfIsRightKanExtension α G β).app (L.obj X) ≫ α.app X = β.app X :=
  NatTrans.congr_app (F'.liftOfIsRightKanExtension_fac α G β) X


lemma hom_ext_of_isRightKanExtension {G : D ⥤ H} (γ₁ γ₂ : G ⟶ F')
    (hγ : whiskerLeft L γ₁ ≫ α = whiskerLeft L γ₂ ≫ α) : γ₁ = γ₂ :=
  (F'.isUniversalOfIsRightKanExtension α).hom_ext hγ


/-- If `(F', α)` is a right Kan extension of `F` along `L`, then this
is the induced bijection `(G ⟶ F') ≃ (L ⋙ G ⟶ F)` for all `G`. -/
noncomputable def homEquivOfIsRightKanExtension (G : D ⥤ H) :
    (G ⟶ F') ≃ (L ⋙ G ⟶ F) where
  toFun β := whiskerLeft _ β ≫ α
  invFun β := liftOfIsRightKanExtension _ α _ β
                                                                   /-
                                                                     C : Type u_1
                                                                     C' : Type u_2
                                                                     H : Type u_3
                                                                     D : Type u_4
                                                                     D' : Type u_5
                                                                     inst✝⁵ : CategoryTheory.Category.{?u.14985, u_1} C
                                                                     inst✝⁴ : CategoryTheory.Category.{?u.14989, u_2} C'
                                                                     inst✝³ : CategoryTheory.Category.{?u.14993, u_3} H
                                                                     inst✝² : CategoryTheory.Category.{?u.14997, u_4} D
                                                                     inst✝¹ : CategoryTheory.Category.{?u.15001, u_5} D'
                                                                     F' : CategoryTheory.Functor D H
                                                                     L : CategoryTheory.Functor C D
                                                                     F : CategoryTheory.Functor C H
                                                                     α : Quiver.Hom (L.comp F') F
                                                                     inst✝ : F'.IsRightKanExtension α
                                                                     G : CategoryTheory.Functor D H
                                                                     β : Quiver.Hom G F'
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L ((fun β …
                                                                   -/
  left_inv β := Functor.hom_ext_of_isRightKanExtension _ α _ _ (by aesop_cat)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                  /-
                    C : Type u_1
                    C' : Type u_2
                    H : Type u_3
                    D : Type u_4
                    D' : Type u_5
                    inst✝⁵ : CategoryTheory.Category.{?u.14985, u_1} C
                    inst✝⁴ : CategoryTheory.Category.{?u.14989, u_2} C'
                    inst✝³ : CategoryTheory.Category.{?u.14993, u_3} H
                    inst✝² : CategoryTheory.Category.{?u.14997, u_4} D
                    inst✝¹ : CategoryTheory.Category.{?u.15001, u_5} D'
                    F' : CategoryTheory.Functor D H
                    L : CategoryTheory.Functor C D
                    F : CategoryTheory.Functor C H
                    α : Quiver.Hom (L.comp F') F
                    inst✝ : F'.IsRightKanExtension α
                    G : CategoryTheory.Functor D H
                    ⊢ Function.RightInverse (fun β => F'.liftOfIsRightKanExtension α G β) fun β => …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


lemma isRightKanExtension_of_iso {F' F'' : D ⥤ H} (e : F' ≅ F'') {L : C ⥤ D} {F : C ⥤ H}
    (α : L ⋙ F' ⟶ F) (α' : L ⋙ F'' ⟶ F) (comm : whiskerLeft L e.hom ≫ α' = α)
    [F'.IsRightKanExtension α] : F''.IsRightKanExtension α' where
  nonempty_isUniversal := ⟨IsTerminal.ofIso (F'.isUniversalOfIsRightKanExtension α)
    (CostructuredArrow.isoMk e comm)⟩


lemma isRightKanExtension_iff_of_iso {F' F'' : D ⥤ H} (e : F' ≅ F'') {L : C ⥤ D} {F : C ⥤ H}
    (α : L ⋙ F' ⟶ F) (α' : L ⋙ F'' ⟶ F) (comm : whiskerLeft L e.hom ≫ α' = α) :
    F'.IsRightKanExtension α ↔ F''.IsRightKanExtension α' := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
    inst✝ : CategoryTheory.Category.{u_6, u_4} D
    F' F'' : CategoryTheory.Functor D H
    e : CategoryTheory.Iso F' F''
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom (L.comp F') F
    α' : Quiver.Hom (L.comp F'') F
    comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e. …
    ⊢ Iff (F'.IsRightKanExtension α) (F''.IsRightKanExtension α')
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e. …
      ⊢ F'.IsRightKanExtension α → F''.IsRightKanExtension α'
    -/
  · intro
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e. …
      a✝ : F'.IsRightKanExtension α
      ⊢ F''.IsRightKanExtension α'
    -/
    exact isRightKanExtension_of_iso e α α' comm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e. …
      ⊢ F''.IsRightKanExtension α' → F'.IsRightKanExtension α
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e. …
      a✝ : F''.IsRightKanExtension α'
      ⊢ F'.IsRightKanExtension α
    -/
    refine isRightKanExtension_of_iso e.symm α' α ?_
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e. …
      a✝ : F''.IsRightKanExtension α'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e.symm. …
    -/
    rw [← comm, ← whiskerLeft_comp_assoc, Iso.symm_hom, e.inv_hom_id, whiskerLeft_id', id_comp]
    /-
      🎉 no goals
    -/


/-- Right Kan extensions of isomorphic functors are isomorphic. -/
@[simps]
noncomputable def rightKanExtensionUniqueOfIso {G : C ⥤ H} (i : F ≅ G) (G' : D ⥤ H)
    (β : L ⋙ G' ⟶ G) [G'.IsRightKanExtension β] : F' ≅ G' where
  hom := liftOfIsRightKanExtension _ β F' (α ≫ i.hom)
  inv := liftOfIsRightKanExtension _ α G' (β ≫ i.inv)
                                                            /-
                                                              C : Type u_1
                                                              C' : Type u_2
                                                              H : Type u_3
                                                              D : Type u_4
                                                              D' : Type u_5
                                                              inst✝⁶ : CategoryTheory.Category.{?u.24348, u_1} C
                                                              inst✝⁵ : CategoryTheory.Category.{?u.24352, u_2} C'
                                                              inst✝⁴ : CategoryTheory.Category.{?u.24356, u_3} H
                                                              inst✝³ : CategoryTheory.Category.{?u.24360, u_4} D
                                                              inst✝² : CategoryTheory.Category.{?u.24364, u_5} D'
                                                              F' : CategoryTheory.Functor D H
                                                              L : CategoryTheory.Functor C D
                                                              F : CategoryTheory.Functor C H
                                                              α : Quiver.Hom (L.comp F') F
                                                              inst✝¹ : F'.IsRightKanExtension α
                                                              G : CategoryTheory.Functor C H
                                                              i : CategoryTheory.Iso F G
                                                              G' : CategoryTheory.Functor D H
                                                              β : Quiver.Hom (L.comp G') G
                                                              inst✝ : G'.IsRightKanExtension β
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L (Catego …
                                                            -/
  hom_inv_id := F'.hom_ext_of_isRightKanExtension α _ _ (by simp)
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              C : Type u_1
                                                              C' : Type u_2
                                                              H : Type u_3
                                                              D : Type u_4
                                                              D' : Type u_5
                                                              inst✝⁶ : CategoryTheory.Category.{?u.24348, u_1} C
                                                              inst✝⁵ : CategoryTheory.Category.{?u.24352, u_2} C'
                                                              inst✝⁴ : CategoryTheory.Category.{?u.24356, u_3} H
                                                              inst✝³ : CategoryTheory.Category.{?u.24360, u_4} D
                                                              inst✝² : CategoryTheory.Category.{?u.24364, u_5} D'
                                                              F' : CategoryTheory.Functor D H
                                                              L : CategoryTheory.Functor C D
                                                              F : CategoryTheory.Functor C H
                                                              α : Quiver.Hom (L.comp F') F
                                                              inst✝¹ : F'.IsRightKanExtension α
                                                              G : CategoryTheory.Functor C H
                                                              i : CategoryTheory.Iso F G
                                                              G' : CategoryTheory.Functor D H
                                                              β : Quiver.Hom (L.comp G') G
                                                              inst✝ : G'.IsRightKanExtension β
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L (Catego …
                                                            -/
  inv_hom_id := G'.hom_ext_of_isRightKanExtension β _ _ (by simp)
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- Two right Kan extensions are (canonically) isomorphic. -/
@[simps!]
noncomputable def rightKanExtensionUnique
    (F'' : D ⥤ H) (α' : L ⋙ F'' ⟶ F) [F''.IsRightKanExtension α'] : F' ≅ F'' :=
  rightKanExtensionUniqueOfIso F' α (Iso.refl _) F'' α'



lemma isRightKanExtension_iff_isIso {F' : D ⥤ H} {F'' : D ⥤ H} (φ : F'' ⟶ F')
    {L : C ⥤ D} {F : C ⥤ H} (α : L ⋙ F' ⟶ F) (α' : L ⋙ F'' ⟶ F)
    (comm : whiskerLeft L φ ≫ α = α') [F'.IsRightKanExtension α] :
    F''.IsRightKanExtension α' ↔ IsIso φ := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C
    inst✝² : CategoryTheory.Category.{u_7, u_3} H
    inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
    F' F'' : CategoryTheory.Functor D H
    φ : Quiver.Hom F'' F'
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom (L.comp F') F
    α' : Quiver.Hom (L.comp F'') F
    comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L φ) …
    inst✝ : F'.IsRightKanExtension α
    ⊢ Iff (F''.IsRightKanExtension α') (CategoryTheory.IsIso φ)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F'' F'
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L φ) …
      inst✝ : F'.IsRightKanExtension α
      ⊢ F''.IsRightKanExtension α' → CategoryTheory.IsIso φ
    -/
  · intro
    rw [F'.hom_ext_of_isRightKanExtension α φ (rightKanExtensionUnique _ α' _ α).hom
      (by simp [comm])]
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F'' F'
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L φ) …
      inst✝ : F'.IsRightKanExtension α
      a✝ : F''.IsRightKanExtension α'
      ⊢ CategoryTheory.IsIso (F''.rightKanExtensionUnique α' F' α).hom
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F'' F'
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L φ) …
      inst✝ : F'.IsRightKanExtension α
      ⊢ CategoryTheory.IsIso φ → F''.IsRightKanExtension α'
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F'' F'
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L φ) …
      inst✝ : F'.IsRightKanExtension α
      a✝ : CategoryTheory.IsIso φ
      ⊢ F''.IsRightKanExtension α'
    -/
    rw [isRightKanExtension_iff_of_iso (asIso φ) α' α comm]
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F'' F'
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      α' : Quiver.Hom (L.comp F'') F
      comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L φ) …
      inst✝ : F'.IsRightKanExtension α
      a✝ : CategoryTheory.IsIso φ
      ⊢ F'.IsRightKanExtension α
    -/
    infer_instance
    /-
      🎉 no goals
    -/

/-- Given `α : F ⟶ L ⋙ F'`, the property `F'.IsLeftKanExtension α` asserts that
`(F', α)` is an initial object in the category `LeftExtension L F`, i.e. that `(F', α)`
is a left Kan extension of `F` along `L`. -/
class IsLeftKanExtension : Prop where
  nonempty_isUniversal : Nonempty (LeftExtension.mk F' α).IsUniversal


/-- If `(F', α)` is a left Kan extension of `F` along `L`, then `(F', α)` is an initial object
in the category `LeftExtension L F`. -/
noncomputable def isUniversalOfIsLeftKanExtension : (LeftExtension.mk F' α).IsUniversal :=
  IsLeftKanExtension.nonempty_isUniversal.some


/-- If `(F', α)` is a left Kan extension of `F` along `L` and `β : F ⟶ L ⋙ G` is
a natural transformation, this is the induced morphism `F' ⟶ G`. -/
noncomputable def descOfIsLeftKanExtension (G : D ⥤ H) (β : F ⟶ L ⋙ G) : F' ⟶ G :=
  (F'.isUniversalOfIsLeftKanExtension α).desc (LeftExtension.mk G β)


@[reassoc (attr := simp)]
lemma descOfIsLeftKanExtension_fac (G : D ⥤ H) (β : F ⟶ L ⋙ G) :
    α ≫ whiskerLeft L (F'.descOfIsLeftKanExtension α G β) = β :=
  (F'.isUniversalOfIsLeftKanExtension α).fac (LeftExtension.mk G β)


@[reassoc (attr := simp)]
lemma descOfIsLeftKanExtension_fac_app (G : D ⥤ H) (β : F ⟶ L ⋙ G) (X : C) :
    α.app X ≫ (F'.descOfIsLeftKanExtension α G β).app (L.obj X) = β.app X :=
  NatTrans.congr_app (F'.descOfIsLeftKanExtension_fac α G β) X


lemma hom_ext_of_isLeftKanExtension {G : D ⥤ H} (γ₁ γ₂ : F' ⟶ G)
    (hγ : α ≫ whiskerLeft L γ₁ = α ≫ whiskerLeft L γ₂) : γ₁ = γ₂ :=
  (F'.isUniversalOfIsLeftKanExtension α).hom_ext hγ


/-- If `(F', α)` is a left Kan extension of `F` along `L`, then this
is the induced bijection `(F' ⟶ G) ≃ (F ⟶ L ⋙ G)` for all `G`. -/
noncomputable def homEquivOfIsLeftKanExtension (G : D ⥤ H) :
    (F' ⟶ G) ≃ (F ⟶ L ⋙ G) where
  toFun β := α ≫ whiskerLeft _ β
  invFun β := descOfIsLeftKanExtension _ α _ β
                                                                  /-
                                                                    C : Type u_1
                                                                    C' : Type u_2
                                                                    H : Type u_3
                                                                    D : Type u_4
                                                                    D' : Type u_5
                                                                    inst✝⁵ : CategoryTheory.Category.{?u.43796, u_1} C
                                                                    inst✝⁴ : CategoryTheory.Category.{?u.43800, u_2} C'
                                                                    inst✝³ : CategoryTheory.Category.{?u.43804, u_3} H
                                                                    inst✝² : CategoryTheory.Category.{?u.43808, u_4} D
                                                                    inst✝¹ : CategoryTheory.Category.{?u.43812, u_5} D'
                                                                    F' : CategoryTheory.Functor D H
                                                                    L : CategoryTheory.Functor C D
                                                                    F : CategoryTheory.Functor C H
                                                                    α : Quiver.Hom F (L.comp F')
                                                                    inst✝ : F'.IsLeftKanExtension α
                                                                    G : CategoryTheory.Functor D H
                                                                    β : Quiver.Hom F' G
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L ((fun …
                                                                  -/
  left_inv β := Functor.hom_ext_of_isLeftKanExtension _ α _ _ (by aesop_cat)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                  /-
                    C : Type u_1
                    C' : Type u_2
                    H : Type u_3
                    D : Type u_4
                    D' : Type u_5
                    inst✝⁵ : CategoryTheory.Category.{?u.43796, u_1} C
                    inst✝⁴ : CategoryTheory.Category.{?u.43800, u_2} C'
                    inst✝³ : CategoryTheory.Category.{?u.43804, u_3} H
                    inst✝² : CategoryTheory.Category.{?u.43808, u_4} D
                    inst✝¹ : CategoryTheory.Category.{?u.43812, u_5} D'
                    F' : CategoryTheory.Functor D H
                    L : CategoryTheory.Functor C D
                    F : CategoryTheory.Functor C H
                    α : Quiver.Hom F (L.comp F')
                    inst✝ : F'.IsLeftKanExtension α
                    G : CategoryTheory.Functor D H
                    ⊢ Function.RightInverse (fun β => F'.descOfIsLeftKanExtension α G β) fun β =>  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


lemma isLeftKanExtension_of_iso {F' : D ⥤ H} {F'' : D ⥤ H} (e : F' ≅ F'')
    {L : C ⥤ D} {F : C ⥤ H} (α : F ⟶ L ⋙ F') (α' : F ⟶ L ⋙ F'')
    (comm : α ≫ whiskerLeft L e.hom = α') [F'.IsLeftKanExtension α] :
    F''.IsLeftKanExtension α' where
  nonempty_isUniversal := ⟨IsInitial.ofIso (F'.isUniversalOfIsLeftKanExtension α)
    (StructuredArrow.isoMk e comm)⟩


lemma isLeftKanExtension_iff_of_iso {F' F'' : D ⥤ H} (e : F' ≅ F'')
    {L : C ⥤ D} {F : C ⥤ H} (α : F ⟶ L ⋙ F') (α' : F ⟶ L ⋙ F'')
    (comm : α ≫ whiskerLeft L e.hom = α') :
    F'.IsLeftKanExtension α ↔ F''.IsLeftKanExtension α' := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
    inst✝ : CategoryTheory.Category.{u_6, u_4} D
    F' F'' : CategoryTheory.Functor D H
    e : CategoryTheory.Iso F' F''
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom F (L.comp F')
    α' : Quiver.Hom F (L.comp F'')
    comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
    ⊢ Iff (F'.IsLeftKanExtension α) (F''.IsLeftKanExtension α')
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      ⊢ F'.IsLeftKanExtension α → F''.IsLeftKanExtension α'
    -/
  · intro
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      a✝ : F'.IsLeftKanExtension α
      ⊢ F''.IsLeftKanExtension α'
    -/
    exact isLeftKanExtension_of_iso e α α' comm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      ⊢ F''.IsLeftKanExtension α' → F'.IsLeftKanExtension α
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      a✝ : F''.IsLeftKanExtension α'
      ⊢ F'.IsLeftKanExtension α
    -/
    refine isLeftKanExtension_of_iso e.symm α' α ?_
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      e : CategoryTheory.Iso F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      a✝ : F''.IsLeftKanExtension α'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp α' (CategoryTheory.whiskerLeft L e.sy …
    -/
    rw [← comm, assoc, ← whiskerLeft_comp, Iso.symm_hom, e.hom_inv_id, whiskerLeft_id', comp_id]
    /-
      🎉 no goals
    -/


/-- Left Kan extensions of isomorphic functors are isomorphic. -/
@[simps]
noncomputable def leftKanExtensionUniqueOfIso {G : C ⥤ H} (i : F ≅ G) (G' : D ⥤ H)
    (β : G ⟶ L ⋙ G') [G'.IsLeftKanExtension β] : F' ≅ G' where
  hom := descOfIsLeftKanExtension _ α G' (i.hom ≫ β)
  inv := descOfIsLeftKanExtension _ β F' (i.inv ≫ α)
                                                           /-
                                                             C : Type u_1
                                                             C' : Type u_2
                                                             H : Type u_3
                                                             D : Type u_4
                                                             D' : Type u_5
                                                             inst✝⁶ : CategoryTheory.Category.{?u.53379, u_1} C
                                                             inst✝⁵ : CategoryTheory.Category.{?u.53383, u_2} C'
                                                             inst✝⁴ : CategoryTheory.Category.{?u.53387, u_3} H
                                                             inst✝³ : CategoryTheory.Category.{?u.53391, u_4} D
                                                             inst✝² : CategoryTheory.Category.{?u.53395, u_5} D'
                                                             F' : CategoryTheory.Functor D H
                                                             L : CategoryTheory.Functor C D
                                                             F : CategoryTheory.Functor C H
                                                             α : Quiver.Hom F (L.comp F')
                                                             inst✝¹ : F'.IsLeftKanExtension α
                                                             G : CategoryTheory.Functor C H
                                                             i : CategoryTheory.Iso F G
                                                             G' : CategoryTheory.Functor D H
                                                             β : Quiver.Hom G (L.comp G')
                                                             inst✝ : G'.IsLeftKanExtension β
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L (Cate …
                                                           -/
  hom_inv_id := F'.hom_ext_of_isLeftKanExtension α _ _ (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             C : Type u_1
                                                             C' : Type u_2
                                                             H : Type u_3
                                                             D : Type u_4
                                                             D' : Type u_5
                                                             inst✝⁶ : CategoryTheory.Category.{?u.53379, u_1} C
                                                             inst✝⁵ : CategoryTheory.Category.{?u.53383, u_2} C'
                                                             inst✝⁴ : CategoryTheory.Category.{?u.53387, u_3} H
                                                             inst✝³ : CategoryTheory.Category.{?u.53391, u_4} D
                                                             inst✝² : CategoryTheory.Category.{?u.53395, u_5} D'
                                                             F' : CategoryTheory.Functor D H
                                                             L : CategoryTheory.Functor C D
                                                             F : CategoryTheory.Functor C H
                                                             α : Quiver.Hom F (L.comp F')
                                                             inst✝¹ : F'.IsLeftKanExtension α
                                                             G : CategoryTheory.Functor C H
                                                             i : CategoryTheory.Iso F G
                                                             G' : CategoryTheory.Functor D H
                                                             β : Quiver.Hom G (L.comp G')
                                                             inst✝ : G'.IsLeftKanExtension β
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp β (CategoryTheory.whiskerLeft L (Cate …
                                                           -/
  inv_hom_id := G'.hom_ext_of_isLeftKanExtension β _ _ (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Two left Kan extensions are (canonically) isomorphic. -/
@[simps!]
noncomputable def leftKanExtensionUnique
    (F'' : D ⥤ H) (α' : F ⟶ L ⋙ F'') [F''.IsLeftKanExtension α'] : F' ≅ F'' :=
  leftKanExtensionUniqueOfIso F' α (Iso.refl _) F'' α'


lemma isLeftKanExtension_iff_isIso {F' : D ⥤ H} {F'' : D ⥤ H} (φ : F' ⟶ F'')
    {L : C ⥤ D} {F : C ⥤ H} (α : F ⟶ L ⋙ F') (α' : F ⟶ L ⋙ F'')
    (comm : α ≫ whiskerLeft L φ = α') [F'.IsLeftKanExtension α] :
    F''.IsLeftKanExtension α' ↔ IsIso φ := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C
    inst✝² : CategoryTheory.Category.{u_7, u_3} H
    inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
    F' F'' : CategoryTheory.Functor D H
    φ : Quiver.Hom F' F''
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom F (L.comp F')
    α' : Quiver.Hom F (L.comp F'')
    comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
    inst✝ : F'.IsLeftKanExtension α
    ⊢ Iff (F''.IsLeftKanExtension α') (CategoryTheory.IsIso φ)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      inst✝ : F'.IsLeftKanExtension α
      ⊢ F''.IsLeftKanExtension α' → CategoryTheory.IsIso φ
    -/
  · intro
    rw [F'.hom_ext_of_isLeftKanExtension α φ (leftKanExtensionUnique _ α _ α').hom
      (by simp [comm])]
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      inst✝ : F'.IsLeftKanExtension α
      a✝ : F''.IsLeftKanExtension α'
      ⊢ CategoryTheory.IsIso (F'.leftKanExtensionUnique α F'' α').hom
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      inst✝ : F'.IsLeftKanExtension α
      ⊢ CategoryTheory.IsIso φ → F''.IsLeftKanExtension α'
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C
      inst✝² : CategoryTheory.Category.{u_7, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_6, u_4} D
      F' F'' : CategoryTheory.Functor D H
      φ : Quiver.Hom F' F''
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      α' : Quiver.Hom F (L.comp F'')
      comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
      inst✝ : F'.IsLeftKanExtension α
      a✝ : CategoryTheory.IsIso φ
      ⊢ F''.IsLeftKanExtension α'
    -/
    exact isLeftKanExtension_of_iso (asIso φ) α α' comm
    /-
      🎉 no goals
    -/


/-- This property `HasRightKanExtension L F` holds when the functor `F` has a right
Kan extension along `L`. -/
abbrev HasRightKanExtension (L : C ⥤ D) (F : C ⥤ H) := HasTerminal (RightExtension L F)


lemma HasRightKanExtension.mk (F' : D ⥤ H) {L : C ⥤ D} {F : C ⥤ H} (α : L ⋙ F' ⟶ F)
    [F'.IsRightKanExtension α] : HasRightKanExtension L F :=
  (F'.isUniversalOfIsRightKanExtension α).hasTerminal


/-- This property `HasLeftKanExtension L F` holds when the functor `F` has a left
Kan extension along `L`. -/
abbrev HasLeftKanExtension (L : C ⥤ D) (F : C ⥤ H) := HasInitial (LeftExtension L F)


lemma HasLeftKanExtension.mk (F' : D ⥤ H) {L : C ⥤ D} {F : C ⥤ H} (α : F ⟶ L ⋙ F')
    [F'.IsLeftKanExtension α] : HasLeftKanExtension L F :=
  (F'.isUniversalOfIsLeftKanExtension α).hasInitial


/-- A chosen right Kan extension when `[HasRightKanExtension L F]` holds. -/
noncomputable def rightKanExtension : D ⥤ H := (⊤_ _ : RightExtension L F).left


/-- The counit of the chosen right Kan extension `rightKanExtension L F`. -/
noncomputable def rightKanExtensionCounit : L ⋙ rightKanExtension L F ⟶ F :=
  (⊤_ _ : RightExtension L F).hom


instance : (L.rightKanExtension F).IsRightKanExtension (L.rightKanExtensionCounit F) where
  nonempty_isUniversal := ⟨terminalIsTerminal⟩


@[ext]
lemma rightKanExtension_hom_ext {G : D ⥤ H} (γ₁ γ₂ : G ⟶ rightKanExtension L F)
    (hγ : whiskerLeft L γ₁ ≫ rightKanExtensionCounit L F =
      whiskerLeft L γ₂ ≫ rightKanExtensionCounit L F) :
    γ₁ = γ₂ :=
  hom_ext_of_isRightKanExtension _ _ _ _ hγ


/-- A chosen left Kan extension when `[HasLeftKanExtension L F]` holds. -/
noncomputable def leftKanExtension : D ⥤ H := (⊥_ _ : LeftExtension L F).right


/-- The unit of the chosen left Kan extension `leftKanExtension L F`. -/
noncomputable def leftKanExtensionUnit : F ⟶ L ⋙ leftKanExtension L F :=
  (⊥_ _ : LeftExtension L F).hom


instance : (L.leftKanExtension F).IsLeftKanExtension (L.leftKanExtensionUnit F) where
  nonempty_isUniversal := ⟨initialIsInitial⟩


@[ext]
lemma leftKanExtension_hom_ext {G : D ⥤ H} (γ₁ γ₂ : leftKanExtension L F ⟶ G)
    (hγ : leftKanExtensionUnit L F ≫ whiskerLeft L γ₁ =
      leftKanExtensionUnit L F ≫ whiskerLeft L γ₂) : γ₁ = γ₂ :=
  hom_ext_of_isLeftKanExtension _ _ _ _ hγ


/-- The functor `LeftExtension L' F ⥤ LeftExtension L F`
induced by a natural transformation `L' ⟶ L ⋙ G'`. -/
@[simps!]
def LeftExtension.postcomp₁ (f : L' ⟶ L ⋙ G) (F : C ⥤ H) :
    LeftExtension L' F ⥤ LeftExtension L F :=
  StructuredArrow.map₂ (F := (whiskeringLeft D D' H).obj G) (G := 𝟭 _) (𝟙 _)
    ((whiskeringLeft C D' H).map f)


/-- The functor `RightExtension L' F ⥤ RightExtension L F`
induced by a natural transformation `L ⋙ G ⟶ L'`. -/
@[simps!]
def RightExtension.postcomp₁ (f : L ⋙ G ⟶ L') (F : C ⥤ H) :
    RightExtension L' F ⥤ RightExtension L F :=
  CostructuredArrow.map₂ (F := (whiskeringLeft D D' H).obj G) (G := 𝟭 _)
    ((whiskeringLeft C D' H).map f) (𝟙 _)


noncomputable instance (f : L' ⟶ L ⋙ G) [IsIso f] (F : C ⥤ H) :
    IsEquivalence (LeftExtension.postcomp₁ G f F) := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁵ : CategoryTheory.Category.{?u.89420, u_2} C'
    inst✝⁴ : CategoryTheory.Category.{u_9, u_3} H
    inst✝³ : CategoryTheory.Category.{u_8, u_4} D
    inst✝² : CategoryTheory.Category.{u_6, u_5} D'
    L : CategoryTheory.Functor C D
    L' : CategoryTheory.Functor C D'
    G : CategoryTheory.Functor D D'
    inst✝¹ : G.IsEquivalence
    f : Quiver.Hom L' (L.comp G)
    inst✝ : CategoryTheory.IsIso f
    F : CategoryTheory.Functor C H
    ⊢ (CategoryTheory.Functor.LeftExtension.postcomp₁ G f F).IsEquivalence
  -/
  apply StructuredArrow.isEquivalenceMap₂
  /-
    🎉 no goals
  -/


noncomputable instance (f : L ⋙ G ⟶ L') [IsIso f] (F : C ⥤ H) :
    IsEquivalence (RightExtension.postcomp₁ G f F) := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁶ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁵ : CategoryTheory.Category.{?u.91963, u_2} C'
    inst✝⁴ : CategoryTheory.Category.{u_9, u_3} H
    inst✝³ : CategoryTheory.Category.{u_8, u_4} D
    inst✝² : CategoryTheory.Category.{u_6, u_5} D'
    L : CategoryTheory.Functor C D
    L' : CategoryTheory.Functor C D'
    G : CategoryTheory.Functor D D'
    inst✝¹ : G.IsEquivalence
    f : Quiver.Hom (L.comp G) L'
    inst✝ : CategoryTheory.IsIso f
    F : CategoryTheory.Functor C H
    ⊢ (CategoryTheory.Functor.RightExtension.postcomp₁ G f F).IsEquivalence
  -/
  apply CostructuredArrow.isEquivalenceMap₂
  /-
    🎉 no goals
  -/


variable {G} in
lemma hasLeftExtension_iff_postcomp₁ (e : L ⋙ G ≅ L') (F : C ⥤ H) :
    HasLeftKanExtension L' F ↔ HasLeftKanExtension L F :=
  (LeftExtension.postcomp₁ G e.inv F).asEquivalence.hasInitial_iff


variable {G} in
lemma hasRightExtension_iff_postcomp₁ (e : L ⋙ G ≅ L') (F : C ⥤ H) :
    HasRightKanExtension L' F ↔ HasRightKanExtension L F :=
  (RightExtension.postcomp₁ G e.hom F).asEquivalence.hasTerminal_iff


/-- Given an isomorphism `e : L ⋙ G ≅ L'`, a left extension of `F` along `L'` is universal
iff the corresponding left extension of `L` along `L` is. -/
noncomputable def LeftExtension.isUniversalPostcomp₁Equiv (ex : LeftExtension L' F) :
    ex.IsUniversal ≃ ((LeftExtension.postcomp₁ G e.inv F).obj ex).IsUniversal := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁵ : CategoryTheory.Category.{?u.98948, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.98952, u_2} C'
    inst✝³ : CategoryTheory.Category.{?u.98956, u_3} H
    inst✝² : CategoryTheory.Category.{?u.98960, u_4} D
    inst✝¹ : CategoryTheory.Category.{?u.98964, u_5} D'
    L : CategoryTheory.Functor C D
    L' : CategoryTheory.Functor C D'
    G : CategoryTheory.Functor D D'
    inst✝ : G.IsEquivalence
    e : CategoryTheory.Iso (L.comp G) L'
    F : CategoryTheory.Functor C H
    ex : L'.LeftExtension F
    ⊢ Equiv (CategoryTheory.StructuredArrow.IsUniversal ex) (CategoryTheory.Struct …
  -/
  apply IsInitial.isInitialIffObj (LeftExtension.postcomp₁ G e.inv F)
  /-
    🎉 no goals
  -/


/-- Given an isomorphism `e : L ⋙ G ≅ L'`, a right extension of `F` along `L'` is universal
iff the corresponding right extension of `L` along `L` is. -/
noncomputable def RightExtension.isUniversalPostcomp₁Equiv (ex : RightExtension L' F) :
    ex.IsUniversal ≃ ((RightExtension.postcomp₁ G e.hom F).obj ex).IsUniversal := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁵ : CategoryTheory.Category.{?u.101533, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.101537, u_2} C'
    inst✝³ : CategoryTheory.Category.{?u.101541, u_3} H
    inst✝² : CategoryTheory.Category.{?u.101545, u_4} D
    inst✝¹ : CategoryTheory.Category.{?u.101549, u_5} D'
    L : CategoryTheory.Functor C D
    L' : CategoryTheory.Functor C D'
    G : CategoryTheory.Functor D D'
    inst✝ : G.IsEquivalence
    e : CategoryTheory.Iso (L.comp G) L'
    F : CategoryTheory.Functor C H
    ex : L'.RightExtension F
    ⊢ Equiv (CategoryTheory.CostructuredArrow.IsUniversal ex) (CategoryTheory.Cost …
  -/
  apply IsTerminal.isTerminalIffObj (RightExtension.postcomp₁ G e.hom F)
  /-
    🎉 no goals
  -/


lemma isLeftKanExtension_iff_postcomp₁ (α : F ⟶ L' ⋙ F') :
    F'.IsLeftKanExtension α ↔ (G ⋙ F').IsLeftKanExtension
      (α ≫ whiskerRight e.inv _ ≫ (Functor.associator _ _ _).hom) := by
  let eq : (LeftExtension.mk _ α).IsUniversal ≃
      (LeftExtension.mk _
        (α ≫ whiskerRight e.inv _ ≫ (Functor.associator _ _ _).hom)).IsUniversal :=
    (LeftExtension.isUniversalPostcomp₁Equiv G e F _).trans
    (IsInitial.equivOfIso (StructuredArrow.isoMk (Iso.refl _)))
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_3} H
    inst✝² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹ : CategoryTheory.Category.{u_8, u_5} D'
    L : CategoryTheory.Functor C D
    L' : CategoryTheory.Functor C D'
    G : CategoryTheory.Functor D D'
    inst✝ : G.IsEquivalence
    e : CategoryTheory.Iso (L.comp G) L'
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D' H
    α : Quiver.Hom F (L'.comp F')
    eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
    ⊢ Iff (F'.IsLeftKanExtension α) ((G.comp F').IsLeftKanExtension (CategoryTheor …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_6, u_3} H
      inst✝² : CategoryTheory.Category.{u_9, u_4} D
      inst✝¹ : CategoryTheory.Category.{u_8, u_5} D'
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      G : CategoryTheory.Functor D D'
      inst✝ : G.IsEquivalence
      e : CategoryTheory.Iso (L.comp G) L'
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D' H
      α : Quiver.Hom F (L'.comp F')
      eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
      ⊢ F'.IsLeftKanExtension α → (G.comp F').IsLeftKanExtension (CategoryTheory.Cat …
    -/
  · exact fun _ => ⟨⟨eq (isUniversalOfIsLeftKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_6, u_3} H
      inst✝² : CategoryTheory.Category.{u_9, u_4} D
      inst✝¹ : CategoryTheory.Category.{u_8, u_5} D'
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      G : CategoryTheory.Functor D D'
      inst✝ : G.IsEquivalence
      e : CategoryTheory.Iso (L.comp G) L'
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D' H
      α : Quiver.Hom F (L'.comp F')
      eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
      ⊢ (G.comp F').IsLeftKanExtension (CategoryTheory.CategoryStruct.comp α (Catego …
    -/
  · exact fun _ => ⟨⟨eq.symm (isUniversalOfIsLeftKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/


lemma isRightKanExtension_iff_postcomp₁ (α : L' ⋙ F' ⟶ F) :
    F'.IsRightKanExtension α ↔ (G ⋙ F').IsRightKanExtension
      ((Functor.associator _ _ _).inv ≫ whiskerRight e.hom F' ≫ α) := by
  let eq : (RightExtension.mk _ α).IsUniversal ≃
    (RightExtension.mk _
      ((Functor.associator _ _ _).inv ≫ whiskerRight e.hom F' ≫ α)).IsUniversal :=
  (RightExtension.isUniversalPostcomp₁Equiv G e F _).trans
    (IsTerminal.equivOfIso (CostructuredArrow.isoMk (Iso.refl _)))
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_3} H
    inst✝² : CategoryTheory.Category.{u_9, u_4} D
    inst✝¹ : CategoryTheory.Category.{u_8, u_5} D'
    L : CategoryTheory.Functor C D
    L' : CategoryTheory.Functor C D'
    G : CategoryTheory.Functor D D'
    inst✝ : G.IsEquivalence
    e : CategoryTheory.Iso (L.comp G) L'
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D' H
    α : Quiver.Hom (L'.comp F') F
    eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
    ⊢ Iff (F'.IsRightKanExtension α) ((G.comp F').IsRightKanExtension (CategoryThe …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_6, u_3} H
      inst✝² : CategoryTheory.Category.{u_9, u_4} D
      inst✝¹ : CategoryTheory.Category.{u_8, u_5} D'
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      G : CategoryTheory.Functor D D'
      inst✝ : G.IsEquivalence
      e : CategoryTheory.Iso (L.comp G) L'
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D' H
      α : Quiver.Hom (L'.comp F') F
      eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
      ⊢ F'.IsRightKanExtension α → (G.comp F').IsRightKanExtension (CategoryTheory.C …
    -/
  · exact fun _ => ⟨⟨eq (isUniversalOfIsRightKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_6, u_3} H
      inst✝² : CategoryTheory.Category.{u_9, u_4} D
      inst✝¹ : CategoryTheory.Category.{u_8, u_5} D'
      L : CategoryTheory.Functor C D
      L' : CategoryTheory.Functor C D'
      G : CategoryTheory.Functor D D'
      inst✝ : G.IsEquivalence
      e : CategoryTheory.Iso (L.comp G) L'
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D' H
      α : Quiver.Hom (L'.comp F') F
      eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
      ⊢ (G.comp F').IsRightKanExtension (CategoryTheory.CategoryStruct.comp (L.assoc …
    -/
  · exact fun _ => ⟨⟨eq.symm (isUniversalOfIsRightKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/


/-- The functor `LeftExtension L F ⥤ LeftExtension (G ⋙ L) (G ⋙ F)`
obtained by precomposition. -/
@[simps!]
def LeftExtension.precomp : LeftExtension L F ⥤ LeftExtension (G ⋙ L) (G ⋙ F) :=
  StructuredArrow.map₂ (F := 𝟭 _) (G := (whiskeringLeft C' C H).obj G) (𝟙 _) (𝟙 _)


/-- The functor `RightExtension L F ⥤ RightExtension (G ⋙ L) (G ⋙ F)`
obtained by precomposition. -/
@[simps!]
def RightExtension.precomp : RightExtension L F ⥤ RightExtension (G ⋙ L) (G ⋙ F) :=
  CostructuredArrow.map₂ (F := 𝟭 _) (G := (whiskeringLeft C' C H).obj G) (𝟙 _) (𝟙 _)


noncomputable instance : IsEquivalence (LeftExtension.precomp L F G) := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁵ : CategoryTheory.Category.{u_8, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_9, u_2} C'
    inst✝³ : CategoryTheory.Category.{u_6, u_3} H
    inst✝² : CategoryTheory.Category.{u_7, u_4} D
    inst✝¹ : CategoryTheory.Category.{?u.135965, u_5} D'
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D H
    G : CategoryTheory.Functor C' C
    inst✝ : G.IsEquivalence
    ⊢ (CategoryTheory.Functor.LeftExtension.precomp L F G).IsEquivalence
  -/
  apply StructuredArrow.isEquivalenceMap₂
  /-
    🎉 no goals
  -/


noncomputable instance : IsEquivalence (RightExtension.precomp L F G) := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁵ : CategoryTheory.Category.{u_8, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_9, u_2} C'
    inst✝³ : CategoryTheory.Category.{u_6, u_3} H
    inst✝² : CategoryTheory.Category.{u_7, u_4} D
    inst✝¹ : CategoryTheory.Category.{?u.138229, u_5} D'
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D H
    G : CategoryTheory.Functor C' C
    inst✝ : G.IsEquivalence
    ⊢ (CategoryTheory.Functor.RightExtension.precomp L F G).IsEquivalence
  -/
  apply CostructuredArrow.isEquivalenceMap₂
  /-
    🎉 no goals
  -/


/-- If `G` is an equivalence, then a left extension of `F` along `L` is universal iff
the corresponding left extension of `G ⋙ F` along `G ⋙ L` is. -/
noncomputable def LeftExtension.isUniversalPrecompEquiv (e : LeftExtension L F) :
    e.IsUniversal ≃ ((LeftExtension.precomp L F G).obj e).IsUniversal := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁵ : CategoryTheory.Category.{?u.140493, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.140497, u_2} C'
    inst✝³ : CategoryTheory.Category.{?u.140501, u_3} H
    inst✝² : CategoryTheory.Category.{?u.140505, u_4} D
    inst✝¹ : CategoryTheory.Category.{?u.140509, u_5} D'
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D H
    G : CategoryTheory.Functor C' C
    inst✝ : G.IsEquivalence
    e : L.LeftExtension F
    ⊢ Equiv (CategoryTheory.StructuredArrow.IsUniversal e) (CategoryTheory.Structu …
  -/
  apply IsInitial.isInitialIffObj (LeftExtension.precomp L F G)
  /-
    🎉 no goals
  -/


/-- If `G` is an equivalence, then a right extension of `F` along `L` is universal iff
the corresponding left extension of `G ⋙ F` along `G ⋙ L` is. -/
noncomputable def RightExtension.isUniversalPrecompEquiv (e : RightExtension L F) :
    e.IsUniversal ≃ ((RightExtension.precomp L F G).obj e).IsUniversal := by
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    D' : Type u_5
    inst✝⁵ : CategoryTheory.Category.{?u.142426, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.142430, u_2} C'
    inst✝³ : CategoryTheory.Category.{?u.142434, u_3} H
    inst✝² : CategoryTheory.Category.{?u.142438, u_4} D
    inst✝¹ : CategoryTheory.Category.{?u.142442, u_5} D'
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D H
    G : CategoryTheory.Functor C' C
    inst✝ : G.IsEquivalence
    e : L.RightExtension F
    ⊢ Equiv (CategoryTheory.CostructuredArrow.IsUniversal e) (CategoryTheory.Costr …
  -/
  apply IsTerminal.isTerminalIffObj (RightExtension.precomp L F G)
  /-
    🎉 no goals
  -/


lemma isLeftKanExtension_iff_precomp (α : F ⟶ L ⋙ F') :
    F'.IsLeftKanExtension α ↔ F'.IsLeftKanExtension
      (whiskerLeft G α ≫ (Functor.associator _ _ _).inv) := by
  let eq : (LeftExtension.mk _ α).IsUniversal ≃ (LeftExtension.mk _
      (whiskerLeft G α ≫ (Functor.associator _ _ _).inv)).IsUniversal :=
    (LeftExtension.isUniversalPrecompEquiv L F G _).trans
    (IsInitial.equivOfIso (StructuredArrow.isoMk (Iso.refl _)))
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_9, u_2} C'
    inst✝² : CategoryTheory.Category.{u_6, u_3} H
    inst✝¹ : CategoryTheory.Category.{u_8, u_4} D
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D H
    G : CategoryTheory.Functor C' C
    inst✝ : G.IsEquivalence
    α : Quiver.Hom F (L.comp F')
    eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
    ⊢ Iff (F'.IsLeftKanExtension α) (F'.IsLeftKanExtension (CategoryTheory.Categor …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_9, u_2} C'
      inst✝² : CategoryTheory.Category.{u_6, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_8, u_4} D
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D H
      G : CategoryTheory.Functor C' C
      inst✝ : G.IsEquivalence
      α : Quiver.Hom F (L.comp F')
      eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
      ⊢ F'.IsLeftKanExtension α → F'.IsLeftKanExtension (CategoryTheory.CategoryStru …
    -/
  · exact fun _ => ⟨⟨eq (isUniversalOfIsLeftKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_9, u_2} C'
      inst✝² : CategoryTheory.Category.{u_6, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_8, u_4} D
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D H
      G : CategoryTheory.Functor C' C
      inst✝ : G.IsEquivalence
      α : Quiver.Hom F (L.comp F')
      eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
      ⊢ F'.IsLeftKanExtension (CategoryTheory.CategoryStruct.comp (CategoryTheory.wh …
    -/
  · exact fun _ => ⟨⟨eq.symm (isUniversalOfIsLeftKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/


lemma isRightKanExtension_iff_precomp (α : L ⋙ F' ⟶ F) :
    F'.IsRightKanExtension α ↔
      F'.IsRightKanExtension ((Functor.associator _ _ _).hom ≫ whiskerLeft G α) := by
  let eq : (RightExtension.mk _ α).IsUniversal ≃ (RightExtension.mk _
      ((Functor.associator _ _ _).hom ≫ whiskerLeft G α)).IsUniversal :=
    (RightExtension.isUniversalPrecompEquiv L F G _).trans
    (IsTerminal.equivOfIso (CostructuredArrow.isoMk (Iso.refl _)))
  /-
    C : Type u_1
    C' : Type u_2
    H : Type u_3
    D : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    inst✝³ : CategoryTheory.Category.{u_9, u_2} C'
    inst✝² : CategoryTheory.Category.{u_6, u_3} H
    inst✝¹ : CategoryTheory.Category.{u_8, u_4} D
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    F' : CategoryTheory.Functor D H
    G : CategoryTheory.Functor C' C
    inst✝ : G.IsEquivalence
    α : Quiver.Hom (L.comp F') F
    eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
    ⊢ Iff (F'.IsRightKanExtension α) (F'.IsRightKanExtension (CategoryTheory.Categ …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_9, u_2} C'
      inst✝² : CategoryTheory.Category.{u_6, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_8, u_4} D
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D H
      G : CategoryTheory.Functor C' C
      inst✝ : G.IsEquivalence
      α : Quiver.Hom (L.comp F') F
      eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
      ⊢ F'.IsRightKanExtension α → F'.IsRightKanExtension (CategoryTheory.CategorySt …
    -/
  · exact fun _ => ⟨⟨eq (isUniversalOfIsRightKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      inst✝³ : CategoryTheory.Category.{u_9, u_2} C'
      inst✝² : CategoryTheory.Category.{u_6, u_3} H
      inst✝¹ : CategoryTheory.Category.{u_8, u_4} D
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      F' : CategoryTheory.Functor D H
      G : CategoryTheory.Functor C' C
      inst✝ : G.IsEquivalence
      α : Quiver.Hom (L.comp F') F
      eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
      ⊢ F'.IsRightKanExtension (CategoryTheory.CategoryStruct.comp (G.associator L F …
    -/
  · exact fun _ => ⟨⟨eq.symm (isUniversalOfIsRightKanExtension _ _)⟩⟩
    /-
      🎉 no goals
    -/


/-- The equivalence `RightExtension L F ≌ RightExtension L' F` induced by
a natural isomorphism `L ≅ L'`. -/
def rightExtensionEquivalenceOfIso₁ : RightExtension L F ≌ RightExtension L' F :=
  CostructuredArrow.mapNatIso ((whiskeringLeft C D H).mapIso iso₁)


include iso₁ in
lemma hasRightExtension_iff_of_iso₁ : HasRightKanExtension L F ↔ HasRightKanExtension L' F :=
  (rightExtensionEquivalenceOfIso₁ iso₁ F).hasTerminal_iff


/-- The equivalence `LeftExtension L F ≌ LeftExtension L' F` induced by
a natural isomorphism `L ≅ L'`. -/
def leftExtensionEquivalenceOfIso₁ : LeftExtension L F ≌ LeftExtension L' F :=
  StructuredArrow.mapNatIso ((whiskeringLeft C D H).mapIso iso₁)


include iso₁ in
lemma hasLeftExtension_iff_of_iso₁ : HasLeftKanExtension L F ↔ HasLeftKanExtension L' F :=
  (leftExtensionEquivalenceOfIso₁ iso₁ F).hasInitial_iff


/-- The equivalence `RightExtension L F ≌ RightExtension L F'` induced by
a natural isomorphism `F ≅ F'`. -/
def rightExtensionEquivalenceOfIso₂ : RightExtension L F ≌ RightExtension L F' :=
  CostructuredArrow.mapIso iso₂


include iso₂ in
lemma hasRightExtension_iff_of_iso₂ : HasRightKanExtension L F ↔ HasRightKanExtension L F' :=
  (rightExtensionEquivalenceOfIso₂ L iso₂).hasTerminal_iff


/-- The equivalence `LeftExtension L F ≌ LeftExtension L F'` induced by
a natural isomorphism `F ≅ F'`. -/
def leftExtensionEquivalenceOfIso₂ : LeftExtension L F ≌ LeftExtension L F' :=
  StructuredArrow.mapIso iso₂


include iso₂ in
lemma hasLeftExtension_iff_of_iso₂ : HasLeftKanExtension L F ↔ HasLeftKanExtension L F' :=
  (leftExtensionEquivalenceOfIso₂ L iso₂).hasInitial_iff


/-- When two left extensions `α₁ : LeftExtension L F₁` and `α₂ : LeftExtension L F₂`
are essentially the same via an isomorphism of functors `F₁ ≅ F₂`,
then `α₁` is universal iff `α₂` is. -/
noncomputable def LeftExtension.isUniversalEquivOfIso₂
    (α₁ : LeftExtension L F₁) (α₂ : LeftExtension L F₂) (e : F₁ ≅ F₂)
    (e' : α₁.right ≅ α₂.right)
    (h : α₁.hom ≫ whiskerLeft L e'.hom = e.hom ≫ α₂.hom) :
    α₁.IsUniversal ≃ α₂.IsUniversal :=
  (IsInitial.isInitialIffObj (leftExtensionEquivalenceOfIso₂ L e).functor α₁).trans
    (IsInitial.equivOfIso (StructuredArrow.isoMk e'
          /-
            C : Type u_1
            C' : Type u_2
            H : Type u_3
            D : Type u_4
            D' : Type u_5
            inst✝⁴ : CategoryTheory.Category.{?u.169905, u_1} C
            inst✝³ : CategoryTheory.Category.{?u.169909, u_2} C'
            inst✝² : CategoryTheory.Category.{?u.169913, u_3} H
            inst✝¹ : CategoryTheory.Category.{?u.169917, u_4} D
            inst✝ : CategoryTheory.Category.{?u.169921, u_5} D'
            L : CategoryTheory.Functor C D
            F₁ F₂ : CategoryTheory.Functor C H
            α₁ : L.LeftExtension F₁
            α₂ : L.LeftExtension F₂
            e : CategoryTheory.Iso F₁ F₂
            e' : CategoryTheory.Iso α₁.right α₂.right
            h : Eq (CategoryTheory.CategoryStruct.comp α₁.hom (CategoryTheory.whiskerLeft  …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.leftExtensionEquivalenceOfIso₂ e) …
          -/
      (by simp [leftExtensionEquivalenceOfIso₂, h])))
          /-
            🎉 no goals
          -/


lemma isLeftKanExtension_iff_of_iso₂ {F₁' F₂' : D ⥤ H} (α₁ : F₁ ⟶ L ⋙ F₁') (α₂ : F₂ ⟶ L ⋙ F₂')
    (e : F₁ ≅ F₂) (e' : F₁' ≅ F₂') (h : α₁ ≫ whiskerLeft L e'.hom = e.hom ≫ α₂) :
    F₁'.IsLeftKanExtension α₁ ↔ F₂'.IsLeftKanExtension α₂ := by
  let eq := LeftExtension.isUniversalEquivOfIso₂ (LeftExtension.mk _ α₁)
    (LeftExtension.mk _ α₂) e e' h
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
    inst✝ : CategoryTheory.Category.{u_6, u_4} D
    L : CategoryTheory.Functor C D
    F₁ F₂ : CategoryTheory.Functor C H
    F₁' F₂' : CategoryTheory.Functor D H
    α₁ : Quiver.Hom F₁ (L.comp F₁')
    α₂ : Quiver.Hom F₂ (L.comp F₂')
    e : CategoryTheory.Iso F₁ F₂
    e' : CategoryTheory.Iso F₁' F₂'
    h : Eq (CategoryTheory.CategoryStruct.comp α₁ (CategoryTheory.whiskerLeft L e' …
    eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
    ⊢ Iff (F₁'.IsLeftKanExtension α₁) (F₂'.IsLeftKanExtension α₂)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      L : CategoryTheory.Functor C D
      F₁ F₂ : CategoryTheory.Functor C H
      F₁' F₂' : CategoryTheory.Functor D H
      α₁ : Quiver.Hom F₁ (L.comp F₁')
      α₂ : Quiver.Hom F₂ (L.comp F₂')
      e : CategoryTheory.Iso F₁ F₂
      e' : CategoryTheory.Iso F₁' F₂'
      h : Eq (CategoryTheory.CategoryStruct.comp α₁ (CategoryTheory.whiskerLeft L e' …
      eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
      ⊢ F₁'.IsLeftKanExtension α₁ → F₂'.IsLeftKanExtension α₂
    -/
  · exact fun _ => ⟨⟨eq.1 (isUniversalOfIsLeftKanExtension F₁' α₁)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      L : CategoryTheory.Functor C D
      F₁ F₂ : CategoryTheory.Functor C H
      F₁' F₂' : CategoryTheory.Functor D H
      α₁ : Quiver.Hom F₁ (L.comp F₁')
      α₂ : Quiver.Hom F₂ (L.comp F₂')
      e : CategoryTheory.Iso F₁ F₂
      e' : CategoryTheory.Iso F₁' F₂'
      h : Eq (CategoryTheory.CategoryStruct.comp α₁ (CategoryTheory.whiskerLeft L e' …
      eq : Equiv (CategoryTheory.StructuredArrow.IsUniversal (CategoryTheory.Functor …
      ⊢ F₂'.IsLeftKanExtension α₂ → F₁'.IsLeftKanExtension α₁
    -/
  · exact fun _ => ⟨⟨eq.2 (isUniversalOfIsLeftKanExtension F₂' α₂)⟩⟩
    /-
      🎉 no goals
    -/


/-- When two right extensions `α₁ : RightExtension L F₁` and `α₂ : RightExtension L F₂`
are essentially the same via an isomorphism of functors `F₁ ≅ F₂`,
then `α₁` is universal iff `α₂` is. -/
noncomputable def RightExtension.isUniversalEquivOfIso₂
    (α₁ : RightExtension L F₁) (α₂ : RightExtension L F₂) (e : F₁ ≅ F₂)
    (e' : α₁.left ≅ α₂.left)
    (h : whiskerLeft L e'.hom ≫ α₂.hom = α₁.hom ≫ e.hom) :
    α₁.IsUniversal ≃ α₂.IsUniversal :=
  (IsTerminal.isTerminalIffObj (rightExtensionEquivalenceOfIso₂ L e).functor α₁).trans
    (IsTerminal.equivOfIso (CostructuredArrow.isoMk e'
          /-
            C : Type u_1
            C' : Type u_2
            H : Type u_3
            D : Type u_4
            D' : Type u_5
            inst✝⁴ : CategoryTheory.Category.{?u.177276, u_1} C
            inst✝³ : CategoryTheory.Category.{?u.177280, u_2} C'
            inst✝² : CategoryTheory.Category.{?u.177284, u_3} H
            inst✝¹ : CategoryTheory.Category.{?u.177288, u_4} D
            inst✝ : CategoryTheory.Category.{?u.177292, u_5} D'
            L : CategoryTheory.Functor C D
            F₁ F₂ : CategoryTheory.Functor C H
            α₁ : L.RightExtension F₁
            α₂ : L.RightExtension F₂
            e : CategoryTheory.Iso F₁ F₂
            e' : CategoryTheory.Iso α₁.left α₂.left
            h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e'.ho …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft C D  …
          -/
      (by simp [rightExtensionEquivalenceOfIso₂, h])))
          /-
            🎉 no goals
          -/


lemma isRightKanExtension_iff_of_iso₂ {F₁' F₂' : D ⥤ H} (α₁ : L ⋙ F₁' ⟶ F₁) (α₂ : L ⋙ F₂' ⟶ F₂)
    (e : F₁ ≅ F₂) (e' : F₁' ≅ F₂') (h : whiskerLeft L e'.hom ≫ α₂ = α₁ ≫ e.hom) :
    F₁'.IsRightKanExtension α₁ ↔ F₂'.IsRightKanExtension α₂ := by
  let eq := RightExtension.isUniversalEquivOfIso₂ (RightExtension.mk _ α₁)
    (RightExtension.mk _ α₂) e e' h
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝² : CategoryTheory.Category.{u_8, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
    inst✝ : CategoryTheory.Category.{u_6, u_4} D
    L : CategoryTheory.Functor C D
    F₁ F₂ : CategoryTheory.Functor C H
    F₁' F₂' : CategoryTheory.Functor D H
    α₁ : Quiver.Hom (L.comp F₁') F₁
    α₂ : Quiver.Hom (L.comp F₂') F₂
    e : CategoryTheory.Iso F₁ F₂
    e' : CategoryTheory.Iso F₁' F₂'
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e'.ho …
    eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
    ⊢ Iff (F₁'.IsRightKanExtension α₁) (F₂'.IsRightKanExtension α₂)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      L : CategoryTheory.Functor C D
      F₁ F₂ : CategoryTheory.Functor C H
      F₁' F₂' : CategoryTheory.Functor D H
      α₁ : Quiver.Hom (L.comp F₁') F₁
      α₂ : Quiver.Hom (L.comp F₂') F₂
      e : CategoryTheory.Iso F₁ F₂
      e' : CategoryTheory.Iso F₁' F₂'
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e'.ho …
      eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
      ⊢ F₁'.IsRightKanExtension α₁ → F₂'.IsRightKanExtension α₂
    -/
  · exact fun _ => ⟨⟨eq.1 (isUniversalOfIsRightKanExtension F₁' α₁)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      H : Type u_3
      D : Type u_4
      inst✝² : CategoryTheory.Category.{u_8, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_7, u_3} H
      inst✝ : CategoryTheory.Category.{u_6, u_4} D
      L : CategoryTheory.Functor C D
      F₁ F₂ : CategoryTheory.Functor C H
      F₁' F₂' : CategoryTheory.Functor D H
      α₁ : Quiver.Hom (L.comp F₁') F₁
      α₂ : Quiver.Hom (L.comp F₂') F₂
      e : CategoryTheory.Iso F₁ F₂
      e' : CategoryTheory.Iso F₁' F₂'
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft L e'.ho …
      eq : Equiv (CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.Funct …
      ⊢ F₂'.IsRightKanExtension α₂ → F₁'.IsRightKanExtension α₁
    -/
  · exact fun _ => ⟨⟨eq.2 (isUniversalOfIsRightKanExtension F₂' α₂)⟩⟩
    /-
      🎉 no goals
    -/


/-- Construct a cocone for a left Kan extension `F' : D ⥤ H` of `F : C ⥤ H` along a functor
`L : C ⥤ D` given a cocone for `F`. -/
@[simps]
noncomputable def coconeOfIsLeftKanExtension (c : Cocone F) : Cocone F' where
  pt := c.pt
  ι := F'.descOfIsLeftKanExtension α _ c.ι


/-- If `c` is a colimit cocone for a functor `F : C ⥤ H` and `α : F ⟶ L ⋙ F'` is the unit of any
left Kan extension `F' : D ⥤ H` of `F` along `L : C ⥤ D`, then `coconeOfIsLeftKanExtension α c` is
a colimit cocone, too. -/
@[simps]
def isColimitCoconeOfIsLeftKanExtension {c : Cocone F} (hc : IsColimit c) :
    IsColimit (F'.coconeOfIsLeftKanExtension α c) where
  desc s := hc.desc (Cocone.mk _ (α ≫ whiskerLeft L s.ι))
  fac s := by
    have : F'.descOfIsLeftKanExtension α ((const D).obj c.pt) c.ι ≫
        (Functor.const _).map (hc.desc (Cocone.mk _ (α ≫ whiskerLeft L s.ι))) = s.ι :=
      F'.hom_ext_of_isLeftKanExtension α _ _ (by aesop_cat)
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.186661, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.186665, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.186669, u_3} H
      inst✝² : CategoryTheory.Category.{?u.186673, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.186677, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      inst✝ : F'.IsLeftKanExtension α
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone F'
      this : Eq (CategoryTheory.CategoryStruct.comp (F'.descOfIsLeftKanExtension α ( …
      ⊢ ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanExte …
    -/
    exact congr_app this
    /-
      🎉 no goals
    -/
  uniq s m hm := hc.hom_ext (fun j ↦ by
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.186661, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.186665, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.186669, u_3} H
      inst✝² : CategoryTheory.Category.{?u.186673, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.186677, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      inst✝ : F'.IsLeftKanExtension α
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone F'
      m : Quiver.Hom (F'.coconeOfIsLeftKanExtension α c).pt s.pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanE …
      j : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (CategoryTheory.Catego …
    -/
    have := hm (L.obj j)
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.186661, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.186665, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.186669, u_3} H
      inst✝² : CategoryTheory.Category.{?u.186673, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.186677, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      inst✝ : F'.IsLeftKanExtension α
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone F'
      m : Quiver.Hom (F'.coconeOfIsLeftKanExtension α c).pt s.pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanE …
      j : C
      this : Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanExtension  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (CategoryTheory.Catego …
    -/
    nth_rw 1 [← F'.descOfIsLeftKanExtension_fac_app α ((const D).obj c.pt)]
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.186661, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.186665, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.186669, u_3} H
      inst✝² : CategoryTheory.Category.{?u.186673, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.186677, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      inst✝ : F'.IsLeftKanExtension α
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone F'
      m : Quiver.Hom (F'.coconeOfIsLeftKanExtension α c).pt s.pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanE …
      j : C
      this : Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanExtension  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp at this ⊢
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.186661, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.186665, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.186669, u_3} H
      inst✝² : CategoryTheory.Category.{?u.186673, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.186677, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom F (L.comp F')
      inst✝ : F'.IsLeftKanExtension α
      c : CategoryTheory.Limits.Cocone F
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone F'
      m : Quiver.Hom (F'.coconeOfIsLeftKanExtension α c).pt s.pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((F'.coconeOfIsLeftKanE …
      j : C
      this : Eq (CategoryTheory.CategoryStruct.comp ((F'.descOfIsLeftKanExtension α  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, this, IsColimit.fac, NatTrans.comp_app, whiskerLeft_app])
    /-
      🎉 no goals
    -/


/-- If `F' : D ⥤ H` is a left Kan extension of `F : C ⥤ H` along `L : C ⥤ D`, the colimit over `F'`
is isomorphic to the colimit over `F`. -/
noncomputable def colimitIsoOfIsLeftKanExtension : colimit F' ≅ colimit F :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit F')
    (F'.isColimitCoconeOfIsLeftKanExtension α (colimit.isColimit F))


@[reassoc (attr := simp)]
lemma ι_colimitIsoOfIsLeftKanExtension_hom (i : C) :
    α.app i ≫ colimit.ι F' (L.obj i) ≫ (F'.colimitIsoOfIsLeftKanExtension α).hom =
      colimit.ι F i := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_3} H
    inst✝³ : CategoryTheory.Category.{u_8, u_4} D
    F' : CategoryTheory.Functor D H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom F (L.comp F')
    inst✝² : F'.IsLeftKanExtension α
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasColimit F'
    i : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app i) (CategoryTheory.CategoryStr …
  -/
  simp [colimitIsoOfIsLeftKanExtension]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_colimitIsoOfIsLeftKanExtension_inv (i : C) :
    colimit.ι F i ≫ (F'.colimitIsoOfIsLeftKanExtension α).inv =
    α.app i ≫ colimit.ι F' (L.obj i) := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_3} H
    inst✝³ : CategoryTheory.Category.{u_8, u_4} D
    F' : CategoryTheory.Functor D H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom F (L.comp F')
    inst✝² : F'.IsLeftKanExtension α
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    inst✝ : CategoryTheory.Limits.HasColimit F'
    i : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F i) …
  -/
  rw [Iso.comp_inv_eq, assoc, ι_colimitIsoOfIsLeftKanExtension_hom]
  /-
    🎉 no goals
  -/


/-- Construct a cone for a right Kan extension `F' : D ⥤ H` of `F : C ⥤ H` along a functor
`L : C ⥤ D` given a cone for `F`. -/
@[simps]
noncomputable def coneOfIsRightKanExtension (c : Cone F) : Cone F' where
  pt := c.pt
  π := F'.liftOfIsRightKanExtension α _ c.π


/-- If `c` is a limit cone for a functor `F : C ⥤ H` and `α : L ⋙ F' ⟶ F` is the counit of any
right Kan extension `F' : D ⥤ H` of `F` along `L : C ⥤ D`, then `coneOfIsRightKanExtension α c` is
a limit cone, too. -/
@[simps]
def isLimitConeOfIsRightKanExtension {c : Cone F} (hc : IsLimit c) :
    IsLimit (F'.coneOfIsRightKanExtension α c) where
  lift s := hc.lift (Cone.mk _ (whiskerLeft L s.π ≫ α))
  fac s := by
    have : (Functor.const _).map (hc.lift (Cone.mk _ (whiskerLeft L s.π ≫ α))) ≫
        F'.liftOfIsRightKanExtension α ((const D).obj c.pt) c.π = s.π :=
      F'.hom_ext_of_isRightKanExtension α _ _ (by aesop_cat)
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.207755, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.207759, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.207763, u_3} H
      inst✝² : CategoryTheory.Category.{?u.207767, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.207771, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      inst✝ : F'.IsRightKanExtension α
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F'
      this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.const D …
      ⊢ ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp ((fun s => hc.lift { pt := …
    -/
    exact congr_app this
    /-
      🎉 no goals
    -/
  uniq s m hm := hc.hom_ext (fun j ↦ by
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.207755, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.207759, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.207763, u_3} H
      inst✝² : CategoryTheory.Category.{?u.207767, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.207771, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      inst✝ : F'.IsRightKanExtension α
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F'
      m : Quiver.Hom s.pt (F'.coneOfIsRightKanExtension α c).pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp m ((F'.coneOfIsRightKan …
      j : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (CategoryTheory.Catego …
    -/
    have := hm (L.obj j)
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.207755, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.207759, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.207763, u_3} H
      inst✝² : CategoryTheory.Category.{?u.207767, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.207771, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      inst✝ : F'.IsRightKanExtension α
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F'
      m : Quiver.Hom s.pt (F'.coneOfIsRightKanExtension α c).pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp m ((F'.coneOfIsRightKan …
      j : C
      this : Eq (CategoryTheory.CategoryStruct.comp m ((F'.coneOfIsRightKanExtension …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (CategoryTheory.Catego …
    -/
    nth_rw 1 [← F'.liftOfIsRightKanExtension_fac_app α ((const D).obj c.pt)]
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.207755, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.207759, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.207763, u_3} H
      inst✝² : CategoryTheory.Category.{?u.207767, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.207771, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      inst✝ : F'.IsRightKanExtension α
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F'
      m : Quiver.Hom s.pt (F'.coneOfIsRightKanExtension α c).pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp m ((F'.coneOfIsRightKan …
      j : C
      this : Eq (CategoryTheory.CategoryStruct.comp m ((F'.coneOfIsRightKanExtension …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
    -/
    dsimp at this ⊢
    /-
      C : Type u_1
      C' : Type u_2
      H : Type u_3
      D : Type u_4
      D' : Type u_5
      inst✝⁵ : CategoryTheory.Category.{?u.207755, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.207759, u_2} C'
      inst✝³ : CategoryTheory.Category.{?u.207763, u_3} H
      inst✝² : CategoryTheory.Category.{?u.207767, u_4} D
      inst✝¹ : CategoryTheory.Category.{?u.207771, u_5} D'
      F' : CategoryTheory.Functor D H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      α : Quiver.Hom (L.comp F') F
      inst✝ : F'.IsRightKanExtension α
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F'
      m : Quiver.Hom s.pt (F'.coneOfIsRightKanExtension α c).pt
      hm : ∀ (j : D), Eq (CategoryTheory.CategoryStruct.comp m ((F'.coneOfIsRightKan …
      j : C
      this : Eq (CategoryTheory.CategoryStruct.comp m ((F'.liftOfIsRightKanExtension …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
    -/
    rw [← assoc, this, IsLimit.fac, NatTrans.comp_app, whiskerLeft_app])
    /-
      🎉 no goals
    -/


/-- If `F' : D ⥤ H` is a right Kan extension of `F : C ⥤ H` along `L : C ⥤ D`, the limit over `F'`
is isomorphic to the limit over `F`. -/
noncomputable def limitIsoOfIsRightKanExtension : limit F' ≅ limit F :=
  IsLimit.conePointUniqueUpToIso (limit.isLimit F')
    (F'.isLimitConeOfIsRightKanExtension α (limit.isLimit F))


@[reassoc (attr := simp)]
lemma limitIsoOfIsRightKanExtension_inv_π (i : C) :
    (F'.limitIsoOfIsRightKanExtension α).inv ≫ limit.π F' (L.obj i) ≫ α.app i = limit.π F i := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_3} H
    inst✝³ : CategoryTheory.Category.{u_8, u_4} D
    F' : CategoryTheory.Functor D H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom (L.comp F') F
    inst✝² : F'.IsRightKanExtension α
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    inst✝ : CategoryTheory.Limits.HasLimit F'
    i : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.limitIsoOfIsRightKanExtension α). …
  -/
  simp [limitIsoOfIsRightKanExtension]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma limitIsoOfIsRightKanExtension_hom_π (i : C) :
    (F'.limitIsoOfIsRightKanExtension α).hom ≫ limit.π F i = limit.π F' (L.obj i) ≫ α.app i := by
  /-
    C : Type u_1
    H : Type u_3
    D : Type u_4
    inst✝⁵ : CategoryTheory.Category.{u_8, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_6, u_3} H
    inst✝³ : CategoryTheory.Category.{u_7, u_4} D
    F' : CategoryTheory.Functor D H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    α : Quiver.Hom (L.comp F') F
    inst✝² : F'.IsRightKanExtension α
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    inst✝ : CategoryTheory.Limits.HasLimit F'
    i : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.limitIsoOfIsRightKanExtension α). …
  -/
  rw [← Iso.eq_inv_comp, limitIsoOfIsRightKanExtension_inv_π]
  /-
    🎉 no goals
  -/


