/-- The condition that a functor `F` has a pointwise left Kan extension along `L` at `Y`.
It means that the functor `CostructuredArrow.proj L Y ⋙ F : CostructuredArrow L Y ⥤ H`
has a colimit. -/
abbrev HasPointwiseLeftKanExtensionAt (Y : D) :=
  HasColimit (CostructuredArrow.proj L Y ⋙ F)


/-- The condition that a functor `F` has a pointwise left Kan extension along `L`: it means
that it has a pointwise left Kan extension at any object. -/
abbrev HasPointwiseLeftKanExtension := ∀ (Y : D), HasPointwiseLeftKanExtensionAt L F Y


/-- The condition that a functor `F` has a pointwise right Kan extension along `L` at `Y`.
It means that the functor `StructuredArrow.proj Y L ⋙ F : StructuredArrow Y L ⥤ H`
has a limit. -/
abbrev HasPointwiseRightKanExtensionAt (Y : D) :=
  HasLimit (StructuredArrow.proj Y L ⋙ F)


/-- The condition that a functor `F` has a pointwise right Kan extension along `L`: it means
that it has a pointwise right Kan extension at any object. -/
abbrev HasPointwiseRightKanExtension := ∀ (Y : D), HasPointwiseRightKanExtensionAt L F Y


/-- The cocone for `CostructuredArrow.proj L Y ⋙ F` attached to `E : LeftExtension L F`.
The point of this cocone is `E.right.obj Y` -/
@[simps]
def coconeAt (Y : D) : Cocone (CostructuredArrow.proj L Y ⋙ F) where
  pt := E.right.obj Y
  ι :=
    { app := fun g => E.hom.app g.left ≫ E.right.map g.hom
      naturality := fun g₁ g₂ φ => by
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.2981, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2985, u_2} D
          inst✝ : CategoryTheory.Category.{?u.2989, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E : L.LeftExtension F
          Y : D
          g₁ g₂ : CategoryTheory.CostructuredArrow L Y
          φ : Quiver.Hom g₁ g₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.p …
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.2981, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2985, u_2} D
          inst✝ : CategoryTheory.Category.{?u.2989, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E : L.LeftExtension F
          Y : D
          g₁ g₂ : CategoryTheory.CostructuredArrow L Y
          φ : Quiver.Hom g₁ g₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.left) (CategoryTheory.Catego …
        -/
        rw [← CostructuredArrow.w φ]
        simp only [assoc, NatTrans.naturality_assoc, Functor.comp_map,
          Functor.map_comp, comp_id] }


variable (L F) in
/-- The cocones for `CostructuredArrow.proj L Y ⋙ F`, as a functor from `LeftExtension L F`. -/
@[simps]
def coconeAtFunctor (Y : D) :
    LeftExtension L F ⥤ Cocone (CostructuredArrow.proj L Y ⋙ F) where
  obj E := E.coconeAt Y
  map {E E'} φ := CoconeMorphism.mk (φ.right.app Y) (fun G => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6339, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6343, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6347, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      E✝ : L.LeftExtension F
      Y : D
      E E' : L.LeftExtension F
      φ : Quiver.Hom E E'
      G : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun E => E.coconeAt Y) E).ι.app G) …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6339, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6343, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6347, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      E✝ : L.LeftExtension F
      Y : D
      E E' : L.LeftExtension F
      φ : Quiver.Hom E E'
      G : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← StructuredArrow.w φ]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6339, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6343, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6347, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      E✝ : L.LeftExtension F
      Y : D
      E E' : L.LeftExtension F
      φ : Quiver.Hom E E'
      G : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp)
    /-
      🎉 no goals
    -/


/-- A left extension `E : LeftExtension L F` is a pointwise left Kan extension at `Y` when
`E.coconeAt Y` is a colimit cocone. -/
def IsPointwiseLeftKanExtensionAt (Y : D) := IsColimit (E.coconeAt Y)


variable {E} in
lemma IsPointwiseLeftKanExtensionAt.hasPointwiseLeftKanExtensionAt
    {Y : D} (h : E.IsPointwiseLeftKanExtensionAt Y) :
    HasPointwiseLeftKanExtensionAt L F Y := ⟨_, h⟩


lemma IsPointwiseLeftKanExtensionAt.isIso_hom_app
    {X : C} (h : E.IsPointwiseLeftKanExtensionAt (L.obj X)) [L.Full] [L.Faithful] :
    IsIso (E.hom.app X) := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Category.{u_6, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    X : C
    h : E.IsPointwiseLeftKanExtensionAt (L.obj X)
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    ⊢ CategoryTheory.IsIso (E.hom.app X)
  -/
  simpa using h.isIso_ι_app_of_isTerminal _ CostructuredArrow.mkIdTerminal
  /-
    🎉 no goals
  -/


/-- A pointwise left Kan extension of `F` along `L` applied to an object `Y` is isomorphic to
`colimit (CostructuredArrow.proj L Y ⋙ F)`. -/
noncomputable def isoColimit :
    E.right.obj Y ≅ colimit (CostructuredArrow.proj L Y ⋙ F) :=
  h.coconePointUniqueUpToIso (colimit.isColimit _)


@[reassoc (attr := simp)]
lemma ι_isoColimit_inv (g : CostructuredArrow L Y) :
    colimit.ι _ g ≫ h.isoColimit.inv = E.hom.app g.left ≫ E.right.map g.hom :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ _ _


@[reassoc (attr := simp)]
lemma ι_isoColimit_hom (g : CostructuredArrow L Y) :
    E.hom.app g.left ≫ E.right.map g.hom ≫ h.isoColimit.hom =
      colimit.ι (CostructuredArrow.proj L Y ⋙ F) g := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    Y : D
    h : E.IsPointwiseLeftKanExtensionAt Y
    inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.CostructuredArrow.pr …
    g : CategoryTheory.CostructuredArrow L Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (E.hom.app g.left) (CategoryTheory.Ca …
  -/
  simpa using h.comp_coconePointUniqueUpToIso_hom (colimit.isColimit _) g
  /-
    🎉 no goals
  -/


/-- A left extension `E : LeftExtension L F` is a pointwise left Kan extension when
it is a pointwise left Kan extension at any object. -/
abbrev IsPointwiseLeftKanExtension := ∀ (Y : D), E.IsPointwiseLeftKanExtensionAt Y


/-- If two left extensions `E` and `E'` are isomorphic, `E` is a pointwise
left Kan extension at `Y` iff `E'` is. -/
def isPointwiseLeftKanExtensionAtEquivOfIso (e : E ≅ E') (Y : D) :
    E.IsPointwiseLeftKanExtensionAt Y ≃ E'.IsPointwiseLeftKanExtensionAt Y :=
  IsColimit.equivIsoColimit ((coconeAtFunctor L F Y).mapIso e)


/-- If two left extensions `E` and `E'` are isomorphic, `E` is a pointwise
left Kan extension iff `E'` is. -/
def isPointwiseLeftKanExtensionEquivOfIso (e : E ≅ E') :
    E.IsPointwiseLeftKanExtension ≃ E'.IsPointwiseLeftKanExtension where
  toFun h := fun Y => (isPointwiseLeftKanExtensionAtEquivOfIso e Y) (h Y)
  invFun h := fun Y => (isPointwiseLeftKanExtensionAtEquivOfIso e Y).symm (h Y)
                   /-
                     C : Type u_1
                     D : Type u_2
                     H : Type u_3
                     inst✝² : CategoryTheory.Category.{?u.37208, u_1} C
                     inst✝¹ : CategoryTheory.Category.{?u.37212, u_2} D
                     inst✝ : CategoryTheory.Category.{?u.37216, u_3} H
                     L : CategoryTheory.Functor C D
                     F : CategoryTheory.Functor C H
                     E E' : L.LeftExtension F
                     e : CategoryTheory.Iso E E'
                     h : E.IsPointwiseLeftKanExtension
                     ⊢ Eq ((fun h Y => (CategoryTheory.Functor.LeftExtension.isPointwiseLeftKanExte …
                   -/
  left_inv h := by aesop
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u_1
                      D : Type u_2
                      H : Type u_3
                      inst✝² : CategoryTheory.Category.{?u.37208, u_1} C
                      inst✝¹ : CategoryTheory.Category.{?u.37212, u_2} D
                      inst✝ : CategoryTheory.Category.{?u.37216, u_3} H
                      L : CategoryTheory.Functor C D
                      F : CategoryTheory.Functor C H
                      E E' : L.LeftExtension F
                      e : CategoryTheory.Iso E E'
                      h : E'.IsPointwiseLeftKanExtension
                      ⊢ Eq ((fun h Y => (CategoryTheory.Functor.LeftExtension.isPointwiseLeftKanExte …
                    -/
  right_inv h := by aesop
                    /-
                      🎉 no goals
                    -/


lemma IsPointwiseLeftKanExtension.hasPointwiseLeftKanExtension :
    HasPointwiseLeftKanExtension L F :=
  fun Y => (h Y).hasPointwiseLeftKanExtensionAt


/-- The (unique) morphism from a pointwise left Kan extension. -/
def IsPointwiseLeftKanExtension.homFrom (G : LeftExtension L F) : E ⟶ G :=
  StructuredArrow.homMk
    { app := fun Y => (h Y).desc (LeftExtension.coconeAt G Y)
      naturality := fun Y₁ Y₂ φ => (h Y₁).hom_ext (fun X => by
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.40007, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.40011, u_2} D
          inst✝ : CategoryTheory.Category.{?u.40015, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E : L.LeftExtension F
          E' : ?m.40436
          h : E.IsPointwiseLeftKanExtension
          G : L.LeftExtension F
          Y₁ Y₂ : D
          φ : Quiver.Hom Y₁ Y₂
          X : CategoryTheory.CostructuredArrow L Y₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((E.coconeAt Y₁).ι.app X) (CategoryTh …
        -/
        rw [(h Y₁).fac_assoc (coconeAt G Y₁) X]
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.40007, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.40011, u_2} D
          inst✝ : CategoryTheory.Category.{?u.40015, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E : L.LeftExtension F
          E' : ?m.40436
          h : E.IsPointwiseLeftKanExtension
          G : L.LeftExtension F
          Y₁ Y₂ : D
          φ : Quiver.Hom Y₁ Y₂
          X : CategoryTheory.CostructuredArrow L Y₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((E.coconeAt Y₁).ι.app X) (CategoryTh …
        -/
        simpa using (h Y₂).fac (coconeAt G Y₂) ((CostructuredArrow.map φ).obj X)) }
        /-
          🎉 no goals
        -/
    (by
      /-
        C : Type u_1
        D : Type u_2
        H : Type u_3
        inst✝² : CategoryTheory.Category.{?u.40007, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.40011, u_2} D
        inst✝ : CategoryTheory.Category.{?u.40015, u_3} H
        L : CategoryTheory.Functor C D
        F : CategoryTheory.Functor C H
        E : L.LeftExtension F
        E' : ?m.40436
        h : E.IsPointwiseLeftKanExtension
        G : L.LeftExtension F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp E.hom (((CategoryTheory.whiskeringLef …
      -/
      ext X
      /-
        case w.h
        C : Type u_1
        D : Type u_2
        H : Type u_3
        inst✝² : CategoryTheory.Category.{?u.40007, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.40011, u_2} D
        inst✝ : CategoryTheory.Category.{?u.40015, u_3} H
        L : CategoryTheory.Functor C D
        F : CategoryTheory.Functor C H
        E : L.LeftExtension F
        E' : ?m.40436
        h : E.IsPointwiseLeftKanExtension
        G : L.LeftExtension F
        X : C
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp E.hom (((CategoryTheory.whiskeringLe …
      -/
      simpa using (h (L.obj X)).fac (LeftExtension.coconeAt G _) (CostructuredArrow.mk (𝟙 _)))
      /-
        🎉 no goals
      -/


lemma IsPointwiseLeftKanExtension.hom_ext
    {G : LeftExtension L F} {f₁ f₂ : E ⟶ G} : f₁ = f₂ := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    ⊢ Eq f₁ f₂
  -/
  ext Y
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    ⊢ Eq (f₁.right.app Y) (f₂.right.app Y)
  -/
  apply (h Y).hom_ext
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow L Y), Eq (CategoryTheory.CategoryStr …
  -/
  intro X
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    X : CategoryTheory.CostructuredArrow L Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((E.coconeAt Y).ι.app X) (f₁.right.ap …
  -/
  have eq₁ := congr_app (StructuredArrow.w f₁) X.left
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    X : CategoryTheory.CostructuredArrow L Y
    eq₁ : Eq ((CategoryTheory.CategoryStruct.comp E.hom (((CategoryTheory.whiskeri …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((E.coconeAt Y).ι.app X) (f₁.right.ap …
  -/
  have eq₂ := congr_app (StructuredArrow.w f₂) X.left
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    X : CategoryTheory.CostructuredArrow L Y
    eq₁ : Eq ((CategoryTheory.CategoryStruct.comp E.hom (((CategoryTheory.whiskeri …
    eq₂ : Eq ((CategoryTheory.CategoryStruct.comp E.hom (((CategoryTheory.whiskeri …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((E.coconeAt Y).ι.app X) (f₁.right.ap …
  -/
  dsimp at eq₁ eq₂ ⊢
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    X : CategoryTheory.CostructuredArrow L Y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (E.hom.app X.left) (f₁.right.app  …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (E.hom.app X.left) (f₂.right.app  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, NatTrans.naturality]
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.LeftExtension F
    h : E.IsPointwiseLeftKanExtension
    G : L.LeftExtension F
    f₁ f₂ : Quiver.Hom E G
    Y : D
    X : CategoryTheory.CostructuredArrow L Y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (E.hom.app X.left) (f₁.right.app  …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (E.hom.app X.left) (f₂.right.app  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (E.hom.app X.left) (CategoryTheory.Ca …
  -/
  rw [reassoc_of% eq₁, reassoc_of% eq₂]
  /-
    🎉 no goals
  -/


/-- A pointwise left Kan extension is universal, i.e. it is a left Kan extension. -/
def IsPointwiseLeftKanExtension.isUniversal : E.IsUniversal :=
  IsInitial.ofUniqueHom h.homFrom (fun _ _ => h.hom_ext)


lemma IsPointwiseLeftKanExtension.isLeftKanExtension :
    E.right.IsLeftKanExtension E.hom where
  nonempty_isUniversal := ⟨h.isUniversal⟩


lemma IsPointwiseLeftKanExtension.hasLeftKanExtension :
    HasLeftKanExtension L F :=
  have := h.isLeftKanExtension
  HasLeftKanExtension.mk E.right E.hom


lemma IsPointwiseLeftKanExtension.isIso_hom [L.Full] [L.Faithful] :
    IsIso (E.hom) :=
  have := fun X => (h (L.obj X)).isIso_hom_app
  NatIso.isIso_of_isIso_app ..


/-- The cone for `StructuredArrow.proj Y L ⋙ F` attached to `E : RightExtension L F`.
The point of this cone is `E.left.obj Y` -/
@[simps]
def coneAt (Y : D) : Cone (StructuredArrow.proj Y L ⋙ F) where
  pt := E.left.obj Y
  π :=
    { app := fun g ↦ E.left.map g.hom ≫ E.hom.app g.right
      naturality := fun g₁ g₂ φ ↦ by
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.58745, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.58749, u_2} D
          inst✝ : CategoryTheory.Category.{?u.58753, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E E' : L.RightExtension F
          Y : D
          g₁ g₂ : CategoryTheory.StructuredArrow Y L
          φ : Quiver.Hom g₁ g₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.58745, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.58749, u_2} D
          inst✝ : CategoryTheory.Category.{?u.58753, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E E' : L.RightExtension F
          Y : D
          g₁ g₂ : CategoryTheory.StructuredArrow Y L
          φ : Quiver.Hom g₁ g₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (E. …
        -/
        rw [assoc, id_comp, ← StructuredArrow.w φ, Functor.map_comp, assoc]
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.58745, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.58749, u_2} D
          inst✝ : CategoryTheory.Category.{?u.58753, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E E' : L.RightExtension F
          Y : D
          g₁ g₂ : CategoryTheory.StructuredArrow Y L
          φ : Quiver.Hom g₁ g₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (E.left.map g₁.hom) (CategoryTheory.C …
        -/
        congr 1
        /-
          case e_a
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.58745, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.58749, u_2} D
          inst✝ : CategoryTheory.Category.{?u.58753, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E E' : L.RightExtension F
          Y : D
          g₁ g₂ : CategoryTheory.StructuredArrow Y L
          φ : Quiver.Hom g₁ g₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (E.left.map (L.map φ.right)) (E.hom.a …
        -/
        apply E.hom.naturality }
        /-
          🎉 no goals
        -/


variable (L F) in
/-- The cones for `StructuredArrow.proj Y L ⋙ F`, as a functor from `RightExtension L F`. -/
@[simps]
def coneAtFunctor (Y : D) :
    RightExtension L F ⥤ Cone (StructuredArrow.proj Y L ⋙ F) where
  obj E := E.coneAt Y
  map {E E'} φ := ConeMorphism.mk (φ.left.app Y) (fun G ↦ by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝² : CategoryTheory.Category.{?u.63992, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.63996, u_2} D
      inst✝ : CategoryTheory.Category.{?u.64000, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      E✝ E'✝ : L.RightExtension F
      Y : D
      E E' : L.RightExtension F
      φ : Quiver.Hom E E'
      G : CategoryTheory.StructuredArrow Y L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.left.app Y) (((fun E => E.coneAt Y …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝² : CategoryTheory.Category.{?u.63992, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.63996, u_2} D
      inst✝ : CategoryTheory.Category.{?u.64000, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      E✝ E'✝ : L.RightExtension F
      Y : D
      E E' : L.RightExtension F
      φ : Quiver.Hom E E'
      G : CategoryTheory.StructuredArrow Y L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.left.app Y) (CategoryTheory.Catego …
    -/
    rw [← CostructuredArrow.w φ]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝² : CategoryTheory.Category.{?u.63992, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.63996, u_2} D
      inst✝ : CategoryTheory.Category.{?u.64000, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      E✝ E'✝ : L.RightExtension F
      Y : D
      E E' : L.RightExtension F
      φ : Quiver.Hom E E'
      G : CategoryTheory.StructuredArrow Y L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.left.app Y) (CategoryTheory.Catego …
    -/
    simp)
    /-
      🎉 no goals
    -/


/-- A right extension `E : RightExtension L F` is a pointwise right Kan extension at `Y` when
`E.coneAt Y` is a limit cone. -/
def IsPointwiseRightKanExtensionAt (Y : D) := IsLimit (E.coneAt Y)


variable {E} in
lemma IsPointwiseRightKanExtensionAt.hasPointwiseRightKanExtensionAt
    {Y : D} (h : E.IsPointwiseRightKanExtensionAt Y) :
    HasPointwiseRightKanExtensionAt L F Y := ⟨_, h⟩


lemma IsPointwiseRightKanExtensionAt.isIso_hom_app
    {X : C} (h : E.IsPointwiseRightKanExtensionAt (L.obj X)) [L.Full] [L.Faithful] :
    IsIso (E.hom.app X) := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Category.{u_6, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    X : C
    h : E.IsPointwiseRightKanExtensionAt (L.obj X)
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    ⊢ CategoryTheory.IsIso (E.hom.app X)
  -/
  simpa using h.isIso_π_app_of_isInitial _ StructuredArrow.mkIdInitial
  /-
    🎉 no goals
  -/


/-- A pointwise right Kan extension of `F` along `L` applied to an object `Y` is isomorphic to
`limit (StructuredArrow.proj Y L ⋙ F)`. -/
noncomputable def isoLimit :
    E.left.obj Y ≅ limit (StructuredArrow.proj Y L ⋙ F) :=
  h.conePointUniqueUpToIso (limit.isLimit _)


@[reassoc (attr := simp)]
lemma isoLimit_hom_π (g : StructuredArrow Y L) :
    h.isoLimit.hom ≫ limit.π _ g = E.left.map g.hom ≫ E.hom.app g.right :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ _


@[reassoc (attr := simp)]
lemma isoLimit_inv_π (g : StructuredArrow Y L) :
    h.isoLimit.inv ≫ E.left.map g.hom ≫ E.hom.app g.right =
      limit.π (StructuredArrow.proj Y L ⋙ F) g := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    Y : D
    h : E.IsPointwiseRightKanExtensionAt Y
    inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.StructuredArrow.proj Y …
    g : CategoryTheory.StructuredArrow Y L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.isoLimit.inv (CategoryTheory.Catego …
  -/
  simpa using h.conePointUniqueUpToIso_inv_comp (limit.isLimit _) g
  /-
    🎉 no goals
  -/


/-- A right extension `E : RightExtension L F` is a pointwise right Kan extension when
it is a pointwise right Kan extension at any object. -/
abbrev IsPointwiseRightKanExtension := ∀ (Y : D), E.IsPointwiseRightKanExtensionAt Y


/-- If two right extensions `E` and `E'` are isomorphic, `E` is a pointwise
right Kan extension at `Y` iff `E'` is. -/
def isPointwiseRightKanExtensionAtEquivOfIso (e : E ≅ E') (Y : D) :
    E.IsPointwiseRightKanExtensionAt Y ≃ E'.IsPointwiseRightKanExtensionAt Y :=
  IsLimit.equivIsoLimit ((coneAtFunctor L F Y).mapIso e)


/-- If two right extensions `E` and `E'` are isomorphic, `E` is a pointwise
right Kan extension iff `E'` is. -/
def isPointwiseRightKanExtensionEquivOfIso (e : E ≅ E') :
    E.IsPointwiseRightKanExtension ≃ E'.IsPointwiseRightKanExtension where
  toFun h := fun Y => (isPointwiseRightKanExtensionAtEquivOfIso e Y) (h Y)
  invFun h := fun Y => (isPointwiseRightKanExtensionAtEquivOfIso e Y).symm (h Y)
                   /-
                     C : Type u_1
                     D : Type u_2
                     H : Type u_3
                     inst✝² : CategoryTheory.Category.{?u.97778, u_1} C
                     inst✝¹ : CategoryTheory.Category.{?u.97782, u_2} D
                     inst✝ : CategoryTheory.Category.{?u.97786, u_3} H
                     L : CategoryTheory.Functor C D
                     F : CategoryTheory.Functor C H
                     E E' : L.RightExtension F
                     e : CategoryTheory.Iso E E'
                     h : E.IsPointwiseRightKanExtension
                     ⊢ Eq ((fun h Y => (CategoryTheory.Functor.RightExtension.isPointwiseRightKanEx …
                   -/
  left_inv h := by aesop
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u_1
                      D : Type u_2
                      H : Type u_3
                      inst✝² : CategoryTheory.Category.{?u.97778, u_1} C
                      inst✝¹ : CategoryTheory.Category.{?u.97782, u_2} D
                      inst✝ : CategoryTheory.Category.{?u.97786, u_3} H
                      L : CategoryTheory.Functor C D
                      F : CategoryTheory.Functor C H
                      E E' : L.RightExtension F
                      e : CategoryTheory.Iso E E'
                      h : E'.IsPointwiseRightKanExtension
                      ⊢ Eq ((fun h Y => (CategoryTheory.Functor.RightExtension.isPointwiseRightKanEx …
                    -/
  right_inv h := by aesop
                    /-
                      🎉 no goals
                    -/


lemma IsPointwiseRightKanExtension.hasPointwiseRightKanExtension :
    HasPointwiseRightKanExtension L F :=
  fun Y => (h Y).hasPointwiseRightKanExtensionAt


/-- The (unique) morphism to a pointwise right Kan extension. -/
def IsPointwiseRightKanExtension.homTo (G : RightExtension L F) : G ⟶ E :=
  CostructuredArrow.homMk
    { app := fun Y ↦ (h Y).lift (RightExtension.coneAt G Y)
      naturality := fun Y₁ Y₂ φ ↦ (h Y₂).hom_ext (fun X ↦ by
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.101543, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.101547, u_2} D
          inst✝ : CategoryTheory.Category.{?u.101551, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E E' : L.RightExtension F
          h : E.IsPointwiseRightKanExtension
          G : L.RightExtension F
          Y₁ Y₂ : D
          φ : Quiver.Hom Y₁ Y₂
          X : CategoryTheory.StructuredArrow Y₂ L
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [assoc, (h Y₂).fac (coneAt G Y₂) X]
        /-
          C : Type u_1
          D : Type u_2
          H : Type u_3
          inst✝² : CategoryTheory.Category.{?u.101543, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.101547, u_2} D
          inst✝ : CategoryTheory.Category.{?u.101551, u_3} H
          L : CategoryTheory.Functor C D
          F : CategoryTheory.Functor C H
          E E' : L.RightExtension F
          h : E.IsPointwiseRightKanExtension
          G : L.RightExtension F
          Y₁ Y₂ : D
          φ : Quiver.Hom Y₁ Y₂
          X : CategoryTheory.StructuredArrow Y₂ L
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.left.map φ) ((G.coneAt Y₂).π.app X …
        -/
        simpa using ((h Y₁).fac (coneAt G Y₁) ((StructuredArrow.map φ).obj X)).symm) }
        /-
          🎉 no goals
        -/
    (by
      /-
        C : Type u_1
        D : Type u_2
        H : Type u_3
        inst✝² : CategoryTheory.Category.{?u.101543, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.101547, u_2} D
        inst✝ : CategoryTheory.Category.{?u.101551, u_3} H
        L : CategoryTheory.Functor C D
        F : CategoryTheory.Functor C H
        E E' : L.RightExtension F
        h : E.IsPointwiseRightKanExtension
        G : L.RightExtension F
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft C D  …
      -/
      ext X
      /-
        case w.h
        C : Type u_1
        D : Type u_2
        H : Type u_3
        inst✝² : CategoryTheory.Category.{?u.101543, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.101547, u_2} D
        inst✝ : CategoryTheory.Category.{?u.101551, u_3} H
        L : CategoryTheory.Functor C D
        F : CategoryTheory.Functor C H
        E E' : L.RightExtension F
        h : E.IsPointwiseRightKanExtension
        G : L.RightExtension F
        X : C
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft C D …
      -/
      simpa using (h (L.obj X)).fac (RightExtension.coneAt G _) (StructuredArrow.mk (𝟙 _)) )
      /-
        🎉 no goals
      -/


lemma IsPointwiseRightKanExtension.hom_ext
    {G : RightExtension L F} {f₁ f₂ : G ⟶ E} : f₁ = f₂ := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    ⊢ Eq f₁ f₂
  -/
  ext Y
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    Y : D
    ⊢ Eq (f₁.left.app Y) (f₂.left.app Y)
  -/
  apply (h Y).hom_ext
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    Y : D
    ⊢ ∀ (j : CategoryTheory.StructuredArrow Y L), Eq (CategoryTheory.CategoryStruc …
  -/
  intro X
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    Y : D
    X : CategoryTheory.StructuredArrow Y L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f₁.left.app Y) ((E.coneAt Y).π.app X …
  -/
  have eq₁ := congr_app (CostructuredArrow.w f₁) X.right
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    Y : D
    X : CategoryTheory.StructuredArrow Y L
    eq₁ : Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f₁.left.app Y) ((E.coneAt Y).π.app X …
  -/
  have eq₂ := congr_app (CostructuredArrow.w f₂) X.right
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    Y : D
    X : CategoryTheory.StructuredArrow Y L
    eq₁ : Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft …
    eq₂ : Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f₁.left.app Y) ((E.coneAt Y).π.app X …
  -/
  dsimp at eq₁ eq₂ ⊢
  /-
    case h.w.h
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    inst✝ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    E : L.RightExtension F
    h : E.IsPointwiseRightKanExtension
    G : L.RightExtension F
    f₁ f₂ : Quiver.Hom G E
    Y : D
    X : CategoryTheory.StructuredArrow Y L
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (f₁.left.app (L.obj X.right)) (E. …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (f₂.left.app (L.obj X.right)) (E. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f₁.left.app Y) (CategoryTheory.Categ …
  -/
  simp only [assoc, ← NatTrans.naturality_assoc, eq₁, eq₂]
  /-
    🎉 no goals
  -/


/-- A pointwise right Kan extension is universal, i.e. it is a right Kan extension. -/
def IsPointwiseRightKanExtension.isUniversal : E.IsUniversal :=
  IsTerminal.ofUniqueHom h.homTo (fun _ _ => h.hom_ext)


lemma IsPointwiseRightKanExtension.isRightKanExtension :
    E.left.IsRightKanExtension E.hom where
  nonempty_isUniversal := ⟨h.isUniversal⟩


lemma IsPointwiseRightKanExtension.hasRightKanExtension :
    HasRightKanExtension L F :=
  have := h.isRightKanExtension
  HasRightKanExtension.mk E.left E.hom


lemma IsPointwiseRightKanExtension.isIso_hom [L.Full] [L.Faithful] :
    IsIso (E.hom) :=
  have := fun X => (h (L.obj X)).isIso_hom_app
  NatIso.isIso_of_isIso_app ..


/-- The constructed pointwise left Kan extension when `HasPointwiseLeftKanExtension L F` holds. -/
@[simps]
noncomputable def pointwiseLeftKanExtension : D ⥤ H where
  obj Y := colimit (CostructuredArrow.proj L Y ⋙ F)
  map {Y₁ Y₂} f :=
    colimit.desc (CostructuredArrow.proj L Y₁ ⋙ F)
      (Cocone.mk (colimit (CostructuredArrow.proj L Y₂ ⋙ F))
        { app := fun g => colimit.ι (CostructuredArrow.proj L Y₂ ⋙ F)
            ((CostructuredArrow.map f).obj g)
          naturality := fun g₁ g₂ φ => by
            simpa using colimit.w (CostructuredArrow.proj L Y₂ ⋙ F)
              ((CostructuredArrow.map f).map φ) })
  map_id Y := colimit.hom_ext (fun j => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y : D
      j : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y : D
      j : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
    -/
    simp only [colimit.ι_desc, comp_id]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y : D
      j : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq (CategoryTheory.Limits.colimit.ι ((CategoryTheory.CostructuredArrow.proj  …
    -/
    congr
    /-
      case h.e_7.h
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y : D
      j : CategoryTheory.CostructuredArrow L Y
      ⊢ Eq ((CategoryTheory.CostructuredArrow.map (CategoryTheory.CategoryStruct.id  …
    -/
    apply CostructuredArrow.map_id)
    /-
      🎉 no goals
    -/
  map_comp {Y₁ Y₂ Y₃} f f' := colimit.hom_ext (fun j => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.CostructuredArrow L Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.CostructuredArrow L Y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
    -/
    simp only [colimit.ι_desc, colimit.ι_desc_assoc, comp_obj, CostructuredArrow.proj_obj]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.CostructuredArrow L Y₁
      ⊢ Eq (CategoryTheory.Limits.colimit.ι ((CategoryTheory.CostructuredArrow.proj  …
    -/
    congr 1
    /-
      case h.e_7.h
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.121946, u_1} C
      inst✝² : CategoryTheory.Category.{?u.121950, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.121954, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.CostructuredArrow L Y₁
      ⊢ Eq ((CategoryTheory.CostructuredArrow.map (CategoryTheory.CategoryStruct.com …
    -/
    apply CostructuredArrow.map_comp)
    /-
      🎉 no goals
    -/


/-- The unit of the constructed pointwise left Kan extension when
`HasPointwiseLeftKanExtension L F` holds. -/
@[simps]
noncomputable def pointwiseLeftKanExtensionUnit : F ⟶ L ⋙ pointwiseLeftKanExtension L F where
  app X := colimit.ι (CostructuredArrow.proj L (L.obj X) ⋙ F)
    (CostructuredArrow.mk (𝟙 (L.obj X)))
  naturality {X₁ X₂} f := by
    simp only [comp_obj, pointwiseLeftKanExtension_obj, comp_map,
      pointwiseLeftKanExtension_map, colimit.ι_desc, CostructuredArrow.map_mk]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.134220, u_1} C
      inst✝² : CategoryTheory.Category.{?u.134224, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.134228, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      X₁ X₂ : C
      f : Quiver.Hom X₁ X₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.Limits.coli …
    -/
    rw [id_comp]
    let φ : CostructuredArrow.mk (L.map f) ⟶ CostructuredArrow.mk (𝟙 (L.obj X₂)) :=
      CostructuredArrow.homMk f
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.134220, u_1} C
      inst✝² : CategoryTheory.Category.{?u.134224, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.134228, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      X₁ X₂ : C
      f : Quiver.Hom X₁ X₂
      φ : Quiver.Hom (CategoryTheory.CostructuredArrow.mk (L.map f)) (CategoryTheory …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.Limits.coli …
    -/
    exact colimit.w (CostructuredArrow.proj L (L.obj X₂) ⋙ F) φ
    /-
      🎉 no goals
    -/


/-- The functor `pointwiseLeftKanExtension L F` is a pointwise left Kan
extension of `F` along `L`. -/
noncomputable def pointwiseLeftKanExtensionIsPointwiseLeftKanExtension :
    (LeftExtension.mk _ (pointwiseLeftKanExtensionUnit L F)).IsPointwiseLeftKanExtension :=
  fun X => IsColimit.ofIsoColimit (colimit.isColimit _) (Cocones.ext (Iso.refl _) (fun j => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.140117, u_1} C
      inst✝² : CategoryTheory.Category.{?u.140121, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.140125, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      X : D
      j : CategoryTheory.CostructuredArrow L X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.140117, u_1} C
      inst✝² : CategoryTheory.Category.{?u.140121, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.140125, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      X : D
      j : CategoryTheory.CostructuredArrow L X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
    -/
    simp only [comp_id, colimit.ι_desc, CostructuredArrow.map_mk]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.140117, u_1} C
      inst✝² : CategoryTheory.Category.{?u.140121, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.140125, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      X : D
      j : CategoryTheory.CostructuredArrow L X
      ⊢ Eq (CategoryTheory.Limits.colimit.ι ((CategoryTheory.CostructuredArrow.proj  …
    -/
    congr 1
    /-
      case h.e_7.h
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.140117, u_1} C
      inst✝² : CategoryTheory.Category.{?u.140121, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.140125, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseLeftKanExtension F
      X : D
      j : CategoryTheory.CostructuredArrow L X
      ⊢ Eq j (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.com …
    -/
    rw [id_comp, ← CostructuredArrow.eq_mk]))
    /-
      🎉 no goals
    -/


/-- The functor `pointwiseLeftKanExtension L F` is a left Kan extension of `F` along `L`. -/
noncomputable def pointwiseLeftKanExtensionIsUniversal :
    (LeftExtension.mk _ (pointwiseLeftKanExtensionUnit L F)).IsUniversal :=
  (pointwiseLeftKanExtensionIsPointwiseLeftKanExtension L F).isUniversal


instance : (pointwiseLeftKanExtension L F).IsLeftKanExtension
    (pointwiseLeftKanExtensionUnit L F) where
  nonempty_isUniversal := ⟨pointwiseLeftKanExtensionIsUniversal L F⟩


instance : HasLeftKanExtension L F :=
  HasLeftKanExtension.mk _ (pointwiseLeftKanExtensionUnit L F)


/-- An auxiliary cocone used in the lemma `pointwiseLeftKanExtension_desc_app` -/
@[simps]
def costructuredArrowMapCocone (G : D ⥤ H) (α : F ⟶ L ⋙ G) (Y : D) :
    Cocone (CostructuredArrow.proj L Y ⋙ F) where
  pt := G.obj Y
  ι := {
    app := fun f ↦ α.app f.left ≫ G.map f.hom
                     /-
                       C : Type u_1
                       D : Type u_2
                       H : Type u_3
                       inst✝³ : CategoryTheory.Category.{?u.147066, u_1} C
                       inst✝² : CategoryTheory.Category.{?u.147070, u_2} D
                       inst✝¹ : CategoryTheory.Category.{?u.147074, u_3} H
                       L : CategoryTheory.Functor C D
                       F : CategoryTheory.Functor C H
                       inst✝ : L.HasPointwiseLeftKanExtension F
                       G : CategoryTheory.Functor D H
                       α : Quiver.Hom F (L.comp G)
                       Y : D
                       ⊢ ∀ ⦃X Y_1 : CategoryTheory.CostructuredArrow L Y⦄ (f : Quiver.Hom X Y_1), Eq  …
                     -/
    naturality := by simp [← G.map_comp] }
                     /-
                       🎉 no goals
                     -/


@[simp]
lemma pointwiseLeftKanExtension_desc_app (G : D ⥤ H) (α :  F ⟶ L ⋙ G) (Y : D) :
    ((pointwiseLeftKanExtension L F).descOfIsLeftKanExtension (pointwiseLeftKanExtensionUnit L F)
      G α |>.app Y) = colimit.desc _ (costructuredArrowMapCocone L F G α Y) := by
  let β : L.pointwiseLeftKanExtension F ⟶ G :=
    { app := fun Y ↦ colimit.desc _ (costructuredArrowMapCocone L F G α Y) }
  have h : (pointwiseLeftKanExtension L F).descOfIsLeftKanExtension
      (pointwiseLeftKanExtensionUnit L F) G α = β := by
    apply hom_ext_of_isLeftKanExtension (α := pointwiseLeftKanExtensionUnit L F)
    aesop
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    inst✝ : L.HasPointwiseLeftKanExtension F
    G : CategoryTheory.Functor D H
    α : Quiver.Hom F (L.comp G)
    Y : D
    β : Quiver.Hom (L.pointwiseLeftKanExtension F) G := { app := fun Y => Category …
    h : Eq ((L.pointwiseLeftKanExtension F).descOfIsLeftKanExtension (L.pointwiseL …
    ⊢ Eq (((L.pointwiseLeftKanExtension F).descOfIsLeftKanExtension (L.pointwiseLe …
  -/
  exact NatTrans.congr_app h Y
  /-
    🎉 no goals
  -/


/-- If `F` admits a pointwise left Kan extension along `L`, then any left Kan extension of `F`
along `L` is a pointwise left Kan extension. -/
noncomputable def isPointwiseLeftKanExtensionOfIsLeftKanExtension (F' : D ⥤ H) (α : F ⟶ L ⋙ F')
    [F'.IsLeftKanExtension α] :
    (LeftExtension.mk _ α).IsPointwiseLeftKanExtension :=
  LeftExtension.isPointwiseLeftKanExtensionEquivOfIso
    (IsColimit.coconePointUniqueUpToIso (pointwiseLeftKanExtensionIsUniversal L F)
      (F'.isUniversalOfIsLeftKanExtension α))
    (pointwiseLeftKanExtensionIsPointwiseLeftKanExtension L F)


/-- The constructed pointwise right Kan extension
when `HasPointwiseRightKanExtension L F` holds. -/
@[simps]
noncomputable def pointwiseRightKanExtension : D ⥤ H where
  obj Y := limit (StructuredArrow.proj Y L ⋙ F)
  map {Y₁ Y₂} f := limit.lift (StructuredArrow.proj Y₂ L ⋙ F)
      (Cone.mk (limit (StructuredArrow.proj Y₁ L ⋙ F))
        { app := fun g ↦ limit.π (StructuredArrow.proj Y₁ L ⋙ F)
            ((StructuredArrow.map f).obj g)
          naturality := fun g₁ g₂ φ ↦ by
            simpa using (limit.w (StructuredArrow.proj Y₁ L ⋙ F)
              ((StructuredArrow.map f).map φ)).symm })
  map_id Y := limit.hom_ext (fun j => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y : D
      j : CategoryTheory.StructuredArrow Y L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun Y => CategoryTheory.Lim …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y : D
      j : CategoryTheory.StructuredArrow Y L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift ((C …
    -/
    simp only [limit.lift_π, id_comp]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y : D
      j : CategoryTheory.StructuredArrow Y L
      ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.StructuredArrow.proj Y L) …
    -/
    congr
    /-
      case h.e_7.h
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y : D
      j : CategoryTheory.StructuredArrow Y L
      ⊢ Eq ((CategoryTheory.StructuredArrow.map (CategoryTheory.CategoryStruct.id Y) …
    -/
    apply StructuredArrow.map_id)
    /-
      🎉 no goals
    -/
  map_comp {Y₁ Y₂ Y₃} f f' := limit.hom_ext (fun j => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.StructuredArrow Y₃ L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun Y => CategoryTheory.Lim …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.StructuredArrow Y₃ L
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift ((C …
    -/
    simp only [limit.lift_π, assoc]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.StructuredArrow Y₃ L
      ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.StructuredArrow.proj Y₁ L …
    -/
    congr 1
    /-
      case h.e_7.h
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.160970, u_1} C
      inst✝² : CategoryTheory.Category.{?u.160974, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.160978, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      Y₁ Y₂ Y₃ : D
      f : Quiver.Hom Y₁ Y₂
      f' : Quiver.Hom Y₂ Y₃
      j : CategoryTheory.StructuredArrow Y₃ L
      ⊢ Eq ((CategoryTheory.StructuredArrow.map (CategoryTheory.CategoryStruct.comp  …
    -/
    apply StructuredArrow.map_comp)
    /-
      🎉 no goals
    -/


/-- The counit of the constructed pointwise right Kan extension when
`HasPointwiseRightKanExtension L F` holds. -/
@[simps]
noncomputable def pointwiseRightKanExtensionCounit :
    L ⋙ pointwiseRightKanExtension L F ⟶ F where
  app X := limit.π (StructuredArrow.proj (L.obj X) L ⋙ F)
    (StructuredArrow.mk (𝟙 (L.obj X)))
  naturality {X₁ X₂} f := by
    simp only [comp_obj, pointwiseRightKanExtension_obj, comp_map,
      pointwiseRightKanExtension_map, limit.lift_π, StructuredArrow.map_mk]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.174032, u_1} C
      inst✝² : CategoryTheory.Category.{?u.174036, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.174040, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      X₁ X₂ : C
      f : Quiver.Hom X₁ X₂
      ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.StructuredArrow.proj (L.o …
    -/
    rw [comp_id]
    let φ : StructuredArrow.mk (𝟙 (L.obj X₁)) ⟶ StructuredArrow.mk (L.map f) :=
      StructuredArrow.homMk f
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.174032, u_1} C
      inst✝² : CategoryTheory.Category.{?u.174036, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.174040, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      X₁ X₂ : C
      f : Quiver.Hom X₁ X₂
      φ : Quiver.Hom (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.StructuredArrow.proj (L.o …
    -/
    exact (limit.w (StructuredArrow.proj (L.obj X₁) L ⋙ F) φ).symm
    /-
      🎉 no goals
    -/


/-- The functor `pointwiseRightKanExtension L F` is a pointwise right Kan
extension of `F` along `L`. -/
noncomputable def pointwiseRightKanExtensionIsPointwiseRightKanExtension :
    (RightExtension.mk _ (pointwiseRightKanExtensionCounit L F)).IsPointwiseRightKanExtension :=
  fun X => IsLimit.ofIsoLimit (limit.isLimit _) (Cones.ext (Iso.refl _) (fun j => by
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.179739, u_1} C
      inst✝² : CategoryTheory.Category.{?u.179743, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.179747, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      X : D
      j : CategoryTheory.StructuredArrow X L
      ⊢ Eq ((CategoryTheory.Limits.limit.cone ((CategoryTheory.StructuredArrow.proj  …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.179739, u_1} C
      inst✝² : CategoryTheory.Category.{?u.179743, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.179747, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      X : D
      j : CategoryTheory.StructuredArrow X L
      ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.StructuredArrow.proj X L) …
    -/
    simp only [limit.lift_π, StructuredArrow.map_mk, id_comp]
    /-
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.179739, u_1} C
      inst✝² : CategoryTheory.Category.{?u.179743, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.179747, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      X : D
      j : CategoryTheory.StructuredArrow X L
      ⊢ Eq (CategoryTheory.Limits.limit.π ((CategoryTheory.StructuredArrow.proj X L) …
    -/
    congr
    /-
      case h.e_7.h
      C : Type u_1
      D : Type u_2
      H : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.179739, u_1} C
      inst✝² : CategoryTheory.Category.{?u.179743, u_2} D
      inst✝¹ : CategoryTheory.Category.{?u.179747, u_3} H
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor C H
      inst✝ : L.HasPointwiseRightKanExtension F
      X : D
      j : CategoryTheory.StructuredArrow X L
      ⊢ Eq j (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
    -/
    rw [comp_id, ← StructuredArrow.eq_mk]))
    /-
      🎉 no goals
    -/


/-- The functor `pointwiseRightKanExtension L F` is a right Kan extension of `F` along `L`. -/
noncomputable def pointwiseRightKanExtensionIsUniversal :
    (RightExtension.mk _ (pointwiseRightKanExtensionCounit L F)).IsUniversal :=
  (pointwiseRightKanExtensionIsPointwiseRightKanExtension L F).isUniversal


instance : (pointwiseRightKanExtension L F).IsRightKanExtension
    (pointwiseRightKanExtensionCounit L F) where
  nonempty_isUniversal := ⟨pointwiseRightKanExtensionIsUniversal L F⟩


instance : HasRightKanExtension L F :=
  HasRightKanExtension.mk _ (pointwiseRightKanExtensionCounit L F)


/-- An auxiliary cocone used in the lemma `pointwiseRightKanExtension_lift_app` -/
@[simps]
def structuredArrowMapCone (G : D ⥤ H) (α : L ⋙ G ⟶ F) (Y : D) :
    Cone (StructuredArrow.proj Y L ⋙ F) where
  pt := G.obj Y
  π := {
    app := fun f ↦ G.map f.hom ≫ α.app f.right
                     /-
                       C : Type u_1
                       D : Type u_2
                       H : Type u_3
                       inst✝³ : CategoryTheory.Category.{?u.186653, u_1} C
                       inst✝² : CategoryTheory.Category.{?u.186657, u_2} D
                       inst✝¹ : CategoryTheory.Category.{?u.186661, u_3} H
                       L : CategoryTheory.Functor C D
                       F : CategoryTheory.Functor C H
                       inst✝ : L.HasPointwiseRightKanExtension F
                       G : CategoryTheory.Functor D H
                       α : Quiver.Hom (L.comp G) F
                       Y : D
                       ⊢ ∀ ⦃X Y_1 : CategoryTheory.StructuredArrow Y L⦄ (f : Quiver.Hom X Y_1), Eq (C …
                     -/
    naturality := by simp [← α.naturality, ← G.map_comp_assoc] }
                     /-
                       🎉 no goals
                     -/


@[simp]
lemma pointwiseRightKanExtension_lift_app (G : D ⥤ H) (α : L ⋙ G ⟶ F) (Y : D) :
    ((pointwiseRightKanExtension L F).liftOfIsRightKanExtension
      (pointwiseRightKanExtensionCounit L F) G α |>.app Y) =
        limit.lift _ (structuredArrowMapCone L F G α Y) := by
  let β : G ⟶ L.pointwiseRightKanExtension F :=
    { app := fun Y ↦ limit.lift _ (structuredArrowMapCone L F G α Y) }
  have h : (pointwiseRightKanExtension L F).liftOfIsRightKanExtension
      (pointwiseRightKanExtensionCounit L F) G α = β := by
    apply hom_ext_of_isRightKanExtension (α := pointwiseRightKanExtensionCounit L F)
    aesop
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} H
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor C H
    inst✝ : L.HasPointwiseRightKanExtension F
    G : CategoryTheory.Functor D H
    α : Quiver.Hom (L.comp G) F
    Y : D
    β : Quiver.Hom G (L.pointwiseRightKanExtension F) := { app := fun Y => Categor …
    h : Eq ((L.pointwiseRightKanExtension F).liftOfIsRightKanExtension (L.pointwis …
    ⊢ Eq (((L.pointwiseRightKanExtension F).liftOfIsRightKanExtension (L.pointwise …
  -/
  exact NatTrans.congr_app h Y
  /-
    🎉 no goals
  -/


/-- If `F` admits a pointwise right Kan extension along `L`, then any right Kan extension of `F`
along `L` is a pointwise right Kan extension. -/
noncomputable def isPointwiseRightKanExtensionOfIsRightKanExtension (F' : D ⥤ H) (α : L ⋙ F' ⟶ F)
    [F'.IsRightKanExtension α] :
    (RightExtension.mk _ α).IsPointwiseRightKanExtension :=
  RightExtension.isPointwiseRightKanExtensionEquivOfIso
    (IsLimit.conePointUniqueUpToIso (pointwiseRightKanExtensionIsUniversal L F)
      (F'.isUniversalOfIsRightKanExtension α))
    (pointwiseRightKanExtensionIsPointwiseRightKanExtension L F)


