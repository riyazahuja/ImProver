/-- The `isoZero` field of the induced shift sequence. -/
noncomputable def isoZero : F' 0 ≅ F :=
  ((whiskeringLeft C D A).obj L).preimageIso (e' 0 ≪≫ G.isoShiftZero M ≪≫ e.symm)


lemma isoZero_hom_app_obj (X : C) :
    (isoZero e M F' e').hom.app (L.obj X) =
      (e' 0).hom.app X ≫ (isoShiftZero G M).hom.app X ≫ e.inv.app X :=
  NatTrans.congr_app (((whiskeringLeft C D A).obj L).map_preimage _) X


/-- The `shiftIso` field of the induced shift sequence. -/
noncomputable def shiftIso (n a a' : M) (ha' : n + a = a') :
    shiftFunctor D n ⋙ F' a ≅ F' a' := by
  exact ((whiskeringLeft C D A).obj L).preimageIso ((Functor.associator _ _ _).symm ≪≫
    isoWhiskerRight (L.commShiftIso n).symm _ ≪≫
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (e' a) ≪≫
    G.shiftIso n a a' ha' ≪≫ (e' a').symm)


lemma shiftIso_hom_app_obj (n a a' : M) (ha' : n + a = a') (X : C) :
    (shiftIso L G M F' e' n a a' ha').hom.app (L.obj X) =
      (F' a).map ((L.commShiftIso n).inv.app X) ≫
        (e' a).hom.app (X⟦n⟧) ≫ (G.shiftIso n a a' ha').hom.app X ≫ (e' a').inv.app X :=
                                                                                   /-
                                                                                     C : Type u_1
                                                                                     D : Type u_2
                                                                                     A : Type u_3
                                                                                     inst✝⁹ : CategoryTheory.Category.{u_7, u_1} C
                                                                                     inst✝⁸ : CategoryTheory.Category.{u_6, u_2} D
                                                                                     inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
                                                                                     L : CategoryTheory.Functor C D
                                                                                     G : CategoryTheory.Functor C A
                                                                                     M : Type u_4
                                                                                     inst✝⁶ : AddMonoid M
                                                                                     inst✝⁵ : CategoryTheory.HasShift C M
                                                                                     inst✝⁴ : G.ShiftSequence M
                                                                                     F' : M → CategoryTheory.Functor D A
                                                                                     e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
                                                                                     inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
                                                                                     inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
                                                                                     inst✝¹ : CategoryTheory.HasShift D M
                                                                                     inst✝ : L.CommShift M
                                                                                     n a a' : M
                                                                                     ha' : Eq (HAdd.hAdd n a) a'
                                                                                     X : C
                                                                                     ⊢ Eq (((L.associator (CategoryTheory.shiftFunctor D n) (F' a)).symm.trans ((Ca …
                                                                                   -/
  (NatTrans.congr_app (((whiskeringLeft C D A).obj L).map_preimage _) X).trans (by simp)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- Given an isomorphism of functors `e : L ⋙ F ≅ G` relating functors `L : C ⥤ D`,
`F : D ⥤ A` and `G : C ⥤ A`, an additive monoid `M`, a family of functors `F' : M → D ⥤ A`
equipped with isomorphisms `e' : ∀ m, L ⋙ F' m ≅ G.shift m`, this is the shift sequence
induced on `F` induced by a shift sequence for the functor `G`, provided that
the functor `(whiskeringLeft C D A).obj L` of precomposition by `L` is fully faithful. -/
noncomputable def induced : F.ShiftSequence M where
  sequence := F'
  isoZero := induced.isoZero e M F' e'
  shiftIso := induced.shiftIso L G M F' e'
  shiftIso_zero a := by
    /-
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      a : M
      ⊢ Eq (CategoryTheory.Functor.ShiftSequence.induced.shiftIso L G M F' e' 0 a a  …
    -/
    ext1
    /-
      case w
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      a : M
      ⊢ Eq (CategoryTheory.Functor.ShiftSequence.induced.shiftIso L G M F' e' 0 a a  …
    -/
    apply ((whiskeringLeft C D A).obj L).map_injective
    /-
      case w.a
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      a : M
      ⊢ Eq (((CategoryTheory.whiskeringLeft C D A).obj L).map (CategoryTheory.Functo …
    -/
    ext K
    /-
      case w.a.w.h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      a : M
      K : C
      ⊢ Eq ((((CategoryTheory.whiskeringLeft C D A).obj L).map (CategoryTheory.Funct …
    -/
    dsimp
    simp only [induced.shiftIso_hom_app_obj, shiftIso_zero_hom_app, id_obj,
      NatTrans.naturality, comp_map, Iso.hom_inv_id_app_assoc,
      comp_id, ← Functor.map_comp, L.commShiftIso_zero, CommShift.isoZero_inv_app, assoc,
      Iso.inv_hom_id_app, Functor.map_id]
  shiftIso_add n m a a' a'' ha' ha'' := by
    /-
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq (CategoryTheory.Functor.ShiftSequence.induced.shiftIso L G M F' e' (HAdd. …
    -/
    ext1
    /-
      case w
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq (CategoryTheory.Functor.ShiftSequence.induced.shiftIso L G M F' e' (HAdd. …
    -/
    apply ((whiskeringLeft C D A).obj L).map_injective
    /-
      case w.a
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq (((CategoryTheory.whiskeringLeft C D A).obj L).map (CategoryTheory.Functo …
    -/
    ext K
    /-
      case w.a.w.h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      K : C
      ⊢ Eq ((((CategoryTheory.whiskeringLeft C D A).obj L).map (CategoryTheory.Funct …
    -/
    dsimp
    simp only [id_comp, induced.shiftIso_hom_app_obj,
      G.shiftIso_add_hom_app n m a a' a'' ha' ha'', L.commShiftIso_add,
      comp_obj, CommShift.isoAdd_inv_app, (F' a).map_comp, assoc,
      ← (e' a).hom.naturality_assoc, comp_map]
    simp only [← NatTrans.naturality_assoc, induced.shiftIso_hom_app_obj,
      ← Functor.map_comp_assoc, ← Functor.map_comp, Iso.inv_hom_id_app, comp_obj,
      Functor.map_id, id_comp]
    /-
      case w.a.w.h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      K : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F' a).map (CategoryTheory.CategoryS …
    -/
    dsimp
    /-
      case w.a.w.h
      C : Type u_1
      D : Type u_2
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{?u.32369, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.32373, u_2} D
      inst✝⁷ : CategoryTheory.Category.{?u.32377, u_3} A
      L : CategoryTheory.Functor C D
      F : CategoryTheory.Functor D A
      G : CategoryTheory.Functor C A
      e : CategoryTheory.Iso (L.comp F) G
      M : Type u_4
      inst✝⁶ : AddMonoid M
      inst✝⁵ : CategoryTheory.HasShift C M
      inst✝⁴ : G.ShiftSequence M
      F' : M → CategoryTheory.Functor D A
      e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
      inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
      inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
      inst✝¹ : CategoryTheory.HasShift D M
      inst✝ : L.CommShift M
      n m a a' a'' : M
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      K : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F' a).map (CategoryTheory.CategoryS …
    -/
    simp only [Functor.map_comp, assoc, Iso.inv_hom_id_app_assoc]
    /-
      🎉 no goals
    -/


@[simp, reassoc]
lemma induced_isoShiftZero_hom_app_obj (X : C) :
    letI := (induced e M F' e')
    (F.isoShiftZero M).hom.app (L.obj X) =
      (e' 0).hom.app X ≫ (isoShiftZero G M).hom.app X ≫ e.inv.app X := by
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    X : C
    ⊢ Eq ((F.isoShiftZero M).hom.app (L.obj X)) (CategoryTheory.CategoryStruct.com …
  -/
  apply induced.isoZero_hom_app_obj
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma induced_shiftIso_hom_app_obj (n a a' : M) (ha' : n + a = a') (X : C) :
    letI := (induced e M F' e')
    (F.shiftIso n a a' ha').hom.app (L.obj X) =
      (F.shift a).map ((L.commShiftIso n).inv.app X) ≫ (e' a).hom.app (X⟦n⟧) ≫
        (G.shiftIso n a a' ha').hom.app X ≫ (e' a').inv.app X := by
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_7, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_6, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n a a' : M
    ha' : Eq (HAdd.hAdd n a) a'
    X : C
    ⊢ Eq ((F.shiftIso n a a' ha').hom.app (L.obj X)) (CategoryTheory.CategoryStruc …
  -/
  apply induced.shiftIso_hom_app_obj
  /-
    🎉 no goals
  -/


@[reassoc]
lemma induced_shiftMap {n : M} {X Y : C} (f : X ⟶ Y⟦n⟧) (a a' : M) (h : n + a = a') :
    letI := induced e M F' e'
    F.shiftMap (L.map f ≫ (L.commShiftIso n).hom.app _) a a' h =
      (e' a).hom.app X ≫ G.shiftMap f a a' h ≫ (e' a').inv.app Y := by
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n : M
    X Y : C
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    a a' : M
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (F.shiftMap (CategoryTheory.CategoryStruct.comp (L.map f) ((L.commShiftIs …
  -/
  dsimp [shiftMap]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n : M
    X Y : C
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    a a' : M
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift a).map (CategoryTheory.Cate …
  -/
  rw [Functor.map_comp, induced_shiftIso_hom_app_obj, assoc, assoc]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n : M
    X Y : C
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    a a' : M
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift a).map (L.map f)) (Category …
  -/
  nth_rw 2 [← Functor.map_comp_assoc]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n : M
    X Y : C
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    a a' : M
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift a).map (L.map f)) (Category …
  -/
  simp only [comp_obj, Iso.hom_inv_id_app, map_id, id_comp]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n : M
    X Y : C
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    a a' : M
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift a).map (L.map f)) (Category …
  -/
  rw [← NatTrans.naturality_assoc]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.Category.{u_7, u_2} D
    inst✝⁷ : CategoryTheory.Category.{u_6, u_3} A
    L : CategoryTheory.Functor C D
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    M : Type u_4
    inst✝⁶ : AddMonoid M
    inst✝⁵ : CategoryTheory.HasShift C M
    inst✝⁴ : G.ShiftSequence M
    F' : M → CategoryTheory.Functor D A
    e' : (m : M) → CategoryTheory.Iso (L.comp (F' m)) (G.shift m)
    inst✝³ : ((CategoryTheory.whiskeringLeft C D A).obj L).Full
    inst✝² : ((CategoryTheory.whiskeringLeft C D A).obj L).Faithful
    inst✝¹ : CategoryTheory.HasShift D M
    inst✝ : L.CommShift M
    n : M
    X Y : C
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y)
    a a' : M
    h : Eq (HAdd.hAdd n a) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift a).map (L.map f)) (Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


