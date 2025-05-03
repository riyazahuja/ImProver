/-- The `zero` field of the `ShiftMkCore` structure for the induced shift. -/
noncomputable def zero : s 0 ≅ 𝟭 D :=
  ((whiskeringLeft C D D).obj F).preimageIso ((i 0) ≪≫
    isoWhiskerRight (shiftFunctorZero C A) F ≪≫ F.leftUnitor ≪≫ F.rightUnitor.symm)


/-- The `add` field of the `ShiftMkCore` structure for the induced shift. -/
noncomputable def add (a b : A) : s (a + b) ≅ s a ⋙ s b :=
  ((whiskeringLeft C D D).obj F).preimageIso
    (i (a + b) ≪≫ isoWhiskerRight (shiftFunctorAdd C a b) F ≪≫
      Functor.associator _ _ _ ≪≫
        isoWhiskerLeft _ (i b).symm ≪≫ (Functor.associator _ _ _).symm ≪≫
        isoWhiskerRight (i a).symm _ ≪≫ Functor.associator _ _ _)


@[simp]
lemma zero_hom_app_obj (X : C) :
    (zero F s i).hom.app (F.obj X) =
      (i 0).hom.app X ≫ F.map ((shiftFunctorZero C A).hom.app X) := by
  have h : whiskerLeft F (zero F s i).hom = _ :=
    ((whiskeringLeft C D D).obj F).map_preimage _
  /-
    C : Type u_5
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_5} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    X : C
    h : Eq (CategoryTheory.whiskerLeft F (CategoryTheory.HasShift.Induced.zero F s …
    ⊢ Eq ((CategoryTheory.HasShift.Induced.zero F s i).hom.app (F.obj X)) (Categor …
  -/
  exact (NatTrans.congr_app h X).trans (by simp)
  /-
    🎉 no goals
  -/


@[simp]
lemma zero_inv_app_obj (X : C) :
    (zero F s i).inv.app (F.obj X) =
      F.map ((shiftFunctorZero C A).inv.app X) ≫ (i 0).inv.app X := by
  have h : whiskerLeft F (zero F s i).inv = _ :=
    ((whiskeringLeft C D D).obj F).map_preimage _
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_5
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    X : C
    h : Eq (CategoryTheory.whiskerLeft F (CategoryTheory.HasShift.Induced.zero F s …
    ⊢ Eq ((CategoryTheory.HasShift.Induced.zero F s i).inv.app (F.obj X)) (Categor …
  -/
  exact (NatTrans.congr_app h X).trans (by simp)
  /-
    🎉 no goals
  -/


@[simp]
lemma add_hom_app_obj (a b : A) (X : C) :
    (add F s i a b).hom.app (F.obj X) =
      (i (a + b)).hom.app X ≫ F.map ((shiftFunctorAdd C a b).hom.app X) ≫
        (i b).inv.app ((shiftFunctor C a).obj X) ≫ (s b).map ((i a).inv.app X) := by
  have h : whiskerLeft F (add F s i a b).hom = _ :=
    ((whiskeringLeft C D D).obj F).map_preimage _
  /-
    C : Type u_5
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_5} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a b : A
    X : C
    h : Eq (CategoryTheory.whiskerLeft F (CategoryTheory.HasShift.Induced.add F s  …
    ⊢ Eq ((CategoryTheory.HasShift.Induced.add F s i a b).hom.app (F.obj X)) (Cate …
  -/
  exact (NatTrans.congr_app h X).trans (by simp)
  /-
    🎉 no goals
  -/


@[simp]
lemma add_inv_app_obj (a b : A) (X : C) :
    (add F s i a b).inv.app (F.obj X) =
      (s b).map ((i a).hom.app X) ≫ (i b).hom.app ((shiftFunctor C a).obj X) ≫
        F.map ((shiftFunctorAdd C a b).inv.app X) ≫ (i (a + b)).inv.app X := by
  have h : whiskerLeft F (add F s i a b).inv = _ :=
    ((whiskeringLeft C D D).obj F).map_preimage _
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_5
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a b : A
    X : C
    h : Eq (CategoryTheory.whiskerLeft F (CategoryTheory.HasShift.Induced.add F s  …
    ⊢ Eq ((CategoryTheory.HasShift.Induced.add F s i a b).inv.app (F.obj X)) (Cate …
  -/
  exact (NatTrans.congr_app h X).trans (by simp)
  /-
    🎉 no goals
  -/


/-- When `F : C ⥤ D` is a functor satisfying suitable technical assumptions,
this is the induced term of type `HasShift D A` deduced from `[HasShift C A]`. -/
noncomputable def induced : HasShift D A :=
  hasShiftMk D A
    { F := s
      zero := Induced.zero F s i
      add := Induced.add F s i
      zero_add_hom_app := fun n => by
        suffices (Induced.add F s i 0 n).hom =
          eqToHom (by rw [zero_add]; rfl) ≫ whiskerRight (Induced.zero F s i ).inv (s n) by
          intro X
          simpa using NatTrans.congr_app this X
        /-
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          ⊢ Eq (CategoryTheory.HasShift.Induced.add F s i 0 n).hom (CategoryTheory.Categ …
        -/
        apply ((whiskeringLeft C D D).obj F).map_injective
        /-
          case a
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          ⊢ Eq (((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.HasShi …
        -/
        ext X
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          X : C
          ⊢ Eq ((((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.HasSh …
        -/
        have eq := dcongr_arg (fun a => (i a).hom.app X) (zero_add n)
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          X : C
          eq : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp (C …
          ⊢ Eq ((((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.HasSh …
        -/
        dsimp
        simp only [Induced.add_hom_app_obj, eq, shiftFunctorAdd_zero_add_hom_app,
          Functor.map_comp, eqToHom_map, Category.assoc, eqToHom_trans_assoc,
          eqToHom_refl, Category.id_comp, eqToHom_app, Induced.zero_inv_app_obj]
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          X : C
          eq : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
        erw [← NatTrans.naturality_assoc, Iso.hom_inv_id_app_assoc]
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          X : C
          eq : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
        rfl
        /-
          🎉 no goals
        -/
      add_zero_hom_app := fun n => by
        suffices (Induced.add F s i n 0).hom =
            eqToHom (by rw [add_zero]; rfl) ≫ whiskerLeft (s n) (Induced.zero F s i).inv by
          intro X
          simpa using NatTrans.congr_app this X
        /-
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          ⊢ Eq (CategoryTheory.HasShift.Induced.add F s i n 0).hom (CategoryTheory.Categ …
        -/
        apply ((whiskeringLeft C D D).obj F).map_injective
        /-
          case a
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          ⊢ Eq (((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.HasShi …
        -/
        ext X
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          X : C
          ⊢ Eq ((((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.HasSh …
        -/
        dsimp
        erw [Induced.add_hom_app_obj, dcongr_arg (fun a => (i a).hom.app X) (add_zero n),
          ← cancel_mono ((s 0).map ((i n).hom.app X)), Category.assoc,
          Category.assoc, Category.assoc, Category.assoc, Category.assoc,
          Category.assoc, ← (s 0).map_comp, Iso.inv_hom_id_app, Functor.map_id, Category.comp_id,
        /-
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.HasShift.Induced.add  …
        -/
          ← NatTrans.naturality, Induced.zero_inv_app_obj,
        /-
          case a
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          ⊢ Eq (((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.Catego …
        -/
          shiftFunctorAdd_add_zero_hom_app]
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          X : C
          ⊢ Eq ((((CategoryTheory.whiskeringLeft C D D).obj F).map (CategoryTheory.Categ …
        -/
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          n : A
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
        simp [eqToHom_map, eqToHom_app]
        /-
          🎉 no goals
        -/
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          X : C
          eq : Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunct …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.HasShift.Induced.add …
        -/
      assoc_hom_app := fun m₁ m₂ m₃ => by
        suffices (Induced.add F s i (m₁ + m₂) m₃).hom ≫
            whiskerRight (Induced.add F s i m₁ m₂).hom (s m₃) =
            eqToHom (by rw [add_assoc]) ≫ (Induced.add F s i m₁ (m₂ + m₃)).hom ≫
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          X : C
          eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.HasShift.Induced.add …
        -/
              whiskerLeft (s m₁) (Induced.add F s i m₂ m₃).hom by
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          X : C
          eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
          intro X
          simpa using NatTrans.congr_app this X
        apply ((whiskeringLeft C D D).obj F).map_injective
        /-
          case a.w.h
          C : Type ?u.42559
          D : Type ?u.42562
          inst✝⁵ : CategoryTheory.Category.{?u.42566, ?u.42559} C
          inst✝⁴ : CategoryTheory.Category.{?u.42570, ?u.42562} D
          F : CategoryTheory.Functor C D
          A : Type ?u.42603
          inst✝³ : AddMonoid A
          inst✝² : CategoryTheory.HasShift C A
          s : A → CategoryTheory.Functor D D
          i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
          inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
          inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
          m₁ m₂ m₃ : A
          X : C
          eq : Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunct …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((i (HAdd.hAdd (HAdd.hAdd m₁ m₂) m₃)) …
        -/
        ext X
        dsimp
        have eq := F.congr_map (shiftFunctorAdd'_assoc_hom_app
          m₁ m₂ m₃ _ _ (m₁+m₂+m₃) rfl rfl rfl X)
        simp only [shiftFunctorAdd'_eq_shiftFunctorAdd] at eq
        simp only [Functor.comp_obj, Functor.map_comp, shiftFunctorAdd',
          Iso.trans_hom, eqToIso.hom, NatTrans.comp_app, eqToHom_app,
          Category.assoc] at eq
        rw [← cancel_mono ((s m₃).map ((s m₂).map ((i m₁).hom.app X)))]
        simp only [Induced.add_hom_app_obj, Category.assoc, Functor.map_comp]
        slice_lhs 4 5 =>
          erw [← Functor.map_comp, Iso.inv_hom_id_app, Functor.map_id]
        erw [Category.id_comp]
        slice_lhs 6 7 =>
          erw [← Functor.map_comp, ← Functor.map_comp, Iso.inv_hom_id_app,
            (s m₂).map_id, (s m₃).map_id]
        erw [Category.comp_id, ← NatTrans.naturality_assoc, reassoc_of% eq,
          dcongr_arg (fun a => (i a).hom.app X) (add_assoc m₁ m₂ m₃).symm]
        simp only [Functor.comp_obj, eqToHom_map, eqToHom_app, NatTrans.naturality_assoc,
          Induced.add_hom_app_obj, Functor.comp_map, Category.assoc, Iso.inv_hom_id_app_assoc,
          eqToHom_trans_assoc, eqToHom_refl, Category.id_comp, Category.comp_id,
          ← Functor.map_comp, Iso.inv_hom_id_app, Functor.map_id] }


@[simp]
lemma shiftFunctor_of_induced (a : A) :
    letI := HasShift.induced F A s i
    shiftFunctor D a = s a := by
  /-
    C : Type u_4
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a : A
    ⊢ Eq (CategoryTheory.shiftFunctor D a) (s a)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftFunctorZero_hom_app_obj_of_induced (X : C) :
    letI := HasShift.induced F A s i
    (shiftFunctorZero D A).hom.app (F.obj X) =
      (i 0).hom.app X ≫ F.map ((shiftFunctorZero C A).hom.app X) := by
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorZero D A).hom.app (F.obj X)) (CategoryTheory …
  -/
  letI := HasShift.induced F A s i
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    X : C
    this : CategoryTheory.HasShift D A := CategoryTheory.HasShift.induced F A s i
    ⊢ Eq ((CategoryTheory.shiftFunctorZero D A).hom.app (F.obj X)) (CategoryTheory …
  -/
  simp only [ShiftMkCore.shiftFunctorZero_eq, HasShift.Induced.zero_hom_app_obj]
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftFunctorZero_inv_app_obj_of_induced (X : C) :
    letI := HasShift.induced F A s i
    (shiftFunctorZero D A).inv.app (F.obj X) =
      F.map ((shiftFunctorZero C A).inv.app X) ≫ (i 0).inv.app X := by
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_5
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorZero D A).inv.app (F.obj X)) (CategoryTheory …
  -/
  letI := HasShift.induced F A s i
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_5
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    X : C
    this : CategoryTheory.HasShift D A := CategoryTheory.HasShift.induced F A s i
    ⊢ Eq ((CategoryTheory.shiftFunctorZero D A).inv.app (F.obj X)) (CategoryTheory …
  -/
  simp only [ShiftMkCore.shiftFunctorZero_eq, HasShift.Induced.zero_inv_app_obj]
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftFunctorAdd_hom_app_obj_of_induced (a b : A) (X : C) :
    letI := HasShift.induced F A s i
    (shiftFunctorAdd D a b).hom.app (F.obj X) =
      (i (a + b)).hom.app X ≫
        F.map ((shiftFunctorAdd C a b).hom.app X) ≫
        (i b).inv.app ((shiftFunctor C a).obj X) ≫
        (s b).map ((i a).inv.app X) := by
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a b : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd D a b).hom.app (F.obj X)) (CategoryTheor …
  -/
  letI := HasShift.induced F A s i
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a b : A
    X : C
    this : CategoryTheory.HasShift D A := CategoryTheory.HasShift.induced F A s i
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd D a b).hom.app (F.obj X)) (CategoryTheor …
  -/
  simp only [ShiftMkCore.shiftFunctorAdd_eq, HasShift.Induced.add_hom_app_obj]
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftFunctorAdd_inv_app_obj_of_induced (a b : A) (X : C) :
    letI := HasShift.induced F A s i
    (shiftFunctorAdd D a b).inv.app (F.obj X) =
      (s b).map ((i a).hom.app X) ≫
      (i b).hom.app ((shiftFunctor C a).obj X) ≫
      F.map ((shiftFunctorAdd C a b).inv.app X) ≫
      (i (a + b)).inv.app X := by
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a b : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd D a b).inv.app (F.obj X)) (CategoryTheor …
  -/
  letI := HasShift.induced F A s i
  /-
    C : Type u_4
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_4} C
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} D
    F : CategoryTheory.Functor C D
    A : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    a b : A
    X : C
    this : CategoryTheory.HasShift D A := CategoryTheory.HasShift.induced F A s i
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd D a b).inv.app (F.obj X)) (CategoryTheor …
  -/
  simp only [ShiftMkCore.shiftFunctorAdd_eq, HasShift.Induced.add_inv_app_obj]
  /-
    🎉 no goals
  -/


/-- When the target category of a functor `F : C ⥤ D` is equipped with
the induced shift, this is the compatibility of `F` with the shifts on
the categories `C` and `D`. -/
def Functor.CommShift.ofInduced :
    letI := HasShift.induced F A s i
    F.CommShift A := by
  /-
    C : Type ?u.120510
    D : Type ?u.120513
    inst✝⁵ : CategoryTheory.Category.{?u.120517, ?u.120510} C
    inst✝⁴ : CategoryTheory.Category.{?u.120521, ?u.120513} D
    F : CategoryTheory.Functor C D
    A : Type ?u.120554
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    s : A → CategoryTheory.Functor D D
    i : (a : A) → CategoryTheory.Iso (F.comp (s a)) ((CategoryTheory.shiftFunctor  …
    inst✝¹ : ((CategoryTheory.whiskeringLeft C D D).obj F).Full
    inst✝ : ((CategoryTheory.whiskeringLeft C D D).obj F).Faithful
    ⊢ F.CommShift A
  -/
  letI := HasShift.induced F A s i
  exact
    { iso := fun a => (i a).symm
      zero := by
        ext X
        dsimp
        simp only [isoZero_hom_app, shiftFunctorZero_inv_app_obj_of_induced,
          ← F.map_comp_assoc, Iso.hom_inv_id_app, F.map_id, Category.id_comp]
      add := fun a b => by
        ext X
        dsimp
        simp only [isoAdd_hom_app, Iso.symm_hom, shiftFunctorAdd_inv_app_obj_of_induced,
          shiftFunctor_of_induced]
        erw [← Functor.map_comp_assoc, Iso.inv_hom_id_app, Functor.map_id,
          Category.id_comp, Iso.inv_hom_id_app_assoc, ← F.map_comp_assoc, Iso.hom_inv_id_app,
          F.map_id, Category.id_comp] }


lemma Functor.commShiftIso_eq_ofInduced (a : A) :
    letI := HasShift.induced F A s i
    letI := Functor.CommShift.ofInduced F A s i
    F.commShiftIso a = (i a).symm := rfl


