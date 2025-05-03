/-- A natural transformation between functors `Karoubi C ⥤ D` is determined
by its value on objects coming from `C`. -/
theorem natTrans_eq {F G : Karoubi C ⥤ D} (φ : F ⟶ G) (P : Karoubi C) :
    φ.app P = F.map (decompId_i P) ≫ φ.app P.X ≫ G.map (decompId_p P) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_5, u_2} D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    φ : Quiver.Hom F G
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (φ.app P) (CategoryTheory.CategoryStruct.comp (F.map P.decompId_i) (Categ …
  -/
  rw [← φ.naturality, ← assoc, ← F.map_comp]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_5, u_2} D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    φ : Quiver.Hom F G
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (φ.app P) (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Cate …
  -/
  conv_lhs => rw [← id_comp (φ.app P), ← F.map_id]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_5, u_2} D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    φ : Quiver.Hom F G
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  congr
  /-
    case e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Category.{u_5, u_2} D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    φ : Quiver.Hom F G
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (CategoryTheory.CategoryStruct.id P) (CategoryTheory.CategoryStruct.comp  …
  -/
  apply decompId
  /-
    🎉 no goals
  -/


/-- The canonical extension of a functor `C ⥤ Karoubi D` to a functor
`Karoubi C ⥤ Karoubi D` -/
@[simps]
def obj (F : C ⥤ Karoubi D) : Karoubi C ⥤ Karoubi D where
  obj P :=
                                      /-
                                        C : Type u_1
                                        D : Type u_2
                                        E : Type u_3
                                        inst✝² : CategoryTheory.Category.{?u.1632, u_1} C
                                        inst✝¹ : CategoryTheory.Category.{?u.1636, u_2} D
                                        inst✝ : CategoryTheory.Category.{?u.1640, u_3} E
                                        F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
                                        P : CategoryTheory.Idempotents.Karoubi C
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (F.map P.p).f) (F.map P …
                                      -/
    ⟨(F.obj P.X).X, (F.map P.p).f, by simpa only [F.map_comp, hom_ext_iff] using F.congr_map P.idem⟩
                                      /-
                                        🎉 no goals
                                      -/
                              /-
                                C : Type u_1
                                D : Type u_2
                                E : Type u_3
                                inst✝² : CategoryTheory.Category.{?u.1632, u_1} C
                                inst✝¹ : CategoryTheory.Category.{?u.1636, u_2} D
                                inst✝ : CategoryTheory.Category.{?u.1640, u_3} E
                                F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
                                X✝ Y✝ : CategoryTheory.Idempotents.Karoubi C
                                f : Quiver.Hom X✝ Y✝
                                ⊢ Eq (F.map f.f).f (CategoryTheory.CategoryStruct.comp ((fun P => { X := (F.ob …
                              -/
  map f := ⟨(F.map f.f).f, by simpa only [F.map_comp, hom_ext_iff] using F.congr_map f.comm⟩
                              /-
                                🎉 no goals
                              -/


/-- Extension of a natural transformation `φ` between functors
`C ⥤ karoubi D` to a natural transformation between the
extension of these functors to `karoubi C ⥤ karoubi D` -/
@[simps]
def map {F G : C ⥤ Karoubi D} (φ : F ⟶ G) : obj F ⟶ obj G where
  app P :=
    { f := (F.map P.p).f ≫ (φ.app P.X).f
      comm := by
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
          inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
          F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
          φ : Quiver.Hom F G
          P : CategoryTheory.Idempotents.Karoubi C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categor …
        -/
        have h := φ.naturality P.p
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
          inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
          F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
          φ : Quiver.Hom F G
          P : CategoryTheory.Idempotents.Karoubi C
          h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p) (φ.app P.X)) (CategoryT …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categor …
        -/
        have h' := F.congr_map P.idem
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
          inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
          F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
          φ : Quiver.Hom F G
          P : CategoryTheory.Idempotents.Karoubi C
          h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p) (φ.app P.X)) (CategoryT …
          h' : Eq (F.map (CategoryTheory.CategoryStruct.comp P.p P.p)) (F.map P.p)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categor …
        -/
        simp only [hom_ext_iff, Karoubi.comp_f, F.map_comp] at h h'
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
          inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
          F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
          φ : Quiver.Hom F G
          P : CategoryTheory.Idempotents.Karoubi C
          h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categ …
          h' : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (F.map P.p).f) (F.ma …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categor …
        -/
        simp only [obj_obj_p, assoc, ← h]
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
          inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
          F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
          φ : Quiver.Hom F G
          P : CategoryTheory.Idempotents.Karoubi C
          h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categ …
          h' : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (F.map P.p).f) (F.ma …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categor …
        -/
        slice_rhs 1 3 => rw [h', h'] }
        /-
          🎉 no goals
        -/
  naturality _ _ f := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.FunctorE …
    -/
    ext
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.FunctorE …
    -/
    dsimp [obj]
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (CategoryTheory.Categor …
    -/
    have h := φ.naturality f.f
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f) (φ.app x✝.X)) (Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (CategoryTheory.Categor …
    -/
    have h' := F.congr_map (comp_p f)
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f) (φ.app x✝.X)) (Category …
      h' : Eq (F.map (CategoryTheory.CategoryStruct.comp f.f x✝.p)) (F.map f.f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (CategoryTheory.Categor …
    -/
    have h'' := F.congr_map (p_comp f)
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f) (φ.app x✝.X)) (Category …
      h' : Eq (F.map (CategoryTheory.CategoryStruct.comp f.f x✝.p)) (F.map f.f)
      h'' : Eq (F.map (CategoryTheory.CategoryStruct.comp x✝¹.p f.f)) (F.map f.f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (CategoryTheory.Categor …
    -/
    simp only [hom_ext_iff, Functor.map_comp, comp_f] at h h' h'' ⊢
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (φ.app x✝.X).f) (Cate …
      h' : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (F.map x✝.p).f) (F.m …
      h'' : Eq (CategoryTheory.CategoryStruct.comp (F.map x✝¹.p).f (F.map f.f).f) (F …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (CategoryTheory.Categor …
    -/
    slice_rhs 2 3 => rw [← h]
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (φ.app x✝.X).f) (Cate …
      h' : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (F.map x✝.p).f) (F.m …
      h'' : Eq (CategoryTheory.CategoryStruct.comp (F.map x✝¹.p).f (F.map f.f).f) (F …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (CategoryTheory.Categor …
    -/
    slice_lhs 1 2 => rw [h']
    /-
      case h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.6208, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.6212, u_2} D
      inst✝ : CategoryTheory.Category.{?u.6216, u_3} E
      F G : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      x✝¹ x✝ : CategoryTheory.Idempotents.Karoubi C
      f : Quiver.Hom x✝¹ x✝
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (φ.app x✝.X).f) (Cate …
      h' : Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (F.map x✝.p).f) (F.m …
      h'' : Eq (CategoryTheory.CategoryStruct.comp (F.map x✝¹.p).f (F.map f.f).f) (F …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.f).f (φ.app x✝.X).f) (Catego …
    -/
    slice_rhs 1 2 => rw [h'']
    /-
      🎉 no goals
    -/


/-- The canonical functor `(C ⥤ Karoubi D) ⥤ (Karoubi C ⥤ Karoubi D)` -/
@[simps]
def functorExtension₁ : (C ⥤ Karoubi D) ⥤ Karoubi C ⥤ Karoubi D where
  obj := FunctorExtension₁.obj
  map := FunctorExtension₁.map
  map_id F := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      ⊢ Eq ({ obj := CategoryTheory.Idempotents.FunctorExtension₁.obj, map := fun {X …
    -/
    ext P
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq (({ obj := CategoryTheory.Idempotents.FunctorExtension₁.obj, map := fun { …
    -/
    exact comp_p (F.map P.p)
    /-
      🎉 no goals
    -/
  map_comp {F G H} φ φ' := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      ⊢ Eq ({ obj := CategoryTheory.Idempotents.FunctorExtension₁.obj, map := fun {X …
    -/
    ext P
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq (({ obj := CategoryTheory.Idempotents.FunctorExtension₁.obj, map := fun { …
    -/
    simp only [comp_f, FunctorExtension₁.map_app_f, NatTrans.comp_app, assoc]
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (CategoryTheory.Categor …
    -/
    have h := φ.naturality P.p
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p) (φ.app P.X)) (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (CategoryTheory.Categor …
    -/
    have h' := F.congr_map P.idem
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p) (φ.app P.X)) (CategoryT …
      h' : Eq (F.map (CategoryTheory.CategoryStruct.comp P.p P.p)) (F.map P.p)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (CategoryTheory.Categor …
    -/
    simp only [hom_ext_iff, comp_f, F.map_comp] at h h'
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categ …
      h' : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (F.map P.p).f) (F.ma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (CategoryTheory.Categor …
    -/
    slice_rhs 2 3 => rw [← h]
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categ …
      h' : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (F.map P.p).f) (F.ma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (CategoryTheory.Categor …
    -/
    slice_rhs 1 2 => rw [h']
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.20525, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.20529, u_2} D
      inst✝ : CategoryTheory.Category.{?u.20533, u_3} E
      F G H : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      φ : Quiver.Hom F G
      φ' : Quiver.Hom G H
      P : CategoryTheory.Idempotents.Karoubi C
      h : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (φ.app P.X).f) (Categ …
      h' : Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (F.map P.p).f) (F.ma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.p).f (CategoryTheory.Categor …
    -/
    simp only [assoc]
    /-
      🎉 no goals
    -/


/-- The natural isomorphism expressing that functors `Karoubi C ⥤ Karoubi D` obtained
using `functorExtension₁` actually extends the original functors `C ⥤ Karoubi D`. -/
@[simps!]
def functorExtension₁CompWhiskeringLeftToKaroubiIso :
    functorExtension₁ C D ⋙ (whiskeringLeft C (Karoubi C) (Karoubi D)).obj (toKaroubi C) ≅ 𝟭 _ :=
  NatIso.ofComponents
    (fun F => NatIso.ofComponents
      (fun X =>
        { hom := { f := (F.obj X).p }
          inv := { f := (F.obj X).p } })
                         /-
                           C : Type u_1
                           D : Type u_2
                           E : Type u_3
                           inst✝² : CategoryTheory.Category.{?u.29939, u_1} C
                           inst✝¹ : CategoryTheory.Category.{?u.29943, u_2} D
                           inst✝ : CategoryTheory.Category.{?u.29947, u_3} E
                           F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
                           X Y : C
                           f : Quiver.Hom X Y
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Idempotents.functo …
                         -/
      (fun {X Y} f => by aesop_cat))
                         /-
                           🎉 no goals
                         -/
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.29939, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.29943, u_2} D
          inst✝ : CategoryTheory.Category.{?u.29947, u_3} E
          ⊢ ∀ {X Y : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)} (f …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- The counit isomorphism of the equivalence `(C ⥤ Karoubi D) ≌ (Karoubi C ⥤ Karoubi D)`. -/
def KaroubiUniversal₁.counitIso :
    (whiskeringLeft C (Karoubi C) (Karoubi D)).obj (toKaroubi C) ⋙ functorExtension₁ C D ≅ 𝟭 _ :=
  NatIso.ofComponents
    (fun G =>
      { hom :=
          { app := fun P =>
              { f := (G.map (decompId_p P)).f
                comm := by
                  simpa only [hom_ext_iff, G.map_comp, G.map_id] using
                    G.congr_map
                      (show P.decompId_p = (toKaroubi C).map P.p ≫ P.decompId_p ≫ 𝟙 _ by simp) }
            naturality := fun P Q f => by
              simpa only [hom_ext_iff, G.map_comp]
                using (G.congr_map (decompId_p_naturality f)).symm }
        inv :=
          { app := fun P =>
              { f := (G.map (decompId_i P)).f
                comm := by
                  simpa only [hom_ext_iff, G.map_comp, G.map_id] using
                    G.congr_map
                      (show P.decompId_i = 𝟙 _ ≫ P.decompId_i ≫ (toKaroubi C).map P.p by simp) }
            naturality := fun P Q f => by
              /-
                C : Type u_1
                D : Type u_2
                E : Type u_3
                inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
                inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
                inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
                G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryThe …
                P Q : CategoryTheory.Idempotents.Karoubi C
                f : Quiver.Hom P Q
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.id (Categor …
              -/
              simpa only [hom_ext_iff, G.map_comp] using G.congr_map (decompId_i_naturality f) }
              /-
                🎉 no goals
              -/
        hom_inv_id := by
          /-
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
            inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
            inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
            G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryThe …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun P => { f := (G.map P.dec …
          -/
          ext P
          /-
            case w.h.h
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
            inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
            inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
            G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryThe …
            P : CategoryTheory.Idempotents.Karoubi C
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun P => { f := (G.map P.de …
          -/
          simpa only [hom_ext_iff, G.map_comp, G.map_id] using G.congr_map P.decomp_p.symm
          /-
            🎉 no goals
          -/
        inv_hom_id := by
          /-
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
            inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
            inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
            G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryThe …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun P => { f := (G.map P.dec …
          -/
          ext P
          /-
            case w.h.h
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
            inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
            inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
            G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryThe …
            P : CategoryTheory.Idempotents.Karoubi C
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun P => { f := (G.map P.de …
          -/
          simpa only [hom_ext_iff, G.map_comp, G.map_id] using G.congr_map P.decompId.symm })
          /-
            🎉 no goals
          -/
    (fun {X Y} φ => by
      /-
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
        inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
        X Y : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryT …
        φ : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.whiskeringLeft C ( …
      -/
      ext P
      /-
        case w.h.h
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
        inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
        X Y : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryT …
        φ : Quiver.Hom X Y
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((((CategoryTheory.whiskeringLeft C  …
      -/
      dsimp
      /-
        case w.h.h
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
        inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
        X Y : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryT …
        φ : Quiver.Hom X Y
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      rw [natTrans_eq φ P, P.decomp_p]
      /-
        case w.h.h
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
        inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
        X Y : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryT …
        φ : Quiver.Hom X Y
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Functor.map_comp, comp_f, assoc]
      /-
        case w.h.h
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝² : CategoryTheory.Category.{?u.48439, u_1} C
        inst✝¹ : CategoryTheory.Category.{?u.48443, u_2} D
        inst✝ : CategoryTheory.Category.{?u.48447, u_3} E
        X Y : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) (CategoryT …
        φ : Quiver.Hom X Y
        P : CategoryTheory.Idempotents.Karoubi C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map P.decompId_p).f (CategoryTheor …
      -/
      rfl)
      /-
        🎉 no goals
      -/


attribute [simps!] KaroubiUniversal₁.counitIso


/-- The equivalence of categories `(C ⥤ Karoubi D) ≌ (Karoubi C ⥤ Karoubi D)`. -/
@[simps]
def karoubiUniversal₁ : C ⥤ Karoubi D ≌ Karoubi C ⥤ Karoubi D where
  functor := functorExtension₁ C D
  inverse := (whiskeringLeft C (Karoubi C) (Karoubi D)).obj (toKaroubi C)
  unitIso := (functorExtension₁CompWhiskeringLeftToKaroubiIso C D).symm
  counitIso := KaroubiUniversal₁.counitIso C D
  functor_unitIso_comp F := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.59076, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.59080, u_2} D
      inst✝ : CategoryTheory.Category.{?u.59084, u_3} E
      F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.functorE …
    -/
    ext P
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.59076, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.59080, u_2} D
      inst✝ : CategoryTheory.Category.{?u.59084, u_3} E
      F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.functor …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝² : CategoryTheory.Category.{?u.59076, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.59080, u_2} D
      inst✝ : CategoryTheory.Category.{?u.59084, u_3} E
      F : CategoryTheory.Functor C (CategoryTheory.Idempotents.Karoubi D)
      P : CategoryTheory.Idempotents.Karoubi C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [comp_p, ← comp_f, ← F.map_comp, P.idem]
    /-
      🎉 no goals
    -/


/-- Compatibility isomorphisms of `functorExtension₁` with respect to the
composition of functors. -/
def functorExtension₁Comp (F : C ⥤ Karoubi D) (G : D ⥤ Karoubi E) :
    (functorExtension₁ C E).obj (F ⋙ (functorExtension₁ D E).obj G) ≅
      (functorExtension₁ C D).obj F ⋙ (functorExtension₁ D E).obj G :=
  Iso.refl _


/-- The canonical functor `(C ⥤ D) ⥤ (Karoubi C ⥤ Karoubi D)` -/
@[simps!]
def functorExtension₂ : (C ⥤ D) ⥤ Karoubi C ⥤ Karoubi D :=
  (whiskeringRight C D (Karoubi D)).obj (toKaroubi D) ⋙ functorExtension₁ C D


/-- The natural isomorphism expressing that functors `Karoubi C ⥤ Karoubi D` obtained
using `functorExtension₂` actually extends the original functors `C ⥤ D`. -/
@[simps!]
def functorExtension₂CompWhiskeringLeftToKaroubiIso :
    functorExtension₂ C D ⋙ (whiskeringLeft C (Karoubi C) (Karoubi D)).obj (toKaroubi C) ≅
      (whiskeringRight C D (Karoubi D)).obj (toKaroubi D) :=
  NatIso.ofComponents
    (fun F => NatIso.ofComponents
      (fun X =>
        { hom := { f := 𝟙 _ }
          inv := { f := 𝟙 _ } })
          /-
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝² : CategoryTheory.Category.{?u.67322, u_1} C
            inst✝¹ : CategoryTheory.Category.{?u.67326, u_2} D
            inst✝ : CategoryTheory.Category.{?u.67330, u_3} E
            F : CategoryTheory.Functor C D
            ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
          -/
      (by aesop_cat))
          /-
            🎉 no goals
          -/
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝² : CategoryTheory.Category.{?u.67322, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.67326, u_2} D
          inst✝ : CategoryTheory.Category.{?u.67330, u_3} E
          ⊢ ∀ {X Y : CategoryTheory.Functor C D} (f : Quiver.Hom X Y), Eq (CategoryTheor …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- The equivalence of categories `(C ⥤ D) ≌ (Karoubi C ⥤ Karoubi D)` when `D`
is idempotent complete. -/
@[simp]
noncomputable def karoubiUniversal₂ : C ⥤ D ≌ Karoubi C ⥤ Karoubi D :=
  (Equivalence.congrRight (toKaroubi D).asEquivalence).trans (karoubiUniversal₁ C D)


theorem karoubiUniversal₂_functor_eq : (karoubiUniversal₂ C D).functor = functorExtension₂ C D :=
  rfl


noncomputable instance : (functorExtension₂ C D).IsEquivalence := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{?u.93425, u_3} E
    inst✝ : CategoryTheory.IsIdempotentComplete D
    ⊢ (CategoryTheory.Idempotents.functorExtension₂ C D).IsEquivalence
  -/
  rw [← karoubiUniversal₂_functor_eq]
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{?u.93425, u_3} E
    inst✝ : CategoryTheory.IsIdempotentComplete D
    ⊢ (CategoryTheory.Idempotents.karoubiUniversal₂ C D).functor.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The extension of functors functor `(C ⥤ D) ⥤ (Karoubi C ⥤ D)`
when `D` is idempotent complete. -/
@[simps!]
noncomputable def functorExtension : (C ⥤ D) ⥤ Karoubi C ⥤ D :=
  functorExtension₂ C D ⋙
    (whiskeringRight (Karoubi C) (Karoubi D) D).obj (toKaroubiEquivalence D).inverse


/-- The equivalence `(C ⥤ D) ≌ (Karoubi C ⥤ D)` when `D` is idempotent complete. -/
@[simp]
noncomputable def karoubiUniversal : C ⥤ D ≌ Karoubi C ⥤ D :=
  (karoubiUniversal₂ C D).trans (Equivalence.congrRight (toKaroubi D).asEquivalence.symm)


theorem karoubiUniversal_functor_eq : (karoubiUniversal C D).functor = functorExtension C D :=
  rfl


noncomputable instance : (functorExtension C D).IsEquivalence := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{?u.97165, u_3} E
    inst✝ : CategoryTheory.IsIdempotentComplete D
    ⊢ (CategoryTheory.Idempotents.functorExtension C D).IsEquivalence
  -/
  rw [← karoubiUniversal_functor_eq]
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{?u.97165, u_3} E
    inst✝ : CategoryTheory.IsIdempotentComplete D
    ⊢ (CategoryTheory.Idempotents.karoubiUniversal C D).functor.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : ((whiskeringLeft C (Karoubi C) D).obj (toKaroubi C)).IsEquivalence := by
  have : ((whiskeringLeft C (Karoubi C) D).obj (toKaroubi C) ⋙
    (whiskeringRight C D (Karoubi D)).obj (toKaroubi D) ⋙
    (whiskeringRight C (Karoubi D) D).obj (Functor.inv (toKaroubi D))).IsEquivalence := by
    change (karoubiUniversal C D).inverse.IsEquivalence
    infer_instance
  exact Functor.isEquivalence_of_comp_right _
    ((whiskeringRight C _ _).obj (toKaroubi D) ⋙
      (whiskeringRight C (Karoubi D) D).obj (Functor.inv (toKaroubi D)))


theorem whiskeringLeft_obj_preimage_app {F G : Karoubi C ⥤ D}
    (τ : toKaroubi _ ⋙ F ⟶ toKaroubi _ ⋙ G) (P : Karoubi C) :
    (((whiskeringLeft _ _ _).obj (toKaroubi _)).preimage τ).app P =
      F.map P.decompId_i ≫ τ.app P.X ≫ G.map P.decompId_p := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    inst✝ : CategoryTheory.IsIdempotentComplete D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    τ : Quiver.Hom ((CategoryTheory.Idempotents.toKaroubi C).comp F) ((CategoryThe …
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq ((((CategoryTheory.whiskeringLeft C (CategoryTheory.Idempotents.Karoubi C …
  -/
  rw [natTrans_eq]
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    inst✝ : CategoryTheory.IsIdempotentComplete D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    τ : Quiver.Hom ((CategoryTheory.Idempotents.toKaroubi C).comp F) ((CategoryThe …
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map P.decompId_i) (CategoryTheory. …
  -/
  congr 2
  /-
    case e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    inst✝ : CategoryTheory.IsIdempotentComplete D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    τ : Quiver.Hom ((CategoryTheory.Idempotents.toKaroubi C).comp F) ((CategoryThe …
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq ((((CategoryTheory.whiskeringLeft C (CategoryTheory.Idempotents.Karoubi C …
  -/
  rw [← congr_app (((whiskeringLeft _ _ _).obj (toKaroubi _)).map_preimage τ) P.X]
  /-
    case e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    inst✝ : CategoryTheory.IsIdempotentComplete D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    τ : Quiver.Hom ((CategoryTheory.Idempotents.toKaroubi C).comp F) ((CategoryThe …
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq ((((CategoryTheory.whiskeringLeft C (CategoryTheory.Idempotents.Karoubi C …
  -/
  dsimp
  /-
    case e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    inst✝ : CategoryTheory.IsIdempotentComplete D
    F G : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi C) D
    τ : Quiver.Hom ((CategoryTheory.Idempotents.toKaroubi C).comp F) ((CategoryThe …
    P : CategoryTheory.Idempotents.Karoubi C
    ⊢ Eq ((((CategoryTheory.whiskeringLeft C (CategoryTheory.Idempotents.Karoubi C …
  -/
  congr
  /-
    🎉 no goals
  -/


