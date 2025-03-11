/-- A category has a shift indexed by an additive monoid `A`
if there is a monoidal functor from `A` to `C ⥤ C`. -/
class HasShift (C : Type u) (A : Type*) [Category.{v} C] [AddMonoid A] where
  /-- a shift is a monoidal functor from `A` to `C ⥤ C` -/
  shift : Discrete A ⥤ C ⥤ C
  /-- `shift` is monoidal -/
  shiftMonoidal : shift.Monoidal := by infer_instance

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- A helper structure to construct the shift functor `(Discrete A) ⥤ (C ⥤ C)`. -/
structure ShiftMkCore where
  /-- the family of shift functors -/
  F : A → C ⥤ C
  /-- the shift by 0 identifies to the identity functor -/
  zero : F 0 ≅ 𝟭 C
  /-- the composition of shift functors identifies to the shift by the sum -/
  add : ∀ n m : A, F (n + m) ≅ F n ⋙ F m
  /-- compatibility with the associativity -/
  assoc_hom_app : ∀ (m₁ m₂ m₃ : A) (X : C),
    (add (m₁ + m₂) m₃).hom.app X ≫ (F m₃).map ((add m₁ m₂).hom.app X) =
                  /-
                    C : Type u
                    A : Type u_1
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    inst✝ : AddMonoid A
                    F : A → CategoryTheory.Functor C C
                    zero : CategoryTheory.Iso (F 0) (CategoryTheory.Functor.id C)
                    add : (n m : A) → CategoryTheory.Iso (F (HAdd.hAdd n m)) ((F n).comp (F m))
                    m₁ m₂ m₃ : A
                    X : C
                    ⊢ Eq ((F (HAdd.hAdd (HAdd.hAdd m₁ m₂) m₃)).obj X) ((F (HAdd.hAdd m₁ (HAdd.hAdd …
                  -/
      eqToHom (by rw [add_assoc]) ≫ (add m₁ (m₂ + m₃)).hom.app X ≫
                  /-
                    🎉 no goals
                  -/
        (add m₂ m₃).hom.app ((F m₁).obj X) := by aesop_cat
  /-- compatibility with the left addition with 0 -/
  zero_add_hom_app : ∀ (n : A) (X : C), (add 0 n).hom.app X =
                /-
                  C : Type u
                  A : Type u_1
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : AddMonoid A
                  F : A → CategoryTheory.Functor C C
                  zero : CategoryTheory.Iso (F 0) (CategoryTheory.Functor.id C)
                  add : (n m : A) → CategoryTheory.Iso (F (HAdd.hAdd n m)) ((F n).comp (F m))
                  assoc_hom_app : autoParam (∀ (m₁ m₂ m₃ : A) (X : C), Eq (CategoryTheory.Catego …
                  n : A
                  X : C
                  ⊢ Eq ((F (HAdd.hAdd 0 n)).obj X) ((F n).obj ((CategoryTheory.Functor.id C).obj …
                -/
    eqToHom (by dsimp; rw [zero_add]) ≫ (F n).map (zero.inv.app X) := by aesop_cat
                       /-
                         🎉 no goals
                       -/
  /-- compatibility with the right addition with 0 -/
  add_zero_hom_app : ∀ (n : A) (X : C), (add n 0).hom.app X =
                /-
                  C : Type u
                  A : Type u_1
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  inst✝ : AddMonoid A
                  F : A → CategoryTheory.Functor C C
                  zero : CategoryTheory.Iso (F 0) (CategoryTheory.Functor.id C)
                  add : (n m : A) → CategoryTheory.Iso (F (HAdd.hAdd n m)) ((F n).comp (F m))
                  assoc_hom_app : autoParam (∀ (m₁ m₂ m₃ : A) (X : C), Eq (CategoryTheory.Catego …
                  zero_add_hom_app : autoParam (∀ (n : A) (X : C), Eq ((add 0 n).hom.app X) (Cat …
                  n : A
                  X : C
                  ⊢ Eq ((F (HAdd.hAdd n 0)).obj X) ((CategoryTheory.Functor.id C).obj ((F n).obj …
                -/
    eqToHom (by dsimp; rw [add_zero]) ≫ zero.inv.app ((F n).obj X) := by aesop_cat
                       /-
                         🎉 no goals
                       -/


attribute [reassoc] assoc_hom_app


@[reassoc]
lemma assoc_inv_app (h : ShiftMkCore C A) (m₁ m₂ m₃ : A) (X : C) :
    (h.F m₃).map ((h.add m₁ m₂).inv.app X) ≫ (h.add (m₁ + m₂) m₃).inv.app X =
    (h.add m₂ m₃).inv.app ((h.F m₁).obj X) ≫ (h.add m₁ (m₂ + m₃)).inv.app X ≫
                  /-
                    C : Type u
                    A : Type u_1
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    inst✝ : AddMonoid A
                    h : CategoryTheory.ShiftMkCore C A
                    m₁ m₂ m₃ : A
                    X : C
                    ⊢ Eq ((h.F (HAdd.hAdd m₁ (HAdd.hAdd m₂ m₃))).obj X) ((h.F (HAdd.hAdd (HAdd.hAd …
                  -/
      eqToHom (by rw [add_assoc]) := by
                  /-
                    🎉 no goals
                  -/
  rw [← cancel_mono ((h.add (m₁ + m₂) m₃).hom.app X ≫ (h.F m₃).map ((h.add m₁ m₂).hom.app X)),
    Category.assoc, Category.assoc, Category.assoc, Iso.inv_hom_id_app_assoc, ← Functor.map_comp,
    Iso.inv_hom_id_app, Functor.map_id, h.assoc_hom_app, eqToHom_trans_assoc, eqToHom_refl,
    Category.id_comp, Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app]
  /-
    C : Type u
    A : Type u_1
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : AddMonoid A
    h : CategoryTheory.ShiftMkCore C A
    m₁ m₂ m₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((h.F m₃).obj (((h.F m₁).comp (h.F m₂)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma zero_add_inv_app (h : ShiftMkCore C A) (n : A) (X : C) :
    (h.add 0 n).inv.app X = (h.F n).map (h.zero.hom.app X) ≫
                  /-
                    C : Type u
                    A : Type u_1
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    inst✝ : AddMonoid A
                    h : CategoryTheory.ShiftMkCore C A
                    n : A
                    X : C
                    ⊢ Eq ((h.F n).obj ((CategoryTheory.Functor.id C).obj X)) ((h.F (HAdd.hAdd 0 n) …
                  -/
      eqToHom (by dsimp; rw [zero_add]) := by
                         /-
                           🎉 no goals
                         -/
  rw [← cancel_epi ((h.add 0 n).hom.app X), Iso.hom_inv_id_app, h.zero_add_hom_app,
    Category.assoc, ← Functor.map_comp_assoc, Iso.inv_hom_id_app, Functor.map_id,
    Category.id_comp, eqToHom_trans, eqToHom_refl]


lemma add_zero_inv_app (h : ShiftMkCore C A) (n : A) (X : C) :
    (h.add n 0).inv.app X = h.zero.hom.app ((h.F n).obj X) ≫
                  /-
                    C : Type u
                    A : Type u_1
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    inst✝ : AddMonoid A
                    h : CategoryTheory.ShiftMkCore C A
                    n : A
                    X : C
                    ⊢ Eq ((CategoryTheory.Functor.id C).obj ((h.F n).obj X)) ((h.F (HAdd.hAdd n 0) …
                  -/
      eqToHom (by dsimp; rw [add_zero]) := by
                         /-
                           🎉 no goals
                         -/
  rw [← cancel_epi ((h.add n 0).hom.app X), Iso.hom_inv_id_app, h.add_zero_hom_app,
    Category.assoc, Iso.inv_hom_id_app_assoc, eqToHom_trans, eqToHom_refl]


instance (h : ShiftMkCore C A) : (Discrete.functor h.F).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := h.zero.symm
      μIso := fun m n ↦ (h.add m.as n.as).symm
      μIso_hom_natural_left := by
        /-
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          ⊢ ∀ {X Y : CategoryTheory.Discrete A} (f : Quiver.Hom X Y) (X' : CategoryTheor …
        -/
        rintro ⟨X⟩ ⟨Y⟩ ⟨⟨⟨rfl⟩⟩⟩ ⟨X'⟩
        /-
          case mk.mk.up.up.refl.mk
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          X X' : A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        ext
        /-
          case mk.mk.up.up.refl.mk.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          X X' : A
          x✝ : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
        -/
        dsimp
        /-
          case mk.mk.up.up.refl.mk.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          X X' : A
          x✝ : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((h.F X').map (CategoryTheory.Categor …
        -/
        simp
        /-
          🎉 no goals
        -/
      μIso_hom_natural_right := by
        /-
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          ⊢ ∀ {X Y : CategoryTheory.Discrete A} (X' : CategoryTheory.Discrete A) (f : Qu …
        -/
        rintro ⟨X⟩ ⟨Y⟩ ⟨X'⟩ ⟨⟨⟨rfl⟩⟩⟩
        /-
          case mk.mk.mk.up.up.refl
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          X X' : A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        ext
        /-
          case mk.mk.mk.up.up.refl.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          X X' : A
          x✝ : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
        -/
        dsimp
        /-
          case mk.mk.mk.up.up.refl.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          X X' : A
          x✝ : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((h …
        -/
        simp
        /-
          🎉 no goals
        -/
      associativity := by
        /-
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          ⊢ ∀ (X Y Z : CategoryTheory.Discrete A), Eq (CategoryTheory.CategoryStruct.com …
        -/
        rintro ⟨m₁⟩ ⟨m₂⟩ ⟨m₃⟩
        /-
          case mk.mk.mk
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          m₁ m₂ m₃ : A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        ext X
        /-
          case mk.mk.mk.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          m₁ m₂ m₃ : A
          X : C
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
        -/
        simp [endofunctorMonoidalCategory, h.assoc_inv_app_assoc]
        /-
          🎉 no goals
        -/
      left_unitality := by
        /-
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          ⊢ ∀ (X : CategoryTheory.Discrete A), Eq (CategoryTheory.MonoidalCategoryStruct …
        -/
        rintro ⟨n⟩
        /-
          case mk
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          n : A
          ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor ((CategoryTheory.Discre …
        -/
        ext X
        /-
          case mk.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          n : A
          X : C
          ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.leftUnitor ((CategoryTheory.Discr …
        -/
        simp [endofunctorMonoidalCategory, h.zero_add_inv_app, ← Functor.map_comp]
        /-
          🎉 no goals
        -/
      right_unitality := by
        /-
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          ⊢ ∀ (X : CategoryTheory.Discrete A), Eq (CategoryTheory.MonoidalCategoryStruct …
        -/
        rintro ⟨n⟩
        /-
          case mk
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          n : A
          ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor ((CategoryTheory.Discr …
        -/
        ext X
        /-
          case mk.w.h
          C : Type u
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : AddMonoid A
          h : CategoryTheory.ShiftMkCore C A
          n : A
          X : C
          ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.rightUnitor ((CategoryTheory.Disc …
        -/
        simp [endofunctorMonoidalCategory, h.add_zero_inv_app] }
        /-
          🎉 no goals
        -/


/-- Constructs a `HasShift C A` instance from `ShiftMkCore`. -/
def hasShiftMk (h : ShiftMkCore C A) : HasShift C A where
  shift := Discrete.functor h.F


/-- The monoidal functor from `A` to `C ⥤ C` given a `HasShift` instance. -/
def shiftMonoidalFunctor : Discrete A ⥤ C ⥤ C :=
  HasShift.shift


instance : (shiftMonoidalFunctor C A).Monoidal := HasShift.shiftMonoidal


/-- The shift autoequivalence, moving objects and morphisms 'up'. -/
def shiftFunctor (i : A) : C ⥤ C :=
  (shiftMonoidalFunctor C A).obj ⟨i⟩


/-- Shifting by `i + j` is the same as shifting by `i` and then shifting by `j`. -/
def shiftFunctorAdd (i j : A) : shiftFunctor C (i + j) ≅ shiftFunctor C i ⋙ shiftFunctor C j :=
  (μIso (shiftMonoidalFunctor C A) ⟨i⟩ ⟨j⟩).symm


/-- When `k = i + j`, shifting by `k` is the same as shifting by `i` and then shifting by `j`. -/
def shiftFunctorAdd' (i j k : A) (h : i + j = k) :
    shiftFunctor C k ≅ shiftFunctor C i ⋙ shiftFunctor C j :=
              /-
                C : Type u
                A : Type u_1
                inst✝² : CategoryTheory.Category.{v, u} C
                inst✝¹ : AddMonoid A
                inst✝ : CategoryTheory.HasShift C A
                i j k : A
                h : Eq (HAdd.hAdd i j) k
                ⊢ Eq (CategoryTheory.shiftFunctor C k) (CategoryTheory.shiftFunctor C (HAdd.hA …
              -/
  eqToIso (by rw [h]) ≪≫ shiftFunctorAdd C i j
              /-
                🎉 no goals
              -/


lemma shiftFunctorAdd'_eq_shiftFunctorAdd (i j : A) :
    shiftFunctorAdd' C i j (i+j) rfl = shiftFunctorAdd C i j := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' C i j (HAdd.hAdd i j) ⋯) (CategoryTheory …
  -/
  ext1
  /-
    case w
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' C i j (HAdd.hAdd i j) ⋯).hom (CategoryTh …
  -/
  apply Category.id_comp
  /-
    🎉 no goals
  -/


variable (A) in
/-- Shifting by zero is the identity functor. -/
def shiftFunctorZero : shiftFunctor C (0 : A) ≅ 𝟭 C :=
  (εIso (shiftMonoidalFunctor C A)).symm


/-- Shifting by `a` such that `a = 0` identifies to the identity functor. -/
def shiftFunctorZero' (a : A) (ha : a = 0) : shiftFunctor C a ≅ 𝟭 C :=
              /-
                C : Type u
                A : Type u_1
                inst✝² : CategoryTheory.Category.{v, u} C
                inst✝¹ : AddMonoid A
                inst✝ : CategoryTheory.HasShift C A
                a : A
                ha : Eq a 0
                ⊢ Eq (CategoryTheory.shiftFunctor C a) (CategoryTheory.shiftFunctor C 0)
              -/
  eqToIso (by rw [ha]) ≪≫ shiftFunctorZero C A
              /-
                🎉 no goals
              -/


lemma ShiftMkCore.shiftFunctor_eq (h : ShiftMkCore C A) (a : A) :
    letI := hasShiftMk C A h
    shiftFunctor C a = h.F a := rfl


lemma ShiftMkCore.shiftFunctorZero_eq (h : ShiftMkCore C A) :
    letI := hasShiftMk C A h
    shiftFunctorZero C A = h.zero := rfl


lemma ShiftMkCore.shiftFunctorAdd_eq (h : ShiftMkCore C A) (a b : A) :
    letI := hasShiftMk C A h
    shiftFunctorAdd C a b = h.add a b := rfl


set_option quotPrecheck false in
/-- shifting an object `X` by `n` is obtained by the notation `X⟦n⟧` -/
notation -- Any better notational suggestions?
X "⟦" n "⟧" => (shiftFunctor _ n).obj X


set_option quotPrecheck false in
/-- shifting a morphism `f` by `n` is obtained by the notation `f⟦n⟧'` -/
notation f "⟦" n "⟧'" => (shiftFunctor _ n).map f


lemma shiftFunctorAdd'_zero_add (a : A) :
    shiftFunctorAdd' C 0 a a (zero_add a) = (Functor.leftUnitor _).symm ≪≫
    isoWhiskerRight (shiftFunctorZero C A).symm (shiftFunctor C a) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' C 0 a a ⋯) ((CategoryTheory.shiftFunctor …
  -/
  ext X
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C 0 a a ⋯).hom.app X) (((CategoryTheory …
  -/
  dsimp [shiftFunctorAdd', shiftFunctorZero, shiftFunctor]
  simp only [eqToHom_app, obj_ε_app, Discrete.addMonoidal_leftUnitor, eqToIso.inv,
    eqToHom_map, Category.id_comp]
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_add_zero (a : A) :
    shiftFunctorAdd' C a 0 a (add_zero a) = (Functor.rightUnitor _).symm ≪≫
    isoWhiskerLeft (shiftFunctor C a) (shiftFunctorZero C A).symm := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' C a 0 a ⋯) ((CategoryTheory.shiftFunctor …
  -/
  ext
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    x✝ : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C a 0 a ⋯).hom.app x✝) (((CategoryTheor …
  -/
  dsimp [shiftFunctorAdd', shiftFunctorZero, shiftFunctor]
  simp only [eqToHom_app, ε_app_obj, Discrete.addMonoidal_rightUnitor, eqToIso.inv,
    eqToHom_map, Category.id_comp]
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    x✝ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) ((Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_assoc (a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A)
    (h₁₂ : a₁ + a₂ = a₁₂) (h₂₃ : a₂ + a₃ = a₂₃) (h₁₂₃ : a₁ + a₂ + a₃ = a₁₂₃) :
                                       /-
                                         C : Type u
                                         A : Type u_1
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         inst✝¹ : AddMonoid A
                                         inst✝ : CategoryTheory.HasShift C A
                                         a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
                                         h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                         h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                         h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
                                         ⊢ Eq (HAdd.hAdd a₁₂ a₃) a₁₂₃
                                       -/
    shiftFunctorAdd' C a₁₂ a₃ a₁₂₃ (by rw [← h₁₂, h₁₂₃]) ≪≫
                                       /-
                                         🎉 no goals
                                       -/
      isoWhiskerRight (shiftFunctorAdd' C a₁ a₂ a₁₂ h₁₂) _ ≪≫ Functor.associator _ _ _ =
                                       /-
                                         C : Type u
                                         A : Type u_1
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         inst✝¹ : AddMonoid A
                                         inst✝ : CategoryTheory.HasShift C A
                                         a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
                                         h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                         h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                         h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
                                         ⊢ Eq (HAdd.hAdd a₁ a₂₃) a₁₂₃
                                       -/
    shiftFunctorAdd' C a₁ a₂₃ a₁₂₃ (by rw [← h₂₃, ← add_assoc, h₁₂₃]) ≪≫
                                       /-
                                         🎉 no goals
                                       -/
      isoWhiskerLeft _ (shiftFunctorAdd' C a₂ a₃ a₂₃ h₂₃) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
    h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
    h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
    h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C a₁₂ a₃ a₁₂₃ ⋯).trans ((CategoryTheory …
  -/
  subst h₁₂ h₂₃ h₁₂₃
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C (HAdd.hAdd a₁ a₂) a₃ (HAdd.hAdd (HAdd …
  -/
  ext X
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (((CategoryTheory.shiftFunctorAdd' C (HAdd.hAdd a₁ a₂) a₃ (HAdd.hAdd (HAd …
  -/
  dsimp
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd' C ( …
  -/
  simp only [shiftFunctorAdd'_eq_shiftFunctorAdd, Category.comp_id]
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C (H …
  -/
  dsimp [shiftFunctorAdd']
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C (H …
  -/
  simp only [eqToHom_app]
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C (H …
  -/
  dsimp [shiftFunctorAdd, shiftFunctor]
  simp only [obj_μ_inv_app, Discrete.addMonoidal_associator, eqToIso.hom, eqToHom_map,
    eqToHom_app]
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.OplaxMonoida …
  -/
  erw [δ_μ_app_assoc, Category.assoc]
  /-
    case w.w.h
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd_assoc (a₁ a₂ a₃ : A) :
    shiftFunctorAdd C (a₁ + a₂) a₃ ≪≫
      isoWhiskerRight (shiftFunctorAdd C a₁ a₂) _ ≪≫ Functor.associator _ _ _ =
    shiftFunctorAdd' C a₁ (a₂ + a₃) _ (add_assoc a₁ a₂ a₃).symm ≪≫
      isoWhiskerLeft _ (shiftFunctorAdd C a₂ a₃) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd C (HAdd.hAdd a₁ a₂) a₃).trans ((Category …
  -/
  ext X
  simpa [shiftFunctorAdd'_eq_shiftFunctorAdd]
    using NatTrans.congr_app (congr_arg Iso.hom
      (shiftFunctorAdd'_assoc C a₁ a₂ a₃ _ _ _ rfl rfl rfl)) X


lemma shiftFunctorAdd'_zero_add_hom_app (a : A) (X : C) :
    (shiftFunctorAdd' C 0 a a (zero_add a)).hom.app X =
    ((shiftFunctorZero C A).inv.app X)⟦a⟧' := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C 0 a a ⋯).hom.app X) ((CategoryTheory. …
  -/
  simpa using NatTrans.congr_app (congr_arg Iso.hom (shiftFunctorAdd'_zero_add C a)) X
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd_zero_add_hom_app (a : A) (X : C) :
    (shiftFunctorAdd C 0 a).hom.app X =
                /-
                  C : Type u
                  A : Type u_1
                  inst✝² : CategoryTheory.Category.{v, u} C
                  inst✝¹ : AddMonoid A
                  inst✝ : CategoryTheory.HasShift C A
                  a : A
                  X : C
                  ⊢ Eq ((CategoryTheory.shiftFunctor C (HAdd.hAdd 0 a)).obj X) ((CategoryTheory. …
                -/
    eqToHom (by dsimp; rw [zero_add]) ≫ ((shiftFunctorZero C A).inv.app X)⟦a⟧' := by
                       /-
                         🎉 no goals
                       -/
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd C 0 a).hom.app X) (CategoryTheory.Catego …
  -/
  simp [← shiftFunctorAdd'_zero_add_hom_app, shiftFunctorAdd']
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_zero_add_inv_app (a : A) (X : C) :
    (shiftFunctorAdd' C 0 a a (zero_add a)).inv.app X =
    ((shiftFunctorZero C A).hom.app X)⟦a⟧' := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C 0 a a ⋯).inv.app X) ((CategoryTheory. …
  -/
  simpa using NatTrans.congr_app (congr_arg Iso.inv (shiftFunctorAdd'_zero_add C a)) X
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd_zero_add_inv_app (a : A) (X : C) : (shiftFunctorAdd C 0 a).inv.app X =
                                                         /-
                                                           C : Type u
                                                           A : Type u_1
                                                           inst✝² : CategoryTheory.Category.{v, u} C
                                                           inst✝¹ : AddMonoid A
                                                           inst✝ : CategoryTheory.HasShift C A
                                                           a : A
                                                           X : C
                                                           ⊢ Eq ((CategoryTheory.shiftFunctor C a).obj ((CategoryTheory.Functor.id C).obj …
                                                         -/
    ((shiftFunctorZero C A).hom.app X)⟦a⟧' ≫ eqToHom (by dsimp; rw [zero_add]) := by
                                                                /-
                                                                  🎉 no goals
                                                                -/
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd C 0 a).inv.app X) (CategoryTheory.Catego …
  -/
  simp [← shiftFunctorAdd'_zero_add_inv_app, shiftFunctorAdd']
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_add_zero_hom_app (a : A) (X : C) :
    (shiftFunctorAdd' C a 0 a (add_zero a)).hom.app X =
    (shiftFunctorZero C A).inv.app (X⟦a⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C a 0 a ⋯).hom.app X) ((CategoryTheory. …
  -/
  simpa using NatTrans.congr_app (congr_arg Iso.hom (shiftFunctorAdd'_add_zero C a)) X
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd_add_zero_hom_app (a : A) (X : C) : (shiftFunctorAdd C a 0).hom.app X =
                /-
                  C : Type u
                  A : Type u_1
                  inst✝² : CategoryTheory.Category.{v, u} C
                  inst✝¹ : AddMonoid A
                  inst✝ : CategoryTheory.HasShift C A
                  a : A
                  X : C
                  ⊢ Eq ((CategoryTheory.shiftFunctor C (HAdd.hAdd a 0)).obj X) ((CategoryTheory. …
                -/
    eqToHom (by dsimp; rw [add_zero]) ≫ (shiftFunctorZero C A).inv.app (X⟦a⟧) := by
                       /-
                         🎉 no goals
                       -/
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd C a 0).hom.app X) (CategoryTheory.Catego …
  -/
  simp [← shiftFunctorAdd'_add_zero_hom_app, shiftFunctorAdd']
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_add_zero_inv_app (a : A) (X : C) :
    (shiftFunctorAdd' C a 0 a (add_zero a)).inv.app X =
    (shiftFunctorZero C A).hom.app (X⟦a⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C a 0 a ⋯).inv.app X) ((CategoryTheory. …
  -/
  simpa using NatTrans.congr_app (congr_arg Iso.inv (shiftFunctorAdd'_add_zero C a)) X
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd_add_zero_inv_app (a : A) (X : C) : (shiftFunctorAdd C a 0).inv.app X =
                                                        /-
                                                          C : Type u
                                                          A : Type u_1
                                                          inst✝² : CategoryTheory.Category.{v, u} C
                                                          inst✝¹ : AddMonoid A
                                                          inst✝ : CategoryTheory.HasShift C A
                                                          a : A
                                                          X : C
                                                          ⊢ Eq ((CategoryTheory.Functor.id C).obj ((CategoryTheory.shiftFunctor C a).obj …
                                                        -/
    (shiftFunctorZero C A).hom.app (X⟦a⟧) ≫ eqToHom (by dsimp; rw [add_zero]) := by
                                                               /-
                                                                 🎉 no goals
                                                               -/
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd C a 0).inv.app X) (CategoryTheory.Catego …
  -/
  simp [← shiftFunctorAdd'_add_zero_inv_app, shiftFunctorAdd']
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shiftFunctorAdd'_assoc_hom_app (a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A)
    (h₁₂ : a₁ + a₂ = a₁₂) (h₂₃ : a₂ + a₃ = a₂₃) (h₁₂₃ : a₁ + a₂ + a₃ = a₁₂₃) (X : C) :
                                        /-
                                          C : Type u
                                          A : Type u_1
                                          inst✝² : CategoryTheory.Category.{v, u} C
                                          inst✝¹ : AddMonoid A
                                          inst✝ : CategoryTheory.HasShift C A
                                          a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
                                          h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                          h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                          h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
                                          X : C
                                          ⊢ Eq (HAdd.hAdd a₁₂ a₃) a₁₂₃
                                        -/
    (shiftFunctorAdd' C a₁₂ a₃ a₁₂₃ (by rw [← h₁₂, h₁₂₃])).hom.app X ≫
                                        /-
                                          🎉 no goals
                                        -/
      ((shiftFunctorAdd' C a₁ a₂ a₁₂ h₁₂).hom.app X)⟦a₃⟧' =
                                        /-
                                          C : Type u
                                          A : Type u_1
                                          inst✝² : CategoryTheory.Category.{v, u} C
                                          inst✝¹ : AddMonoid A
                                          inst✝ : CategoryTheory.HasShift C A
                                          a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
                                          h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                          h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                          h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
                                          X : C
                                          ⊢ Eq (HAdd.hAdd a₁ a₂₃) a₁₂₃
                                        -/
    (shiftFunctorAdd' C a₁ a₂₃ a₁₂₃ (by rw [← h₂₃, ← add_assoc, h₁₂₃])).hom.app X ≫
                                        /-
                                          🎉 no goals
                                        -/
      (shiftFunctorAdd' C a₂ a₃ a₂₃ h₂₃).hom.app (X⟦a₁⟧) := by
  simpa using NatTrans.congr_app (congr_arg Iso.hom
    (shiftFunctorAdd'_assoc C _ _ _ _ _ _ h₁₂ h₂₃ h₁₂₃)) X


@[reassoc]
lemma shiftFunctorAdd'_assoc_inv_app (a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A)
    (h₁₂ : a₁ + a₂ = a₁₂) (h₂₃ : a₂ + a₃ = a₂₃) (h₁₂₃ : a₁ + a₂ + a₃ = a₁₂₃) (X : C) :
    ((shiftFunctorAdd' C a₁ a₂ a₁₂ h₁₂).inv.app X)⟦a₃⟧' ≫
                                          /-
                                            C : Type u
                                            A : Type u_1
                                            inst✝² : CategoryTheory.Category.{v, u} C
                                            inst✝¹ : AddMonoid A
                                            inst✝ : CategoryTheory.HasShift C A
                                            a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
                                            h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                            h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                            h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
                                            X : C
                                            ⊢ Eq (HAdd.hAdd a₁₂ a₃) a₁₂₃
                                          -/
      (shiftFunctorAdd' C a₁₂ a₃ a₁₂₃ (by rw [← h₁₂, h₁₂₃])).inv.app X =
                                          /-
                                            🎉 no goals
                                          -/
    (shiftFunctorAdd' C a₂ a₃ a₂₃ h₂₃).inv.app (X⟦a₁⟧) ≫
                                          /-
                                            C : Type u
                                            A : Type u_1
                                            inst✝² : CategoryTheory.Category.{v, u} C
                                            inst✝¹ : AddMonoid A
                                            inst✝ : CategoryTheory.HasShift C A
                                            a₁ a₂ a₃ a₁₂ a₂₃ a₁₂₃ : A
                                            h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                            h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                            h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a₁₂₃
                                            X : C
                                            ⊢ Eq (HAdd.hAdd a₁ a₂₃) a₁₂₃
                                          -/
      (shiftFunctorAdd' C a₁ a₂₃ a₁₂₃ (by rw [← h₂₃, ← add_assoc, h₁₂₃])).inv.app X := by
                                          /-
                                            🎉 no goals
                                          -/
  simpa using NatTrans.congr_app (congr_arg Iso.inv
    (shiftFunctorAdd'_assoc C _ _ _ _ _ _ h₁₂ h₂₃ h₁₂₃)) X


@[reassoc]
lemma shiftFunctorAdd_assoc_hom_app (a₁ a₂ a₃ : A) (X : C) :
    (shiftFunctorAdd C (a₁ + a₂) a₃).hom.app X ≫
      ((shiftFunctorAdd C a₁ a₂).hom.app X)⟦a₃⟧' =
    (shiftFunctorAdd' C a₁ (a₂ + a₃) (a₁ + a₂ + a₃) (add_assoc _ _ _).symm).hom.app X ≫
      (shiftFunctorAdd C a₂ a₃).hom.app (X⟦a₁⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C (H …
  -/
  simpa using NatTrans.congr_app (congr_arg Iso.hom (shiftFunctorAdd_assoc C a₁ a₂ a₃)) X
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shiftFunctorAdd_assoc_inv_app (a₁ a₂ a₃ : A) (X : C) :
    ((shiftFunctorAdd C a₁ a₂).inv.app X)⟦a₃⟧' ≫
      (shiftFunctorAdd C (a₁ + a₂) a₃).inv.app X =
    (shiftFunctorAdd C a₂ a₃).inv.app (X⟦a₁⟧) ≫
      (shiftFunctorAdd' C a₁ (a₂ + a₃) (a₁ + a₂ + a₃) (add_assoc _ _ _).symm).inv.app X := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    a₁ a₂ a₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a₃).m …
  -/
  simpa using NatTrans.congr_app (congr_arg Iso.inv (shiftFunctorAdd_assoc C a₁ a₂ a₃)) X
  /-
    🎉 no goals
  -/


/-- Shifting by `i + j` is the same as shifting by `i` and then shifting by `j`. -/
abbrev shiftAdd (i j : A) : X⟦i + j⟧ ≅ X⟦i⟧⟦j⟧ :=
  (shiftFunctorAdd C i j).app _


theorem shift_shift' (i j : A) :
    f⟦i⟧'⟦j⟧' = (shiftAdd X i j).inv ≫ f⟦i + j⟧' ≫ (shiftAdd Y i j).hom := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq ((CategoryTheory.shiftFunctor C j).map ((CategoryTheory.shiftFunctor C i) …
  -/
  symm
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.shiftAdd X i j).inv ( …
  -/
  rw [← Functor.comp_map, NatIso.app_inv]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C i  …
  -/
  apply NatIso.naturality_1
  /-
    🎉 no goals
  -/


/-- Shifting by zero is the identity functor. -/
abbrev shiftZero : X⟦(0 : A)⟧ ≅ X :=
  (shiftFunctorZero C A).app _


theorem shiftZero' : f⟦(0 : A)⟧' = (shiftZero A X).hom ≫ f ≫ (shiftZero A Y).inv := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.shiftFunctor C 0).map f) (CategoryTheory.CategoryStruct. …
  -/
  symm
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.shiftZero A X).hom (C …
  -/
  rw [NatIso.app_inv, NatIso.app_hom]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C A …
  -/
  apply NatIso.naturality_2
  /-
    🎉 no goals
  -/


/-- When `i + j = 0`, shifting by `i` and by `j` gives the identity functor -/
def shiftFunctorCompIsoId (i j : A) (h : i + j = 0) :
    shiftFunctor C i ⋙ shiftFunctor C j ≅ 𝟭 C :=
  (shiftFunctorAdd' C i j 0 h).symm ≪≫ shiftFunctorZero C A


/-- Shifting by `i` and shifting by `j` forms an equivalence when `i + j = 0`. -/
@[simps]
def shiftEquiv' (i j : A) (h : i + j = 0) : C ≌ C where
  functor := shiftFunctor C i
  inverse := shiftFunctor C j
  unitIso := (shiftFunctorCompIsoId C i j h).symm
  counitIso := shiftFunctorCompIsoId C j i
        /-
          C : Type u
          A : Type u_1
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : AddGroup A
          inst✝ : CategoryTheory.HasShift C A
          i j : A
          h : Eq (HAdd.hAdd i j) 0
          ⊢ Eq (HAdd.hAdd j i) 0
        -/
    (by rw [← add_left_inj j, add_assoc, h, zero_add, add_zero])
        /-
          🎉 no goals
        -/
  functor_unitIso_comp X := by
    convert (equivOfTensorIsoUnit (shiftMonoidalFunctor C A) ⟨i⟩ ⟨j⟩ (Discrete.eqToIso h)
      (Discrete.eqToIso (by dsimp; rw [← add_left_inj j, add_assoc, h, zero_add, add_zero]))
      (Subsingleton.elim _ _)).functor_unitIso_comp X
    all_goals
      ext X
      dsimp [shiftFunctorCompIsoId, unitOfTensorIsoUnit,
        shiftFunctorAdd']
      simp only [Category.assoc, eqToHom_map]
      rfl


/-- Shifting by `n` and shifting by `-n` forms an equivalence. -/
abbrev shiftEquiv (n : A) : C ≌ C := shiftEquiv' C n (-n) (add_neg_cancel n)


/-- Shifting by `i` is an equivalence. -/
instance (i : A) : (shiftFunctor C i).IsEquivalence := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i : A
    ⊢ (CategoryTheory.shiftFunctor C i).IsEquivalence
  -/
  change (shiftEquiv C i).functor.IsEquivalence
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i : A
    ⊢ (CategoryTheory.shiftEquiv C i).functor.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Shifting by `i` and then shifting by `-i` is the identity. -/
abbrev shiftShiftNeg (i : A) : X⟦i⟧⟦-i⟧ ≅ X :=
  (shiftEquiv C i).unitIso.symm.app X


/-- Shifting by `-i` and then shifting by `i` is the identity. -/
abbrev shiftNegShift (i : A) : X⟦-i⟧⟦i⟧ ≅ X :=
  (shiftEquiv C i).counitIso.app X


theorem shift_shift_neg' (i : A) :
    f⟦i⟧'⟦-i⟧' = (shiftFunctorCompIsoId C i (-i) (add_neg_cancel i)).hom.app X ≫
      f ≫ (shiftFunctorCompIsoId C i (-i) (add_neg_cancel i)).inv.app Y :=
  (NatIso.naturality_2 (shiftFunctorCompIsoId C i (-i) (add_neg_cancel i)) f).symm


theorem shift_neg_shift' (i : A) :
    f⟦-i⟧'⟦i⟧' = (shiftFunctorCompIsoId C (-i) i (neg_add_cancel i)).hom.app X ≫ f ≫
      (shiftFunctorCompIsoId C (-i) i (neg_add_cancel i)).inv.app Y :=
  (NatIso.naturality_2 (shiftFunctorCompIsoId C (-i) i (neg_add_cancel i)) f).symm


theorem shift_equiv_triangle (n : A) (X : C) :
    (shiftShiftNeg X n).inv⟦n⟧' ≫ (shiftNegShift (X⟦n⟧) n).hom = 𝟙 (X⟦n⟧) :=
  (shiftEquiv C n).functor_unitIso_comp X


theorem shift_shiftFunctorCompIsoId_hom_app (n m : A) (h : n + m = 0) (X : C) :
    ((shiftFunctorCompIsoId C n m h).hom.app X)⟦n⟧' =
    (shiftFunctorCompIsoId C m n
          /-
            C : Type u
            A : Type u_1
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : AddGroup A
            inst✝ : CategoryTheory.HasShift C A
            X✝ Y : C
            f : Quiver.Hom X✝ Y
            n m : A
            h : Eq (HAdd.hAdd n m) 0
            X : C
            ⊢ Eq (HAdd.hAdd m n) 0
          -/
      (by rw [← neg_eq_of_add_eq_zero_left h, add_neg_cancel])).hom.app (X⟦n⟧) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    n m : A
    h : Eq (HAdd.hAdd n m) 0
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C n).map ((CategoryTheory.shiftFunctorCompI …
  -/
  dsimp [shiftFunctorCompIsoId]
  simpa only [Functor.map_comp, ← shiftFunctorAdd'_zero_add_inv_app n X,
    ← shiftFunctorAdd'_add_zero_inv_app n X]
    using shiftFunctorAdd'_assoc_inv_app n m n 0 0 n h
      (by rw [← neg_eq_of_add_eq_zero_left h, add_neg_cancel]) (by rw [h, zero_add]) X


theorem shift_shiftFunctorCompIsoId_inv_app (n m : A) (h : n + m = 0) (X : C) :
    ((shiftFunctorCompIsoId C n m h).inv.app X)⟦n⟧' =
    ((shiftFunctorCompIsoId C m n
          /-
            C : Type u
            A : Type u_1
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : AddGroup A
            inst✝ : CategoryTheory.HasShift C A
            X✝ Y : C
            f : Quiver.Hom X✝ Y
            n m : A
            h : Eq (HAdd.hAdd n m) 0
            X : C
            ⊢ Eq (HAdd.hAdd m n) 0
          -/
      (by rw [← neg_eq_of_add_eq_zero_left h, add_neg_cancel])).inv.app (X⟦n⟧)) := by
          /-
            🎉 no goals
          -/
  rw [← cancel_mono (((shiftFunctorCompIsoId C n m h).hom.app X)⟦n⟧'),
    ← Functor.map_comp, Iso.inv_hom_id_app, Functor.map_id,
    shift_shiftFunctorCompIsoId_hom_app, Iso.inv_hom_id_app]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    n m : A
    h : Eq (HAdd.hAdd n m) 0
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor C n).obj  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem shift_shiftFunctorCompIsoId_add_neg_cancel_hom_app (n : A) (X : C) :
    ((shiftFunctorCompIsoId C n (-n) (add_neg_cancel n)).hom.app X)⟦n⟧' =
    (shiftFunctorCompIsoId C (-n) n (neg_add_cancel n)).hom.app (X⟦n⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    n : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C n).map ((CategoryTheory.shiftFunctorCompI …
  -/
  apply shift_shiftFunctorCompIsoId_hom_app
  /-
    🎉 no goals
  -/


theorem shift_shiftFunctorCompIsoId_add_neg_cancel_inv_app (n : A) (X : C) :
    ((shiftFunctorCompIsoId C n (-n) (add_neg_cancel n)).inv.app X)⟦n⟧' =
    (shiftFunctorCompIsoId C (-n) n (neg_add_cancel n)).inv.app (X⟦n⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    n : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C n).map ((CategoryTheory.shiftFunctorCompI …
  -/
  apply shift_shiftFunctorCompIsoId_inv_app
  /-
    🎉 no goals
  -/


theorem shift_shiftFunctorCompIsoId_neg_add_cancel_hom_app (n : A) (X : C) :
    ((shiftFunctorCompIsoId C (-n) n (neg_add_cancel n)).hom.app X)⟦-n⟧' =
    (shiftFunctorCompIsoId C n (-n) (add_neg_cancel n)).hom.app (X⟦-n⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    n : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C (Neg.neg n)).map ((CategoryTheory.shiftFu …
  -/
  apply shift_shiftFunctorCompIsoId_hom_app
  /-
    🎉 no goals
  -/


theorem shift_shiftFunctorCompIsoId_neg_add_cancel_inv_app (n : A) (X : C) :
    ((shiftFunctorCompIsoId C (-n) n (neg_add_cancel n)).inv.app X)⟦-n⟧' =
    (shiftFunctorCompIsoId C n (-n) (add_neg_cancel n)).inv.app (X⟦-n⟧) := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    n : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C (Neg.neg n)).map ((CategoryTheory.shiftFu …
  -/
  apply shift_shiftFunctorCompIsoId_inv_app
  /-
    🎉 no goals
  -/


lemma shiftFunctorCompIsoId_zero_zero_hom_app (X : C) :
    (shiftFunctorCompIsoId C 0 0 (add_zero 0)).hom.app X =
      ((shiftFunctorZero C A).hom.app X)⟦0⟧' ≫ (shiftFunctorZero C A).hom.app X := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorCompIsoId C 0 0 ⋯).hom.app X) (CategoryTheor …
  -/
  simp [shiftFunctorCompIsoId, shiftFunctorAdd'_zero_add_inv_app]
  /-
    🎉 no goals
  -/


lemma shiftFunctorCompIsoId_zero_zero_inv_app (X : C) :
    (shiftFunctorCompIsoId C 0 0 (add_zero 0)).inv.app X =
      (shiftFunctorZero C A).inv.app X ≫ ((shiftFunctorZero C A).inv.app X)⟦0⟧' := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctorCompIsoId C 0 0 ⋯).inv.app X) (CategoryTheor …
  -/
  simp [shiftFunctorCompIsoId, shiftFunctorAdd'_zero_add_hom_app]
  /-
    🎉 no goals
  -/


lemma shiftFunctorCompIsoId_add'_inv_app :
    (shiftFunctorCompIsoId C p' p hp).inv.app X =
      (shiftFunctorCompIsoId C n' n hn).inv.app X ≫
      (shiftFunctorCompIsoId C m' m hm).inv.app (X⟦n'⟧)⟦n⟧' ≫
      (shiftFunctorAdd' C m n p h).inv.app (X⟦n'⟧⟦m'⟧) ≫
      ((shiftFunctorAdd' C n' m' p'
        (by rw [← add_left_inj p, hp, ← h, add_assoc,
          ← add_assoc m', hm, zero_add, hn])).inv.app X)⟦p⟧' := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    m n p m' n' p' : A
    hm : Eq (HAdd.hAdd m' m) 0
    hn : Eq (HAdd.hAdd n' n) 0
    hp : Eq (HAdd.hAdd p' p) 0
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq ((CategoryTheory.shiftFunctorCompIsoId C p' p hp).inv.app X) (CategoryThe …
  -/
  dsimp [shiftFunctorCompIsoId]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    m n p m' n' p' : A
    hm : Eq (HAdd.hAdd m' m) 0
    hn : Eq (HAdd.hAdd n' n) 0
    hp : Eq (HAdd.hAdd p' p) 0
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C A …
  -/
  simp only [Functor.map_comp, Category.assoc]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    m n p m' n' p' : A
    hm : Eq (HAdd.hAdd m' m) 0
    hn : Eq (HAdd.hAdd n' n) 0
    hp : Eq (HAdd.hAdd p' p) 0
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C A …
  -/
  congr 1
  /-
    case e_a
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    m n p m' n' p' : A
    hm : Eq (HAdd.hAdd m' m) 0
    hn : Eq (HAdd.hAdd n' n) 0
    hp : Eq (HAdd.hAdd p' p) 0
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C p' p 0 hp).hom.app X) (CategoryTheory …
  -/
  rw [← NatTrans.naturality]
  /-
    case e_a
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    m n p m' n' p' : A
    hm : Eq (HAdd.hAdd m' m) 0
    hn : Eq (HAdd.hAdd n' n) 0
    hp : Eq (HAdd.hAdd p' p) 0
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C p' p 0 hp).hom.app X) (CategoryTheory …
  -/
  dsimp
  rw [← cancel_mono ((shiftFunctorAdd' C p' p 0 hp).inv.app X), Iso.hom_inv_id_app,
    Category.assoc, Category.assoc, Category.assoc, Category.assoc,
    ← shiftFunctorAdd'_assoc_inv_app p' m n n' p 0
      (by rw [← add_left_inj n, hn, add_assoc, h, hp]) h (by rw [add_assoc, h, hp]),
    ← Functor.map_comp_assoc, ← Functor.map_comp_assoc, ← Functor.map_comp_assoc,
    Category.assoc, Category.assoc,
    shiftFunctorAdd'_assoc_inv_app n' m' m p' 0 n' _ _
      (by rw [add_assoc, hm, add_zero]), Iso.hom_inv_id_app_assoc,
    ← shiftFunctorAdd'_add_zero_hom_app, Iso.hom_inv_id_app,
    Functor.map_id, Category.id_comp, Iso.hom_inv_id_app]


lemma shiftFunctorCompIsoId_add'_hom_app :
    (shiftFunctorCompIsoId C p' p hp).hom.app X =
      ((shiftFunctorAdd' C n' m' p'
          (by rw [← add_left_inj p, hp, ← h, add_assoc,
            ← add_assoc m', hm, zero_add, hn])).hom.app X)⟦p⟧' ≫
      (shiftFunctorAdd' C m n p h).hom.app (X⟦n'⟧⟦m'⟧) ≫
      (shiftFunctorCompIsoId C m' m hm).hom.app (X⟦n'⟧)⟦n⟧' ≫
      (shiftFunctorCompIsoId C n' n hn).hom.app X := by
  rw [← cancel_mono ((shiftFunctorCompIsoId C p' p hp).inv.app X), Iso.hom_inv_id_app,
    shiftFunctorCompIsoId_add'_inv_app m n p m' n' p' hm hn hp h,
    Category.assoc, Category.assoc, Category.assoc, Iso.hom_inv_id_app_assoc,
    ← Functor.map_comp_assoc, Iso.hom_inv_id_app]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddGroup A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    m n p m' n' p' : A
    hm : Eq (HAdd.hAdd m' m) 0
    hn : Eq (HAdd.hAdd n' n) 0
    hp : Eq (HAdd.hAdd p' p) 0
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.shiftFunctor C p').co …
  -/
  dsimp
  rw [Functor.map_id, Category.id_comp, Iso.hom_inv_id_app_assoc,
    ← Functor.map_comp, Iso.hom_inv_id_app, Functor.map_id]


theorem shift_zero_eq_zero (X Y : C) (n : A) : (0 : X ⟶ Y)⟦n⟧' = (0 : X⟦n⟧ ⟶ Y⟦n⟧) :=
  CategoryTheory.Functor.map_zero _ _ _


/-- When shifts are indexed by an additive commutative monoid, then shifts commute. -/
def shiftFunctorComm (i j : A) :
    shiftFunctor C i ⋙ shiftFunctor C j ≅
      shiftFunctor C j ⋙ shiftFunctor C i :=
  (shiftFunctorAdd C i j).symm ≪≫ shiftFunctorAdd' C j i (i + j) (add_comm j i)


lemma shiftFunctorComm_eq (i j k : A) (h : i + j = k) :
    shiftFunctorComm C i j = (shiftFunctorAdd' C i j k h).symm ≪≫
                                   /-
                                     C : Type u
                                     A : Type u_1
                                     inst✝² : CategoryTheory.Category.{v, u} C
                                     inst✝¹ : AddCommMonoid A
                                     inst✝ : CategoryTheory.HasShift C A
                                     i j k : A
                                     h : Eq (HAdd.hAdd i j) k
                                     ⊢ Eq (HAdd.hAdd j i) k
                                   -/
      shiftFunctorAdd' C j i k (by rw [add_comm j i, h]) := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j k : A
    h : Eq (HAdd.hAdd i j) k
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i j) ((CategoryTheory.shiftFunctorAdd' …
  -/
  subst h
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i j) ((CategoryTheory.shiftFunctorAdd' …
  -/
  rw [shiftFunctorAdd'_eq_shiftFunctorAdd]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i j) ((CategoryTheory.shiftFunctorAdd  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma shiftFunctorComm_eq_refl (i : A) :
    shiftFunctorComm C i i = Iso.refl _ := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i : A
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i i) (CategoryTheory.Iso.refl ((Catego …
  -/
  rw [shiftFunctorComm_eq C i i (i + i) rfl, Iso.symm_self_id]
  /-
    🎉 no goals
  -/


lemma shiftFunctorComm_symm (i j : A) :
    (shiftFunctorComm C i j).symm = shiftFunctorComm C j i := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i j).symm (CategoryTheory.shiftFunctor …
  -/
  ext1
  /-
    case w
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i j).symm.hom (CategoryTheory.shiftFun …
  -/
  dsimp
  /-
    case w
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq (CategoryTheory.shiftFunctorComm C i j).inv (CategoryTheory.shiftFunctorC …
  -/
  rw [shiftFunctorComm_eq C i j (i+j) rfl, shiftFunctorComm_eq C j i (i+j) (add_comm j i)]
  /-
    case w
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    i j : A
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' C i j (HAdd.hAdd i j) ⋯).symm.trans (Ca …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- When shifts are indexed by an additive commutative monoid, then shifts commute. -/
abbrev shiftComm (i j : A) : X⟦i⟧⟦j⟧ ≅ X⟦j⟧⟦i⟧ :=
  (shiftFunctorComm C i j).app X


@[simp]
theorem shiftComm_symm (i j : A) : (shiftComm X i j).symm = shiftComm X j i := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    i j : A
    ⊢ Eq (CategoryTheory.shiftComm X i j).symm (CategoryTheory.shiftComm X j i)
  -/
  ext
  /-
    case w
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    i j : A
    ⊢ Eq (CategoryTheory.shiftComm X i j).symm.hom (CategoryTheory.shiftComm X j i …
  -/
  exact NatTrans.congr_app (congr_arg Iso.hom (shiftFunctorComm_symm C i j)) X
  /-
    🎉 no goals
  -/


/-- When shifts are indexed by an additive commutative monoid, then shifts commute. -/
theorem shiftComm' (i j : A) :
    f⟦i⟧'⟦j⟧' = (shiftComm _ _ _).hom ≫ f⟦j⟧'⟦i⟧' ≫ (shiftComm _ _ _).hom := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq ((CategoryTheory.shiftFunctor C j).map ((CategoryTheory.shiftFunctor C i) …
  -/
  erw [← shiftComm_symm Y i j, ← ((shiftFunctorComm C i j).hom.naturality_assoc f)]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq ((CategoryTheory.shiftFunctor C j).map ((CategoryTheory.shiftFunctor C i) …
  -/
  dsimp
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq ((CategoryTheory.shiftFunctor C j).map ((CategoryTheory.shiftFunctor C i) …
  -/
  simp only [Iso.hom_inv_id_app, Functor.comp_obj, Category.comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem shiftComm_hom_comp (i j : A) :
    (shiftComm X i j).hom ≫ f⟦j⟧'⟦i⟧' = f⟦i⟧'⟦j⟧' ≫ (shiftComm Y i j).hom := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X Y : C
    f : Quiver.Hom X Y
    i j : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.shiftComm X i j).hom  …
  -/
  rw [shiftComm', ← shiftComm_symm, Iso.symm_hom, Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


lemma shiftFunctorZero_hom_app_shift (n : A) :
    (shiftFunctorZero C A).hom.app (X⟦n⟧) =
    (shiftFunctorComm C n 0).hom.app X ≫ ((shiftFunctorZero C A).hom.app X)⟦n⟧' := by
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    n : A
    ⊢ Eq ((CategoryTheory.shiftFunctorZero C A).hom.app ((CategoryTheory.shiftFunc …
  -/
  rw [← shiftFunctorAdd'_zero_add_inv_app n X, shiftFunctorComm_eq C n 0 n (add_zero n)]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    n : A
    ⊢ Eq ((CategoryTheory.shiftFunctorZero C A).hom.app ((CategoryTheory.shiftFunc …
  -/
  dsimp
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    n : A
    ⊢ Eq ((CategoryTheory.shiftFunctorZero C A).hom.app ((CategoryTheory.shiftFunc …
  -/
  rw [Category.assoc, Iso.hom_inv_id_app, Category.comp_id, shiftFunctorAdd'_add_zero_inv_app]
  /-
    🎉 no goals
  -/


lemma shiftFunctorZero_inv_app_shift (n : A) :
    (shiftFunctorZero C A).inv.app (X⟦n⟧) =
  ((shiftFunctorZero C A).inv.app X)⟦n⟧' ≫ (shiftFunctorComm C n 0).inv.app X := by
  rw [← cancel_mono ((shiftFunctorZero C A).hom.app (X⟦n⟧)), Category.assoc, Iso.inv_hom_id_app,
    shiftFunctorZero_hom_app_shift, Iso.inv_hom_id_app_assoc, ← Functor.map_comp,
    Iso.inv_hom_id_app]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    n : A
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.Functor.id C).obj ((Ca …
  -/
  dsimp
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : C
    n : A
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor C n).obj  …
  -/
  rw [Functor.map_id]
  /-
    🎉 no goals
  -/


lemma shiftFunctorComm_zero_hom_app (a : A) :
    (shiftFunctorComm C a 0).hom.app X =
      (shiftFunctorZero C A).hom.app (X⟦a⟧) ≫ ((shiftFunctorZero C A).inv.app X)⟦a⟧' := by
  simp only [shiftFunctorZero_hom_app_shift, Category.assoc, ← Functor.map_comp,
    Iso.hom_inv_id_app, Functor.map_id, Functor.comp_obj, Category.comp_id]


@[reassoc]
lemma shiftFunctorComm_hom_app_comp_shift_shiftFunctorAdd_hom_app (m₁ m₂ m₃ : A) (X : C) :
    (shiftFunctorComm C m₁ (m₂ + m₃)).hom.app X ≫
    ((shiftFunctorAdd C m₂ m₃).hom.app X)⟦m₁⟧' =
  (shiftFunctorAdd C m₂ m₃).hom.app (X⟦m₁⟧) ≫
    ((shiftFunctorComm C m₁ m₂).hom.app X)⟦m₃⟧' ≫
    (shiftFunctorComm C m₁ m₃).hom.app (X⟦m₂⟧) := by
  rw [← cancel_mono ((shiftFunctorComm C m₁ m₃).inv.app (X⟦m₂⟧)),
    ← cancel_mono (((shiftFunctorComm C m₁ m₂).inv.app X)⟦m₃⟧')]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    m₁ m₂ m₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, Iso.hom_inv_id_app]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    m₁ m₂ m₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorComm C m …
  -/
  dsimp
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    m₁ m₂ m₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorComm C m …
  -/
  simp only [Category.id_comp, ← Functor.map_comp, Iso.hom_inv_id_app]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    m₁ m₂ m₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorComm C m …
  -/
  dsimp
  simp only [Functor.map_id, Category.comp_id,
    shiftFunctorComm_eq C _ _ _ rfl, ← shiftFunctorAdd'_eq_shiftFunctorAdd]
  /-
    C : Type u
    A : Type u_1
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : AddCommMonoid A
    inst✝ : CategoryTheory.HasShift C A
    m₁ m₂ m₃ : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.shiftFunctorAdd' C  …
  -/
  dsimp
  simp only [Category.assoc, Iso.hom_inv_id_app_assoc, Iso.inv_hom_id_app_assoc,
    ← Functor.map_comp,
    shiftFunctorAdd'_assoc_hom_app_assoc m₂ m₃ m₁ (m₂ + m₃) (m₁ + m₃) (m₁ + (m₂ + m₃)) rfl
      (add_comm m₃ m₁) (add_comm _ m₁) X,
    ← shiftFunctorAdd'_assoc_hom_app_assoc m₂ m₁ m₃ (m₁ + m₂) (m₁ + m₃)
      (m₁ + (m₂ + m₃)) (add_comm _ _) rfl (by rw [add_comm m₂ m₁, add_assoc]) X,
    shiftFunctorAdd'_assoc_hom_app m₁ m₂ m₃
      (m₁ + m₂) (m₂ + m₃) (m₁ + (m₂ + m₃)) rfl rfl (add_assoc _ _ _) X]


/-- auxiliary definition for `FullyFaithful.hasShift` -/
def zero : s 0 ≅ 𝟭 C :=
  (hF.whiskeringRight C).preimageIso ((i 0) ≪≫ isoWhiskerLeft F (shiftFunctorZero D A) ≪≫
    Functor.rightUnitor _ ≪≫ (Functor.leftUnitor _).symm)


@[simp]
lemma map_zero_hom_app (X : C) :
    F.map ((zero hF s i).hom.app X) =
      (i 0).hom.app X ≫ (shiftFunctorZero D A).hom.app (F.obj X) := by
  /-
    C : Type u
    A : Type u_1
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    s : A → CategoryTheory.Functor C C
    i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
    X : C
    ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.zero hF s i).hom.a …
  -/
  simp [zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_zero_inv_app (X : C) :
    F.map ((zero hF s i).inv.app X) =
      (shiftFunctorZero D A).inv.app (F.obj X) ≫ (i 0).inv.app X := by
  /-
    C : Type u
    A : Type u_1
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    s : A → CategoryTheory.Functor C C
    i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
    X : C
    ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.zero hF s i).inv.a …
  -/
  simp [zero]
  /-
    🎉 no goals
  -/


/-- auxiliary definition for `FullyFaithful.hasShift` -/
def add (a b : A) : s (a + b) ≅ s a ⋙ s b :=
  (hF.whiskeringRight C).preimageIso (i (a + b) ≪≫ isoWhiskerLeft _ (shiftFunctorAdd D a b) ≪≫
      (Functor.associator _ _ _).symm ≪≫ (isoWhiskerRight (i a).symm _) ≪≫
      Functor.associator _ _ _ ≪≫ (isoWhiskerLeft _ (i b).symm) ≪≫
      (Functor.associator _ _ _).symm)


@[simp]
lemma map_add_hom_app (a b : A) (X : C) :
    F.map ((add hF s i a b).hom.app X) =
      (i (a + b)).hom.app X ≫ (shiftFunctorAdd D a b).hom.app (F.obj X) ≫
        ((i a).inv.app X)⟦b⟧' ≫ (i b).inv.app ((s a).obj X) := by
  /-
    C : Type u
    A : Type u_1
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    s : A → CategoryTheory.Functor C C
    i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
    a b : A
    X : C
    ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.add hF s i a b).ho …
  -/
  dsimp [add]
  /-
    C : Type u
    A : Type u_1
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    s : A → CategoryTheory.Functor C C
    i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
    a b : A
    X : C
    ⊢ Eq (F.map (hF.preimage (CategoryTheory.CategoryStruct.comp ((i (HAdd.hAdd a  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_add_inv_app (a b : A) (X : C) :
    F.map ((add hF s i a b).inv.app X) =
      (i b).hom.app ((s a).obj X) ≫ ((i a).hom.app X)⟦b⟧' ≫
        (shiftFunctorAdd D a b).inv.app (F.obj X) ≫ (i (a + b)).inv.app X := by
  /-
    C : Type u
    A : Type u_1
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    s : A → CategoryTheory.Functor C C
    i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
    a b : A
    X : C
    ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.add hF s i a b).in …
  -/
  dsimp [add]
  /-
    C : Type u
    A : Type u_1
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift D A
    F : CategoryTheory.Functor C D
    hF : F.FullyFaithful
    s : A → CategoryTheory.Functor C C
    i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
    a b : A
    X : C
    ⊢ Eq (F.map (hF.preimage (CategoryTheory.CategoryStruct.comp (CategoryTheory.C …
  -/
  simp
  /-
    🎉 no goals
  -/


open hasShift in
/-- Given a family of endomorphisms of `C` which are intertwined by a fully faithful `F : C ⥤ D`
with shift functors on `D`, we can promote that family to shift functors on `C`. -/
def hasShift :
    HasShift C A :=
  hasShiftMk C A
    { F := s
      zero := zero hF s i
      add := add hF s i
      assoc_hom_app := fun m₁ m₂ m₃ X => hF.map_injective (by
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          m₁ m₂ m₃ : A
          X : C
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.Fully …
        -/
        have h := shiftFunctorAdd'_assoc_hom_app m₁ m₂ m₃ _ _ (m₁+m₂+m₃) rfl rfl rfl (F.obj X)
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          m₁ m₂ m₃ : A
          X : C
          h : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd' D …
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.Fully …
        -/
        simp only [shiftFunctorAdd'_eq_shiftFunctorAdd] at h
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          m₁ m₂ m₃ : A
          X : C
          h : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd D  …
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.Fully …
        -/
        rw [← cancel_mono ((i m₃).hom.app ((s m₂).obj ((s m₁).obj X)))]
        simp only [Functor.comp_obj, Functor.map_comp, map_add_hom_app,
          Category.assoc, Iso.inv_hom_id_app_assoc, NatTrans.naturality_assoc, Functor.comp_map,
          Iso.inv_hom_id_app, Category.comp_id]
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          m₁ m₂ m₃ : A
          X : C
          h : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd D  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((i (HAdd.hAdd (HAdd.hAdd m₁ m₂) m₃)) …
        -/
        erw [(i m₃).hom.naturality]
        rw [Functor.comp_map, map_add_hom_app,
          Functor.map_comp, Functor.map_comp, Iso.inv_hom_id_app_assoc,
          ← Functor.map_comp_assoc _ ((i (m₁ + m₂)).inv.app X), Iso.inv_hom_id_app,
          Functor.map_id, Category.id_comp, reassoc_of% h,
          dcongr_arg (fun a => (i a).hom.app X) (add_assoc m₁ m₂ m₃)]
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          m₁ m₂ m₃ : A
          X : C
          h : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd D  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp [shiftFunctorAdd', eqToHom_map])
        /-
          🎉 no goals
        -/
      zero_add_hom_app := fun n X => hF.map_injective (by
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.add hF s i 0 n).ho …
        -/
        have this := dcongr_arg (fun a => (i a).hom.app X) (zero_add n)
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          this : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.add hF s i 0 n).ho …
        -/
        rw [← cancel_mono ((i n).hom.app ((s 0).obj X)) ]
        simp [this, map_add_hom_app,
          shiftFunctorAdd_zero_add_hom_app, eqToHom_map]
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          this : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
        -/
        congr 1
        /-
          case e_a
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          this : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((i n).hom.app X) (CategoryTheory.Cat …
        -/
        erw [(i n).hom.naturality]
        /-
          case e_a
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          this : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((i n).hom.app X) (CategoryTheory.Cat …
        -/
        dsimp
        /-
          case e_a
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          this : Eq ((i (HAdd.hAdd 0 n)).hom.app X) (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((i n).hom.app X) (CategoryTheory.Cat …
        -/
        simp)
        /-
          🎉 no goals
        -/
      add_zero_hom_app := fun n X => hF.map_injective (by
        /-
          C : Type u
          A : Type u_1
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.301214, u_2} D
          inst✝¹ : AddMonoid A
          inst✝ : CategoryTheory.HasShift D A
          F : CategoryTheory.Functor C D
          hF : F.FullyFaithful
          s : A → CategoryTheory.Functor C C
          i : (i : A) → CategoryTheory.Iso ((s i).comp F) (F.comp (CategoryTheory.shiftF …
          n : A
          X : C
          ⊢ Eq (F.map ((CategoryTheory.Functor.FullyFaithful.hasShift.add hF s i n 0).ho …
        -/
        have := dcongr_arg (fun a => (i a).hom.app X) (add_zero n)
        simp [this, ← NatTrans.naturality_assoc, eqToHom_map,
          shiftFunctorAdd_add_zero_hom_app]) }


