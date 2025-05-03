/-- A complex of functors gives a functor to complexes. -/
@[simps obj map]
def asFunctor {T : Type*} [Category T] (C : HomologicalComplex (T ⥤ V) c) :
    T ⥤ HomologicalComplex V c where
  obj t :=
    { X := fun i => (C.X i).obj t
      d := fun i j => (C.d i j).app t
      d_comp_d' := fun i j k _ _ => by
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.81, u_2} T
          C : HomologicalComplex (CategoryTheory.Functor T V) c
          t : T
          i j k : ι
          x✝¹ : c.Rel i j
          x✝ : c.Rel j k
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (C.d i j).app t) i j) (( …
        -/
        have := C.d_comp_d i j k
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.81, u_2} T
          C : HomologicalComplex (CategoryTheory.Functor T V) c
          t : T
          i j k : ι
          x✝¹ : c.Rel i j
          x✝ : c.Rel j k
          this : Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (C.d i j).app t) i j) (( …
        -/
        rw [NatTrans.ext_iff, funext_iff] at this
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.81, u_2} T
          C : HomologicalComplex (CategoryTheory.Functor T V) c
          t : T
          i j : ι
          h : Not (c.Rel i j)
          ⊢ Eq ((fun i j => (C.d i j).app t) i j) 0
        -/
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.81, u_2} T
          C : HomologicalComplex (CategoryTheory.Functor T V) c
          t : T
          i j k : ι
          x✝¹ : c.Rel i j
          x✝ : c.Rel j k
          this : ∀ (x : T), Eq ((CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (C.d i j).app t) i j) (( …
        -/
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.81, u_2} T
          C : HomologicalComplex (CategoryTheory.Functor T V) c
          t : T
          i j : ι
          h : Not (c.Rel i j)
          this : Eq (C.d i j) 0
          ⊢ Eq ((fun i j => (C.d i j).app t) i j) 0
        -/
        exact this t
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.81, u_2} T
          C : HomologicalComplex (CategoryTheory.Functor T V) c
          t : T
          i j : ι
          h : Not (c.Rel i j)
          this : ∀ (x : T), Eq ((C.d i j).app x) (CategoryTheory.NatTrans.app 0 x)
          ⊢ Eq ((fun i j => (C.d i j).app t) i j) 0
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      shape := fun i j h => by
        have := C.shape _ _ h
        rw [NatTrans.ext_iff, funext_iff] at this
        exact this t }
  map h :=
    { f := fun i => (C.X i).map h
      comm' := fun _ _ _ => NatTrans.naturality _ _ }
  map_id t := by
    /-
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      ι : Type u_1
      c : ComplexShape ι
      T : Type u_2
      inst✝ : CategoryTheory.Category.{?u.81, u_2} T
      C : HomologicalComplex (CategoryTheory.Functor T V) c
      t : T
      ⊢ Eq ({ obj := fun t => { X := fun i => (C.X i).obj t, d := fun i j => (C.d i  …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      ι : Type u_1
      c : ComplexShape ι
      T : Type u_2
      inst✝ : CategoryTheory.Category.{?u.81, u_2} T
      C : HomologicalComplex (CategoryTheory.Functor T V) c
      t : T
      i : ι
      ⊢ Eq (({ obj := fun t => { X := fun i => (C.X i).obj t, d := fun i j => (C.d i …
    -/
    dsimp
    /-
      case h
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      ι : Type u_1
      c : ComplexShape ι
      T : Type u_2
      inst✝ : CategoryTheory.Category.{?u.81, u_2} T
      C : HomologicalComplex (CategoryTheory.Functor T V) c
      t : T
      i : ι
      ⊢ Eq ((C.X i).map (CategoryTheory.CategoryStruct.id t)) (CategoryTheory.Catego …
    -/
    rw [(C.X i).map_id]
    /-
      🎉 no goals
    -/
  map_comp h₁ h₂ := by
    /-
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      ι : Type u_1
      c : ComplexShape ι
      T : Type u_2
      inst✝ : CategoryTheory.Category.{?u.81, u_2} T
      C : HomologicalComplex (CategoryTheory.Functor T V) c
      X✝ Y✝ Z✝ : T
      h₁ : Quiver.Hom X✝ Y✝
      h₂ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun t => { X := fun i => (C.X i).obj t, d := fun i j => (C.d i  …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      ι : Type u_1
      c : ComplexShape ι
      T : Type u_2
      inst✝ : CategoryTheory.Category.{?u.81, u_2} T
      C : HomologicalComplex (CategoryTheory.Functor T V) c
      X✝ Y✝ Z✝ : T
      h₁ : Quiver.Hom X✝ Y✝
      h₂ : Quiver.Hom Y✝ Z✝
      i : ι
      ⊢ Eq (({ obj := fun t => { X := fun i => (C.X i).obj t, d := fun i j => (C.d i …
    -/
    dsimp
    /-
      case h
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      ι : Type u_1
      c : ComplexShape ι
      T : Type u_2
      inst✝ : CategoryTheory.Category.{?u.81, u_2} T
      C : HomologicalComplex (CategoryTheory.Functor T V) c
      X✝ Y✝ Z✝ : T
      h₁ : Quiver.Hom X✝ Y✝
      h₂ : Quiver.Hom Y✝ Z✝
      i : ι
      ⊢ Eq ((C.X i).map (CategoryTheory.CategoryStruct.comp h₁ h₂)) (CategoryTheory. …
    -/
    rw [Functor.map_comp]
    /-
      🎉 no goals
    -/

-- TODO in fact, this is an equivalence of categories.

/-- The functorial version of `HomologicalComplex.asFunctor`. -/
@[simps]
def complexOfFunctorsToFunctorToComplex {T : Type*} [Category T] :
    HomologicalComplex (T ⥤ V) c ⥤ T ⥤ HomologicalComplex V c where
  obj C := C.asFunctor
  map f :=
    { app := fun t =>
        { f := fun i => (f.f i).app t
          comm' := fun i j _ => NatTrans.congr_app (f.comm i j) t }
      naturality := fun t t' g => by
        /-
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.3849, u_2} T
          X✝ Y✝ : HomologicalComplex (CategoryTheory.Functor T V) c
          f : Quiver.Hom X✝ Y✝
          t t' : T
          g : Quiver.Hom t t'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun C => C.asFunctor) X✝).map g) ( …
        -/
        ext i
        /-
          case h
          V : Type u
          inst✝² : CategoryTheory.Category.{v, u} V
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
          ι : Type u_1
          c : ComplexShape ι
          T : Type u_2
          inst✝ : CategoryTheory.Category.{?u.3849, u_2} T
          X✝ Y✝ : HomologicalComplex (CategoryTheory.Functor T V) c
          f : Quiver.Hom X✝ Y✝
          t t' : T
          g : Quiver.Hom t t'
          i : ι
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((fun C => C.asFunctor) X✝).map g)  …
        -/
        exact (f.f i).naturality g }
        /-
          🎉 no goals
        -/


