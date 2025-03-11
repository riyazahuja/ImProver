/-- Since `eqToHom` only preserves the fact that `X.X i = X.X j` but not `i = j`, this definition
is used to aid the simplifier. -/
abbrev objEqToHom {i j : β} (h : i = j) :
    X.obj i ⟶ X.obj j :=
  eqToHom (congr_arg X.obj h)


@[simp]
theorem objEqToHom_refl (i : β) : X.objEqToHom (refl i) = 𝟙 _ :=
  rfl


@[reassoc (attr := simp)]
theorem objEqToHom_d {x y : β} (h : x = y) :
                                                      /-
                                                        β : Type u_1
                                                        inst✝² : AddCommGroup β
                                                        b : β
                                                        V : Type u_2
                                                        inst✝¹ : CategoryTheory.Category.{?u.1280, u_2} V
                                                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                        X : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithShif …
                                                        x y : β
                                                        h : Eq x y
                                                        ⊢ Eq ((fun b_1 => HAdd.hAdd b_1 (HSMul.hSMul { as := 1 }.as b)) x) ((fun b_1 = …
                                                      -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
    X.objEqToHom h ≫ X.d y = X.d x ≫ X.objEqToHom (by cases h; rfl) := by cases h; dsimp; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[reassoc (attr := simp)]
theorem d_squared_apply {x : β} : X.d x ≫ X.d _ = 0 := congr_fun X.d_squared _


@[reassoc (attr := simp)]
theorem eqToHom_f' {X Y : DifferentialObject ℤ (GradedObjectWithShift b V)} (f : X ⟶ Y) {x y : β}
                                                                        /-
                                                                          β : Type u_1
                                                                          inst✝² : AddCommGroup β
                                                                          b : β
                                                                          V : Type u_2
                                                                          inst✝¹ : CategoryTheory.Category.{u_3, u_2} V
                                                                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                          X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithSh …
                                                                          f : Quiver.Hom X Y
                                                                          x y : β
                                                                          h : Eq x y
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.objEqToHom h) (f.f y)) (CategoryTh …
                                                                        -/
    (h : x = y) : X.objEqToHom h ≫ f.f y = f.f x ≫ Y.objEqToHom h := by cases h; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[reassoc (attr := simp, nolint simpNF)]
theorem d_eqToHom (X : HomologicalComplex V (ComplexShape.up' b)) {x y z : β} (h : y = z) :
                                                        /-
                                                          β : Type u_1
                                                          inst✝² : AddCommGroup β
                                                          b : β
                                                          V : Type u_2
                                                          inst✝¹ : CategoryTheory.Category.{u_3, u_2} V
                                                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                          X : HomologicalComplex V (ComplexShape.up' b)
                                                          x y z : β
                                                          h : Eq y z
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.d x y) (CategoryTheory.eqToHom ⋯)) …
                                                        -/
    X.d x y ≫ eqToHom (congr_arg X.X h) = X.d x z := by cases h; simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


open Classical in
set_option maxHeartbeats 400000 in
/-- The functor from differential graded objects to homological complexes.
-/
@[simps]
def dgoToHomologicalComplex :
    DifferentialObject ℤ (GradedObjectWithShift b V) ⥤
      HomologicalComplex V (ComplexShape.up' b) where
  obj X :=
    { X := fun i => X.obj i
      d := fun i j =>
                                                                                /-
                                                                                  β : Type u_1
                                                                                  inst✝² : AddCommGroup β
                                                                                  b : β
                                                                                  V : Type u_2
                                                                                  inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
                                                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                                  X : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithShif …
                                                                                  i j : β
                                                                                  h : Eq (HAdd.hAdd i b) j
                                                                                  ⊢ Eq (HAdd.hAdd i (HSMul.hSMul 1 b)) j
                                                                                -/
        if h : i + b = j then X.d i ≫ X.objEqToHom (show i + (1 : ℤ) • b = j by simp [h]) else 0
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                               /-
                                 β : Type u_1
                                 inst✝² : AddCommGroup β
                                 b : β
                                 V : Type u_2
                                 inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                 X : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithShif …
                                 i j : β
                                 w : Not ((ComplexShape.up' b).Rel i j)
                                 ⊢ Eq ((fun i j => dite (Eq (HAdd.hAdd i b) j) (fun h => CategoryTheory.Categor …
                               -/
      shape := fun i j w => by dsimp at w; convert dif_neg w
                                           /-
                                             🎉 no goals
                                           -/
      d_comp_d' := fun i j k hij hjk => by
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithShif …
          i j k : β
          hij : (ComplexShape.up' b).Rel i j
          hjk : (ComplexShape.up' b).Rel j k
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => dite (Eq (HAdd.hAdd i b) …
        -/
        dsimp at hij hjk; substs hij hjk
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithShif …
          i : β
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => dite (Eq (HAdd.hAdd i b) …
        -/
        simp }
        /-
          🎉 no goals
        -/
  map {X Y} f :=
    { f := f.f
      comm' := fun i j h => by
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithSh …
          f : Quiver.Hom X Y
          i j : β
          h : (ComplexShape.up' b).Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (((fun X => { X := fun i => X …
        -/
        dsimp at h ⊢
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithSh …
          f : Quiver.Hom X Y
          i j : β
          h : Eq (HAdd.hAdd i b) j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (dite (Eq (HAdd.hAdd i b) j)  …
        -/
        subst h
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithSh …
          f : Quiver.Hom X Y
          i : β
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (dite (Eq (HAdd.hAdd i b) (HA …
        -/
        simp only [dite_true, Category.assoc, eqToHom_f']
        -- Porting note: this `rw` used to be part of the `simp`.
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithSh …
          f : Quiver.Hom X Y
          i : β
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (CategoryTheory.CategoryStruc …
        -/
        have : f.f i ≫ Y.d i = X.d i ≫ f.f _ := (congr_fun f.comm i).symm
        /-
          β : Type u_1
          inst✝² : AddCommGroup β
          b : β
          V : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.7360, u_2} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectWithSh …
          f : Quiver.Hom X Y
          i : β
          this : Eq (CategoryTheory.CategoryStruct.comp (f.f i) (Y.d i)) (CategoryTheory …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (CategoryTheory.CategoryStruc …
        -/
        rw [reassoc_of% this] }
        /-
          🎉 no goals
        -/


/-- The functor from homological complexes to differential graded objects.
-/
@[simps]
def homologicalComplexToDGO :
    HomologicalComplex V (ComplexShape.up' b) ⥤
      DifferentialObject ℤ (GradedObjectWithShift b V) where
  obj X :=
    { obj := fun i => X.X i
      d := fun i => X.d i _ }
  map {X Y} f := { f := f.f }


/-- The unit isomorphism for `dgoEquivHomologicalComplex`.
-/
@[simps!]
def dgoEquivHomologicalComplexUnitIso :
    𝟭 (DifferentialObject ℤ (GradedObjectWithShift b V)) ≅
      dgoToHomologicalComplex b V ⋙ homologicalComplexToDGO b V :=
  /-
    β : Type u_1
    inst✝² : AddCommGroup β
    b : β
    V : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.183960, u_2} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    ⊢ ∀ {X Y : CategoryTheory.DifferentialObject Int (CategoryTheory.GradedObjectW …
  -/
  NatIso.ofComponents (fun X =>
  /-
    🎉 no goals
  -/
    { hom := { f := fun i => 𝟙 (X.obj i) }
      inv := { f := fun i => 𝟙 (X.obj i) } })


/-- The counit isomorphism for `dgoEquivHomologicalComplex`.
-/
@[simps!]
def dgoEquivHomologicalComplexCounitIso :
    homologicalComplexToDGO b V ⋙ dgoToHomologicalComplex b V ≅
      𝟭 (HomologicalComplex V (ComplexShape.up' b)) :=
  /-
    β : Type u_1
    inst✝² : AddCommGroup β
    b : β
    V : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.198842, u_2} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    ⊢ ∀ {X Y : HomologicalComplex V (ComplexShape.up' b)} (f : Quiver.Hom X Y), Eq …
  -/
  NatIso.ofComponents (fun X =>
  /-
    🎉 no goals
  -/
    { hom := { f := fun i => 𝟙 (X.X i) }
      inv := { f := fun i => 𝟙 (X.X i) } })


/-- The category of differential graded objects in `V` is equivalent
to the category of homological complexes in `V`.
-/
@[simps]
def dgoEquivHomologicalComplex :
    DifferentialObject ℤ (GradedObjectWithShift b V) ≌
      HomologicalComplex V (ComplexShape.up' b) where
  functor := dgoToHomologicalComplex b V
  inverse := homologicalComplexToDGO b V
  unitIso := dgoEquivHomologicalComplexUnitIso b V
  counitIso := dgoEquivHomologicalComplexCounitIso b V


