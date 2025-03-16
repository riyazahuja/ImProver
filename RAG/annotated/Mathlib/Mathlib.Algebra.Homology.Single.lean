/-- The functor `V ⥤ HomologicalComplex V c` creating a chain complex supported in a single degree.
-/
noncomputable def single (j : ι) : V ⥤ HomologicalComplex V c where
  obj A :=
    { X := fun i => if i = j then A else 0
      d := fun _ _ => 0 }
  map f :=
                                                  /-
                                                    V : Type u
                                                    inst✝³ : CategoryTheory.Category.{v, u} V
                                                    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                                    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                                                    ι : Type u_1
                                                    inst✝ : DecidableEq ι
                                                    c : ComplexShape ι
                                                    j : ι
                                                    X✝ Y✝ : V
                                                    f : Quiver.Hom X✝ Y✝
                                                    i : ι
                                                    h : Eq i j
                                                    ⊢ Eq (((fun A => { X := fun i => ite (Eq i j) A 0, d := fun x x_1 => 0, shape  …
                                                  -/
    { f := fun i => if h : i = j then eqToHom (by dsimp; rw [if_pos h]) ≫ f ≫
                                                         /-
                                                           🎉 no goals
                                                         -/
                          /-
                            V : Type u
                            inst✝³ : CategoryTheory.Category.{v, u} V
                            inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                            inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                            ι : Type u_1
                            inst✝ : DecidableEq ι
                            c : ComplexShape ι
                            j : ι
                            X✝ Y✝ : V
                            f : Quiver.Hom X✝ Y✝
                            i : ι
                            h : Eq i j
                            ⊢ Eq Y✝ (((fun A => { X := fun i => ite (Eq i j) A 0, d := fun x x_1 => 0, sha …
                          -/
              eqToHom (by dsimp; rw [if_pos h]) else 0 }
                                 /-
                                   🎉 no goals
                                 -/
  map_id A := by
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      j : ι
      A : V
      ⊢ Eq ({ obj := fun A => { X := fun i => ite (Eq i j) A 0, d := fun x x_1 => 0, …
    -/
    ext
    /-
      case h
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      j : ι
      A : V
      i✝ : ι
      ⊢ Eq (({ obj := fun A => { X := fun i => ite (Eq i j) A 0, d := fun x x_1 => 0 …
    -/
    dsimp
    /-
      case h
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      j : ι
      A : V
      i✝ : ι
      ⊢ Eq (dite (Eq i✝ j) (fun h => CategoryTheory.CategoryStruct.comp (CategoryThe …
    -/
    split_ifs with h
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        A : V
        i✝ : ι
        h : Eq i✝ j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
    · subst h
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        A : V
        i✝ : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp
      /-
        🎉 no goals
      -/
    · #adaptation_note /-- after nightly-2024-03-07, the previous sensible proof
      `rw [if_neg h]; simp` fails with "motive not type correct".
      The following is horrible. -/
      /-
        case neg
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        A : V
        i✝ : ι
        h : Not (Eq i✝ j)
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.id (ite (Eq i✝ j) A 0))
      -/
      convert (id_zero (C := V)).symm
      /-
        case h.e'_1.h.e'_3
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        A : V
        i✝ : ι
        h : Not (Eq i✝ j)
        ⊢ Eq (ite (Eq i✝ j) A 0) 0
      -/
      all_goals simp [if_neg h]
      /-
        🎉 no goals
      -/
  map_comp f g := by
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      j : ι
      X✝ Y✝ Z✝ : V
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun A => { X := fun i => ite (Eq i j) A 0, d := fun x x_1 => 0, …
    -/
    ext
    /-
      case h
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      j : ι
      X✝ Y✝ Z✝ : V
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      i✝ : ι
      ⊢ Eq (({ obj := fun A => { X := fun i => ite (Eq i j) A 0, d := fun x x_1 => 0 …
    -/
    dsimp
    /-
      case h
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      j : ι
      X✝ Y✝ Z✝ : V
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      i✝ : ι
      ⊢ Eq (dite (Eq i✝ j) (fun h => CategoryTheory.CategoryStruct.comp (CategoryThe …
    -/
    split_ifs with h
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        X✝ Y✝ Z✝ : V
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        i✝ : ι
        h : Eq i✝ j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
    · subst h
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        X✝ Y✝ Z✝ : V
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        i✝ : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        X✝ Y✝ Z✝ : V
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        i✝ : ι
        h : Not (Eq i✝ j)
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp 0 0)
      -/
    · simp
      /-
        🎉 no goals
      -/


@[simp]
lemma single_obj_X_self (j : ι) (A : V) :
    ((single V c j).obj A).X j = A := if_pos rfl


lemma isZero_single_obj_X (j : ι) (A : V) (i : ι) (hi : i ≠ j) :
    IsZero (((single V c j).obj A).X i) := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    j : ι
    A : V
    i : ι
    hi : Ne i j
    ⊢ CategoryTheory.Limits.IsZero (((HomologicalComplex.single V c j).obj A).X i)
  -/
  dsimp [single]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    j : ι
    A : V
    i : ι
    hi : Ne i j
    ⊢ CategoryTheory.Limits.IsZero (ite (Eq i j) A 0)
  -/
  rw [if_neg hi]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    j : ι
    A : V
    i : ι
    hi : Ne i j
    ⊢ CategoryTheory.Limits.IsZero 0
  -/
  exact Limits.isZero_zero V
  /-
    🎉 no goals
  -/


/-- The object in degree `i` of `(single V c h).obj A` is just `A` when `i = j`. -/
noncomputable def singleObjXIsoOfEq (j : ι) (A : V) (i : ι) (hi : i = j) :
    ((single V c j).obj A).X i ≅ A :=
              /-
                V : Type u
                inst✝³ : CategoryTheory.Category.{v, u} V
                inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                ι : Type u_1
                inst✝ : DecidableEq ι
                c : ComplexShape ι
                j : ι
                A : V
                i : ι
                hi : Eq i j
                ⊢ Eq (((HomologicalComplex.single V c j).obj A).X i) A
              -/
  eqToIso (by subst hi; simp [single])
                        /-
                          🎉 no goals
                        -/


/-- The object in degree `j` of `(single V c h).obj A` is just `A`. -/
noncomputable def singleObjXSelf (j : ι) (A : V) : ((single V c j).obj A).X j ≅ A :=
  singleObjXIsoOfEq c j A j rfl


@[simp]
lemma single_obj_d (j : ι) (A : V) (k l : ι) :
    ((single V c j).obj A).d k l = 0 := rfl


@[reassoc]
theorem single_map_f_self (j : ι) {A B : V} (f : A ⟶ B) :
    ((single V c j).map f).f j = (singleObjXSelf c j A).hom ≫
      f ≫ (singleObjXSelf c j B).inv := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    j : ι
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (((HomologicalComplex.single V c j).map f).f j) (CategoryTheory.CategoryS …
  -/
  dsimp [single]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    j : ι
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (dite (Eq j j) (fun h => CategoryTheory.CategoryStruct.comp (CategoryTheo …
  -/
  rw [dif_pos rfl]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    j : ι
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The natural isomorphism `single V c j ⋙ eval V c j ≅ 𝟭 V`. -/
@[simps!]
noncomputable def singleCompEvalIsoSelf (j : ι) : single V c j ⋙ eval V c j ≅ 𝟭 V :=
                                                              /-
                                                                V : Type u
                                                                inst✝³ : CategoryTheory.Category.{v, u} V
                                                                inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                                                inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                                                                ι : Type u_1
                                                                inst✝ : DecidableEq ι
                                                                c : ComplexShape ι
                                                                j : ι
                                                                A B : V
                                                                f : Quiver.Hom A B
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.single V c j).c …
                                                              -/
  NatIso.ofComponents (singleObjXSelf c j) (fun {A B} f => by simp [single_map_f_self])
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma isZero_single_comp_eval (j i : ι) (hi : i ≠ j) : IsZero (single V c j ⋙ eval V c i) :=
  Functor.isZero _ (fun _ ↦ isZero_single_obj_X c _ _ _ hi)


@[ext]
lemma from_single_hom_ext {K : HomologicalComplex V c} {j : ι} {A : V}
    {f g : (single V c j).obj A ⟶ K} (hfg : f.f j = g.f j) : f = g := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    f g : Quiver.Hom ((HomologicalComplex.single V c j).obj A) K
    hfg : Eq (f.f j) (g.f j)
    ⊢ Eq f g
  -/
  ext i
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    f g : Quiver.Hom ((HomologicalComplex.single V c j).obj A) K
    hfg : Eq (f.f j) (g.f j)
    i : ι
    ⊢ Eq (f.f i) (g.f i)
  -/
  by_cases h : i = j
    /-
      case pos
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      f g : Quiver.Hom ((HomologicalComplex.single V c j).obj A) K
      hfg : Eq (f.f j) (g.f j)
      i : ι
      h : Eq i j
      ⊢ Eq (f.f i) (g.f i)
    -/
  · subst h
    /-
      case pos
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      A : V
      i : ι
      f g : Quiver.Hom ((HomologicalComplex.single V c i).obj A) K
      hfg : Eq (f.f i) (g.f i)
      ⊢ Eq (f.f i) (g.f i)
    -/
    exact hfg
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      f g : Quiver.Hom ((HomologicalComplex.single V c j).obj A) K
      hfg : Eq (f.f j) (g.f j)
      i : ι
      h : Not (Eq i j)
      ⊢ Eq (f.f i) (g.f i)
    -/
  · apply (isZero_single_obj_X c j A i h).eq_of_src
    /-
      🎉 no goals
    -/


@[ext]
lemma to_single_hom_ext {K : HomologicalComplex V c} {j : ι} {A : V}
    {f g : K ⟶ (single V c j).obj A} (hfg : f.f j = g.f j) : f = g := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    f g : Quiver.Hom K ((HomologicalComplex.single V c j).obj A)
    hfg : Eq (f.f j) (g.f j)
    ⊢ Eq f g
  -/
  ext i
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    f g : Quiver.Hom K ((HomologicalComplex.single V c j).obj A)
    hfg : Eq (f.f j) (g.f j)
    i : ι
    ⊢ Eq (f.f i) (g.f i)
  -/
  by_cases h : i = j
    /-
      case pos
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      f g : Quiver.Hom K ((HomologicalComplex.single V c j).obj A)
      hfg : Eq (f.f j) (g.f j)
      i : ι
      h : Eq i j
      ⊢ Eq (f.f i) (g.f i)
    -/
  · subst h
    /-
      case pos
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      A : V
      i : ι
      f g : Quiver.Hom K ((HomologicalComplex.single V c i).obj A)
      hfg : Eq (f.f i) (g.f i)
      ⊢ Eq (f.f i) (g.f i)
    -/
    exact hfg
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      f g : Quiver.Hom K ((HomologicalComplex.single V c j).obj A)
      hfg : Eq (f.f j) (g.f j)
      i : ι
      h : Not (Eq i j)
      ⊢ Eq (f.f i) (g.f i)
    -/
  · apply (isZero_single_obj_X c j A i h).eq_of_tgt
    /-
      🎉 no goals
    -/


instance (j : ι) : (single V c j).Faithful where
  map_injective {A B f g} w := by
    rw [← cancel_mono (singleObjXSelf c j B).inv,
      ← cancel_epi (singleObjXSelf c j A).hom, ← single_map_f_self,
      ← single_map_f_self, w]


instance (j : ι) : (single V c j).Full where
  map_surjective {A B} f :=
    ⟨(singleObjXSelf c j A).inv ≫ f.f j ≫ (singleObjXSelf c j B).hom, by
      /-
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        A B : V
        f : Quiver.Hom ((HomologicalComplex.single V c j).obj A) ((HomologicalComplex. …
        ⊢ Eq ((HomologicalComplex.single V c j).map (CategoryTheory.CategoryStruct.com …
      -/
      ext
      /-
        case hfg
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        j : ι
        A B : V
        f : Quiver.Hom ((HomologicalComplex.single V c j).obj A) ((HomologicalComplex. …
        ⊢ Eq (((HomologicalComplex.single V c j).map (CategoryTheory.CategoryStruct.co …
      -/
      simp [single_map_f_self]⟩
      /-
        🎉 no goals
      -/


/-- Constructor for morphisms to a single homological complex. -/
noncomputable def mkHomToSingle {K : HomologicalComplex V c} {j : ι} {A : V} (φ : K.X j ⟶ A)
    (hφ : ∀ (i : ι), c.Rel i j → K.d i j ≫ φ = 0) :
    K ⟶ (single V c j).obj A where
  f i :=
    if hi : i = j
      then (K.XIsoOfEq hi).hom ≫ φ ≫ (singleObjXIsoOfEq c j A i hi).inv
      else 0
  comm' i k hik := by
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      φ : Quiver.Hom (K.X j) A
      hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
      i k : ι
      hik : c.Rel i k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => dite (Eq i j) (fun hi => C …
    -/
    dsimp
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      φ : Quiver.Hom (K.X j) A
      hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
      i k : ι
      hik : c.Rel i k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i j) (fun hi => CategoryThe …
    -/
    rw [comp_zero]
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      φ : Quiver.Hom (K.X j) A
      hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
      i k : ι
      hik : c.Rel i k
      ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i k) (dite (Eq k j) (fun hi => …
    -/
    split_ifs with hk
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        K : HomologicalComplex V c
        j : ι
        A : V
        φ : Quiver.Hom (K.X j) A
        hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
        i k : ι
        hik : c.Rel i k
        hk : Eq k j
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i k) (CategoryTheory.CategoryS …
      -/
    · subst hk
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        K : HomologicalComplex V c
        A : V
        i k : ι
        hik : c.Rel i k
        φ : Quiver.Hom (K.X k) A
        hφ : ∀ (i : ι), c.Rel i k → Eq (CategoryTheory.CategoryStruct.comp (K.d i k) φ …
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i k) (CategoryTheory.CategoryS …
      -/
      simp only [XIsoOfEq_rfl, Iso.refl_hom, id_comp, reassoc_of% hφ i hik, zero_comp]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        K : HomologicalComplex V c
        j : ι
        A : V
        φ : Quiver.Hom (K.X j) A
        hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
        i k : ι
        hik : c.Rel i k
        hk : Not (Eq k j)
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i k) 0)
      -/
    · apply (isZero_single_obj_X c j A k hk).eq_of_tgt
      /-
        🎉 no goals
      -/


@[simp]
lemma mkHomToSingle_f {K : HomologicalComplex V c} {j : ι} {A : V} (φ : K.X j ⟶ A)
    (hφ : ∀ (i : ι), c.Rel i j → K.d i j ≫ φ = 0) :
    (mkHomToSingle φ hφ).f j = φ ≫ (singleObjXSelf c j A).inv := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    φ : Quiver.Hom (K.X j) A
    hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
    ⊢ Eq ((HomologicalComplex.mkHomToSingle φ hφ).f j) (CategoryTheory.CategoryStr …
  -/
  dsimp [mkHomToSingle]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    φ : Quiver.Hom (K.X j) A
    hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
    ⊢ Eq (dite (Eq j j) (fun hi => CategoryTheory.CategoryStruct.comp (CategoryThe …
  -/
  rw [dif_pos rfl, id_comp]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    φ : Quiver.Hom (K.X j) A
    hφ : ∀ (i : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (K.d i j) φ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (HomologicalComplex.singleObjXIsoOf …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Constructor for morphisms from a single homological complex. -/
noncomputable def mkHomFromSingle {K : HomologicalComplex V c} {j : ι} {A : V} (φ : A ⟶ K.X j)
    (hφ : ∀ (k : ι), c.Rel j k → φ ≫ K.d j k = 0) :
    (single V c j).obj A ⟶ K where
  f i :=
    if hi : i = j
      then (singleObjXIsoOfEq c j A i hi).hom ≫ φ ≫ (K.XIsoOfEq hi).inv
      else 0
  comm' i k hik := by
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      φ : Quiver.Hom A (K.X j)
      hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
      i k : ι
      hik : c.Rel i k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => dite (Eq i j) (fun hi => C …
    -/
    dsimp
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      φ : Quiver.Hom A (K.X j)
      hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
      i k : ι
      hik : c.Rel i k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i j) (fun hi => CategoryThe …
    -/
    rw [zero_comp]
    /-
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝¹ : CategoryTheory.Limits.HasZeroObject V
      ι : Type u_1
      inst✝ : DecidableEq ι
      c : ComplexShape ι
      K : HomologicalComplex V c
      j : ι
      A : V
      φ : Quiver.Hom A (K.X j)
      hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
      i k : ι
      hik : c.Rel i k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i j) (fun hi => CategoryThe …
    -/
    split_ifs with hi
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        K : HomologicalComplex V c
        j : ι
        A : V
        φ : Quiver.Hom A (K.X j)
        hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
        i k : ι
        hik : c.Rel i k
        hi : Eq i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · subst hi
      /-
        case pos
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        K : HomologicalComplex V c
        A : V
        i k : ι
        hik : c.Rel i k
        φ : Quiver.Hom A (K.X i)
        hφ : ∀ (k : ι), c.Rel i k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d i k) …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [XIsoOfEq_rfl, Iso.refl_inv, comp_id, assoc, hφ k hik, comp_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u
        inst✝³ : CategoryTheory.Category.{v, u} V
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
        inst✝¹ : CategoryTheory.Limits.HasZeroObject V
        ι : Type u_1
        inst✝ : DecidableEq ι
        c : ComplexShape ι
        K : HomologicalComplex V c
        j : ι
        A : V
        φ : Quiver.Hom A (K.X j)
        hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
        i k : ι
        hik : c.Rel i k
        hi : Not (Eq i j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (K.d i k)) 0
      -/
    · apply (isZero_single_obj_X c j A i hi).eq_of_src
      /-
        🎉 no goals
      -/


@[simp]
lemma mkHomFromSingle_f {K : HomologicalComplex V c} {j : ι} {A : V} (φ : A ⟶ K.X j)
    (hφ : ∀ (k : ι), c.Rel j k → φ ≫ K.d j k = 0) :
    (mkHomFromSingle φ hφ).f j = (singleObjXSelf c j A).hom ≫ φ := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    φ : Quiver.Hom A (K.X j)
    hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
    ⊢ Eq ((HomologicalComplex.mkHomFromSingle φ hφ).f j) (CategoryTheory.CategoryS …
  -/
  dsimp [mkHomFromSingle]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    φ : Quiver.Hom A (K.X j)
    hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
    ⊢ Eq (dite (Eq j j) (fun hi => CategoryTheory.CategoryStruct.comp (Homological …
  -/
  rw [dif_pos rfl, comp_id]
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝¹ : CategoryTheory.Limits.HasZeroObject V
    ι : Type u_1
    inst✝ : DecidableEq ι
    c : ComplexShape ι
    K : HomologicalComplex V c
    j : ι
    A : V
    φ : Quiver.Hom A (K.X j)
    hφ : ∀ (k : ι), c.Rel j k → Eq (CategoryTheory.CategoryStruct.comp φ (K.d j k) …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.singleObjXIsoOfEq …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance (j : ι) : (single V c j).PreservesZeroMorphisms where


/-- The functor `V ⥤ ChainComplex V ℕ` creating a chain complex supported in degree zero. -/
noncomputable abbrev single₀ : V ⥤ ChainComplex V ℕ :=
  HomologicalComplex.single V (ComplexShape.down ℕ) 0


@[simp]
lemma single₀_obj_zero (A : V) :
    ((single₀ V).obj A).X 0 = A := rfl


@[simp]
lemma single₀_map_f_zero {A B : V} (f : A ⟶ B) :
    ((single₀ V).map f).f 0 = f := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (((ChainComplex.single₀ V).map f).f 0) f
  -/
  rw [HomologicalComplex.single_map_f_self]
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.singleObjXSelf (C …
  -/
  dsimp [HomologicalComplex.singleObjXSelf, HomologicalComplex.singleObjXIsoOfEq]
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id A)  …
  -/
  rw [comp_id, id_comp]
  /-
    🎉 no goals
  -/



@[simp]
lemma single₀ObjXSelf (X : V) :
    HomologicalComplex.singleObjXSelf (ComplexShape.down ℕ) 0 X = Iso.refl _ := rfl


/-- Morphisms from an `ℕ`-indexed chain complex `C`
to a single object chain complex with `X` concentrated in degree 0
are the same as morphisms `f : C.X 0 ⟶ X` such that `C.d 1 0 ≫ f = 0`.
-/
@[simps apply_coe]
noncomputable def toSingle₀Equiv (C : ChainComplex V ℕ) (X : V) :
    (C ⟶ (single₀ V).obj X) ≃ { f : C.X 0 ⟶ X // C.d 1 0 ≫ f = 0 } where
                        /-
                          V : Type u
                          inst✝² : CategoryTheory.Category.{v, u} V
                          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                          inst✝ : CategoryTheory.Limits.HasZeroObject V
                          C : ChainComplex V Nat
                          X : V
                          φ : Quiver.Hom C ((ChainComplex.single₀ V).obj X)
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) (φ.f 0)) 0
                        -/
  toFun φ := ⟨φ.f 0, by rw [← φ.comm 1 0, HomologicalComplex.single_obj_d, comp_zero]⟩
                        /-
                          🎉 no goals
                        -/
  invFun f := HomologicalComplex.mkHomToSingle f.1 (fun i hi => by
    /-
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝ : CategoryTheory.Limits.HasZeroObject V
      C : ChainComplex V Nat
      X : V
      f : Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
      i : Nat
      hi : (ComplexShape.down Nat).Rel i 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i 0) ↑f) 0
    -/
    obtain rfl : i = 1 := by simpa using hi.symm
    /-
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝ : CategoryTheory.Limits.HasZeroObject V
      C : ChainComplex V Nat
      X : V
      f : Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
      hi : (ComplexShape.down Nat).Rel 1 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) ↑f) 0
    -/
    exact f.2)
    /-
      🎉 no goals
    -/
                   /-
                     V : Type u
                     inst✝² : CategoryTheory.Category.{v, u} V
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                     inst✝ : CategoryTheory.Limits.HasZeroObject V
                     C : ChainComplex V Nat
                     X : V
                     φ : Quiver.Hom C ((ChainComplex.single₀ V).obj X)
                     ⊢ Eq ((fun f => HomologicalComplex.mkHomToSingle ↑f ⋯) ((fun φ => ⟨φ.f 0, ⋯⟩)  …
                   -/
  left_inv φ := by aesop_cat
                   /-
                     🎉 no goals
                   -/
                    /-
                      V : Type u
                      inst✝² : CategoryTheory.Category.{v, u} V
                      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                      inst✝ : CategoryTheory.Limits.HasZeroObject V
                      C : ChainComplex V Nat
                      X : V
                      f : Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
                      ⊢ Eq ((fun φ => ⟨φ.f 0, ⋯⟩) ((fun f => HomologicalComplex.mkHomToSingle ↑f ⋯)  …
                    -/
  right_inv f := by aesop_cat
                    /-
                      🎉 no goals
                    -/


@[simp]
lemma toSingle₀Equiv_symm_apply_f_zero {C : ChainComplex V ℕ} {X : V}
    (f : C.X 0 ⟶ X) (hf : C.d 1 0 ≫ f = 0) :
    ((toSingle₀Equiv C X).symm ⟨f, hf⟩).f 0 = f := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    C : ChainComplex V Nat
    X : V
    f : Quiver.Hom (C.X 0) X
    hf : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
    ⊢ Eq (((C.toSingle₀Equiv X).symm ⟨f, hf⟩).f 0) f
  -/
  simp [toSingle₀Equiv]
  /-
    🎉 no goals
  -/


/-- Morphisms from a single object chain complex with `X` concentrated in degree 0
to an `ℕ`-indexed chain complex `C` are the same as morphisms `f : X → C.X 0`.
-/
@[simps apply]
noncomputable def fromSingle₀Equiv (C : ChainComplex V ℕ) (X : V) :
    ((single₀ V).obj X ⟶ C) ≃ (X ⟶ C.X 0) where
  toFun f := f.f 0
                                                                   /-
                                                                     V : Type u
                                                                     inst✝² : CategoryTheory.Category.{v, u} V
                                                                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                     inst✝ : CategoryTheory.Limits.HasZeroObject V
                                                                     C : ChainComplex V Nat
                                                                     X : V
                                                                     f : Quiver.Hom X (C.X 0)
                                                                     i : Nat
                                                                     hi : (ComplexShape.down Nat).Rel 0 i
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 i)) 0
                                                                   -/
  invFun f := HomologicalComplex.mkHomFromSingle f (fun i hi => by simp at hi)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                 /-
                   V : Type u
                   inst✝² : CategoryTheory.Category.{v, u} V
                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                   inst✝ : CategoryTheory.Limits.HasZeroObject V
                   C : ChainComplex V Nat
                   X : V
                   ⊢ Function.LeftInverse (fun f => HomologicalComplex.mkHomFromSingle f ⋯) fun f …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    V : Type u
                    inst✝² : CategoryTheory.Category.{v, u} V
                    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                    inst✝ : CategoryTheory.Limits.HasZeroObject V
                    C : ChainComplex V Nat
                    X : V
                    ⊢ Function.RightInverse (fun f => HomologicalComplex.mkHomFromSingle f ⋯) fun  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma fromSingle₀Equiv_symm_apply_f_zero
    {C : ChainComplex V ℕ} {X : V} (f : X ⟶ C.X 0) :
    ((fromSingle₀Equiv C X).symm f).f 0 = f := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    C : ChainComplex V Nat
    X : V
    f : Quiver.Hom X (C.X 0)
    ⊢ Eq (((C.fromSingle₀Equiv X).symm f).f 0) f
  -/
  simp [fromSingle₀Equiv]
  /-
    🎉 no goals
  -/


@[simp]
lemma fromSingle₀Equiv_symm_apply_f_succ
    {C : ChainComplex V ℕ} {X : V} (f : X ⟶ C.X 0) (n : ℕ) :
    ((fromSingle₀Equiv C X).symm f).f (n + 1) = 0 := rfl


/-- The functor `V ⥤ CochainComplex V ℕ` creating a cochain complex supported in degree zero. -/
noncomputable abbrev single₀ : V ⥤ CochainComplex V ℕ :=
  HomologicalComplex.single V (ComplexShape.up ℕ) 0


@[simp]
lemma single₀_map_f_zero {A B : V} (f : A ⟶ B) :
    ((single₀ V).map f).f 0 = f := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (((CochainComplex.single₀ V).map f).f 0) f
  -/
  rw [HomologicalComplex.single_map_f_self]
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.singleObjXSelf (C …
  -/
  dsimp [HomologicalComplex.singleObjXSelf, HomologicalComplex.singleObjXIsoOfEq]
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    A B : V
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id A)  …
  -/
  rw [comp_id, id_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma single₀ObjXSelf (X : V) :
    HomologicalComplex.singleObjXSelf (ComplexShape.up ℕ) 0 X = Iso.refl _ := rfl


/-- Morphisms from a single object cochain complex with `X` concentrated in degree 0
to an `ℕ`-indexed cochain complex `C`
are the same as morphisms `f : X ⟶ C.X 0` such that `f ≫ C.d 0 1 = 0`. -/
@[simps apply_coe]
noncomputable def fromSingle₀Equiv (C : CochainComplex V ℕ) (X : V) :
    ((single₀ V).obj X ⟶ C) ≃ { f : X ⟶ C.X 0 // f ≫ C.d 0 1 = 0 } where
                        /-
                          V : Type u
                          inst✝² : CategoryTheory.Category.{v, u} V
                          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                          inst✝ : CategoryTheory.Limits.HasZeroObject V
                          C : CochainComplex V Nat
                          X : V
                          φ : Quiver.Hom ((CochainComplex.single₀ V).obj X) C
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.f 0) (C.d 0 1)) 0
                        -/
  toFun φ := ⟨φ.f 0, by rw [φ.comm 0 1, HomologicalComplex.single_obj_d, zero_comp]⟩
                        /-
                          🎉 no goals
                        -/
  invFun f := HomologicalComplex.mkHomFromSingle f.1 (fun i hi => by
    /-
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝ : CategoryTheory.Limits.HasZeroObject V
      C : CochainComplex V Nat
      X : V
      f : Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      i : Nat
      hi : (ComplexShape.up Nat).Rel 0 i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑f) (C.d 0 i)) 0
    -/
    obtain rfl : i = 1 := by simpa using hi.symm
    /-
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      inst✝ : CategoryTheory.Limits.HasZeroObject V
      C : CochainComplex V Nat
      X : V
      f : Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      hi : (ComplexShape.up Nat).Rel 0 1
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑f) (C.d 0 1)) 0
    -/
    exact f.2)
    /-
      🎉 no goals
    -/
                   /-
                     V : Type u
                     inst✝² : CategoryTheory.Category.{v, u} V
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                     inst✝ : CategoryTheory.Limits.HasZeroObject V
                     C : CochainComplex V Nat
                     X : V
                     φ : Quiver.Hom ((CochainComplex.single₀ V).obj X) C
                     ⊢ Eq ((fun f => HomologicalComplex.mkHomFromSingle ↑f ⋯) ((fun φ => ⟨φ.f 0, ⋯⟩ …
                   -/
  left_inv φ := by aesop_cat
                   /-
                     🎉 no goals
                   -/
                  /-
                    V : Type u
                    inst✝² : CategoryTheory.Category.{v, u} V
                    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                    inst✝ : CategoryTheory.Limits.HasZeroObject V
                    C : CochainComplex V Nat
                    X : V
                    ⊢ Function.RightInverse (fun f => HomologicalComplex.mkHomFromSingle ↑f ⋯) fun …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma fromSingle₀Equiv_symm_apply_f_zero {C : CochainComplex V ℕ} {X : V}
    (f : X ⟶ C.X 0) (hf : f ≫ C.d 0 1 = 0) :
    ((fromSingle₀Equiv C X).symm ⟨f, hf⟩).f 0 = f := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    C : CochainComplex V Nat
    X : V
    f : Quiver.Hom X (C.X 0)
    hf : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
    ⊢ Eq (((C.fromSingle₀Equiv X).symm ⟨f, hf⟩).f 0) f
  -/
  simp [fromSingle₀Equiv]
  /-
    🎉 no goals
  -/


/-- Morphisms to a single object cochain complex with `X` concentrated in degree 0
to an `ℕ`-indexed cochain complex `C` are the same as morphisms `f : C.X 0 ⟶ X`.
-/
@[simps apply]
noncomputable def toSingle₀Equiv (C : CochainComplex V ℕ) (X : V) :
    (C ⟶ (single₀ V).obj X) ≃ (C.X 0 ⟶ X) where
  toFun f := f.f 0
                                                                 /-
                                                                   V : Type u
                                                                   inst✝² : CategoryTheory.Category.{v, u} V
                                                                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                   inst✝ : CategoryTheory.Limits.HasZeroObject V
                                                                   C : CochainComplex V Nat
                                                                   X : V
                                                                   f : Quiver.Hom (C.X 0) X
                                                                   i : Nat
                                                                   hi : (ComplexShape.up Nat).Rel i 0
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i 0) f) 0
                                                                 -/
  invFun f := HomologicalComplex.mkHomToSingle f (fun i hi => by simp at hi)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                 /-
                   V : Type u
                   inst✝² : CategoryTheory.Category.{v, u} V
                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                   inst✝ : CategoryTheory.Limits.HasZeroObject V
                   C : CochainComplex V Nat
                   X : V
                   ⊢ Function.LeftInverse (fun f => HomologicalComplex.mkHomToSingle f ⋯) fun f = …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    V : Type u
                    inst✝² : CategoryTheory.Category.{v, u} V
                    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
                    inst✝ : CategoryTheory.Limits.HasZeroObject V
                    C : CochainComplex V Nat
                    X : V
                    ⊢ Function.RightInverse (fun f => HomologicalComplex.mkHomToSingle f ⋯) fun f  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma toSingle₀Equiv_symm_apply_f_zero
    {C : CochainComplex V ℕ} {X : V} (f : C.X 0 ⟶ X) :
    ((toSingle₀Equiv C X).symm f).f 0 = f := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    C : CochainComplex V Nat
    X : V
    f : Quiver.Hom (C.X 0) X
    ⊢ Eq (((C.toSingle₀Equiv X).symm f).f 0) f
  -/
  simp [toSingle₀Equiv]
  /-
    🎉 no goals
  -/


@[simp]
lemma toSingle₀Equiv_symm_apply_f_succ
    {C : CochainComplex V ℕ} {X : V} (f : C.X 0 ⟶ X) (n : ℕ) :
    ((toSingle₀Equiv C X).symm f).f (n + 1) = 0 := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    C : CochainComplex V Nat
    X : V
    f : Quiver.Hom (C.X 0) X
    n : Nat
    ⊢ Eq (((C.toSingle₀Equiv X).symm f).f (HAdd.hAdd n 1)) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


