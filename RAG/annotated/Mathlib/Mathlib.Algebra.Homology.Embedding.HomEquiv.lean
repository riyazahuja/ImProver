/-- The condition on a morphism `K.restriction e ⟶ L` which allows to
extend it as a morphism `K ⟶ L.extend e`, see `Embedding.homEquiv`. -/
def HasLift (φ : K.restriction e ⟶ L) : Prop :=
  ∀ (j : ι) (_ : e.BoundaryGE j) (i' : ι')
    (_ : c'.Rel i' (e.f j)), K.d i' _ ≫ φ.f j = 0


open Classical in
/-- Auxiliary definition for `liftExtend`. -/
noncomputable def f (i' : ι') : K.X i' ⟶ (L.extend e).X i' :=
  if hi' : ∃ i, e.f i = i' then
    (K.restrictionXIso e hi'.choose_spec).inv ≫ φ.f hi'.choose ≫
      (L.extendXIso e hi'.choose_spec).inv
  else 0


lemma f_eq {i' : ι'} {i : ι} (hi : e.f i = i') :
    f φ i' = (K.restrictionXIso e hi).inv ≫ φ.f i ≫ (L.extendXIso e hi).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    i' : ι'
    i : ι
    hi : Eq (e.f i) i'
    ⊢ Eq (ComplexShape.Embedding.liftExtend.f φ i') (CategoryTheory.CategoryStruct …
  -/
  have hi' : ∃ k, e.f k = i' := ⟨i, hi⟩
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    i' : ι'
    i : ι
    hi : Eq (e.f i) i'
    hi' : Exists fun k => Eq (e.f k) i'
    ⊢ Eq (ComplexShape.Embedding.liftExtend.f φ i') (CategoryTheory.CategoryStruct …
  -/
  have : hi'.choose = i := e.injective_f (by rw [hi'.choose_spec, hi])
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    i' : ι'
    i : ι
    hi : Eq (e.f i) i'
    hi' : Exists fun k => Eq (e.f k) i'
    this : Eq hi'.choose i
    ⊢ Eq (ComplexShape.Embedding.liftExtend.f φ i') (CategoryTheory.CategoryStruct …
  -/
  dsimp [f]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    i' : ι'
    i : ι
    hi : Eq (e.f i) i'
    hi' : Exists fun k => Eq (e.f k) i'
    this : Eq hi'.choose i
    ⊢ Eq (dite (Exists fun i => Eq (e.f i) i') (fun hi' => CategoryTheory.Category …
  -/
  rw [dif_pos ⟨i, hi⟩]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    i' : ι'
    i : ι
    hi : Eq (e.f i) i'
    hi' : Exists fun k => Eq (e.f k) i'
    this : Eq hi'.choose i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.restrictionXIso e ⋯).inv (Category …
  -/
  subst this
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    i' : ι'
    hi' : Exists fun k => Eq (e.f k) i'
    hi : Eq (e.f hi'.choose) i'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.restrictionXIso e ⋯).inv (Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma comm (hφ : e.HasLift φ) (i' j' : ι') :
    f φ i' ≫ (L.extend e).d i' j' = K.d i' j' ≫ f φ j' := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    hφ : e.HasLift φ
    i' j' : ι'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.liftExtend.f  …
  -/
  by_cases hij' : c'.Rel i' j'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      L : HomologicalComplex C c
      inst✝ : e.IsRelIff
      φ : Quiver.Hom (K.restriction e) L
      hφ : e.HasLift φ
      i' j' : ι'
      hij' : c'.Rel i' j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.liftExtend.f  …
    -/
  · by_cases hi' : ∃ i, e.f i = i'
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        e : c.Embedding c'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        L : HomologicalComplex C c
        inst✝ : e.IsRelIff
        φ : Quiver.Hom (K.restriction e) L
        hφ : e.HasLift φ
        i' j' : ι'
        hij' : c'.Rel i' j'
        hi' : Exists fun i => Eq (e.f i) i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.liftExtend.f  …
      -/
    · obtain ⟨i, hi⟩ := hi'
      /-
        case pos.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        e : c.Embedding c'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        L : HomologicalComplex C c
        inst✝ : e.IsRelIff
        φ : Quiver.Hom (K.restriction e) L
        hφ : e.HasLift φ
        i' j' : ι'
        hij' : c'.Rel i' j'
        i : ι
        hi : Eq (e.f i) i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.liftExtend.f  …
      -/
      rw [f_eq φ hi]
      /-
        case pos.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        e : c.Embedding c'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        L : HomologicalComplex C c
        inst✝ : e.IsRelIff
        φ : Quiver.Hom (K.restriction e) L
        hφ : e.HasLift φ
        i' j' : ι'
        hij' : c'.Rel i' j'
        i : ι
        hi : Eq (e.f i) i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      by_cases hj' : ∃ j, e.f j = j'
        /-
          case pos
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' j' : ι'
          hij' : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          hj' : Exists fun j => Eq (e.f j) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · obtain ⟨j, hj⟩ := hj'
        /-
          case pos.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' j' : ι'
          hij' : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          j : ι
          hj : Eq (e.f j) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [f_eq φ hj, L.extend_d_eq e hi hj]
        /-
          case pos.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' j' : ι'
          hij' : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          j : ι
          hj : Eq (e.f j) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        subst hi hj
        /-
          case pos.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i j : ι
          hij' : c'.Rel (e.f i) (e.f j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp [HomologicalComplex.restrictionXIso]
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' j' : ι'
          hij' : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          hj' : Not (Exists fun j => Eq (e.f j) j')
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · apply (L.isZero_extend_X e j' (by simpa using hj')).eq_of_tgt
        /-
          🎉 no goals
        -/
    · have : (L.extend e).d i' j' = 0 := by
        apply (L.isZero_extend_X e i' (by simpa using hi')).eq_of_src
      /-
        case neg
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        e : c.Embedding c'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        L : HomologicalComplex C c
        inst✝ : e.IsRelIff
        φ : Quiver.Hom (K.restriction e) L
        hφ : e.HasLift φ
        i' j' : ι'
        hij' : c'.Rel i' j'
        hi' : Not (Exists fun i => Eq (e.f i) i')
        this : Eq ((L.extend e).d i' j') 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.liftExtend.f  …
      -/
      rw [this, comp_zero]
      /-
        case neg
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        e : c.Embedding c'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        L : HomologicalComplex C c
        inst✝ : e.IsRelIff
        φ : Quiver.Hom (K.restriction e) L
        hφ : e.HasLift φ
        i' j' : ι'
        hij' : c'.Rel i' j'
        hi' : Not (Exists fun i => Eq (e.f i) i')
        this : Eq ((L.extend e).d i' j') 0
        ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i' j') (ComplexShape.Embedding …
      -/
      by_cases hj' : ∃ j, e.f j = j'
        /-
          case pos
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' j' : ι'
          hij' : c'.Rel i' j'
          hi' : Not (Exists fun i => Eq (e.f i) i')
          this : Eq ((L.extend e).d i' j') 0
          hj' : Exists fun j => Eq (e.f j) j'
          ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i' j') (ComplexShape.Embedding …
        -/
      · obtain ⟨j, rfl⟩ := hj'
        /-
          case pos.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' : ι'
          hi' : Not (Exists fun i => Eq (e.f i) i')
          j : ι
          hij' : c'.Rel i' (e.f j)
          this : Eq ((L.extend e).d i' (e.f j)) 0
          ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i' (e.f j)) (ComplexShape.Embe …
        -/
        rw [f_eq φ rfl]
        /-
          case pos.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' : ι'
          hi' : Not (Exists fun i => Eq (e.f i) i')
          j : ι
          hij' : c'.Rel i' (e.f j)
          this : Eq ((L.extend e).d i' (e.f j)) 0
          ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i' (e.f j)) (CategoryTheory.Ca …
        -/
        dsimp [restrictionXIso]
        rw [id_comp, reassoc_of% (hφ j (e.boundaryGE hij'
          (by simpa using hi')) i' hij'), zero_comp]
      · have : f φ j' = 0 := by
          apply (L.isZero_extend_X e j' (by simpa using hj')).eq_of_tgt
        /-
          case neg
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{u_4, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K : HomologicalComplex C c'
          L : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' j' : ι'
          hij' : c'.Rel i' j'
          hi' : Not (Exists fun i => Eq (e.f i) i')
          this✝ : Eq ((L.extend e).d i' j') 0
          hj' : Not (Exists fun j => Eq (e.f j) j')
          this : Eq (ComplexShape.Embedding.liftExtend.f φ j') 0
          ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (K.d i' j') (ComplexShape.Embedding …
        -/
        rw [this, comp_zero]
        /-
          🎉 no goals
        -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      L : HomologicalComplex C c
      inst✝ : e.IsRelIff
      φ : Quiver.Hom (K.restriction e) L
      hφ : e.HasLift φ
      i' j' : ι'
      hij' : Not (c'.Rel i' j')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.liftExtend.f  …
    -/
  · simp [HomologicalComplex.shape _ _ _ hij']
    /-
      🎉 no goals
    -/


/-- The morphism  `K ⟶ L.extend e` given by a morphism `K.restriction e ⟶ L`
which satisfy `e.HasLift φ`. -/
noncomputable def liftExtend :
    K ⟶ L.extend e where
  f i' := liftExtend.f φ i'
  comm' _ _ _ := liftExtend.comm φ hφ _ _


lemma liftExtend_f :
    (e.liftExtend φ hφ).f i' = (K.restrictionXIso e hi).inv ≫ φ.f i ≫
      (L.extendXIso e hi).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    hφ : e.HasLift φ
    i' : ι'
    i : ι
    hi : Eq (e.f i) i'
    ⊢ Eq ((e.liftExtend φ hφ).f i') (CategoryTheory.CategoryStruct.comp (K.restric …
  -/
  apply liftExtend.f_eq
  /-
    🎉 no goals
  -/


/-- Given `φ : K.restriction e ⟶ L` such that `hφ : e.HasLift φ`, this is
the isomorphisms in the category of arrows between the maps
`(e.liftExtend φ hφ).f i'` and `φ.f i` when `e.f i = i'`. -/
noncomputable def liftExtendfArrowIso :
    Arrow.mk ((e.liftExtend φ hφ).f i') ≅ Arrow.mk (φ.f i) :=
  Arrow.isoMk (K.restrictionXIso e hi).symm (L.extendXIso e hi)
        /-
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          e : c.Embedding c'
          C : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.22405, u_3} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          K K' : HomologicalComplex C c'
          L L' : HomologicalComplex C c
          inst✝ : e.IsRelIff
          φ : Quiver.Hom (K.restriction e) L
          hφ : e.HasLift φ
          i' : ι'
          i : ι
          hi : Eq (e.f i) i'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.restrictionXIso e hi).symm.hom (Ca …
        -/
    (by simp [e.liftExtend_f φ hφ hi])
        /-
          🎉 no goals
        -/


lemma isIso_liftExtend_f_iff (hi : e.f i = i') :
    IsIso ((e.liftExtend φ hφ).f i') ↔ IsIso (φ.f i) :=
  (MorphismProperty.isomorphisms C).arrow_mk_iso_iff (e.liftExtendfArrowIso φ hφ hi)


lemma mono_liftExtend_f_iff (hi : e.f i = i') :
    Mono ((e.liftExtend φ hφ).f i') ↔ Mono (φ.f i) :=
  (MorphismProperty.monomorphisms C).arrow_mk_iso_iff (e.liftExtendfArrowIso φ hφ hi)


lemma epi_liftExtend_f_iff (hi : e.f i = i') :
    Epi ((e.liftExtend φ hφ).f i') ↔ Epi (φ.f i) :=
  (MorphismProperty.epimorphisms C).arrow_mk_iso_iff (e.liftExtendfArrowIso φ hφ hi)


/-- Auxiliary definition for `Embedding.homRestrict`. -/
noncomputable def f (i : ι) : (K.restriction e).X i ⟶ L.X i :=
  ψ.f (e.f i) ≫ (L.extendXIso e rfl).hom


lemma f_eq {i : ι} {i' : ι'} (h : e.f i = i') :
    f ψ i = (K.restrictionXIso e h).hom ≫ ψ.f i' ≫ (L.extendXIso e h).hom := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    i : ι
    i' : ι'
    h : Eq (e.f i) i'
    ⊢ Eq (ComplexShape.Embedding.homRestrict.f ψ i) (CategoryTheory.CategoryStruct …
  -/
  subst h
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    i : ι
    ⊢ Eq (ComplexShape.Embedding.homRestrict.f ψ i) (CategoryTheory.CategoryStruct …
  -/
  simp [f, restrictionXIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma comm (i j : ι) :
    f ψ i ≫ L.d i j = K.d (e.f i) (e.f j) ≫ f ψ j := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ComplexShape.Embedding.homRestrict.f …
  -/
  dsimp [f]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, ← ψ.comm_assoc, L.extend_d_eq e rfl rfl, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


/-- The morphism `K.restriction e ⟶ L` induced by a morphism `K ⟶ L.extend e`. -/
noncomputable def homRestrict (ψ : K ⟶ L.extend e) : K.restriction e ⟶ L where
  f i := homRestrict.f ψ i


lemma homRestrict_f (ψ : K ⟶ L.extend e) {i : ι} {i' : ι'} (h : e.f i = i') :
    (e.homRestrict ψ).f i = (K.restrictionXIso e h).hom ≫ ψ.f i' ≫ (L.extendXIso e h).hom :=
  homRestrict.f_eq ψ h


lemma homRestrict_hasLift (ψ : K ⟶ L.extend e) :
    e.HasLift (e.homRestrict ψ) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    ⊢ e.HasLift (e.homRestrict ψ)
  -/
  intro j hj i' hij'
  have : (L.extend e).d i' (e.f j) = 0 := by
    apply (L.isZero_extend_X e i' (hj.not_mem hij')).eq_of_src
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    j : ι
    hj : e.BoundaryGE j
    i' : ι'
    hij' : c'.Rel i' (e.f j)
    this : Eq ((L.extend e).d i' (e.f j)) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d i' (e.f j)) ((e.homRestrict ψ).f …
  -/
  dsimp [homRestrict]
  rw [homRestrict.f_eq ψ rfl, restrictionXIso, eqToIso_refl, Iso.refl_hom, id_comp,
    ← ψ.comm_assoc, this, zero_comp, comp_zero]


@[simp]
lemma liftExtend_homRestrict (ψ : K ⟶ L.extend e) :
    e.liftExtend (e.homRestrict ψ) (e.homRestrict_hasLift ψ) = ψ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    ⊢ Eq (e.liftExtend (e.homRestrict ψ) ⋯) ψ
  -/
  ext i'
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    ψ : Quiver.Hom K (L.extend e)
    i' : ι'
    ⊢ Eq ((e.liftExtend (e.homRestrict ψ) ⋯).f i') (ψ.f i')
  -/
  by_cases hi' : ∃ i, e.f i = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      L : HomologicalComplex C c
      inst✝ : e.IsRelIff
      ψ : Quiver.Hom K (L.extend e)
      i' : ι'
      hi' : Exists fun i => Eq (e.f i) i'
      ⊢ Eq ((e.liftExtend (e.homRestrict ψ) ⋯).f i') (ψ.f i')
    -/
  · obtain ⟨i, rfl⟩ := hi'
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      L : HomologicalComplex C c
      inst✝ : e.IsRelIff
      ψ : Quiver.Hom K (L.extend e)
      i : ι
      ⊢ Eq ((e.liftExtend (e.homRestrict ψ) ⋯).f (e.f i)) (ψ.f (e.f i))
    -/
    simp [e.homRestrict_f _ rfl, e.liftExtend_f _ _ rfl]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      L : HomologicalComplex C c
      inst✝ : e.IsRelIff
      ψ : Quiver.Hom K (L.extend e)
      i' : ι'
      hi' : Not (Exists fun i => Eq (e.f i) i')
      ⊢ Eq ((e.liftExtend (e.homRestrict ψ) ⋯).f i') (ψ.f i')
    -/
  · apply (L.isZero_extend_X e i' (by simpa using hi')).eq_of_tgt
    /-
      🎉 no goals
    -/


@[simp]
lemma homRestrict_liftExtend (φ : K.restriction e ⟶ L) (hφ : e.HasLift φ) :
    e.homRestrict (e.liftExtend φ hφ) = φ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    hφ : e.HasLift φ
    ⊢ Eq (e.homRestrict (e.liftExtend φ hφ)) φ
  -/
  ext i
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    φ : Quiver.Hom (K.restriction e) L
    hφ : e.HasLift φ
    i : ι
    ⊢ Eq ((e.homRestrict (e.liftExtend φ hφ)).f i) (φ.f i)
  -/
  simp [e.homRestrict_f _ rfl, e.liftExtend_f _ _ rfl]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homRestrict_precomp (α : K' ⟶ K) (ψ : K ⟶ L.extend e) :
    e.homRestrict (α ≫ ψ) = restrictionMap α e ≫ e.homRestrict ψ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K K' : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    α : Quiver.Hom K' K
    ψ : Quiver.Hom K (L.extend e)
    ⊢ Eq (e.homRestrict (CategoryTheory.CategoryStruct.comp α ψ)) (CategoryTheory. …
  -/
  ext i
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K K' : HomologicalComplex C c'
    L : HomologicalComplex C c
    inst✝ : e.IsRelIff
    α : Quiver.Hom K' K
    ψ : Quiver.Hom K (L.extend e)
    i : ι
    ⊢ Eq ((e.homRestrict (CategoryTheory.CategoryStruct.comp α ψ)).f i) ((Category …
  -/
  simp [homRestrict_f _ _ rfl, restrictionXIso]
  /-
    🎉 no goals
  -/


/-- The bijection between `K ⟶ L.extend e` and the subtype of `K.restriction e ⟶ L`
consisting of morphisms `φ` such that `e.HasLift φ`. -/
@[simps]
noncomputable def homEquiv :
    (K ⟶ L.extend e) ≃ { φ : K.restriction e ⟶ L // e.HasLift φ } where
  toFun ψ := ⟨e.homRestrict ψ, e.homRestrict_hasLift ψ⟩
  invFun φ := e.liftExtend φ.1 φ.2
                   /-
                     ι : Type u_1
                     ι' : Type u_2
                     c : ComplexShape ι
                     c' : ComplexShape ι'
                     e : c.Embedding c'
                     C : Type u_3
                     inst✝³ : CategoryTheory.Category.{?u.52792, u_3} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                     K K' : HomologicalComplex C c'
                     L L' : HomologicalComplex C c
                     inst✝ : e.IsRelIff
                     ψ : Quiver.Hom K (L.extend e)
                     ⊢ Eq ((fun φ => e.liftExtend ↑φ ⋯) ((fun ψ => ⟨e.homRestrict ψ, ⋯⟩) ψ)) ψ
                   -/
  left_inv ψ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      ι : Type u_1
                      ι' : Type u_2
                      c : ComplexShape ι
                      c' : ComplexShape ι'
                      e : c.Embedding c'
                      C : Type u_3
                      inst✝³ : CategoryTheory.Category.{?u.52792, u_3} C
                      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                      K K' : HomologicalComplex C c'
                      L L' : HomologicalComplex C c
                      inst✝ : e.IsRelIff
                      φ : Subtype fun φ => e.HasLift φ
                      ⊢ Eq ((fun ψ => ⟨e.homRestrict ψ, ⋯⟩) ((fun φ => e.liftExtend ↑φ ⋯) φ)) φ
                    -/
  right_inv φ := by simp
                    /-
                      🎉 no goals
                    -/


