/-- Auxiliary definition for the `X` field of `HomologicalComplex.extend`. -/
noncomputable def X : Option ι → C
  | some x => K.X x
  | none => 0


/-- The isomorphism `X K i ≅ K.X j` when `i = some j`. -/
noncomputable def XIso {i : Option ι} {j : ι} (hj : i = some j) :
                                 /-
                                   ι : Type u_1
                                   ι' : Type u_2
                                   c : ComplexShape ι
                                   c' : ComplexShape ι'
                                   C : Type u_3
                                   inst✝² : CategoryTheory.Category.{?u.808, u_3} C
                                   inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                   K L M : HomologicalComplex C c
                                   φ : Quiver.Hom K L
                                   φ' : Quiver.Hom L M
                                   e : c.Embedding c'
                                   i : Option ι
                                   j : ι
                                   hj : Eq i (Option.some j)
                                   ⊢ Eq (HomologicalComplex.extend.X K i) (K.X j)
                                 -/
    X K i ≅ K.X j := eqToIso (by subst hj; rfl)
                                           /-
                                             🎉 no goals
                                           -/


lemma isZero_X {i : Option ι} (hi : i = none) :
    IsZero (X K i) := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    i : Option ι
    hi : Eq i Option.none
    ⊢ CategoryTheory.Limits.IsZero (HomologicalComplex.extend.X K i)
  -/
  subst hi
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    ⊢ CategoryTheory.Limits.IsZero (HomologicalComplex.extend.X K Option.none)
  -/
  exact Limits.isZero_zero _
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `X K.op i ≅ Opposite.op (X K i)`. -/
noncomputable def XOpIso (i : Option ι) : X K.op i ≅ Opposite.op (X K i) :=
  match i with
  | some _ => Iso.refl _
  | none => IsZero.iso (isZero_X _ rfl) (isZero_X K rfl).op


/-- Auxiliary definition for the `d` field of `HomologicalComplex.extend`. -/
noncomputable def d : ∀ (i j : Option ι), extend.X K i ⟶ extend.X K j
  | none, _ => 0
  | some i, some j => K.d i j
  | some _, none => 0


lemma d_none_eq_zero (i j : Option ι) (hi : i = none) :
                      /-
                        ι : Type u_1
                        c : ComplexShape ι
                        C : Type u_3
                        inst✝² : CategoryTheory.Category.{u_4, u_3} C
                        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                        K : HomologicalComplex C c
                        i j : Option ι
                        hi : Eq i Option.none
                        ⊢ Eq (HomologicalComplex.extend.d K i j) 0
                      -/
    d K i j = 0 := by subst hi; rfl
                                /-
                                  🎉 no goals
                                -/


lemma d_none_eq_zero' (i j : Option ι) (hj : j = none) :
                      /-
                        ι : Type u_1
                        c : ComplexShape ι
                        C : Type u_3
                        inst✝² : CategoryTheory.Category.{u_4, u_3} C
                        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                        K : HomologicalComplex C c
                        i j : Option ι
                        hj : Eq j Option.none
                        ⊢ Eq (HomologicalComplex.extend.d K i j) 0
                      -/
                                            /-
                                              🎉 no goals
                                            -/
    d K i j = 0 := by subst hj; cases i <;> rfl
                                            /-
                                              🎉 no goals
                                            -/


lemma d_eq {i j : Option ι} {a b : ι} (hi : i = some a) (hj : j = some b) :
    d K i j = (XIso K hi).hom ≫ K.d a b ≫ (XIso K hj).inv := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    i j : Option ι
    a b : ι
    hi : Eq i (Option.some a)
    hj : Eq j (Option.some b)
    ⊢ Eq (HomologicalComplex.extend.d K i j) (CategoryTheory.CategoryStruct.comp ( …
  -/
  subst hi hj
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    a b : ι
    ⊢ Eq (HomologicalComplex.extend.d K (Option.some a) (Option.some b)) (Category …
  -/
  dsimp [XIso, d]
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    a b : ι
    ⊢ Eq (K.d a b) (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStr …
  -/
  erw [id_comp, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma XOpIso_hom_d_op (i j : Option ι) :
    (XOpIso K i).hom ≫ (d K j i).op =
      d K.op i j ≫ (XOpIso K j).hom :=
  match i, j with
  | none, _ => by
      /-
        ι : Type u_1
        c : ComplexShape ι
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        i j x✝ : Option ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.XOpIso K O …
      -/
      simp only [d_none_eq_zero, d_none_eq_zero', comp_zero, zero_comp, op_zero]
      /-
        🎉 no goals
      -/
  | some i, some j => by
      /-
        ι : Type u_1
        c : ComplexShape ι
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        i✝ j✝ : Option ι
        i j : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.XOpIso K ( …
      -/
      dsimp [XOpIso]
      simp only [d_eq _ rfl rfl, Option.some.injEq, d_eq, op_comp, assoc,
        id_comp, comp_id]
      /-
        ι : Type u_1
        c : ComplexShape ι
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        i✝ j✝ : Option ι
        i j : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.XIso K ⋯). …
      -/
      rfl
      /-
        🎉 no goals
      -/
  | some _, none => by
      /-
        ι : Type u_1
        c : ComplexShape ι
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        i j : Option ι
        val✝ : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.XOpIso K ( …
      -/
      simp only [d_none_eq_zero, d_none_eq_zero', comp_zero, zero_comp, op_zero]
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `HomologicalComplex.extendMap`. -/
noncomputable def mapX : ∀ (i : Option ι), X K i ⟶ X L i
  | some i => φ.f i
  | none => 0


lemma mapX_some {i : Option ι} {a : ι} (hi : i = some a) :
    mapX φ i = (XIso K hi).hom ≫ φ.f a ≫ (XIso L hi).inv := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    i : Option ι
    a : ι
    hi : Eq i (Option.some a)
    ⊢ Eq (HomologicalComplex.extend.mapX φ i) (CategoryTheory.CategoryStruct.comp  …
  -/
  subst hi
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    a : ι
    ⊢ Eq (HomologicalComplex.extend.mapX φ (Option.some a)) (CategoryTheory.Catego …
  -/
  dsimp [XIso]
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    a : ι
    ⊢ Eq (HomologicalComplex.extend.mapX φ (Option.some a)) (CategoryTheory.Catego …
  -/
  erw [id_comp, comp_id]
  /-
    ι : Type u_1
    c : ComplexShape ι
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    a : ι
    ⊢ Eq (HomologicalComplex.extend.mapX φ (Option.some a)) (φ.f a)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma mapX_none {i : Option ι} (hi : i = none) :
                       /-
                         ι : Type u_1
                         c : ComplexShape ι
                         C : Type u_3
                         inst✝² : CategoryTheory.Category.{u_4, u_3} C
                         inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                         K L : HomologicalComplex C c
                         φ : Quiver.Hom K L
                         i : Option ι
                         hi : Eq i Option.none
                         ⊢ Eq (HomologicalComplex.extend.mapX φ i) 0
                       -/
    mapX φ i = 0 := by subst hi; rfl
                                 /-
                                   🎉 no goals
                                 -/


/-- Given `K : HomologicalComplex C c` and `e : c.Embedding c'`,
this is the extension of `K` in `HomologicalComplex C c'`: it is
zero in the degrees that are not in the image of `e.f`. -/
noncomputable def extend : HomologicalComplex C c' where
  X i' := extend.X K (e.r i')
  d i' j' := extend.d K (e.r i') (e.r j')
  shape i' j' h := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' j' : ι'
      h : Not (c'.Rel i' j')
      ⊢ Eq ((fun i' j' => HomologicalComplex.extend.d K (e.r i') (e.r j')) i' j') 0
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' j' : ι'
      h : Not (c'.Rel i' j')
      ⊢ Eq (HomologicalComplex.extend.d K (e.r i') (e.r j')) 0
    -/
    obtain hi'|⟨i, hi⟩ := (e.r i').eq_none_or_eq_some
      /-
        case inl
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' : ι'
        h : Not (c'.Rel i' j')
        hi' : Eq (e.r i') Option.none
        ⊢ Eq (HomologicalComplex.extend.d K (e.r i') (e.r j')) 0
      -/
    · rw [extend.d_none_eq_zero K _ _ hi']
      /-
        🎉 no goals
      -/
      /-
        case inr.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' : ι'
        h : Not (c'.Rel i' j')
        i : ι
        hi : Eq (e.r i') (Option.some i)
        ⊢ Eq (HomologicalComplex.extend.d K (e.r i') (e.r j')) 0
      -/
    · obtain hj'|⟨j, hj⟩ := (e.r j').eq_none_or_eq_some
        /-
          case inr.intro.inl
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          h : Not (c'.Rel i' j')
          i : ι
          hi : Eq (e.r i') (Option.some i)
          hj' : Eq (e.r j') Option.none
          ⊢ Eq (HomologicalComplex.extend.d K (e.r i') (e.r j')) 0
        -/
      · rw [extend.d_none_eq_zero' K _ _ hj']
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.inr.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          h : Not (c'.Rel i' j')
          i : ι
          hi : Eq (e.r i') (Option.some i)
          j : ι
          hj : Eq (e.r j') (Option.some j)
          ⊢ Eq (HomologicalComplex.extend.d K (e.r i') (e.r j')) 0
        -/
      · rw [extend.d_eq K hi hj,K.shape, zero_comp, comp_zero]
        /-
          case inr.intro.inr.intro.a
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          h : Not (c'.Rel i' j')
          i : ι
          hi : Eq (e.r i') (Option.some i)
          j : ι
          hj : Eq (e.r j') (Option.some j)
          ⊢ Not (c.Rel i j)
        -/
        obtain rfl := e.f_eq_of_r_eq_some hi
        /-
          case inr.intro.inr.intro.a
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          j' : ι'
          i j : ι
          hj : Eq (e.r j') (Option.some j)
          h : Not (c'.Rel (e.f i) j')
          hi : Eq (e.r (e.f i)) (Option.some i)
          ⊢ Not (c.Rel i j)
        -/
        obtain rfl := e.f_eq_of_r_eq_some hj
        /-
          case inr.intro.inr.intro.a
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i j : ι
          hi : Eq (e.r (e.f i)) (Option.some i)
          hj : Eq (e.r (e.f j)) (Option.some j)
          h : Not (c'.Rel (e.f i) (e.f j))
          ⊢ Not (c.Rel i j)
        -/
        intro hij
        /-
          case inr.intro.inr.intro.a
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i j : ι
          hi : Eq (e.r (e.f i)) (Option.some i)
          hj : Eq (e.r (e.f j)) (Option.some j)
          h : Not (c'.Rel (e.f i) (e.f j))
          hij : c.Rel i j
          ⊢ False
        -/
        exact h (e.rel hij)
        /-
          🎉 no goals
        -/
  d_comp_d' i' j' k' _ _ := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' j' k' : ι'
      x✝¹ : c'.Rel i' j'
      x✝ : c'.Rel j' k'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i' j' => HomologicalComplex.ext …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' j' k' : ι'
      x✝¹ : c'.Rel i' j'
      x✝ : c'.Rel j' k'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.d K (e.r i …
    -/
    obtain hi'|⟨i, hi⟩ := (e.r i').eq_none_or_eq_some
      /-
        case inl
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' k' : ι'
        x✝¹ : c'.Rel i' j'
        x✝ : c'.Rel j' k'
        hi' : Eq (e.r i') Option.none
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.d K (e.r i …
      -/
    · rw [extend.d_none_eq_zero K _ _ hi', zero_comp]
      /-
        🎉 no goals
      -/
      /-
        case inr.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' k' : ι'
        x✝¹ : c'.Rel i' j'
        x✝ : c'.Rel j' k'
        i : ι
        hi : Eq (e.r i') (Option.some i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.d K (e.r i …
      -/
    · obtain hj'|⟨j, hj⟩ := (e.r j').eq_none_or_eq_some
        /-
          case inr.intro.inl
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' k' : ι'
          x✝¹ : c'.Rel i' j'
          x✝ : c'.Rel j' k'
          i : ι
          hi : Eq (e.r i') (Option.some i)
          hj' : Eq (e.r j') Option.none
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.d K (e.r i …
        -/
      · rw [extend.d_none_eq_zero K _ _ hj', comp_zero]
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.inr.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' k' : ι'
          x✝¹ : c'.Rel i' j'
          x✝ : c'.Rel j' k'
          i : ι
          hi : Eq (e.r i') (Option.some i)
          j : ι
          hj : Eq (e.r j') (Option.some j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.d K (e.r i …
        -/
      · obtain hk'|⟨k, hk⟩ := (e.r k').eq_none_or_eq_some
          /-
            case inr.intro.inr.intro.inl
            ι : Type u_1
            ι' : Type u_2
            c : ComplexShape ι
            c' : ComplexShape ι'
            C : Type u_3
            inst✝² : CategoryTheory.Category.{?u.14642, u_3} C
            inst✝¹ : CategoryTheory.Limits.HasZeroObject C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            K L M : HomologicalComplex C c
            φ : Quiver.Hom K L
            φ' : Quiver.Hom L M
            e : c.Embedding c'
            i' j' k' : ι'
            x✝¹ : c'.Rel i' j'
            x✝ : c'.Rel j' k'
            i : ι
            hi : Eq (e.r i') (Option.some i)
            j : ι
            hj : Eq (e.r j') (Option.some j)
            hk' : Eq (e.r k') Option.none
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.d K (e.r i …
          -/
        · rw [extend.d_none_eq_zero' K _ _ hk', comp_zero]
          /-
            🎉 no goals
          -/
        · rw [extend.d_eq K hi hj, extend.d_eq K hj hk, assoc, assoc,
            Iso.inv_hom_id_assoc, K.d_comp_d_assoc, zero_comp, comp_zero]


/-- The isomorphism `(K.extend e).X i' ≅ K.X i` when `e.f i = i'`. -/
noncomputable def extendXIso {i' : ι'} {i : ι} (h : e.f i = i') :
    (K.extend e).X i' ≅ K.X i :=
  extend.XIso K (e.r_eq_some h)


lemma isZero_extend_X' (i' : ι') (hi' : e.r i' = none) :
    IsZero ((K.extend e).X i') :=
  extend.isZero_X K hi'


lemma isZero_extend_X (i' : ι') (hi' : ∀ i, e.f i ≠ i') :
    IsZero ((K.extend e).X i') :=
  K.isZero_extend_X' e i' (by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ Eq (e.r i') Option.none
    -/
    obtain hi'|⟨i, hi⟩ := (e.r i').eq_none_or_eq_some
      /-
        case inl
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        e : c.Embedding c'
        i' : ι'
        hi'✝ : ∀ (i : ι), Ne (e.f i) i'
        hi' : Eq (e.r i') Option.none
        ⊢ Eq (e.r i') Option.none
      -/
    · exact hi'
      /-
        🎉 no goals
      -/
      /-
        case inr.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        e : c.Embedding c'
        i' : ι'
        hi' : ∀ (i : ι), Ne (e.f i) i'
        i : ι
        hi : Eq (e.r i') (Option.some i)
        ⊢ Eq (e.r i') Option.none
      -/
    · exfalso
      /-
        case inr.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{u_4, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c
        e : c.Embedding c'
        i' : ι'
        hi' : ∀ (i : ι), Ne (e.f i) i'
        i : ι
        hi : Eq (e.r i') (Option.some i)
        ⊢ False
      -/
      exact hi' _ (e.f_eq_of_r_eq_some hi))
      /-
        🎉 no goals
      -/


instance : (K.extend e).IsStrictlySupported e where
  isZero i' hi' := K.isZero_extend_X e i' hi'


lemma extend_d_eq {i' j' : ι'} {i j : ι} (hi : e.f i = i') (hj : e.f j = j') :
    (K.extend e).d i' j' = (K.extendXIso e hi).hom ≫ K.d i j ≫
      (K.extendXIso e hj).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' j' : ι'
    i j : ι
    hi : Eq (e.f i) i'
    hj : Eq (e.f j) j'
    ⊢ Eq ((K.extend e).d i' j') (CategoryTheory.CategoryStruct.comp (K.extendXIso  …
  -/
  apply extend.d_eq
  /-
    🎉 no goals
  -/


lemma extend_d_from_eq_zero (i' j' : ι') (i : ι) (hi : e.f i = i') (hi' : ¬ c.Rel i (c.next i)) :
    (K.extend e).d i' j' = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' j' : ι'
    i : ι
    hi : Eq (e.f i) i'
    hi' : Not (c.Rel i (c.next i))
    ⊢ Eq ((K.extend e).d i' j') 0
  -/
  obtain hj'|⟨j, hj⟩ := (e.r j').eq_none_or_eq_some
    /-
      case inl
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      i : ι
      hi : Eq (e.f i) i'
      hi' : Not (c.Rel i (c.next i))
      hj' : Eq (e.r j') Option.none
      ⊢ Eq ((K.extend e).d i' j') 0
    -/
  · exact extend.d_none_eq_zero' _ _ _ hj'
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      i : ι
      hi : Eq (e.f i) i'
      hi' : Not (c.Rel i (c.next i))
      j : ι
      hj : Eq (e.r j') (Option.some j)
      ⊢ Eq ((K.extend e).d i' j') 0
    -/
  · rw [extend_d_eq K e hi (e.f_eq_of_r_eq_some hj), K.shape, zero_comp, comp_zero]
    /-
      case inr.intro.a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      i : ι
      hi : Eq (e.f i) i'
      hi' : Not (c.Rel i (c.next i))
      j : ι
      hj : Eq (e.r j') (Option.some j)
      ⊢ Not (c.Rel i j)
    -/
    intro hij
    /-
      case inr.intro.a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      i : ι
      hi : Eq (e.f i) i'
      hi' : Not (c.Rel i (c.next i))
      j : ι
      hj : Eq (e.r j') (Option.some j)
      hij : c.Rel i j
      ⊢ False
    -/
    obtain rfl := c.next_eq' hij
    /-
      case inr.intro.a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      i : ι
      hi : Eq (e.f i) i'
      hi' : Not (c.Rel i (c.next i))
      hj : Eq (e.r j') (Option.some (c.next i))
      hij : c.Rel i (c.next i)
      ⊢ False
    -/
    exact hi' hij
    /-
      🎉 no goals
    -/


lemma extend_d_to_eq_zero (i' j' : ι') (j : ι) (hj : e.f j = j') (hj' : ¬ c.Rel (c.prev j) j) :
    (K.extend e).d i' j' = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' j' : ι'
    j : ι
    hj : Eq (e.f j) j'
    hj' : Not (c.Rel (c.prev j) j)
    ⊢ Eq ((K.extend e).d i' j') 0
  -/
  obtain hi'|⟨i, hi⟩ := (e.r i').eq_none_or_eq_some
    /-
      case inl
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      j : ι
      hj : Eq (e.f j) j'
      hj' : Not (c.Rel (c.prev j) j)
      hi' : Eq (e.r i') Option.none
      ⊢ Eq ((K.extend e).d i' j') 0
    -/
  · exact extend.d_none_eq_zero _ _ _ hi'
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      j : ι
      hj : Eq (e.f j) j'
      hj' : Not (c.Rel (c.prev j) j)
      i : ι
      hi : Eq (e.r i') (Option.some i)
      ⊢ Eq ((K.extend e).d i' j') 0
    -/
  · rw [extend_d_eq K e (e.f_eq_of_r_eq_some hi) hj, K.shape, zero_comp, comp_zero]
    /-
      case inr.intro.a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      j : ι
      hj : Eq (e.f j) j'
      hj' : Not (c.Rel (c.prev j) j)
      i : ι
      hi : Eq (e.r i') (Option.some i)
      ⊢ Not (c.Rel i j)
    -/
    intro hij
    /-
      case inr.intro.a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      j : ι
      hj : Eq (e.f j) j'
      hj' : Not (c.Rel (c.prev j) j)
      i : ι
      hi : Eq (e.r i') (Option.some i)
      hij : c.Rel i j
      ⊢ False
    -/
    obtain rfl := c.prev_eq' hij
    /-
      case inr.intro.a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' j' : ι'
      j : ι
      hj : Eq (e.f j) j'
      hj' : Not (c.Rel (c.prev j) j)
      hi : Eq (e.r i') (Option.some (c.prev j))
      hij : c.Rel (c.prev j) j
      ⊢ False
    -/
    exact hj' hij
    /-
      🎉 no goals
    -/


/-- Given an ambedding `e : c.Embedding c'` of complexes shapes, this is the
morphism `K.extend e ⟶ L.extend e` induced by a morphism `K ⟶ L` in
`HomologicalComplex C c`. -/
noncomputable def extendMap : K.extend e ⟶ L.extend e where
  f _ := extend.mapX φ _
  comm' i' j' _ := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' j' : ι'
      x✝ : c'.Rel i' j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => HomologicalComplex.extend. …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' j' : ι'
      x✝ : c'.Rel i' j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
    -/
    by_cases hi : ∃ i, e.f i = i'
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' : ι'
        x✝ : c'.Rel i' j'
        hi : Exists fun i => Eq (e.f i) i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
      -/
    · obtain ⟨i, hi⟩ := hi
      /-
        case pos.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' : ι'
        x✝ : c'.Rel i' j'
        i : ι
        hi : Eq (e.f i) i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
      -/
      by_cases hj : ∃ j, e.f j = j'
        /-
          case pos
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          x✝ : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          hj : Exists fun j => Eq (e.f j) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
        -/
      · obtain ⟨j, hj⟩ := hj
        rw [K.extend_d_eq e hi hj, L.extend_d_eq e hi hj,
          extend.mapX_some φ (e.r_eq_some hi),
          extend.mapX_some φ (e.r_eq_some hj)]
        /-
          case pos.intro
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          x✝ : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          j : ι
          hj : Eq (e.f j) j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [extendXIso, assoc, Iso.inv_hom_id_assoc, Hom.comm_assoc]
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          x✝ : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          hj : Not (Exists fun j => Eq (e.f j) j')
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
        -/
      · have hj' := e.r_eq_none j' (fun j'' hj'' => hj ⟨j'', hj''⟩)
        /-
          case neg
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          C : Type u_3
          inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          K L M : HomologicalComplex C c
          φ : Quiver.Hom K L
          φ' : Quiver.Hom L M
          e : c.Embedding c'
          i' j' : ι'
          x✝ : c'.Rel i' j'
          i : ι
          hi : Eq (e.f i) i'
          hj : Not (Exists fun j => Eq (e.f j) j')
          hj' : Eq (e.r j') Option.none
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
        -/
        dsimp [extend]
        rw [extend.d_none_eq_zero' _ _ _ hj', extend.d_none_eq_zero' _ _ _ hj',
          comp_zero, zero_comp]
      /-
        case neg
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' : ι'
        x✝ : c'.Rel i' j'
        hi : Not (Exists fun i => Eq (e.f i) i')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
      -/
    · have hi' := e.r_eq_none i' (fun i'' hi'' => hi ⟨i'', hi''⟩)
      /-
        case neg
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝² : CategoryTheory.Category.{?u.29811, u_3} C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        i' j' : ι'
        x✝ : c'.Rel i' j'
        hi : Not (Exists fun i => Eq (e.f i) i')
        hi' : Eq (e.r i') Option.none
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.mapX φ (e. …
      -/
      dsimp [extend]
      rw [extend.d_none_eq_zero _ _ _ hi', extend.d_none_eq_zero _ _ _ hi',
        comp_zero, zero_comp]


lemma extendMap_f {i : ι} {i' : ι'} (h : e.f i = i') :
    (extendMap φ e).f i' =
      (extendXIso K e h).hom ≫ φ.f i ≫ (extendXIso L e h).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    e : c.Embedding c'
    i : ι
    i' : ι'
    h : Eq (e.f i) i'
    ⊢ Eq ((HomologicalComplex.extendMap φ e).f i') (CategoryTheory.CategoryStruct. …
  -/
  dsimp [extendMap]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    e : c.Embedding c'
    i : ι
    i' : ι'
    h : Eq (e.f i) i'
    ⊢ Eq (HomologicalComplex.extend.mapX φ (e.r i')) (CategoryTheory.CategoryStruc …
  -/
  rw [extend.mapX_some φ (e.r_eq_some h)]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    e : c.Embedding c'
    i : ι
    i' : ι'
    h : Eq (e.f i) i'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.extend.XIso K ⋯). …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma extendMap_f_eq_zero (i' : ι') (hi' : ∀ i, e.f i ≠ i') :
    (extendMap φ e).f i' = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    e : c.Embedding c'
    i' : ι'
    hi' : ∀ (i : ι), Ne (e.f i) i'
    ⊢ Eq ((HomologicalComplex.extendMap φ e).f i') 0
  -/
  dsimp [extendMap]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    e : c.Embedding c'
    i' : ι'
    hi' : ∀ (i : ι), Ne (e.f i) i'
    ⊢ Eq (HomologicalComplex.extend.mapX φ (e.r i')) 0
  -/
  rw [extend.mapX_none φ (e.r_eq_none i' hi')]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
lemma extendMap_comp :
    extendMap (φ ≫ φ') e = extendMap φ e ≫ extendMap φ' e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    ⊢ Eq (HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.comp φ φ') e …
  -/
  ext i'
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L M : HomologicalComplex C c
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    i' : ι'
    ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.comp φ φ')  …
  -/
  by_cases hi' : ∃ i, e.f i = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' : ι'
      hi' : Exists fun i => Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.comp φ φ')  …
    -/
  · obtain ⟨i, hi⟩ := hi'
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' : ι'
      i : ι
      hi : Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.comp φ φ')  …
    -/
    simp [extendMap_f _ e hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      i' : ι'
      hi' : Not (Exists fun i => Eq (e.f i) i')
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.comp φ φ')  …
    -/
  · simp [extendMap_f_eq_zero _ e i' (fun i hi => hi' ⟨i, hi⟩)]
    /-
      🎉 no goals
    -/


lemma extendMap_id_f (i' : ι') : (extendMap (𝟙 K) e).f i' = 𝟙 _ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' : ι'
    ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
  -/
  by_cases hi' : ∃ i, e.f i = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : Exists fun i => Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
    -/
  · obtain ⟨i, hi⟩ := hi'
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      i : ι
      hi : Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
    -/
    simp [extendMap_f _ e hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : Not (Exists fun i => Eq (e.f i) i')
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
    -/
  · apply (K.isZero_extend_X e i' (fun i hi => hi' ⟨i, hi⟩)).eq_of_src
    /-
      🎉 no goals
    -/


@[simp]
lemma extendMap_id : extendMap (𝟙 K) e = 𝟙 _ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    ⊢ Eq (HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e) (Ca …
  -/
  ext i'
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' : ι'
    ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
  -/
  by_cases hi' : ∃ i, e.f i = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : Exists fun i => Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
    -/
  · obtain ⟨i, hi⟩ := hi'
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      i : ι
      hi : Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
    -/
    simp [extendMap_f _ e hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : Not (Exists fun i => Eq (e.f i) i')
      ⊢ Eq ((HomologicalComplex.extendMap (CategoryTheory.CategoryStruct.id K) e).f  …
    -/
  · apply (K.isZero_extend_X e i' (fun i hi => hi' ⟨i, hi⟩)).eq_of_src
    /-
      🎉 no goals
    -/


@[simp]
lemma extendMap_zero : extendMap (0 : K ⟶ L) e = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    e : c.Embedding c'
    ⊢ Eq (HomologicalComplex.extendMap 0 e) 0
  -/
  ext i'
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c
    e : c.Embedding c'
    i' : ι'
    ⊢ Eq ((HomologicalComplex.extendMap 0 e).f i') (HomologicalComplex.Hom.f 0 i')
  -/
  by_cases hi' : ∃ i, e.f i = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : Exists fun i => Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap 0 e).f i') (HomologicalComplex.Hom.f 0 i')
    -/
  · obtain ⟨i, hi⟩ := hi'
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      i : ι
      hi : Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap 0 e).f i') (HomologicalComplex.Hom.f 0 i')
    -/
    simp [extendMap_f _ e hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K L : HomologicalComplex C c
      e : c.Embedding c'
      i' : ι'
      hi' : Not (Exists fun i => Eq (e.f i) i')
      ⊢ Eq ((HomologicalComplex.extendMap 0 e).f i') (HomologicalComplex.Hom.f 0 i')
    -/
  · apply (K.isZero_extend_X e i' (fun i hi => hi' ⟨i, hi⟩)).eq_of_src
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism `K.op.extend e.op ≅ (K.extend e).op`. -/
noncomputable def extendOpIso : K.op.extend e.op ≅ (K.extend e).op :=
  Hom.isoOfComponents (fun _ ↦ extend.XOpIso _ _) (fun _ _ _ ↦
    extend.XOpIso_hom_d_op _ _ _)


@[reassoc]
lemma extend_op_d (i' j' : ι') :
    (K.op.extend e.op).d i' j' =
      (K.extendOpIso e).hom.f i' ≫ ((K.extend e).d j' i').op ≫
        (K.extendOpIso e).inv.f j' := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' j' : ι'
    ⊢ Eq ((K.op.extend e.op).d i' j') (CategoryTheory.CategoryStruct.comp ((K.exte …
  -/
  have := (K.extendOpIso e).inv.comm i' j'
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' j' : ι'
    this : Eq (CategoryTheory.CategoryStruct.comp ((K.extendOpIso e).inv.f i') ((K …
    ⊢ Eq ((K.op.extend e.op).d i' j') (CategoryTheory.CategoryStruct.comp ((K.exte …
  -/
  dsimp at this
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c
    e : c.Embedding c'
    i' j' : ι'
    this : Eq (CategoryTheory.CategoryStruct.comp ((K.extendOpIso e).inv.f i') ((K …
    ⊢ Eq ((K.op.extend e.op).d i' j') (CategoryTheory.CategoryStruct.comp ((K.exte …
  -/
  rw [← this, ← comp_f_assoc, Iso.hom_inv_id, id_f, id_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma extendMap_add [Preadditive C] {K L : HomologicalComplex C c} (φ φ' : K ⟶ L)
    (e : c.Embedding c') : extendMap (φ + φ' : K ⟶ L) e = extendMap φ e + extendMap φ' e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Preadditive C
    K L : HomologicalComplex C c
    φ φ' : Quiver.Hom K L
    e : c.Embedding c'
    ⊢ Eq (HomologicalComplex.extendMap (HAdd.hAdd φ φ') e) (HAdd.hAdd (Homological …
  -/
  ext i'
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝² : CategoryTheory.Category.{u_4, u_3} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Preadditive C
    K L : HomologicalComplex C c
    φ φ' : Quiver.Hom K L
    e : c.Embedding c'
    i' : ι'
    ⊢ Eq ((HomologicalComplex.extendMap (HAdd.hAdd φ φ') e).f i') ((HAdd.hAdd (Hom …
  -/
  by_cases hi' : ∃ i, e.f i = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Preadditive C
      K L : HomologicalComplex C c
      φ φ' : Quiver.Hom K L
      e : c.Embedding c'
      i' : ι'
      hi' : Exists fun i => Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (HAdd.hAdd φ φ') e).f i') ((HAdd.hAdd (Hom …
    -/
  · obtain ⟨i, hi⟩ := hi'
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Preadditive C
      K L : HomologicalComplex C c
      φ φ' : Quiver.Hom K L
      e : c.Embedding c'
      i' : ι'
      i : ι
      hi : Eq (e.f i) i'
      ⊢ Eq ((HomologicalComplex.extendMap (HAdd.hAdd φ φ') e).f i') ((HAdd.hAdd (Hom …
    -/
    simp [extendMap_f _ e hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Preadditive C
      K L : HomologicalComplex C c
      φ φ' : Quiver.Hom K L
      e : c.Embedding c'
      i' : ι'
      hi' : Not (Exists fun i => Eq (e.f i) i')
      ⊢ Eq ((HomologicalComplex.extendMap (HAdd.hAdd φ φ') e).f i') ((HAdd.hAdd (Hom …
    -/
  · apply (K.isZero_extend_X e i' (fun i hi => hi' ⟨i, hi⟩)).eq_of_src
    /-
      🎉 no goals
    -/


/-- Given an embedding `e : c.Embedding c'` of complex shapes, this is
the functor `HomologicalComplex C c ⥤ HomologicalComplex C c'` which
extend complexes along `e`: the extended complexes are zero
in the degrees that are not in the image of `e.f`. -/
@[simps]
noncomputable def extendFunctor [HasZeroMorphisms C] :
    HomologicalComplex C c ⥤ HomologicalComplex C c' where
  obj K := K.extend e
  map φ := HomologicalComplex.extendMap φ e


instance [HasZeroMorphisms C] : (e.extendFunctor C).PreservesZeroMorphisms where


instance [Preadditive C] : (e.extendFunctor C).Additive where


