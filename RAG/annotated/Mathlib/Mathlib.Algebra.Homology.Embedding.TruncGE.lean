open Classical in
/-- The `X` field of `truncGE'`. -/
noncomputable def X (i : ι) : C :=
  if e.BoundaryGE i
  then K.opcycles (e.f i)
  else K.X (e.f i)


/-- The isomorphism `truncGE'.X K e i ≅ K.opcycles (e.f i)` when `e.BoundaryGE i` holds.-/
noncomputable def XIsoOpcycles {i : ι} (hi : e.BoundaryGE i) :
    X K e i ≅ K.opcycles (e.f i) :=
  eqToIso (if_pos hi)


/-- The isomorphism `truncGE'.X K e i ≅ K.X (e.f i)` when `e.BoundaryGE i` does not hold.-/
noncomputable def XIso {i : ι} (hi : ¬ e.BoundaryGE i) :
    X K e i ≅ K.X (e.f i) :=
  eqToIso (if_neg hi)


open Classical in
/-- The `d` field of `truncGE'`. -/
noncomputable def d (i j : ι) : X K e i ⟶ X K e j :=
  if hij : c.Rel i j
  then
    if hi : e.BoundaryGE i
    then (truncGE'.XIsoOpcycles K e hi).hom ≫ K.fromOpcycles (e.f i) (e.f j) ≫
      (XIso K e (e.not_boundaryGE_next hij)).inv
    else (XIso K e hi).hom ≫ K.d (e.f i) (e.f j) ≫
      (XIso K e (e.not_boundaryGE_next hij)).inv
  else 0


@[reassoc (attr := simp)]
lemma d_comp_d (i j k : ι) : d K e i j ≫ d K e j k = 0 := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j k : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.truncGE'.d K e i  …
  -/
  dsimp [d]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j k : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i j) (fun hij => dite (e …
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝¹ : e.IsTruncGE
      inst✝ : ∀ (i' : ι'), K.HasHomology i'
      i j k : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i j) (fun hij => dite (e …
    -/
  · by_cases hjk : c.Rel j k
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        inst✝¹ : e.IsTruncGE
        inst✝ : ∀ (i' : ι'), K.HasHomology i'
        i j k : ι
        hij : c.Rel i j
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i j) (fun hij => dite (e …
      -/
    · rw [dif_pos hij, dif_pos hjk, dif_neg (e.not_boundaryGE_next hij)]
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        inst✝¹ : e.IsTruncGE
        inst✝ : ∀ (i' : ι'), K.HasHomology i'
        i j k : ι
        hij : c.Rel i j
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (e.BoundaryGE i) (fun hi => Cat …
      -/
                    /-
                      🎉 no goals
                    -/
      split_ifs <;> simp
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
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        inst✝¹ : e.IsTruncGE
        inst✝ : ∀ (i' : ι'), K.HasHomology i'
        i j k : ι
        hij : c.Rel i j
        hjk : Not (c.Rel j k)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i j) (fun hij => dite (e …
      -/
    · rw [dif_neg hjk, comp_zero]
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
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝¹ : e.IsTruncGE
      inst✝ : ∀ (i' : ι'), K.HasHomology i'
      i j k : ι
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i j) (fun hij => dite (e …
    -/
  · rw [dif_neg hij, zero_comp]
    /-
      🎉 no goals
    -/


/-- The canonical truncation of a homological complex relative to an embedding
of complex shapes `e` which satisfies `e.IsTruncGE`. -/
noncomputable def truncGE' : HomologicalComplex C c where
  X := truncGE'.X K e
  d := truncGE'.d K e
  shape _ _ h := dif_neg h


/-- The isomorphism `(K.truncGE' e).X i ≅ K.X i'` when `e.f i = i'`
and `e.BoundaryGE i` does not hold. -/
noncomputable def truncGE'XIso {i : ι} {i' : ι'} (hi' : e.f i = i') (hi : ¬ e.BoundaryGE i) :
    (K.truncGE' e).X i ≅ K.X i' :=
                                        /-
                                          ι : Type u_1
                                          ι' : Type u_2
                                          c : ComplexShape ι
                                          c' : ComplexShape ι'
                                          C : Type u_3
                                          inst✝⁵ : CategoryTheory.Category.{?u.11446, u_3} C
                                          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                          K L M : HomologicalComplex C c'
                                          φ : Quiver.Hom K L
                                          φ' : Quiver.Hom L M
                                          e : c.Embedding c'
                                          inst✝³ : e.IsTruncGE
                                          inst✝² : ∀ (i' : ι'), K.HasHomology i'
                                          inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
                                          inst✝ : ∀ (i' : ι'), M.HasHomology i'
                                          i : ι
                                          i' : ι'
                                          hi' : Eq (e.f i) i'
                                          hi : Not (e.BoundaryGE i)
                                          ⊢ Eq (K.X (e.f i)) (K.X i')
                                        -/
  (truncGE'.XIso K e hi) ≪≫ eqToIso (by subst hi'; rfl)
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The isomorphism `(K.truncGE' e).X i ≅ K.opcycles i'` when `e.f i = i'`
and `e.BoundaryGE i` holds. -/
noncomputable def truncGE'XIsoOpcycles {i : ι} {i' : ι'} (hi' : e.f i = i') (hi : e.BoundaryGE i) :
    (K.truncGE' e).X i ≅ K.opcycles i' :=
                                                /-
                                                  ι : Type u_1
                                                  ι' : Type u_2
                                                  c : ComplexShape ι
                                                  c' : ComplexShape ι'
                                                  C : Type u_3
                                                  inst✝⁵ : CategoryTheory.Category.{?u.12609, u_3} C
                                                  inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  K L M : HomologicalComplex C c'
                                                  φ : Quiver.Hom K L
                                                  φ' : Quiver.Hom L M
                                                  e : c.Embedding c'
                                                  inst✝³ : e.IsTruncGE
                                                  inst✝² : ∀ (i' : ι'), K.HasHomology i'
                                                  inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
                                                  inst✝ : ∀ (i' : ι'), M.HasHomology i'
                                                  i : ι
                                                  i' : ι'
                                                  hi' : Eq (e.f i) i'
                                                  hi : e.BoundaryGE i
                                                  ⊢ Eq (K.opcycles (e.f i)) (K.opcycles i')
                                                -/
  (truncGE'.XIsoOpcycles K e hi) ≪≫ eqToIso (by subst hi'; rfl)
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma truncGE'_d_eq {i j : ι} (hij : c.Rel i j) {i' j' : ι'}
    (hi' : e.f i = i') (hj' : e.f j = j') (hi : ¬ e.BoundaryGE i) :
    (K.truncGE' e).d i j = (K.truncGE'XIso e hi' hi).hom ≫ K.d i' j' ≫
      (K.truncGE'XIso e hj' (e.not_boundaryGE_next hij)).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    i' j' : ι'
    hi' : Eq (e.f i) i'
    hj' : Eq (e.f j) j'
    hi : Not (e.BoundaryGE i)
    ⊢ Eq ((K.truncGE' e).d i j) (CategoryTheory.CategoryStruct.comp (K.truncGE'XIs …
  -/
  dsimp [truncGE', truncGE'.d]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    i' j' : ι'
    hi' : Eq (e.f i) i'
    hj' : Eq (e.f j) j'
    hi : Not (e.BoundaryGE i)
    ⊢ Eq (dite (c.Rel i j) (fun hij => dite (e.BoundaryGE i) (fun hi => CategoryTh …
  -/
  rw [dif_pos hij, dif_neg hi]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    i' j' : ι'
    hi' : Eq (e.f i) i'
    hj' : Eq (e.f j) j'
    hi : Not (e.BoundaryGE i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.truncGE'.XIso K e …
  -/
  subst hi' hj'
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    hi : Not (e.BoundaryGE i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.truncGE'.XIso K e …
  -/
  simp [truncGE'XIso]
  /-
    🎉 no goals
  -/


lemma truncGE'_d_eq_fromOpcycles {i j : ι} (hij : c.Rel i j) {i' j' : ι'}
    (hi' : e.f i = i') (hj' : e.f j = j') (hi : e.BoundaryGE i) :
    (K.truncGE' e).d i j = (K.truncGE'XIsoOpcycles e hi' hi).hom ≫ K.fromOpcycles i' j' ≫
      (K.truncGE'XIso e hj' (e.not_boundaryGE_next hij)).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    i' j' : ι'
    hi' : Eq (e.f i) i'
    hj' : Eq (e.f j) j'
    hi : e.BoundaryGE i
    ⊢ Eq ((K.truncGE' e).d i j) (CategoryTheory.CategoryStruct.comp (K.truncGE'XIs …
  -/
  dsimp [truncGE', truncGE'.d]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    i' j' : ι'
    hi' : Eq (e.f i) i'
    hj' : Eq (e.f j) j'
    hi : e.BoundaryGE i
    ⊢ Eq (dite (c.Rel i j) (fun hij => dite (e.BoundaryGE i) (fun hi => CategoryTh …
  -/
  rw [dif_pos hij, dif_pos hi]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    i' j' : ι'
    hi' : Eq (e.f i) i'
    hj' : Eq (e.f j) j'
    hi : e.BoundaryGE i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.truncGE'.XIsoOpcy …
  -/
  subst hi' hj'
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i j : ι
    hij : c.Rel i j
    hi : e.BoundaryGE i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.truncGE'.XIsoOpcy …
  -/
  simp [truncGE'XIso, truncGE'XIsoOpcycles]
  /-
    🎉 no goals
  -/


/-- The canonical truncation of a homological complex relative to an embedding
of complex shapes `e` which satisfies `e.IsTruncGE`. -/
noncomputable def truncGE : HomologicalComplex C c' := (K.truncGE' e).extend e


/-- The isomorphism `(K.truncGE e).X i' ≅ K.X i'` when `e.f i = i'`
and `e.BoundaryGE i` does not hold. -/
noncomputable def truncGEXIso {i : ι} {i' : ι'} (hi' : e.f i = i') (hi : ¬ e.BoundaryGE i) :
    (K.truncGE e).X i' ≅ K.X i' :=
  (K.truncGE' e).extendXIso e hi' ≪≫ K.truncGE'XIso e hi' hi


/-- The isomorphism `(K.truncGE e).X i' ≅ K.opcycles i'` when `e.f i = i'`
and `e.BoundaryGE i` holds. -/
noncomputable def truncGEXIsoOpcycles {i : ι} {i' : ι'} (hi' : e.f i = i') (hi : e.BoundaryGE i) :
    (K.truncGE e).X i' ≅ K.opcycles i' :=
  (K.truncGE' e).extendXIso e hi' ≪≫ K.truncGE'XIsoOpcycles e hi' hi


open Classical in
/-- The morphism `K.truncGE' e ⟶ L.truncGE' e` induced by a morphism `K ⟶ L`. -/
noncomputable def truncGE'Map : K.truncGE' e ⟶ L.truncGE' e where
  f i :=
    if hi : e.BoundaryGE i
    then
      (K.truncGE'XIsoOpcycles e rfl hi).hom ≫ opcyclesMap φ (e.f i) ≫
        (L.truncGE'XIsoOpcycles e rfl hi).inv
    else
      (K.truncGE'XIso e rfl hi).hom ≫ φ.f (e.f i) ≫ (L.truncGE'XIso e rfl hi).inv
  comm' i j hij := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.20292, u_3} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c'
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝³ : e.IsTruncGE
      inst✝² : ∀ (i' : ι'), K.HasHomology i'
      inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
      inst✝ : ∀ (i' : ι'), M.HasHomology i'
      i j : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => dite (e.BoundaryGE i) (fun …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.20292, u_3} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c'
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝³ : e.IsTruncGE
      inst✝² : ∀ (i' : ι'), K.HasHomology i'
      inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
      inst✝ : ∀ (i' : ι'), M.HasHomology i'
      i j : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (e.BoundaryGE i) (fun hi => Cat …
    -/
    rw [dif_neg (e.not_boundaryGE_next hij)]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝⁵ : CategoryTheory.Category.{?u.20292, u_3} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c'
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝³ : e.IsTruncGE
      inst✝² : ∀ (i' : ι'), K.HasHomology i'
      inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
      inst✝ : ∀ (i' : ι'), M.HasHomology i'
      i j : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (e.BoundaryGE i) (fun hi => Cat …
    -/
    by_cases hi : e.BoundaryGE i
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝⁵ : CategoryTheory.Category.{?u.20292, u_3} C
        inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝³ : e.IsTruncGE
        inst✝² : ∀ (i' : ι'), K.HasHomology i'
        inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
        inst✝ : ∀ (i' : ι'), M.HasHomology i'
        i j : ι
        hij : c.Rel i j
        hi : e.BoundaryGE i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (e.BoundaryGE i) (fun hi => Cat …
      -/
    · rw [dif_pos hi]
      simp [truncGE'_d_eq_fromOpcycles _ e hij rfl rfl hi,
        ← cancel_epi (K.pOpcycles (e.f i))]
      /-
        case neg
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝⁵ : CategoryTheory.Category.{?u.20292, u_3} C
        inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝³ : e.IsTruncGE
        inst✝² : ∀ (i' : ι'), K.HasHomology i'
        inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
        inst✝ : ∀ (i' : ι'), M.HasHomology i'
        i j : ι
        hij : c.Rel i j
        hi : Not (e.BoundaryGE i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (e.BoundaryGE i) (fun hi => Cat …
      -/
    · rw [dif_neg hi]
      /-
        case neg
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝⁵ : CategoryTheory.Category.{?u.20292, u_3} C
        inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝³ : e.IsTruncGE
        inst✝² : ∀ (i' : ι'), K.HasHomology i'
        inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
        inst✝ : ∀ (i' : ι'), M.HasHomology i'
        i j : ι
        hij : c.Rel i j
        hi : Not (e.BoundaryGE i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp [truncGE'_d_eq _ e hij rfl rfl hi]
      /-
        🎉 no goals
      -/


lemma truncGE'Map_f_eq_opcyclesMap {i : ι} (hi : e.BoundaryGE i) {i' : ι'} (h : e.f i = i') :
    (truncGE'Map φ e).f i =
      (K.truncGE'XIsoOpcycles e h hi).hom ≫ opcyclesMap φ i' ≫
        (L.truncGE'XIsoOpcycles e h hi).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_3} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝² : e.IsTruncGE
    inst✝¹ : ∀ (i' : ι'), K.HasHomology i'
    inst✝ : ∀ (i' : ι'), L.HasHomology i'
    i : ι
    hi : e.BoundaryGE i
    i' : ι'
    h : Eq (e.f i) i'
    ⊢ Eq ((HomologicalComplex.truncGE'Map φ e).f i) (CategoryTheory.CategoryStruct …
  -/
  subst h
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_3} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝² : e.IsTruncGE
    inst✝¹ : ∀ (i' : ι'), K.HasHomology i'
    inst✝ : ∀ (i' : ι'), L.HasHomology i'
    i : ι
    hi : e.BoundaryGE i
    ⊢ Eq ((HomologicalComplex.truncGE'Map φ e).f i) (CategoryTheory.CategoryStruct …
  -/
  exact dif_pos hi
  /-
    🎉 no goals
  -/


lemma truncGE'Map_f_eq {i : ι} (hi : ¬ e.BoundaryGE i) {i' : ι'} (h : e.f i = i') :
    (truncGE'Map φ e).f i =
      (K.truncGE'XIso e h hi).hom ≫ φ.f i' ≫ (L.truncGE'XIso e h hi).inv := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_3} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝² : e.IsTruncGE
    inst✝¹ : ∀ (i' : ι'), K.HasHomology i'
    inst✝ : ∀ (i' : ι'), L.HasHomology i'
    i : ι
    hi : Not (e.BoundaryGE i)
    i' : ι'
    h : Eq (e.f i) i'
    ⊢ Eq ((HomologicalComplex.truncGE'Map φ e).f i) (CategoryTheory.CategoryStruct …
  -/
  subst h
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_3} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝² : e.IsTruncGE
    inst✝¹ : ∀ (i' : ι'), K.HasHomology i'
    inst✝ : ∀ (i' : ι'), L.HasHomology i'
    i : ι
    hi : Not (e.BoundaryGE i)
    ⊢ Eq ((HomologicalComplex.truncGE'Map φ e).f i) (CategoryTheory.CategoryStruct …
  -/
  exact dif_neg hi
  /-
    🎉 no goals
  -/


variable (K) in
@[simp]
lemma truncGE'Map_id : truncGE'Map (𝟙 K) e = 𝟙 _ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    ⊢ Eq (HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.id K) e) ( …
  -/
  ext i
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝¹ : e.IsTruncGE
    inst✝ : ∀ (i' : ι'), K.HasHomology i'
    i : ι
    ⊢ Eq ((HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.id K) e). …
  -/
  by_cases hi : e.BoundaryGE i
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝¹ : e.IsTruncGE
      inst✝ : ∀ (i' : ι'), K.HasHomology i'
      i : ι
      hi : e.BoundaryGE i
      ⊢ Eq ((HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.id K) e). …
    -/
  · simp [truncGE'Map_f_eq_opcyclesMap _ _ hi rfl]
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
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝¹ : e.IsTruncGE
      inst✝ : ∀ (i' : ι'), K.HasHomology i'
      i : ι
      hi : Not (e.BoundaryGE i)
      ⊢ Eq ((HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.id K) e). …
    -/
  · simp [truncGE'Map_f_eq _ _ hi rfl]
    /-
      🎉 no goals
    -/


@[reassoc, simp]
lemma truncGE'Map_comp : truncGE'Map (φ ≫ φ') e = truncGE'Map φ e ≫ truncGE'Map φ' e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_3} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    K L M : HomologicalComplex C c'
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝³ : e.IsTruncGE
    inst✝² : ∀ (i' : ι'), K.HasHomology i'
    inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
    inst✝ : ∀ (i' : ι'), M.HasHomology i'
    ⊢ Eq (HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.comp φ φ') …
  -/
  ext i
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_3} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    K L M : HomologicalComplex C c'
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝³ : e.IsTruncGE
    inst✝² : ∀ (i' : ι'), K.HasHomology i'
    inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
    inst✝ : ∀ (i' : ι'), M.HasHomology i'
    i : ι
    ⊢ Eq ((HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.comp φ φ' …
  -/
  by_cases hi : e.BoundaryGE i
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_3} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c'
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝³ : e.IsTruncGE
      inst✝² : ∀ (i' : ι'), K.HasHomology i'
      inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
      inst✝ : ∀ (i' : ι'), M.HasHomology i'
      i : ι
      hi : e.BoundaryGE i
      ⊢ Eq ((HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.comp φ φ' …
    -/
  · simp [truncGE'Map_f_eq_opcyclesMap _ _ hi rfl, opcyclesMap_comp]
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
      inst✝⁵ : CategoryTheory.Category.{u_4, u_3} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
      K L M : HomologicalComplex C c'
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝³ : e.IsTruncGE
      inst✝² : ∀ (i' : ι'), K.HasHomology i'
      inst✝¹ : ∀ (i' : ι'), L.HasHomology i'
      inst✝ : ∀ (i' : ι'), M.HasHomology i'
      i : ι
      hi : Not (e.BoundaryGE i)
      ⊢ Eq ((HomologicalComplex.truncGE'Map (CategoryTheory.CategoryStruct.comp φ φ' …
    -/
  · simp [truncGE'Map_f_eq _ _ hi rfl]
    /-
      🎉 no goals
    -/


/-- The morphism `K.truncGE e ⟶ L.truncGE e` induced by a morphism `K ⟶ L`. -/
noncomputable def truncGEMap : K.truncGE e ⟶ L.truncGE e :=
  (e.extendFunctor C).map (truncGE'Map φ e)


variable (K) in
@[simp]
lemma truncGEMap_id : truncGEMap (𝟙 K) e = 𝟙 _ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_3} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝² : e.IsTruncGE
    inst✝¹ : ∀ (i' : ι'), K.HasHomology i'
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (HomologicalComplex.truncGEMap (CategoryTheory.CategoryStruct.id K) e) (C …
  -/
  simp [truncGEMap, truncGE]
  /-
    🎉 no goals
  -/


@[reassoc, simp]
lemma truncGEMap_comp : truncGEMap (φ ≫ φ') e = truncGEMap φ e ≫ truncGEMap φ' e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝⁶ : CategoryTheory.Category.{u_4, u_3} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    K L M : HomologicalComplex C c'
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝⁴ : e.IsTruncGE
    inst✝³ : ∀ (i' : ι'), K.HasHomology i'
    inst✝² : ∀ (i' : ι'), L.HasHomology i'
    inst✝¹ : ∀ (i' : ι'), M.HasHomology i'
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (HomologicalComplex.truncGEMap (CategoryTheory.CategoryStruct.comp φ φ')  …
  -/
  simp [truncGEMap, truncGE]
  /-
    🎉 no goals
  -/


/-- Given an embedding `e : Embedding c c'` of complex shapes which satisfy `e.IsTruncGE`,
this is the (canonical) truncation functor
`HomologicalComplex C c' ⥤ HomologicalComplex C c`. -/
@[simps]
noncomputable def truncGE'Functor :
    HomologicalComplex C c' ⥤ HomologicalComplex C c where
  obj K := K.truncGE' e
  map φ := HomologicalComplex.truncGE'Map φ e


/-- Given an embedding `e : Embedding c c'` of complex shapes which satisfy `e.IsTruncGE`,
this is the (canonical) truncation functor
`HomologicalComplex C c' ⥤ HomologicalComplex C c'`. -/
@[simps]
noncomputable def truncGEFunctor :
    HomologicalComplex C c' ⥤ HomologicalComplex C c' where
  obj K := K.truncGE e
  map φ := HomologicalComplex.truncGEMap φ e


