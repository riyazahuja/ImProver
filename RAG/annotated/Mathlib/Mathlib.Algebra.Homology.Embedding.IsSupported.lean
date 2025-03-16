/-- If `K : HomologicalComplex C c'`, then `K.IsStrictlySupported e` holds for
an embedding `e : c.Embedding c'` of complex shapes if `K.X i'` is zero
wheneverm `i'` is not of the form `e.f i` for some `i`.-/
class IsStrictlySupported : Prop where
  isZero (i' : ι') (hi' : ∀ i, e.f i ≠ i') : IsZero (K.X i')


lemma isZero_X_of_isStrictlySupported [K.IsStrictlySupported e]
    (i' : ι') (hi' : ∀ i, e.f i ≠ i') :
    IsZero (K.X i') :=
  IsStrictlySupported.isZero i' hi'


include e' in
variable {K L} in
lemma isStrictlySupported_of_iso [K.IsStrictlySupported e] : L.IsStrictlySupported e where
  isZero i' hi' := (K.isZero_X_of_isStrictlySupported e i' hi').of_iso
    ((eval _ _ i').mapIso e'.symm)


/-- If `K : HomologicalComplex C c'`, then `K.IsStrictlySupported e` holds for
an embedding `e : c.Embedding c'` of complex shapes if `K` is exact at `i'`
whenever `i'` is not of the form `e.f i` for some `i`.-/
class IsSupported : Prop where
  exactAt (i' : ι') (hi' : ∀ i, e.f i ≠ i') : K.ExactAt i'


lemma exactAt_of_isSupported [K.IsSupported e] (i' : ι') (hi' : ∀ i, e.f i ≠ i') :
    K.ExactAt i' :=
  IsSupported.exactAt i' hi'


include e' in
variable {K L} in
lemma isSupported_of_iso [K.IsSupported e] : L.IsSupported e where
  exactAt i' hi' :=
    (K.exactAt_of_isSupported e i' hi').of_iso e'


instance [K.IsStrictlySupported e] : K.IsSupported e where
  exactAt i' hi' := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      K L : HomologicalComplex C c'
      e' : CategoryTheory.Iso K L
      e : c.Embedding c'
      inst✝ : K.IsStrictlySupported e
      i' : ι'
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ K.ExactAt i'
    -/
    rw [exactAt_iff]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝² : CategoryTheory.Category.{u_4, u_3} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      K L : HomologicalComplex C c'
      e' : CategoryTheory.Iso K L
      e : c.Embedding c'
      inst✝ : K.IsStrictlySupported e
      i' : ι'
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ (K.sc i').Exact
    -/
    exact ShortComplex.exact_of_isZero_X₂ _ (K.isZero_X_of_isStrictlySupported e i' hi')
    /-
      🎉 no goals
    -/


/-- If `K : HomologicalComplex C c'`, then `K.IsStrictlySupportedOutside e` holds for
an embedding `e : c.Embedding c'` of complex shapes if `K.X (e.f i)` is zero for all `i`. -/
structure IsStrictlySupportedOutside : Prop where
  isZero (i : ι) : IsZero (K.X (e.f i))


/-- If `K : HomologicalComplex C c'`, then `K.IsSupportedOutside e` holds for
an embedding `e : c.Embedding c'` of complex shapes if `K` is exact at `e.f i` for all `i`. -/
structure IsSupportedOutside : Prop where
  exactAt (i : ι) : K.ExactAt (e.f i)


variable {K e} in
lemma IsStrictlySupportedOutside.isSupportedOutside (h : K.IsStrictlySupportedOutside e) :
    K.IsSupportedOutside e where
  exactAt i := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      h : K.IsStrictlySupportedOutside e
      i : ι
      ⊢ K.ExactAt (e.f i)
    -/
    rw [exactAt_iff]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      h : K.IsStrictlySupportedOutside e
      i : ι
      ⊢ (K.sc (e.f i)).Exact
    -/
    exact ShortComplex.exact_of_isZero_X₂ _ (h.isZero i)
    /-
      🎉 no goals
    -/


instance [HasZeroObject C] : (0 : HomologicalComplex C c').IsStrictlySupported e where
  isZero i _ := (eval _ _ i).map_isZero (Limits.isZero_zero _)


lemma isZero_iff_isStrictlySupported_and_isStrictlySupportedOutside :
    IsZero K ↔ K.IsStrictlySupported e ∧ K.IsStrictlySupportedOutside e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    ⊢ Iff (CategoryTheory.Limits.IsZero K) (And (K.IsStrictlySupported e) (K.IsStr …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      ⊢ CategoryTheory.Limits.IsZero K → And (K.IsStrictlySupported e) (K.IsStrictly …
    -/
  · intro hK
    /-
      case mp
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      hK : CategoryTheory.Limits.IsZero K
      ⊢ And (K.IsStrictlySupported e) (K.IsStrictlySupportedOutside e)
    -/
    constructor
    all_goals
      constructor
      intros
      exact (eval _ _ _).map_isZero hK
    /-
      case mpr
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      ⊢ And (K.IsStrictlySupported e) (K.IsStrictlySupportedOutside e) → CategoryThe …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      h₁ : K.IsStrictlySupported e
      h₂ : K.IsStrictlySupportedOutside e
      ⊢ CategoryTheory.Limits.IsZero K
    -/
    rw [IsZero.iff_id_eq_zero]
    /-
      case mpr.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      h₁ : K.IsStrictlySupported e
      h₂ : K.IsStrictlySupportedOutside e
      ⊢ Eq (CategoryTheory.CategoryStruct.id K) 0
    -/
    ext n
    /-
      case mpr.intro.h
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      h₁ : K.IsStrictlySupported e
      h₂ : K.IsStrictlySupportedOutside e
      n : ι'
      ⊢ Eq ((CategoryTheory.CategoryStruct.id K).f n) (HomologicalComplex.Hom.f 0 n)
    -/
    apply IsZero.eq_of_src
    /-
      case mpr.intro.h.hX
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      h₁ : K.IsStrictlySupported e
      h₂ : K.IsStrictlySupportedOutside e
      n : ι'
      ⊢ CategoryTheory.Limits.IsZero (K.X n)
    -/
    by_cases hn : ∃ i, e.f i = n
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        h₁ : K.IsStrictlySupported e
        h₂ : K.IsStrictlySupportedOutside e
        n : ι'
        hn : Exists fun i => Eq (e.f i) n
        ⊢ CategoryTheory.Limits.IsZero (K.X n)
      -/
    · obtain ⟨i, rfl⟩ := hn
      /-
        case pos.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        h₁ : K.IsStrictlySupported e
        h₂ : K.IsStrictlySupportedOutside e
        i : ι
        ⊢ CategoryTheory.Limits.IsZero (K.X (e.f i))
      -/
      exact h₂.isZero i
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
        inst✝¹ : CategoryTheory.Category.{u_4, u_3} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        h₁ : K.IsStrictlySupported e
        h₂ : K.IsStrictlySupportedOutside e
        n : ι'
        hn : Not (Exists fun i => Eq (e.f i) n)
        ⊢ CategoryTheory.Limits.IsZero (K.X n)
      -/
    · exact K.isZero_X_of_isStrictlySupported e _ (by simpa using hn)
      /-
        🎉 no goals
      -/


instance [K.IsStrictlySupported e] : K.op.IsStrictlySupported e.op where
  isZero j hj' := (K.isZero_X_of_isStrictlySupported e j hj').op


instance map_isStrictlySupported [K.IsStrictlySupported e] :
    ((F.mapHomologicalComplex c').obj K).IsStrictlySupported e where
  isZero i' hi' := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      D : Type u_4
      inst✝⁵ : CategoryTheory.Category.{u_5, u_3} C
      inst✝⁴ : CategoryTheory.Category.{u_6, u_4} D
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      K : HomologicalComplex C c'
      F : CategoryTheory.Functor C D
      inst✝¹ : F.PreservesZeroMorphisms
      e : c.Embedding c'
      inst✝ : K.IsStrictlySupported e
      i' : ι'
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ CategoryTheory.Limits.IsZero (((F.mapHomologicalComplex c').obj K).X i')
    -/
    rw [IsZero.iff_id_eq_zero]
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      D : Type u_4
      inst✝⁵ : CategoryTheory.Category.{u_5, u_3} C
      inst✝⁴ : CategoryTheory.Category.{u_6, u_4} D
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      K : HomologicalComplex C c'
      F : CategoryTheory.Functor C D
      inst✝¹ : F.PreservesZeroMorphisms
      e : c.Embedding c'
      inst✝ : K.IsStrictlySupported e
      i' : ι'
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ Eq (CategoryTheory.CategoryStruct.id (((F.mapHomologicalComplex c').obj K).X …
    -/
    dsimp
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      D : Type u_4
      inst✝⁵ : CategoryTheory.Category.{u_5, u_3} C
      inst✝⁴ : CategoryTheory.Category.{u_6, u_4} D
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      K : HomologicalComplex C c'
      F : CategoryTheory.Functor C D
      inst✝¹ : F.PreservesZeroMorphisms
      e : c.Embedding c'
      inst✝ : K.IsStrictlySupported e
      i' : ι'
      hi' : ∀ (i : ι), Ne (e.f i) i'
      ⊢ Eq (CategoryTheory.CategoryStruct.id (F.obj (K.X i'))) 0
    -/
    rw [← F.map_id, (K.isZero_X_of_isStrictlySupported e i' hi').eq_of_src (𝟙 _) 0, F.map_zero]
    /-
      🎉 no goals
    -/


