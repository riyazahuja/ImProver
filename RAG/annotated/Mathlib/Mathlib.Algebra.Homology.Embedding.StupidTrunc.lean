/-- The stupid truncation of a complex `K : HomologicalComplex C c'` relatively to
an embedding `e : c.Embedding c'` of complex shapes. -/
noncomputable def stupidTrunc : HomologicalComplex C c' := ((K.restriction e).extend e)


instance : IsStrictlySupported (K.stupidTrunc e) e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K L M : HomologicalComplex C c'
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    ⊢ (K.stupidTrunc e).IsStrictlySupported e
  -/
  dsimp [stupidTrunc]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K L M : HomologicalComplex C c'
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    ⊢ ((K.restriction e).extend e).IsStrictlySupported e
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The isomorphism `(K.stupidTrunc e).X i' ≅ K.X i'` when `i` is in the image of `e.f`. -/
noncomputable def stupidTruncXIso {i : ι} {i' : ι'} (hi' : e.f i = i') :
    (K.stupidTrunc e).X i' ≅ K.X i' :=
                                                    /-
                                                      ι : Type u_1
                                                      ι' : Type u_2
                                                      c : ComplexShape ι
                                                      c' : ComplexShape ι'
                                                      C : Type u_3
                                                      inst✝³ : CategoryTheory.Category.{?u.1532, u_3} C
                                                      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                      K L M : HomologicalComplex C c'
                                                      φ : Quiver.Hom K L
                                                      φ' : Quiver.Hom L M
                                                      e : c.Embedding c'
                                                      inst✝ : e.IsRelIff
                                                      i : ι
                                                      i' : ι'
                                                      hi' : Eq (e.f i) i'
                                                      ⊢ Eq ((K.restriction e).X i) (K.X i')
                                                    -/
  (K.restriction e).extendXIso e hi' ≪≫ eqToIso (by subst hi'; rfl)
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma isZero_stupidTrunc_X (i' : ι') (hi' : ∀ i, e.f i ≠ i') :
    IsZero ((K.stupidTrunc e).X i') :=
  isZero_extend_X _ _ _ hi'


instance {ι'' : Type*} {c'' : ComplexShape ι''} (e' : c''.Embedding c')
    [K.IsStrictlySupported e'] :
    IsStrictlySupported (K.stupidTrunc e) e' where
  isZero i' hi' := by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝⁴ : CategoryTheory.Category.{u_5, u_3} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      K L M : HomologicalComplex C c'
      φ : Quiver.Hom K L
      φ' : Quiver.Hom L M
      e : c.Embedding c'
      inst✝¹ : e.IsRelIff
      ι'' : Type u_4
      c'' : ComplexShape ι''
      e' : c''.Embedding c'
      inst✝ : K.IsStrictlySupported e'
      i' : ι'
      hi' : ∀ (i : ι''), Ne (e'.f i) i'
      ⊢ CategoryTheory.Limits.IsZero ((K.stupidTrunc e).X i')
    -/
    by_cases hi'' : ∃ i, e.f i = i'
      /-
        case pos
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_3} C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝¹ : e.IsRelIff
        ι'' : Type u_4
        c'' : ComplexShape ι''
        e' : c''.Embedding c'
        inst✝ : K.IsStrictlySupported e'
        i' : ι'
        hi' : ∀ (i : ι''), Ne (e'.f i) i'
        hi'' : Exists fun i => Eq (e.f i) i'
        ⊢ CategoryTheory.Limits.IsZero ((K.stupidTrunc e).X i')
      -/
    · obtain ⟨i, hi⟩ := hi''
      /-
        case pos.intro
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_3} C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝¹ : e.IsRelIff
        ι'' : Type u_4
        c'' : ComplexShape ι''
        e' : c''.Embedding c'
        inst✝ : K.IsStrictlySupported e'
        i' : ι'
        hi' : ∀ (i : ι''), Ne (e'.f i) i'
        i : ι
        hi : Eq (e.f i) i'
        ⊢ CategoryTheory.Limits.IsZero ((K.stupidTrunc e).X i')
      -/
      exact (K.isZero_X_of_isStrictlySupported e' i' hi').of_iso (K.stupidTruncXIso e hi)
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
        inst✝⁴ : CategoryTheory.Category.{u_5, u_3} C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝¹ : e.IsRelIff
        ι'' : Type u_4
        c'' : ComplexShape ι''
        e' : c''.Embedding c'
        inst✝ : K.IsStrictlySupported e'
        i' : ι'
        hi' : ∀ (i : ι''), Ne (e'.f i) i'
        hi'' : Not (Exists fun i => Eq (e.f i) i')
        ⊢ CategoryTheory.Limits.IsZero ((K.stupidTrunc e).X i')
      -/
    · apply isZero_stupidTrunc_X
      /-
        case neg.hi'
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝⁴ : CategoryTheory.Category.{u_5, u_3} C
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        K L M : HomologicalComplex C c'
        φ : Quiver.Hom K L
        φ' : Quiver.Hom L M
        e : c.Embedding c'
        inst✝¹ : e.IsRelIff
        ι'' : Type u_4
        c'' : ComplexShape ι''
        e' : c''.Embedding c'
        inst✝ : K.IsStrictlySupported e'
        i' : ι'
        hi' : ∀ (i : ι''), Ne (e'.f i) i'
        hi'' : Not (Exists fun i => Eq (e.f i) i')
        ⊢ ∀ (i : ι), Ne (e.f i) i'
      -/
      simpa using hi''
      /-
        🎉 no goals
      -/


lemma isZero_stupidTrunc_iff :
    IsZero (K.stupidTrunc e) ↔ K.IsStrictlySupportedOutside e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    ⊢ Iff (CategoryTheory.Limits.IsZero (K.stupidTrunc e)) (K.IsStrictlySupportedO …
  -/
  constructor
  · exact fun h ↦ ⟨fun i ↦
      ((eval _ _ (e.f i)).map_isZero h).of_iso (K.stupidTruncXIso e rfl).symm⟩
    /-
      case mpr
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      ⊢ K.IsStrictlySupportedOutside e → CategoryTheory.Limits.IsZero (K.stupidTrunc …
    -/
  · intro h
    /-
      case mpr
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      h : K.IsStrictlySupportedOutside e
      ⊢ CategoryTheory.Limits.IsZero (K.stupidTrunc e)
    -/
    rw [isZero_iff_isStrictlySupported_and_isStrictlySupportedOutside _ e]
    /-
      case mpr
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      C : Type u_3
      inst✝³ : CategoryTheory.Category.{u_4, u_3} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      K : HomologicalComplex C c'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      h : K.IsStrictlySupportedOutside e
      ⊢ And ((K.stupidTrunc e).IsStrictlySupported e) ((K.stupidTrunc e).IsStrictlyS …
    -/
    constructor
      /-
        case mpr.left
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        inst✝ : e.IsRelIff
        h : K.IsStrictlySupportedOutside e
        ⊢ (K.stupidTrunc e).IsStrictlySupported e
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
      /-
        case mpr.right
        ι : Type u_1
        ι' : Type u_2
        c : ComplexShape ι
        c' : ComplexShape ι'
        C : Type u_3
        inst✝³ : CategoryTheory.Category.{u_4, u_3} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        K : HomologicalComplex C c'
        e : c.Embedding c'
        inst✝ : e.IsRelIff
        h : K.IsStrictlySupportedOutside e
        ⊢ (K.stupidTrunc e).IsStrictlySupportedOutside e
      -/
    · exact ⟨fun i ↦ (h.isZero i).of_iso (K.stupidTruncXIso e rfl)⟩
      /-
        🎉 no goals
      -/


/-- The morphism `K.stupidTrunc e ⟶ L.stupidTrunc e` induced by a morphism `K ⟶ L`. -/
noncomputable def stupidTruncMap : K.stupidTrunc e ⟶ L.stupidTrunc e :=
  extendMap (restrictionMap φ e) e


variable (K) in
@[simp]
lemma stupidTruncMap_id : stupidTruncMap (𝟙 K) e = 𝟙 _ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K : HomologicalComplex C c'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    ⊢ Eq (HomologicalComplex.stupidTruncMap (CategoryTheory.CategoryStruct.id K) e …
  -/
  simp [stupidTruncMap, stupidTrunc]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma stupidTruncMap_comp :
    stupidTruncMap (φ ≫ φ') e = stupidTruncMap φ e ≫ stupidTruncMap φ' e := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K L M : HomologicalComplex C c'
    φ : Quiver.Hom K L
    φ' : Quiver.Hom L M
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    ⊢ Eq (HomologicalComplex.stupidTruncMap (CategoryTheory.CategoryStruct.comp φ  …
  -/
  simp [stupidTruncMap, stupidTrunc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma stupidTruncMap_stupidTruncXIso_hom {i : ι} {i' : ι'} (hi : e.f i = i') :
    (stupidTruncMap φ e).f i' ≫ (L.stupidTruncXIso e hi).hom =
      (K.stupidTruncXIso e hi).hom ≫ φ.f i' := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i : ι
    i' : ι'
    hi : Eq (e.f i) i'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.stupidTruncMap φ …
  -/
  subst hi
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    C : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_3} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    K L : HomologicalComplex C c'
    φ : Quiver.Hom K L
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.stupidTruncMap φ …
  -/
  simp [stupidTruncMap, stupidTruncXIso, extendMap_f _ _ rfl]
  /-
    🎉 no goals
  -/


/-- The stupid truncation functor `HomologicalComplex C c' ⥤ HomologicalComplex C c'`
given by an embedding `e : Embedding c c'` of complex shapes. -/
@[simps]
noncomputable def stupidTruncFunctor [e.IsRelIff] :
    HomologicalComplex C c' ⥤ HomologicalComplex C c' where
  obj K := K.stupidTrunc e
  map φ := HomologicalComplex.stupidTruncMap φ e


