/-- An object is simple if monomorphisms into it are (exclusively) either isomorphisms or zero. -/
class Simple (X : C) : Prop where
  mono_isIso_iff_nonzero : ∀ {Y : C} (f : Y ⟶ X) [Mono f], IsIso f ↔ f ≠ 0


/-- A nonzero monomorphism to a simple object is an isomorphism. -/
theorem isIso_of_mono_of_nonzero {X Y : C} [Simple Y] {f : X ⟶ Y} [Mono f] (w : f ≠ 0) : IsIso f :=
  (Simple.mono_isIso_iff_nonzero f).mpr w


theorem Simple.of_iso {X Y : C} [Simple Y] (i : X ≅ Y) : Simple X :=
  { mono_isIso_iff_nonzero := fun f m => by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        inst✝ : CategoryTheory.Simple Y
        i : CategoryTheory.Iso X Y
        Y✝ : C
        f : Quiver.Hom Y✝ X
        m : CategoryTheory.Mono f
        ⊢ Iff (CategoryTheory.IsIso f) (Ne f 0)
      -/
      constructor
        /-
          case mp
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          ⊢ CategoryTheory.IsIso f → Ne f 0
        -/
      · intro h w
        /-
          case mp
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          h : CategoryTheory.IsIso f
          w : Eq f 0
          ⊢ False
        -/
        have j : IsIso (f ≫ i.hom) := by infer_instance
        /-
          case mp
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          h : CategoryTheory.IsIso f
          w : Eq f 0
          j : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f i.hom)
          ⊢ False
        -/
        rw [Simple.mono_isIso_iff_nonzero] at j
        /-
          case mp
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          h : CategoryTheory.IsIso f
          w : Eq f 0
          j : Ne (CategoryTheory.CategoryStruct.comp f i.hom) 0
          ⊢ False
        -/
        subst w
        /-
          case mp
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          m : CategoryTheory.Mono 0
          h : CategoryTheory.IsIso 0
          j : Ne (CategoryTheory.CategoryStruct.comp 0 i.hom) 0
          ⊢ False
        -/
        simp at j
        /-
          🎉 no goals
        -/
        /-
          case mpr
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          ⊢ Ne f 0 → CategoryTheory.IsIso f
        -/
      · intro h
        have j : IsIso (f ≫ i.hom) := by
          apply isIso_of_mono_of_nonzero
          intro w
          apply h
          simpa using (cancel_mono i.inv).2 w
        /-
          case mpr
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          h : Ne f 0
          j : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f i.hom)
          ⊢ CategoryTheory.IsIso f
        -/
        rw [← Category.comp_id f, ← i.hom_inv_id, ← Category.assoc]
        /-
          case mpr
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          inst✝ : CategoryTheory.Simple Y
          i : CategoryTheory.Iso X Y
          Y✝ : C
          f : Quiver.Hom Y✝ X
          m : CategoryTheory.Mono f
          h : Ne f 0
          j : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f i.hom)
          ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
        -/
        infer_instance }
        /-
          🎉 no goals
        -/


theorem Simple.iff_of_iso {X Y : C} (i : X ≅ Y) : Simple X ↔ Simple Y :=
  ⟨fun _ => Simple.of_iso i.symm, fun _ => Simple.of_iso i⟩


theorem kernel_zero_of_nonzero_from_simple {X Y : C} [Simple X] {f : X ⟶ Y} [HasKernel f]
    (w : f ≠ 0) : kernel.ι f = 0 := by
  classical
    by_contra h
    haveI := isIso_of_mono_of_nonzero h
    exact w (eq_zero_of_epi_kernel f)

-- See also `mono_of_nonzero_from_simple`, which requires `Preadditive C`.

/-- A nonzero morphism `f` to a simple object is an epimorphism
(assuming `f` has an image, and `C` has equalizers).
-/
theorem epi_of_nonzero_to_simple [HasEqualizers C] {X Y : C} [Simple Y] {f : X ⟶ Y} [HasImage f]
    (w : f ≠ 0) : Epi f := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    X Y : C
    inst✝¹ : CategoryTheory.Simple Y
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    w : Ne f 0
    ⊢ CategoryTheory.Epi f
  -/
  rw [← image.fac f]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    X Y : C
    inst✝¹ : CategoryTheory.Simple Y
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    w : Ne f 0
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  haveI : IsIso (image.ι f) := isIso_of_mono_of_nonzero fun h => w (eq_zero_of_image_eq_zero h)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    X Y : C
    inst✝¹ : CategoryTheory.Simple Y
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    w : Ne f 0
    this : CategoryTheory.IsIso (CategoryTheory.Limits.image.ι f)
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  apply epi_comp
  /-
    🎉 no goals
  -/


theorem mono_to_simple_zero_of_not_iso {X Y : C} [Simple Y] {f : X ⟶ Y} [Mono f]
    (w : IsIso f → False) : f = 0 := by
  classical
    by_contra h
    exact w (isIso_of_mono_of_nonzero h)


theorem id_nonzero (X : C) [Simple.{v} X] : 𝟙 X ≠ 0 :=
                                               /-
                                                 C : Type u
                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                 X : C
                                                 inst✝ : CategoryTheory.Simple X
                                                 ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id X)
                                               -/
  (Simple.mono_isIso_iff_nonzero (𝟙 X)).mp (by infer_instance)
                                               /-
                                                 🎉 no goals
                                               -/


instance (X : C) [Simple.{v} X] : Nontrivial (End X) :=
  nontrivial_of_ne 1 _ (id_nonzero X)


theorem Simple.not_isZero (X : C) [Simple X] : ¬IsZero X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    inst✝ : CategoryTheory.Simple X
    ⊢ Not (CategoryTheory.Limits.IsZero X)
  -/
  simpa [Limits.IsZero.iff_id_eq_zero] using id_nonzero X
  /-
    🎉 no goals
  -/


/-- We don't want the definition of 'simple' to include the zero object, so we check that here. -/
theorem zero_not_simple [Simple (0 : C)] : False :=
                                                                     /-
                                                                       C : Type u
                                                                       inst✝³ : CategoryTheory.Category.{v, u} C
                                                                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                       inst✝ : CategoryTheory.Simple 0
                                                                       ⊢ And (Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStr …
                                                                     -/
  (Simple.mono_isIso_iff_nonzero (0 : (0 : C) ⟶ (0 : C))).mp ⟨⟨0, by aesop_cat⟩⟩ rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- In an abelian category, an object satisfying the dual of the definition of a simple object is
    simple. -/
theorem simple_of_cosimple (X : C) (h : ∀ {Z : C} (f : X ⟶ Z) [Epi f], IsIso f ↔ f ≠ 0) :
    Simple X :=
  ⟨fun {Y} f I => by
    classical
      fconstructor
      · intros
        have hx := cokernel.π_of_epi f
        by_contra h
        subst h
        exact (h _).mp (cokernel.π_of_zero _ _) hx
      · intro hf
        suffices Epi f by exact isIso_of_mono_of_epi _
        apply Preadditive.epi_of_cokernel_zero
        by_contra h'
        exact cokernel_not_iso_of_nonzero hf ((h _).mpr h')⟩


/-- A nonzero epimorphism from a simple object is an isomorphism. -/
theorem isIso_of_epi_of_nonzero {X Y : C} [Simple X] {f : X ⟶ Y} [Epi f] (w : f ≠ 0) : IsIso f :=
  -- `f ≠ 0` means that `kernel.ι f` is not an iso, and hence zero, and hence `f` is a mono.
  haveI : Mono f :=
    Preadditive.mono_of_kernel_zero (mono_to_simple_zero_of_not_iso (kernel_not_iso_of_nonzero w))
  isIso_of_mono_of_epi f


theorem cokernel_zero_of_nonzero_to_simple {X Y : C} [Simple Y] {f : X ⟶ Y} (w : f ≠ 0) :
    cokernel.π f = 0 := by
  classical
    by_contra h
    haveI := isIso_of_epi_of_nonzero h
    exact w (eq_zero_of_mono_cokernel f)


theorem epi_from_simple_zero_of_not_iso {X Y : C} [Simple X] {f : X ⟶ Y} [Epi f]
    (w : IsIso f → False) : f = 0 := by
  classical
    by_contra h
    exact w (isIso_of_epi_of_nonzero h)


theorem Biprod.isIso_inl_iff_isZero (X Y : C) : IsIso (biprod.inl : X ⟶ X ⊞ Y) ↔ IsZero Y := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X Y : C
    ⊢ Iff (CategoryTheory.IsIso CategoryTheory.Limits.biprod.inl) (CategoryTheory. …
  -/
  rw [biprod.isIso_inl_iff_id_eq_fst_comp_inl, ← biprod.total, add_right_eq_self]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X Y : C
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.snd …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.snd Cate …
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      X Y : C
      h : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.snd Ca …
      ⊢ CategoryTheory.Limits.IsZero Y
    -/
    replace h := h =≫ biprod.snd
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      X Y : C
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ CategoryTheory.Limits.IsZero Y
    -/
    simpa [← IsZero.iff_isSplitEpi_eq_zero (biprod.snd : X ⊞ Y ⟶ Y)] using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      X Y : C
      ⊢ CategoryTheory.Limits.IsZero Y → Eq (CategoryTheory.CategoryStruct.comp Cate …
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      X Y : C
      h : CategoryTheory.Limits.IsZero Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.snd Cate …
    -/
    rw [IsZero.iff_isSplitEpi_eq_zero (biprod.snd : X ⊞ Y ⟶ Y)] at h
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      X Y : C
      h : Eq CategoryTheory.Limits.biprod.snd 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.snd Cate …
    -/
    rw [h, zero_comp]
    /-
      🎉 no goals
    -/


/-- Any simple object in a preadditive category is indecomposable. -/
theorem indecomposable_of_simple (X : C) [Simple X] : Indecomposable X :=
  ⟨Simple.not_isZero X, fun Y Z i => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X : C
      inst✝ : CategoryTheory.Simple X
      Y Z : C
      i : CategoryTheory.Iso X (CategoryTheory.Limits.biprod Y Z)
      ⊢ Or (CategoryTheory.Limits.IsZero Y) (CategoryTheory.Limits.IsZero Z)
    -/
    refine or_iff_not_imp_left.mpr fun h => ?_
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X : C
      inst✝ : CategoryTheory.Simple X
      Y Z : C
      i : CategoryTheory.Iso X (CategoryTheory.Limits.biprod Y Z)
      h : Not (CategoryTheory.Limits.IsZero Y)
      ⊢ CategoryTheory.Limits.IsZero Z
    -/
    rw [IsZero.iff_isSplitMono_eq_zero (biprod.inl : Y ⟶ Y ⊞ Z)] at h
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X : C
      inst✝ : CategoryTheory.Simple X
      Y Z : C
      i : CategoryTheory.Iso X (CategoryTheory.Limits.biprod Y Z)
      h : Not (Eq CategoryTheory.Limits.biprod.inl 0)
      ⊢ CategoryTheory.Limits.IsZero Z
    -/
    change biprod.inl ≠ 0 at h
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X : C
      inst✝ : CategoryTheory.Simple X
      Y Z : C
      i : CategoryTheory.Iso X (CategoryTheory.Limits.biprod Y Z)
      h : Ne CategoryTheory.Limits.biprod.inl 0
      ⊢ CategoryTheory.Limits.IsZero Z
    -/
    have : Simple (Y ⊞ Z) := Simple.of_iso i.symm -- Porting note: this instance is needed
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X : C
      inst✝ : CategoryTheory.Simple X
      Y Z : C
      i : CategoryTheory.Iso X (CategoryTheory.Limits.biprod Y Z)
      h : Ne CategoryTheory.Limits.biprod.inl 0
      this : CategoryTheory.Simple (CategoryTheory.Limits.biprod Y Z)
      ⊢ CategoryTheory.Limits.IsZero Z
    -/
    rw [← Simple.mono_isIso_iff_nonzero biprod.inl] at h
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X : C
      inst✝ : CategoryTheory.Simple X
      Y Z : C
      i : CategoryTheory.Iso X (CategoryTheory.Limits.biprod Y Z)
      h : CategoryTheory.IsIso CategoryTheory.Limits.biprod.inl
      this : CategoryTheory.Simple (CategoryTheory.Limits.biprod Y Z)
      ⊢ CategoryTheory.Limits.IsZero Z
    -/
    rwa [Biprod.isIso_inl_iff_isZero] at h⟩
    /-
      🎉 no goals
    -/


instance {X : C} [Simple X] : Nontrivial (Subobject X) :=
  nontrivial_of_not_isZero (Simple.not_isZero X)


instance {X : C} [Simple X] : IsSimpleOrder (Subobject X) where
  eq_bot_or_eq_top := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : CategoryTheory.Simple X
      ⊢ ∀ (a : CategoryTheory.Subobject X), Or (Eq a Bot.bot) (Eq a Top.top)
    -/
    rintro ⟨⟨⟨Y : C, ⟨⟨⟩⟩, f : Y ⟶ X⟩, m : Mono f⟩⟩
    /-
      case mk.mk.mk.mk.unit
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : CategoryTheory.Simple X
      a✝ : CategoryTheory.Subobject X
      Y : C
      f : Quiver.Hom Y X
      m : CategoryTheory.Mono f
      ⊢ Or (Eq (Quot.mk ⇑(CategoryTheory.isIsomorphicSetoid (CategoryTheory.MonoOver …
    -/
    change mk f = ⊥ ∨ mk f = ⊤
    /-
      case mk.mk.mk.mk.unit
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : CategoryTheory.Simple X
      a✝ : CategoryTheory.Subobject X
      Y : C
      f : Quiver.Hom Y X
      m : CategoryTheory.Mono f
      ⊢ Or (Eq (CategoryTheory.Subobject.mk f) Bot.bot) (Eq (CategoryTheory.Subobjec …
    -/
    by_cases h : f = 0
      /-
        case pos
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        X : C
        inst✝ : CategoryTheory.Simple X
        a✝ : CategoryTheory.Subobject X
        Y : C
        f : Quiver.Hom Y X
        m : CategoryTheory.Mono f
        h : Eq f 0
        ⊢ Or (Eq (CategoryTheory.Subobject.mk f) Bot.bot) (Eq (CategoryTheory.Subobjec …
      -/
    · exact Or.inl (mk_eq_bot_iff_zero.mpr h)
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        X : C
        inst✝ : CategoryTheory.Simple X
        a✝ : CategoryTheory.Subobject X
        Y : C
        f : Quiver.Hom Y X
        m : CategoryTheory.Mono f
        h : Not (Eq f 0)
        ⊢ Or (Eq (CategoryTheory.Subobject.mk f) Bot.bot) (Eq (CategoryTheory.Subobjec …
      -/
    · refine Or.inr ((isIso_iff_mk_eq_top _).mp ((Simple.mono_isIso_iff_nonzero f).mpr h))
      /-
        🎉 no goals
      -/


/-- If `X` has subobject lattice `{⊥, ⊤}`, then `X` is simple. -/
theorem simple_of_isSimpleOrder_subobject (X : C) [IsSimpleOrder (Subobject X)] : Simple X := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
    ⊢ CategoryTheory.Simple X
  -/
  constructor; intros Y f hf; constructor
    /-
      case mono_isIso_iff_nonzero.mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      ⊢ CategoryTheory.IsIso f → Ne f 0
    -/
  · intro i
    /-
      case mono_isIso_iff_nonzero.mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      i : CategoryTheory.IsIso f
      ⊢ Ne f 0
    -/
    rw [Subobject.isIso_iff_mk_eq_top] at i
    /-
      case mono_isIso_iff_nonzero.mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      i : Eq (CategoryTheory.Subobject.mk f) Top.top
      ⊢ Ne f 0
    -/
    intro w
    /-
      case mono_isIso_iff_nonzero.mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      i : Eq (CategoryTheory.Subobject.mk f) Top.top
      w : Eq f 0
      ⊢ False
    -/
    rw [← Subobject.mk_eq_bot_iff_zero] at w
    /-
      case mono_isIso_iff_nonzero.mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      i : Eq (CategoryTheory.Subobject.mk f) Top.top
      w : Eq (CategoryTheory.Subobject.mk f) Bot.bot
      ⊢ False
    -/
    exact IsSimpleOrder.bot_ne_top (w.symm.trans i)
    /-
      🎉 no goals
    -/
    /-
      case mono_isIso_iff_nonzero.mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      ⊢ Ne f 0 → CategoryTheory.IsIso f
    -/
  · intro i
    /-
      case mono_isIso_iff_nonzero.mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      X : C
      inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
      Y : C
      f : Quiver.Hom Y X
      hf : CategoryTheory.Mono f
      i : Ne f 0
      ⊢ CategoryTheory.IsIso f
    -/
    rcases IsSimpleOrder.eq_bot_or_eq_top (Subobject.mk f) with (h | h)
      /-
        case mono_isIso_iff_nonzero.mpr.inl
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        X : C
        inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
        Y : C
        f : Quiver.Hom Y X
        hf : CategoryTheory.Mono f
        i : Ne f 0
        h : Eq (CategoryTheory.Subobject.mk f) Bot.bot
        ⊢ CategoryTheory.IsIso f
      -/
    · rw [Subobject.mk_eq_bot_iff_zero] at h
      /-
        case mono_isIso_iff_nonzero.mpr.inl
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        X : C
        inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
        Y : C
        f : Quiver.Hom Y X
        hf : CategoryTheory.Mono f
        i : Ne f 0
        h : Eq f 0
        ⊢ CategoryTheory.IsIso f
      -/
      exact False.elim (i h)
      /-
        🎉 no goals
      -/
      /-
        case mono_isIso_iff_nonzero.mpr.inr
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        X : C
        inst✝ : IsSimpleOrder (CategoryTheory.Subobject X)
        Y : C
        f : Quiver.Hom Y X
        hf : CategoryTheory.Mono f
        i : Ne f 0
        h : Eq (CategoryTheory.Subobject.mk f) Top.top
        ⊢ CategoryTheory.IsIso f
      -/
    · exact (Subobject.isIso_iff_mk_eq_top _).mpr h
      /-
        🎉 no goals
      -/


/-- `X` is simple iff it has subobject lattice `{⊥, ⊤}`. -/
theorem simple_iff_subobject_isSimpleOrder (X : C) : Simple X ↔ IsSimpleOrder (Subobject X) :=
  ⟨by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : C
      ⊢ CategoryTheory.Simple X → IsSimpleOrder (CategoryTheory.Subobject X)
    -/
    intro h
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : C
      h : CategoryTheory.Simple X
      ⊢ IsSimpleOrder (CategoryTheory.Subobject X)
    -/
    infer_instance, by
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : C
      ⊢ IsSimpleOrder (CategoryTheory.Subobject X) → CategoryTheory.Simple X
    -/
    intro h
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : C
      h : IsSimpleOrder (CategoryTheory.Subobject X)
      ⊢ CategoryTheory.Simple X
    -/
    exact simple_of_isSimpleOrder_subobject X⟩
    /-
      🎉 no goals
    -/


/-- A subobject is simple iff it is an atom in the subobject lattice. -/
theorem subobject_simple_iff_isAtom {X : C} (Y : Subobject X) : Simple (Y : C) ↔ IsAtom Y :=
  (simple_iff_subobject_isSimpleOrder _).trans
    ((OrderIso.isSimpleOrder_iff (subobjectOrderIso Y)).trans Set.isSimpleOrder_Iic_iff_isAtom)


