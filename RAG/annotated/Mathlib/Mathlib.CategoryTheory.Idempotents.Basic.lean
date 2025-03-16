/-- A category is idempotent complete iff all idempotent endomorphisms `p`
split as a composition `p = e ≫ i` with `i ≫ e = 𝟙 _` -/
class IsIdempotentComplete : Prop where
  /-- A category is idempotent complete iff all idempotent endomorphisms `p`
    split as a composition `p = e ≫ i` with `i ≫ e = 𝟙 _` -/
  idempotents_split :
    ∀ (X : C) (p : X ⟶ X), p ≫ p = p → ∃ (Y : C) (i : Y ⟶ X) (e : X ⟶ Y), i ≫ e = 𝟙 Y ∧ e ≫ i = p


/-- A category is idempotent complete iff for all idempotent endomorphisms,
the equalizer of the identity and this idempotent exists. -/
theorem isIdempotentComplete_iff_hasEqualizer_of_id_and_idempotent :
    IsIdempotentComplete C ↔ ∀ (X : C) (p : X ⟶ X), p ≫ p = p → HasEqualizer (𝟙 X) p := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    ⊢ Iff (CategoryTheory.IsIdempotentComplete C) (∀ (X : C) (p : Quiver.Hom X X), …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      ⊢ CategoryTheory.IsIdempotentComplete C → ∀ (X : C) (p : Quiver.Hom X X), Eq ( …
    -/
  · intro
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      a✝ : CategoryTheory.IsIdempotentComplete C
      ⊢ ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p)  …
    -/
    intro X p hp
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      a✝ : CategoryTheory.IsIdempotentComplete C
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      ⊢ CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) p
    -/
    rcases IsIdempotentComplete.idempotents_split X p hp with ⟨Y, i, e, ⟨h₁, h₂⟩⟩
    exact
      ⟨Nonempty.intro
          { cone := Fork.ofι i (show i ≫ 𝟙 X = i ≫ p by rw [comp_id, ← h₂, ← assoc, h₁, id_comp])
            isLimit := by
              apply Fork.IsLimit.mk'
              intro s
              refine ⟨s.ι ≫ e, ?_⟩
              constructor
              · erw [assoc, h₂, ← Limits.Fork.condition s, comp_id]
              · intro m hm
                rw [Fork.ι_ofι] at hm
                rw [← hm]
                simp only [← hm, assoc, h₁]
                exact (comp_id m).symm }⟩
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      ⊢ (∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p) …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      ⊢ CategoryTheory.IsIdempotentComplete C
    -/
    refine ⟨?_⟩
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      ⊢ ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p)  …
    -/
    intro X p hp
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
    -/
    haveI : HasEqualizer (𝟙 X) p := h X p hp
    refine ⟨equalizer (𝟙 X) p, equalizer.ι (𝟙 X) p,
      equalizer.lift p (show p ≫ 𝟙 X = p ≫ p by rw [hp, comp_id]), ?_, equalizer.lift_ι _ _⟩
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      this : CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) p
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι (C …
    -/
    ext
    simp only [assoc, limit.lift_π, Eq.ndrec, id_eq, eq_mpr_eq_cast, Fork.ofι_pt,
      Fork.ofι_π_app, id_comp]
    /-
      case mpr.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      this : CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) p
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι (C …
    -/
    rw [← equalizer.condition, comp_id]
    /-
      🎉 no goals
    -/


/-- In a preadditive category, when `p : X ⟶ X` is idempotent,
then `𝟙 X - p` is also idempotent. -/
theorem idem_of_id_sub_idem [Preadditive C] {X : C} (p : X ⟶ X) (hp : p ≫ p = p) :
    (𝟙 _ - p) ≫ (𝟙 _ - p) = 𝟙 _ - p := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : C
    p : Quiver.Hom X X
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
  -/
  simp only [comp_sub, sub_comp, id_comp, comp_id, hp, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


/-- A preadditive category is pseudoabelian iff all idempotent endomorphisms have a kernel. -/
theorem isIdempotentComplete_iff_idempotents_have_kernels [Preadditive C] :
    IsIdempotentComplete C ↔ ∀ (X : C) (p : X ⟶ X), p ≫ p = p → HasKernel p := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    ⊢ Iff (CategoryTheory.IsIdempotentComplete C) (∀ (X : C) (p : Quiver.Hom X X), …
  -/
  rw [isIdempotentComplete_iff_hasEqualizer_of_id_and_idempotent]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    ⊢ Iff (∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ⊢ (∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p) …
    -/
  · intro h X p hp
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      ⊢ CategoryTheory.Limits.HasKernel p
    -/
    haveI : HasEqualizer (𝟙 X) (𝟙 X - p) := h X (𝟙 _ - p) (idem_of_id_sub_idem p hp)
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      this : CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) …
      ⊢ CategoryTheory.Limits.HasKernel p
    -/
    convert hasKernel_of_hasEqualizer (𝟙 X) (𝟙 X - p)
    /-
      case h.e'_6
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      this : CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) …
      ⊢ Eq p (HSub.hSub (CategoryTheory.CategoryStruct.id X) (HSub.hSub (CategoryThe …
    -/
    rw [sub_sub_cancel]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      ⊢ (∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p) …
    -/
  · intro h X p hp
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      ⊢ CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) p
    -/
    haveI : HasKernel (𝟙 _ - p) := h X (𝟙 _ - p) (idem_of_id_sub_idem p hp)
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      h : ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p …
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      this : CategoryTheory.Limits.HasKernel (HSub.hSub (CategoryTheory.CategoryStru …
      ⊢ CategoryTheory.Limits.HasEqualizer (CategoryTheory.CategoryStruct.id X) p
    -/
    apply Preadditive.hasEqualizer_of_hasKernel
    /-
      🎉 no goals
    -/


/-- An abelian category is idempotent complete. -/
instance (priority := 100) isIdempotentComplete_of_abelian (D : Type*) [Category D] [Abelian D] :
    IsIdempotentComplete D := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.10565, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    inst✝ : CategoryTheory.Abelian D
    ⊢ CategoryTheory.IsIdempotentComplete D
  -/
  rw [isIdempotentComplete_iff_idempotents_have_kernels]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.10565, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    inst✝ : CategoryTheory.Abelian D
    ⊢ ∀ (X : D) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p)  …
  -/
  intros
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.10565, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    inst✝ : CategoryTheory.Abelian D
    X✝ : D
    p✝ : Quiver.Hom X✝ X✝
    a✝ : Eq (CategoryTheory.CategoryStruct.comp p✝ p✝) p✝
    ⊢ CategoryTheory.Limits.HasKernel p✝
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem split_imp_of_iso {X X' : C} (φ : X ≅ X') (p : X ⟶ X) (p' : X' ⟶ X')
    (hpp' : p ≫ φ.hom = φ.hom ≫ p')
    (h : ∃ (Y : C) (i : Y ⟶ X) (e : X ⟶ Y), i ≫ e = 𝟙 Y ∧ e ≫ i = p) :
    ∃ (Y' : C) (i' : Y' ⟶ X') (e' : X' ⟶ Y'), i' ≫ e' = 𝟙 Y' ∧ e' ≫ i' = p' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X X' : C
    φ : CategoryTheory.Iso X X'
    p : Quiver.Hom X X
    p' : Quiver.Hom X' X'
    hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
    h : Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Ca …
    ⊢ Exists fun Y' => Exists fun i' => Exists fun e' => And (Eq (CategoryTheory.C …
  -/
  rcases h with ⟨Y, i, e, ⟨h₁, h₂⟩⟩
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X X' : C
    φ : CategoryTheory.Iso X X'
    p : Quiver.Hom X X
    p' : Quiver.Hom X' X'
    hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
    Y : C
    i : Quiver.Hom Y X
    e : Quiver.Hom X Y
    h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p
    ⊢ Exists fun Y' => Exists fun i' => Exists fun e' => And (Eq (CategoryTheory.C …
  -/
  use Y, i ≫ φ.hom, φ.inv ≫ e
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X X' : C
    φ : CategoryTheory.Iso X X'
    p : Quiver.Hom X X
    p' : Quiver.Hom X' X'
    hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
    Y : C
    i : Quiver.Hom Y X
    e : Quiver.Hom X Y
    h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.c …
  -/
  constructor
    /-
      case h.left
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      Y : C
      i : Quiver.Hom Y X
      e : Quiver.Hom X Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
    -/
  · slice_lhs 2 3 => rw [φ.hom_inv_id]
    /-
      case h.left
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      Y : C
      i : Quiver.Hom Y X
      e : Quiver.Hom X Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i (CategoryTheory.CategoryStruct.comp …
    -/
    rw [id_comp, h₁]
    /-
      🎉 no goals
    -/
    /-
      case h.right
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      Y : C
      i : Quiver.Hom Y X
      e : Quiver.Hom X Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
    -/
  · slice_lhs 2 3 => rw [h₂]
    /-
      case h.right
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      Y : C
      i : Quiver.Hom Y X
      e : Quiver.Hom X Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.inv (CategoryTheory.CategoryStruct. …
    -/
    rw [hpp', ← assoc, φ.inv_hom_id, id_comp]
    /-
      🎉 no goals
    -/


theorem split_iff_of_iso {X X' : C} (φ : X ≅ X') (p : X ⟶ X) (p' : X' ⟶ X')
    (hpp' : p ≫ φ.hom = φ.hom ≫ p') :
    (∃ (Y : C) (i : Y ⟶ X) (e : X ⟶ Y), i ≫ e = 𝟙 Y ∧ e ≫ i = p) ↔
      ∃ (Y' : C) (i' : Y' ⟶ X') (e' : X' ⟶ Y'), i' ≫ e' = 𝟙 Y' ∧ e' ≫ i' = p' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X X' : C
    φ : CategoryTheory.Iso X X'
    p : Quiver.Hom X X
    p' : Quiver.Hom X' X'
    hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
    ⊢ Iff (Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ (Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cat …
    -/
  · exact split_imp_of_iso φ p p' hpp'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ (Exists fun Y' => Exists fun i' => Exists fun e' => And (Eq (CategoryTheory. …
    -/
  · apply split_imp_of_iso φ.symm p' p
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp p' φ.symm.hom) (CategoryTheory.Catego …
    -/
    rw [← comp_id p, ← φ.hom_inv_id]
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp p' φ.symm.hom) (CategoryTheory.Catego …
    -/
    slice_rhs 2 3 => rw [hpp']
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp p' φ.symm.hom) (CategoryTheory.Catego …
    -/
    slice_rhs 1 2 => erw [φ.inv_hom_id]
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp p' φ.symm.hom) (CategoryTheory.Catego …
    -/
    simp only [id_comp]
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X X' : C
      φ : CategoryTheory.Iso X X'
      p : Quiver.Hom X X
      p' : Quiver.Hom X' X'
      hpp' : Eq (CategoryTheory.CategoryStruct.comp p φ.hom) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp p' φ.symm.hom) (CategoryTheory.Catego …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Equivalence.isIdempotentComplete {D : Type*} [Category D] (ε : C ≌ D)
    (h : IsIdempotentComplete C) : IsIdempotentComplete D := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    ε : CategoryTheory.Equivalence C D
    h : CategoryTheory.IsIdempotentComplete C
    ⊢ CategoryTheory.IsIdempotentComplete D
  -/
  refine ⟨?_⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    ε : CategoryTheory.Equivalence C D
    h : CategoryTheory.IsIdempotentComplete C
    ⊢ ∀ (X : D) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p)  …
  -/
  intro X' p hp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    ε : CategoryTheory.Equivalence C D
    h : CategoryTheory.IsIdempotentComplete C
    X' : D
    p : Quiver.Hom X' X'
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  let φ := ε.counitIso.symm.app X'
  erw [split_iff_of_iso φ p (φ.inv ≫ p ≫ φ.hom)
      (by
        slice_rhs 1 2 => rw [φ.hom_inv_id]
        rw [id_comp])]
  rcases IsIdempotentComplete.idempotents_split (ε.inverse.obj X') (ε.inverse.map p)
      (by rw [← ε.inverse.map_comp, hp]) with
    ⟨Y, i, e, ⟨h₁, h₂⟩⟩
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    ε : CategoryTheory.Equivalence C D
    h : CategoryTheory.IsIdempotentComplete C
    X' : D
    p : Quiver.Hom X' X'
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    φ : CategoryTheory.Iso ((CategoryTheory.Functor.id D).obj X') ((ε.inverse.comp …
    Y : C
    i : Quiver.Hom Y (ε.inverse.obj X')
    e : Quiver.Hom (ε.inverse.obj X') Y
    h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) (ε.inverse.map p)
    ⊢ Exists fun Y' => Exists fun i' => Exists fun e' => And (Eq (CategoryTheory.C …
  -/
  use ε.functor.obj Y, ε.functor.map i, ε.functor.map e
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    ε : CategoryTheory.Equivalence C D
    h : CategoryTheory.IsIdempotentComplete C
    X' : D
    p : Quiver.Hom X' X'
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    φ : CategoryTheory.Iso ((CategoryTheory.Functor.id D).obj X') ((ε.inverse.comp …
    Y : C
    i : Quiver.Hom Y (ε.inverse.obj X')
    e : Quiver.Hom (ε.inverse.obj X') Y
    h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) (ε.inverse.map p)
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (ε.functor.map i) (ε.functor.map …
  -/
  constructor
    /-
      case h.left
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      ε : CategoryTheory.Equivalence C D
      h : CategoryTheory.IsIdempotentComplete C
      X' : D
      p : Quiver.Hom X' X'
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      φ : CategoryTheory.Iso ((CategoryTheory.Functor.id D).obj X') ((ε.inverse.comp …
      Y : C
      i : Quiver.Hom Y (ε.inverse.obj X')
      e : Quiver.Hom (ε.inverse.obj X') Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) (ε.inverse.map p)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ε.functor.map i) (ε.functor.map e))  …
    -/
  · rw [← ε.functor.map_comp, h₁, ε.functor.map_id]
    /-
      🎉 no goals
    -/
    /-
      case h.right
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      ε : CategoryTheory.Equivalence C D
      h : CategoryTheory.IsIdempotentComplete C
      X' : D
      p : Quiver.Hom X' X'
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      φ : CategoryTheory.Iso ((CategoryTheory.Functor.id D).obj X') ((ε.inverse.comp …
      Y : C
      i : Quiver.Hom Y (ε.inverse.obj X')
      e : Quiver.Hom (ε.inverse.obj X') Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) (ε.inverse.map p)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ε.functor.map e) (ε.functor.map i))  …
    -/
  · simp only [← ε.functor.map_comp, h₂, Equivalence.fun_inv_map]
    /-
      case h.right
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      ε : CategoryTheory.Equivalence C D
      h : CategoryTheory.IsIdempotentComplete C
      X' : D
      p : Quiver.Hom X' X'
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      φ : CategoryTheory.Iso ((CategoryTheory.Functor.id D).obj X') ((ε.inverse.comp …
      Y : C
      i : Quiver.Hom Y (ε.inverse.obj X')
      e : Quiver.Hom (ε.inverse.obj X') Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) (ε.inverse.map p)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ε.counit.app X') (CategoryTheory.Cat …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `C` and `D` are equivalent categories, that `C` is idempotent complete iff `D` is. -/
theorem isIdempotentComplete_iff_of_equivalence {D : Type*} [Category D] (ε : C ≌ D) :
    IsIdempotentComplete C ↔ IsIdempotentComplete D := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} D
    ε : CategoryTheory.Equivalence C D
    ⊢ Iff (CategoryTheory.IsIdempotentComplete C) (CategoryTheory.IsIdempotentComp …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      ε : CategoryTheory.Equivalence C D
      ⊢ CategoryTheory.IsIdempotentComplete C → CategoryTheory.IsIdempotentComplete D
    -/
  · exact Equivalence.isIdempotentComplete ε
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} D
      ε : CategoryTheory.Equivalence C D
      ⊢ CategoryTheory.IsIdempotentComplete D → CategoryTheory.IsIdempotentComplete C
    -/
  · exact Equivalence.isIdempotentComplete ε.symm
    /-
      🎉 no goals
    -/


theorem isIdempotentComplete_of_isIdempotentComplete_opposite (h : IsIdempotentComplete Cᵒᵖ) :
    IsIdempotentComplete C := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    h : CategoryTheory.IsIdempotentComplete (Opposite C)
    ⊢ CategoryTheory.IsIdempotentComplete C
  -/
  refine ⟨?_⟩
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    h : CategoryTheory.IsIdempotentComplete (Opposite C)
    ⊢ ∀ (X : C) (p : Quiver.Hom X X), Eq (CategoryTheory.CategoryStruct.comp p p)  …
  -/
  intro X p hp
  rcases IsIdempotentComplete.idempotents_split (op X) p.op (by rw [← op_comp, hp]) with
    ⟨Y, i, e, ⟨h₁, h₂⟩⟩
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    h : CategoryTheory.IsIdempotentComplete (Opposite C)
    X : C
    p : Quiver.Hom X X
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    Y : Opposite C
    i : Quiver.Hom Y { unop := X }
    e : Quiver.Hom { unop := X } Y
    h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p.op
    ⊢ Exists fun Y => Exists fun i => Exists fun e => And (Eq (CategoryTheory.Cate …
  -/
  use Y.unop, e.unop, i.unop
  /-
    case h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    h : CategoryTheory.IsIdempotentComplete (Opposite C)
    X : C
    p : Quiver.Hom X X
    hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
    Y : Opposite C
    i : Quiver.Hom Y { unop := X }
    e : Quiver.Hom { unop := X } Y
    h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p.op
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp e.unop i.unop) (CategoryTheory.C …
  -/
  constructor
    /-
      case h.left
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete (Opposite C)
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      Y : Opposite C
      i : Quiver.Hom Y { unop := X }
      e : Quiver.Hom { unop := X } Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p.op
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e.unop i.unop) (CategoryTheory.Catego …
    -/
  · simp only [← unop_comp, h₁]
    /-
      case h.left
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete (Opposite C)
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      Y : Opposite C
      i : Quiver.Hom Y { unop := X }
      e : Quiver.Hom { unop := X } Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p.op
      ⊢ Eq (CategoryTheory.CategoryStruct.id Y).unop (CategoryTheory.CategoryStruct. …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.right
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete (Opposite C)
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      Y : Opposite C
      i : Quiver.Hom Y { unop := X }
      e : Quiver.Hom { unop := X } Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p.op
      ⊢ Eq (CategoryTheory.CategoryStruct.comp i.unop e.unop) p
    -/
  · simp only [← unop_comp, h₂]
    /-
      case h.right
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete (Opposite C)
      X : C
      p : Quiver.Hom X X
      hp : Eq (CategoryTheory.CategoryStruct.comp p p) p
      Y : Opposite C
      i : Quiver.Hom Y { unop := X }
      e : Quiver.Hom { unop := X } Y
      h₁ : Eq (CategoryTheory.CategoryStruct.comp i e) (CategoryTheory.CategoryStruc …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp e i) p.op
      ⊢ Eq p.op.unop p
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem isIdempotentComplete_iff_opposite : IsIdempotentComplete Cᵒᵖ ↔ IsIdempotentComplete C := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    ⊢ Iff (CategoryTheory.IsIdempotentComplete (Opposite C)) (CategoryTheory.IsIde …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      ⊢ CategoryTheory.IsIdempotentComplete (Opposite C) → CategoryTheory.IsIdempote …
    -/
  · exact isIdempotentComplete_of_isIdempotentComplete_opposite
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      ⊢ CategoryTheory.IsIdempotentComplete C → CategoryTheory.IsIdempotentComplete  …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete C
      ⊢ CategoryTheory.IsIdempotentComplete (Opposite C)
    -/
    apply isIdempotentComplete_of_isIdempotentComplete_opposite
    /-
      case mpr.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete C
      ⊢ CategoryTheory.IsIdempotentComplete (Opposite (Opposite C))
    -/
    rw [isIdempotentComplete_iff_of_equivalence (opOpEquivalence C)]
    /-
      case mpr.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : CategoryTheory.IsIdempotentComplete C
      ⊢ CategoryTheory.IsIdempotentComplete C
    -/
    exact h
    /-
      🎉 no goals
    -/


instance [IsIdempotentComplete C] : IsIdempotentComplete Cᵒᵖ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.IsIdempotentComplete C
    ⊢ CategoryTheory.IsIdempotentComplete (Opposite C)
  -/
  rwa [isIdempotentComplete_iff_opposite]
  /-
    🎉 no goals
  -/


