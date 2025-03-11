theorem mono_of_nonzero_from_simple [HasKernels C] {X Y : C} [Simple X] {f : X ⟶ Y} (w : f ≠ 0) :
    Mono f :=
  Preadditive.mono_of_kernel_zero (kernel_zero_of_nonzero_from_simple w)


/-- The part of **Schur's lemma** that holds in any preadditive category with kernels:
that a nonzero morphism between simple objects is an isomorphism.
-/
theorem isIso_of_hom_simple
    [HasKernels C] {X Y : C} [Simple X] [Simple Y] {f : X ⟶ Y} (w : f ≠ 0) : IsIso f :=
  haveI := mono_of_nonzero_from_simple w
  isIso_of_mono_of_nonzero w


/-- As a corollary of Schur's lemma for preadditive categories,
any morphism between simple objects is (exclusively) either an isomorphism or zero.
-/
theorem isIso_iff_nonzero [HasKernels C] {X Y : C} [Simple X] [Simple Y] (f : X ⟶ Y) :
    IsIso f ↔ f ≠ 0 :=
  ⟨fun I => by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : Quiver.Hom X Y
      I : CategoryTheory.IsIso f
      ⊢ Ne f 0
    -/
    intro h
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : Quiver.Hom X Y
      I : CategoryTheory.IsIso f
      h : Eq f 0
      ⊢ False
    -/
    apply id_nonzero X
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : Quiver.Hom X Y
      I : CategoryTheory.IsIso f
      h : Eq f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0
    -/
    simp only [← IsIso.hom_inv_id f, h, zero_comp],
    /-
      🎉 no goals
    -/
   fun w => isIso_of_hom_simple w⟩


open scoped Classical in
/-- In any preadditive category with kernels,
the endomorphisms of a simple object form a division ring. -/
noncomputable instance [HasKernels C] {X : C} [Simple X] : DivisionRing (End X) where
  inv f := if h : f = 0 then 0 else haveI := isIso_of_hom_simple h; inv f
  exists_pair_ne := ⟨𝟙 X, 0, id_nonzero _⟩
  inv_zero := dif_pos rfl
  mul_inv_cancel f hf := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.5925, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      X : C
      inst✝ : CategoryTheory.Simple X
      f : CategoryTheory.End X
      hf : Ne f 0
      ⊢ Eq (HMul.hMul f (Inv.inv f)) 1
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.5925, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      X : C
      inst✝ : CategoryTheory.Simple X
      f : CategoryTheory.End X
      hf : Ne f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq f 0) (fun h => 0) fun h =>  …
    -/
    rw [dif_neg hf]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.5925, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      X : C
      inst✝ : CategoryTheory.Simple X
      f : CategoryTheory.End X
      hf : Ne f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv f) f) (CategoryTh …
    -/
    haveI := isIso_of_hom_simple hf
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.5925, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      X : C
      inst✝ : CategoryTheory.Simple X
      f : CategoryTheory.End X
      hf : Ne f 0
      this : CategoryTheory.IsIso f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv f) f) (CategoryTh …
    -/
    exact IsIso.inv_hom_id f
    /-
      🎉 no goals
    -/
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


/-- Part of **Schur's lemma** for `𝕜`-linear categories:
the hom space between two non-isomorphic simple objects is 0-dimensional.
-/
theorem finrank_hom_simple_simple_eq_zero_of_not_iso [HasKernels C] [Linear 𝕜 C] {X Y : C}
    [Simple X] [Simple Y] (h : (X ≅ Y) → False) : finrank 𝕜 (X ⟶ Y) = 0 :=
  haveI :=
    subsingleton_of_forall_eq (0 : X ⟶ Y) fun f => by
      /-
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        𝕜 : Type u_2
        inst✝⁴ : DivisionRing 𝕜
        inst✝³ : CategoryTheory.Limits.HasKernels C
        inst✝² : CategoryTheory.Linear 𝕜 C
        X Y : C
        inst✝¹ : CategoryTheory.Simple X
        inst✝ : CategoryTheory.Simple Y
        h : CategoryTheory.Iso X Y → False
        f : Quiver.Hom X Y
        ⊢ Eq f 0
      -/
      have p := not_congr (isIso_iff_nonzero f)
      /-
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        𝕜 : Type u_2
        inst✝⁴ : DivisionRing 𝕜
        inst✝³ : CategoryTheory.Limits.HasKernels C
        inst✝² : CategoryTheory.Linear 𝕜 C
        X Y : C
        inst✝¹ : CategoryTheory.Simple X
        inst✝ : CategoryTheory.Simple Y
        h : CategoryTheory.Iso X Y → False
        f : Quiver.Hom X Y
        p : Iff (Not (CategoryTheory.IsIso f)) (Not (Ne f 0))
        ⊢ Eq f 0
      -/
      simp only [Classical.not_not, Ne] at p
      /-
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁵ : CategoryTheory.Preadditive C
        𝕜 : Type u_2
        inst✝⁴ : DivisionRing 𝕜
        inst✝³ : CategoryTheory.Limits.HasKernels C
        inst✝² : CategoryTheory.Linear 𝕜 C
        X Y : C
        inst✝¹ : CategoryTheory.Simple X
        inst✝ : CategoryTheory.Simple Y
        h : CategoryTheory.Iso X Y → False
        f : Quiver.Hom X Y
        p : Iff (Not (CategoryTheory.IsIso f)) (Eq f 0)
        ⊢ Eq f 0
      -/
      exact p.mp fun _ => h (asIso f)
      /-
        🎉 no goals
      -/
  finrank_zero_of_subsingleton


/-- An auxiliary lemma for Schur's lemma.

If `X ⟶ X` is finite dimensional, and every nonzero endomorphism is invertible,
then `X ⟶ X` is 1-dimensional.
-/
theorem finrank_endomorphism_eq_one {X : C} (isIso_iff_nonzero : ∀ f : X ⟶ X, IsIso f ↔ f ≠ 0)
    [I : FiniteDimensional 𝕜 (X ⟶ X)] : finrank 𝕜 (X ⟶ X) = 1 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X X)) 1
  -/
  have id_nonzero := (isIso_iff_nonzero (𝟙 X)).mp (by infer_instance)
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X X)) 1
  -/
  refine finrank_eq_one (𝟙 X) id_nonzero ?_
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    ⊢ ∀ (w : Quiver.Hom X X), Exists fun c => Eq (HSMul.hSMul c (CategoryTheory.Ca …
  -/
  intro f
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    f : Quiver.Hom X X
    ⊢ Exists fun c => Eq (HSMul.hSMul c (CategoryTheory.CategoryStruct.id X)) f
  -/
  have : Nontrivial (End X) := nontrivial_of_ne _ _ id_nonzero
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    f : Quiver.Hom X X
    this : Nontrivial (CategoryTheory.End X)
    ⊢ Exists fun c => Eq (HSMul.hSMul c (CategoryTheory.CategoryStruct.id X)) f
  -/
  have : FiniteDimensional 𝕜 (End X) := I
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    f : Quiver.Hom X X
    this✝ : Nontrivial (CategoryTheory.End X)
    this : FiniteDimensional 𝕜 (CategoryTheory.End X)
    ⊢ Exists fun c => Eq (HSMul.hSMul c (CategoryTheory.CategoryStruct.id X)) f
  -/
  obtain ⟨c, nu⟩ := spectrum.nonempty_of_isAlgClosed_of_finiteDimensional 𝕜 (End.of f)
  /-
    case intro
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    f : Quiver.Hom X X
    this✝ : Nontrivial (CategoryTheory.End X)
    this : FiniteDimensional 𝕜 (CategoryTheory.End X)
    c : 𝕜
    nu : Membership.mem (spectrum 𝕜 (CategoryTheory.End.of f)) c
    ⊢ Exists fun c => Eq (HSMul.hSMul c (CategoryTheory.CategoryStruct.id X)) f
  -/
  use c
  rw [spectrum.mem_iff, IsUnit.sub_iff, isUnit_iff_isIso, isIso_iff_nonzero, Ne,
    Classical.not_not, sub_eq_zero, Algebra.algebraMap_eq_smul_one] at nu
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : IsAlgClosed 𝕜
    inst✝ : CategoryTheory.Linear 𝕜 C
    X : C
    isIso_iff_nonzero : ∀ (f : Quiver.Hom X X), Iff (CategoryTheory.IsIso f) (Ne f …
    I : FiniteDimensional 𝕜 (Quiver.Hom X X)
    id_nonzero : Ne (CategoryTheory.CategoryStruct.id X) 0
    f : Quiver.Hom X X
    this✝ : Nontrivial (CategoryTheory.End X)
    this : FiniteDimensional 𝕜 (CategoryTheory.End X)
    c : 𝕜
    nu : Eq (CategoryTheory.End.of f) (HSMul.hSMul c 1)
    ⊢ Eq (HSMul.hSMul c (CategoryTheory.CategoryStruct.id X)) f
  -/
  exact nu.symm
  /-
    🎉 no goals
  -/


/-- **Schur's lemma** for endomorphisms in `𝕜`-linear categories.
-/
theorem finrank_endomorphism_simple_eq_one (X : C) [Simple X] [FiniteDimensional 𝕜 (X ⟶ X)] :
    finrank 𝕜 (X ⟶ X) = 1 :=
  finrank_endomorphism_eq_one 𝕜 isIso_iff_nonzero


theorem endomorphism_simple_eq_smul_id {X : C} [Simple X] [FiniteDimensional 𝕜 (X ⟶ X)]
    (f : X ⟶ X) : ∃ c : 𝕜, c • 𝟙 X = f :=
  (finrank_eq_one_iff_of_nonzero' (𝟙 X) (id_nonzero X)).mp (finrank_endomorphism_simple_eq_one 𝕜 X)
    f


/-- Endomorphisms of a simple object form a field if they are finite dimensional.
This can't be an instance as `𝕜` would be undetermined.
-/
noncomputable def fieldEndOfFiniteDimensional (X : C) [Simple X] [I : FiniteDimensional 𝕜 (X ⟶ X)] :
    Field (End X) := by
  classical exact
    { (inferInstance : DivisionRing (End X)) with
      mul_comm := fun f g => by
        obtain ⟨c, rfl⟩ := endomorphism_simple_eq_smul_id 𝕜 f
        obtain ⟨d, rfl⟩ := endomorphism_simple_eq_smul_id 𝕜 g
        simp [← mul_smul, mul_comm c d] }

-- There is a symmetric argument that uses `[FiniteDimensional 𝕜 (Y ⟶ Y)]` instead,
-- but we don't bother proving that here.

/-- **Schur's lemma** for `𝕜`-linear categories:
if hom spaces are finite dimensional, then the hom space between simples is at most 1-dimensional.

See `finrank_hom_simple_simple_eq_one_iff` and `finrank_hom_simple_simple_eq_zero_iff` below
for the refinements when we know whether or not the simples are isomorphic.
-/
theorem finrank_hom_simple_simple_le_one (X Y : C) [FiniteDimensional 𝕜 (X ⟶ X)] [Simple X]
    [Simple Y] : finrank 𝕜 (X ⟶ Y) ≤ 1 := by
  /-
    C : Type u_1
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝⁶ : Field 𝕜
    inst✝⁵ : IsAlgClosed 𝕜
    inst✝⁴ : CategoryTheory.Linear 𝕜 C
    inst✝³ : CategoryTheory.Limits.HasKernels C
    X Y : C
    inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
    inst✝¹ : CategoryTheory.Simple X
    inst✝ : CategoryTheory.Simple Y
    ⊢ LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
  -/
  obtain (h|h) := subsingleton_or_nontrivial (X ⟶ Y)
    /-
      case inl
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Subsingleton (Quiver.Hom X Y)
      ⊢ LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
  · rw [finrank_zero_of_subsingleton]
    /-
      case inl
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Subsingleton (Quiver.Hom X Y)
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/
    /-
      case inr
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nontrivial (Quiver.Hom X Y)
      ⊢ LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
  · obtain ⟨f, nz⟩ := (nontrivial_iff_exists_ne 0).mp h
    /-
      case inr.intro
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nontrivial (Quiver.Hom X Y)
      f : Quiver.Hom X Y
      nz : Ne f 0
      ⊢ LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
    haveI fi := (isIso_iff_nonzero f).mpr nz
    /-
      case inr.intro
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nontrivial (Quiver.Hom X Y)
      f : Quiver.Hom X Y
      nz : Ne f 0
      fi : CategoryTheory.IsIso f
      ⊢ LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
    refine finrank_le_one f ?_
    /-
      case inr.intro
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nontrivial (Quiver.Hom X Y)
      f : Quiver.Hom X Y
      nz : Ne f 0
      fi : CategoryTheory.IsIso f
      ⊢ ∀ (w : Quiver.Hom X Y), Exists fun c => Eq (HSMul.hSMul c f) w
    -/
    intro g
    /-
      case inr.intro
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nontrivial (Quiver.Hom X Y)
      f : Quiver.Hom X Y
      nz : Ne f 0
      fi : CategoryTheory.IsIso f
      g : Quiver.Hom X Y
      ⊢ Exists fun c => Eq (HSMul.hSMul c f) g
    -/
    obtain ⟨c, w⟩ := endomorphism_simple_eq_smul_id 𝕜 (g ≫ inv f)
    /-
      case inr.intro.intro
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nontrivial (Quiver.Hom X Y)
      f : Quiver.Hom X Y
      nz : Ne f 0
      fi : CategoryTheory.IsIso f
      g : Quiver.Hom X Y
      c : 𝕜
      w : Eq (HSMul.hSMul c (CategoryTheory.CategoryStruct.id X)) (CategoryTheory.Ca …
      ⊢ Exists fun c => Eq (HSMul.hSMul c f) g
    -/
    exact ⟨c, by simpa using w =≫ f⟩
    /-
      🎉 no goals
    -/


theorem finrank_hom_simple_simple_eq_one_iff (X Y : C) [FiniteDimensional 𝕜 (X ⟶ X)]
    [FiniteDimensional 𝕜 (X ⟶ Y)] [Simple X] [Simple Y] :
    finrank 𝕜 (X ⟶ Y) = 1 ↔ Nonempty (X ≅ Y) := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁸ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝⁷ : Field 𝕜
    inst✝⁶ : IsAlgClosed 𝕜
    inst✝⁵ : CategoryTheory.Linear 𝕜 C
    inst✝⁴ : CategoryTheory.Limits.HasKernels C
    X Y : C
    inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
    inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
    inst✝¹ : CategoryTheory.Simple X
    inst✝ : CategoryTheory.Simple Y
    ⊢ Iff (Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1) (Nonempty (CategoryTheory.Iso …
  -/
  fconstructor
    /-
      case mp
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1 → Nonempty (CategoryTheory.Iso X Y)
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
      ⊢ Nonempty (CategoryTheory.Iso X Y)
    -/
    rw [finrank_eq_one_iff'] at h
    /-
      case mp
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Exists fun v => And (Ne v 0) (∀ (w : Quiver.Hom X Y), Exists fun c => Eq ( …
      ⊢ Nonempty (CategoryTheory.Iso X Y)
    -/
    obtain ⟨f, nz, -⟩ := h
    /-
      case mp.intro.intro
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : Quiver.Hom X Y
      nz : Ne f 0
      ⊢ Nonempty (CategoryTheory.Iso X Y)
    -/
    rw [← isIso_iff_nonzero] at nz
    /-
      case mp.intro.intro
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : Quiver.Hom X Y
      nz : CategoryTheory.IsIso f
      ⊢ Nonempty (CategoryTheory.Iso X Y)
    -/
    exact ⟨asIso f⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      ⊢ Nonempty (CategoryTheory.Iso X Y) → Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
  · rintro ⟨f⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : CategoryTheory.Iso X Y
      ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
    have le_one := finrank_hom_simple_simple_le_one 𝕜 X Y
    have zero_lt : 0 < finrank 𝕜 (X ⟶ Y) :=
      finrank_pos_iff_exists_ne_zero.mpr ⟨f.hom, (isIso_iff_nonzero f.hom).mp inferInstance⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁸ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁷ : Field 𝕜
      inst✝⁶ : IsAlgClosed 𝕜
      inst✝⁵ : CategoryTheory.Linear 𝕜 C
      inst✝⁴ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
      inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      f : CategoryTheory.Iso X Y
      le_one : LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
      zero_lt : LT.lt 0 (Module.finrank 𝕜 (Quiver.Hom X Y))
      ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
    omega
    /-
      🎉 no goals
    -/


theorem finrank_hom_simple_simple_eq_zero_iff (X Y : C) [FiniteDimensional 𝕜 (X ⟶ X)]
    [FiniteDimensional 𝕜 (X ⟶ Y)] [Simple X] [Simple Y] :
    finrank 𝕜 (X ⟶ Y) = 0 ↔ IsEmpty (X ≅ Y) := by
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁸ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝⁷ : Field 𝕜
    inst✝⁶ : IsAlgClosed 𝕜
    inst✝⁵ : CategoryTheory.Linear 𝕜 C
    inst✝⁴ : CategoryTheory.Limits.HasKernels C
    X Y : C
    inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
    inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
    inst✝¹ : CategoryTheory.Simple X
    inst✝ : CategoryTheory.Simple Y
    ⊢ Iff (Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 0) (IsEmpty (CategoryTheory.Iso  …
  -/
  rw [← not_nonempty_iff, ← not_congr (finrank_hom_simple_simple_eq_one_iff 𝕜 X Y)]
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁸ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝⁷ : Field 𝕜
    inst✝⁶ : IsAlgClosed 𝕜
    inst✝⁵ : CategoryTheory.Linear 𝕜 C
    inst✝⁴ : CategoryTheory.Limits.HasKernels C
    X Y : C
    inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
    inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
    inst✝¹ : CategoryTheory.Simple X
    inst✝ : CategoryTheory.Simple Y
    ⊢ Iff (Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 0) (Not (Eq (Module.finrank 𝕜 (Q …
  -/
  have := finrank_hom_simple_simple_le_one 𝕜 X Y
  /-
    C : Type u_1
    inst✝⁹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁸ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝⁷ : Field 𝕜
    inst✝⁶ : IsAlgClosed 𝕜
    inst✝⁵ : CategoryTheory.Linear 𝕜 C
    inst✝⁴ : CategoryTheory.Limits.HasKernels C
    X Y : C
    inst✝³ : FiniteDimensional 𝕜 (Quiver.Hom X X)
    inst✝² : FiniteDimensional 𝕜 (Quiver.Hom X Y)
    inst✝¹ : CategoryTheory.Simple X
    inst✝ : CategoryTheory.Simple Y
    this : LE.le (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    ⊢ Iff (Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 0) (Not (Eq (Module.finrank 𝕜 (Q …
  -/
  omega
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem finrank_hom_simple_simple (X Y : C) [∀ X Y : C, FiniteDimensional 𝕜 (X ⟶ Y)] [Simple X]
    [Simple Y] : finrank 𝕜 (X ⟶ Y) = if Nonempty (X ≅ Y) then 1 else 0 := by
  /-
    C : Type u_1
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Preadditive C
    𝕜 : Type u_2
    inst✝⁶ : Field 𝕜
    inst✝⁵ : IsAlgClosed 𝕜
    inst✝⁴ : CategoryTheory.Linear 𝕜 C
    inst✝³ : CategoryTheory.Limits.HasKernels C
    X Y : C
    inst✝² : ∀ (X Y : C), FiniteDimensional 𝕜 (Quiver.Hom X Y)
    inst✝¹ : CategoryTheory.Simple X
    inst✝ : CategoryTheory.Simple Y
    ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) (ite (Nonempty (CategoryTheory.Iso X  …
  -/
  split_ifs with h
    /-
      case pos
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : ∀ (X Y : C), FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Nonempty (CategoryTheory.Iso X Y)
      ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 1
    -/
  · exact (finrank_hom_simple_simple_eq_one_iff 𝕜 X Y).2 h
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Preadditive C
      𝕜 : Type u_2
      inst✝⁶ : Field 𝕜
      inst✝⁵ : IsAlgClosed 𝕜
      inst✝⁴ : CategoryTheory.Linear 𝕜 C
      inst✝³ : CategoryTheory.Limits.HasKernels C
      X Y : C
      inst✝² : ∀ (X Y : C), FiniteDimensional 𝕜 (Quiver.Hom X Y)
      inst✝¹ : CategoryTheory.Simple X
      inst✝ : CategoryTheory.Simple Y
      h : Not (Nonempty (CategoryTheory.Iso X Y))
      ⊢ Eq (Module.finrank 𝕜 (Quiver.Hom X Y)) 0
    -/
  · exact (finrank_hom_simple_simple_eq_zero_iff 𝕜 X Y).2 (not_nonempty_iff.mp h)
    /-
      🎉 no goals
    -/


