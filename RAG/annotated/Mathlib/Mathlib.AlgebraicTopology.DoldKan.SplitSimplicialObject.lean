/-- The projection on a summand of the coproduct decomposition given
by a splitting of a simplicial object. -/
noncomputable def πSummand [HasZeroMorphisms C] {Δ : SimplexCategoryᵒᵖ} (A : IndexSet Δ) :
    X.obj Δ ⟶ s.N A.1.unop.len :=
  s.desc Δ (fun B => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.231, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      Δ : Opposite SimplexCategory
      A B : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Quiver.Hom (s.N (Opposite.unop B.fst).len) (s.N (Opposite.unop A.fst).len)
    -/
    by_cases h : B = A
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.231, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        Δ : Opposite SimplexCategory
        A B : SimplicialObject.Splitting.IndexSet Δ
        h : Eq B A
        ⊢ Quiver.Hom (s.N (Opposite.unop B.fst).len) (s.N (Opposite.unop A.fst).len)
      -/
    · exact eqToHom (by subst h; rfl)
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.231, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        Δ : Opposite SimplexCategory
        A B : SimplicialObject.Splitting.IndexSet Δ
        h : Not (Eq B A)
        ⊢ Quiver.Hom (s.N (Opposite.unop B.fst).len) (s.N (Opposite.unop A.fst).len)
      -/
    · exact 0)
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem cofan_inj_πSummand_eq_id [HasZeroMorphisms C] {Δ : SimplexCategoryᵒᵖ} (A : IndexSet Δ) :
    (s.cofan Δ).inj A ≫ s.πSummand A = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (s.πSummand A)) ( …
  -/
  simp [πSummand]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem cofan_inj_πSummand_eq_zero [HasZeroMorphisms C] {Δ : SimplexCategoryᵒᵖ} (A B : IndexSet Δ)
    (h : B ≠ A) : (s.cofan Δ).inj A ≫ s.πSummand B = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Δ : Opposite SimplexCategory
    A B : SimplicialObject.Splitting.IndexSet Δ
    h : Ne B A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (s.πSummand B)) 0
  -/
  dsimp [πSummand]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Δ : Opposite SimplexCategory
    A B : SimplicialObject.Splitting.IndexSet Δ
    h : Ne B A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (s.desc Δ fun B_1 …
  -/
  rw [ι_desc, dif_neg h.symm]
  /-
    🎉 no goals
  -/


theorem decomposition_id (Δ : SimplexCategoryᵒᵖ) :
    𝟙 (X.obj Δ) = ∑ A : IndexSet Δ, s.πSummand A ≫ (s.cofan Δ).inj A := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    Δ : Opposite SimplexCategory
    ⊢ Eq (CategoryTheory.CategoryStruct.id (X.obj Δ)) (Finset.univ.sum fun A => Ca …
  -/
  apply s.hom_ext'
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    Δ : Opposite SimplexCategory
    ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet Δ), Eq (CategoryTheory.CategorySt …
  -/
  intro A
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (CategoryTheory.C …
  -/
  dsimp
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    Δ : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (CategoryTheory.C …
  -/
  erw [comp_id, comp_sum, Finset.sum_eq_single A, cofan_inj_πSummand_eq_id_assoc]
    /-
      case h.h₀
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ ∀ (b : SimplicialObject.Splitting.IndexSet Δ), Membership.mem Finset.univ b  …
    -/
  · intro B _ h₂
    /-
      case h.h₀
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Δ : Opposite SimplexCategory
      A B : SimplicialObject.Splitting.IndexSet Δ
      a✝ : Membership.mem Finset.univ B
      h₂ : Ne B A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan Δ).inj A) (CategoryTheory.C …
    -/
    rw [s.cofan_inj_πSummand_eq_zero_assoc _ _ h₂, zero_comp]
    /-
      🎉 no goals
    -/
    /-
      case h.h₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Δ : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Not (Membership.mem Finset.univ A) → Eq (CategoryTheory.CategoryStruct.comp  …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem σ_comp_πSummand_id_eq_zero {n : ℕ} (i : Fin (n + 1)) :
    X.σ i ≫ s.πSummand (IndexSet.id (op [n + 1])) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (s.πSummand (SimplicialObject …
  -/
  apply s.hom_ext'
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }) …
  -/
  intro A
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
  -/
  dsimp only [SimplicialObject.σ]
  rw [comp_zero, s.cofan_inj_epi_naturality_assoc A (SimplexCategory.σ i).op,
    cofan_inj_πSummand_eq_zero]
  /-
    case h.h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Ne (SimplicialObject.Splitting.IndexSet.id { unop := SimplexCategory.mk (HAd …
  -/
  rw [ne_comm]
  /-
    case h.h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Ne (A.epiComp (SimplexCategory.σ i).op) (SimplicialObject.Splitting.IndexSet …
  -/
  change ¬(A.epiComp (SimplexCategory.σ i).op).EqId
  /-
    case h.h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Not (A.epiComp (SimplexCategory.σ i).op).EqId
  -/
  rw [IndexSet.eqId_iff_len_eq]
  /-
    case h.h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Not (Eq (Opposite.unop (A.epiComp (SimplexCategory.σ i).op).fst).len (Opposi …
  -/
  have h := SimplexCategory.len_le_of_epi (inferInstance : Epi A.e)
  /-
    case h.h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    h : LE.le (Opposite.unop A.fst).len (Opposite.unop { unop := SimplexCategory.m …
    ⊢ Not (Eq (Opposite.unop (A.epiComp (SimplexCategory.σ i).op).fst).len (Opposi …
  -/
  dsimp at h ⊢
  /-
    case h.h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    h : LE.le (Opposite.unop A.fst).len n
    ⊢ Not (Eq (Opposite.unop A.fst).len (HAdd.hAdd n 1))
  -/
  omega
  /-
    🎉 no goals
  -/


/-- If a simplicial object `X` in an additive category is split,
then `PInfty` vanishes on all the summands of `X _[n]` which do
not correspond to the identity of `[n]`. -/
theorem cofan_inj_comp_PInfty_eq_zero {X : SimplicialObject C} (s : SimplicialObject.Splitting X)
    {n : ℕ} (A : SimplicialObject.Splitting.IndexSet (op [n])) (hA : ¬A.EqId) :
    (s.cofan _).inj A ≫ PInfty.f n = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    hA : Not A.EqId
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
  -/
  rw [SimplicialObject.Splitting.IndexSet.eqId_iff_mono] at hA
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    hA : Not (CategoryTheory.Mono A.e)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
  -/
  rw [SimplicialObject.Splitting.cofan_inj_eq, assoc, degeneracy_comp_PInfty X n A.e hA, comp_zero]
  /-
    🎉 no goals
  -/


theorem comp_PInfty_eq_zero_iff {Z : C} {n : ℕ} (f : Z ⟶ X _[n]) :
    f ≫ PInfty.f n = 0 ↔ f ≫ s.πSummand (IndexSet.id (op [n])) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    Z : C
    n : Nat
    f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PIn …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Z : C
      n : Nat
      f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty.f …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Z : C
      n : Nat
      f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
      h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
    -/
    rcases n with _|n
      /-
        case mp.zero
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk 0 })
        h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
      -/
    · dsimp at h
      /-
        case mp.zero
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk 0 })
        h : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
      -/
      rw [comp_id] at h
      /-
        case mp.zero
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk 0 })
        h : Eq f 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
      -/
      rw [h, zero_comp]
      /-
        🎉 no goals
      -/
      /-
        case mp.succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
      -/
    · have h' := f ≫= PInfty_f_add_QInfty_f (n + 1)
      /-
        case mp.succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
        h' : Eq (CategoryTheory.CategoryStruct.comp f (HAdd.hAdd (AlgebraicTopology.Do …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
      -/
      dsimp at h'
      /-
        case mp.succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
        h' : Eq (CategoryTheory.CategoryStruct.comp f (HAdd.hAdd (AlgebraicTopology.Do …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
      -/
      rw [comp_id, comp_add, h, zero_add] at h'
      rw [← h', assoc, QInfty_f, decomposition_Q, Preadditive.sum_comp, Preadditive.comp_sum,
        Finset.sum_eq_zero]
      /-
        case mp.succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
        h' : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.QInft …
        ⊢ ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem (Finset.filter (fun i => LT.lt ( …
      -/
      intro i _
      /-
        case mp.succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        h : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.PInfty …
        h' : Eq (CategoryTheory.CategoryStruct.comp f (AlgebraicTopology.DoldKan.QInft …
        i : Fin (HAdd.hAdd n 1)
        a✝ : Membership.mem (Finset.filter (fun i => LT.lt (↑i) (HAdd.hAdd n 1)) Finse …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      simp only [assoc, σ_comp_πSummand_id_eq_zero, comp_zero]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Z : C
      n : Nat
      f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Split …
    -/
  · intro h
    rw [← comp_id f, assoc, s.decomposition_id, Preadditive.sum_comp, Preadditive.comp_sum,
      Fintype.sum_eq_zero]
    /-
      case mpr.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Z : C
      n : Nat
      f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
      h : Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Spl …
      ⊢ ∀ (a : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }) …
    -/
    intro A
    /-
      case mpr.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      Z : C
      n : Nat
      f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
      h : Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Spl …
      A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    by_cases hA : A.EqId
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
        h : Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Spl …
        A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
        hA : A.EqId
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
    · dsimp at hA
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
        h : Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Spl …
        A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
        hA : Eq A (SimplicialObject.Splitting.IndexSet.id { unop := SimplexCategory.mk …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      subst hA
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
        h : Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Spl …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      rw [assoc, reassoc_of% h, zero_comp]
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        X : CategoryTheory.SimplicialObject C
        s : SimplicialObject.Splitting X
        inst✝ : CategoryTheory.Preadditive C
        Z : C
        n : Nat
        f : Quiver.Hom Z (X.obj { unop := SimplexCategory.mk n })
        h : Eq (CategoryTheory.CategoryStruct.comp f (s.πSummand (SimplicialObject.Spl …
        A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
        hA : Not A.EqId
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
    · simp only [assoc, s.cofan_inj_comp_PInfty_eq_zero A hA, comp_zero]
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem PInfty_comp_πSummand_id (n : ℕ) :
    PInfty.f n ≫ s.πSummand (IndexSet.id (op [n])) = s.πSummand (IndexSet.id (op [n])) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  conv_rhs => rw [← id_comp (s.πSummand _)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  symm
  rw [← sub_eq_zero, ← sub_comp, ← comp_PInfty_eq_zero_iff, sub_comp, id_comp, PInfty_f_idem,
    sub_self]


@[reassoc (attr := simp)]
theorem πSummand_comp_cofan_inj_id_comp_PInfty_eq_PInfty (n : ℕ) :
    s.πSummand (IndexSet.id (op [n])) ≫ (s.cofan _).inj (IndexSet.id (op [n])) ≫ PInfty.f n =
      PInfty.f n := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.πSummand (SimplicialObject.Splitti …
  -/
  conv_rhs => rw [← id_comp (PInfty.f n)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.πSummand (SimplicialObject.Splitti …
  -/
  erw [s.decomposition_id, Preadditive.sum_comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.πSummand (SimplicialObject.Splitti …
  -/
  rw [Fintype.sum_eq_single (IndexSet.id (op [n])), assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    ⊢ ∀ (x : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }) …
  -/
  rintro A (hA : ¬A.EqId)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    hA : Not A.EqId
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, s.cofan_inj_comp_PInfty_eq_zero A hA, comp_zero]
  /-
    🎉 no goals
  -/


/-- The differentials `s.d i j : s.N i ⟶ s.N j` on nondegenerate simplices of a split
simplicial object are induced by the differentials on the alternating face map complex. -/
@[simp]
noncomputable def d (i j : ℕ) : s.N i ⟶ s.N j :=
  (s.cofan _).inj (IndexSet.id (op [i])) ≫ K[X].d i j ≫ s.πSummand (IndexSet.id (op [j]))


theorem ιSummand_comp_d_comp_πSummand_eq_zero (j k : ℕ) (A : IndexSet (op [j])) (hA : ¬A.EqId) :
    (s.cofan _).inj A ≫ K[X].d j k ≫ s.πSummand (IndexSet.id (op [k])) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X : CategoryTheory.SimplicialObject C
    s : SimplicialObject.Splitting X
    inst✝ : CategoryTheory.Preadditive C
    j k : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }
    hA : Not A.EqId
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
  -/
  rw [A.eqId_iff_mono] at hA
  rw [← assoc, ← s.comp_PInfty_eq_zero_iff, assoc, ← PInfty.comm j k, s.cofan_inj_eq, assoc,
    degeneracy_comp_PInfty_assoc X j A.e hA, zero_comp, comp_zero]


/-- If `s` is a splitting of a simplicial object `X` in a preadditive category,
`s.nondegComplex` is a chain complex which is given in degree `n` by
the nondegenerate `n`-simplices of `X`. -/
@[simps]
noncomputable def nondegComplex : ChainComplex C ℕ where
  X := s.N
  d := s.d
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{?u.51898, u_1} C
                        X : CategoryTheory.SimplicialObject C
                        s : SimplicialObject.Splitting X
                        inst✝ : CategoryTheory.Preadditive C
                        i j : Nat
                        hij : Not ((ComplexShape.down Nat).Rel i j)
                        ⊢ Eq (s.d i j) 0
                      -/
  shape i j hij := by simp only [d, K[X].shape i j hij, zero_comp, comp_zero]
                      /-
                        🎉 no goals
                      -/
  d_comp_d' i j k _ _ := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.51898, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      i j k : Nat
      x✝¹ : (ComplexShape.down Nat).Rel i j
      x✝ : (ComplexShape.down Nat).Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.d i j) (s.d j k)) 0
    -/
    simp only [d, assoc]
    have eq : K[X].d i j ≫ 𝟙 (X.obj (op [j])) ≫ K[X].d j k ≫
        s.πSummand (IndexSet.id (op [k])) = 0 := by
      erw [id_comp, HomologicalComplex.d_comp_d_assoc, zero_comp]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.51898, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      i j k : Nat
      x✝¹ : (ComplexShape.down Nat).Rel i j
      x✝ : (ComplexShape.down Nat).Rel j k
      eq : Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.AlternatingFac …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
    -/
    rw [s.decomposition_id] at eq
    classical
    rw [Fintype.sum_eq_add_sum_compl (IndexSet.id (op [j])), add_comp, comp_add, assoc,
      Preadditive.sum_comp, Preadditive.comp_sum, Finset.sum_eq_zero, add_zero] at eq
    swap
    · intro A hA
      simp only [Finset.mem_compl, Finset.mem_singleton] at hA
      simp only [assoc, ιSummand_comp_d_comp_πSummand_eq_zero _ _ _ _ hA, comp_zero]
    rw [eq, comp_zero]


/-- The chain complex `s.nondegComplex` attached to a splitting of a simplicial object `X`
becomes isomorphic to the normalized Moore complex `N₁.obj X` defined as a formal direct
factor in the category `Karoubi (ChainComplex C ℕ)`. -/
@[simps]
noncomputable def toKaroubiNondegComplexIsoN₁ :
    (toKaroubi _).obj s.nondegComplex ≅ N₁.obj X where
  hom :=
    { f :=
        { f := fun n => (s.cofan _).inj (IndexSet.id (op [n])) ≫ PInfty.f n
          comm' := fun i j _ => by
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
              X : CategoryTheory.SimplicialObject C
              s : SimplicialObject.Splitting X
              inst✝ : CategoryTheory.Preadditive C
              i j : Nat
              x✝ : (ComplexShape.down Nat).Rel i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
            -/
            dsimp
            rw [assoc, assoc, assoc, πSummand_comp_cofan_inj_id_comp_PInfty_eq_PInfty,
              HomologicalComplex.Hom.comm] }
      comm := by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
          X : CategoryTheory.SimplicialObject C
          s : SimplicialObject.Splitting X
          inst✝ : CategoryTheory.Preadditive C
          ⊢ Eq { f := fun n => CategoryTheory.CategoryStruct.comp ((s.cofan { unop := Si …
        -/
        ext n
        /-
          case h
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
          X : CategoryTheory.SimplicialObject C
          s : SimplicialObject.Splitting X
          inst✝ : CategoryTheory.Preadditive C
          n : Nat
          ⊢ Eq ({ f := fun n => CategoryTheory.CategoryStruct.comp ((s.cofan { unop := S …
        -/
        dsimp
        /-
          case h
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
          X : CategoryTheory.SimplicialObject C
          s : SimplicialObject.Splitting X
          inst✝ : CategoryTheory.Preadditive C
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((s.cofan { unop := SimplexCategory.m …
        -/
        rw [id_comp, assoc, PInfty_f_idem] }
        /-
          🎉 no goals
        -/
  inv :=
    { f :=
        { f := fun n => s.πSummand (IndexSet.id (op [n]))
          comm' := fun i j _ => by
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
              X : CategoryTheory.SimplicialObject C
              s : SimplicialObject.Splitting X
              inst✝ : CategoryTheory.Preadditive C
              i j : Nat
              x✝ : (ComplexShape.down Nat).Rel i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => s.πSummand (SimplicialObje …
            -/
            dsimp
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
              X : CategoryTheory.SimplicialObject C
              s : SimplicialObject.Splitting X
              inst✝ : CategoryTheory.Preadditive C
              i j : Nat
              x✝ : (ComplexShape.down Nat).Rel i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.πSummand (SimplicialObject.Splitti …
            -/
            slice_rhs 1 1 => rw [← id_comp (K[X].d i j)]
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
              X : CategoryTheory.SimplicialObject C
              s : SimplicialObject.Splitting X
              inst✝ : CategoryTheory.Preadditive C
              i j : Nat
              x✝ : (ComplexShape.down Nat).Rel i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.πSummand (SimplicialObject.Splitti …
            -/
            erw [s.decomposition_id]
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
              X : CategoryTheory.SimplicialObject C
              s : SimplicialObject.Splitting X
              inst✝ : CategoryTheory.Preadditive C
              i j : Nat
              x✝ : (ComplexShape.down Nat).Rel i j
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.πSummand (SimplicialObject.Splitti …
            -/
            rw [sum_comp, sum_comp, Finset.sum_eq_single (IndexSet.id (op [i])), assoc, assoc]
              /-
                case h₀
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
                X : CategoryTheory.SimplicialObject C
                s : SimplicialObject.Splitting X
                inst✝ : CategoryTheory.Preadditive C
                i j : Nat
                x✝ : (ComplexShape.down Nat).Rel i j
                ⊢ ∀ (b : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk i }) …
              -/
            · intro A _ hA
              /-
                case h₀
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
                X : CategoryTheory.SimplicialObject C
                s : SimplicialObject.Splitting X
                inst✝ : CategoryTheory.Preadditive C
                i j : Nat
                x✝ : (ComplexShape.down Nat).Rel i j
                A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk i }
                a✝ : Membership.mem Finset.univ A
                hA : Ne A (SimplicialObject.Splitting.IndexSet.id { unop := SimplexCategory.mk …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              simp only [assoc, s.ιSummand_comp_d_comp_πSummand_eq_zero _ _ _ hA, comp_zero]
              /-
                🎉 no goals
              -/
              /-
                case h₁
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
                X : CategoryTheory.SimplicialObject C
                s : SimplicialObject.Splitting X
                inst✝ : CategoryTheory.Preadditive C
                i j : Nat
                x✝ : (ComplexShape.down Nat).Rel i j
                ⊢ Not (Membership.mem Finset.univ (SimplicialObject.Splitting.IndexSet.id { un …
              -/
            · simp only [Finset.mem_univ, not_true, IsEmpty.forall_iff] }
              /-
                🎉 no goals
              -/
      comm := by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
          X : CategoryTheory.SimplicialObject C
          s : SimplicialObject.Splitting X
          inst✝ : CategoryTheory.Preadditive C
          ⊢ Eq { f := fun n => s.πSummand (SimplicialObject.Splitting.IndexSet.id { unop …
        -/
        ext n
        /-
          case h
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
          X : CategoryTheory.SimplicialObject C
          s : SimplicialObject.Splitting X
          inst✝ : CategoryTheory.Preadditive C
          n : Nat
          ⊢ Eq ({ f := fun n => s.πSummand (SimplicialObject.Splitting.IndexSet.id { uno …
        -/
        dsimp
        /-
          case h
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
          X : CategoryTheory.SimplicialObject C
          s : SimplicialObject.Splitting X
          inst✝ : CategoryTheory.Preadditive C
          n : Nat
          ⊢ Eq (s.πSummand (SimplicialObject.Splitting.IndexSet.id { unop := SimplexCate …
        -/
        simp only [comp_id, PInfty_comp_πSummand_id] }
        /-
          🎉 no goals
        -/
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := { f := fun n => CategoryTheory …
    -/
    ext n
    simp only [assoc, PInfty_comp_πSummand_id, Karoubi.comp_f, HomologicalComplex.comp_f,
      cofan_inj_πSummand_eq_id]
    /-
      case h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.id (SimplicialObject.Splitting.summand s.N …
    -/
    rfl
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.69081, u_1} C
      X : CategoryTheory.SimplicialObject C
      s : SimplicialObject.Splitting X
      inst✝ : CategoryTheory.Preadditive C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := { f := fun n => s.πSummand (Si …
    -/
    ext n
    simp only [πSummand_comp_cofan_inj_id_comp_PInfty_eq_PInfty, Karoubi.comp_f,
      HomologicalComplex.comp_f, N₁_obj_p, Karoubi.id_f]


/-- The functor which sends a split simplicial object in a preadditive category to
the chain complex which consists of nondegenerate simplices. -/
@[simps]
noncomputable def nondegComplexFunctor : Split C ⥤ ChainComplex C ℕ where
  obj S := S.s.nondegComplex
  map {S₁ S₂} Φ :=
    { f := Φ.f
      comm' := fun i j _ => by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f i) (((fun S => S.s.nondegComplex …
        -/
        dsimp
        erw [← cofan_inj_naturality_symm_assoc Φ (Splitting.IndexSet.id (op [i])),
          ((alternatingFaceMapComplex C).map Φ.F).comm_assoc i j]
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((S₁.s.cofan { unop := SimplexCategor …
        -/
        simp only [assoc]
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((S₁.s.cofan { unop := SimplexCategor …
        -/
        congr 2
        /-
          case e_a.e_a
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.alternatingFaceM …
        -/
        apply S₁.s.hom_ext'
        /-
          case e_a.e_a.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }) …
        -/
        intro A
        /-
          case e_a.e_a.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((S₁.s.cofan { unop := SimplexCategor …
        -/
        dsimp [alternatingFaceMapComplex]
        /-
          case e_a.e_a.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((S₁.s.cofan { unop := SimplexCategor …
        -/
        rw [cofan_inj_naturality_symm_assoc Φ A]
        /-
          case e_a.e_a.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          S₁ S₂ : SimplicialObject.Split C
          Φ : Quiver.Hom S₁ S₂
          i j : Nat
          x✝ : (ComplexShape.down Nat).Rel i j
          A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f (Opposite.unop A.fst).len) (Cate …
        -/
        by_cases h : A.EqId
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            S₁ S₂ : SimplicialObject.Split C
            Φ : Quiver.Hom S₁ S₂
            i j : Nat
            x✝ : (ComplexShape.down Nat).Rel i j
            A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }
            h : A.EqId
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f (Opposite.unop A.fst).len) (Cate …
          -/
        · dsimp at h
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            S₁ S₂ : SimplicialObject.Split C
            Φ : Quiver.Hom S₁ S₂
            i j : Nat
            x✝ : (ComplexShape.down Nat).Rel i j
            A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk j }
            h : Eq A (SimplicialObject.Splitting.IndexSet.id { unop := SimplexCategory.mk  …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f (Opposite.unop A.fst).len) (Cate …
          -/
          subst h
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            S₁ S₂ : SimplicialObject.Split C
            Φ : Quiver.Hom S₁ S₂
            i j : Nat
            x✝ : (ComplexShape.down Nat).Rel i j
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f (Opposite.unop (SimplicialObject …
          -/
          rw [Splitting.cofan_inj_πSummand_eq_id]
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            S₁ S₂ : SimplicialObject.Split C
            Φ : Quiver.Hom S₁ S₂
            i j : Nat
            x✝ : (ComplexShape.down Nat).Rel i j
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f (Opposite.unop (SimplicialObject …
          -/
          dsimp
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.95353, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            S₁ S₂ : SimplicialObject.Split C
            Φ : Quiver.Hom S₁ S₂
            i j : Nat
            x✝ : (ComplexShape.down Nat).Rel i j
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f j) (CategoryTheory.CategoryStruc …
          -/
          rw [comp_id, Splitting.cofan_inj_πSummand_eq_id_assoc]
          /-
            🎉 no goals
          -/
        · rw [S₁.s.cofan_inj_πSummand_eq_zero_assoc _ _ (Ne.symm h),
            S₂.s.cofan_inj_πSummand_eq_zero _ _ (Ne.symm h), zero_comp, comp_zero] }


/-- The natural isomorphism (in `Karoubi (ChainComplex C ℕ)`) between the chain complex
of nondegenerate simplices of a split simplicial object and the normalized Moore complex
defined as a formal direct factor of the alternating face map complex. -/
@[simps!]
noncomputable def toKaroubiNondegComplexFunctorIsoN₁ :
    nondegComplexFunctor ⋙ toKaroubi (ChainComplex C ℕ) ≅ forget C ⋙ DoldKan.N₁ :=
  NatIso.ofComponents (fun S => S.s.toKaroubiNondegComplexIsoN₁) fun Φ => by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.107523, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X✝ Y✝ : SimplicialObject.Split C
      Φ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((SimplicialObject.Split.nondegComple …
    -/
    ext n
    /-
      case h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.107523, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X✝ Y✝ : SimplicialObject.Split C
      Φ : Quiver.Hom X✝ Y✝
      n : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((SimplicialObject.Split.nondegCompl …
    -/
    dsimp
    simp only [Karoubi.comp_f, toKaroubi_map_f, HomologicalComplex.comp_f,
      nondegComplexFunctor_map_f, Splitting.toKaroubiNondegComplexIsoN₁_hom_f_f, N₁_map_f,
      AlternatingFaceMapComplex.map_f, assoc, PInfty_f_idem_assoc]
    /-
      case h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.107523, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X✝ Y✝ : SimplicialObject.Split C
      Φ : Quiver.Hom X✝ Y✝
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (Φ.f n) (CategoryTheory.CategoryStruc …
    -/
    erw [← Split.cofan_inj_naturality_symm_assoc Φ (Splitting.IndexSet.id (op [n]))]
    /-
      case h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.107523, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X✝ Y✝ : SimplicialObject.Split C
      Φ : Quiver.Hom X✝ Y✝
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((X✝.s.cofan { unop := SimplexCategor …
    -/
    rw [PInfty_f_naturality]
    /-
      🎉 no goals
    -/


