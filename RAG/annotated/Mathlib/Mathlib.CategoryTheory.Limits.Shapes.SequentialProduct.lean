lemma functorObj_eq_pos {n m : ℕ} (h : m < n) :
    (fun i ↦ if _ : i < n then M i else N i) m = M m := dif_pos h


lemma functorObj_eq_neg {n m : ℕ} (h : ¬(m < n)) :
    (fun i ↦ if _ : i < n then M i else N i) m = N m := dif_neg h


variable (M N) in
/-- The product of the `m` first objects of `M` and the rest of the rest of `N` -/
noncomputable def functorObj : ℕ → C :=
  fun n ↦ ∏ᶜ (fun m ↦ if _ : m < n then M m else N m)


/-- The projection map from `functorObj M N n` to `M m`, when `m < n`  -/
noncomputable def functorObjProj_pos (n m : ℕ) (h : m < n) :
    functorObj M N n ⟶ M m :=
                                                                                   /-
                                                                                     C : Type u_1
                                                                                     M N : Nat → C
                                                                                     inst✝¹ : CategoryTheory.Category.{?u.7050, u_1} C
                                                                                     f : (n : Nat) → Quiver.Hom (M n) (N n)
                                                                                     inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                                                                     n m : Nat
                                                                                     h : LT.lt m n
                                                                                     ⊢ LT.lt m n
                                                                                   -/
  Pi.π (fun m ↦ if _ : m < n then M m else N m) m ≫ eqToHom (functorObj_eq_pos (by omega))
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- The projection map from `functorObj M N n` to `N m`, when `m ≥ n`  -/
noncomputable def functorObjProj_neg (n m : ℕ) (h : ¬(m < n)) :
    functorObj M N n ⟶ N m :=
                                                                                   /-
                                                                                     C : Type u_1
                                                                                     M N : Nat → C
                                                                                     inst✝¹ : CategoryTheory.Category.{?u.14000, u_1} C
                                                                                     f : (n : Nat) → Quiver.Hom (M n) (N n)
                                                                                     inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                                                                     n m : Nat
                                                                                     h : Not (LT.lt m n)
                                                                                     ⊢ Not (LT.lt m n)
                                                                                   -/
  Pi.π (fun m ↦ if _ : m < n then M m else N m) m ≫ eqToHom (functorObj_eq_neg (by omega))
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- The transition maps in the sequential limit of products -/
noncomputable def functorMap : ∀ n,
    functorObj M N (n + 1) ⟶ functorObj M N n := by
  /-
    C : Type u_1
    M N : Nat → C
    inst✝¹ : CategoryTheory.Category.{?u.20939, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
    ⊢ (n : Nat) → Quiver.Hom (CategoryTheory.Limits.SequentialProduct.functorObj M …
  -/
  intro n
  refine Limits.Pi.map fun m ↦ if h : m < n then eqToHom ?_ else
    if h' : m < n + 1 then eqToHom ?_ ≫ f m ≫ eqToHom ?_ else eqToHom ?_
  /-
    case refine_1
    C : Type u_1
    M N : Nat → C
    inst✝¹ : CategoryTheory.Category.{?u.20939, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
    n m : Nat
    h : LT.lt m n
    ⊢ Eq (dite (LT.lt m (HAdd.hAdd n 1)) (fun x => M m) fun x => N m) (dite (LT.lt …
  -/
  all_goals split_ifs; try rfl; try omega
  /-
    🎉 no goals
  -/


lemma functorMap_commSq_succ (n : ℕ) :
                                                           /-
                                                             C : Type u_1
                                                             M N : Nat → C
                                                             inst✝¹ : CategoryTheory.Category.{?u.35455, u_1} C
                                                             f : (n : Nat) → Quiver.Hom (M n) (N n)
                                                             inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                                             n : Nat
                                                             ⊢ LE.le n (HAdd.hAdd n 1)
                                                           -/
    (Functor.ofOpSequence (functorMap f)).map (homOfLE (by omega : n ≤ n+1)).op ≫ Pi.π _ n ≫
                                                           /-
                                                             🎉 no goals
                                                           -/
                                     /-
                                       C : Type u_1
                                       M N : Nat → C
                                       inst✝¹ : CategoryTheory.Category.{?u.35455, u_1} C
                                       f : (n : Nat) → Quiver.Hom (M n) (N n)
                                       inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                       n : Nat
                                       ⊢ Not (LT.lt n n)
                                     -/
      eqToHom (functorObj_eq_neg (by omega : ¬(n < n))) =
                                     /-
                                       🎉 no goals
                                     -/
        (Pi.π (fun i ↦ if _ : i < (n + 1) then M i else N i) n) ≫
                                         /-
                                           C : Type u_1
                                           M N : Nat → C
                                           inst✝¹ : CategoryTheory.Category.{?u.35455, u_1} C
                                           f : (n : Nat) → Quiver.Hom (M n) (N n)
                                           inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                           n : Nat
                                           ⊢ LT.lt n (HAdd.hAdd n 1)
                                         -/
          eqToHom (functorObj_eq_pos (by omega)) ≫ f n := by
                                         /-
                                           🎉 no goals
                                         -/
  /-
    C : Type u_1
    M N : Nat → C
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
  -/
  simp [functorMap]
  /-
    🎉 no goals
  -/


lemma functorMap_commSq_aux {n m k : ℕ} (h : n ≤ m) (hh : ¬(k < m)) :
    (Functor.ofOpSequence (functorMap f)).map (homOfLE h).op ≫ Pi.π _ k ≫
                                     /-
                                       C : Type u_1
                                       M N : Nat → C
                                       inst✝¹ : CategoryTheory.Category.{?u.77029, u_1} C
                                       f : (n : Nat) → Quiver.Hom (M n) (N n)
                                       inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                       n m k : Nat
                                       h : LE.le n m
                                       hh : Not (LT.lt k m)
                                       ⊢ Not (LT.lt k n)
                                     -/
      eqToHom (functorObj_eq_neg (by omega : ¬(k < n))) =
                                     /-
                                       🎉 no goals
                                     -/
        (Pi.π (fun i ↦ if _ : i < m then M i else N i) k) ≫
          eqToHom (functorObj_eq_neg hh) := by
  /-
    C : Type u_1
    M N : Nat → C
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
    n m k : Nat
    h : LE.le n m
    hh : Not (LT.lt k m)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
  -/
  induction' h using Nat.leRec with m h ih
    /-
      case refl
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m k : Nat
      hh : Not (LT.lt k n)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case le_succ_of_le
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m✝ k m : Nat
      h : LE.le n m
      ih : ∀ (hh : Not (LT.lt k m)), Eq (CategoryTheory.CategoryStruct.comp ((Catego …
      hh : Not (LT.lt k (HAdd.hAdd m 1))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
    -/
  · specialize ih (by omega)
    have : homOfLE (by omega : n ≤ m + 1) =
        homOfLE (by omega : n ≤ m) ≫ homOfLE (by omega : m ≤ m + 1) := by simp
    /-
      case le_succ_of_le
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m✝ k m : Nat
      h : LE.le n m
      hh : Not (LT.lt k (HAdd.hAdd m 1))
      ih : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSeque …
      this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
    -/
    rw [this, op_comp, Functor.map_comp]
    /-
      case le_succ_of_le
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m✝ k m : Nat
      h : LE.le n m
      hh : Not (LT.lt k (HAdd.hAdd m 1))
      ih : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSeque …
      this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 2 4 => rw [ih]
    simp only [Functor.ofOpSequence_obj, homOfLE_leOfHom, Functor.ofOpSequence_map_homOfLE_succ,
      functorMap, dite_eq_ite, limMap_π_assoc, Discrete.functor_obj_eq_as, Discrete.natTrans_app]
    /-
      case le_succ_of_le
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m✝ k m : Nat
      h : LE.le n m
      hh : Not (LT.lt k (HAdd.hAdd m 1))
      ih : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSeque …
      this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
    -/
    split_ifs
    /-
      case le_succ_of_le
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m✝ k m : Nat
      h : LE.le n m
      hh : Not (LT.lt k (HAdd.hAdd m 1))
      ih : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSeque …
      this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
    -/
    simp [dif_neg (by omega : ¬(k < m))]
    /-
      🎉 no goals
    -/


lemma functorMap_commSq {n m : ℕ} (h : ¬(m < n)) :
                                                           /-
                                                             C : Type u_1
                                                             M N : Nat → C
                                                             inst✝¹ : CategoryTheory.Category.{?u.111798, u_1} C
                                                             f : (n : Nat) → Quiver.Hom (M n) (N n)
                                                             inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                                             n m : Nat
                                                             h : Not (LT.lt m n)
                                                             ⊢ LE.le n (HAdd.hAdd m 1)
                                                           -/
    (Functor.ofOpSequence (functorMap f)).map (homOfLE (by omega : n ≤ m + 1)).op ≫ Pi.π _ m ≫
                                                           /-
                                                             🎉 no goals
                                                           -/
                                     /-
                                       C : Type u_1
                                       M N : Nat → C
                                       inst✝¹ : CategoryTheory.Category.{?u.111798, u_1} C
                                       f : (n : Nat) → Quiver.Hom (M n) (N n)
                                       inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                       n m : Nat
                                       h : Not (LT.lt m n)
                                       ⊢ Not (LT.lt m n)
                                     -/
      eqToHom (functorObj_eq_neg (by omega : ¬(m < n))) =
                                     /-
                                       🎉 no goals
                                     -/
        (Pi.π (fun i ↦ if _ : i < m + 1 then M i else N i) m) ≫
                                         /-
                                           C : Type u_1
                                           M N : Nat → C
                                           inst✝¹ : CategoryTheory.Category.{?u.111798, u_1} C
                                           f : (n : Nat) → Quiver.Hom (M n) (N n)
                                           inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                                           n m : Nat
                                           h : Not (LT.lt m n)
                                           ⊢ LT.lt m (HAdd.hAdd m 1)
                                         -/
          eqToHom (functorObj_eq_pos (by omega)) ≫ f m := by
                                         /-
                                           🎉 no goals
                                         -/
  cases m with
  | zero =>
      have : n = 0 := by omega
      subst this
      simp [functorMap]
  | succ m =>
      rw [← functorMap_commSq_succ f (m + 1)]
      simp only [Functor.ofOpSequence_obj, homOfLE_leOfHom, dite_eq_ite,
        Functor.ofOpSequence_map_homOfLE_succ, add_le_iff_nonpos_right, nonpos_iff_eq_zero,
        one_ne_zero]
      have : homOfLE (by omega : n ≤ m + 1 + 1) =
          homOfLE (by omega : n ≤ m + 1) ≫ homOfLE (by omega : m + 1 ≤ m + 1 + 1) := by simp
      rw [this, op_comp, Functor.map_comp]
      simp only [Functor.ofOpSequence_obj, homOfLE_leOfHom, Functor.ofOpSequence_map_homOfLE_succ,
        Category.assoc, add_le_iff_nonpos_right, nonpos_iff_eq_zero, one_ne_zero]
      congr 1
      exact functorMap_commSq_aux f (by omega) (by omega)


/--
The cone over the tower
```
⋯ → ∏_{n < m} M n × ∏_{n ≥ m} N n → ⋯ → ∏ N
```
with cone point `∏ M`. This is a limit cone, see `CategoryTheory.Limits.SequentialProduct.isLimit`.
 -/
noncomputable def cone : Cone (Functor.ofOpSequence (functorMap f)) where
  pt := ∏ᶜ M
  π := by
    refine NatTrans.ofOpSequence
      (fun n ↦ Limits.Pi.map fun m ↦ if h : m < n then eqToHom (functorObj_eq_pos h).symm else
        f m ≫ eqToHom (functorObj_eq_neg h).symm) (fun n ↦ ?_)
    /-
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.140540, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
    -/
    apply Limits.Pi.hom_ext
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.140540, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n : Nat
      ⊢ ∀ (b : Nat), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
    -/
    intro m
    simp only [Functor.const_obj_obj, Functor.ofOpSequence_obj, homOfLE_leOfHom,
      Functor.const_obj_map, Category.id_comp, limMap_π, Discrete.functor_obj_eq_as,
      Discrete.natTrans_app, Functor.ofOpSequence_map_homOfLE_succ, functorMap, Category.assoc,
      limMap_π_assoc]
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.140540, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      n m : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
    -/
    split
      /-
        case h.isTrue
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.140540, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        n m : Nat
        h✝ : LT.lt m n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
      -/
    · simp [dif_pos (by omega : m < n + 1)]
      /-
        🎉 no goals
      -/
      /-
        case h.isFalse
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.140540, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        n m : Nat
        h✝ : Not (LT.lt m n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
      -/
    · split
      /-
        case h.isFalse.isTrue
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.140540, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        n m : Nat
        h✝¹ : Not (LT.lt m n)
        h✝ : LT.lt m (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
      -/
      all_goals simp
      /-
        🎉 no goals
      -/


lemma cone_π_app (n : ℕ) : (cone f).π.app ⟨n⟩ =
  Limits.Pi.map fun m ↦ if h : m < n then eqToHom (functorObj_eq_pos h).symm else
    f m ≫ eqToHom (functorObj_eq_neg h).symm := rfl


@[reassoc]
lemma cone_π_app_comp_Pi_π_pos (m n : ℕ) (h : n < m) : (cone f).π.app ⟨m⟩ ≫
    Pi.π (fun i ↦ if _ : i < m then M i else N i) n =
    Pi.π _ n ≫ eqToHom (functorObj_eq_pos h).symm := by
  simp only [Functor.const_obj_obj, dite_eq_ite, Functor.ofOpSequence_obj, cone_π_app, limMap_π,
    Discrete.functor_obj_eq_as, Discrete.natTrans_app]
  /-
    C : Type u_1
    M N : Nat → C
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
    m n : Nat
    h : LT.lt n m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
  -/
  rw [dif_pos h]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma cone_π_app_comp_Pi_π_neg (m n : ℕ) (h : ¬(n < m)) : (cone f).π.app ⟨m⟩ ≫ Pi.π _ n =
    Pi.π _ n ≫ f n ≫ eqToHom (functorObj_eq_neg h).symm := by
  simp only [Functor.const_obj_obj, dite_eq_ite, Functor.ofOpSequence_obj, cone_π_app, limMap_π,
    Discrete.functor_obj_eq_as, Discrete.natTrans_app]
  /-
    C : Type u_1
    M N : Nat → C
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
    m n : Nat
    h : Not (LT.lt n m)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
  -/
  rw [dif_neg h]
  /-
    🎉 no goals
  -/


/--
The cone over the tower
```
⋯ → ∏_{n < m} M n × ∏_{n ≥ m} N n → ⋯ → ∏ N
```
with cone point `∏ M` is indeed a limit cone.
 -/
noncomputable def isLimit : IsLimit (cone f) where
  lift s := Pi.lift fun m ↦
    s.π.app ⟨m + 1⟩ ≫ Pi.π (fun i ↦ if _ : i < m + 1 then M i else N i) m ≫
                           /-
                             C : Type u_1
                             M N : Nat → C
                             inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
                             f : (n : Nat) → Quiver.Hom (M n) (N n)
                             inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
                             s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
                             m : Nat
                             ⊢ LT.lt m (HAdd.hAdd m 1)
                           -/
      eqToHom (dif_pos (by omega : m < m + 1))
                           /-
                             🎉 no goals
                           -/
  fac s := by
    /-
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      ⊢ ∀ (j : Opposite Nat), Eq (CategoryTheory.CategoryStruct.comp ((fun s => Cate …
    -/
    intro ⟨n⟩
    /-
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.Pi.l …
    -/
    apply Pi.hom_ext
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      n : Nat
      ⊢ ∀ (b : Nat), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
    -/
    intro m
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      n m : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    by_cases h : m < n
      /-
        case pos
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : LT.lt m n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [le_refl, Category.assoc, cone_π_app_comp_Pi_π_pos f _ _ h]
      simp only [dite_eq_ite, Functor.ofOpSequence_obj, le_refl, limit.lift_π_assoc, Fan.mk_pt,
        Discrete.functor_obj_eq_as, Fan.mk_π_app, Category.assoc, eqToHom_trans]
      /-
        case pos
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : LT.lt m n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { unop := HAdd.hAdd m 1 }) ( …
      -/
      have hh : m + 1 ≤ n := by omega
      /-
        case pos
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : LT.lt m n
        hh : LE.le (HAdd.hAdd m 1) n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { unop := HAdd.hAdd m 1 }) ( …
      -/
      rw [← s.w (homOfLE hh).op]
      simp only [Functor.const_obj_obj, Functor.ofOpSequence_obj, homOfLE_leOfHom, le_refl,
        Category.assoc]
      /-
        case pos
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : LT.lt m n
        hh : LE.le (HAdd.hAdd m 1) n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { unop := n }) (CategoryTheo …
      -/
      congr
      /-
        case pos.e_a
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : LT.lt m n
        hh : LE.le (HAdd.hAdd m 1) n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
      -/
      induction' hh using Nat.leRec with n hh ih
        /-
          case pos.e_a.refl
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n m : Nat
          h : LT.lt m (HAdd.hAdd m 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
        -/
      · have : homOfLE (Nat.le_succ_of_le hh) = homOfLE hh ≫ homOfLE (Nat.le_succ n) := by simp
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofOpSequence …
        -/
        rw [this, op_comp, Functor.map_comp]
        simp only [Functor.ofOpSequence_obj, Nat.succ_eq_add_one, homOfLE_leOfHom,
          Functor.ofOpSequence_map_homOfLE_succ, le_refl, Category.assoc]
        have h₁ : (if _ : m < m + 1 then M m else N m) = if _ : m < n then M m else N m := by
          rw [dif_pos (by omega), dif_pos (by omega)]
        have h₂ : (if _ : m < n then M m else N m) = if _ : m < n + 1 then M m else N m := by
          rw [dif_pos h, dif_pos (by omega)]
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
          h₁ : Eq (dite (LT.lt m (HAdd.hAdd m 1)) (fun x => M m) fun x => N m) (dite (LT …
          h₂ : Eq (dite (LT.lt m n) (fun x => M m) fun x => N m) (dite (LT.lt m (HAdd.hA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.SequentialProd …
        -/
        rw [← eqToHom_trans h₁ h₂]
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
          h₁ : Eq (dite (LT.lt m (HAdd.hAdd m 1)) (fun x => M m) fun x => N m) (dite (LT …
          h₂ : Eq (dite (LT.lt m n) (fun x => M m) fun x => N m) (dite (LT.lt m (HAdd.hA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.SequentialProd …
        -/
        slice_lhs 2 4 => rw [ih (by omega)]
        simp only [functorMap, dite_eq_ite, Pi.π, limMap_π_assoc, Discrete.functor_obj_eq_as,
          Discrete.natTrans_app]
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
          h₁ : Eq (dite (LT.lt m (HAdd.hAdd m 1)) (fun x => M m) fun x => N m) (dite (LT …
          h₂ : Eq (dite (LT.lt m n) (fun x => M m) fun x => N m) (dite (LT.lt m (HAdd.hA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
        -/
        split_ifs
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
          h₁ : Eq (dite (LT.lt m (HAdd.hAdd m 1)) (fun x => M m) fun x => N m) (dite (LT …
          h₂ : Eq (dite (LT.lt m n) (fun x => M m) fun x => N m) (dite (LT.lt m (HAdd.hA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
        -/
        rw [dif_pos (by omega)]
        /-
          case pos.e_a.le_succ_of_le
          C : Type u_1
          M N : Nat → C
          inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
          f : (n : Nat) → Quiver.Hom (M n) (N n)
          inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
          s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
          n✝ m n : Nat
          hh : LE.le (HAdd.hAdd m 1) n
          ih : ∀ (h : LT.lt m n), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          h : LT.lt m (HAdd.hAdd n 1)
          this : Eq (CategoryTheory.homOfLE ⋯) (CategoryTheory.CategoryStruct.comp (Cate …
          h₁ : Eq (dite (LT.lt m (HAdd.hAdd m 1)) (fun x => M m) fun x => N m) (dite (LT …
          h₂ : Eq (dite (LT.lt m n) (fun x => M m) fun x => N m) (dite (LT.lt m (HAdd.hA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
        -/
        simp
        /-
          🎉 no goals
        -/
      /-
        case neg
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : Not (LT.lt m n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [le_refl, Category.assoc]
      /-
        case neg
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : Not (LT.lt m n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.lift fun m  …
      -/
      rw [cone_π_app_comp_Pi_π_neg f _ _ h]
      simp only [dite_eq_ite, Functor.ofOpSequence_obj, le_refl, limit.lift_π_assoc, Fan.mk_pt,
        Discrete.functor_obj_eq_as, Fan.mk_π_app, Category.assoc]
      /-
        case neg
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : Not (LT.lt m n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { unop := HAdd.hAdd m 1 }) ( …
      -/
      slice_lhs 2 4 => erw [← functorMap_commSq f h]
      /-
        case neg
        C : Type u_1
        M N : Nat → C
        inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
        f : (n : Nat) → Quiver.Hom (M n) (N n)
        inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
        s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
        n m : Nat
        h : Not (LT.lt m n)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { unop := HAdd.hAdd m 1 }) ( …
      -/
      simp
      /-
        🎉 no goals
      -/
  uniq s m h := by
    /-
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.SequentialProduct.cone f).pt
      h : ∀ (j : Opposite Nat), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryT …
      ⊢ Eq m ((fun s => CategoryTheory.Limits.Pi.lift fun m => CategoryTheory.Catego …
    -/
    apply Pi.hom_ext
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.SequentialProduct.cone f).pt
      h : ∀ (j : Opposite Nat), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryT …
      ⊢ ∀ (b : Nat), Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits …
    -/
    intro n
    simp only [Functor.ofOpSequence_obj, le_refl, dite_eq_ite, limit.lift_π, Fan.mk_pt,
      Fan.mk_π_app, ← h ⟨n + 1⟩, Category.assoc]
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.SequentialProduct.cone f).pt
      h : ∀ (j : Opposite Nat), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryT …
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Pi.π M n)) ( …
    -/
    slice_rhs 2 3 => erw [cone_π_app_comp_Pi_π_pos f (n + 1) _ (by omega)]
    /-
      case h
      C : Type u_1
      M N : Nat → C
      inst✝¹ : CategoryTheory.Category.{?u.230079, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : CategoryTheory.Limits.HasProductsOfShape Nat C
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.ofOpSequence (CategoryT …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.SequentialProduct.cone f).pt
      h : ∀ (j : Opposite Nat), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryT …
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Pi.π M n)) ( …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma functorMap_epi (n : ℕ) : Epi (functorMap f n) := by
  /-
    C : Type u_1
    M N : Nat → C
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    n : Nat
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.SequentialProduct.functorMap f n)
  -/
  rw [functorMap, Pi.map_eq_prod_map (P := fun m : ℕ ↦ m < n + 1)]
  /-
    C : Type u_1
    M N : Nat → C
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    n : Nat
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limi …
  -/
  apply ( config := { allowSynthFailures := true } ) epi_comp
  /-
    case inst
    C : Type u_1
    M N : Nat → C
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    n : Nat
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  apply ( config := { allowSynthFailures := true } ) epi_comp
  /-
    case inst
    C : Type u_1
    M N : Nat → C
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    n : Nat
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.prod.map (CategoryTheory.Limits.Pi …
  -/
  apply ( config := { allowSynthFailures := true } ) prod.map_epi
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n : Nat
      ⊢ CategoryTheory.Epi (CategoryTheory.Limits.Pi.map fun i => dite (LT.lt (↑i) n …
    -/
  · apply ( config := { allowSynthFailures := true } ) Pi.map_epi
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n : Nat
      ⊢ ∀ (j : Subtype fun x => LT.lt x (HAdd.hAdd n 1)), CategoryTheory.Epi (dite ( …
    -/
    intro ⟨_, _⟩
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n val✝ : Nat
      property✝ : LT.lt val✝ (HAdd.hAdd n 1)
      ⊢ CategoryTheory.Epi (dite (LT.lt (↑⟨val✝, property✝⟩) n) (fun h => CategoryTh …
    -/
    split
    /-
      case inst.isTrue
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n val✝ : Nat
      property✝ : LT.lt val✝ (HAdd.hAdd n 1)
      h✝ : LT.lt (↑⟨val✝, property✝⟩) n
      ⊢ CategoryTheory.Epi (CategoryTheory.eqToHom ⋯)
    -/
    all_goals infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n : Nat
      ⊢ CategoryTheory.Epi (CategoryTheory.Limits.Pi.map fun i => dite (LT.lt (↑i) n …
    -/
  · apply ( config := { allowSynthFailures := true } ) IsIso.epi_of_iso
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n : Nat
      ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.Pi.map fun i => dite (LT.lt (↑i) …
    -/
    apply ( config := { allowSynthFailures := true } ) Pi.map_isIso
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n : Nat
      ⊢ ∀ (b : Subtype fun x => Not (LT.lt x (HAdd.hAdd n 1))), CategoryTheory.IsIso …
    -/
    intro ⟨_, _⟩
    /-
      case inst
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n val✝ : Nat
      property✝ : Not (LT.lt val✝ (HAdd.hAdd n 1))
      ⊢ CategoryTheory.IsIso (dite (LT.lt (↑⟨val✝, property✝⟩) n) (fun h => Category …
    -/
    split
    /-
      case inst.isTrue
      C : Type u_1
      M N : Nat → C
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      f : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝⁴ : CategoryTheory.Limits.HasProductsOfShape Nat C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteBiproducts C
      inst✝¹ : CategoryTheory.Limits.HasCountableProducts C
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
      n val✝ : Nat
      property✝ : Not (LT.lt val✝ (HAdd.hAdd n 1))
      h✝ : LT.lt (↑⟨val✝, property✝⟩) n
      ⊢ CategoryTheory.IsIso (CategoryTheory.eqToHom ⋯)
    -/
    all_goals infer_instance
    /-
      🎉 no goals
    -/

