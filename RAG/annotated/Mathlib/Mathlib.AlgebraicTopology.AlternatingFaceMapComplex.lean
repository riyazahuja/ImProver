/-- The differential on the alternating face map complex is the alternate
sum of the face maps -/
@[simp]
def objD (n : ℕ) : X _[n + 1] ⟶ X _[n] :=
  ∑ i : Fin (n + 2), (-1 : ℤ) ^ (i : ℕ) • X.δ i


/-- ## The chain complex relation `d ≫ d`
-/
theorem d_squared (n : ℕ) : objD X (n + 1) ≫ objD X n = 0 := by
  -- we start by expanding d ≫ d as a double sum
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMap …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Finset.univ.sum fun i => HSMul.hSMul …
  -/
  simp only [comp_sum, sum_comp, ← Finset.sum_product']
  -- then, we decompose the index set P into a subset S and its complement Sᶜ
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq ((SProd.sprod Finset.univ Finset.univ).sum fun x => CategoryTheory.Catego …
  -/
  let P := Fin (n + 2) × Fin (n + 3)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
    ⊢ Eq ((SProd.sprod Finset.univ Finset.univ).sum fun x => CategoryTheory.Catego …
  -/
  let S := Finset.univ.filter fun ij : P => (ij.2 : ℕ) ≤ (ij.1 : ℕ)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
    S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
    ⊢ Eq ((SProd.sprod Finset.univ Finset.univ).sum fun x => CategoryTheory.Catego …
  -/
  erw [← Finset.sum_add_sum_compl S, ← eq_neg_iff_add_eq_zero, ← Finset.sum_neg_distrib]
  /- we are reduced to showing that two sums are equal, and this is obtained
    by constructing a bijection φ : S -> Sᶜ, which maps (i,j) to (j,i+1),
    and by comparing the terms -/
  let φ : ∀ ij : P, ij ∈ S → P := fun ij hij =>
    (Fin.castLT ij.2 (lt_of_le_of_lt (Finset.mem_filter.mp hij).right (Fin.is_lt ij.1)), ij.1.succ)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
    S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
    φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
    ⊢ Eq (S.sum fun i => CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPo …
  -/
  apply Finset.sum_bij φ
  · -- φ(S) is contained in Sᶜ
    /-
      case hi
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      ⊢ ∀ (a : P) (ha : Membership.mem S a), Membership.mem (HasCompl.compl S) (φ a  …
    -/
    intro ij hij
    simp only [S, φ, Finset.mem_univ, Finset.compl_filter, Finset.mem_filter, true_and,
      Fin.val_succ, Fin.coe_castLT] at hij ⊢
    /-
      case hi
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      ij : P
      hij : LE.le ↑ij.2 ↑ij.1
      ⊢ Not (LE.le (HAdd.hAdd (↑ij.1) 1) ↑ij.2)
    -/
    omega
    /-
      🎉 no goals
    -/
  · -- φ : S → Sᶜ is injective
    /-
      case i_inj
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      ⊢ ∀ (a₁ : P) (ha₁ : Membership.mem S a₁) (a₂ : P) (ha₂ : Membership.mem S a₂), …
    -/
    rintro ⟨i, j⟩ hij ⟨i', j'⟩ hij' h
    /-
      case i_inj.mk.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 3)
      hij : Membership.mem S { fst := i, snd := j }
      i' : Fin (HAdd.hAdd n 2)
      j' : Fin (HAdd.hAdd n 3)
      hij' : Membership.mem S { fst := i', snd := j' }
      h : Eq (φ { fst := i, snd := j } hij) (φ { fst := i', snd := j' } hij')
      ⊢ Eq { fst := i, snd := j } { fst := i', snd := j' }
    -/
    rw [Prod.mk.inj_iff]
    exact ⟨by simpa [φ] using congr_arg Prod.snd h,
      by simpa [φ, Fin.castSucc_castLT] using congr_arg Fin.castSucc (congr_arg Prod.fst h)⟩
  · -- φ : S → Sᶜ is surjective
    /-
      case i_surj
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      ⊢ ∀ (b : P), Membership.mem (HasCompl.compl S) b → Exists fun a => Exists fun  …
    -/
    rintro ⟨i', j'⟩ hij'
    simp only [S, Finset.mem_univ, forall_true_left, Prod.forall, Finset.compl_filter,
      not_le, Finset.mem_filter, true_and] at hij'
    /-
      case i_surj.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      i' : Fin (HAdd.hAdd n 2)
      j' : Fin (HAdd.hAdd n 3)
      hij' : LT.lt ↑i' ↑j'
      ⊢ Exists fun a => Exists fun ha => Eq (φ a ha) { fst := i', snd := j' }
    -/
    refine ⟨(j'.pred <| ?_, Fin.castSucc i'), ?_, ?_⟩
      /-
        case i_surj.mk.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i' : Fin (HAdd.hAdd n 2)
        j' : Fin (HAdd.hAdd n 3)
        hij' : LT.lt ↑i' ↑j'
        ⊢ Ne j' 0
      -/
    · rintro rfl
      /-
        case i_surj.mk.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i' : Fin (HAdd.hAdd n 2)
        hij' : LT.lt ↑i' ↑0
        ⊢ False
      -/
      simp only [Fin.val_zero, not_lt_zero'] at hij'
      /-
        🎉 no goals
      -/
    · simpa only [S, Finset.mem_univ, forall_true_left, Prod.forall, Finset.mem_filter,
        Fin.coe_castSucc, Fin.coe_pred, true_and] using Nat.le_sub_one_of_lt hij'
      /-
        case i_surj.mk.refine_3
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i' : Fin (HAdd.hAdd n 2)
        j' : Fin (HAdd.hAdd n 3)
        hij' : LT.lt ↑i' ↑j'
        ⊢ Eq (φ { fst := j'.pred ⋯, snd := i'.castSucc } ⋯) { fst := i', snd := j' }
      -/
    · simp only [φ, Fin.castLT_castSucc, Fin.succ_pred]
      /-
        🎉 no goals
      -/
  · -- identification of corresponding terms in both sums
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      ⊢ ∀ (a : P) (ha : Membership.mem S a), Eq (CategoryTheory.CategoryStruct.comp  …
    -/
    rintro ⟨i, j⟩ hij
    /-
      case h.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 3)
      hij : Membership.mem S { fst := i, snd := j }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPow (-1) ↑{ fst : …
    -/
    dsimp
    /-
      case h.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 3)
      hij : Membership.mem S { fst := i, snd := j }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPow (-1) ↑j) (X.δ …
    -/
    simp only [zsmul_comp, comp_zsmul, smul_smul, ← neg_smul]
    /-
      case h.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
      S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
      φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 3)
      hij : Membership.mem S { fst := i, snd := j }
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) ↑i) (HPow.hPow (-1) ↑j)) (Categor …
    -/
    congr 1
      /-
        case h.mk.e_a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 3)
        hij : Membership.mem S { fst := i, snd := j }
        ⊢ Eq (HMul.hMul (HPow.hPow (-1) ↑i) (HPow.hPow (-1) ↑j)) (Neg.neg (HMul.hMul ( …
      -/
    · simp only [φ, Fin.val_succ, pow_add, pow_one, mul_neg, neg_neg, mul_one]
      /-
        case h.mk.e_a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 3)
        hij : Membership.mem S { fst := i, snd := j }
        ⊢ Eq (HMul.hMul (HPow.hPow (-1) ↑i) (HPow.hPow (-1) ↑j)) (HMul.hMul (HPow.hPow …
      -/
      apply mul_comm
      /-
        🎉 no goals
      -/
      /-
        case h.mk.e_a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 3)
        hij : Membership.mem S { fst := i, snd := j }
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j) (X.δ i)) (CategoryTheory.Cate …
      -/
    · rw [CategoryTheory.SimplicialObject.δ_comp_δ'']
      /-
        case h.mk.e_a.H
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        P : Type := Prod (Fin (HAdd.hAdd n 2)) (Fin (HAdd.hAdd n 3))
        S : Finset P := Finset.filter (fun ij => LE.le ↑ij.2 ↑ij.1) Finset.univ
        φ : (ij : P) → Membership.mem S ij → P := fun ij hij => { fst := ij.2.castLT ⋯ …
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 3)
        hij : Membership.mem S { fst := i, snd := j }
        ⊢ LE.le j i.castSucc
      -/
      simpa [S] using hij
      /-
        🎉 no goals
      -/


/-- The alternating face map complex, on objects -/
def obj : ChainComplex C ℕ :=
  ChainComplex.of (fun n => X _[n]) (objD X) (d_squared X)


@[simp]
theorem obj_X (X : SimplicialObject C) (n : ℕ) : (AlternatingFaceMapComplex.obj X).X n = X _[n] :=
  rfl


@[simp]
theorem obj_d_eq (X : SimplicialObject C) (n : ℕ) :
    (AlternatingFaceMapComplex.obj X).d (n + 1) n
      = ∑ i : Fin (n + 2), (-1 : ℤ) ^ (i : ℕ) • X.δ i := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq ((AlgebraicTopology.AlternatingFaceMapComplex.obj X).d (HAdd.hAdd n 1) n) …
  -/
  apply ChainComplex.of_d
  /-
    🎉 no goals
  -/


/-- The alternating face map complex, on morphisms -/
def map (f : X ⟶ Y) : obj X ⟶ obj Y :=
  ChainComplex.ofHom _ _ _ _ _ _ (fun n => f.app (op [n])) fun n => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => f.app { unop := SimplexCat …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk ( …
    -/
    rw [comp_sum, sum_comp]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (f.app { uno …
    -/
    refine Finset.sum_congr rfl fun _ _ => ?_
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x✝¹ : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk ( …
    -/
    rw [comp_zsmul, zsmul_comp]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x✝¹ : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x✝¹
      ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) ↑x✝¹) (CategoryTheory.CategoryStruct.comp (f …
    -/
    congr 1
    /-
      case e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x✝¹ : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk ( …
    -/
    symm
    /-
      case e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.22357, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x✝¹ : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x✝¹
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ x✝¹) (f.app { unop := SimplexCat …
    -/
    apply f.naturality
    /-
      🎉 no goals
    -/


@[simp]
theorem map_f (f : X ⟶ Y) (n : ℕ) : (map f).f n = f.app (op [n]) :=
  rfl


/-- The alternating face map complex, as a functor -/
def alternatingFaceMapComplex : SimplicialObject C ⥤ ChainComplex C ℕ where
  obj := AlternatingFaceMapComplex.obj
  map f := AlternatingFaceMapComplex.map f


@[simp]
theorem alternatingFaceMapComplex_obj_X (X : SimplicialObject C) (n : ℕ) :
    ((alternatingFaceMapComplex C).obj X).X n = X _[n] :=
  rfl


@[simp]
theorem alternatingFaceMapComplex_obj_d (X : SimplicialObject C) (n : ℕ) :
    ((alternatingFaceMapComplex C).obj X).d (n + 1) n = AlternatingFaceMapComplex.objD X n := by
 /-
   C : Type u_1
   inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
   inst✝ : CategoryTheory.Preadditive C
   X : CategoryTheory.SimplicialObject C
   n : Nat
   ⊢ Eq (((AlgebraicTopology.alternatingFaceMapComplex C).obj X).d (HAdd.hAdd n 1 …
 -/
 dsimp only [alternatingFaceMapComplex, AlternatingFaceMapComplex.obj]
 /-
   C : Type u_1
   inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
   inst✝ : CategoryTheory.Preadditive C
   X : CategoryTheory.SimplicialObject C
   n : Nat
   ⊢ Eq ((ChainComplex.of (fun n => X.obj { unop := SimplexCategory.mk n }) (Alge …
 -/
 apply ChainComplex.of_d
 /-
   🎉 no goals
 -/


@[simp]
theorem alternatingFaceMapComplex_map_f {X Y : SimplicialObject C} (f : X ⟶ Y) (n : ℕ) :
    ((alternatingFaceMapComplex C).map f).f n = f.app (op [n]) :=
  rfl


theorem map_alternatingFaceMapComplex {D : Type*} [Category D] [Preadditive D] (F : C ⥤ D)
    [F.Additive] :
    alternatingFaceMapComplex C ⋙ F.mapHomologicalComplex _ =
      (SimplicialObject.whiskering C D).obj F ⋙ alternatingFaceMapComplex D := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ Eq ((AlgebraicTopology.alternatingFaceMapComplex C).comp (F.mapHomologicalCo …
  -/
  apply CategoryTheory.Functor.ext
    /-
      case h_map
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ autoParam (∀ (X Y : CategoryTheory.SimplicialObject C) (f : Quiver.Hom X Y), …
    -/
  · intro X Y f
    /-
      case h_map
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      ⊢ Eq (((AlgebraicTopology.alternatingFaceMapComplex C).comp (F.mapHomologicalC …
    -/
    ext n
    simp only [Functor.comp_map, HomologicalComplex.comp_f, alternatingFaceMapComplex_map_f,
      Functor.mapHomologicalComplex_map_f, HomologicalComplex.eqToHom_f, eqToHom_refl, comp_id,
      id_comp, SimplicialObject.whiskering_obj_map_app]
    /-
      case h_obj
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ ∀ (X : CategoryTheory.SimplicialObject C), Eq (((AlgebraicTopology.alternati …
    -/
  · intro X
    /-
      case h_obj
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X : CategoryTheory.SimplicialObject C
      ⊢ Eq (((AlgebraicTopology.alternatingFaceMapComplex C).comp (F.mapHomologicalC …
    -/
    apply HomologicalComplex.ext
      /-
        case h_obj.h_d
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_2} D
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.SimplicialObject C
        ⊢ ∀ (i j : Nat), (ComplexShape.down Nat).Rel i j → Eq (CategoryTheory.Category …
      -/
    · rintro i j (rfl : j + 1 = i)
      /-
        case h_obj.h_d
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_2} D
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.SimplicialObject C
        j : Nat
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((AlgebraicTopology.alternatingFace …
      -/
      dsimp only [Functor.comp_obj]
      simp only [Functor.mapHomologicalComplex_obj_d, alternatingFaceMapComplex_obj_d,
        eqToHom_refl, id_comp, comp_id, AlternatingFaceMapComplex.objD, Functor.map_sum,
        Functor.map_zsmul]
      /-
        case h_obj.h_d
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_2} D
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.SimplicialObject C
        j : Nat
        ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (HPow.hPow (-1) ↑x) (F.map (X.δ x)) …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case h_obj.h_X
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_2} D
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.SimplicialObject C
        ⊢ Eq (((AlgebraicTopology.alternatingFaceMapComplex C).comp (F.mapHomologicalC …
      -/
    · ext n
      /-
        case h_obj.h_X.h
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        D : Type u_2
        inst✝² : CategoryTheory.Category.{u_3, u_2} D
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.SimplicialObject C
        n : Nat
        ⊢ Eq ((((AlgebraicTopology.alternatingFaceMapComplex C).comp (F.mapHomological …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem karoubi_alternatingFaceMapComplex_d (P : Karoubi (SimplicialObject C)) (n : ℕ) :
    ((AlternatingFaceMapComplex.obj (KaroubiFunctorCategoryEmbedding.obj P)).d (n + 1) n).f =
      P.p.app (op [n + 1]) ≫ (AlternatingFaceMapComplex.obj P.X).d (n + 1) n := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    ⊢ Eq ((AlgebraicTopology.AlternatingFaceMapComplex.obj (CategoryTheory.Idempot …
  -/
  dsimp
  simp only [AlternatingFaceMapComplex.obj_d_eq, Karoubi.sum_hom, Preadditive.comp_sum,
    Karoubi.zsmul_hom, Preadditive.comp_zsmul]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    ⊢ Eq (Finset.univ.sum fun x => HSMul.hSMul (HPow.hPow (-1) ↑x) (CategoryTheory …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The natural transformation which gives the augmentation of the alternating face map
complex attached to an augmented simplicial object. -/
def ε [Limits.HasZeroObject C] :
    SimplicialObject.Augmented.drop ⋙ AlgebraicTopology.alternatingFaceMapComplex C ⟶
      SimplicialObject.Augmented.point ⋙ ChainComplex.single₀ C where
  app X := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ⊢ Quiver.Hom ((CategoryTheory.SimplicialObject.Augmented.drop.comp (AlgebraicT …
    -/
    refine (ChainComplex.toSingle₀Equiv _ _).symm ?_
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ⊢ Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Si …
    -/
    refine ⟨X.hom.app (op [0]), ?_⟩
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.SimplicialObject.Au …
    -/
    dsimp
    rw [alternatingFaceMapComplex_obj_d, objD, Fin.sum_univ_two, Fin.val_zero,
      pow_zero, one_smul, Fin.val_one, pow_one, neg_smul, one_smul, add_comp,
      neg_comp, SimplicialObject.δ_naturality, SimplicialObject.δ_naturality]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (X.hom.app { unop := Simpl …
    -/
    apply add_neg_cancel
    /-
      🎉 no goals
    -/
  naturality X Y f := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialObject.Aug …
    -/
    apply HomologicalComplex.to_single_hom_ext
    /-
      case hfg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      f : Quiver.Hom X Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialObject.Au …
    -/
    dsimp
    erw [ChainComplex.toSingle₀Equiv_symm_apply_f_zero,
      ChainComplex.toSingle₀Equiv_symm_apply_f_zero]
    /-
      case hfg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.left.app { unop := SimplexCategory …
    -/
    simp only [ChainComplex.single₀_map_f_zero]
    /-
      case hfg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.38970, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.left.app { unop := SimplexCategory …
    -/
    exact congr_app f.w _
    /-
      🎉 no goals
    -/


@[simp]
lemma ε_app_f_zero [Limits.HasZeroObject C] (X : SimplicialObject.Augmented C) :
    (ε.app X).f 0 = X.hom.app (op [0]) :=
  ChainComplex.toSingle₀Equiv_symm_apply_f_zero _ _


@[simp]
lemma ε_app_f_succ [Limits.HasZeroObject C] (X : SimplicialObject.Augmented C) (n : ℕ) :
    (ε.app X).f (n + 1) = 0 := rfl


/-- The inclusion map of the Moore complex in the alternating face map complex -/
def inclusionOfMooreComplexMap (X : SimplicialObject A) :
    (normalizedMooreComplex A).obj X ⟶ (alternatingFaceMapComplex A).obj X := by
  dsimp only [normalizedMooreComplex, NormalizedMooreComplex.obj,
    alternatingFaceMapComplex, AlternatingFaceMapComplex.obj]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    ⊢ Quiver.Hom (ChainComplex.of (fun n => CategoryTheory.Subobject.underlying.ob …
  -/
  apply ChainComplex.ofHom _ _ _ _ _ _ (fun n => (NormalizedMooreComplex.objX X n).arrow)
  /- we have to show the compatibility of the differentials on the alternating
           face map complex with those defined on the normalized Moore complex:
           we first get rid of the terms of the alternating sum that are obviously
           zero on the normalized_Moore_complex -/
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    ⊢ ∀ (i : Nat), Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.Norma …
  -/
  intro i
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.NormalizedMooreCom …
  -/
  simp only [AlternatingFaceMapComplex.objD, comp_sum]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    i : Nat
    ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (AlgebraicTo …
  -/
  rw [Fin.sum_univ_succ, Fintype.sum_eq_zero]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    i : Nat
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.Normali …
  -/
  swap
    /-
      case h
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      i : Nat
      ⊢ ∀ (a : Fin (HAdd.hAdd i 1)), Eq (CategoryTheory.CategoryStruct.comp (Algebra …
    -/
  · intro j
    rw [NormalizedMooreComplex.objX_add_one, comp_zsmul,
      ← factorThru_arrow _ _ (finset_inf_arrow_factors Finset.univ _ _ (Finset.mem_univ j)),
      Category.assoc, kernelSubobject_arrow_comp, comp_zero, smul_zero]
  -- finally, we study the remaining term which is induced by X.δ 0
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    i : Nat
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.Normali …
  -/
  rw [add_zero, Fin.val_zero, pow_zero, one_zsmul]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.NormalizedMooreCom …
  -/
  dsimp [NormalizedMooreComplex.objD, NormalizedMooreComplex.objX]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.54792, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.54815, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Finset.univ.inf fun k => CategoryThe …
  -/
              /-
                🎉 no goals
              -/
  cases i <;> simp
              /-
                🎉 no goals
              -/


@[simp]
theorem inclusionOfMooreComplexMap_f (X : SimplicialObject A) (n : ℕ) :
    (inclusionOfMooreComplexMap X).f n = (NormalizedMooreComplex.objX X n).arrow := by
  /-
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    n : Nat
    ⊢ Eq ((AlgebraicTopology.inclusionOfMooreComplexMap X).f n) (AlgebraicTopology …
  -/
  dsimp only [inclusionOfMooreComplexMap]
  /-
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    n : Nat
    ⊢ Eq ((id (ChainComplex.ofHom (fun n => CategoryTheory.Subobject.underlying.ob …
  -/
  exact ChainComplex.ofHom_f _ _ _ _ _ _ _ _ n
  /-
    🎉 no goals
  -/


/-- The inclusion map of the Moore complex in the alternating face map complex,
as a natural transformation -/
@[simps]
def inclusionOfMooreComplex : normalizedMooreComplex A ⟶ alternatingFaceMapComplex A where
  app := inclusionOfMooreComplexMap


/-- The differential on the alternating coface map complex is the alternate
sum of the coface maps -/
@[simp]
def objD (n : ℕ) : X.obj [n] ⟶ X.obj [n + 1] :=
  ∑ i : Fin (n + 2), (-1 : ℤ) ^ (i : ℕ) • X.δ i


theorem d_eq_unop_d (n : ℕ) :
    objD X n =
      (AlternatingFaceMapComplex.objD ((cosimplicialSimplicialEquiv C).functor.obj (op X))
          n).unop := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    ⊢ Eq (AlgebraicTopology.AlternatingCofaceMapComplex.objD X n) (AlgebraicTopolo …
  -/
  simp only [objD, AlternatingFaceMapComplex.objD, unop_sum, unop_zsmul]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (HPow.hPow (-1) ↑i) (X.δ i)) (Finse …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem d_squared (n : ℕ) : objD X n ≫ objD X (n + 1) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingCofaceM …
  -/
  simp only [d_eq_unop_d, ← unop_comp, AlternatingFaceMapComplex.d_squared, unop_zero]
  /-
    🎉 no goals
  -/


/-- The alternating coface map complex, on objects -/
def obj : CochainComplex C ℕ :=
  CochainComplex.of (fun n => X.obj [n]) (objD X) (d_squared X)


/-- The alternating face map complex, on morphisms -/
@[simp]
def map (f : X ⟶ Y) : obj X ⟶ obj Y :=
  CochainComplex.ofHom _ _ _ _ _ _ (fun n => f.app [n]) fun n => by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => f.app (SimplexCategory.mk  …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app (SimplexCategory.mk n)) (Finse …
    -/
    rw [comp_sum, sum_comp]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (f.app (Simp …
    -/
    refine Finset.sum_congr rfl fun x _ => ?_
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app (SimplexCategory.mk n)) (HSMul …
    -/
    rw [comp_zsmul, zsmul_comp]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x
      ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) ↑x) (CategoryTheory.CategoryStruct.comp (f.a …
    -/
    congr 1
    /-
      case e_a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app (SimplexCategory.mk n)) (Y.δ x …
    -/
    symm
    /-
      case e_a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.73355, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      A : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.73378, u_2} A
      inst✝ : CategoryTheory.Abelian A
      X Y : CategoryTheory.CosimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      x : Fin (HAdd.hAdd n 2)
      x✝ : Membership.mem Finset.univ x
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ x) (f.app (SimplexCategory.mk (H …
    -/
    apply f.naturality
    /-
      🎉 no goals
    -/


/-- The alternating coface map complex, as a functor -/
@[simps]
def alternatingCofaceMapComplex : CosimplicialObject C ⥤ CochainComplex C ℕ where
  obj := AlternatingCofaceMapComplex.obj
  map f := AlternatingCofaceMapComplex.map f


