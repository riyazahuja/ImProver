/-- A morphism `φ : Y ⟶ X _[n+1]` satisfies `HigherFacesVanish q φ`
when the compositions `φ ≫ X.δ j` are `0` for `j ≥ max 1 (n+2-q)`. When `q ≤ n+1`,
it basically means that the composition `φ ≫ X.δ j` are `0` for the `q` highest
possible values of a nonzero `j`. Otherwise, when `q ≥ n+2`, all the compositions
`φ ≫ X.δ j` for nonzero `j` vanish. See also the lemma `comp_P_eq_self_iff` in
`Projections.lean` which states that `HigherFacesVanish q φ` is equivalent to
the identity `φ ≫ (P q).f (n+1) = φ`. -/
def HigherFacesVanish {Y : C} {n : ℕ} (q : ℕ) (φ : Y ⟶ X _[n + 1]) : Prop :=
  ∀ j : Fin (n + 1), n + 1 ≤ (j : ℕ) + q → φ ≫ X.δ j.succ = 0


@[reassoc]
theorem comp_δ_eq_zero {Y : C} {n : ℕ} {q : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish q φ)
    (j : Fin (n + 2)) (hj₁ : j ≠ 0) (hj₂ : n + 2 ≤ (j : ℕ) + q) : φ ≫ X.δ j = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 2)
    hj₁ : Ne j 0
    hj₂ : LE.le (HAdd.hAdd n 2) (HAdd.hAdd (↑j) q)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j)) 0
  -/
  obtain ⟨i, rfl⟩ := Fin.eq_succ_of_ne_zero hj₁
  /-
    case intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    i : Fin (HAdd.hAdd n 1)
    hj₁ : Ne i.succ 0
    hj₂ : LE.le (HAdd.hAdd n 2) (HAdd.hAdd (↑i.succ) q)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ i.succ)) 0
  -/
  apply v i
  /-
    case intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    i : Fin (HAdd.hAdd n 1)
    hj₁ : Ne i.succ 0
    hj₂ : LE.le (HAdd.hAdd n 2) (HAdd.hAdd (↑i.succ) q)
    ⊢ LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
  -/
  simp only [Fin.val_succ] at hj₂
  /-
    case intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    i : Fin (HAdd.hAdd n 1)
    hj₁ : Ne i.succ 0
    hj₂ : LE.le (HAdd.hAdd n 2) (HAdd.hAdd (HAdd.hAdd (↑i) 1) q)
    ⊢ LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem of_succ {Y : C} {n q : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish (q + 1) φ) :
                                                 /-
                                                   C : Type u_1
                                                   inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                                   inst✝ : CategoryTheory.Preadditive C
                                                   X : CategoryTheory.SimplicialObject C
                                                   Y : C
                                                   n q : Nat
                                                   φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
                                                   v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
                                                   j : Fin (HAdd.hAdd n 1)
                                                   hj : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) q)
                                                   ⊢ LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
                                                 -/
    HigherFacesVanish q φ := fun j hj => v j (by simpa only [← add_assoc] using le_add_right hj)
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem of_comp {Y Z : C} {q n : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish q φ) (f : Z ⟶ Y) :
                                                  /-
                                                    C : Type u_1
                                                    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                                    inst✝ : CategoryTheory.Preadditive C
                                                    X : CategoryTheory.SimplicialObject C
                                                    Y Z : C
                                                    q n : Nat
                                                    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
                                                    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
                                                    f : Quiver.Hom Z Y
                                                    j : Fin (HAdd.hAdd n 1)
                                                    hj : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) q)
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                                  -/
    HigherFacesVanish q (f ≫ φ) := fun j hj => by rw [assoc, v j hj, comp_zero]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem comp_Hσ_eq {Y : C} {n a q : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish q φ)
    (hnaq : n = a + q) :
    φ ≫ (Hσ q).f (n + 1) =
      -φ ≫ X.δ ⟨a + 1, Nat.succ_lt_succ (Nat.lt_succ_iff.mpr (Nat.le.intro hnaq.symm))⟩ ≫
        X.σ ⟨a, Nat.lt_succ_iff.mpr (Nat.le.intro hnaq.symm)⟩ := by
  have hnaq_shift : ∀ d : ℕ, n + d = a + d + q := by
    intro d
    rw [add_assoc, add_comm d, ← add_assoc, hnaq]
  rw [Hσ, Homotopy.nullHomotopicMap'_f (c_mk (n + 2) (n + 1) rfl) (c_mk (n + 1) n rfl),
    hσ'_eq hnaq (c_mk (n + 1) n rfl), hσ'_eq (hnaq_shift 1) (c_mk (n + 2) (n + 1) rfl)]
  simp only [AlternatingFaceMapComplex.obj_d_eq, eqToHom_refl, comp_id, comp_sum, sum_comp,
    comp_add]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp φ …
  -/
  simp only [comp_zsmul, zsmul_comp, ← assoc, ← mul_zsmul]
  -- cleaning up the first sum
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun x => HSMul.hSMul (HMul.hMul (HPow.hPow (- …
  -/
  rw [← Fin.sum_congr' _ (hnaq_shift 2).symm, Fin.sum_trunc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (HPow.hPow (- …
  -/
  swap
    /-
      case hf
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      ⊢ ∀ (j : Fin q), Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (-1) …
    -/
  · rintro ⟨k, hk⟩
    suffices φ ≫ X.δ (⟨a + 2 + k, by omega⟩ : Fin (n + 2)) = 0 by
      simp only [this, Fin.natAdd_mk, Fin.cast_mk, zero_comp, smul_zero]
    /-
      case hf.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      k : Nat
      hk : LT.lt k q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ ⟨HAdd.hAdd (HAdd.hAdd a 2) k,  …
    -/
    convert v ⟨a + k + 1, by omega⟩ (by rw [Fin.val_mk]; omega)
    /-
      case h.e'_2.h.e'_7.h.e'_5.h.e'_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      k : Nat
      hk : LT.lt k q
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd a 2) k) ↑⟨HAdd.hAdd (HAdd.hAdd a k) 1, ⋯⟩.succ
    -/
    dsimp
    /-
      case h.e'_2.h.e'_7.h.e'_5.h.e'_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      k : Nat
      hk : LT.lt k q
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd a 2) k) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd a k) 1) 1)
    -/
    omega
    /-
      🎉 no goals
    -/
  -- cleaning up the second sum
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (HPow.hPow (- …
  -/
  rw [← Fin.sum_congr' _ (hnaq_shift 3).symm, @Fin.sum_trunc _ _ (a + 3)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (HPow.hPow (- …
  -/
  swap
    /-
      case hf
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      ⊢ ∀ (j : Fin q), Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) ↑(Fin.cast ⋯ (Fin. …
    -/
  · rintro ⟨k, hk⟩
    /-
      case hf.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      k : Nat
      hk : LT.lt k q
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) ↑(Fin.cast ⋯ (Fin.natAdd (HAdd.hA …
    -/
    rw [assoc, X.δ_comp_σ_of_gt', v.comp_δ_eq_zero_assoc, zero_comp, zsmul_zero]
      /-
        case hf.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ LT.lt ⟨HAdd.hAdd a 1, ⋯⟩.succ (Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd a 3) ⟨k, hk …
      -/
    · simp only [Fin.lt_iff_val_lt_val]
      /-
        case hf.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ LT.lt ↑⟨HAdd.hAdd a 1, ⋯⟩.succ ↑(Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd a 3) ⟨k,  …
      -/
      dsimp [Fin.natAdd, Fin.cast]
      /-
        case hf.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd a 1) 1) (HAdd.hAdd (HAdd.hAdd a 3) k)
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case hf.mk.hj₁
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ Ne ((Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd a 3) ⟨k, hk⟩)).pred ⋯) 0
      -/
    · intro h
      /-
        case hf.mk.hj₁
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        h : Eq ((Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd a 3) ⟨k, hk⟩)).pred ⋯) 0
        ⊢ False
      -/
      rw [Fin.pred_eq_iff_eq_succ, Fin.ext_iff] at h
      /-
        case hf.mk.hj₁
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        h : Eq ↑(Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd a 3) ⟨k, hk⟩)) ↑(Fin.succ 0)
        ⊢ False
      -/
      dsimp [Fin.cast] at h
      /-
        case hf.mk.hj₁
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        h : Eq (HAdd.hAdd (HAdd.hAdd a 3) k) 1
        ⊢ False
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case hf.mk.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ LE.le (HAdd.hAdd n 2) (HAdd.hAdd (↑((Fin.cast ⋯ (Fin.natAdd (HAdd.hAdd a 3)  …
      -/
    · dsimp [Fin.cast, Fin.pred]
      /-
        case hf.mk.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ LE.le (HAdd.hAdd n 2) (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd a 3) k) 1) …
      -/
      rw [Nat.add_right_comm, Nat.add_sub_assoc (by norm_num : 1 ≤ 3)]
      /-
        case hf.mk.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n a q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hnaq : Eq n (HAdd.hAdd a q)
        hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
        k : Nat
        hk : LT.lt k q
        ⊢ LE.le (HAdd.hAdd n 2) (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd a k) (HSub.hSub 3 1)) …
      -/
      omega
      /-
        🎉 no goals
      -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (HPow.hPow (- …
  -/
  simp only [assoc]
  conv_lhs =>
    congr
    · rw [Fin.sum_univ_castSucc]
    · rw [Fin.sum_univ_castSucc, Fin.sum_univ_castSucc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (H …
  -/
  dsimp [Fin.cast, Fin.castLE, Fin.castLT]
  /- the purpose of the following `simplif` is to create three subgoals in order
      to finish the proof -/
  have simplif :
    ∀ a b c d e f : Y ⟶ X _[n + 1], b = f → d + e = 0 → c + a = 0 → a + b + (c + d + e) = f := by
    intro a b c d e f h1 h2 h3
    rw [add_assoc c d e, h2, add_zero, add_comm a, add_assoc, add_comm a, h3, add_zero, h1]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n a q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hnaq : Eq n (HAdd.hAdd a q)
    hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
    simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (H …
  -/
  apply simplif
  · -- b = f
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) a) (HPow.hPow (-1) (HAdd.hAdd a 1 …
    -/
    rw [← pow_add, Odd.neg_one_pow, neg_smul, one_zsmul]
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      ⊢ Odd (HAdd.hAdd a (HAdd.hAdd a 1))
    -/
    exact ⟨a, by omega⟩
    /-
      🎉 no goals
    -/
  · -- d + e = 0
    rw [X.δ_comp_σ_self' (Fin.castSucc_mk _ _ _).symm,
      X.δ_comp_σ_succ' (Fin.succ_mk _ _ _).symm]
    simp only [comp_id, pow_add _ (a + 1) 1, pow_one, mul_neg, mul_one, neg_mul, neg_smul,
      add_neg_cancel]
  · -- c + a = 0
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      ⊢ Eq (HAdd.hAdd (Finset.univ.sum fun i => HSMul.hSMul (HMul.hMul (HPow.hPow (- …
    -/
    rw [← Finset.sum_add_distrib]
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      ⊢ Eq (Finset.univ.sum fun x => HAdd.hAdd (HSMul.hSMul (HMul.hMul (HPow.hPow (- …
    -/
    apply Finset.sum_eq_zero
    /-
      case a.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      ⊢ ∀ (x : Fin (HAdd.hAdd a 1)), Membership.mem Finset.univ x → Eq (HAdd.hAdd (H …
    -/
    rintro ⟨i, hi⟩ _
    /-
      case a.h.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      i : Nat
      hi : LT.lt i (HAdd.hAdd a 1)
      a✝ : Membership.mem Finset.univ ⟨i, hi⟩
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) ↑⟨i, hi⟩) (HPow.hPow ( …
    -/
    simp only
    have hia : (⟨i, by omega⟩ : Fin (n + 2)) ≤
        Fin.castSucc (⟨a, by omega⟩ : Fin (n + 1)) := by
      rw [Fin.le_iff_val_le_val]
      dsimp
      omega
    /-
      case a.h.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      i : Nat
      hi : LT.lt i (HAdd.hAdd a 1)
      a✝ : Membership.mem Finset.univ ⟨i, hi⟩
      hia : LE.le ⟨i, ⋯⟩ ⟨a, ⋯⟩.castSucc
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) i) (HPow.hPow (-1) (HA …
    -/
    erw [δ_comp_σ_of_le X hia, add_eq_zero_iff_eq_neg, ← neg_zsmul]
    /-
      case a.h.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      i : Nat
      hi : LT.lt i (HAdd.hAdd a 1)
      a✝ : Membership.mem Finset.univ ⟨i, hi⟩
      hia : LE.le ⟨i, ⋯⟩ ⟨a, ⋯⟩.castSucc
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) i) (HPow.hPow (-1) (HAdd.hAdd a 1 …
    -/
    congr 2
    /-
      case a.h.mk.e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n a q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnaq : Eq n (HAdd.hAdd a q)
      hnaq_shift : ∀ (d : Nat), Eq (HAdd.hAdd n d) (HAdd.hAdd (HAdd.hAdd a d) q)
      simplif : ∀ (a b c d e f : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (H …
      i : Nat
      hi : LT.lt i (HAdd.hAdd a 1)
      a✝ : Membership.mem Finset.univ ⟨i, hi⟩
      hia : LE.le ⟨i, ⋯⟩ ⟨a, ⋯⟩.castSucc
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) i) (HPow.hPow (-1) (HAdd.hAdd a 1))) (Neg.neg  …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem comp_Hσ_eq_zero {Y : C} {n q : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish q φ)
    (hqn : n < q) : φ ≫ (Hσ q).f (n + 1) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hqn : LT.lt n q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.Hσ q).f …
  -/
  simp only [Hσ, Homotopy.nullHomotopicMap'_f (c_mk (n + 2) (n + 1) rfl) (c_mk (n + 1) n rfl)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hqn : LT.lt n q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (HAdd.hAdd (CategoryTheory.Category …
  -/
  rw [hσ'_eq_zero hqn (c_mk (n + 1) n rfl), comp_zero, zero_add]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    hqn : LT.lt n q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
  -/
  by_cases hqn' : n + 1 < q
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hqn : LT.lt n q
      hqn' : LT.lt (HAdd.hAdd n 1) q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
    -/
  · rw [hσ'_eq_zero hqn' (c_mk (n + 2) (n + 1) rfl), zero_comp, comp_zero]
    /-
      🎉 no goals
    -/
  · simp only [hσ'_eq (show n + 1 = 0 + q by omega) (c_mk (n + 2) (n + 1) rfl), pow_zero,
      Fin.mk_zero, one_zsmul, eqToHom_refl, comp_id, comp_sum,
      AlternatingFaceMapComplex.obj_d_eq]
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hqn : LT.lt n q
      hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
      ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp φ (CategoryT …
    -/
    rw [← Fin.sum_congr' _ (show 2 + (n + 1) = n + 1 + 2 by omega), Fin.sum_trunc]
    · simp only [Fin.sum_univ_castSucc, Fin.sum_univ_zero, zero_add, Fin.last, Fin.castLE_mk,
        Fin.cast_mk, Fin.castSucc_mk]
      simp only [Fin.mk_zero, Fin.val_zero, pow_zero, one_zsmul, Fin.mk_one, Fin.val_one, pow_one,
        neg_smul, comp_neg]
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : LT.lt n q
        hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.Category …
      -/
      erw [δ_comp_σ_self, δ_comp_σ_succ, add_neg_cancel]
      /-
        🎉 no goals
      -/
      /-
        case neg.hf
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : LT.lt n q
        hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
        ⊢ ∀ (j : Fin (HAdd.hAdd n 1)), Eq (CategoryTheory.CategoryStruct.comp φ (Categ …
      -/
    · intro j
      /-
        case neg.hf
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : LT.lt n q
        hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
      -/
      dsimp [Fin.cast, Fin.castLE, Fin.castLT]
      /-
        case neg.hf
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n q : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : LT.lt n q
        hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
      -/
      rw [comp_zsmul, comp_zsmul, δ_comp_σ_of_gt', v.comp_δ_eq_zero_assoc, zero_comp, zsmul_zero]
        /-
          case neg.hf.H
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          Y : C
          n q : Nat
          φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
          hqn : LT.lt n q
          hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
          j : Fin (HAdd.hAdd n 1)
          ⊢ LT.lt (Fin.succ 0) ⟨HAdd.hAdd 2 ↑j, ⋯⟩
        -/
      · simp only [Fin.lt_iff_val_lt_val]
        /-
          case neg.hf.H
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          Y : C
          n q : Nat
          φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
          hqn : LT.lt n q
          hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
          j : Fin (HAdd.hAdd n 1)
          ⊢ LT.lt (↑(Fin.succ 0)) (HAdd.hAdd 2 ↑j)
        -/
        dsimp [Fin.succ]
        /-
          case neg.hf.H
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          Y : C
          n q : Nat
          φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
          hqn : LT.lt n q
          hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
          j : Fin (HAdd.hAdd n 1)
          ⊢ LT.lt 1 (HAdd.hAdd 2 ↑j)
        -/
        omega
        /-
          🎉 no goals
        -/
        /-
          case neg.hf.hj₁
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          Y : C
          n q : Nat
          φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
          hqn : LT.lt n q
          hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
          j : Fin (HAdd.hAdd n 1)
          ⊢ Ne (⟨HAdd.hAdd 2 ↑j, ⋯⟩.pred ⋯) 0
        -/
      · intro h
        simp only [Fin.pred, Fin.subNat, Fin.ext_iff, Nat.succ_add_sub_one,
          Fin.val_zero, add_eq_zero, false_and, reduceCtorEq] at h
        /-
          case neg.hf.hj₂
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          Y : C
          n q : Nat
          φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
          hqn : LT.lt n q
          hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
          j : Fin (HAdd.hAdd n 1)
          ⊢ LE.le (HAdd.hAdd n 2) (HAdd.hAdd (↑(⟨HAdd.hAdd 2 ↑j, ⋯⟩.pred ⋯)) q)
        -/
      · simp only [Fin.pred, Fin.subNat, Nat.pred_eq_sub_one, Nat.succ_add_sub_one]
        /-
          case neg.hf.hj₂
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          Y : C
          n q : Nat
          φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
          hqn : LT.lt n q
          hqn' : Not (LT.lt (HAdd.hAdd n 1) q)
          j : Fin (HAdd.hAdd n 1)
          ⊢ LE.le (HAdd.hAdd n 2) (HAdd.hAdd (HAdd.hAdd 1 ↑j) q)
        -/
        omega
        /-
          🎉 no goals
        -/


theorem induction {Y : C} {n q : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish q φ) :
    HigherFacesVanish (q + 1) (φ ≫ (𝟙 _ + Hσ q).f (n + 1)) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) (CategoryTheory. …
  -/
  intro j hj₁
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 1)
    hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 1)
    hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
  -/
  simp only [comp_add, add_comp, comp_id]
  -- when n < q, the result follows immediately from the assumption
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 1)
    hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryT …
  -/
  by_cases hqn : n < q
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd n 1)
      hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : LT.lt n q
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryT …
    -/
  · rw [v.comp_Hσ_eq_zero hqn, zero_comp, add_zero, v j (by omega)]
    /-
      🎉 no goals
    -/
  -- we now assume that n≥q, and write n=a+q
  /-
    case neg
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 1)
    hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt n q)
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryT …
  -/
  cases' Nat.le.dest (not_lt.mp hqn) with a ha
  /-
    case neg.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 1)
    hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt n q)
    a : Nat
    ha : Eq (HAdd.hAdd q a) n
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryT …
  -/
  rw [v.comp_Hσ_eq (show n = a + q by omega), neg_comp, add_neg_eq_zero, assoc, assoc]
  /-
    case neg.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd n 1)
    hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt n q)
    a : Nat
    ha : Eq (HAdd.hAdd q a) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  cases' n with m hm
  -- the boundary case n=0
  · simp only [Nat.eq_zero_of_add_eq_zero_left ha, Fin.eq_zero j, Fin.mk_zero, Fin.mk_one,
      δ_comp_σ_succ, comp_id]
    /-
      case neg.intro.zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd 0 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd 0 1)
      hj₁ : LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt 0 q)
      ha : Eq (HAdd.hAdd q a) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ (Fin.succ 0))) (CategoryTheory …
    -/
    rfl
    /-
      🎉 no goals
    -/
  -- in the other case, we need to write n as m+1
  -- then, we first consider the particular case j = a
  /-
    case neg.intro.succ
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    q a m : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
    hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt (HAdd.hAdd m 1) q)
    ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  by_cases hj₂ : a = (j : ℕ)
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Eq a ↑j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
  · simp only [hj₂, Fin.eta, δ_comp_σ_succ, comp_id]
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Eq a ↑j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  -- now, we assume j ≠ a (i.e. a < j)
  /-
    case neg
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    q a m : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
    hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt (HAdd.hAdd m 1) q)
    ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
    hj₂ : Not (Eq a ↑j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  have haj : a < j := (Ne.le_iff_lt hj₂).mp (by omega)
  /-
    case neg
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    q a m : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
    hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt (HAdd.hAdd m 1) q)
    ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
    hj₂ : Not (Eq a ↑j)
    haj : LT.lt a ↑j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  have ham : a ≤ m := by omega
  /-
    case neg
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    q a m : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
    hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt (HAdd.hAdd m 1) q)
    ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
    hj₂ : Not (Eq a ↑j)
    haj : LT.lt a ↑j
    ham : LE.le a m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  rw [X.δ_comp_σ_of_gt', j.pred_succ]
  /-
    case neg
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    q a m : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
    hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt (HAdd.hAdd m 1) q)
    ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
    hj₂ : Not (Eq a ↑j)
    haj : LT.lt a ↑j
    ham : LE.le a m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  swap
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      ⊢ LT.lt ⟨a, ⋯⟩.succ j.succ
    -/
  · rw [Fin.lt_iff_val_lt_val]
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      ⊢ LT.lt ↑⟨a, ⋯⟩.succ ↑j.succ
    -/
    simpa only [Fin.val_mk, Fin.val_succ, add_lt_add_iff_right] using haj
    /-
      🎉 no goals
    -/
  /-
    case neg
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    q a m : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
    hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
    hqn : Not (LT.lt (HAdd.hAdd m 1) q)
    ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
    hj₂ : Not (Eq a ↑j)
    haj : LT.lt a ↑j
    ham : LE.le a m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
  -/
  obtain _ | ham'' := ham.lt_or_eq
  · -- case where `a<m`
    /-
      case neg.inl
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      h✝ : LT.lt a m
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    rw [← X.δ_comp_δ''_assoc]
    /-
      case neg.inl
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      h✝ : LT.lt a m
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    swap
      /-
        case neg.inl.H
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        hj₂ : Not (Eq a ↑j)
        haj : LT.lt a ↑j
        ham : LE.le a m
        h✝ : LT.lt a m
        ⊢ LE.le ⟨HAdd.hAdd a 1, ⋯⟩ j.castSucc
      -/
    · rw [Fin.le_iff_val_le_val]
      /-
        case neg.inl.H
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        hj₂ : Not (Eq a ↑j)
        haj : LT.lt a ↑j
        ham : LE.le a m
        h✝ : LT.lt a m
        ⊢ LE.le ↑⟨HAdd.hAdd a 1, ⋯⟩ ↑j.castSucc
      -/
      dsimp
      /-
        case neg.inl.H
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        hj₂ : Not (Eq a ↑j)
        haj : LT.lt a ↑j
        ham : LE.le a m
        h✝ : LT.lt a m
        ⊢ LE.le (HAdd.hAdd a 1) ↑j
      -/
      omega
      /-
        🎉 no goals
      -/
    /-
      case neg.inl
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      h✝ : LT.lt a m
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    simp only [← assoc, v j (by omega), zero_comp]
    /-
      🎉 no goals
    -/
  · -- in the last case, a=m, q=1 and j=a+1
    /-
      case neg.inr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      ham'' : Eq a m
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    rw [X.δ_comp_δ_self'_assoc]
    /-
      case neg.inr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      ham'' : Eq a m
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    swap
      /-
        case neg.inr.H
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        hj₂ : Not (Eq a ↑j)
        haj : LT.lt a ↑j
        ham : LE.le a m
        ham'' : Eq a m
        ⊢ Eq ⟨HAdd.hAdd a 1, ⋯⟩ j.castSucc
      -/
    · ext
      /-
        case neg.inr.H.h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        hj₂ : Not (Eq a ↑j)
        haj : LT.lt a ↑j
        ham : LE.le a m
        ham'' : Eq a m
        ⊢ Eq ↑⟨HAdd.hAdd a 1, ⋯⟩ ↑j.castSucc
      -/
      cases j
      /-
        case neg.inr.H.h.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        ham : LE.le a m
        ham'' : Eq a m
        val✝ : Nat
        isLt✝ : LT.lt val✝ (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑⟨val✝, isLt✝⟩) (HAdd.hA …
        hj₂ : Not (Eq a ↑⟨val✝, isLt✝⟩)
        haj : LT.lt a ↑⟨val✝, isLt✝⟩
        ⊢ Eq ↑⟨HAdd.hAdd a 1, ⋯⟩ ↑⟨val✝, isLt✝⟩.castSucc
      -/
      dsimp
      /-
        case neg.inr.H.h.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        ham : LE.le a m
        ham'' : Eq a m
        val✝ : Nat
        isLt✝ : LT.lt val✝ (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑⟨val✝, isLt✝⟩) (HAdd.hA …
        hj₂ : Not (Eq a ↑⟨val✝, isLt✝⟩)
        haj : LT.lt a ↑⟨val✝, isLt✝⟩
        ⊢ Eq (HAdd.hAdd a 1) val✝
      -/
      dsimp only [Nat.succ_eq_add_one] at *
      /-
        case neg.inr.H.h.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        q a m : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
        hqn : Not (LT.lt (HAdd.hAdd m 1) q)
        ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
        ham : LE.le a m
        ham'' : Eq a m
        val✝ : Nat
        isLt✝ : LT.lt val✝ (HAdd.hAdd (HAdd.hAdd m 1) 1)
        hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd val✝ (HAdd.hAdd q 1))
        hj₂ : Not (Eq a val✝)
        haj : LT.lt a val✝
        ⊢ Eq (HAdd.hAdd a 1) val✝
      -/
      omega
      /-
        🎉 no goals
      -/
    /-
      case neg.inr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      q a m : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd m 1 …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      j : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)
      hj₁ : LE.le (HAdd.hAdd (HAdd.hAdd m 1) 1) (HAdd.hAdd (↑j) (HAdd.hAdd q 1))
      hqn : Not (LT.lt (HAdd.hAdd m 1) q)
      ha : Eq (HAdd.hAdd q a) (HAdd.hAdd m 1)
      hj₂ : Not (Eq a ↑j)
      haj : LT.lt a ↑j
      ham : LE.le a m
      ham'' : Eq a m
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (X.δ j.succ)) (CategoryTheory.Categ …
    -/
    simp only [← assoc, v j (by omega), zero_comp]
    /-
      🎉 no goals
    -/


