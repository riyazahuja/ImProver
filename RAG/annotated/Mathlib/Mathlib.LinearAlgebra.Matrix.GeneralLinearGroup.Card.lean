local notation "q" => Fintype.card K

local notation "n" => Module.finrank K V


attribute [local instance] Fintype.ofFinite in
open Fintype in
/-- The cardinal of the set of linearly independent vectors over a finite dimensional vector space
over a finite field. -/
theorem card_linearIndependent {k : ℕ} (hk : k ≤ n) :
    Nat.card { s : Fin k → V // LinearIndependent K s } =
      ∏ i : Fin k, (q ^ n - q ^ i.val) := by
  /-
    K : Type u_1
    V : Type u_2
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Fintype K
    inst✝ : Finite V
    k : Nat
    hk : LE.le k (Module.finrank K V)
    ⊢ Eq (Nat.card (Subtype fun s => LinearIndependent K s)) (Finset.univ.prod fun …
  -/
  rw [Nat.card_eq_fintype_card]
  induction k with
  | zero => simp only [linearIndependent_iff_ker, Finsupp.linearCombination_fin_zero, ker_zero,
      card_ofSubsingleton, Finset.univ_eq_empty, Finset.prod_empty]
  | succ k ih =>
      have (s : { s : Fin k → V // LinearIndependent K s }) :
          card ((Submodule.span K (Set.range (s : Fin k → V)))ᶜ : Set (V)) =
          (q) ^ n - (q) ^ k := by
            rw [card_compl_set, card_eq_pow_finrank (K := K)
            (V := ((Submodule.span K (Set.range (s : Fin k → V))) : Set (V)))]
            simp only [SetLike.coe_sort_coe, finrank_span_eq_card s.2, card_fin]
            rw [card_eq_pow_finrank (K := K)]
      simp [card_congr (equiv_linearIndependent k), sum_congr _ _ this, ih (Nat.le_of_succ_le hk),
        mul_comm, Fin.prod_univ_succAbove _ k]


local notation "q" => Fintype.card 𝔽


/-- Equivalence between `GL n F` and `n` vectors of length `n` that are linearly independent. Given
by sending a matrix to its columns. -/
noncomputable def equiv_GL_linearindependent (hn : 0 < n) :
    GL (Fin n) 𝔽 ≃ { s : Fin n → Fin n → 𝔽 // LinearIndependent 𝔽 s } where
  toFun M := ⟨transpose M, by
    /-
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : LT.lt 0 n
      M : Matrix.GeneralLinearGroup (Fin n) 𝔽
      ⊢ LinearIndependent 𝔽 (↑M).transpose
    -/
    apply linearIndependent_iff_card_eq_finrank_span.2
    /-
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : LT.lt 0 n
      M : Matrix.GeneralLinearGroup (Fin n) 𝔽
      ⊢ Eq (Fintype.card (Fin n)) (Set.finrank 𝔽 (Set.range (↑M).transpose))
    -/
    rw [Set.finrank, ← rank_eq_finrank_span_cols, rank_unit]⟩
    /-
      🎉 no goals
    -/
  invFun M := GeneralLinearGroup.mk'' (transpose (M.1)) <| by
    /-
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : LT.lt 0 n
      M : Subtype fun s => LinearIndependent 𝔽 s
      ⊢ IsUnit (Matrix.transpose ↑M).det
    -/
    have : Nonempty (Fin n) := Fin.pos_iff_nonempty.1 hn
    /-
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : LT.lt 0 n
      M : Subtype fun s => LinearIndependent 𝔽 s
      this : Nonempty (Fin n)
      ⊢ IsUnit (Matrix.transpose ↑M).det
    -/
    let b := basisOfLinearIndependentOfCardEqFinrank M.2 (by simp)
    /-
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : LT.lt 0 n
      M : Subtype fun s => LinearIndependent 𝔽 s
      this : Nonempty (Fin n)
      b : Basis (Fin n) 𝔽 (Fin n → 𝔽) := basisOfLinearIndependentOfCardEqFinrank ⋯ ⋯
      ⊢ IsUnit (Matrix.transpose ↑M).det
    -/
    have := (Pi.basisFun 𝔽 (Fin n)).invertibleToMatrix b
    rw [← Basis.coePiBasisFun.toMatrix_eq_transpose,
      ← coe_basisOfLinearIndependentOfCardEqFinrank M.2]
    /-
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : LT.lt 0 n
      M : Subtype fun s => LinearIndependent 𝔽 s
      this✝ : Nonempty (Fin n)
      b : Basis (Fin n) 𝔽 (Fin n → 𝔽) := basisOfLinearIndependentOfCardEqFinrank ⋯ ⋯
      this : Invertible ((Pi.basisFun 𝔽 (Fin n)).toMatrix ⇑b)
      ⊢ IsUnit ((Pi.basisFun 𝔽 (Fin n)).toMatrix ⇑(basisOfLinearIndependentOfCardEqF …
    -/
    exact isUnit_det_of_invertible _
    /-
      🎉 no goals
    -/
  left_inv := fun _ ↦ Units.ext (ext fun _ _ ↦ rfl)
                  /-
                    𝔽 : Type u_1
                    inst✝¹ : Field 𝔽
                    inst✝ : Fintype 𝔽
                    n : Nat
                    hn : LT.lt 0 n
                    ⊢ Function.RightInverse (fun M => Matrix.GeneralLinearGroup.mk'' (Matrix.trans …
                  -/
  right_inv := by exact congrFun rfl
                  /-
                    🎉 no goals
                  -/


/-- The cardinal of the general linear group over a finite field. -/
theorem card_GL_field :
    Nat.card (GL (Fin n) 𝔽) = ∏ i : (Fin n), (q ^ n - q ^ ( i : ℕ )) := by
  /-
    𝔽 : Type u_1
    inst✝¹ : Field 𝔽
    inst✝ : Fintype 𝔽
    n : Nat
    ⊢ Eq (Nat.card (Matrix.GeneralLinearGroup (Fin n) 𝔽)) (Finset.univ.prod fun i  …
  -/
  rcases Nat.eq_zero_or_pos n with rfl | hn
    /-
      case inl
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      ⊢ Eq (Nat.card (Matrix.GeneralLinearGroup (Fin 0) 𝔽)) (Finset.univ.prod fun i  …
    -/
  · simp [Nat.card_eq_fintype_card]
    /-
      🎉 no goals
    -/
  · rw [Nat.card_congr (equiv_GL_linearindependent n hn), card_linearIndependent,
    Module.finrank_fintype_fun_eq_card, Fintype.card_fin]
    /-
      case inr
      𝔽 : Type u_1
      inst✝¹ : Field 𝔽
      inst✝ : Fintype 𝔽
      n : Nat
      hn : GT.gt n 0
      ⊢ LE.le n (Module.finrank 𝔽 (Fin n → 𝔽))
    -/
    simp only [Module.finrank_fintype_fun_eq_card, Fintype.card_fin, le_refl]
    /-
      🎉 no goals
    -/


