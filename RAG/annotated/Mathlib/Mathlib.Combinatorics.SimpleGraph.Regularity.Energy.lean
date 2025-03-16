/-- The energy of a partition, also known as index. Auxiliary quantity for Szemerédi's regularity
lemma. -/
def energy : ℚ :=
  ((∑ uv ∈ P.parts.offDiag, G.edgeDensity uv.1 uv.2 ^ 2) : ℚ) / (#P.parts : ℚ) ^ 2


theorem energy_nonneg : 0 ≤ P.energy G := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    s : Finset α
    P : Finpartition s
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ⊢ LE.le 0 (P.energy G)
  -/
  exact div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) <| sq_nonneg _
  /-
    🎉 no goals
  -/


theorem energy_le_one : P.energy G ≤ 1 :=
  div_le_of_le_mul₀ (sq_nonneg _) zero_le_one <|
    calc
      ∑ uv ∈ P.parts.offDiag, G.edgeDensity uv.1 uv.2 ^ 2 ≤ #P.parts.offDiag • (1 : ℚ) :=
        sum_le_card_nsmul _ _ 1 fun _ _ =>
          (sq_le_one_iff₀ <| G.edgeDensity_nonneg _ _).2 <| G.edgeDensity_le_one _ _
      _ = #P.parts.offDiag := Nat.smul_one_eq_cast _
      _ ≤ _ := by
        /-
          α : Type u_1
          inst✝¹ : DecidableEq α
          s : Finset α
          P : Finpartition s
          G : SimpleGraph α
          inst✝ : DecidableRel G.Adj
          ⊢ LE.le (↑P.parts.offDiag.card) (HMul.hMul 1 (HPow.hPow (↑P.parts.card) 2))
        -/
        rw [offDiag_card, one_mul]
        /-
          α : Type u_1
          inst✝¹ : DecidableEq α
          s : Finset α
          P : Finpartition s
          G : SimpleGraph α
          inst✝ : DecidableRel G.Adj
          ⊢ LE.le (↑(HSub.hSub (HMul.hMul P.parts.card P.parts.card) P.parts.card)) (HPo …
        -/
        norm_cast
        /-
          α : Type u_1
          inst✝¹ : DecidableEq α
          s : Finset α
          P : Finpartition s
          G : SimpleGraph α
          inst✝ : DecidableRel G.Adj
          ⊢ LE.le (HSub.hSub (HMul.hMul P.parts.card P.parts.card) P.parts.card) (HPow.h …
        -/
        rw [sq]
        /-
          α : Type u_1
          inst✝¹ : DecidableEq α
          s : Finset α
          P : Finpartition s
          G : SimpleGraph α
          inst✝ : DecidableRel G.Adj
          ⊢ LE.le (HSub.hSub (HMul.hMul P.parts.card P.parts.card) P.parts.card) (HMul.h …
        -/
        exact tsub_le_self
        /-
          🎉 no goals
        -/


@[simp, norm_cast]
theorem coe_energy {𝕜 : Type*} [LinearOrderedField 𝕜] : (P.energy G : 𝕜) =
    (∑ uv ∈ P.parts.offDiag, (G.edgeDensity uv.1 uv.2 : 𝕜) ^ 2) / (#P.parts : 𝕜) ^ 2 := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    s : Finset α
    P : Finpartition s
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    𝕜 : Type u_2
    inst✝ : LinearOrderedField 𝕜
    ⊢ Eq (↑(P.energy G)) (HDiv.hDiv (P.parts.offDiag.sum fun uv => HPow.hPow (↑(G. …
  -/
  rw [energy]; norm_cast
               /-
                 🎉 no goals
               -/


