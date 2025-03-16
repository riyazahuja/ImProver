/-- The doubling constant `σₘ[A, B]` of two finsets `A` and `B` in a group is `|A * B| / |A|`.

The notation `σₘ[A, B]` is available in scope `Combinatorics.Additive`. -/
@[to_additive
"The doubling constant `σ[A, B]` of two finsets `A` and `B` in a group is `|A + B| / |A|`.

The notation `σ[A, B]` is available in scope `Combinatorics.Additive`."]
def mulConst (A B : Finset G) : ℚ≥0 := #(A * B) / #A


/-- The difference constant `δₘ[A, B]` of two finsets `A` and `B` in a group is `|A / B| / |A|`.

The notation `δₘ[A, B]` is available in scope `Combinatorics.Additive`. -/
@[to_additive
"The difference constant `σ[A, B]` of two finsets `A` and `B` in a group is `|A - B| / |A|`.

The notation `δ[A, B]` is available in scope `Combinatorics.Additive`."]
def divConst (A B : Finset G) : ℚ≥0 := #(A / B) / #A


/-- The doubling constant `σₘ[A, B]` of two finsets `A` and `B` in a group is `|A * B| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "σₘ[" A ", " B "]" => Finset.mulConst A B


/-- The doubling constant `σₘ[A]` of a finset `A` in a group is `|A * A| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "σₘ[" A "]" => Finset.mulConst A A


/-- The doubling constant `σ[A, B]` of two finsets `A` and `B` in a group is `|A + B| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "σ[" A ", " B "]" => Finset.addConst A B


/-- The doubling constant `σ[A]` of a finset `A` in a group is `|A + A| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "σ[" A "]" => Finset.addConst A A


/-- The difference constant `σₘ[A, B]` of two finsets `A` and `B` in a group is `|A / B| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "δₘ[" A ", " B "]" => Finset.divConst A B


/-- The difference constant `σₘ[A]` of a finset `A` in a group is `|A / A| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "δₘ[" A "]" => Finset.divConst A A


/-- The difference constant `σ[A, B]` of two finsets `A` and `B` in a group is `|A - B| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "δ[" A ", " B "]" => Finset.subConst A B


/-- The difference constant `σ[A]` of a finset `A` in a group is `|A - A| / |A|`. -/
scoped[Combinatorics.Additive] notation3:max "δ[" A "]" => Finset.subConst A A


@[to_additive (attr := simp) addConst_mul_card]
lemma mulConst_mul_card (A B : Finset G) : σₘ[A, B] * #A = #(A * B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq (HMul.hMul (A.mulConst B) ↑A.card) ↑(HMul.hMul A B).card
  -/
  obtain rfl | hA := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      B : Finset G
      ⊢ Eq (HMul.hMul (EmptyCollection.emptyCollection.mulConst B) ↑EmptyCollection. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      A B : Finset G
      hA : A.Nonempty
      ⊢ Eq (HMul.hMul (A.mulConst B) ↑A.card) ↑(HMul.hMul A B).card
    -/
  · exact div_mul_cancel₀ _ (by positivity)
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp) subConst_mul_card]
lemma divConst_mul_card (A B : Finset G) : δₘ[A, B] * #A = #(A / B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq (HMul.hMul (A.divConst B) ↑A.card) ↑(HDiv.hDiv A B).card
  -/
  obtain rfl | hA := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      B : Finset G
      ⊢ Eq (HMul.hMul (EmptyCollection.emptyCollection.divConst B) ↑EmptyCollection. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : DecidableEq G
      A B : Finset G
      hA : A.Nonempty
      ⊢ Eq (HMul.hMul (A.divConst B) ↑A.card) ↑(HDiv.hDiv A B).card
    -/
  · exact div_mul_cancel₀ _ (by positivity)
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp) card_mul_addConst]
lemma card_mul_mulConst (A B : Finset G) : #A * σₘ[A, B] = #(A * B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq (HMul.hMul (↑A.card) (A.mulConst B)) ↑(HMul.hMul A B).card
  -/
  rw [mul_comm, mulConst_mul_card]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) card_mul_subConst]
lemma card_mul_divConst (A B : Finset G) : #A * δₘ[A, B] = #(A / B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq (HMul.hMul (↑A.card) (A.divConst B)) ↑(HDiv.hDiv A B).card
  -/
  rw [mul_comm, divConst_mul_card]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                              /-
                                                                G : Type u_1
                                                                inst✝¹ : Group G
                                                                inst✝ : DecidableEq G
                                                                B : Finset G
                                                                ⊢ Eq (EmptyCollection.emptyCollection.mulConst B) 0
                                                              -/
lemma mulConst_empty_left (B : Finset G) : σₘ[∅, B] = 0 := by simp [mulConst]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[to_additive (attr := simp)]
                                                              /-
                                                                G : Type u_1
                                                                inst✝¹ : Group G
                                                                inst✝ : DecidableEq G
                                                                B : Finset G
                                                                ⊢ Eq (EmptyCollection.emptyCollection.divConst B) 0
                                                              -/
lemma divConst_empty_left (B : Finset G) : δₘ[∅, B] = 0 := by simp [divConst]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[to_additive (attr := simp)]
                                                               /-
                                                                 G : Type u_1
                                                                 inst✝¹ : Group G
                                                                 inst✝ : DecidableEq G
                                                                 A : Finset G
                                                                 ⊢ Eq (A.mulConst EmptyCollection.emptyCollection) 0
                                                               -/
lemma mulConst_empty_right (A : Finset G) : σₘ[A, ∅] = 0 := by simp [mulConst]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive (attr := simp)]
                                                               /-
                                                                 G : Type u_1
                                                                 inst✝¹ : Group G
                                                                 inst✝ : DecidableEq G
                                                                 A : Finset G
                                                                 ⊢ Eq (A.divConst EmptyCollection.emptyCollection) 0
                                                               -/
lemma divConst_empty_right (A : Finset G) : δₘ[A, ∅] = 0 := by simp [divConst]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive (attr := simp)]
lemma mulConst_inv_right (A B : Finset G) : σₘ[A, B⁻¹] = δₘ[A, B] := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq (A.mulConst (Inv.inv B)) (A.divConst B)
  -/
  rw [mulConst, divConst, ← div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma divConst_inv_right (A B : Finset G) : δₘ[A, B⁻¹] = σₘ[A, B] := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq (A.divConst (Inv.inv B)) (A.mulConst B)
  -/
  rw [mulConst, divConst, div_inv_eq_mul]
  /-
    🎉 no goals
  -/


/-- Dense sets have small doubling. -/
@[to_additive addConst_le_inv_dens "Dense sets have small doubling."]
lemma mulConst_le_inv_dens : σₘ[A, B] ≤ A.dens⁻¹ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : DecidableEq G
    A B : Finset G
    inst✝ : Fintype G
    ⊢ LE.le (A.mulConst B) (Inv.inv A.dens)
  -/
  rw [dens, inv_div, mulConst]; gcongr; exact card_le_univ _
                                        /-
                                          🎉 no goals
                                        -/


/-- Dense sets have small difference constant. -/
@[to_additive subConst_le_inv_dens "Dense sets have small difference constant."]
lemma divConst_le_inv_dens : δₘ[A, B] ≤ A.dens⁻¹ := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : DecidableEq G
    A B : Finset G
    inst✝ : Fintype G
    ⊢ LE.le (A.divConst B) (Inv.inv A.dens)
  -/
  rw [dens, inv_div, divConst]; gcongr; exact card_le_univ _
                                        /-
                                          🎉 no goals
                                        -/


lemma cast_addConst (A B : Finset G') : (σ[A, B] : 𝕜) = #(A + B) / #A := by
  /-
    G' : Type u_2
    inst✝³ : AddGroup G'
    inst✝² : DecidableEq G'
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G'
    ⊢ Eq (↑(A.addConst B)) (HDiv.hDiv ↑(HAdd.hAdd A B).card ↑A.card)
  -/
  simp [addConst]
  /-
    🎉 no goals
  -/


lemma cast_subConst (A B : Finset G') : (δ[A, B] : 𝕜) = #(A - B) / #A := by
  /-
    G' : Type u_2
    inst✝³ : AddGroup G'
    inst✝² : DecidableEq G'
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G'
    ⊢ Eq (↑(A.subConst B)) (HDiv.hDiv ↑(HSub.hSub A B).card ↑A.card)
  -/
  simp [subConst]
  /-
    🎉 no goals
  -/


@[to_additive existing]
                                                                            /-
                                                                              G : Type u_1
                                                                              inst✝³ : Group G
                                                                              inst✝² : DecidableEq G
                                                                              𝕜 : Type u_3
                                                                              inst✝¹ : Semifield 𝕜
                                                                              inst✝ : CharZero 𝕜
                                                                              A B : Finset G
                                                                              ⊢ Eq (↑(A.mulConst B)) (HDiv.hDiv ↑(HMul.hMul A B).card ↑A.card)
                                                                            -/
lemma cast_mulConst (A B : Finset G) : (σₘ[A, B] : 𝕜) = #(A * B) / #A := by simp [mulConst]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[to_additive existing]
                                                                            /-
                                                                              G : Type u_1
                                                                              inst✝³ : Group G
                                                                              inst✝² : DecidableEq G
                                                                              𝕜 : Type u_3
                                                                              inst✝¹ : Semifield 𝕜
                                                                              inst✝ : CharZero 𝕜
                                                                              A B : Finset G
                                                                              ⊢ Eq (↑(A.divConst B)) (HDiv.hDiv ↑(HDiv.hDiv A B).card ↑A.card)
                                                                            -/
lemma cast_divConst (A B : Finset G) : (δₘ[A, B] : 𝕜) = #(A / B) / #A := by simp [divConst]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


lemma cast_addConst_mul_card (A B : Finset G') : (σ[A, B] * #A : 𝕜) = #(A + B) := by
  /-
    G' : Type u_2
    inst✝³ : AddGroup G'
    inst✝² : DecidableEq G'
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G'
    ⊢ Eq (HMul.hMul ↑(A.addConst B) ↑A.card) ↑(HAdd.hAdd A B).card
  -/
  norm_cast; exact addConst_mul_card _ _
             /-
               🎉 no goals
             -/


lemma cast_subConst_mul_card (A B : Finset G') : (δ[A, B] * #A : 𝕜) = #(A - B) := by
  /-
    G' : Type u_2
    inst✝³ : AddGroup G'
    inst✝² : DecidableEq G'
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G'
    ⊢ Eq (HMul.hMul ↑(A.subConst B) ↑A.card) ↑(HSub.hSub A B).card
  -/
  norm_cast; exact subConst_mul_card _ _
             /-
               🎉 no goals
             -/


lemma card_mul_cast_addConst (A B : Finset G') : (#A * σ[A, B] : 𝕜) = #(A + B) := by
  /-
    G' : Type u_2
    inst✝³ : AddGroup G'
    inst✝² : DecidableEq G'
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G'
    ⊢ Eq (HMul.hMul ↑A.card ↑(A.addConst B)) ↑(HAdd.hAdd A B).card
  -/
  norm_cast; exact card_mul_addConst _ _
             /-
               🎉 no goals
             -/


lemma card_mul_cast_subConst (A B : Finset G') : (#A * δ[A, B] : 𝕜) = #(A - B) := by
  /-
    G' : Type u_2
    inst✝³ : AddGroup G'
    inst✝² : DecidableEq G'
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G'
    ⊢ Eq (HMul.hMul ↑A.card ↑(A.subConst B)) ↑(HSub.hSub A B).card
  -/
  norm_cast; exact card_mul_subConst _ _
             /-
               🎉 no goals
             -/


@[to_additive (attr := simp) existing cast_addConst_mul_card]
lemma cast_mulConst_mul_card (A B : Finset G) : (σₘ[A, B] * #A : 𝕜) = #(A * B) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : DecidableEq G
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G
    ⊢ Eq (HMul.hMul ↑(A.mulConst B) ↑A.card) ↑(HMul.hMul A B).card
  -/
  norm_cast; exact mulConst_mul_card _ _
             /-
               🎉 no goals
             -/


@[to_additive (attr := simp) existing cast_subConst_mul_card]
lemma cast_divConst_mul_card (A B : Finset G) : (δₘ[A, B] * #A : 𝕜) = #(A / B) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : DecidableEq G
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G
    ⊢ Eq (HMul.hMul ↑(A.divConst B) ↑A.card) ↑(HDiv.hDiv A B).card
  -/
  norm_cast; exact divConst_mul_card _ _
             /-
               🎉 no goals
             -/


@[to_additive (attr := simp) existing card_mul_cast_addConst]
lemma card_mul_cast_mulConst (A B : Finset G) : (#A * σₘ[A, B] : 𝕜) = #(A * B) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : DecidableEq G
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G
    ⊢ Eq (HMul.hMul ↑A.card ↑(A.mulConst B)) ↑(HMul.hMul A B).card
  -/
  norm_cast; exact card_mul_mulConst _ _
             /-
               🎉 no goals
             -/


@[to_additive (attr := simp) existing card_mul_cast_subConst]
lemma card_mul_cast_divConst (A B : Finset G) : (#A * δₘ[A, B] : 𝕜) = #(A / B) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : DecidableEq G
    𝕜 : Type u_3
    inst✝¹ : Semifield 𝕜
    inst✝ : CharZero 𝕜
    A B : Finset G
    ⊢ Eq (HMul.hMul ↑A.card ↑(A.divConst B)) ↑(HDiv.hDiv A B).card
  -/
  norm_cast; exact card_mul_divConst _ _
             /-
               🎉 no goals
             -/


@[to_additive (attr := simp)]
lemma mulConst_inv_left (A B : Finset G) : σₘ[A⁻¹, B] = δₘ[A, B] := by
  /-
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq ((Inv.inv A).mulConst B) (A.divConst B)
  -/
  rw [mulConst, divConst, card_inv, ← card_inv, mul_inv_rev, inv_inv, inv_mul_eq_div]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma divConst_inv_left (A B : Finset G) : δₘ[A⁻¹, B] = σₘ[A, B] := by
  /-
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : DecidableEq G
    A B : Finset G
    ⊢ Eq ((Inv.inv A).divConst B) (A.mulConst B)
  -/
  rw [mulConst, divConst, card_inv, ← card_inv, inv_div, div_inv_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


/-- If `A` has small difference, then it has small doubling, with the constant squared.

This is a consequence of the Ruzsa triangle inequality. -/
@[to_additive
"If `A` has small difference, then it has small doubling, with the constant squared.

This is a consequence of the Ruzsa triangle inequality."]
lemma mulConst_le_divConst_sq : σₘ[A] ≤ δₘ[A] ^ 2 := by
  /-
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : DecidableEq G
    A : Finset G
    ⊢ LE.le (A.mulConst A) (HPow.hPow (A.divConst A) 2)
  -/
  obtain rfl | hA' := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : CommGroup G
      inst✝ : DecidableEq G
      ⊢ LE.le (EmptyCollection.emptyCollection.mulConst EmptyCollection.emptyCollect …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : DecidableEq G
    A : Finset G
    hA' : A.Nonempty
    ⊢ LE.le (A.mulConst A) (HPow.hPow (A.divConst A) 2)
  -/
  refine le_of_mul_le_mul_right ?_ (by positivity : (0 : ℚ≥0) < #A * #A)
  calc
    _ = #(A * A) * (#A : ℚ≥0) := by rw [← mul_assoc, mulConst_mul_card]
    _ ≤ #(A / A) * #(A / A) := by norm_cast; exact ruzsa_triangle_inequality_mul_div_div ..
    _ = _ := by rw [← divConst_mul_card]; ring


/-- If `A` has small doubling, then it has small difference, with the constant squared.

This is a consequence of the Ruzsa triangle inequality. -/
@[to_additive
"If `A` has small doubling, then it has small difference, with the constant squared.

This is a consequence of the Ruzsa triangle inequality."]
lemma divConst_le_mulConst_sq : δₘ[A] ≤ σₘ[A] ^ 2 := by
  /-
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : DecidableEq G
    A : Finset G
    ⊢ LE.le (A.divConst A) (HPow.hPow (A.mulConst A) 2)
  -/
  obtain rfl | hA' := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : CommGroup G
      inst✝ : DecidableEq G
      ⊢ LE.le (EmptyCollection.emptyCollection.divConst EmptyCollection.emptyCollect …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : DecidableEq G
    A : Finset G
    hA' : A.Nonempty
    ⊢ LE.le (A.divConst A) (HPow.hPow (A.mulConst A) 2)
  -/
  refine le_of_mul_le_mul_right ?_ (by positivity : (0 : ℚ≥0) < #A * #A)
  calc
    _ = #(A / A) * (#A : ℚ≥0) := by rw [← mul_assoc, divConst_mul_card]
    _ ≤ #(A * A) * #(A * A) := by norm_cast; exact ruzsa_triangle_inequality_div_mul_mul ..
    _ = _ := by rw [← mulConst_mul_card]; ring


