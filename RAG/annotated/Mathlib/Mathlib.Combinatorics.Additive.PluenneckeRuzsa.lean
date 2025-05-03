/-- **Ruzsa's triangle inequality**. Division version. -/
@[to_additive "**Ruzsa's triangle inequality**. Subtraction version."]
theorem ruzsa_triangle_inequality_div_div_div (A B C : Finset G) :
    #(A / C) * #B ≤ #(A / B) * #(C / B) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HDiv.hDiv A C).card B.card) (HMul.hMul (HDiv.hDiv A B).car …
  -/
  rw [← card_product (A / B), ← mul_one #((A / B) ×ˢ (C / B))]
  refine card_mul_le_card_mul (fun b (a, c) ↦ a / c = b) (fun x hx ↦ ?_)
    fun x _ ↦ card_le_one_iff.2 fun hu hv ↦
      ((mem_bipartiteBelow _).1 hu).2.symm.trans ?_
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A B C : Finset G
      x : G
      hx : Membership.mem (HDiv.hDiv A C) x
      ⊢ LE.le B.card (Finset.bipartiteAbove (fun b x => Finset.ruzsa_triangle_inequa …
    -/
  · obtain ⟨a, ha, c, hc, rfl⟩ := mem_div.1 hx
    /-
      case refine_1.intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A B C : Finset G
      a : G
      ha : Membership.mem A a
      c : G
      hc : Membership.mem C c
      hx : Membership.mem (HDiv.hDiv A C) (HDiv.hDiv a c)
      ⊢ LE.le B.card (Finset.bipartiteAbove (fun b x => Finset.ruzsa_triangle_inequa …
    -/
    refine card_le_card_of_injOn (fun b ↦ (a / b, c / b)) (fun b hb ↦ ?_) fun b₁ _ b₂ _ h ↦ ?_
      /-
        case refine_1.intro.intro.intro.intro.refine_1
        G : Type u_1
        inst✝¹ : DecidableEq G
        inst✝ : Group G
        A B C : Finset G
        a : G
        ha : Membership.mem A a
        c : G
        hc : Membership.mem C c
        hx : Membership.mem (HDiv.hDiv A C) (HDiv.hDiv a c)
        b : G
        hb : Membership.mem B b
        ⊢ Membership.mem (Finset.bipartiteAbove (fun b x => Finset.ruzsa_triangle_ineq …
      -/
    · rw [mem_bipartiteAbove]
      /-
        case refine_1.intro.intro.intro.intro.refine_1
        G : Type u_1
        inst✝¹ : DecidableEq G
        inst✝ : Group G
        A B C : Finset G
        a : G
        ha : Membership.mem A a
        c : G
        hc : Membership.mem C c
        hx : Membership.mem (HDiv.hDiv A C) (HDiv.hDiv a c)
        b : G
        hb : Membership.mem B b
        ⊢ And (Membership.mem (SProd.sprod (HDiv.hDiv A B) (HDiv.hDiv C B)) ((fun b => …
      -/
      exact ⟨mk_mem_product (div_mem_div ha hb) (div_mem_div hc hb), div_div_div_cancel_right ..⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.refine_2
        G : Type u_1
        inst✝¹ : DecidableEq G
        inst✝ : Group G
        A B C : Finset G
        a : G
        ha : Membership.mem A a
        c : G
        hc : Membership.mem C c
        hx : Membership.mem (HDiv.hDiv A C) (HDiv.hDiv a c)
        b₁ : G
        x✝¹ : Membership.mem (↑B) b₁
        b₂ : G
        x✝ : Membership.mem (↑B) b₂
        h : Eq ((fun b => { fst := HDiv.hDiv a b, snd := HDiv.hDiv c b }) b₁) ((fun b  …
        ⊢ Eq b₁ b₂
      -/
    · exact div_right_injective (Prod.ext_iff.1 h).1
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A B C : Finset G
      x : Prod G G
      x✝ : Membership.mem (SProd.sprod (HDiv.hDiv A B) (HDiv.hDiv C B)) x
      a✝ b✝ : G
      hu : Membership.mem (Finset.bipartiteBelow (fun b x => Finset.ruzsa_triangle_i …
      hv : Membership.mem (Finset.bipartiteBelow (fun b x => Finset.ruzsa_triangle_i …
      ⊢ Eq (HDiv.hDiv x.1 x.2) b✝
    -/
  · exact ((mem_bipartiteBelow _).1 hv).2
    /-
      🎉 no goals
    -/


/-- **Ruzsa's triangle inequality**. Mulinv-mulinv-mulinv version. -/
@[to_additive "**Ruzsa's triangle inequality**. Addneg-addneg-addneg version."]
theorem ruzsa_triangle_inequality_mulInv_mulInv_mulInv (A B C : Finset G) :
    #(A * C⁻¹) * #B ≤ #(A * B⁻¹) * #(C * B⁻¹) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A (Inv.inv C)).card B.card) (HMul.hMul (HMul.hMu …
  -/
  simpa [div_eq_mul_inv] using ruzsa_triangle_inequality_div_div_div A B C
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Invmul-invmul-invmul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Negadd-negadd-negadd version."]
theorem ruzsa_triangle_inequality_invMul_invMul_invMul (A B C : Finset G) :
    #B * #(A⁻¹ * C) ≤ #(B⁻¹ * A) * #(B⁻¹ * C) := by
  simpa [mul_comm, div_eq_mul_inv, ← map_op_mul, ← map_op_inv] using
    ruzsa_triangle_inequality_div_div_div (G := Gᵐᵒᵖ) (C.map opEquiv.toEmbedding)
      (B.map opEquiv.toEmbedding) (A.map opEquiv.toEmbedding)



/-- **Ruzsa's triangle inequality**. Div-mul-mul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Sub-add-add version."]
theorem ruzsa_triangle_inequality_div_mul_mul (A B C : Finset G) :
    #(A / C) * #B ≤ #(A * B) * #(C * B) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HDiv.hDiv A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  simpa using ruzsa_triangle_inequality_div_div_div A B⁻¹ C
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Mulinv-mul-mul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Addneg-add-add version."]
theorem ruzsa_triangle_inequality_mulInv_mul_mul (A B C : Finset G) :
    #(A * C⁻¹) * #B ≤ #(A * B) * #(C * B) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A (Inv.inv C)).card B.card) (HMul.hMul (HMul.hMu …
  -/
  simpa using ruzsa_triangle_inequality_mulInv_mulInv_mulInv A B⁻¹ C
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Invmul-mul-mul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Negadd-add-add version."]
theorem ruzsa_triangle_inequality_invMul_mul_mul (A B C : Finset G) :
    #B * #(A⁻¹ * C) ≤ #(B * A) * #(B * C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul B.card (HMul.hMul (Inv.inv A) C).card) (HMul.hMul (HMul.hMu …
  -/
  simpa using ruzsa_triangle_inequality_invMul_invMul_invMul A B⁻¹ C
  /-
    🎉 no goals
  -/



/-- **Ruzsa's triangle inequality**. Mul-div-mul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Add-sub-add version."]
theorem ruzsa_triangle_inequality_mul_div_mul (A B C : Finset G) :
    #B * #(A * C) ≤ #(B / A) * #(B * C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul B.card (HMul.hMul A C).card) (HMul.hMul (HDiv.hDiv B A).car …
  -/
  simpa [div_eq_mul_inv] using ruzsa_triangle_inequality_invMul_mul_mul A⁻¹ B C
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Mul-mulinv-mul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Add-addneg-add version."]
theorem ruzsa_triangle_inequality_mul_mulInv_mul (A B C : Finset G) :
    #B * #(A * C) ≤ #(B * A⁻¹) * #(B * C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul B.card (HMul.hMul A C).card) (HMul.hMul (HMul.hMul B (Inv.i …
  -/
  simpa [div_eq_mul_inv] using ruzsa_triangle_inequality_mul_div_mul A B C
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Mul-mul-invmul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Add-add-negadd version."]
theorem ruzsa_triangle_inequality_mul_mul_invMul (A B C : Finset G) :
    #(A * C) * #B ≤ #(A * B) * #(C⁻¹ * B) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  simpa using ruzsa_triangle_inequality_mulInv_mul_mul A B C⁻¹
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pluennecke_petridis_inequality_mul (C : Finset G)
    (hA : ∀ A' ⊆ A, #(A * B) * #A' ≤ #(A' * B) * #A) :
    #(A * B * C) * #A ≤ #(A * B) * #(A * C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A.card) (HMul.hMul (HMul …
  -/
  induction' C using Finset.induction_on with x C _ ih
    /-
      case empty
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : CommGroup G
      A B : Finset G
      hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
      ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) EmptyCollection.emptyCollection) …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case insert
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
    x : G
    C : Finset G
    a✝ : Not (Membership.mem C x)
    ih : LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A.card) (HMul.hMul (H …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) (Insert.insert x C)).card A.card …
  -/
  set A' := A ∩ (A * C / {x}) with hA'
  /-
    case insert
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
    x : G
    C : Finset G
    a✝ : Not (Membership.mem C x)
    ih : LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A.card) (HMul.hMul (H …
    A' : Finset G := Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton …
    hA' : Eq A' (Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton x)))
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) (Insert.insert x C)).card A.card …
  -/
  set C' := insert x C with hC'
  have h₀ : A' * {x} = A * {x} ∩ (A * C) := by
    rw [hA', inter_mul_singleton, (isUnit_singleton x).div_mul_cancel]
  have h₁ : A * B * C' = A * B * C ∪ (A * B * {x}) \ (A' * B * {x}) := by
    rw [hC', insert_eq, union_comm, mul_union]
    refine (sup_sdiff_eq_sup ?_).symm
    rw [mul_right_comm, mul_right_comm A, h₀]
    exact mul_subset_mul_right inter_subset_right
  have h₂ : A' * B * {x} ⊆ A * B * {x} :=
    mul_subset_mul_right (mul_subset_mul_right inter_subset_left)
  have h₃ : #(A * B * C') ≤ #(A * B * C) + #(A * B) - #(A' * B) := by
    rw [h₁]
    refine (card_union_le _ _).trans_eq ?_
    rw [card_sdiff h₂, ← add_tsub_assoc_of_le (card_le_card h₂), card_mul_singleton,
      card_mul_singleton]
  /-
    case insert
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
    x : G
    C : Finset G
    a✝ : Not (Membership.mem C x)
    ih : LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A.card) (HMul.hMul (H …
    A' : Finset G := Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton …
    hA' : Eq A' (Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton x)))
    C' : Finset G := Insert.insert x C
    hC' : Eq C' (Insert.insert x C)
    h₀ : Eq (HMul.hMul A' (Singleton.singleton x)) (Inter.inter (HMul.hMul A (Sing …
    h₁ : Eq (HMul.hMul (HMul.hMul A B) C') (Union.union (HMul.hMul (HMul.hMul A B) …
    h₂ : HasSubset.Subset (HMul.hMul (HMul.hMul A' B) (Singleton.singleton x)) (HM …
    h₃ : LE.le (HMul.hMul (HMul.hMul A B) C').card (HSub.hSub (HAdd.hAdd (HMul.hMu …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C').card A.card) (HMul.hMul (HMu …
  -/
  refine (mul_le_mul_right' h₃ _).trans ?_
  /-
    case insert
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
    x : G
    C : Finset G
    a✝ : Not (Membership.mem C x)
    ih : LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A.card) (HMul.hMul (H …
    A' : Finset G := Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton …
    hA' : Eq A' (Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton x)))
    C' : Finset G := Insert.insert x C
    hC' : Eq C' (Insert.insert x C)
    h₀ : Eq (HMul.hMul A' (Singleton.singleton x)) (Inter.inter (HMul.hMul A (Sing …
    h₁ : Eq (HMul.hMul (HMul.hMul A B) C') (Union.union (HMul.hMul (HMul.hMul A B) …
    h₂ : HasSubset.Subset (HMul.hMul (HMul.hMul A' B) (Singleton.singleton x)) (HM …
    h₃ : LE.le (HMul.hMul (HMul.hMul A B) C').card (HSub.hSub (HAdd.hAdd (HMul.hMu …
    ⊢ LE.le (HMul.hMul (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul A B) C).card (H …
  -/
  rw [tsub_mul, add_mul]
  /-
    case insert
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hA : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A  …
    x : G
    C : Finset G
    a✝ : Not (Membership.mem C x)
    ih : LE.le (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A.card) (HMul.hMul (H …
    A' : Finset G := Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton …
    hA' : Eq A' (Inter.inter A (HDiv.hDiv (HMul.hMul A C) (Singleton.singleton x)))
    C' : Finset G := Insert.insert x C
    hC' : Eq C' (Insert.insert x C)
    h₀ : Eq (HMul.hMul A' (Singleton.singleton x)) (Inter.inter (HMul.hMul A (Sing …
    h₁ : Eq (HMul.hMul (HMul.hMul A B) C') (Union.union (HMul.hMul (HMul.hMul A B) …
    h₂ : HasSubset.Subset (HMul.hMul (HMul.hMul A' B) (Singleton.singleton x)) (HM …
    h₃ : LE.le (HMul.hMul (HMul.hMul A B) C').card (HSub.hSub (HAdd.hAdd (HMul.hMu …
    ⊢ LE.le (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul A B) C).card A. …
  -/
  refine (tsub_le_tsub (add_le_add_right ih _) <| hA _ inter_subset_left).trans_eq ?_
  rw [← mul_add, ← mul_tsub, ← hA', hC', insert_eq, mul_union, ← card_mul_singleton A x, ←
    card_mul_singleton A' x, add_comm #_, h₀,
    eq_tsub_of_add_eq (card_union_add_card_inter _ _)]


@[to_additive]
private theorem mul_aux (hA : A.Nonempty) (hAB : A ⊆ B)
    (h : ∀ A' ∈ B.powerset.erase ∅, (#(A * C) : ℚ≥0) / #A ≤ #(A' * C) / #A') :
    ∀ A' ⊆ A, #(A * C) * #A' ≤ #(A' * C) * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hA : A.Nonempty
    hAB : HasSubset.Subset A B
    h : ∀ (A' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.emptyC …
    ⊢ ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A C). …
  -/
  rintro A' hAA'
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hA : A.Nonempty
    hAB : HasSubset.Subset A B
    h : ∀ (A' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.emptyC …
    A' : Finset G
    hAA' : HasSubset.Subset A' A
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card A'.card) (HMul.hMul (HMul.hMul A' C).c …
  -/
  obtain rfl | hA' := A'.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : CommGroup G
      A B C : Finset G
      hA : A.Nonempty
      hAB : HasSubset.Subset A B
      h : ∀ (A' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.emptyC …
      hAA' : HasSubset.Subset EmptyCollection.emptyCollection A
      ⊢ LE.le (HMul.hMul (HMul.hMul A C).card EmptyCollection.emptyCollection.card)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hA : A.Nonempty
    hAB : HasSubset.Subset A B
    h : ∀ (A' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.emptyC …
    A' : Finset G
    hAA' : HasSubset.Subset A' A
    hA' : A'.Nonempty
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card A'.card) (HMul.hMul (HMul.hMul A' C).c …
  -/
  have hA₀ : (0 : ℚ≥0) < #A := cast_pos.2 hA.card_pos
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hA : A.Nonempty
    hAB : HasSubset.Subset A B
    h : ∀ (A' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.emptyC …
    A' : Finset G
    hAA' : HasSubset.Subset A' A
    hA' : A'.Nonempty
    hA₀ : LT.lt 0 ↑A.card
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card A'.card) (HMul.hMul (HMul.hMul A' C).c …
  -/
  have hA₀' : (0 : ℚ≥0) < #A' := cast_pos.2 hA'.card_pos
  exact mod_cast
    (div_le_div_iff₀ hA₀ hA₀').1
      (h _ <| mem_erase_of_ne_of_mem hA'.ne_empty <| mem_powerset.2 <| hAA'.trans hAB)


/-- **Ruzsa's triangle inequality**. Multiplication version. -/
@[to_additive "**Ruzsa's triangle inequality**. Addition version."]
theorem ruzsa_triangle_inequality_mul_mul_mul (A B C : Finset G) :
    #(A * C) * #B ≤ #(A * B) * #(B * C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  obtain rfl | hB := B.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : CommGroup G
      A C : Finset G
      ⊢ LE.le (HMul.hMul (HMul.hMul A C).card EmptyCollection.emptyCollection.card)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  have hB' : B ∈ B.powerset.erase ∅ := mem_erase_of_ne_of_mem hB.ne_empty (mem_powerset_self _)
  obtain ⟨U, hU, hUA⟩ :=
    exists_min_image (B.powerset.erase ∅) (fun U ↦ #(U * A) / #U : _ → ℚ≥0) ⟨B, hB'⟩
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) U
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  rw [mem_erase, mem_powerset, ← nonempty_iff_ne_empty] at hU
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : And U.Nonempty (HasSubset.Subset U B)
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  refine cast_le.1 (?_ : (_ : ℚ≥0) ≤ _)
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : And U.Nonempty (HasSubset.Subset U B)
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le ↑(HMul.hMul (HMul.hMul A C).card B.card) ↑(HMul.hMul (HMul.hMul A B).c …
  -/
  push_cast
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : And U.Nonempty (HasSubset.Subset U B)
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le (HMul.hMul ↑(HMul.hMul A C).card ↑B.card) (HMul.hMul ↑(HMul.hMul A B). …
  -/
  rw [← le_div_iff₀ (cast_pos.2 hB.card_pos), mul_div_right_comm, mul_comm _ B]
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : And U.Nonempty (HasSubset.Subset U B)
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le (↑(HMul.hMul A C).card) (HMul.hMul (HDiv.hDiv ↑(HMul.hMul B A).card ↑B …
  -/
  refine (Nat.cast_le.2 <| card_le_card_mul_left hU.1).trans ?_
  refine le_trans ?_
    (mul_le_mul (hUA _ hB') (cast_le.2 <| card_le_card <| mul_subset_mul_right hU.2)
      (zero_le _) (zero_le _))
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : And U.Nonempty (HasSubset.Subset U B)
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le (↑(HMul.hMul U (HMul.hMul A C)).card) (HMul.hMul (HDiv.hDiv ↑(HMul.hMu …
  -/
  #adaptation_note /-- 2024-11-01 `le_div_iff₀` is synthesizing wrong `GroupWithZero` without `@` -/
  rw [← mul_div_right_comm, ← mul_assoc,
    @le_div_iff₀ _ (_) _ _ _ _ _ _ _ (cast_pos.2 hU.1.card_pos)]
  /-
    case inr.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    hB : B.Nonempty
    hB' : Membership.mem (B.powerset.erase EmptyCollection.emptyCollection) B
    U : Finset G
    hU : And U.Nonempty (HasSubset.Subset U B)
    hUA : ∀ (x' : Finset G), Membership.mem (B.powerset.erase EmptyCollection.empt …
    ⊢ LE.le (HMul.hMul ↑(HMul.hMul (HMul.hMul U A) C).card ↑U.card) (HMul.hMul ↑(H …
  -/
  exact mod_cast pluennecke_petridis_inequality_mul C (mul_aux hU.1 hU.2 hUA)
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Mul-div-div version. -/
@[to_additive "**Ruzsa's triangle inequality**. Add-sub-sub version."]
theorem ruzsa_triangle_inequality_mul_div_div (A B C : Finset G) :
    #(A * C) * #B ≤ #(A / B) * #(B / C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card B.card) (HMul.hMul (HDiv.hDiv A B).car …
  -/
  rw [div_eq_mul_inv, ← card_inv B, ← card_inv (B / C), inv_div', div_inv_eq_mul]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A C).card (Inv.inv B).card) (HMul.hMul (HMul.hMu …
  -/
  exact ruzsa_triangle_inequality_mul_mul_mul _ _ _
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Div-mul-div version. -/
@[to_additive "**Ruzsa's triangle inequality**. Sub-add-sub version."]
theorem ruzsa_triangle_inequality_div_mul_div (A B C : Finset G) :
    #(A / C) * #B ≤ #(A * B) * #(B / C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HDiv.hDiv A C).card B.card) (HMul.hMul (HMul.hMul A B).car …
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A (Inv.inv C)).card B.card) (HMul.hMul (HMul.hMu …
  -/
  exact ruzsa_triangle_inequality_mul_mul_mul _ _ _
  /-
    🎉 no goals
  -/


/-- **Ruzsa's triangle inequality**. Div-div-mul version. -/
@[to_additive "**Ruzsa's triangle inequality**. Sub-sub-add version."]
theorem card_div_mul_le_card_div_mul_card_mul (A B C : Finset G) :
    #(A / C) * #B ≤ #(A / B) * #(B * C) := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HDiv.hDiv A C).card B.card) (HMul.hMul (HDiv.hDiv A B).car …
  -/
  rw [← div_inv_eq_mul, div_eq_mul_inv]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B C : Finset G
    ⊢ LE.le (HMul.hMul (HMul.hMul A (Inv.inv C)).card B.card) (HMul.hMul (HDiv.hDi …
  -/
  exact ruzsa_triangle_inequality_mul_div_div _ _ _
  /-
    🎉 no goals
  -/

-- Auxiliary lemma towards the Plünnecke-Ruzsa inequality

@[to_additive]
private lemma card_mul_pow_le (hAB : ∀ A' ⊆ A, #(A * B) * #A' ≤ #(A' * B) * #A) (n : ℕ) :
    #(A * B ^ n) ≤ (#(A * B) / #A : ℚ≥0) ^ n * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hAB : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A …
    n : Nat
    ⊢ LE.le (↑(HMul.hMul A (HPow.hPow B n)).card) (HMul.hMul (HPow.hPow (HDiv.hDiv …
  -/
  obtain rfl | hA := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : CommGroup G
      B : Finset G
      n : Nat
      hAB : ∀ (A' : Finset G), HasSubset.Subset A' EmptyCollection.emptyCollection → …
      ⊢ LE.le (↑(HMul.hMul EmptyCollection.emptyCollection (HPow.hPow B n)).card) (H …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hAB : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A …
    n : Nat
    hA : A.Nonempty
    ⊢ LE.le (↑(HMul.hMul A (HPow.hPow B n)).card) (HMul.hMul (HPow.hPow (HDiv.hDiv …
  -/
  induction' n with n ih
    /-
      case inr.zero
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : CommGroup G
      A B : Finset G
      hAB : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A …
      hA : A.Nonempty
      ⊢ LE.le (↑(HMul.hMul A (HPow.hPow B 0)).card) (HMul.hMul (HPow.hPow (HDiv.hDiv …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.succ
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A B : Finset G
    hAB : ∀ (A' : Finset G), HasSubset.Subset A' A → LE.le (HMul.hMul (HMul.hMul A …
    hA : A.Nonempty
    n : Nat
    ih : LE.le (↑(HMul.hMul A (HPow.hPow B n)).card) (HMul.hMul (HPow.hPow (HDiv.h …
    ⊢ LE.le (↑(HMul.hMul A (HPow.hPow B (HAdd.hAdd n 1))).card) (HMul.hMul (HPow.h …
  -/
  refine le_of_mul_le_mul_right ?_ (by positivity : (0 : ℚ≥0) < #A)
  calc
    ((#(A * B ^ (n + 1))) * #A : ℚ≥0)
      = #(A * B * B ^ n) * #A := by rw [_root_.pow_succ', ← mul_assoc]
    _ ≤ #(A * B) * #(A * B ^ n) := mod_cast pluennecke_petridis_inequality_mul _ hAB
    _ ≤ #(A * B) * ((#(A * B) / #A) ^ n * #A) := by gcongr
    _ = (#(A * B) / #A) ^ (n + 1) * #A * #A := by field_simp; ring


/-- The **Plünnecke-Ruzsa inequality**. Multiplication version. Note that this is genuinely harder
than the division version because we cannot use a double counting argument. -/
@[to_additive "The **Plünnecke-Ruzsa inequality**. Addition version. Note that this is genuinely
harder than the subtraction version because we cannot use a double counting argument."]
theorem pluennecke_ruzsa_inequality_pow_div_pow_mul (hA : A.Nonempty) (B : Finset G) (m n : ℕ) :
    #(B ^ m / B ^ n) ≤ (#(A * B) / #A : ℚ≥0) ^ (m + n) * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    m n : Nat
    ⊢ LE.le (↑(HDiv.hDiv (HPow.hPow B m) (HPow.hPow B n)).card) (HMul.hMul (HPow.h …
  -/
  have hA' : A ∈ A.powerset.erase ∅ := mem_erase_of_ne_of_mem hA.ne_empty (mem_powerset_self _)
  obtain ⟨C, hC, hCmin⟩ :=
    exists_min_image (A.powerset.erase ∅) (fun C ↦ #(C * B) / #C : _ → ℚ≥0) ⟨A, hA'⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    m n : Nat
    hA' : Membership.mem (A.powerset.erase EmptyCollection.emptyCollection) A
    C : Finset G
    hC : Membership.mem (A.powerset.erase EmptyCollection.emptyCollection) C
    hCmin : ∀ (x' : Finset G), Membership.mem (A.powerset.erase EmptyCollection.em …
    ⊢ LE.le (↑(HDiv.hDiv (HPow.hPow B m) (HPow.hPow B n)).card) (HMul.hMul (HPow.h …
  -/
  rw [mem_erase, mem_powerset, ← nonempty_iff_ne_empty] at hC
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    m n : Nat
    hA' : Membership.mem (A.powerset.erase EmptyCollection.emptyCollection) A
    C : Finset G
    hC : And C.Nonempty (HasSubset.Subset C A)
    hCmin : ∀ (x' : Finset G), Membership.mem (A.powerset.erase EmptyCollection.em …
    ⊢ LE.le (↑(HDiv.hDiv (HPow.hPow B m) (HPow.hPow B n)).card) (HMul.hMul (HPow.h …
  -/
  obtain ⟨hC, hCA⟩ := hC
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    m n : Nat
    hA' : Membership.mem (A.powerset.erase EmptyCollection.emptyCollection) A
    C : Finset G
    hCmin : ∀ (x' : Finset G), Membership.mem (A.powerset.erase EmptyCollection.em …
    hC : C.Nonempty
    hCA : HasSubset.Subset C A
    ⊢ LE.le (↑(HDiv.hDiv (HPow.hPow B m) (HPow.hPow B n)).card) (HMul.hMul (HPow.h …
  -/
  refine le_of_mul_le_mul_right ?_ (by positivity : (0 : ℚ≥0) < #C)
  calc
    (#(B ^ m / B ^ n) * #C : ℚ≥0)
      ≤ #(B ^ m * C) * #(B ^ n * C) := mod_cast ruzsa_triangle_inequality_div_mul_mul ..
    _ = #(C * B ^ m) * #(C * B ^ n) := by simp_rw [mul_comm]
    _ ≤ ((#(C * B) / #C) ^ m * #C) * ((#(C * B) / #C : ℚ≥0) ^ n * #C) := by
      gcongr <;> exact card_mul_pow_le (mul_aux hC hCA hCmin) _
    _ = (#(C * B) / #C) ^ (m + n) * #C * #C := by ring
    _ ≤ (#(A * B) / #A) ^ (m + n) * #A * #C := by gcongr (?_ ^ _) * #?_ * _; exact hCmin _ hA'


/-- The **Plünnecke-Ruzsa inequality**. Division version. -/
@[to_additive "The **Plünnecke-Ruzsa inequality**. Subtraction version."]
theorem pluennecke_ruzsa_inequality_pow_div_pow_div (hA : A.Nonempty) (B : Finset G) (m n : ℕ) :
    #(B ^ m / B ^ n) ≤ (#(A / B) / #A : ℚ≥0) ^ (m + n) * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    m n : Nat
    ⊢ LE.le (↑(HDiv.hDiv (HPow.hPow B m) (HPow.hPow B n)).card) (HMul.hMul (HPow.h …
  -/
  rw [← card_inv, inv_div', ← inv_pow, ← inv_pow, div_eq_mul_inv A]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    m n : Nat
    ⊢ LE.le (↑(HDiv.hDiv (HPow.hPow (Inv.inv B) m) (HPow.hPow (Inv.inv B) n)).card …
  -/
  exact pluennecke_ruzsa_inequality_pow_div_pow_mul hA _ _ _
  /-
    🎉 no goals
  -/


/-- Special case of the **Plünnecke-Ruzsa inequality**. Multiplication version. -/
@[to_additive "Special case of the **Plünnecke-Ruzsa inequality**. Addition version."]
theorem pluennecke_ruzsa_inequality_pow_mul (hA : A.Nonempty) (B : Finset G) (n : ℕ) :
    #(B ^ n) ≤ (#(A * B) / #A : ℚ≥0) ^ n * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    n : Nat
    ⊢ LE.le (↑(HPow.hPow B n).card) (HMul.hMul (HPow.hPow (HDiv.hDiv ↑(HMul.hMul A …
  -/
  simpa only [_root_.pow_zero, div_one] using pluennecke_ruzsa_inequality_pow_div_pow_mul hA _ _ 0
  /-
    🎉 no goals
  -/


/-- Special case of the **Plünnecke-Ruzsa inequality**. Division version. -/
@[to_additive "Special case of the **Plünnecke-Ruzsa inequality**. Subtraction version."]
theorem pluennecke_ruzsa_inequality_pow_div (hA : A.Nonempty) (B : Finset G) (n : ℕ) :
    #(B ^ n) ≤ (#(A / B) / #A : ℚ≥0) ^ n * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : CommGroup G
    A : Finset G
    hA : A.Nonempty
    B : Finset G
    n : Nat
    ⊢ LE.le (↑(HPow.hPow B n).card) (HMul.hMul (HPow.hPow (HDiv.hDiv ↑(HDiv.hDiv A …
  -/
  simpa only [_root_.pow_zero, div_one] using pluennecke_ruzsa_inequality_pow_div_pow_div hA _ _ 0
  /-
    🎉 no goals
  -/


