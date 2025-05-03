local notation3 "m" => (card α / stepBound #P.parts : ℕ)


/-- The portion of `SzemerediRegularity.increment` which partitions `U`. -/
noncomputable def chunk : Finpartition U :=
  if hUcard : #U = m * 4 ^ #P.parts + (card α / #P.parts - m * 4 ^ #P.parts) then
    (atomise U <| P.nonuniformWitnesses G ε U).equitabilise <| card_aux₁ hUcard
  else (atomise U <| P.nonuniformWitnesses G ε U).equitabilise <| card_aux₂ hP hU hUcard

-- `hP` and `hU` are used to get that `U` has size
-- `m * 4 ^ #P.parts + a or m * 4 ^ #P.parts + a + 1`

/-- The portion of `SzemerediRegularity.chunk` which is contained in the witness of non-uniformity
of `U` and `V`. -/
noncomputable def star (V : Finset α) : Finset (Finset α) :=
  {A ∈ (chunk hP G ε hU).parts | A ⊆ G.nonuniformWitness ε U V}


theorem biUnion_star_subset_nonuniformWitness :
    (star hP G ε hU V).biUnion id ⊆ G.nonuniformWitness ε U V :=
  biUnion_subset_iff_forall_subset.2 fun _ hA => (mem_filter.1 hA).2


theorem star_subset_chunk : star hP G ε hU V ⊆ (chunk hP G ε hU).parts :=
  filter_subset _ _


private theorem card_nonuniformWitness_sdiff_biUnion_star (hV : V ∈ P.parts) (hUV : U ≠ V)
    (h₂ : ¬G.IsUniform ε U V) :
    #(G.nonuniformWitness ε U V \ (star hP G ε hU V).biUnion id) ≤ 2 ^ (#P.parts - 1) * m := by
  have hX : G.nonuniformWitness ε U V ∈ P.nonuniformWitnesses G ε U :=
    nonuniformWitness_mem_nonuniformWitnesses h₂ hV hUV
  have q : G.nonuniformWitness ε U V \ (star hP G ε hU V).biUnion id ⊆
      {B ∈ (atomise U <| P.nonuniformWitnesses G ε U).parts |
        B ⊆ G.nonuniformWitness ε U V ∧ B.Nonempty}.biUnion
        fun B => B \ {A ∈ (chunk hP G ε hU).parts | A ⊆ B}.biUnion id := by
    intro x hx
    rw [← biUnion_filter_atomise hX (G.nonuniformWitness_subset h₂), star, mem_sdiff,
      mem_biUnion] at hx
    simp only [not_exists, mem_biUnion, and_imp, exists_prop, mem_filter,
      not_and, mem_sdiff, id, mem_sdiff] at hx ⊢
    obtain ⟨⟨B, hB₁, hB₂⟩, hx⟩ := hx
    exact ⟨B, hB₁, hB₂, fun A hA AB => hx A hA <| AB.trans hB₁.2.1⟩
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    h₂ : Not (G.IsUniform ε U V)
    hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
    q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
    ⊢ LE.le (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegularity.star hP …
  -/
  apply (card_le_card q).trans (card_biUnion_le.trans _)
  trans ∑ B ∈ (atomise U <| P.nonuniformWitnesses G ε U).parts with
    B ⊆ G.nonuniformWitness ε U V ∧ B.Nonempty, m
  · suffices ∀ B ∈ (atomise U <| P.nonuniformWitnesses G ε U).parts,
        #(B \ {A ∈ (chunk hP G ε hU).parts | A ⊆ B}.biUnion id) ≤ m by
      exact sum_le_sum fun B hB => this B <| filter_subset _ _ hB
    /-
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      V : Finset α
      hV : Membership.mem P.parts V
      hUV : Ne U V
      h₂ : Not (G.IsUniform ε U V)
      hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
      q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
      ⊢ ∀ (B : Finset α), Membership.mem (Finpartition.atomise U (P.nonuniformWitnes …
    -/
    intro B hB
    /-
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      V : Finset α
      hV : Membership.mem P.parts V
      hUV : Ne U V
      h₂ : Not (G.IsUniform ε U V)
      hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
      q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
      B : Finset α
      hB : Membership.mem (Finpartition.atomise U (P.nonuniformWitnesses G ε U)).par …
      ⊢ LE.le (SDiff.sdiff B ((Finset.filter (fun A => HasSubset.Subset A B) (Szemer …
    -/
    unfold chunk
    /-
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      V : Finset α
      hV : Membership.mem P.parts V
      hUV : Ne U V
      h₂ : Not (G.IsUniform ε U V)
      hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
      q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
      B : Finset α
      hB : Membership.mem (Finpartition.atomise U (P.nonuniformWitnesses G ε U)).par …
      ⊢ LE.le (SDiff.sdiff B ((Finset.filter (fun A => HasSubset.Subset A B) (dite ( …
    -/
    split_ifs with h₁
      /-
        case pos
        α : Type u_1
        inst✝² : Fintype α
        inst✝¹ : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝ : DecidableRel G.Adj
        ε : Real
        U : Finset α
        hU : Membership.mem P.parts U
        V : Finset α
        hV : Membership.mem P.parts V
        hUV : Ne U V
        h₂ : Not (G.IsUniform ε U V)
        hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
        q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
        B : Finset α
        hB : Membership.mem (Finpartition.atomise U (P.nonuniformWitnesses G ε U)).par …
        h₁ : Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (SzemerediReg …
        ⊢ LE.le (SDiff.sdiff B ((Finset.filter (fun A => HasSubset.Subset A B) (Finpar …
      -/
    · convert card_parts_equitabilise_subset_le _ (card_aux₁ h₁) hB
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝² : Fintype α
        inst✝¹ : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝ : DecidableRel G.Adj
        ε : Real
        U : Finset α
        hU : Membership.mem P.parts U
        V : Finset α
        hV : Membership.mem P.parts V
        hUV : Ne U V
        h₂ : Not (G.IsUniform ε U V)
        hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
        q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
        B : Finset α
        hB : Membership.mem (Finpartition.atomise U (P.nonuniformWitnesses G ε U)).par …
        h₁ : Not (Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (Szemere …
        ⊢ LE.le (SDiff.sdiff B ((Finset.filter (fun A => HasSubset.Subset A B) (Finpar …
      -/
    · convert card_parts_equitabilise_subset_le _ (card_aux₂ hP hU h₁) hB
      /-
        🎉 no goals
      -/
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    h₂ : Not (G.IsUniform ε U V)
    hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
    q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
    ⊢ LE.le ((Finset.filter (fun B => And (HasSubset.Subset B (G.nonuniformWitness …
  -/
  rw [sum_const]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    h₂ : Not (G.IsUniform ε U V)
    hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
    q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
    ⊢ LE.le (HSMul.hSMul (Finset.filter (fun B => And (HasSubset.Subset B (G.nonun …
  -/
  refine mul_le_mul_right' ?_ _
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    h₂ : Not (G.IsUniform ε U V)
    hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
    q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
    ⊢ LE.le (Finset.filter (fun B => And (HasSubset.Subset B (G.nonuniformWitness  …
  -/
  have t := card_filter_atomise_le_two_pow (s := U) hX
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    h₂ : Not (G.IsUniform ε U V)
    hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
    q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
    t : LE.le (Finset.filter (fun u => And (HasSubset.Subset u (G.nonuniformWitnes …
    ⊢ LE.le (Finset.filter (fun B => And (HasSubset.Subset B (G.nonuniformWitness  …
  -/
  refine t.trans (pow_right_mono₀ (by norm_num) <| tsub_le_tsub_right ?_ _)
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    h₂ : Not (G.IsUniform ε U V)
    hX : Membership.mem (P.nonuniformWitnesses G ε U) (G.nonuniformWitness ε U V)
    q : HasSubset.Subset (SDiff.sdiff (G.nonuniformWitness ε U V) ((SzemerediRegul …
    t : LE.le (Finset.filter (fun u => And (HasSubset.Subset u (G.nonuniformWitnes …
    ⊢ LE.le (P.nonuniformWitnesses G ε U).card P.parts.card
  -/
  exact card_image_le.trans (card_le_card <| filter_subset _ _)
  /-
    🎉 no goals
  -/


private theorem one_sub_eps_mul_card_nonuniformWitness_le_card_star (hV : V ∈ P.parts)
    (hUV : U ≠ V) (hunif : ¬G.IsUniform ε U V) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    (hε₁ : ε ≤ 1) :
    (1 - ε / 10) * #(G.nonuniformWitness ε U V) ≤ #((star hP G ε hU V).biUnion id) := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    V : Finset α
    hV : Membership.mem P.parts V
    hUV : Ne U V
    hunif : Not (G.IsUniform ε U V)
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    ⊢ LE.le (HMul.hMul (HSub.hSub 1 (HDiv.hDiv ε 10)) ↑(G.nonuniformWitness ε U V) …
  -/
  have hP₁ : 0 < #P.parts := Finset.card_pos.2 ⟨_, hU⟩
  have : (↑2 ^ #P.parts : ℝ) * m / (#U * ε) ≤ ε / 10 := by
    rw [← div_div, div_le_iff₀']
    swap
    · sz_positivity
    refine le_of_mul_le_mul_left ?_ (pow_pos zero_lt_two #P.parts)
    calc
      ↑2 ^ #P.parts * ((↑2 ^ #P.parts * m : ℝ) / #U) =
          ((2 : ℝ) * 2) ^ #P.parts * m / #U := by
        rw [mul_pow, ← mul_div_assoc, mul_assoc]
      _ = ↑4 ^ #P.parts * m / #U := by norm_num
      _ ≤ 1 := div_le_one_of_le₀ (pow_mul_m_le_card_part hP hU) (cast_nonneg _)
      _ ≤ ↑2 ^ #P.parts * ε ^ 2 / 10 := by
        refine (one_le_sq_iff₀ <| by positivity).1 ?_
        rw [div_pow, mul_pow, pow_right_comm, ← pow_mul ε,
          one_le_div (sq_pos_of_ne_zero <| by norm_num)]
        calc
          (↑10 ^ 2) = 100 := by norm_num
          _ ≤ ↑4 ^ #P.parts * ε ^ 5 := hPε
          _ ≤ ↑4 ^ #P.parts * ε ^ 4 :=
            (mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (by sz_positivity) hε₁ <| le_succ _)
              (by positivity))
          _ = (↑2 ^ 2) ^ #P.parts * ε ^ (2 * 2) := by norm_num
      _ = ↑2 ^ #P.parts * (ε * (ε / 10)) := by rw [mul_div_assoc, sq, mul_div_assoc]
  calc
    (↑1 - ε / 10) * #(G.nonuniformWitness ε U V) ≤
        (↑1 - ↑2 ^ #P.parts * m / (#U * ε)) * #(G.nonuniformWitness ε U V) :=
      mul_le_mul_of_nonneg_right (sub_le_sub_left this _) (cast_nonneg _)
    _ = #(G.nonuniformWitness ε U V) -
        ↑2 ^ #P.parts * m / (#U * ε) * #(G.nonuniformWitness ε U V) := by
      rw [sub_mul, one_mul]
    _ ≤ #(G.nonuniformWitness ε U V) - ↑2 ^ (#P.parts - 1) * m := by
      refine sub_le_sub_left ?_ _
      have : (2 : ℝ) ^ #P.parts = ↑2 ^ (#P.parts - 1) * 2 := by
        rw [← _root_.pow_succ, tsub_add_cancel_of_le (succ_le_iff.2 hP₁)]
      rw [← mul_div_right_comm, this, mul_right_comm _ (2 : ℝ), mul_assoc, le_div_iff₀]
      · refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        exact (G.le_card_nonuniformWitness hunif).trans
          (le_mul_of_one_le_left (cast_nonneg _) one_le_two)
      have := Finset.card_pos.mpr (P.nonempty_of_mem_parts hU)
      sz_positivity
    _ ≤ #((star hP G ε hU V).biUnion id) := by
      rw [sub_le_comm, ←
        cast_sub (card_le_card <| biUnion_star_subset_nonuniformWitness hP G ε hU V), ←
        card_sdiff (biUnion_star_subset_nonuniformWitness hP G ε hU V)]
      exact mod_cast card_nonuniformWitness_sdiff_biUnion_star hV hUV hunif


theorem card_chunk (hm : m ≠ 0) : #(chunk hP G ε hU).parts = 4 ^ #P.parts := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    hm : Ne (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.car …
    ⊢ Eq (SzemerediRegularity.chunk hP G ε hU).parts.card (HPow.hPow 4 P.parts.card)
  -/
  unfold chunk
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    hm : Ne (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.car …
    ⊢ Eq (dite (Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (Szeme …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      hm : Ne (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.car …
      h✝ : Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (SzemerediReg …
      ⊢ Eq (Finpartition.equitabilise ⋯).parts.card (HPow.hPow 4 P.parts.card)
    -/
  · rw [card_parts_equitabilise _ _ hm, tsub_add_cancel_of_le]
    /-
      case pos
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      hm : Ne (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.car …
      h✝ : Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (SzemerediReg …
      ⊢ LE.le (HSub.hSub (HDiv.hDiv (Fintype.card α) P.parts.card) (HMul.hMul (HDiv. …
    -/
    exact le_of_lt a_add_one_le_four_pow_parts_card
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      hm : Ne (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.car …
      h✝ : Not (Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype.card α) (Szemere …
      ⊢ Eq (Finpartition.equitabilise ⋯).parts.card (HPow.hPow 4 P.parts.card)
    -/
  · rw [card_parts_equitabilise _ _ hm, tsub_add_cancel_of_le a_add_one_le_four_pow_parts_card]
    /-
      🎉 no goals
    -/


theorem card_eq_of_mem_parts_chunk (hs : s ∈ (chunk hP G ε hU).parts) :
    #s = m ∨ #s = m + 1 := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    s : Finset α
    hs : Membership.mem (SzemerediRegularity.chunk hP G ε hU).parts s
    ⊢ Or (Eq s.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.p …
  -/
  unfold chunk at hs
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    s : Finset α
    hs : Membership.mem (dite (Eq U.card (HAdd.hAdd (HMul.hMul (HDiv.hDiv (Fintype …
    ⊢ Or (Eq s.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.p …
  -/
                      /-
                        🎉 no goals
                      -/
  split_ifs at hs <;> exact card_eq_of_mem_parts_equitabilise hs
                      /-
                        🎉 no goals
                      -/


theorem m_le_card_of_mem_chunk_parts (hs : s ∈ (chunk hP G ε hU).parts) : m ≤ #s :=
                                                            /-
                                                              α : Type u_1
                                                              inst✝² : Fintype α
                                                              inst✝¹ : DecidableEq α
                                                              P : Finpartition Finset.univ
                                                              hP : P.IsEquipartition
                                                              G : SimpleGraph α
                                                              inst✝ : DecidableRel G.Adj
                                                              ε : Real
                                                              U : Finset α
                                                              hU : Membership.mem P.parts U
                                                              s : Finset α
                                                              hs : Membership.mem (SzemerediRegularity.chunk hP G ε hU).parts s
                                                              i : Eq s.card (HAdd.hAdd (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.step …
                                                              ⊢ LE.le (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.car …
                                                            -/
  (card_eq_of_mem_parts_chunk hs).elim ge_of_eq fun i => by simp [i]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem card_le_m_add_one_of_mem_chunk_parts (hs : s ∈ (chunk hP G ε hU).parts) : #s ≤ m + 1 :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝² : Fintype α
                                                      inst✝¹ : DecidableEq α
                                                      P : Finpartition Finset.univ
                                                      hP : P.IsEquipartition
                                                      G : SimpleGraph α
                                                      inst✝ : DecidableRel G.Adj
                                                      ε : Real
                                                      U : Finset α
                                                      hU : Membership.mem P.parts U
                                                      s : Finset α
                                                      hs : Membership.mem (SzemerediRegularity.chunk hP G ε hU).parts s
                                                      i : Eq s.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.par …
                                                      ⊢ LE.le s.card (HAdd.hAdd (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.ste …
                                                    -/
  (card_eq_of_mem_parts_chunk hs).elim (fun i => by simp [i]) fun i => i.le
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem card_biUnion_star_le_m_add_one_card_star_mul :
    (#((star hP G ε hU V).biUnion id) : ℝ) ≤ #(star hP G ε hU V) * (m + 1) :=
  mod_cast card_biUnion_le_card_mul _ _ _ fun _ hs =>
    card_le_m_add_one_of_mem_chunk_parts <| star_subset_chunk hs


private theorem le_sum_card_subset_chunk_parts (h𝒜 : 𝒜 ⊆ (chunk hP G ε hU).parts) (hs : s ∈ 𝒜) :
    (#𝒜 : ℝ) * #s * (m / (m + 1)) ≤ #(𝒜.sup id) := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    𝒜 : Finset (Finset α)
    s : Finset α
    h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
    hs : Membership.mem 𝒜 s
    ⊢ LE.le (HMul.hMul (HMul.hMul ↑𝒜.card ↑s.card) (HDiv.hDiv (↑(HDiv.hDiv (Fintyp …
  -/
  rw [mul_div_assoc', div_le_iff₀ coe_m_add_one_pos, mul_right_comm]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    𝒜 : Finset (Finset α)
    s : Finset α
    h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
    hs : Membership.mem 𝒜 s
    ⊢ LE.le (HMul.hMul (HMul.hMul ↑𝒜.card ↑(HDiv.hDiv (Fintype.card α) (SzemerediR …
  -/
  refine mul_le_mul ?_ ?_ (cast_nonneg _) (cast_nonneg _)
    /-
      case refine_1
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      ⊢ LE.le (HMul.hMul ↑𝒜.card ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.s …
    -/
  · rw [← (ofSubset _ h𝒜 rfl).sum_card_parts, ofSubset_parts, ← cast_mul, cast_le]
    /-
      case refine_1
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      ⊢ LE.le (HMul.hMul 𝒜.card (HDiv.hDiv (Fintype.card α) (SzemerediRegularity.ste …
    -/
    exact card_nsmul_le_sum _ _ _ fun x hx => m_le_card_of_mem_chunk_parts <| h𝒜 hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      ⊢ LE.le (↑s.card) (HAdd.hAdd (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularit …
    -/
  · exact mod_cast card_le_m_add_one_of_mem_chunk_parts (h𝒜 hs)
    /-
      🎉 no goals
    -/


private theorem sum_card_subset_chunk_parts_le (m_pos : (0 : ℝ) < m)
    (h𝒜 : 𝒜 ⊆ (chunk hP G ε hU).parts) (hs : s ∈ 𝒜) :
    (#(𝒜.sup id) : ℝ) ≤ #𝒜 * #s * ((m + 1) / m) := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    𝒜 : Finset (Finset α)
    s : Finset α
    m_pos : LT.lt 0 ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P. …
    h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
    hs : Membership.mem 𝒜 s
    ⊢ LE.le (↑(𝒜.sup id).card) (HMul.hMul (HMul.hMul ↑𝒜.card ↑s.card) (HDiv.hDiv ( …
  -/
  rw [sup_eq_biUnion, mul_div_assoc', le_div_iff₀ m_pos, mul_right_comm]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U : Finset α
    hU : Membership.mem P.parts U
    𝒜 : Finset (Finset α)
    s : Finset α
    m_pos : LT.lt 0 ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P. …
    h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
    hs : Membership.mem 𝒜 s
    ⊢ LE.le (HMul.hMul ↑(𝒜.biUnion id).card ↑(HDiv.hDiv (Fintype.card α) (Szemered …
  -/
  refine mul_le_mul ?_ ?_ (cast_nonneg _) (by positivity)
    /-
      case refine_1
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      m_pos : LT.lt 0 ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P. …
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      ⊢ LE.le (↑(𝒜.biUnion id).card) (HMul.hMul (↑𝒜.card) (HAdd.hAdd (↑(HDiv.hDiv (F …
    -/
  · norm_cast
    /-
      case refine_1
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      m_pos : LT.lt 0 ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P. …
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      ⊢ LE.le (𝒜.biUnion id).card (HMul.hMul 𝒜.card (HAdd.hAdd (HDiv.hDiv (Fintype.c …
    -/
    refine card_biUnion_le_card_mul _ _ _ fun x hx => ?_
    /-
      case refine_1
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      m_pos : LT.lt 0 ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P. …
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      x : Finset α
      hx : Membership.mem 𝒜 x
      ⊢ LE.le (id x).card (HAdd.hAdd (HDiv.hDiv (Fintype.card α) (SzemerediRegularit …
    -/
    apply card_le_m_add_one_of_mem_chunk_parts (h𝒜 hx)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      U : Finset α
      hU : Membership.mem P.parts U
      𝒜 : Finset (Finset α)
      s : Finset α
      m_pos : LT.lt 0 ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P. …
      h𝒜 : HasSubset.Subset 𝒜 (SzemerediRegularity.chunk hP G ε hU).parts
      hs : Membership.mem 𝒜 s
      ⊢ LE.le ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBound P.parts.ca …
    -/
  · exact mod_cast m_le_card_of_mem_chunk_parts (h𝒜 hs)
    /-
      🎉 no goals
    -/


private theorem one_sub_le_m_div_m_add_one_sq [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5) :
    ↑1 - ε ^ 5 / ↑50 ≤ (m / (m + 1 : ℝ)) ^ 2 := by
  have : (m : ℝ) / (m + 1) = 1 - 1 / (m + 1) := by
    rw [one_sub_div coe_m_add_one_pos.ne', add_sub_cancel_right]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    this : Eq (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (HPow.hPow ε 5) 50)) (HPow.hPow (HDiv.hDiv (↑( …
  -/
  rw [this, sub_sq, one_pow, mul_one]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    this : Eq (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
    ⊢ LE.le (HSub.hSub 1 (HDiv.hDiv (HPow.hPow ε 5) 50)) (HAdd.hAdd (HSub.hSub 1 ( …
  -/
  refine le_trans ?_ (le_add_of_nonneg_right <| sq_nonneg _)
  rw [sub_le_sub_iff_left, ← le_div_iff₀' (show (0 : ℝ) < 2 by norm_num), div_div,
    one_div_le coe_m_add_one_pos, one_div_div]
    /-
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      ε : Real
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      this : Eq (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
      ⊢ LE.le (HDiv.hDiv (HMul.hMul 50 2) (HPow.hPow ε 5)) (HAdd.hAdd (↑(HDiv.hDiv ( …
    -/
  · refine le_trans ?_ (le_add_of_nonneg_right zero_le_one)
    /-
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      ε : Real
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      this : Eq (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
      ⊢ LE.le (HDiv.hDiv (HMul.hMul 50 2) (HPow.hPow ε 5)) ↑(HDiv.hDiv (Fintype.card …
    -/
    norm_num
    /-
      α : Type u_1
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      P : Finpartition Finset.univ
      ε : Real
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      this : Eq (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
      ⊢ LE.le (HDiv.hDiv 100 (HPow.hPow ε 5)) ↑(HDiv.hDiv (Fintype.card α) (Szemered …
    -/
    apply hundred_div_ε_pow_five_le_m hPα hPε
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    this : Eq (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (SzemerediRegularity.stepBo …
    ⊢ LT.lt 0 (HDiv.hDiv (HPow.hPow ε 5) (HMul.hMul 50 2))
  -/
  sz_positivity
  /-
    🎉 no goals
  -/


private theorem m_add_one_div_m_le_one_add [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5) (hε₁ : ε ≤ 1) :
    ((m + 1 : ℝ) / m) ^ 2 ≤ ↑1 + ε ^ 5 / 49 := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    ⊢ LE.le (HPow.hPow (HDiv.hDiv (HAdd.hAdd (↑(HDiv.hDiv (Fintype.card α) (Szemer …
  -/
  have : 0 ≤ ε := by sz_positivity
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    this : LE.le 0 ε
    ⊢ LE.le (HPow.hPow (HDiv.hDiv (HAdd.hAdd (↑(HDiv.hDiv (Fintype.card α) (Szemer …
  -/
  rw [same_add_div (by sz_positivity)]
  calc
    _ ≤ (1 + ε ^ 5 / 100) ^ 2 := by
      gcongr (1 + ?_) ^ 2
      rw [← one_div_div (100 : ℝ)]
      exact one_div_le_one_div_of_le (by sz_positivity) (hundred_div_ε_pow_five_le_m hPα hPε)
    _ = 1 + ε ^ 5 * (50⁻¹ + ε ^ 5 / 10000) := by ring
    _ ≤ 1 + ε ^ 5 * (50⁻¹ + 1 ^ 5 / 10000) := by gcongr
    _ ≤ 1 + ε ^ 5 * 49⁻¹ := by gcongr; norm_num
    _ = 1 + ε ^ 5 / 49 := by rw [div_eq_mul_inv]


private theorem density_sub_eps_le_sum_density_div_card [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    {hU : U ∈ P.parts} {hV : V ∈ P.parts} {A B : Finset (Finset α)}
    (hA : A ⊆ (chunk hP G ε hU).parts) (hB : B ⊆ (chunk hP G ε hV).parts) :
    (G.edgeDensity (A.biUnion id) (B.biUnion id)) - ε ^ 5 / 50 ≤
    (∑ ab ∈ A.product B, (G.edgeDensity ab.1 ab.2 : ℝ)) / (#A * #B) := by
  have : ↑(G.edgeDensity (A.biUnion id) (B.biUnion id)) - ε ^ 5 / ↑50 ≤
      (↑1 - ε ^ 5 / 50) * G.edgeDensity (A.biUnion id) (B.biUnion id) := by
    rw [sub_mul, one_mul, sub_le_sub_iff_left]
    refine mul_le_of_le_one_right (by sz_positivity) ?_
    exact mod_cast G.edgeDensity_le_one _ _
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    ⊢ LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv.hDiv …
  -/
  refine this.trans ?_
  conv_rhs => -- Porting note: LHS and RHS need separate treatment to get the desired form
    simp only [SimpleGraph.edgeDensity_def, sum_div, Rat.cast_div, div_div]
  conv_lhs =>
    rw [SimpleGraph.edgeDensity_def, SimpleGraph.interedges, ← sup_eq_biUnion, ← sup_eq_biUnion,
      Rel.card_interedges_finpartition _ (ofSubset _ hA rfl) (ofSubset _ hB rfl), ofSubset_parts,
      ofSubset_parts]
    simp only [cast_sum, sum_div, mul_sum, Rat.cast_sum, Rat.cast_div,
      mul_div_left_comm ((1 : ℝ) - _)]
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    ⊢ LE.le ((SProd.sprod A B).sum fun x => HMul.hMul (↑↑(Rel.interedges G.Adj x.1 …
  -/
  push_cast
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    ⊢ LE.le ((SProd.sprod A B).sum fun x => HMul.hMul (↑(Rel.interedges G.Adj x.1  …
  -/
  apply sum_le_sum
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    ⊢ ∀ (i : Prod (Finset α) (Finset α)), Membership.mem (SProd.sprod A B) i → LE. …
  -/
  simp only [and_imp, Prod.forall, mem_product]
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    ⊢ ∀ (a b : Finset α), Membership.mem A a → Membership.mem B b → LE.le (HMul.hM …
  -/
  rintro x y hx hy
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    x y : Finset α
    hx : Membership.mem A x
    hy : Membership.mem B y
    ⊢ LE.le (HMul.hMul (↑(Rel.interedges G.Adj x y).card) (HDiv.hDiv (HSub.hSub 1  …
  -/
  rw [mul_mul_mul_comm, mul_comm (#x : ℝ), mul_comm (#y : ℝ), le_div_iff₀, mul_assoc]
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (HMul.hMul (↑(Rel.interedges G.Adj x y).card) (HMul.hMul (HDiv.hDiv (H …
    -/
  · refine mul_le_of_le_one_right (cast_nonneg _) ?_
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (HMul.hMul (HDiv.hDiv (HSub.hSub 1 (HDiv.hDiv (HPow.hPow ε 5) 50)) (HM …
    -/
    rw [div_mul_eq_mul_div, ← mul_assoc, mul_assoc]
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (HDiv.hDiv (HMul.hMul (HSub.hSub 1 (HDiv.hDiv (HPow.hPow ε 5) 50)) (HM …
    -/
    refine div_le_one_of_le₀ ?_ (by positivity)
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (HMul.hMul (HSub.hSub 1 (HDiv.hDiv (HPow.hPow ε 5) 50)) (HMul.hMul (HM …
    -/
    refine (mul_le_mul_of_nonneg_right (one_sub_le_m_div_m_add_one_sq hPα hPε) ?_).trans ?_
      /-
        case h.refine_1
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ LE.le 0 (HMul.hMul (HMul.hMul ↑A.card ↑x.card) (HMul.hMul ↑B.card ↑y.card))
      -/
    · exact mod_cast _root_.zero_le _
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (HMul.hMul (HPow.hPow (HDiv.hDiv (↑(HDiv.hDiv (Fintype.card α) (Szemer …
    -/
    rw [sq, mul_mul_mul_comm, mul_comm ((m : ℝ) / _), mul_comm ((m : ℝ) / _)]
    /-
      case h.refine_2
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul ↑A.card ↑x.card) (HDiv.hDiv (↑(HDiv.h …
    -/
    refine mul_le_mul ?_ ?_ ?_ (cast_nonneg _)
      /-
        case h.refine_2.refine_1
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ LE.le (HMul.hMul (HMul.hMul ↑A.card ↑x.card) (HDiv.hDiv (↑(HDiv.hDiv (Fintyp …
      -/
    · apply le_sum_card_subset_chunk_parts hA hx
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.refine_2
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ LE.le (HMul.hMul (HMul.hMul ↑B.card ↑y.card) (HDiv.hDiv (↑(HDiv.hDiv (Fintyp …
      -/
    · apply le_sum_card_subset_chunk_parts hB hy
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.refine_3
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ LE.le 0 (HMul.hMul (HMul.hMul ↑B.card ↑y.card) (HDiv.hDiv (↑(HDiv.hDiv (Fint …
      -/
    · positivity
      /-
        🎉 no goals
      -/
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    x y : Finset α
    hx : Membership.mem A x
    hy : Membership.mem B y
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul ↑A.card ↑x.card) (HMul.hMul ↑B.card ↑y.card))
  -/
  refine mul_pos (mul_pos ?_ ?_) (mul_pos ?_ ?_) <;> rw [cast_pos, Finset.card_pos]
  /-
    case h.refine_1
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv …
    x y : Finset α
    hx : Membership.mem A x
    hy : Membership.mem B y
    ⊢ A.Nonempty
  -/
  exacts [⟨_, hx⟩, nonempty_of_mem_parts _ (hA hx), ⟨_, hy⟩, nonempty_of_mem_parts _ (hB hy)]
  /-
    🎉 no goals
  -/


private theorem sum_density_div_card_le_density_add_eps [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    (hε₁ : ε ≤ 1) {hU : U ∈ P.parts} {hV : V ∈ P.parts} {A B : Finset (Finset α)}
    (hA : A ⊆ (chunk hP G ε hU).parts) (hB : B ⊆ (chunk hP G ε hV).parts) :
    (∑ ab ∈ A.product B, G.edgeDensity ab.1 ab.2 : ℝ) / (#A * #B) ≤
    G.edgeDensity (A.biUnion id) (B.biUnion id) + ε ^ 5 / 49 := by
  have : (↑1 + ε ^ 5 / ↑49) * G.edgeDensity (A.biUnion id) (B.biUnion id) ≤
      G.edgeDensity (A.biUnion id) (B.biUnion id) + ε ^ 5 / 49 := by
    rw [add_mul, one_mul, add_le_add_iff_left]
    refine mul_le_of_le_one_right (by sz_positivity) ?_
    exact mod_cast G.edgeDensity_le_one _ _
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    ⊢ LE.le (HDiv.hDiv ((A.product B).sum fun ab => ↑(G.edgeDensity ab.1 ab.2)) (H …
  -/
  refine le_trans ?_ this
  conv_lhs => -- Porting note: LHS and RHS need separate treatment to get the desired form
    simp only [SimpleGraph.edgeDensity, edgeDensity, sum_div, Rat.cast_div, div_div]
  conv_rhs =>
    rw [SimpleGraph.edgeDensity, edgeDensity, ← sup_eq_biUnion, ← sup_eq_biUnion,
      Rel.card_interedges_finpartition _ (ofSubset _ hA rfl) (ofSubset _ hB rfl)]
    simp only [cast_sum, mul_sum, sum_div, Rat.cast_sum, Rat.cast_div,
      mul_div_left_comm ((1 : ℝ) + _)]
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    ⊢ LE.le ((A.product B).sum fun x => HDiv.hDiv (↑↑(Rel.interedges G.Adj x.1 x.2 …
  -/
  push_cast
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    ⊢ LE.le ((A.product B).sum fun x => HDiv.hDiv (↑(Rel.interedges G.Adj x.1 x.2) …
  -/
  apply sum_le_sum
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    ⊢ ∀ (i : Prod (Finset α) (Finset α)), Membership.mem (A.product B) i → LE.le ( …
  -/
  simp only [and_imp, Prod.forall, mem_product, show A.product B = A ×ˢ B by rfl]
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    ⊢ ∀ (a b : Finset α), Membership.mem A a → Membership.mem B b → LE.le (HDiv.hD …
  -/
  intro x y hx hy
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    x y : Finset α
    hx : Membership.mem A x
    hy : Membership.mem B y
    ⊢ LE.le (HDiv.hDiv (↑(Rel.interedges G.Adj x y).card) (HMul.hMul (HMul.hMul ↑x …
  -/
  rw [mul_mul_mul_comm, mul_comm (#x : ℝ), mul_comm (#y : ℝ), div_le_iff₀, mul_assoc]
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le (↑(Rel.interedges G.Adj x y).card) (HMul.hMul (↑(Rel.interedges G.Adj  …
    -/
  · refine le_mul_of_one_le_right (cast_nonneg _) ?_
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LE.le 1 (HMul.hMul (HDiv.hDiv (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ( …
    -/
    rw [div_mul_eq_mul_div, one_le_div]
      /-
        case h
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ LE.le (HMul.hMul ↑(A.sup id).card ↑(B.sup id).card) (HMul.hMul (HAdd.hAdd 1  …
      -/
    · refine le_trans ?_ (mul_le_mul_of_nonneg_right (m_add_one_div_m_le_one_add hPα hPε hε₁) ?_)
        /-
          case h.refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          A B : Finset (Finset α)
          hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
          hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
          this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
          x y : Finset α
          hx : Membership.mem A x
          hy : Membership.mem B y
          ⊢ LE.le (HMul.hMul ↑(A.sup id).card ↑(B.sup id).card) (HMul.hMul (HPow.hPow (H …
        -/
      · rw [sq, mul_mul_mul_comm, mul_comm (_ / (m : ℝ)), mul_comm (_ / (m : ℝ))]
        exact mul_le_mul (sum_card_subset_chunk_parts_le (by sz_positivity) hA hx)
          (sum_card_subset_chunk_parts_le (by sz_positivity) hB hy) (by positivity) (by positivity)
        /-
          case h.refine_2
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          A B : Finset (Finset α)
          hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
          hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
          this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
          x y : Finset α
          hx : Membership.mem A x
          hy : Membership.mem B y
          ⊢ LE.le 0 (HMul.hMul (HMul.hMul ↑A.card ↑x.card) (HMul.hMul ↑B.card ↑y.card))
        -/
      · exact mod_cast _root_.zero_le _
        /-
          🎉 no goals
        -/
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LT.lt 0 (HMul.hMul ↑(A.sup id).card ↑(B.sup id).card)
    -/
    rw [← cast_mul, cast_pos]
    /-
      case h
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
      x y : Finset α
      hx : Membership.mem A x
      hy : Membership.mem B y
      ⊢ LT.lt 0 (HMul.hMul (A.sup id).card (B.sup id).card)
    -/
    apply mul_pos <;> rw [Finset.card_pos, sup_eq_biUnion, biUnion_nonempty]
      /-
        case h.ha
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ Exists fun x => And (Membership.mem A x) (id x).Nonempty
      -/
    · exact ⟨_, hx, nonempty_of_mem_parts _ (hA hx)⟩
      /-
        🎉 no goals
      -/
      /-
        case h.hb
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        A B : Finset (Finset α)
        hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
        hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
        this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
        x y : Finset α
        hx : Membership.mem A x
        hy : Membership.mem B y
        ⊢ Exists fun x => And (Membership.mem B x) (id x).Nonempty
      -/
    · exact ⟨_, hy, nonempty_of_mem_parts _ (hB hy)⟩
      /-
        🎉 no goals
      -/
  /-
    case h
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    x y : Finset α
    hx : Membership.mem A x
    hy : Membership.mem B y
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul ↑A.card ↑x.card) (HMul.hMul ↑B.card ↑y.card))
  -/
  refine mul_pos (mul_pos ?_ ?_) (mul_pos ?_ ?_) <;> rw [cast_pos, Finset.card_pos]
  /-
    case h.refine_1
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    this : LE.le (HMul.hMul (HAdd.hAdd 1 (HDiv.hDiv (HPow.hPow ε 5) 49)) ↑(G.edgeD …
    x y : Finset α
    hx : Membership.mem A x
    hy : Membership.mem B y
    ⊢ A.Nonempty
  -/
  exacts [⟨_, hx⟩, nonempty_of_mem_parts _ (hA hx), ⟨_, hy⟩, nonempty_of_mem_parts _ (hB hy)]
  /-
    🎉 no goals
  -/


private theorem average_density_near_total_density [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    (hε₁ : ε ≤ 1) {hU : U ∈ P.parts} {hV : V ∈ P.parts} {A B : Finset (Finset α)}
    (hA : A ⊆ (chunk hP G ε hU).parts) (hB : B ⊆ (chunk hP G ε hV).parts) :
    |(∑ ab ∈ A.product B, G.edgeDensity ab.1 ab.2 : ℝ) / (#A * #B) -
      G.edgeDensity (A.biUnion id) (B.biUnion id)| ≤ ε ^ 5 / 49 := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    ⊢ LE.le (abs (HSub.hSub (HDiv.hDiv ((A.product B).sum fun ab => ↑(G.edgeDensit …
  -/
  rw [abs_sub_le_iff]
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    ⊢ And (LE.le (HSub.hSub (HDiv.hDiv ((A.product B).sum fun ab => ↑(G.edgeDensit …
  -/
  constructor
    /-
      case left
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      ⊢ LE.le (HSub.hSub (HDiv.hDiv ((A.product B).sum fun ab => ↑(G.edgeDensity ab. …
    -/
  · rw [sub_le_iff_le_add']
    /-
      case left
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      A B : Finset (Finset α)
      hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
      hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
      ⊢ LE.le (HDiv.hDiv ((A.product B).sum fun ab => ↑(G.edgeDensity ab.1 ab.2)) (H …
    -/
    exact sum_density_div_card_le_density_add_eps hPα hPε hε₁ hA hB
    /-
      🎉 no goals
    -/
  suffices (G.edgeDensity (A.biUnion id) (B.biUnion id) : ℝ) -
      (∑ ab ∈ A.product B, (G.edgeDensity ab.1 ab.2 : ℝ)) / (#A * #B) ≤ ε ^ 5 / 50 by
    apply this.trans
    gcongr <;> [sz_positivity; norm_num]
  /-
    case right
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    ⊢ LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv.hDiv …
  -/
  rw [sub_le_iff_le_add, ← sub_le_iff_le_add']
  /-
    case right
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    A B : Finset (Finset α)
    hA : HasSubset.Subset A (SzemerediRegularity.chunk hP G ε hU).parts
    hB : HasSubset.Subset B (SzemerediRegularity.chunk hP G ε hV).parts
    ⊢ LE.le (HSub.hSub (↑(G.edgeDensity (A.biUnion id) (B.biUnion id))) (HDiv.hDiv …
  -/
  apply density_sub_eps_le_sum_density_div_card hPα hPε hA hB
  /-
    🎉 no goals
  -/


private theorem edgeDensity_chunk_aux [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    (hU : U ∈ P.parts) (hV : V ∈ P.parts) :
    (G.edgeDensity U V : ℝ) ^ 2 - ε ^ 5 / ↑25 ≤
    ((∑ ab ∈ (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts,
      (G.edgeDensity ab.1 ab.2 : ℝ)) / ↑16 ^ #P.parts) ^ 2 := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
  -/
  obtain hGε | hGε := le_total (G.edgeDensity U V : ℝ) (ε ^ 5 / 50)
    /-
      case inl
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      hGε : LE.le (↑(G.edgeDensity U V)) (HDiv.hDiv (HPow.hPow ε 5) 50)
      ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
    -/
  · refine (sub_nonpos_of_le <| (sq_le ?_ ?_).trans <| hGε.trans ?_).trans (sq_nonneg _)
      /-
        case inl.refine_1
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hGε : LE.le (↑(G.edgeDensity U V)) (HDiv.hDiv (HPow.hPow ε 5) 50)
        ⊢ LE.le 0 ↑(G.edgeDensity U V)
      -/
    · exact mod_cast G.edgeDensity_nonneg _ _
      /-
        🎉 no goals
      -/
      /-
        case inl.refine_2
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hGε : LE.le (↑(G.edgeDensity U V)) (HDiv.hDiv (HPow.hPow ε 5) 50)
        ⊢ LE.le (↑(G.edgeDensity U V)) 1
      -/
    · exact mod_cast G.edgeDensity_le_one _ _
      /-
        🎉 no goals
      -/
      /-
        case inl.refine_3
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hGε : LE.le (↑(G.edgeDensity U V)) (HDiv.hDiv (HPow.hPow ε 5) 50)
        ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 5) 50) (HDiv.hDiv (HPow.hPow ε 5) 25)
      -/
    · exact div_le_div_of_nonneg_left (by sz_positivity) (by norm_num) (by norm_num)
      /-
        🎉 no goals
      -/
  /-
    case inr
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hGε : LE.le (HDiv.hDiv (HPow.hPow ε 5) 50) ↑(G.edgeDensity U V)
    ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
  -/
  rw [← sub_nonneg] at hGε
  /-
    case inr
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hGε : LE.le 0 (HSub.hSub (↑(G.edgeDensity U V)) (HDiv.hDiv (HPow.hPow ε 5) 50))
    ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
  -/
  have : 0 ≤ ε := by sz_positivity
  calc
    _ = G.edgeDensity U V ^ 2 - 1 * ε ^ 5 / 25 + 0 ^ 10 / 2500 := by ring
    _ ≤ G.edgeDensity U V ^ 2 - G.edgeDensity U V * ε ^ 5 / 25 + ε ^ 10 / 2500 := by
      gcongr; exact mod_cast G.edgeDensity_le_one ..
    _ = (G.edgeDensity U V - ε ^ 5 / 50) ^ 2 := by ring
    _ ≤ _ := by
      gcongr
      have rflU := Set.Subset.refl (chunk hP G ε hU).parts.toSet
      have rflV := Set.Subset.refl (chunk hP G ε hV).parts.toSet
      refine (le_trans ?_ <| density_sub_eps_le_sum_density_div_card hPα hPε rflU rflV).trans ?_
      · rw [biUnion_parts, biUnion_parts]
      · rw [card_chunk (m_pos hPα).ne', card_chunk (m_pos hPα).ne', ← cast_mul, ← mul_pow, cast_pow]
        norm_cast


private theorem abs_density_star_sub_density_le_eps (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    (hε₁ : ε ≤ 1) {hU : U ∈ P.parts} {hV : V ∈ P.parts} (hUV' : U ≠ V) (hUV : ¬G.IsUniform ε U V) :
    |(G.edgeDensity ((star hP G ε hU V).biUnion id) ((star hP G ε hV U).biUnion id) : ℝ) -
      G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε V U)| ≤ ε / 5 := by
  convert abs_edgeDensity_sub_edgeDensity_le_two_mul G.Adj
    (biUnion_star_subset_nonuniformWitness hP G ε hU V)
    (biUnion_star_subset_nonuniformWitness hP G ε hV U) (by sz_positivity)
    (one_sub_eps_mul_card_nonuniformWitness_le_card_star hV hUV' hUV hPε hε₁)
    (one_sub_eps_mul_card_nonuniformWitness_le_card_star hU hUV'.symm (fun hVU => hUV hVU.symm)
      hPε hε₁) using 1
  /-
    case h.e'_4
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUV' : Ne U V
    hUV : Not (G.IsUniform ε U V)
    ⊢ Eq (HDiv.hDiv ε 5) (HMul.hMul 2 (HDiv.hDiv ε 10))
  -/
  linarith
  /-
    🎉 no goals
  -/


private theorem eps_le_card_star_div [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α)
    (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5) (hε₁ : ε ≤ 1) (hU : U ∈ P.parts) (hV : V ∈ P.parts)
    (hUV : U ≠ V) (hunif : ¬G.IsUniform ε U V) :
    ↑4 / ↑5 * ε ≤ #(star hP G ε hU V) / ↑4 ^ #P.parts := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUV : Ne U V
    hunif : Not (G.IsUniform ε U V)
    ⊢ LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε hU …
  -/
  have hm : (0 : ℝ) ≤ 1 - (↑m)⁻¹ := sub_nonneg_of_le (inv_le_one_of_one_le₀ <| one_le_m_coe hPα)
  have hε : 0 ≤ 1 - ε / 10 :=
    sub_nonneg_of_le (div_le_one_of_le₀ (hε₁.trans <| by norm_num) <| by norm_num)
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUV : Ne U V
    hunif : Not (G.IsUniform ε U V)
    hm : LE.le 0 (HSub.hSub 1 (Inv.inv ↑(HDiv.hDiv (Fintype.card α) (SzemerediRegu …
    hε : LE.le 0 (HSub.hSub 1 (HDiv.hDiv ε 10))
    ⊢ LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε hU …
  -/
  have hε₀ : 0 < ε := by sz_positivity
  calc
    4 / 5 * ε = (1 - 1 / 10) * (1 - 9⁻¹) * ε := by norm_num
    _ ≤ (1 - ε / 10) * (1 - (↑m)⁻¹) * (#(G.nonuniformWitness ε U V) / #U) := by
        gcongr
        exacts [mod_cast (show 9 ≤ 100 by norm_num).trans (hundred_le_m hPα hPε hε₁),
          (le_div_iff₀' <| cast_pos.2 (P.nonempty_of_mem_parts hU).card_pos).2 <|
           G.le_card_nonuniformWitness hunif]
    _ = (1 - ε / 10) * #(G.nonuniformWitness ε U V) * ((1 - (↑m)⁻¹) / #U) := by
      rw [mul_assoc, mul_assoc, mul_div_left_comm]
    _ ≤ #((star hP G ε hU V).biUnion id) * ((1 - (↑m)⁻¹) / #U) :=
      (mul_le_mul_of_nonneg_right
        (one_sub_eps_mul_card_nonuniformWitness_le_card_star hV hUV hunif hPε hε₁) (by positivity))
    _ ≤ #(star hP G ε hU V) * (m + 1) * ((1 - (↑m)⁻¹) / #U) :=
      (mul_le_mul_of_nonneg_right card_biUnion_star_le_m_add_one_card_star_mul (by positivity))
    _ ≤ #(star hP G ε hU V) * (m + ↑1) * ((↑1 - (↑m)⁻¹) / (↑4 ^ #P.parts * m)) :=
      (mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_left hm (by sz_positivity) <|
        pow_mul_m_le_card_part hP hU) (by positivity))
    _ ≤ #(star hP G ε hU V) / ↑4 ^ #P.parts := by
      rw [mul_assoc, mul_comm ((4 : ℝ) ^ #P.parts), ← div_div, ← mul_div_assoc, ← mul_comm_div]
      refine mul_le_of_le_one_right (by positivity) ?_
      have hm : (0 : ℝ) < m := by sz_positivity
      rw [mul_div_assoc', div_le_one hm, ← one_div, one_sub_div hm.ne', mul_div_assoc',
        div_le_iff₀ hm]
      linarith


/-- Lower bound on the edge densities between non-uniform parts of `SzemerediRegularity.star`. -/
private theorem edgeDensity_star_not_uniform [Nonempty α]
    (hPα : #P.parts * 16 ^ #P.parts ≤ card α) (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5)
    (hε₁ : ε ≤ 1) {hU : U ∈ P.parts} {hV : V ∈ P.parts} (hUVne : U ≠ V) (hUV : ¬G.IsUniform ε U V) :
    ↑3 / ↑4 * ε ≤
    |(∑ ab ∈ (star hP G ε hU V).product (star hP G ε hV U), (G.edgeDensity ab.1 ab.2 : ℝ)) /
      (#(star hP G ε hU V) * #(star hP G ε hV U)) -
        (∑ ab ∈ (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts,
          (G.edgeDensity ab.1 ab.2 : ℝ)) / (16 : ℝ) ^ #P.parts| := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub (HDiv.hDiv (((SzemerediRegularit …
  -/
  rw [show (16 : ℝ) = ↑4 ^ 2 by norm_num, pow_right_comm, sq ((4 : ℝ) ^ _)]
  set p : ℝ :=
    (∑ ab ∈ (star hP G ε hU V).product (star hP G ε hV U), (G.edgeDensity ab.1 ab.2 : ℝ)) /
      (#(star hP G ε hU V) * #(star hP G ε hV U))
  set q : ℝ :=
    (∑ ab ∈ (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts,
      (G.edgeDensity ab.1 ab.2 : ℝ)) / (↑4 ^ #P.parts * ↑4 ^ #P.parts)
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub p q))
  -/
  set r : ℝ := ↑(G.edgeDensity ((star hP G ε hU V).biUnion id) ((star hP G ε hV U).biUnion id))
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub p q))
  -/
  set s : ℝ := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε V U))
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
    s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub p q))
  -/
  set t : ℝ := ↑(G.edgeDensity U V)
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
    s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
    t : Real := ↑(G.edgeDensity U V)
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub p q))
  -/
  have hrs : |r - s| ≤ ε / 5 := abs_density_star_sub_density_le_eps hPε hε₁ hUVne hUV
  have hst : ε ≤ |s - t| := by
    -- After https://github.com/leanprover/lean4/pull/2734, we need to do the zeta reduction before `mod_cast`.
    unfold s t
    exact mod_cast G.nonuniformWitness_spec hUVne hUV
  have hpr : |p - r| ≤ ε ^ 5 / 49 :=
    average_density_near_total_density hPα hPε hε₁ star_subset_chunk star_subset_chunk
  have hqt : |q - t| ≤ ε ^ 5 / 49 := by
    have := average_density_near_total_density hPα hPε hε₁
      (Subset.refl (chunk hP G ε hU).parts) (Subset.refl (chunk hP G ε hV).parts)
    simp_rw [← sup_eq_biUnion, sup_parts, card_chunk (m_pos hPα).ne', cast_pow] at this
    norm_num at this
    exact this
  have hε' : ε ^ 5 ≤ ε := by
    simpa using pow_le_pow_of_le_one (by sz_positivity) hε₁ (show 1 ≤ 5 by norm_num)
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
    s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
    t : Real := ↑(G.edgeDensity U V)
    hrs : LE.le (abs (HSub.hSub r s)) (HDiv.hDiv ε 5)
    hst : LE.le ε (abs (HSub.hSub s t))
    hpr : LE.le (abs (HSub.hSub p r)) (HDiv.hDiv (HPow.hPow ε 5) 49)
    hqt : LE.le (abs (HSub.hSub q t)) (HDiv.hDiv (HPow.hPow ε 5) 49)
    hε' : LE.le (HPow.hPow ε 5) ε
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub p q))
  -/
  rw [abs_sub_le_iff] at hrs hpr hqt
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
    s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
    t : Real := ↑(G.edgeDensity U V)
    hrs : And (LE.le (HSub.hSub r s) (HDiv.hDiv ε 5)) (LE.le (HSub.hSub s r) (HDiv …
    hst : LE.le ε (abs (HSub.hSub s t))
    hpr : And (LE.le (HSub.hSub p r) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
    hqt : And (LE.le (HSub.hSub q t) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
    hε' : LE.le (HPow.hPow ε 5) ε
    ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub p q))
  -/
  rw [le_abs] at hst ⊢
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hε₁ : LE.le ε 1
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    hUVne : Ne U V
    hUV : Not (G.IsUniform ε U V)
    p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
    q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
    r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
    s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
    t : Real := ↑(G.edgeDensity U V)
    hrs : And (LE.le (HSub.hSub r s) (HDiv.hDiv ε 5)) (LE.le (HSub.hSub s r) (HDiv …
    hst : Or (LE.le ε (HSub.hSub s t)) (LE.le ε (Neg.neg (HSub.hSub s t)))
    hpr : And (LE.le (HSub.hSub p r) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
    hqt : And (LE.le (HSub.hSub q t) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
    hε' : LE.le (HPow.hPow ε 5) ε
    ⊢ Or (LE.le (HMul.hMul (3 / 4) ε) (HSub.hSub p q)) (LE.le (HMul.hMul (3 / 4) ε …
  -/
  cases hst
    /-
      case inl
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      hUVne : Ne U V
      hUV : Not (G.IsUniform ε U V)
      p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
      q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
      r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
      s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
      t : Real := ↑(G.edgeDensity U V)
      hrs : And (LE.le (HSub.hSub r s) (HDiv.hDiv ε 5)) (LE.le (HSub.hSub s r) (HDiv …
      hpr : And (LE.le (HSub.hSub p r) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
      hqt : And (LE.le (HSub.hSub q t) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
      hε' : LE.le (HPow.hPow ε 5) ε
      h✝ : LE.le ε (HSub.hSub s t)
      ⊢ Or (LE.le (HMul.hMul (3 / 4) ε) (HSub.hSub p q)) (LE.le (HMul.hMul (3 / 4) ε …
    -/
  · left; linarith
          /-
            🎉 no goals
          -/
    /-
      case inr
      α : Type u_1
      inst✝³ : Fintype α
      inst✝² : DecidableEq α
      P : Finpartition Finset.univ
      hP : P.IsEquipartition
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      ε : Real
      U V : Finset α
      inst✝ : Nonempty α
      hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
      hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
      hε₁ : LE.le ε 1
      hU : Membership.mem P.parts U
      hV : Membership.mem P.parts V
      hUVne : Ne U V
      hUV : Not (G.IsUniform ε U V)
      p : Real := HDiv.hDiv (((SzemerediRegularity.star hP G ε hU V).product (Szemer …
      q : Real := HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.product (S …
      r : Real := ↑(G.edgeDensity ((SzemerediRegularity.star hP G ε hU V).biUnion id …
      s : Real := ↑(G.edgeDensity (G.nonuniformWitness ε U V) (G.nonuniformWitness ε …
      t : Real := ↑(G.edgeDensity U V)
      hrs : And (LE.le (HSub.hSub r s) (HDiv.hDiv ε 5)) (LE.le (HSub.hSub s r) (HDiv …
      hpr : And (LE.le (HSub.hSub p r) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
      hqt : And (LE.le (HSub.hSub q t) (HDiv.hDiv (HPow.hPow ε 5) 49)) (LE.le (HSub. …
      hε' : LE.le (HPow.hPow ε 5) ε
      h✝ : LE.le ε (Neg.neg (HSub.hSub s t))
      ⊢ Or (LE.le (HMul.hMul (3 / 4) ε) (HSub.hSub p q)) (LE.le (HMul.hMul (3 / 4) ε …
    -/
  · right; linarith
           /-
             🎉 no goals
           -/


/-- Lower bound on the edge densities between non-uniform parts of `SzemerediRegularity.increment`.
-/
theorem edgeDensity_chunk_not_uniform [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α)
    (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5) (hε₁ : ε ≤ 1) {hU : U ∈ P.parts} {hV : V ∈ P.parts}
    (hUVne : U ≠ V) (hUV : ¬G.IsUniform ε U V) :
    (G.edgeDensity U V : ℝ) ^ 2 - ε ^ 5 / ↑25 + ε ^ 4 / ↑3 ≤
    (∑ ab ∈ (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts,
      (G.edgeDensity ab.1 ab.2 : ℝ) ^ 2) / ↑16 ^ #P.parts :=
  calc
    ↑(G.edgeDensity U V) ^ 2 - ε ^ 5 / 25 + ε ^ 4 / ↑3 ≤ ↑(G.edgeDensity U V) ^ 2 - ε ^ 5 / ↑25 +
        #(star hP G ε hU V) * #(star hP G ε hV U) / ↑16 ^ #P.parts *
          (↑9 / ↑16) * ε ^ 2 := by
      /-
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv  …
      -/
      apply add_le_add_left
      have Ul : 4 / 5 * ε ≤ #(star hP G ε hU V) / _ :=
        eps_le_card_star_div hPα hPε hε₁ hU hV hUVne hUV
      have Vl : 4 / 5 * ε ≤ #(star hP G ε hV U) / _ :=
        eps_le_card_star_div hPα hPε hε₁ hV hU hUVne.symm fun h => hUV h.symm
      rw [show (16 : ℝ) = ↑4 ^ 2 by norm_num, pow_right_comm, sq ((4 : ℝ) ^ _), ←
        _root_.div_mul_div_comm, mul_assoc]
      /-
        case bc
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 4) 3) (HMul.hMul (HMul.hMul (HDiv.hDiv (↑(Szem …
      -/
      have : 0 < ε := by sz_positivity
      /-
        case bc
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        this : LT.lt 0 ε
        ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 4) 3) (HMul.hMul (HMul.hMul (HDiv.hDiv (↑(Szem …
      -/
      have UVl := mul_le_mul Ul Vl (by positivity) ?_
      /-
        case bc.refine_2
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        this : LT.lt 0 ε
        UVl : LE.le (HMul.hMul (HMul.hMul (4 / 5) ε) (HMul.hMul (4 / 5) ε)) (HMul.hMul …
        ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 4) 3) (HMul.hMul (HMul.hMul (HDiv.hDiv (↑(Szem …
      -/
      swap
      · -- This seems faster than `exact div_nonneg (by positivity) (by positivity)` and *much*
        -- (tens of seconds) faster than `positivity` on its own.
        /-
          case bc.refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          this : LT.lt 0 ε
          ⊢ LE.le 0 (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε hU V).card) (HPow.hPow …
        -/
                             /-
                               🎉 no goals
                             -/
        apply div_nonneg <;> positivity
                             /-
                               🎉 no goals
                             -/
      /-
        case bc.refine_2
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
        this : LT.lt 0 ε
        UVl : LE.le (HMul.hMul (HMul.hMul (4 / 5) ε) (HMul.hMul (4 / 5) ε)) (HMul.hMul …
        ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 4) 3) (HMul.hMul (HMul.hMul (HDiv.hDiv (↑(Szem …
      -/
      refine le_trans ?_ (mul_le_mul_of_nonneg_right UVl ?_)
        /-
          case bc.refine_2.refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          this : LT.lt 0 ε
          UVl : LE.le (HMul.hMul (HMul.hMul (4 / 5) ε) (HMul.hMul (4 / 5) ε)) (HMul.hMul …
          ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 4) 3) (HMul.hMul (HMul.hMul (HMul.hMul (4 / 5) …
        -/
      · norm_num
        /-
          case bc.refine_2.refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          this : LT.lt 0 ε
          UVl : LE.le (HMul.hMul (HMul.hMul (4 / 5) ε) (HMul.hMul (4 / 5) ε)) (HMul.hMul …
          ⊢ LE.le (HDiv.hDiv (HPow.hPow ε 4) 3) (HMul.hMul (HMul.hMul (HMul.hMul (4 / 5) …
        -/
        nlinarith
        /-
          🎉 no goals
        -/
        /-
          case bc.refine_2.refine_2
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          this : LT.lt 0 ε
          UVl : LE.le (HMul.hMul (HMul.hMul (4 / 5) ε) (HMul.hMul (4 / 5) ε)) (HMul.hMul …
          ⊢ LE.le 0 (HMul.hMul (HDiv.hDiv 9 (HPow.hPow 4 2)) (HPow.hPow ε 2))
        -/
      · norm_num
        /-
          case bc.refine_2.refine_2
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          Ul : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          Vl : LE.le (HMul.hMul (4 / 5) ε) (HDiv.hDiv (↑(SzemerediRegularity.star hP G ε …
          this : LT.lt 0 ε
          UVl : LE.le (HMul.hMul (HMul.hMul (4 / 5) ε) (HMul.hMul (4 / 5) ε)) (HMul.hMul …
          ⊢ LE.le 0 (HPow.hPow ε 2)
        -/
        positivity
        /-
          🎉 no goals
        -/
    _ ≤ (∑ ab ∈ (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts,
        (G.edgeDensity ab.1 ab.2 : ℝ) ^ 2) / ↑16 ^ #P.parts := by
      have t : (star hP G ε hU V).product (star hP G ε hV U) ⊆
          (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts :=
        product_subset_product star_subset_chunk star_subset_chunk
      /-
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
        ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv  …
      -/
      have hε : 0 ≤ ε := by sz_positivity
      /-
        α : Type u_1
        inst✝³ : Fintype α
        inst✝² : DecidableEq α
        P : Finpartition Finset.univ
        hP : P.IsEquipartition
        G : SimpleGraph α
        inst✝¹ : DecidableRel G.Adj
        ε : Real
        U V : Finset α
        inst✝ : Nonempty α
        hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
        hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
        hε₁ : LE.le ε 1
        hU : Membership.mem P.parts U
        hV : Membership.mem P.parts V
        hUVne : Ne U V
        hUV : Not (G.IsUniform ε U V)
        t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
        hε : LE.le 0 ε
        ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv  …
      -/
      have sp : ∀ (a b : Finset (Finset α)), a.product b = a ×ˢ b := fun a b => rfl
      have := add_div_le_sum_sq_div_card t (fun x => (G.edgeDensity x.1 x.2 : ℝ))
        ((G.edgeDensity U V : ℝ) ^ 2 - ε ^ 5 / ↑25) (show 0 ≤ 3 / 4 * ε by linarith) ?_ ?_
      · simp_rw [sp, card_product, card_chunk (m_pos hPα).ne', ← mul_pow, cast_pow, mul_pow,
          div_pow, ← mul_assoc] at this
        /-
          case refine_3
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          this : LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv. …
          ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv  …
        -/
        norm_num at this
        /-
          case refine_3
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          this : LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv. …
          ⊢ LE.le (HAdd.hAdd (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv  …
        -/
        exact this
        /-
          🎉 no goals
        -/
        /-
          case refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub (HDiv.hDiv (((SzemerediRegularit …
        -/
      · simp_rw [sp, card_product, card_chunk (m_pos hPα).ne', ← mul_pow]
        /-
          case refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub (HDiv.hDiv ((SProd.sprod (Szemer …
        -/
        norm_num
        /-
          case refine_1
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          ⊢ LE.le (HMul.hMul (3 / 4) ε) (abs (HSub.hSub (HDiv.hDiv ((SProd.sprod (Szemer …
        -/
        exact edgeDensity_star_not_uniform hPα hPε hε₁ hUVne hUV
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
        -/
      · rw [sp, card_product]
        /-
          case refine_2
          α : Type u_1
          inst✝³ : Fintype α
          inst✝² : DecidableEq α
          P : Finpartition Finset.univ
          hP : P.IsEquipartition
          G : SimpleGraph α
          inst✝¹ : DecidableRel G.Adj
          ε : Real
          U V : Finset α
          inst✝ : Nonempty α
          hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
          hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
          hε₁ : LE.le ε 1
          hU : Membership.mem P.parts U
          hV : Membership.mem P.parts V
          hUVne : Ne U V
          hUV : Not (G.IsUniform ε U V)
          t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
          hε : LE.le 0 ε
          sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
          ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
        -/
        apply (edgeDensity_chunk_aux hPα hPε hU hV).trans
          /-
            case refine_2
            α : Type u_1
            inst✝³ : Fintype α
            inst✝² : DecidableEq α
            P : Finpartition Finset.univ
            hP : P.IsEquipartition
            G : SimpleGraph α
            inst✝¹ : DecidableRel G.Adj
            ε : Real
            U V : Finset α
            inst✝ : Nonempty α
            hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
            hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
            hε₁ : LE.le ε 1
            hU : Membership.mem P.parts U
            hV : Membership.mem P.parts V
            hUVne : Ne U V
            hUV : Not (G.IsUniform ε U V)
            t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
            hε : LE.le 0 ε
            sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
            ⊢ LE.le (HPow.hPow (HDiv.hDiv (((SzemerediRegularity.chunk ?m.666086 G ε hU).p …
          -/
        · rw [card_chunk (m_pos hPα).ne', card_chunk (m_pos hPα).ne', ← mul_pow]
            /-
              case refine_2
              α : Type u_1
              inst✝³ : Fintype α
              inst✝² : DecidableEq α
              P : Finpartition Finset.univ
              hP : P.IsEquipartition
              G : SimpleGraph α
              inst✝¹ : DecidableRel G.Adj
              ε : Real
              U V : Finset α
              inst✝ : Nonempty α
              hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
              hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
              hε₁ : LE.le ε 1
              hU : Membership.mem P.parts U
              hV : Membership.mem P.parts V
              hUVne : Ne U V
              hUV : Not (G.IsUniform ε U V)
              t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
              hε : LE.le 0 ε
              sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
              ⊢ LE.le (HPow.hPow (HDiv.hDiv (((SzemerediRegularity.chunk ?m.666086 G ε hU).p …
            -/
          · norm_num
            /-
              case refine_2
              α : Type u_1
              inst✝³ : Fintype α
              inst✝² : DecidableEq α
              P : Finpartition Finset.univ
              hP : P.IsEquipartition
              G : SimpleGraph α
              inst✝¹ : DecidableRel G.Adj
              ε : Real
              U V : Finset α
              inst✝ : Nonempty α
              hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
              hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
              hε₁ : LE.le ε 1
              hU : Membership.mem P.parts U
              hV : Membership.mem P.parts V
              hUVne : Ne U V
              hUV : Not (G.IsUniform ε U V)
              t : HasSubset.Subset ((SzemerediRegularity.star hP G ε hU V).product (Szemered …
              hε : LE.le 0 ε
              sp : ∀ (a b : Finset (Finset α)), Eq (a.product b) (SProd.sprod a b)
              ⊢ LE.le (HPow.hPow (HDiv.hDiv (((SzemerediRegularity.chunk ?m.666086 G ε hU).p …
            -/
            rfl
            /-
              🎉 no goals
            -/


/-- Lower bound on the edge densities between parts of `SzemerediRegularity.increment`. This is the
blanket lower bound used the uniform parts. -/
theorem edgeDensity_chunk_uniform [Nonempty α] (hPα : #P.parts * 16 ^ #P.parts ≤ card α)
    (hPε : ↑100 ≤ ↑4 ^ #P.parts * ε ^ 5) (hU : U ∈ P.parts) (hV : V ∈ P.parts) :
    (G.edgeDensity U V : ℝ) ^ 2 - ε ^ 5 / ↑25 ≤
    (∑ ab ∈ (chunk hP G ε hU).parts.product (chunk hP G ε hV).parts,
      (G.edgeDensity ab.1 ab.2 : ℝ) ^ 2) / ↑16 ^ #P.parts := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    ⊢ LE.le (HSub.hSub (HPow.hPow (↑(G.edgeDensity U V)) 2) (HDiv.hDiv (HPow.hPow  …
  -/
  apply (edgeDensity_chunk_aux (hP := hP) hPα hPε hU hV).trans
  have key : (16 : ℝ) ^ #P.parts = #((chunk hP G ε hU).parts ×ˢ (chunk hP G ε hV).parts) := by
    rw [card_product, cast_mul, card_chunk (m_pos hPα).ne', card_chunk (m_pos hPα).ne', ←
      cast_mul, ← mul_pow]; norm_cast
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    key : Eq (HPow.hPow 16 P.parts.card) ↑(SProd.sprod (SzemerediRegularity.chunk  …
    ⊢ LE.le (HPow.hPow (HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.pr …
  -/
  simp_rw [key]
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    P : Finpartition Finset.univ
    hP : P.IsEquipartition
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    U V : Finset α
    inst✝ : Nonempty α
    hPα : LE.le (HMul.hMul P.parts.card (HPow.hPow 16 P.parts.card)) (Fintype.card …
    hPε : LE.le 100 (HMul.hMul (HPow.hPow 4 P.parts.card) (HPow.hPow ε 5))
    hU : Membership.mem P.parts U
    hV : Membership.mem P.parts V
    key : Eq (HPow.hPow 16 P.parts.card) ↑(SProd.sprod (SzemerediRegularity.chunk  …
    ⊢ LE.le (HPow.hPow (HDiv.hDiv (((SzemerediRegularity.chunk hP G ε hU).parts.pr …
  -/
  convert sum_div_card_sq_le_sum_sq_div_card (α := ℝ)
  /-
    🎉 no goals
  -/


