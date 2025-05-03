/-- The commuting probability of a finite type with a multiplication operation. -/
def commProb : ℚ :=
  Nat.card { p : M × M // Commute p.1 p.2 } / (Nat.card M : ℚ) ^ 2


theorem commProb_def :
    commProb M = Nat.card { p : M × M // Commute p.1 p.2 } / (Nat.card M : ℚ) ^ 2 :=
  rfl


theorem commProb_prod (M' : Type*) [Mul M'] : commProb (M × M') = commProb M * commProb M' := by
  simp_rw [commProb_def, div_mul_div_comm, Nat.card_prod, Nat.cast_mul, mul_pow, ← Nat.cast_mul,
    ← Nat.card_prod, Commute, SemiconjBy, Prod.ext_iff]
  /-
    M : Type u_1
    inst✝¹ : Mul M
    M' : Type u_2
    inst✝ : Mul M'
    ⊢ Eq (HDiv.hDiv (↑(Nat.card (Subtype fun p => And (Eq (HMul.hMul p.1 p.2).1 (H …
  -/
  congr 2
  exact Nat.card_congr ⟨fun x => ⟨⟨⟨x.1.1.1, x.1.2.1⟩, x.2.1⟩, ⟨⟨x.1.1.2, x.1.2.2⟩, x.2.2⟩⟩,
    fun x => ⟨⟨⟨x.1.1.1, x.2.1.1⟩, ⟨x.1.1.2, x.2.1.2⟩⟩, ⟨x.1.2, x.2.2⟩⟩, fun x => rfl, fun x => rfl⟩


theorem commProb_pi {α : Type*} (i : α → Type*) [Fintype α] [∀ a, Mul (i a)] :
    commProb (∀ a, i a) = ∏ a, commProb (i a) := by
  simp_rw [commProb_def, Finset.prod_div_distrib, Finset.prod_pow, ← Nat.cast_prod,
    ← Nat.card_pi, Commute, SemiconjBy, funext_iff]
  /-
    α : Type u_2
    i : α → Type u_3
    inst✝¹ : Fintype α
    inst✝ : (a : α) → Mul (i a)
    ⊢ Eq (HDiv.hDiv (↑(Nat.card (Subtype fun p => ∀ (x : α), Eq (HMul.hMul p.1 p.2 …
  -/
  congr 2
  exact Nat.card_congr ⟨fun x a => ⟨⟨x.1.1 a, x.1.2 a⟩, x.2 a⟩, fun x => ⟨⟨fun a => (x a).1.1,
    fun a => (x a).1.2⟩, fun a => (x a).2⟩, fun x => rfl, fun x => rfl⟩


theorem commProb_function {α β : Type*} [Fintype α] [Mul β] :
    commProb (α → β) = (commProb β) ^ Fintype.card α := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Fintype α
    inst✝ : Mul β
    ⊢ Eq (commProb (α → β)) (HPow.hPow (commProb β) (Fintype.card α))
  -/
  rw [commProb_pi, Finset.prod_const, Finset.card_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem commProb_eq_zero_of_infinite [Infinite M] : commProb M = 0 :=
  div_eq_zero_iff.2 (Or.inl (Nat.cast_eq_zero.2 Nat.card_eq_zero_of_infinite))


theorem commProb_pos [h : Nonempty M] : 0 < commProb M :=
  h.elim fun x ↦
    div_pos (Nat.cast_pos.mpr (Finite.card_pos_iff.mpr ⟨⟨(x, x), rfl⟩⟩))
      (pow_pos (Nat.cast_pos.mpr Finite.card_pos) 2)


theorem commProb_le_one : commProb M ≤ 1 := by
  /-
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Finite M
    ⊢ LE.le (commProb M) 1
  -/
  refine div_le_one_of_le₀ ?_ (sq_nonneg (Nat.card M : ℚ))
  /-
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Finite M
    ⊢ LE.le (↑(Nat.card (Subtype fun p => Commute p.1 p.2))) (HPow.hPow (↑(Nat.car …
  -/
  rw [← Nat.cast_pow, Nat.cast_le, sq, ← Nat.card_prod]
  /-
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Finite M
    ⊢ LE.le (Nat.card (Subtype fun p => Commute p.1 p.2)) (Nat.card (Prod M M))
  -/
  apply Finite.card_subtype_le
  /-
    🎉 no goals
  -/


theorem commProb_eq_one_iff [h : Nonempty M] :
    commProb M = 1 ↔ Std.Commutative ((· * ·) : M → M → M) := by
  /-
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Finite M
    h : Nonempty M
    ⊢ Iff (Eq (commProb M) 1) (Std.Commutative fun x1 x2 => HMul.hMul x1 x2)
  -/
  haveI := Fintype.ofFinite M
  /-
    M : Type u_1
    inst✝¹ : Mul M
    inst✝ : Finite M
    h : Nonempty M
    this : Fintype M
    ⊢ Iff (Eq (commProb M) 1) (Std.Commutative fun x1 x2 => HMul.hMul x1 x2)
  -/
  rw [commProb, ← Set.coe_setOf, Nat.card_eq_fintype_card, Nat.card_eq_fintype_card]
  rw [div_eq_one_iff_eq, ← Nat.cast_pow, Nat.cast_inj, sq, ← card_prod,
    set_fintype_card_eq_univ_iff, Set.eq_univ_iff_forall]
    /-
      M : Type u_1
      inst✝¹ : Mul M
      inst✝ : Finite M
      h : Nonempty M
      this : Fintype M
      ⊢ Iff (∀ (x : Prod M M), Membership.mem (setOf fun x => Commute x.1 x.2) x) (S …
    -/
  · exact ⟨fun h ↦ ⟨fun x y ↦ h (x, y)⟩, fun h x ↦ h.comm x.1 x.2⟩
    /-
      🎉 no goals
    -/
    /-
      M : Type u_1
      inst✝¹ : Mul M
      inst✝ : Finite M
      h : Nonempty M
      this : Fintype M
      ⊢ Ne (HPow.hPow (↑(Fintype.card M)) 2) 0
    -/
  · exact pow_ne_zero 2 (Nat.cast_ne_zero.mpr card_ne_zero)
    /-
      🎉 no goals
    -/


theorem commProb_def' : commProb G = Nat.card (ConjClasses G) / Nat.card G := by
  /-
    G : Type u_2
    inst✝ : Group G
    ⊢ Eq (commProb G) (HDiv.hDiv ↑(Nat.card (ConjClasses G)) ↑(Nat.card G))
  -/
  rw [commProb, card_comm_eq_card_conjClasses_mul_card, Nat.cast_mul, sq]
  /-
    G : Type u_2
    inst✝ : Group G
    ⊢ Eq (HDiv.hDiv (HMul.hMul ↑(Nat.card (ConjClasses G)) ↑(Nat.card G)) (HMul.hM …
  -/
  by_cases h : (Nat.card G : ℚ) = 0
    /-
      case pos
      G : Type u_2
      inst✝ : Group G
      h : Eq (↑(Nat.card G)) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul ↑(Nat.card (ConjClasses G)) ↑(Nat.card G)) (HMul.hM …
    -/
  · rw [h, zero_mul, div_zero, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_2
      inst✝ : Group G
      h : Not (Eq (↑(Nat.card G)) 0)
      ⊢ Eq (HDiv.hDiv (HMul.hMul ↑(Nat.card (ConjClasses G)) ↑(Nat.card G)) (HMul.hM …
    -/
  · exact mul_div_mul_right _ _ h
    /-
      🎉 no goals
    -/


theorem Subgroup.commProb_subgroup_le : commProb H ≤ commProb G * (H.index : ℚ) ^ 2 := by
  /- After rewriting with `commProb_def`, we reduce to showing that `G` has at least as many
      commuting pairs as `H`. -/
  rw [commProb_def, commProb_def, div_le_iff₀, mul_assoc, ← mul_pow, ← Nat.cast_mul,
    mul_comm H.index, H.card_mul_index, div_mul_cancel₀, Nat.cast_le]
    /-
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Finite G
      H : Subgroup G
      ⊢ LE.le (Nat.card (Subtype fun p => Commute p.1 p.2)) (Nat.card (Subtype fun p …
    -/
  · refine Finite.card_le_of_injective (fun p ↦ ⟨⟨p.1.1, p.1.2⟩, Subtype.ext_iff.mp p.2⟩) ?_
    /-
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Finite G
      H : Subgroup G
      ⊢ Function.Injective fun p => ⟨{ fst := ↑(↑p).1, snd := ↑(↑p).2 }, ⋯⟩
    -/
    exact fun p q h ↦ by simpa only [Subtype.ext_iff, Prod.ext_iff] using h
    /-
      🎉 no goals
    -/
    /-
      case h
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Finite G
      H : Subgroup G
      ⊢ Ne (HPow.hPow (↑(Nat.card G)) 2) 0
    -/
  · exact pow_ne_zero 2 (Nat.cast_ne_zero.mpr Finite.card_pos.ne')
    /-
      🎉 no goals
    -/
    /-
      G : Type u_2
      inst✝¹ : Group G
      inst✝ : Finite G
      H : Subgroup G
      ⊢ LT.lt 0 (HPow.hPow (↑(Nat.card (Subtype fun x => Membership.mem H x))) 2)
    -/
  · exact pow_pos (Nat.cast_pos.mpr Finite.card_pos) 2
    /-
      🎉 no goals
    -/


theorem Subgroup.commProb_quotient_le [H.Normal] : commProb (G ⧸ H) ≤ commProb G * Nat.card H := by
  /- After rewriting with `commProb_def'`, we reduce to showing that `G` has at least as many
      conjugacy classes as `G ⧸ H`. -/
  rw [commProb_def', commProb_def', div_le_iff₀, mul_assoc, ← Nat.cast_mul, ← Subgroup.index,
    H.card_mul_index, div_mul_cancel₀, Nat.cast_le]
    /-
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : Finite G
      H : Subgroup G
      inst✝ : H.Normal
      ⊢ LE.le (Nat.card (ConjClasses (HasQuotient.Quotient G H))) (Nat.card (ConjCla …
    -/
  · apply Finite.card_le_of_surjective
    /-
      case hf
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : Finite G
      H : Subgroup G
      inst✝ : H.Normal
      ⊢ Function.Surjective ?f
    -/
    show Function.Surjective (ConjClasses.map (QuotientGroup.mk' H))
    /-
      case hf
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : Finite G
      H : Subgroup G
      inst✝ : H.Normal
      ⊢ Function.Surjective (ConjClasses.map (QuotientGroup.mk' H))
    -/
    exact ConjClasses.map_surjective Quotient.mk''_surjective
    /-
      🎉 no goals
    -/
    /-
      case h
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : Finite G
      H : Subgroup G
      inst✝ : H.Normal
      ⊢ Ne (↑(Nat.card G)) 0
    -/
  · exact Nat.cast_ne_zero.mpr Finite.card_pos.ne'
    /-
      🎉 no goals
    -/
    /-
      G : Type u_2
      inst✝² : Group G
      inst✝¹ : Finite G
      H : Subgroup G
      inst✝ : H.Normal
      ⊢ LT.lt 0 ↑(Nat.card (HasQuotient.Quotient G H))
    -/
  · exact Nat.cast_pos.mpr Finite.card_pos
    /-
      🎉 no goals
    -/


theorem inv_card_commutator_le_commProb : (↑(Nat.card (commutator G)))⁻¹ ≤ commProb G :=
  (inv_le_iff_one_le_mul₀ (Nat.cast_pos.mpr Finite.card_pos)).mpr
    (le_trans (ge_of_eq (commProb_eq_one_iff.mpr ⟨(Abelianization.commGroup G).mul_comm⟩))
      (commutator G).commProb_quotient_le)

-- Construction of group with commuting probability 1/n

lemma commProb_odd {n : ℕ} (hn : Odd n) :
    commProb (DihedralGroup n) = (n + 3) / (4 * n) := by
  /-
    n : Nat
    hn : Odd n
    ⊢ Eq (commProb (DihedralGroup n)) (HDiv.hDiv (HAdd.hAdd (↑n) 3) (HMul.hMul 4 ↑ …
  -/
  rw [commProb_def', DihedralGroup.card_conjClasses_odd hn, nat_card]
  /-
    n : Nat
    hn : Odd n
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv (HAdd.hAdd n 3) 2) ↑(HMul.hMul 2 n)) (HDiv.hDiv (H …
  -/
  qify [show 2 ∣ n + 3 by rw [Nat.dvd_iff_mod_eq_zero, Nat.add_mod, Nat.odd_iff.mp hn]]
  /-
    n : Nat
    hn : Odd n
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (HAdd.hAdd (↑n) 3) 2) (HMul.hMul 2 ↑n)) (HDiv.hDiv  …
  -/
  rw [div_div, ← mul_assoc]
  /-
    n : Nat
    hn : Odd n
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (↑n) 3) (HMul.hMul (HMul.hMul 2 2) ↑n)) (HDiv.hDiv  …
  -/
  congr
  /-
    case e_a.e_a
    n : Nat
    hn : Odd n
    ⊢ Eq (HMul.hMul 2 2) 4
  -/
  norm_num
  /-
    🎉 no goals
  -/


private lemma div_two_lt {n : ℕ} (h0 : n ≠ 0) : n / 2 < n :=
  Nat.div_lt_self (Nat.pos_of_ne_zero h0) (lt_add_one 1)


private lemma div_four_lt : {n : ℕ} → (h0 : n ≠ 0) → (h1 : n ≠ 1) → n / 4 + 1 < n
                        /-
                          ⊢ Ne 0 0 → Ne 0 1 → LT.lt (HAdd.hAdd (0 / 4) 1) 0
                        -/
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
  | 0 | 1 | 2 | 3 => by decide
                        /-
                          🎉 no goals
                        -/
                /-
                  n : Nat
                  ⊢ Ne (HAdd.hAdd n 4) 0 → Ne (HAdd.hAdd n 4) 1 → LT.lt (HAdd.hAdd (HDiv.hDiv (H …
                -/
  | n + 4 => by omega
                /-
                  🎉 no goals
                -/


/-- A list of Dihedral groups whose product will have commuting probability `1 / n`. -/
def reciprocalFactors (n : ℕ) : List ℕ :=
  if _ : n = 0 then [0]
  else if _ : n = 1 then []
  else if Even n then
    3 :: reciprocalFactors (n / 2)
  else
    n % 4 * n :: reciprocalFactors (n / 4 + 1)


@[simp] lemma reciprocalFactors_zero : reciprocalFactors 0 = [0] := by
  /-
    ⊢ Eq (DihedralGroup.reciprocalFactors 0) (List.cons 0 List.nil)
  -/
  unfold reciprocalFactors; rfl
                            /-
                              🎉 no goals
                            -/


@[simp] lemma reciprocalFactors_one : reciprocalFactors 1 = [] := by
  /-
    ⊢ Eq (DihedralGroup.reciprocalFactors 1) List.nil
  -/
  unfold reciprocalFactors; rfl
                            /-
                              🎉 no goals
                            -/


lemma reciprocalFactors_even {n : ℕ} (h0 : n ≠ 0) (h2 : Even n) :
    reciprocalFactors n = 3 :: reciprocalFactors (n / 2) := by
  have h1 : n ≠ 1 := by
    rintro rfl
    norm_num at h2
  /-
    n : Nat
    h0 : Ne n 0
    h2 : Even n
    h1 : Ne n 1
    ⊢ Eq (DihedralGroup.reciprocalFactors n) (List.cons 3 (DihedralGroup.reciproca …
  -/
  rw [reciprocalFactors, dif_neg h0, dif_neg h1, if_pos h2]
  /-
    🎉 no goals
  -/


lemma reciprocalFactors_odd {n : ℕ} (h1 : n ≠ 1) (h2 : Odd n) :
    reciprocalFactors n = n % 4 * n :: reciprocalFactors (n / 4 + 1) := by
  have h0 : n ≠ 0 := by
    rintro rfl
    norm_num [← Nat.not_even_iff_odd] at h2
  /-
    n : Nat
    h1 : Ne n 1
    h2 : Odd n
    h0 : Ne n 0
    ⊢ Eq (DihedralGroup.reciprocalFactors n) (List.cons (HMul.hMul (HMod.hMod n 4) …
  -/
  rw [reciprocalFactors, dif_neg h0, dif_neg h1, if_neg (Nat.not_even_iff_odd.2 h2)]
  /-
    🎉 no goals
  -/


/-- A finite product of Dihedral groups. -/
abbrev Product (l : List ℕ) : Type :=
  ∀ i : Fin l.length, DihedralGroup l[i]


lemma commProb_nil : commProb (Product []) = 1 := by
  /-
    ⊢ Eq (commProb (DihedralGroup.Product List.nil)) 1
  -/
  simp [Product, commProb_pi]
  /-
    🎉 no goals
  -/


lemma commProb_cons (n : ℕ) (l : List ℕ) :
    commProb (Product (n :: l)) = commProb (DihedralGroup n) * commProb (Product l) := by
  /-
    n : Nat
    l : List Nat
    ⊢ Eq (commProb (DihedralGroup.Product (List.cons n l))) (HMul.hMul (commProb ( …
  -/
  simp [Product, commProb_pi, Fin.prod_univ_succ]
  /-
    🎉 no goals
  -/


/-- Construction of a group with commuting probability `1 / n`. -/
theorem commProb_reciprocal (n : ℕ) :
    commProb (Product (reciprocalFactors n)) = 1 / n := by
  /-
    n : Nat
    ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
  -/
  by_cases h0 : n = 0
    /-
      case pos
      n : Nat
      h0 : Eq n 0
      ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
    -/
  · rw [h0, reciprocalFactors_zero, commProb_cons, commProb_nil, mul_one, Nat.cast_zero, div_zero]
    /-
      case pos
      n : Nat
      h0 : Eq n 0
      ⊢ Eq (commProb (DihedralGroup 0)) 0
    -/
    apply commProb_eq_zero_of_infinite
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    h0 : Not (Eq n 0)
    ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
  -/
  by_cases h1 : n = 1
    /-
      case pos
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Eq n 1
      ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
    -/
  · rw [h1, reciprocalFactors_one, commProb_nil, Nat.cast_one, div_one]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    h0 : Not (Eq n 0)
    h1 : Not (Eq n 1)
    ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
  -/
  rcases Nat.even_or_odd n with h2 | h2
    /-
      case neg.inl
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Even n
      ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
    -/
  · have := div_two_lt h0
    rw [reciprocalFactors_even h0 h2, commProb_cons, commProb_reciprocal (n / 2),
        commProb_odd (by decide)]
    /-
      case neg.inl
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Even n
      this : LT.lt (HDiv.hDiv n 2) n
      ⊢ Eq (HMul.hMul (HDiv.hDiv (HAdd.hAdd (↑3) 3) (HMul.hMul 4 ↑3)) (HDiv.hDiv 1 ↑ …
    -/
    field_simp [h0, h2.two_dvd]
    /-
      case neg.inl
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Even n
      this : LT.lt (HDiv.hDiv n 2) n
      ⊢ Eq (HMul.hMul (HAdd.hAdd 3 3) 2) (HMul.hMul 4 3)
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Odd n
      ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
    -/
  · have := div_four_lt h0 h1
    /-
      case neg.inr
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Odd n
      this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
      ⊢ Eq (commProb (DihedralGroup.Product (DihedralGroup.reciprocalFactors n))) (H …
    -/
    rw [reciprocalFactors_odd h1 h2, commProb_cons, commProb_reciprocal (n / 4 + 1)]
    /-
      case neg.inr
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Odd n
      this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
      ⊢ Eq (HMul.hMul (commProb (DihedralGroup (HMul.hMul (HMod.hMod n 4) n))) (HDiv …
    -/
    have key : n % 4 = 1 ∨ n % 4 = 3 := Nat.odd_mod_four_iff.mp (Nat.odd_iff.mp h2)
    /-
      case neg.inr
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Odd n
      this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
      key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
      ⊢ Eq (HMul.hMul (commProb (DihedralGroup (HMul.hMul (HMod.hMod n 4) n))) (HDiv …
    -/
    have hn : Odd (n % 4) := by rcases key with h | h <;> rw [h] <;> decide
    /-
      case neg.inr
      n : Nat
      h0 : Not (Eq n 0)
      h1 : Not (Eq n 1)
      h2 : Odd n
      this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
      key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
      hn : Odd (HMod.hMod n 4)
      ⊢ Eq (HMul.hMul (commProb (DihedralGroup (HMul.hMul (HMod.hMod n 4) n))) (HDiv …
    -/
    rw [commProb_odd (hn.mul h2), div_mul_div_comm, mul_one, div_eq_div_iff, one_mul] <;> norm_cast
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
      /-
        case neg.inr
        n : Nat
        h0 : Not (Eq n 0)
        h1 : Not (Eq n 1)
        h2 : Odd n
        this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
        key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
        hn : Odd (HMod.hMod n 4)
        ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HMod.hMod n 4) n) 3) n) (HMul.hMul (HMu …
      -/
    · have h0 : (n % 4) ^ 2 + 3 = n % 4 * 4 := by rcases key with h | h <;> rw [h] <;> norm_num
      /-
        case neg.inr
        n : Nat
        h0✝ : Not (Eq n 0)
        h1 : Not (Eq n 1)
        h2 : Odd n
        this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
        key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
        hn : Odd (HMod.hMod n 4)
        h0 : Eq (HAdd.hAdd (HPow.hPow (HMod.hMod n 4) 2) 3) (HMul.hMul (HMod.hMod n 4) …
        ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HMod.hMod n 4) n) 3) n) (HMul.hMul (HMu …
      -/
      have h1 := (Nat.div_add_mod n 4).symm
      /-
        case neg.inr
        n : Nat
        h0✝ : Not (Eq n 0)
        h1✝ : Not (Eq n 1)
        h2 : Odd n
        this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
        key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
        hn : Odd (HMod.hMod n 4)
        h0 : Eq (HAdd.hAdd (HPow.hPow (HMod.hMod n 4) 2) 3) (HMul.hMul (HMod.hMod n 4) …
        h1 : Eq n (HAdd.hAdd (HMul.hMul 4 (HDiv.hDiv n 4)) (HMod.hMod n 4))
        ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HMod.hMod n 4) n) 3) n) (HMul.hMul (HMu …
      -/
      zify at h0 h1 ⊢
      /-
        case neg.inr
        n : Nat
        h0✝ : Not (Eq n 0)
        h1✝ : Not (Eq n 1)
        h2 : Odd n
        this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
        key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
        hn : Odd (HMod.hMod n 4)
        h0 : Eq (HAdd.hAdd (HPow.hPow (HMod.hMod (↑n) 4) 2) 3) (HMul.hMul (HMod.hMod ( …
        h1 : Eq (↑n) (HAdd.hAdd (HMul.hMul 4 (HDiv.hDiv (↑n) 4)) (HMod.hMod (↑n) 4))
        ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HMod.hMod (↑n) 4) ↑n) 3) ↑n) (HMul.hMul …
      -/
      linear_combination (h0 + h1 * (n % 4)) * n
      /-
        🎉 no goals
      -/
      /-
        case neg.inr.hb
        n : Nat
        h0 : Not (Eq n 0)
        h1 : Not (Eq n 1)
        h2 : Odd n
        this : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
        key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
        hn : Odd (HMod.hMod n 4)
        ⊢ Not (Eq (HMul.hMul (HMul.hMul 4 (HMul.hMul (HMod.hMod n 4) n)) (HAdd.hAdd (H …
      -/
    · have := hn.pos.ne'
      /-
        case neg.inr.hb
        n : Nat
        h0 : Not (Eq n 0)
        h1 : Not (Eq n 1)
        h2 : Odd n
        this✝ : LT.lt (HAdd.hAdd (HDiv.hDiv n 4) 1) n
        key : Or (Eq (HMod.hMod n 4) 1) (Eq (HMod.hMod n 4) 3)
        hn : Odd (HMod.hMod n 4)
        this : Ne (HMod.hMod n 4) 0
        ⊢ Not (Eq (HMul.hMul (HMul.hMul 4 (HMul.hMul (HMod.hMod n 4) n)) (HAdd.hAdd (H …
      -/
      positivity
      /-
        🎉 no goals
      -/


