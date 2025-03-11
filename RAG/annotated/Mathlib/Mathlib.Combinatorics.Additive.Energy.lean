/-- The multiplicative energy `Eₘ[s, t]` of two finsets `s` and `t` in a group is the number of
quadruples `(a₁, a₂, b₁, b₂) ∈ s × s × t × t` such that `a₁ * b₁ = a₂ * b₂`.

The notation `Eₘ[s, t]` is available in scope `Combinatorics.Additive`. -/
@[to_additive "The additive energy `E[s, t]` of two finsets `s` and `t` in a group is the number of
quadruples `(a₁, a₂, b₁, b₂) ∈ s × s × t × t` such that `a₁ + b₁ = a₂ + b₂`.

The notation `E[s, t]` is available in scope `Combinatorics.Additive`."]
def mulEnergy (s t : Finset α) : ℕ :=
  (((s ×ˢ s) ×ˢ t ×ˢ t).filter fun x : (α × α) × α × α => x.1.1 * x.2.1 = x.1.2 * x.2.2).card


/-- The multiplicative energy of two finsets `s` and `t` in a group is the number of quadruples
`(a₁, a₂, b₁, b₂) ∈ s × s × t × t` such that `a₁ * b₁ = a₂ * b₂`. -/
scoped[Combinatorics.Additive] notation3:max "Eₘ[" s ", " t "]" => Finset.mulEnergy s t


/-- The additive energy of two finsets `s` and `t` in a group is the number of quadruples
`(a₁, a₂, b₁, b₂) ∈ s × s × t × t` such that `a₁ + b₁ = a₂ + b₂`.-/
scoped[Combinatorics.Additive] notation3:max "E[" s ", " t "]" => Finset.addEnergy s t


/-- The multiplicative energy of a finset `s` in a group is the number of quadruples
`(a₁, a₂, b₁, b₂) ∈ s × s × s × s` such that `a₁ * b₁ = a₂ * b₂`. -/
scoped[Combinatorics.Additive] notation3:max "Eₘ[" s "]" => Finset.mulEnergy s s


/-- The additive energy of a finset `s` in a group is the number of quadruples
`(a₁, a₂, b₁, b₂) ∈ s × s × s × s` such that `a₁ + b₁ = a₂ + b₂`. -/
scoped[Combinatorics.Additive] notation3:max "E[" s "]" => Finset.addEnergy s s


@[to_additive (attr := gcongr)]
lemma mulEnergy_mono (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) : Eₘ[s₁, t₁] ≤ Eₘ[s₂, t₂] := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s₁ s₂ t₁ t₂ : Finset α
    hs : HasSubset.Subset s₁ s₂
    ht : HasSubset.Subset t₁ t₂
    ⊢ LE.le (s₁.mulEnergy t₁) (s₂.mulEnergy t₂)
  -/
  unfold mulEnergy; gcongr
                    /-
                      🎉 no goals
                    -/


@[to_additive] lemma mulEnergy_mono_left (hs : s₁ ⊆ s₂) : Eₘ[s₁, t] ≤ Eₘ[s₂, t] :=
  mulEnergy_mono hs Subset.rfl


@[to_additive] lemma mulEnergy_mono_right (ht : t₁ ⊆ t₂) : Eₘ[s, t₁] ≤ Eₘ[s, t₂] :=
  mulEnergy_mono Subset.rfl ht


@[to_additive] lemma le_mulEnergy : s.card * t.card ≤ Eₘ[s, t] := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    ⊢ LE.le (HMul.hMul s.card t.card) (s.mulEnergy t)
  -/
  rw [← card_product]
  refine
    card_le_card_of_injOn (@fun x => ((x.1, x.1), x.2, x.2)) (by
    -- Porting note: changed this from a `simp` proof without `only` because of a timeout
      simp only [← and_imp, mem_product, Prod.forall, mem_filter, and_self, and_true, imp_self,
        implies_true]) fun a _ b _ => ?_
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    a : Prod α α
    x✝¹ : Membership.mem (↑(SProd.sprod s t)) a
    b : Prod α α
    x✝ : Membership.mem (↑(SProd.sprod s t)) b
    ⊢ Eq ((fun x => { fst := { fst := x.1, snd := x.1 }, snd := { fst := x.2, snd  …
  -/
  simp only [Prod.mk.inj_iff, and_self_iff, and_imp]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    a : Prod α α
    x✝¹ : Membership.mem (↑(SProd.sprod s t)) a
    b : Prod α α
    x✝ : Membership.mem (↑(SProd.sprod s t)) b
    ⊢ Eq a.1 b.1 → Eq a.2 b.2 → Eq a b
  -/
  exact Prod.ext
  /-
    🎉 no goals
  -/


@[to_additive] lemma mulEnergy_pos (hs : s.Nonempty) (ht : t.Nonempty) : 0 < Eₘ[s, t] :=
  (mul_pos hs.card_pos ht.card_pos).trans_le le_mulEnergy


                                                                              /-
                                                                                α : Type u_1
                                                                                inst✝¹ : DecidableEq α
                                                                                inst✝ : Mul α
                                                                                t : Finset α
                                                                                ⊢ Eq (EmptyCollection.emptyCollection.mulEnergy t) 0
                                                                              -/
@[to_additive (attr := simp)] lemma mulEnergy_empty_left : Eₘ[∅, t] = 0 := by simp [mulEnergy]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝¹ : DecidableEq α
                                                                                 inst✝ : Mul α
                                                                                 s : Finset α
                                                                                 ⊢ Eq (s.mulEnergy EmptyCollection.emptyCollection) 0
                                                                               -/
@[to_additive (attr := simp)] lemma mulEnergy_empty_right : Eₘ[s, ∅] = 0 := by simp [mulEnergy]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[to_additive (attr := simp)] lemma mulEnergy_pos_iff : 0 < Eₘ[s, t] ↔ s.Nonempty ∧ t.Nonempty where
  mp h := of_not_not fun H => by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      h : LT.lt 0 (s.mulEnergy t)
      H : Not (And s.Nonempty t.Nonempty)
      ⊢ False
    -/
    simp_rw [not_and_or, not_nonempty_iff_eq_empty] at H
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      h : LT.lt 0 (s.mulEnergy t)
      H : Or (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyColle …
      ⊢ False
    -/
                              /-
                                🎉 no goals
                              -/
    obtain rfl | rfl := H <;> simp [Nat.not_lt_zero] at h
                              /-
                                🎉 no goals
                              -/
  mpr h := mulEnergy_pos h.1 h.2


@[to_additive (attr := simp)] lemma mulEnergy_eq_zero_iff : Eₘ[s, t] = 0 ↔ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    ⊢ Iff (Eq (s.mulEnergy t) 0) (Or (Eq s EmptyCollection.emptyCollection) (Eq t  …
  -/
  simp [← (Nat.zero_le _).not_gt_iff_eq, not_and_or, imp_iff_or_not, or_comm]
  /-
    🎉 no goals
  -/


@[to_additive] lemma mulEnergy_eq_card_filter (s t : Finset α) :
    Eₘ[s, t] = (((s ×ˢ t) ×ˢ s ×ˢ t).filter fun ((a, b), c, d) ↦ a * b = c * d).card :=
                                             /-
                                               α : Type u_1
                                               inst✝¹ : DecidableEq α
                                               inst✝ : Mul α
                                               s t : Finset α
                                               ⊢ ∀ (i : Prod (Prod α α) (Prod α α)), Iff (Membership.mem (Finset.filter (fun  …
                                             -/
  card_equiv (.prodProdProdComm _ _ _ _) (by simp [and_and_and_comm])
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive] lemma mulEnergy_eq_sum_sq' (s t : Finset α) :
    Eₘ[s, t] = ∑ a ∈ s * t, ((s ×ˢ t).filter fun (x, y) ↦ x * y = a).card ^ 2 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    ⊢ Eq (s.mulEnergy t) ((HMul.hMul s t).sum fun a => HPow.hPow (Finset.filter (f …
  -/
  simp_rw [mulEnergy_eq_card_filter, sq, ← card_product]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.1.2) (HMul.hMul x.2.1 x.2. …
  -/
  rw [← card_disjiUnion]
  -- The `swap`, `ext` and `simp` calls significantly reduce heartbeats
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Mul α
    s t : Finset α
    ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.1.2) (HMul.hMul x.2.1 x.2. …
  -/
  swap
  · simp only [Set.PairwiseDisjoint, Set.Pairwise, coe_mul, ne_eq, disjoint_left, mem_product,
      mem_filter, not_and, and_imp, Prod.forall]
    /-
      case h
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      ⊢ ∀ ⦃x : α⦄, Membership.mem (HMul.hMul ↑s ↑t) x → ∀ ⦃y : α⦄, Membership.mem (H …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.1.2) (HMul.hMul x.2.1 x.2. …
    -/
  · congr
    /-
      case e_s
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.1.2) (HMul.hMul x.2.1 x.2. …
    -/
    ext
    /-
      case e_s.h
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      a✝ : Prod (Prod α α) (Prod α α)
      ⊢ Iff (Membership.mem (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.1.2) (HMu …
    -/
    simp only [mem_filter, mem_product, disjiUnion_eq_biUnion, mem_biUnion]
    /-
      case e_s.h
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Mul α
      s t : Finset α
      a✝ : Prod (Prod α α) (Prod α α)
      ⊢ Iff (And (And (And (Membership.mem s a✝.1.1) (Membership.mem t a✝.1.2)) (And …
    -/
    aesop (add unsafe mul_mem_mul)
    /-
      🎉 no goals
    -/


@[to_additive] lemma mulEnergy_eq_sum_sq [Fintype α] (s t : Finset α) :
    Eₘ[s, t] = ∑ a, ((s ×ˢ t).filter fun (x, y) ↦ x * y = a).card ^ 2 := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Mul α
    inst✝ : Fintype α
    s t : Finset α
    ⊢ Eq (s.mulEnergy t) (Finset.univ.sum fun a => HPow.hPow (Finset.filter (fun x …
  -/
  rw [mulEnergy_eq_sum_sq']
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Mul α
    inst✝ : Fintype α
    s t : Finset α
    ⊢ Eq ((HMul.hMul s t).sum fun a => HPow.hPow (Finset.filter (fun x => Finset.m …
  -/
  exact Fintype.sum_subset <| by aesop (add simp [filter_eq_empty_iff, mul_mem_mul])
  /-
    🎉 no goals
  -/


@[to_additive card_sq_le_card_mul_addEnergy]
lemma card_sq_le_card_mul_mulEnergy (s t u : Finset α) :
    ((s ×ˢ t).filter fun (a, b) ↦ a * b ∈ u).card ^ 2 ≤ u.card * Eₘ[s, t] := by
  calc
    _ = (∑ c ∈ u, ((s ×ˢ t).filter fun (a, b) ↦ a * b = c).card) ^ 2 := by
        rw [← sum_card_fiberwise_eq_card_filter]
    _ ≤ u.card * ∑ c ∈ u, ((s ×ˢ t).filter fun (a, b) ↦ a * b = c).card ^ 2 := by
        simpa using sum_mul_sq_le_sq_mul_sq (R := ℕ) _ 1 _
    _ ≤ u.card * ∑ c ∈ s * t, ((s ×ˢ t).filter fun (a, b) ↦ a * b = c).card ^ 2 := by
        refine mul_le_mul_left' (sum_le_sum_of_ne_zero ?_) _
        aesop (add simp [filter_eq_empty_iff]) (add unsafe mul_mem_mul)
    _ = u.card * Eₘ[s, t] := by rw [mulEnergy_eq_sum_sq']


@[to_additive le_card_add_mul_addEnergy] lemma le_card_add_mul_mulEnergy (s t : Finset α) :
    s.card ^ 2 * t.card ^ 2 ≤ (s * t).card * Eₘ[s, t] :=
  calc
    _ = ((s ×ˢ t).filter fun (a, b) ↦ a * b ∈ s * t).card ^ 2 := by
      /-
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Mul α
        s t : Finset α
        ⊢ Eq (HMul.hMul (HPow.hPow s.card 2) (HPow.hPow t.card 2)) (HPow.hPow (Finset. …
      -/
      rw [filter_eq_self.2, card_product, mul_pow]; aesop (add unsafe mul_mem_mul)
                                                    /-
                                                      🎉 no goals
                                                    -/
    _ ≤ (s * t).card * Eₘ[s, t] := card_sq_le_card_mul_mulEnergy _ _ _


@[to_additive] lemma mulEnergy_comm (s t : Finset α) : Eₘ[s, t] = Eₘ[t, s] := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    s t : Finset α
    ⊢ Eq (s.mulEnergy t) (t.mulEnergy s)
  -/
  rw [mulEnergy, ← Finset.card_map (Equiv.prodComm _ _).toEmbedding, map_filter]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    s t : Finset α
    ⊢ Eq (Finset.filter (Function.comp (fun x => Eq (HMul.hMul x.1.1 x.2.1) (HMul. …
  -/
  simp [-Finset.card_map, eq_comm, mulEnergy, mul_comm, map_eq_image, Function.comp_def]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma mulEnergy_univ_left : Eₘ[univ, t] = Fintype.card α * t.card ^ 2 := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    ⊢ Eq (Finset.univ.mulEnergy t) (HMul.hMul (Fintype.card α) (HPow.hPow t.card 2))
  -/
  simp only [mulEnergy, univ_product_univ, Fintype.card, sq, ← card_product]
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.2.1) (HMul.hMul x.1.2 x.2. …
  -/
  let f : α × α × α → (α × α) × α × α := fun x => ((x.1 * x.2.2, x.1 * x.2.1), x.2)
  have : (↑((univ : Finset α) ×ˢ t ×ˢ t) : Set (α × α × α)).InjOn f := by
    rintro ⟨a₁, b₁, c₁⟩ _ ⟨a₂, b₂, c₂⟩ h₂ h
    simp_rw [f, Prod.ext_iff] at h
    obtain ⟨h, rfl, rfl⟩ := h
    rw [mul_right_cancel h.1]
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    f : Prod α (Prod α α) → Prod (Prod α α) (Prod α α) := fun x => { fst := { fst  …
    this : Set.InjOn f ↑(SProd.sprod Finset.univ (SProd.sprod t t))
    ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.2.1) (HMul.hMul x.1.2 x.2. …
  -/
  rw [← card_image_of_injOn this]
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    f : Prod α (Prod α α) → Prod (Prod α α) (Prod α α) := fun x => { fst := { fst  …
    this : Set.InjOn f ↑(SProd.sprod Finset.univ (SProd.sprod t t))
    ⊢ Eq (Finset.filter (fun x => Eq (HMul.hMul x.1.1 x.2.1) (HMul.hMul x.1.2 x.2. …
  -/
  congr with a
  simp only [mem_filter, mem_product, mem_univ, true_and, mem_image, exists_prop,
    Prod.exists]
  /-
    case e_s.h
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    f : Prod α (Prod α α) → Prod (Prod α α) (Prod α α) := fun x => { fst := { fst  …
    this : Set.InjOn f ↑(SProd.sprod Finset.univ (SProd.sprod t t))
    a : Prod (Prod α α) (Prod α α)
    ⊢ Iff (And (And (Membership.mem t a.2.1) (Membership.mem t a.2.2)) (Eq (HMul.h …
  -/
  refine ⟨fun h => ⟨a.1.1 * a.2.2⁻¹, _, _, h.1, by simp [f, mul_right_comm, h.2]⟩, ?_⟩
  /-
    case e_s.h
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    f : Prod α (Prod α α) → Prod (Prod α α) (Prod α α) := fun x => { fst := { fst  …
    this : Set.InjOn f ↑(SProd.sprod Finset.univ (SProd.sprod t t))
    a : Prod (Prod α α) (Prod α α)
    ⊢ (Exists fun a_1 => Exists fun a_2 => Exists fun b => And (And (Membership.me …
  -/
  rintro ⟨b, c, d, hcd, rfl⟩
  /-
    case e_s.h.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    t : Finset α
    f : Prod α (Prod α α) → Prod (Prod α α) (Prod α α) := fun x => { fst := { fst  …
    this : Set.InjOn f ↑(SProd.sprod Finset.univ (SProd.sprod t t))
    b c d : α
    hcd : And (Membership.mem t c) (Membership.mem t d)
    ⊢ And (And (Membership.mem t (f { fst := b, snd := { fst := c, snd := d } }).2 …
  -/
  simpa [f, mul_right_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma mulEnergy_univ_right : Eₘ[s, univ] = Fintype.card α * s.card ^ 2 := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CommGroup α
    inst✝ : Fintype α
    s : Finset α
    ⊢ Eq (s.mulEnergy Finset.univ) (HMul.hMul (Fintype.card α) (HPow.hPow s.card 2))
  -/
  rw [mulEnergy_comm, mulEnergy_univ_left]
  /-
    🎉 no goals
  -/


