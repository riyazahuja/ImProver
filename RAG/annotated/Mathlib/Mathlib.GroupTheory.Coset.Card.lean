@[to_additive AddSubgroup.card_eq_card_quotient_mul_card_addSubgroup]
theorem card_eq_card_quotient_mul_card_subgroup (s : Subgroup α) :
    Nat.card α = Nat.card (α ⧸ s) * Nat.card s := by
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    ⊢ Eq (Nat.card α) (HMul.hMul (Nat.card (HasQuotient.Quotient α s)) (Nat.card ( …
  -/
  rw [← Nat.card_prod]; exact Nat.card_congr Subgroup.groupEquivQuotientProdSubgroup
                        /-
                          🎉 no goals
                        -/


@[to_additive]
lemma card_mul_eq_card_subgroup_mul_card_quotient (s : Subgroup α) (t : Set α) :
    Nat.card (t * s : Set α) = Nat.card s * Nat.card (t.image (↑) : Set (α ⧸ s)) := by
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    t : Set α
    ⊢ Eq (Nat.card ↑(HMul.hMul t ↑s)) (HMul.hMul (Nat.card (Subtype fun x => Membe …
  -/
  rw [← Nat.card_prod, Nat.card_congr]
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    t : Set α
    ⊢ Equiv (↑(HMul.hMul t ↑s)) (Prod (Subtype fun x => Membership.mem s x) ↑(Set. …
  -/
  apply Equiv.trans _ (QuotientGroup.preimageMkEquivSubgroupProdSet _ _)
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    t : Set α
    ⊢ Equiv ↑(HMul.hMul t ↑s) ↑(Set.preimage QuotientGroup.mk (Set.image QuotientG …
  -/
  rw [QuotientGroup.preimage_image_mk]
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    t : Set α
    ⊢ Equiv ↑(HMul.hMul t ↑s) ↑(Set.iUnion fun x => Set.preimage (fun x_1 => HMul. …
  -/
  convert Equiv.refl ↑(t * s)
  /-
    case h.e'_2.h.e'_2
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    t : Set α
    ⊢ Eq (Set.iUnion fun x => Set.preimage (fun x_1 => HMul.hMul x_1 ↑x) t) (HMul. …
  -/
  aesop (add simp [Set.mem_mul])
  /-
    🎉 no goals
  -/


/-- **Lagrange's Theorem**: The order of a subgroup divides the order of its ambient group. -/
@[to_additive "**Lagrange's Theorem**: The order of an additive subgroup divides the order of its
 ambient additive group."]
theorem card_subgroup_dvd_card (s : Subgroup α) : Nat.card s ∣ Nat.card α := by
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem s x)) (Nat.card α)
  -/
  classical simp [card_eq_card_quotient_mul_card_subgroup s, @dvd_mul_left ℕ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem card_quotient_dvd_card (s : Subgroup α) : Nat.card (α ⧸ s) ∣ Nat.card α := by
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    ⊢ Dvd.dvd (Nat.card (HasQuotient.Quotient α s)) (Nat.card α)
  -/
  simp [card_eq_card_quotient_mul_card_subgroup s, @dvd_mul_right ℕ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem card_dvd_of_injective (f : α →* H) (hf : Function.Injective f) :
    Nat.card α ∣ Nat.card H := by
  classical calc
      Nat.card α = Nat.card (f.range : Subgroup H) := Nat.card_congr (Equiv.ofInjective f hf)
      _ ∣ Nat.card H := card_subgroup_dvd_card _


@[to_additive]
theorem card_dvd_of_le {H K : Subgroup α} (hHK : H ≤ K) : Nat.card H ∣ Nat.card K :=
  card_dvd_of_injective (inclusion hHK) (inclusion_injective hHK)


@[to_additive]
theorem card_comap_dvd_of_injective (K : Subgroup H) (f : α →* H)
    (hf : Function.Injective f) : Nat.card (K.comap f) ∣ Nat.card K :=
  calc Nat.card (K.comap f) = Nat.card ((K.comap f).map f) :=
      Nat.card_congr (equivMapOfInjective _ _ hf).toEquiv
    _ ∣ Nat.card K := card_dvd_of_le (map_comap_le _ _)


