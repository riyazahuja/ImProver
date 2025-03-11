@[norm_cast]
theorem coe_list_sum (l : List ℚ≥0) : (l.sum : ℚ) = (l.map (↑)).sum :=
  map_list_sum coeHom _


@[norm_cast]
theorem coe_list_prod (l : List ℚ≥0) : (l.prod : ℚ) = (l.map (↑)).prod :=
  map_list_prod coeHom _


@[norm_cast]
theorem coe_multiset_sum (s : Multiset ℚ≥0) : (s.sum : ℚ) = (s.map (↑)).sum :=
  map_multiset_sum coeHom _


@[norm_cast]
theorem coe_multiset_prod (s : Multiset ℚ≥0) : (s.prod : ℚ) = (s.map (↑)).prod :=
  map_multiset_prod coeHom _


@[norm_cast]
theorem coe_sum {s : Finset α} {f : α → ℚ≥0} : ↑(∑ a ∈ s, f a) = ∑ a ∈ s, (f a : ℚ) :=
  map_sum coeHom _ _


theorem toNNRat_sum_of_nonneg {s : Finset α} {f : α → ℚ} (hf : ∀ a, a ∈ s → 0 ≤ f a) :
    (∑ a ∈ s, f a).toNNRat = ∑ a ∈ s, (f a).toNNRat := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Rat
    hf : ∀ (a : α), Membership.mem s a → LE.le 0 (f a)
    ⊢ Eq (s.sum fun a => f a).toNNRat (s.sum fun a => (f a).toNNRat)
  -/
  rw [← coe_inj, coe_sum, Rat.coe_toNNRat _ (Finset.sum_nonneg hf)]
  /-
    α : Type u_1
    s : Finset α
    f : α → Rat
    hf : ∀ (a : α), Membership.mem s a → LE.le 0 (f a)
    ⊢ Eq (s.sum fun i => f i) (s.sum fun a => ↑(f a).toNNRat)
  -/
  exact Finset.sum_congr rfl fun x hxs ↦ by rw [Rat.coe_toNNRat _ (hf x hxs)]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_prod {s : Finset α} {f : α → ℚ≥0} : ↑(∏ a ∈ s, f a) = ∏ a ∈ s, (f a : ℚ) :=
  map_prod coeHom _ _


theorem toNNRat_prod_of_nonneg {s : Finset α} {f : α → ℚ} (hf : ∀ a ∈ s, 0 ≤ f a) :
    (∏ a ∈ s, f a).toNNRat = ∏ a ∈ s, (f a).toNNRat := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Rat
    hf : ∀ (a : α), Membership.mem s a → LE.le 0 (f a)
    ⊢ Eq (s.prod fun a => f a).toNNRat (s.prod fun a => (f a).toNNRat)
  -/
  rw [← coe_inj, coe_prod, Rat.coe_toNNRat _ (Finset.prod_nonneg hf)]
  /-
    α : Type u_1
    s : Finset α
    f : α → Rat
    hf : ∀ (a : α), Membership.mem s a → LE.le 0 (f a)
    ⊢ Eq (s.prod fun i => f i) (s.prod fun a => ↑(f a).toNNRat)
  -/
  exact Finset.prod_congr rfl fun x hxs ↦ by rw [Rat.coe_toNNRat _ (hf x hxs)]
  /-
    🎉 no goals
  -/


