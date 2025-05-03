/-- `Nat.card α` is the cardinality of `α` as a natural number.
  If `α` is infinite, `Nat.card α = 0`. -/
protected def card (α : Type*) : ℕ :=
  toNat (mk α)


@[simp]
theorem card_eq_fintype_card [Fintype α] : Nat.card α = Fintype.card α :=
  mk_toNat_eq_card


/-- Because this theorem takes `Fintype α` as a non-instance argument, it can be used in particular
when `Fintype.card` ends up with different instance than the one found by inference  -/
theorem _root_.Fintype.card_eq_nat_card {_ : Fintype α} : Fintype.card α = Nat.card α :=
  mk_toNat_eq_card.symm


lemma card_eq_finsetCard (s : Finset α) : Nat.card s = s.card := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem s x)) s.card
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.card_coe]
  /-
    🎉 no goals
  -/


lemma card_eq_card_toFinset (s : Set α) [Fintype s] : Nat.card s = s.toFinset.card := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    ⊢ Eq (Nat.card ↑s) s.toFinset.card
  -/
  simp only [← Nat.card_eq_finsetCard, s.mem_toFinset]
  /-
    🎉 no goals
  -/


lemma card_eq_card_finite_toFinset {s : Set α} (hs : s.Finite) : Nat.card s = hs.toFinset.card := by
  /-
    α : Type u_1
    s : Set α
    hs : s.Finite
    ⊢ Eq (Nat.card ↑s) hs.toFinset.card
  -/
  simp only [← Nat.card_eq_finsetCard, hs.mem_toFinset]
  /-
    🎉 no goals
  -/


                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : IsEmpty α
                                                                     ⊢ Eq (Nat.card α) 0
                                                                   -/
@[simp] theorem card_of_isEmpty [IsEmpty α] : Nat.card α = 0 := by simp [Nat.card]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp] lemma card_eq_zero_of_infinite [Infinite α] : Nat.card α = 0 := mk_toNat_of_infinite


lemma cast_card [Finite α] : (Nat.card α : Cardinal) = Cardinal.mk α := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Eq (↑(Nat.card α)) (Cardinal.mk α)
  -/
  rw [Nat.card, Cardinal.cast_toNat_of_lt_aleph0]
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ LT.lt (Cardinal.mk α) Cardinal.aleph0
  -/
  exact Cardinal.lt_aleph0_of_finite _
  /-
    🎉 no goals
  -/


lemma _root_.Set.Infinite.card_eq_zero {s : Set α} (hs : s.Infinite) : Nat.card s = 0 :=
  @card_eq_zero_of_infinite _ hs.to_subtype


lemma card_eq_zero : Nat.card α = 0 ↔ IsEmpty α ∨ Infinite α := by
  /-
    α : Type u_1
    ⊢ Iff (Eq (Nat.card α) 0) (Or (IsEmpty α) (Infinite α))
  -/
  simp [Nat.card, mk_eq_zero_iff, aleph0_le_mk_iff]
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    α : Type u_1
                                                                    ⊢ Iff (Ne (Nat.card α) 0) (And (Nonempty α) (Finite α))
                                                                  -/
lemma card_ne_zero : Nat.card α ≠ 0 ↔ Nonempty α ∧ Finite α := by simp [card_eq_zero, not_or]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma card_pos_iff : 0 < Nat.card α ↔ Nonempty α ∧ Finite α := by
  /-
    α : Type u_1
    ⊢ Iff (LT.lt 0 (Nat.card α)) (And (Nonempty α) (Finite α))
  -/
  simp [Nat.card, mk_eq_zero_iff, mk_lt_aleph0_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma card_pos [Nonempty α] [Finite α] : 0 < Nat.card α := card_pos_iff.2 ⟨‹_›, ‹_›⟩


theorem finite_of_card_ne_zero (h : Nat.card α ≠ 0) : Finite α := (card_ne_zero.1 h).2


theorem card_congr (f : α ≃ β) : Nat.card α = Nat.card β :=
  Cardinal.toNat_congr f


lemma card_le_card_of_injective {α : Type u} {β : Type v} [Finite β] (f : α → β)
    (hf : Injective f) : Nat.card α ≤ Nat.card β := by
  /-
    α : Type u
    β : Type v
    inst✝ : Finite β
    f : α → β
    hf : Function.Injective f
    ⊢ LE.le (Nat.card α) (Nat.card β)
  -/
  simpa using toNat_le_toNat (lift_mk_le_lift_mk_of_injective hf) (by simp [lt_aleph0_of_finite])
  /-
    🎉 no goals
  -/


lemma card_le_card_of_surjective {α : Type u} {β : Type v} [Finite α] (f : α → β)
    (hf : Surjective f) : Nat.card β ≤ Nat.card α := by
  /-
    α : Type u
    β : Type v
    inst✝ : Finite α
    f : α → β
    hf : Function.Surjective f
    ⊢ LE.le (Nat.card β) (Nat.card α)
  -/
  have : lift.{u} #β ≤ lift.{v} #α := mk_le_of_surjective (ULift.map_surjective.2 hf)
  /-
    α : Type u
    β : Type v
    inst✝ : Finite α
    f : α → β
    hf : Function.Surjective f
    this : LE.le (Cardinal.lift.{u, v} (Cardinal.mk β)) (Cardinal.lift.{v, u} (Car …
    ⊢ LE.le (Nat.card β) (Nat.card α)
  -/
  simpa using toNat_le_toNat this (by simp [lt_aleph0_of_finite])
  /-
    🎉 no goals
  -/


theorem card_eq_of_bijective (f : α → β) (hf : Function.Bijective f) : Nat.card α = Nat.card β :=
  card_congr (Equiv.ofBijective f hf)


protected theorem bijective_iff_injective_and_card [Finite β] (f : α → β) :
    Bijective f ↔ Injective f ∧ Nat.card α = Nat.card β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    ⊢ Iff (Function.Bijective f) (And (Function.Injective f) (Eq (Nat.card α) (Nat …
  -/
  rw [Bijective, and_congr_right_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    ⊢ Function.Injective f → Iff (Function.Surjective f) (Eq (Nat.card α) (Nat.car …
  -/
  intro h
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    h : Function.Injective f
    ⊢ Iff (Function.Surjective f) (Eq (Nat.card α) (Nat.card β))
  -/
  have := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    h : Function.Injective f
    this : Fintype β
    ⊢ Iff (Function.Surjective f) (Eq (Nat.card α) (Nat.card β))
  -/
  have := Fintype.ofInjective f h
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    h : Function.Injective f
    this✝ : Fintype β
    this : Fintype α
    ⊢ Iff (Function.Surjective f) (Eq (Nat.card α) (Nat.card β))
  -/
  revert h
  rw [← and_congr_right_iff, ← Bijective,
    card_eq_fintype_card, card_eq_fintype_card, Fintype.bijective_iff_injective_and_card]


protected theorem bijective_iff_surjective_and_card [Finite α] (f : α → β) :
    Bijective f ↔ Surjective f ∧ Nat.card α = Nat.card β := by
  classical
  rw [_root_.and_comm, Bijective, and_congr_left_iff]
  intro h
  have := Fintype.ofFinite α
  have := Fintype.ofSurjective f h
  revert h
  rw [← and_congr_left_iff, ← Bijective, ← and_comm,
    card_eq_fintype_card, card_eq_fintype_card, Fintype.bijective_iff_surjective_and_card]


theorem _root_.Function.Injective.bijective_of_nat_card_le [Finite β] {f : α → β}
    (inj : Injective f) (hc : Nat.card β ≤ Nat.card α) : Bijective f :=
  (Nat.bijective_iff_injective_and_card f).mpr
    ⟨inj, hc.antisymm (card_le_card_of_injective f inj) |>.symm⟩


theorem _root_.Function.Surjective.bijective_of_nat_card_le [Finite α] {f : α → β}
    (surj : Surjective f) (hc : Nat.card α ≤ Nat.card β) : Bijective f :=
  (Nat.bijective_iff_surjective_and_card f).mpr
    ⟨surj, hc.antisymm (card_le_card_of_surjective f surj)⟩


theorem card_eq_of_equiv_fin {α : Type*} {n : ℕ} (f : α ≃ Fin n) : Nat.card α = n := by
  /-
    α : Type u_3
    n : Nat
    f : Equiv α (Fin n)
    ⊢ Eq (Nat.card α) n
  -/
  simpa only [card_eq_fintype_card, Fintype.card_fin] using card_congr f
  /-
    🎉 no goals
  -/


lemma card_mono (ht : t.Finite) (h : s ⊆ t) : Nat.card s ≤ Nat.card t :=
  toNat_le_toNat (mk_le_mk_of_subset h) ht.lt_aleph0


lemma card_image_le {f : α → β} (hs : s.Finite) : Nat.card (f '' s) ≤ Nat.card s :=
  have := hs.to_subtype; card_le_card_of_surjective (imageFactorization f s) surjective_onto_image


lemma card_image_of_injOn {f : α → β} (hf : s.InjOn f) : Nat.card (f '' s) = Nat.card s := by
  classical
  obtain hs | hs := s.finite_or_infinite
  · have := hs.fintype
    have := fintypeImage s f
    simp_rw [Nat.card_eq_fintype_card, Set.card_image_of_inj_on hf]
  · have := hs.to_subtype
    have := (hs.image hf).to_subtype
    simp [Nat.card_eq_zero_of_infinite]


lemma card_image_of_injective {f : α → β} (hf : Injective f) (s : Set α) :
    Nat.card (f '' s) = Nat.card s := card_image_of_injOn hf.injOn


lemma card_image_equiv (e : α ≃ β) : Nat.card (e '' s) = Nat.card s :=
    Nat.card_congr (e.image s).symm


lemma card_preimage_of_injOn {f : α → β} {s : Set β} (hf : (f ⁻¹' s).InjOn f) (hsf : s ⊆ range f) :
    Nat.card (f ⁻¹' s) = Nat.card s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    hf : Set.InjOn f (Set.preimage f s)
    hsf : HasSubset.Subset s (Set.range f)
    ⊢ Eq (Nat.card ↑(Set.preimage f s)) (Nat.card ↑s)
  -/
  rw [← Nat.card_image_of_injOn hf, image_preimage_eq_iff.2 hsf]
  /-
    🎉 no goals
  -/


lemma card_preimage_of_injective {f : α → β} {s : Set β} (hf : Injective f) (hsf : s ⊆ range f) :
    Nat.card (f ⁻¹' s) = Nat.card s := card_preimage_of_injOn hf.injOn hsf


@[simp] lemma card_univ : Nat.card (univ : Set α) = Nat.card α :=
  card_congr (Equiv.Set.univ α)


lemma card_range_of_injective {f : α → β} (hf : Injective f) :
    Nat.card (range f) = Nat.card α := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    ⊢ Eq (Nat.card ↑(Set.range f)) (Nat.card α)
  -/
  rw [← Nat.card_preimage_of_injective hf le_rfl]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    ⊢ Eq (Nat.card ↑(Set.preimage f (Set.range f))) (Nat.card α)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If the cardinality is positive, that means it is a finite type, so there is
an equivalence between `α` and `Fin (Nat.card α)`. See also `Finite.equivFin`. -/
def equivFinOfCardPos {α : Type*} (h : Nat.card α ≠ 0) : α ≃ Fin (Nat.card α) := by
  /-
    α✝ : Type u_1
    β : Type u_2
    α : Type u_3
    h : Ne (Nat.card α) 0
    ⊢ Equiv α (Fin (Nat.card α))
  -/
  cases fintypeOrInfinite α
    /-
      case inl
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      h : Ne (Nat.card α) 0
      val✝ : Fintype α
      ⊢ Equiv α (Fin (Nat.card α))
    -/
  · simpa only [card_eq_fintype_card] using Fintype.equivFin α
    /-
      🎉 no goals
    -/
    /-
      case inr
      α✝ : Type u_1
      β : Type u_2
      α : Type u_3
      h : Ne (Nat.card α) 0
      val✝ : Infinite α
      ⊢ Equiv α (Fin (Nat.card α))
    -/
  · simp only [card_eq_zero_of_infinite, ne_eq, not_true_eq_false] at h
    /-
      🎉 no goals
    -/


theorem card_of_subsingleton (a : α) [Subsingleton α] : Nat.card α = 1 := by
  /-
    α : Type u_1
    a : α
    inst✝ : Subsingleton α
    ⊢ Eq (Nat.card α) 1
  -/
  letI := Fintype.ofSubsingleton a
  /-
    α : Type u_1
    a : α
    inst✝ : Subsingleton α
    this : Fintype α := Fintype.ofSubsingleton a
    ⊢ Eq (Nat.card α) 1
  -/
  rw [card_eq_fintype_card, Fintype.card_ofSubsingleton a]
  /-
    🎉 no goals
  -/


theorem card_eq_one_iff_unique : Nat.card α = 1 ↔ Subsingleton α ∧ Nonempty α :=
  Cardinal.toNat_eq_one_iff_unique


@[simp]
theorem card_unique [Nonempty α] [Subsingleton α] : Nat.card α = 1 := by
  /-
    α : Type u_1
    inst✝¹ : Nonempty α
    inst✝ : Subsingleton α
    ⊢ Eq (Nat.card α) 1
  -/
  simp [card_eq_one_iff_unique, *]
  /-
    🎉 no goals
  -/


theorem card_eq_one_iff_exists : Nat.card α = 1 ↔ ∃ x : α, ∀ y : α, y = x := by
  /-
    α : Type u_1
    ⊢ Iff (Eq (Nat.card α) 1) (Exists fun x => ∀ (y : α), Eq y x)
  -/
  rw [card_eq_one_iff_unique]
  /-
    α : Type u_1
    ⊢ Iff (And (Subsingleton α) (Nonempty α)) (Exists fun x => ∀ (y : α), Eq y x)
  -/
  exact ⟨fun ⟨s, ⟨a⟩⟩ ↦ ⟨a, fun x ↦ s.elim x a⟩, fun ⟨x, h⟩ ↦ ⟨subsingleton_of_forall_eq x h, ⟨x⟩⟩⟩
  /-
    🎉 no goals
  -/


theorem card_eq_two_iff : Nat.card α = 2 ↔ ∃ x y : α, x ≠ y ∧ {x, y} = @Set.univ α :=
  toNat_eq_ofNat.trans mk_eq_two_iff


theorem card_eq_two_iff' (x : α) : Nat.card α = 2 ↔ ∃! y, y ≠ x :=
  toNat_eq_ofNat.trans (mk_eq_two_iff' x)


@[simp]
theorem card_sum [Finite α] [Finite β] : Nat.card (α ⊕ β) = Nat.card α + Nat.card β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    ⊢ Eq (Nat.card (Sum α β)) (HAdd.hAdd (Nat.card α) (Nat.card β))
  -/
  have := Fintype.ofFinite α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this : Fintype α
    ⊢ Eq (Nat.card (Sum α β)) (HAdd.hAdd (Nat.card α) (Nat.card β))
  -/
  have := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this✝ : Fintype α
    this : Fintype β
    ⊢ Eq (Nat.card (Sum α β)) (HAdd.hAdd (Nat.card α) (Nat.card β))
  -/
  simp_rw [Nat.card_eq_fintype_card, Fintype.card_sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_prod (α β : Type*) : Nat.card (α × β) = Nat.card α * Nat.card β := by
  /-
    α : Type u_3
    β : Type u_4
    ⊢ Eq (Nat.card (Prod α β)) (HMul.hMul (Nat.card α) (Nat.card β))
  -/
  simp only [Nat.card, mk_prod, toNat_mul, toNat_lift]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_ulift (α : Type*) : Nat.card (ULift α) = Nat.card α :=
  card_congr Equiv.ulift


@[simp]
theorem card_plift (α : Type*) : Nat.card (PLift α) = Nat.card α :=
  card_congr Equiv.plift


theorem card_pi {β : α → Type*} [Fintype α] : Nat.card (∀ a, β a) = ∏ a, Nat.card (β a) := by
  /-
    α : Type u_1
    β : α → Type u_3
    inst✝ : Fintype α
    ⊢ Eq (Nat.card ((a : α) → β a)) (Finset.univ.prod fun a => Nat.card (β a))
  -/
  simp_rw [Nat.card, mk_pi, prod_eq_of_fintype, toNat_lift, map_prod]
  /-
    🎉 no goals
  -/


theorem card_fun [Finite α] : Nat.card (α → β) = Nat.card β ^ Nat.card α := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite α
    ⊢ Eq (Nat.card (α → β)) (HPow.hPow (Nat.card β) (Nat.card α))
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite α
    this : Fintype α
    ⊢ Eq (Nat.card (α → β)) (HPow.hPow (Nat.card β) (Nat.card α))
  -/
  rw [Nat.card_pi, Finset.prod_const, Finset.card_univ, ← Nat.card_eq_fintype_card]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_zmod (n : ℕ) : Nat.card (ZMod n) = n := by
  /-
    n : Nat
    ⊢ Eq (Nat.card (ZMod n)) n
  -/
  cases n
    /-
      case zero
      ⊢ Eq (Nat.card (ZMod 0)) 0
    -/
  · exact @Nat.card_eq_zero_of_infinite _ Int.infinite
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (Nat.card (ZMod (HAdd.hAdd n✝ 1))) (HAdd.hAdd n✝ 1)
    -/
  · rw [Nat.card_eq_fintype_card, ZMod.card]
    /-
      🎉 no goals
    -/


lemma card_singleton_prod (a : α) (t : Set β) : Nat.card ({a} ×ˢ t) = Nat.card t := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    t : Set β
    ⊢ Eq (Nat.card ↑(SProd.sprod (Singleton.singleton a) t)) (Nat.card ↑t)
  -/
  rw [singleton_prod, Nat.card_image_of_injective (Prod.mk.inj_left a)]
  /-
    🎉 no goals
  -/


lemma card_prod_singleton (s : Set α) (b : β) : Nat.card (s ×ˢ {b}) = Nat.card s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    b : β
    ⊢ Eq (Nat.card ↑(SProd.sprod s (Singleton.singleton b))) (Nat.card ↑s)
  -/
  rw [prod_singleton, Nat.card_image_of_injective (Prod.mk.inj_right b)]
  /-
    🎉 no goals
  -/


theorem natCard_pos (hs : s.Finite) : 0 < Nat.card s ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    hs : s.Finite
    ⊢ Iff (LT.lt 0 (Nat.card ↑s)) s.Nonempty
  -/
  simp [pos_iff_ne_zero, Nat.card_eq_zero, hs.to_subtype, nonempty_iff_ne_empty]
  /-
    🎉 no goals
  -/


protected alias ⟨_, Nonempty.natCard_pos⟩ := natCard_pos


@[simp] lemma natCard_graphOn (s : Set α) (f : α → β) : Nat.card (s.graphOn f) = Nat.card s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    ⊢ Eq (Nat.card ↑(Set.graphOn f s)) (Nat.card ↑s)
  -/
  rw [← Nat.card_image_of_injOn fst_injOn_graph, image_fst_graphOn]
  /-
    🎉 no goals
  -/


/-- `ENat.card α` is the cardinality of `α` as an extended natural number.
  If `α` is infinite, `ENat.card α = ⊤`. -/
def card (α : Type*) : ℕ∞ :=
  toENat (mk α)


@[simp]
theorem card_eq_coe_fintype_card [Fintype α] : card α = Fintype.card α := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    ⊢ Eq (ENat.card α) ↑(Fintype.card α)
  -/
  simp [card]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_eq_top_of_infinite [Infinite α] : card α = ⊤ := by
  /-
    α : Type u_1
    inst✝ : Infinite α
    ⊢ Eq (ENat.card α) Top.top
  -/
  simp [card]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_sum (α β : Type*) :
    card (α ⊕ β) = card α + card β := by
  /-
    α : Type u_3
    β : Type u_4
    ⊢ Eq (ENat.card (Sum α β)) (HAdd.hAdd (ENat.card α) (ENat.card β))
  -/
  simp only [card, mk_sum, map_add, toENat_lift]
  /-
    🎉 no goals
  -/


theorem card_congr {α β : Type*} (f : α ≃ β) : card α = card β :=
  Cardinal.toENat_congr f


@[simp] lemma card_ulift (α : Type*) : card (ULift α) = card α := card_congr Equiv.ulift


@[simp] lemma card_plift (α : Type*) : card (PLift α) = card α := card_congr Equiv.plift


theorem card_image_of_injOn {α β : Type*} {f : α → β} {s : Set α} (h : Set.InjOn f s) :
    card (f '' s) = card s :=
  card_congr (Equiv.Set.imageOfInjOn f s h).symm


theorem card_image_of_injective {α β : Type*} (f : α → β) (s : Set α)
    (h : Function.Injective f) : card (f '' s) = card s := card_image_of_injOn h.injOn


@[simp]
theorem _root_.Cardinal.natCast_le_toENat_iff {n : ℕ} {c : Cardinal} :
    ↑n ≤ toENat c ↔ ↑n ≤ c := by
  /-
    n : Nat
    c : Cardinal.{u_3}
    ⊢ Iff (LE.le (↑n) (Cardinal.toENat c)) (LE.le (↑n) c)
  -/
  rw [← toENat_nat n, toENat_le_iff_of_le_aleph0 (le_of_lt (nat_lt_aleph0 n))]
  /-
    🎉 no goals
  -/


theorem _root_.Cardinal.toENat_le_natCast_iff {c : Cardinal} {n : ℕ} :
                               /-
                                 c : Cardinal.{u_3}
                                 n : Nat
                                 ⊢ Iff (LE.le (Cardinal.toENat c) ↑n) (LE.le c ↑n)
                               -/
    toENat c ≤ n ↔ c ≤ n := by simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem _root_.Cardinal.natCast_eq_toENat_iff {n : ℕ} {c : Cardinal} :
    ↑n = toENat c ↔ ↑n = c := by
  rw [le_antisymm_iff, le_antisymm_iff, Cardinal.toENat_le_natCast_iff,
    Cardinal.natCast_le_toENat_iff]


theorem _root_.Cardinal.toENat_eq_natCast_iff {c : Cardinal} {n : ℕ} :
                                        /-
                                          c : Cardinal.{u_3}
                                          n : Nat
                                          ⊢ Iff (Eq (Cardinal.toENat c) ↑n) (Eq c ↑n)
                                        -/
    Cardinal.toENat c = n ↔ c = n := by simp
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem _root_.Cardinal.natCast_lt_toENat_iff {n : ℕ} {c : Cardinal} :
    ↑n < toENat c ↔ ↑n < c := by
  /-
    n : Nat
    c : Cardinal.{u_3}
    ⊢ Iff (LT.lt (↑n) (Cardinal.toENat c)) (LT.lt (↑n) c)
  -/
  simp only [← not_le, Cardinal.toENat_le_natCast_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Cardinal.toENat_lt_natCast_iff {n : ℕ} {c : Cardinal} :
    toENat c < ↑n ↔ c < ↑n := by
  /-
    n : Nat
    c : Cardinal.{u_3}
    ⊢ Iff (LT.lt (Cardinal.toENat c) ↑n) (LT.lt c ↑n)
  -/
  simp only [← not_le, Cardinal.natCast_le_toENat_iff]
  /-
    🎉 no goals
  -/


theorem card_eq_zero_iff_empty (α : Type*) : card α = 0 ↔ IsEmpty α := by
  /-
    α : Type u_3
    ⊢ Iff (Eq (ENat.card α) 0) (IsEmpty α)
  -/
  rw [← Cardinal.mk_eq_zero_iff]
  /-
    α : Type u_3
    ⊢ Iff (Eq (ENat.card α) 0) (Eq (Cardinal.mk α) 0)
  -/
  simp [card]
  /-
    🎉 no goals
  -/


theorem card_le_one_iff_subsingleton (α : Type*) : card α ≤ 1 ↔ Subsingleton α := by
  /-
    α : Type u_3
    ⊢ Iff (LE.le (ENat.card α) 1) (Subsingleton α)
  -/
  rw [← le_one_iff_subsingleton]
  /-
    α : Type u_3
    ⊢ Iff (LE.le (ENat.card α) 1) (LE.le (Cardinal.mk α) 1)
  -/
  simp [card]
  /-
    🎉 no goals
  -/


theorem one_lt_card_iff_nontrivial (α : Type*) : 1 < card α ↔ Nontrivial α := by
  /-
    α : Type u_3
    ⊢ Iff (LT.lt 1 (ENat.card α)) (Nontrivial α)
  -/
  rw [← Cardinal.one_lt_iff_nontrivial]
  /-
    α : Type u_3
    ⊢ Iff (LT.lt 1 (ENat.card α)) (LT.lt 1 (Cardinal.mk α))
  -/
  conv_rhs => rw [← Nat.cast_one]
  /-
    α : Type u_3
    ⊢ Iff (LT.lt 1 (ENat.card α)) (LT.lt (↑1) (Cardinal.mk α))
  -/
  rw [← natCast_lt_toENat_iff]
  /-
    α : Type u_3
    ⊢ Iff (LT.lt 1 (ENat.card α)) (LT.lt (↑1) (Cardinal.toENat (Cardinal.mk α)))
  -/
  simp only [ENat.card, Nat.cast_one]
  /-
    🎉 no goals
  -/


