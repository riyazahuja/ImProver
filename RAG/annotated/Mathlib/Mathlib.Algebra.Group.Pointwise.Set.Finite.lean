@[to_additive (attr := simp)]
theorem finite_one : (1 : Set α).Finite :=
  finite_singleton _


@[to_additive]
theorem Finite.mul : s.Finite → t.Finite → (s * t).Finite :=
  Finite.image2 _


/-- Multiplication preserves finiteness. -/
@[to_additive "Addition preserves finiteness."]
instance fintypeMul [DecidableEq α] (s t : Set α) [Fintype s] [Fintype t] : Fintype (s * t) :=
  Set.fintypeImage2 _ _ _


@[to_additive]
instance decidableMemMul [Fintype α] [DecidableEq α] [DecidablePred (· ∈ s)]
    [DecidablePred (· ∈ t)] : DecidablePred (· ∈ s * t) := fun _ ↦ decidable_of_iff _ mem_mul.symm


@[to_additive]
instance decidableMemPow [Fintype α] [DecidableEq α] [DecidablePred (· ∈ s)] (n : ℕ) :
    DecidablePred (· ∈ s ^ n) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝³ : Monoid α
    s t : Set α
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : DecidablePred fun x => Membership.mem s x
    n : Nat
    ⊢ DecidablePred fun x => Membership.mem (HPow.hPow s n) x
  -/
  induction' n with n ih
    /-
      case zero
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : Monoid α
      s t : Set α
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidablePred fun x => Membership.mem s x
      ⊢ DecidablePred fun x => Membership.mem (HPow.hPow s 0) x
    -/
  · simp only [pow_zero, mem_one]
    /-
      case zero
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : Monoid α
      s t : Set α
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidablePred fun x => Membership.mem s x
      ⊢ DecidablePred fun x => Eq x 1
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case succ
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : Monoid α
      s t : Set α
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      ih : DecidablePred fun x => Membership.mem (HPow.hPow s n) x
      ⊢ DecidablePred fun x => Membership.mem (HPow.hPow s (HAdd.hAdd n 1)) x
    -/
  · letI := ih
    /-
      case succ
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : Monoid α
      s t : Set α
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      ih : DecidablePred fun x => Membership.mem (HPow.hPow s n) x
      this : DecidablePred fun x => Membership.mem (HPow.hPow s n) x := ih
      ⊢ DecidablePred fun x => Membership.mem (HPow.hPow s (HAdd.hAdd n 1)) x
    -/
    rw [pow_succ]
    /-
      case succ
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : Monoid α
      s t : Set α
      inst✝² : Fintype α
      inst✝¹ : DecidableEq α
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      ih : DecidablePred fun x => Membership.mem (HPow.hPow s n) x
      this : DecidablePred fun x => Membership.mem (HPow.hPow s n) x := ih
      ⊢ DecidablePred fun x => Membership.mem (HMul.hMul (HPow.hPow s n) s) x
    -/
    infer_instance
    /-
      🎉 no goals
    -/


@[to_additive]
theorem Finite.smul : s.Finite → t.Finite → (s • t).Finite :=
  Finite.image2 _


@[to_additive]
theorem Finite.smul_set : s.Finite → (a • s).Finite :=
  Finite.image _


@[to_additive]
theorem Infinite.of_smul_set : (a • s).Infinite → s.Infinite :=
  Infinite.of_image _


theorem Finite.vsub (hs : s.Finite) (ht : t.Finite) : Set.Finite (s -ᵥ t) :=
  hs.image2 _ ht


@[to_additive]
lemma finite_mul : (s * t).Finite ↔ s.Finite ∧ t.Finite ∨ s = ∅ ∨ t = ∅ :=
  finite_image2 (fun _ _ ↦ (mul_left_injective _).injOn) fun _ _ ↦ (mul_right_injective _).injOn


@[to_additive]
lemma infinite_mul : (s * t).Infinite ↔ s.Infinite ∧ t.Nonempty ∨ t.Infinite ∧ s.Nonempty :=
  infinite_image2 (fun _ _ => (mul_left_injective _).injOn) fun _ _ => (mul_right_injective _).injOn


@[to_additive (attr := simp)] lemma finite_inv : s⁻¹.Finite ↔ s.Finite := by
  /-
    α : Type u_2
    inst✝ : InvolutiveInv α
    s : Set α
    ⊢ Iff (Inv.inv s).Finite s.Finite
  -/
  rw [← image_inv_eq_inv, finite_image_iff inv_injective.injOn]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)] lemma infinite_inv : s⁻¹.Infinite ↔ s.Infinite := finite_inv.not


@[to_additive] alias ⟨Finite.of_inv, Finite.inv⟩ := finite_inv


@[to_additive] lemma Finite.div : s.Finite → t.Finite → (s / t).Finite := .image2 _


/-- Division preserves finiteness. -/
@[to_additive "Subtraction preserves finiteness."]
instance fintypeDiv [DecidableEq α] (s t : Set α) [Fintype s] [Fintype t] : Fintype (s / t) :=
  Set.fintypeImage2 _ _ _


@[to_additive]
lemma finite_div : (s / t).Finite ↔ s.Finite ∧ t.Finite ∨ s = ∅ ∨ t = ∅ :=
  finite_image2 (fun _ _ ↦ div_left_injective.injOn) fun _ _ ↦ div_right_injective.injOn


@[to_additive]
lemma infinite_div : (s / t).Infinite ↔ s.Infinite ∧ t.Nonempty ∨ t.Infinite ∧ s.Nonempty :=
  infinite_image2 (fun _ _ ↦ div_left_injective.injOn) fun _ _ ↦ div_right_injective.injOn


@[to_additive (attr := simp)]
theorem finite_smul_set : (a • s).Finite ↔ s.Finite :=
  finite_image_iff (MulAction.injective _).injOn


@[to_additive (attr := simp)]
theorem infinite_smul_set : (a • s).Infinite ↔ s.Infinite :=
  infinite_image_iff (MulAction.injective _).injOn


@[to_additive] alias ⟨Finite.of_smul_set, _⟩ := finite_smul_set

@[to_additive] alias ⟨_, Infinite.smul_set⟩ := infinite_smul_set


@[to_additive]
theorem card_pow_eq_card_pow_card_univ [∀ k : ℕ, DecidablePred (· ∈ S ^ k)] :
    ∀ k, Fintype.card G ≤ k → Fintype.card (↥(S ^ k)) = Fintype.card (↥(S ^ Fintype.card G)) := by
  /-
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : Fintype G
    S : Set G
    inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow S k) x
    ⊢ ∀ (k : Nat), LE.le (Fintype.card G) k → Eq (Fintype.card ↑(HPow.hPow S k)) ( …
  -/
  have hG : 0 < Fintype.card G := Fintype.card_pos
  /-
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : Fintype G
    S : Set G
    inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow S k) x
    hG : LT.lt 0 (Fintype.card G)
    ⊢ ∀ (k : Nat), LE.le (Fintype.card G) k → Eq (Fintype.card ↑(HPow.hPow S k)) ( …
  -/
  rcases S.eq_empty_or_nonempty with (rfl | ⟨a, ha⟩)
    /-
      case inl
      G : Type u_5
      inst✝² : Group G
      inst✝¹ : Fintype G
      hG : LT.lt 0 (Fintype.card G)
      inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow EmptyColl …
      ⊢ ∀ (k : Nat), LE.le (Fintype.card G) k → Eq (Fintype.card ↑(HPow.hPow EmptyCo …
    -/
  · refine fun k hk ↦ Fintype.card_congr ?_
    /-
      case inl
      G : Type u_5
      inst✝² : Group G
      inst✝¹ : Fintype G
      hG : LT.lt 0 (Fintype.card G)
      inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow EmptyColl …
      k : Nat
      hk : LE.le (Fintype.card G) k
      ⊢ Equiv ↑(HPow.hPow EmptyCollection.emptyCollection k) ↑(HPow.hPow EmptyCollec …
    -/
    rw [empty_pow (hG.trans_le hk).ne', empty_pow (ne_of_gt hG)]
    /-
      🎉 no goals
    -/
  have key : ∀ (a) (s t : Set G) [Fintype s] [Fintype t],
      (∀ b : G, b ∈ s → b * a ∈ t) → Fintype.card s ≤ Fintype.card t := by
    refine fun a s t _ _ h ↦ Fintype.card_le_of_injective (fun ⟨b, hb⟩ ↦ ⟨b * a, h b hb⟩) ?_
    rintro ⟨b, hb⟩ ⟨c, hc⟩ hbc
    exact Subtype.ext (mul_right_cancel (Subtype.ext_iff.mp hbc))
  have mono : Monotone (fun n ↦ Fintype.card (↥(S ^ n)) : ℕ → ℕ) :=
    monotone_nat_of_le_succ fun n ↦ key a _ _ fun b hb ↦ Set.mul_mem_mul hb ha
  refine fun _ ↦ Nat.stabilises_of_monotone mono (fun n ↦ set_fintype_card_le_univ (S ^ n))
    fun n h ↦ le_antisymm (mono (n + 1).le_succ) (key a⁻¹ (S ^ (n + 2)) (S ^ (n + 1)) ?_)
  replace h₂ : S ^ n * {a} = S ^ (n + 1) := by
    have : Fintype (S ^ n * Set.singleton a) := by
      classical
      apply fintypeMul
    refine Set.eq_of_subset_of_card_le ?_ (le_trans (ge_of_eq h) ?_)
    · exact mul_subset_mul Set.Subset.rfl (Set.singleton_subset_iff.mpr ha)
    · convert key a (S ^ n) (S ^ n * {a}) fun b hb ↦ Set.mul_mem_mul hb (Set.mem_singleton a)
  /-
    case inr.intro
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : Fintype G
    S : Set G
    inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow S k) x
    hG : LT.lt 0 (Fintype.card G)
    a : G
    ha : Membership.mem S a
    key : ∀ (a : G) (s t : Set G) [inst : Fintype ↑s] [inst_1 : Fintype ↑t], (∀ (b …
    mono : Monotone fun n => Fintype.card ↑(HPow.hPow S n)
    x✝ n : Nat
    h : Eq (Fintype.card ↑(HPow.hPow S n)) (Fintype.card ↑(HPow.hPow S (HAdd.hAdd  …
    h₂ : Eq (HMul.hMul (HPow.hPow S n) (Singleton.singleton a)) (HPow.hPow S (HAdd …
    ⊢ ∀ (b : G), Membership.mem (HPow.hPow S (HAdd.hAdd n 2)) b → Membership.mem ( …
  -/
  rw [pow_succ', ← h₂, ← mul_assoc, ← pow_succ', h₂, mul_singleton, forall_mem_image]
  /-
    case inr.intro
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : Fintype G
    S : Set G
    inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow S k) x
    hG : LT.lt 0 (Fintype.card G)
    a : G
    ha : Membership.mem S a
    key : ∀ (a : G) (s t : Set G) [inst : Fintype ↑s] [inst_1 : Fintype ↑t], (∀ (b …
    mono : Monotone fun n => Fintype.card ↑(HPow.hPow S n)
    x✝ n : Nat
    h : Eq (Fintype.card ↑(HPow.hPow S n)) (Fintype.card ↑(HPow.hPow S (HAdd.hAdd  …
    h₂ : Eq (HMul.hMul (HPow.hPow S n) (Singleton.singleton a)) (HPow.hPow S (HAdd …
    ⊢ ∀ ⦃x : G⦄, Membership.mem (HPow.hPow S (HAdd.hAdd n 1)) x → Membership.mem ( …
  -/
  intro x hx
  /-
    case inr.intro
    G : Type u_5
    inst✝² : Group G
    inst✝¹ : Fintype G
    S : Set G
    inst✝ : (k : Nat) → DecidablePred fun x => Membership.mem (HPow.hPow S k) x
    hG : LT.lt 0 (Fintype.card G)
    a : G
    ha : Membership.mem S a
    key : ∀ (a : G) (s t : Set G) [inst : Fintype ↑s] [inst_1 : Fintype ↑t], (∀ (b …
    mono : Monotone fun n => Fintype.card ↑(HPow.hPow S n)
    x✝ n : Nat
    h : Eq (Fintype.card ↑(HPow.hPow S n)) (Fintype.card ↑(HPow.hPow S (HAdd.hAdd  …
    h₂ : Eq (HMul.hMul (HPow.hPow S n) (Singleton.singleton a)) (HPow.hPow S (HAdd …
    x : G
    hx : Membership.mem (HPow.hPow S (HAdd.hAdd n 1)) x
    ⊢ Membership.mem (HPow.hPow S (HAdd.hAdd n 1)) (HMul.hMul (HMul.hMul x a) (Inv …
  -/
  rwa [mul_inv_cancel_right]
  /-
    🎉 no goals
  -/


