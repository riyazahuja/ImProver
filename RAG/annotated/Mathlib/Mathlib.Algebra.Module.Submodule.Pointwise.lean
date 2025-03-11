/-- The submodule with every element negated. Note if `R` is a ring and not just a semiring, this
is a no-op, as shown by `Submodule.neg_eq_self`.

Recall that When `R` is the semiring corresponding to the nonnegative elements of `R'`,
`Submodule R' M` is the type of cones of `M`. This instance reflects such cones about `0`.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseNeg : Neg (Submodule R M) where
  neg p :=
    { -p.toAddSubmonoid with
      smul_mem' := fun r m hm => Set.mem_neg.2 <| smul_neg r m ▸ p.smul_mem r <| Set.mem_neg.1 hm }


@[simp]
theorem coe_set_neg (S : Submodule R M) : ↑(-S) = -(S : Set M) :=
  rfl


@[simp]
theorem neg_toAddSubmonoid (S : Submodule R M) : (-S).toAddSubmonoid = -S.toAddSubmonoid :=
  rfl


@[simp]
theorem mem_neg {g : M} {S : Submodule R M} : g ∈ -S ↔ -g ∈ S :=
  Iff.rfl


/-- `Submodule.pointwiseNeg` is involutive.

This is available as an instance in the `Pointwise` locale. -/
protected def involutivePointwiseNeg : InvolutiveNeg (Submodule R M) where
  neg := Neg.neg
  neg_neg _S := SetLike.coe_injective <| neg_neg _


@[simp]
theorem neg_le_neg (S T : Submodule R M) : -S ≤ -T ↔ S ≤ T :=
  SetLike.coe_subset_coe.symm.trans Set.neg_subset_neg


theorem neg_le (S T : Submodule R M) : -S ≤ T ↔ S ≤ -T :=
  SetLike.coe_subset_coe.symm.trans Set.neg_subset


/-- `Submodule.pointwiseNeg` as an order isomorphism. -/
def negOrderIso : Submodule R M ≃o Submodule R M where
  toEquiv := Equiv.neg _
  map_rel_iff' := @neg_le_neg _ _ _ _ _


theorem closure_neg (s : Set M) : span R (-s) = -span R s := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : Set M
    ⊢ Eq (Submodule.span R (Neg.neg s)) (Neg.neg (Submodule.span R s))
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_2
      M : Type u_3
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set M
      ⊢ LE.le (Submodule.span R (Neg.neg s)) (Neg.neg (Submodule.span R s))
    -/
  · rw [span_le, coe_set_neg, ← Set.neg_subset, neg_neg]
    /-
      case a
      R : Type u_2
      M : Type u_3
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set M
      ⊢ HasSubset.Subset s ↑(Submodule.span R s)
    -/
    exact subset_span
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_2
      M : Type u_3
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set M
      ⊢ LE.le (Neg.neg (Submodule.span R s)) (Submodule.span R (Neg.neg s))
    -/
  · rw [neg_le, span_le, coe_set_neg, ← Set.neg_subset]
    /-
      case a
      R : Type u_2
      M : Type u_3
      inst✝² : Semiring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : Set M
      ⊢ HasSubset.Subset (Neg.neg s) ↑(Submodule.span R (Neg.neg s))
    -/
    exact subset_span
    /-
      🎉 no goals
    -/


@[simp]
theorem neg_inf (S T : Submodule R M) : -(S ⊓ T) = -S ⊓ -T :=
  SetLike.coe_injective Set.inter_neg


@[simp]
theorem neg_sup (S T : Submodule R M) : -(S ⊔ T) = -S ⊔ -T :=
  (negOrderIso : Submodule R M ≃o Submodule R M).map_sup S T


@[simp]
theorem neg_bot : -(⊥ : Submodule R M) = ⊥ :=
  SetLike.coe_injective <| (Set.neg_singleton 0).trans <| congr_arg _ neg_zero


@[simp]
theorem neg_top : -(⊤ : Submodule R M) = ⊤ :=
  SetLike.coe_injective <| Set.neg_univ


@[simp]
theorem neg_iInf {ι : Sort*} (S : ι → Submodule R M) : (-⨅ i, S i) = ⨅ i, -S i :=
  (negOrderIso : Submodule R M ≃o Submodule R M).map_iInf _


@[simp]
theorem neg_iSup {ι : Sort*} (S : ι → Submodule R M) : (-⨆ i, S i) = ⨆ i, -S i :=
  (negOrderIso : Submodule R M ≃o Submodule R M).map_iSup _


@[simp]
theorem neg_eq_self [Ring R] [AddCommGroup M] [Module R M] (p : Submodule R M) : -p = p :=
  ext fun _ => p.neg_mem_iff


instance pointwiseZero : Zero (Submodule R M) where
  zero := ⊥


instance pointwiseAdd : Add (Submodule R M) where
  add := (· ⊔ ·)


instance pointwiseAddCommMonoid : AddCommMonoid (Submodule R M) where
  add_assoc := sup_assoc
  zero_add := bot_sup_eq
  add_zero := sup_bot_eq
  add_comm := sup_comm
  nsmul := nsmulRec


@[simp]
theorem add_eq_sup (p q : Submodule R M) : p + q = p ⊔ q :=
  rfl


@[simp]
theorem zero_eq_bot : (0 : Submodule R M) = ⊥ :=
  rfl


instance : CanonicallyOrderedAddCommMonoid (Submodule R M) :=
  { Submodule.pointwiseAddCommMonoid,
    Submodule.completeLattice with
    add_le_add_left := fun _a _b => sup_le_sup_left
    exists_add_of_le := @fun _a b h => ⟨b, (sup_eq_right.2 h).symm⟩
    le_self_add := fun _a _b => le_sup_left }


/-- The action on a submodule corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseDistribMulAction : DistribMulAction α (Submodule R M) where
  smul a S := S.map (DistribMulAction.toLinearMap R M a : M →ₗ[R] M)
  one_smul S :=
    (congr_arg (fun f : Module.End R M => S.map f) (LinearMap.ext <| one_smul α)).trans S.map_id
  mul_smul _a₁ _a₂ S :=
    (congr_arg (fun f : Module.End R M => S.map f) (LinearMap.ext <| mul_smul _ _)).trans
      (S.map_comp _ _)
  smul_zero _a := map_bot _
  smul_add _a _S₁ _S₂ := map_sup _ _ _


@[simp]
theorem coe_pointwise_smul (a : α) (S : Submodule R M) : ↑(a • S) = a • (S : Set M) :=
  rfl


@[simp]
theorem pointwise_smul_toAddSubmonoid (a : α) (S : Submodule R M) :
    (a • S).toAddSubmonoid = a • S.toAddSubmonoid :=
  rfl


@[simp]
theorem pointwise_smul_toAddSubgroup {R M : Type*} [Ring R] [AddCommGroup M] [DistribMulAction α M]
    [Module R M] [SMulCommClass α R M] (a : α) (S : Submodule R M) :
    (a • S).toAddSubgroup = a • S.toAddSubgroup :=
  rfl


theorem smul_mem_pointwise_smul (m : M) (a : α) (S : Submodule R M) : m ∈ S → a • m ∈ a • S :=
  (Set.smul_mem_smul_set : _ → _ ∈ a • (S : Set M))


instance : CovariantClass α (Submodule R M) HSMul.hSMul LE.le :=
  ⟨fun _ _ => map_mono⟩


/-- See also `Submodule.smul_bot`. -/
@[simp]
theorem smul_bot' (a : α) : a • (⊥ : Submodule R M) = ⊥ :=
  map_bot _


/-- See also `Submodule.smul_sup`. -/
theorem smul_sup' (a : α) (S T : Submodule R M) : a • (S ⊔ T) = a • S ⊔ a • T :=
  map_sup _ _ _


theorem smul_span (a : α) (s : Set M) : a • span R s = span R (a • s) :=
  map_span _ _


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    R : Type u_2
                                                                                    M : Type u_3
                                                                                    inst✝⁵ : Semiring R
                                                                                    inst✝⁴ : AddCommMonoid M
                                                                                    inst✝³ : Module R M
                                                                                    inst✝² : Monoid α
                                                                                    inst✝¹ : DistribMulAction α M
                                                                                    inst✝ : SMulCommClass α R M
                                                                                    a : α
                                                                                    S : Submodule R M
                                                                                    ⊢ Eq (HSMul.hSMul a S) (Submodule.span R (HSMul.hSMul a ↑S))
                                                                                  -/
lemma smul_def (a : α) (S : Submodule R M) : a • S = span R (a • S : Set M) := by simp [← smul_span]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem span_smul (a : α) (s : Set M) : span R (a • s) = a • span R s :=
  Eq.symm (span_image _).symm


instance pointwiseCentralScalar [DistribMulAction αᵐᵒᵖ M] [SMulCommClass αᵐᵒᵖ R M]
    [IsCentralScalar α M] : IsCentralScalar α (Submodule R M) :=
  ⟨fun _a S => (congr_arg fun f : Module.End R M => S.map f) <| LinearMap.ext <| op_smul_eq_smul _⟩


@[simp]
theorem smul_le_self_of_tower {α : Type*} [Semiring α] [Module α R] [Module α M]
    [SMulCommClass α R M] [IsScalarTower α R M] (a : α) (S : Submodule R M) : a • S ≤ S := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    α : Type u_4
    inst✝⁴ : Semiring α
    inst✝³ : Module α R
    inst✝² : Module α M
    inst✝¹ : SMulCommClass α R M
    inst✝ : IsScalarTower α R M
    a : α
    S : Submodule R M
    ⊢ LE.le (HSMul.hSMul a S) S
  -/
  rintro y ⟨x, hx, rfl⟩
  /-
    case intro.intro
    R : Type u_2
    M : Type u_3
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    α : Type u_4
    inst✝⁴ : Semiring α
    inst✝³ : Module α R
    inst✝² : Module α M
    inst✝¹ : SMulCommClass α R M
    inst✝ : IsScalarTower α R M
    a : α
    S : Submodule R M
    x : M
    hx : Membership.mem (↑S) x
    ⊢ Membership.mem S ((DistribMulAction.toLinearMap R M a) x)
  -/
  exact smul_of_tower_mem _ a hx
  /-
    🎉 no goals
  -/


/-- The action on a submodule corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale.

This is a stronger version of `Submodule.pointwiseDistribMulAction`. Note that `add_smul` does
not hold so this cannot be stated as a `Module`. -/
protected def pointwiseMulActionWithZero : MulActionWithZero α (Submodule R M) :=
  { Submodule.pointwiseDistribMulAction with
    zero_smul := fun S =>
      (congr_arg (fun f : M →ₗ[R] M => S.map f) (LinearMap.ext <| zero_smul α)).trans S.map_zero }


/--
Let `s ⊆ R` be a set and `N ≤ M` be a submodule, then `s • N` is the smallest submodule containing
all `r • n` where `r ∈ s` and `n ∈ N`.
-/
protected def pointwiseSetSMul : SMul (Set S) (Submodule R M) where
  smul s N := sInf { p | ∀ ⦃r : S⦄ ⦃n : M⦄, r ∈ s → n ∈ N → r • n ∈ p }


lemma mem_set_smul_def (x : M) :
    x ∈ s • N ↔
  x ∈ sInf { p : Submodule R M | ∀ ⦃r : S⦄ {n : M}, r ∈ s → n ∈ N → r • n ∈ p } := Iff.rfl


variable {s N} in
@[aesop safe]
lemma mem_set_smul_of_mem_mem {r : S} {m : M} (mem1 : r ∈ s) (mem2 : m ∈ N) :
    r • m ∈ s • N := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_4
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S M
    s : Set S
    N : Submodule R M
    r : S
    m : M
    mem1 : Membership.mem s r
    mem2 : Membership.mem N m
    ⊢ Membership.mem (HSMul.hSMul s N) (HSMul.hSMul r m)
  -/
  rw [mem_set_smul_def, mem_sInf]
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_4
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S M
    s : Set S
    N : Submodule R M
    r : S
    m : M
    mem1 : Membership.mem s r
    mem2 : Membership.mem N m
    ⊢ ∀ (p : Submodule R M), Membership.mem (setOf fun p => ∀ ⦃r : S⦄ {n : M}, Mem …
  -/
  exact fun _ h => h mem1 mem2
  /-
    🎉 no goals
  -/


lemma set_smul_le (p : Submodule R M)
    (closed_under_smul : ∀ ⦃r : S⦄ ⦃n : M⦄, r ∈ s → n ∈ N → r • n ∈ p) :
    s • N ≤ p :=
  sInf_le closed_under_smul


lemma set_smul_le_iff (p : Submodule R M) :
    s • N ≤ p ↔
    ∀ ⦃r : S⦄ ⦃n : M⦄, r ∈ s → n ∈ N → r • n ∈ p := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_4
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S M
    s : Set S
    N p : Submodule R M
    ⊢ Iff (LE.le (HSMul.hSMul s N) p) (∀ ⦃r : S⦄ ⦃n : M⦄, Membership.mem s r → Mem …
  -/
  fconstructor
    /-
      case mp
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      s : Set S
      N p : Submodule R M
      ⊢ LE.le (HSMul.hSMul s N) p → ∀ ⦃r : S⦄ ⦃n : M⦄, Membership.mem s r → Membersh …
    -/
  · intro h r n hr hn
    /-
      case mp
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      s : Set S
      N p : Submodule R M
      h : LE.le (HSMul.hSMul s N) p
      r : S
      n : M
      hr : Membership.mem s r
      hn : Membership.mem N n
      ⊢ Membership.mem p (HSMul.hSMul r n)
    -/
    exact h <| mem_set_smul_of_mem_mem hr hn
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      s : Set S
      N p : Submodule R M
      ⊢ (∀ ⦃r : S⦄ ⦃n : M⦄, Membership.mem s r → Membership.mem N n → Membership.mem …
    -/
  · apply set_smul_le
    /-
      🎉 no goals
    -/


lemma set_smul_eq_of_le (p : Submodule R M)
    (closed_under_smul : ∀ ⦃r : S⦄ ⦃n : M⦄, r ∈ s → n ∈ N → r • n ∈ p)
    (le : p ≤ s • N) :
    s • N = p :=
  le_antisymm (set_smul_le s N p closed_under_smul) le


instance : CovariantClass (Set S) (Submodule R M) HSMul.hSMul LE.le :=
  ⟨fun _ _ _ le => set_smul_le _ _ _ fun _ _ hr hm => mem_set_smul_of_mem_mem (mem1 := hr)
    (mem2 := le hm)⟩


@[deprecated smul_mono_right (since := "2024-03-31")]
theorem set_smul_mono_right {p q : Submodule R M} (le : p ≤ q) :
    s • p ≤ s • q :=
  smul_mono_right s le


lemma set_smul_mono_left {s t : Set S} (le : s ≤ t) :
    s • N ≤ t • N :=
  set_smul_le _ _ _ fun _ _ hr hm => mem_set_smul_of_mem_mem (mem1 := le hr)
    (mem2 := hm)


lemma set_smul_le_of_le_le {s t : Set S} {p q : Submodule R M}
    (le_set : s ≤ t) (le_submodule : p ≤ q) : s • p ≤ t • q :=
  le_trans (set_smul_mono_left _ le_set) <| smul_mono_right _ le_submodule


lemma set_smul_eq_iSup [SMulCommClass S R M] (s : Set S) (N : Submodule R M) :
    s • N = ⨆ (a ∈ s), a • N := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    inst✝ : SMulCommClass S R M
    s : Set S
    N : Submodule R M
    ⊢ Eq (HSMul.hSMul s N) (iSup fun a => iSup fun h => HSMul.hSMul a N)
  -/
  refine Eq.trans (congrArg sInf ?_) csInf_Ici
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    inst✝ : SMulCommClass S R M
    s : Set S
    N : Submodule R M
    ⊢ Eq (setOf fun p => ∀ ⦃r : S⦄ ⦃n : M⦄, Membership.mem s r → Membership.mem N  …
  -/
  simp_rw [← Set.Ici_def, iSup_le_iff, @forall_comm M]
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    inst✝ : SMulCommClass S R M
    s : Set S
    N : Submodule R M
    ⊢ Eq (setOf fun p => ∀ ⦃r : S⦄, Membership.mem s r → ∀ (a : M), Membership.mem …
  -/
  exact Set.ext fun _ => forall₂_congr (fun _ _ => Iff.symm map_le_iff_le_comap)
  /-
    🎉 no goals
  -/


theorem set_smul_span [SMulCommClass S R M] (s : Set S) (t : Set M) :
    s • span R t = span R (s • t) := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    inst✝ : SMulCommClass S R M
    s : Set S
    t : Set M
    ⊢ Eq (HSMul.hSMul s (Submodule.span R t)) (Submodule.span R (HSMul.hSMul s t))
  -/
  simp_rw [set_smul_eq_iSup, smul_span, iSup_span, Set.iUnion_smul_set]
  /-
    🎉 no goals
  -/


theorem span_set_smul [SMulCommClass S R M] (s : Set S) (t : Set M) :
    span R (s • t) = s • span R t := (set_smul_span s t).symm


variable {s N} in
/--
Induction principle for set acting on submodules. To prove `P` holds for all `s • N`, it is enough
to prove:
- for all `r ∈ s` and `n ∈ N`, `P (r • n)`;
- for all `r` and `m ∈ s • N`, `P (r • n)`;
- for all `m₁, m₂`, `P m₁` and `P m₂` implies `P (m₁ + m₂)`;
- `P 0`.

To invoke this induction principle, use `induction x, hx using Submodule.set_smul_inductionOn` where
`x : M` and `hx : x ∈ s • N`
-/
@[elab_as_elim]
lemma set_smul_inductionOn {motive : (x : M) → (_ : x ∈ s • N) → Prop}
    (x : M)
    (hx : x ∈ s • N)
    (smul₀ : ∀ ⦃r : S⦄ ⦃n : M⦄ (mem₁ : r ∈ s) (mem₂ : n ∈ N),
      motive (r • n) (mem_set_smul_of_mem_mem mem₁ mem₂))
    (smul₁ : ∀ (r : R) ⦃m : M⦄ (mem : m ∈ s • N) ,
      motive m mem → motive (r • m) (Submodule.smul_mem _ r mem)) --
    (add : ∀ ⦃m₁ m₂ : M⦄ (mem₁ : m₁ ∈ s • N) (mem₂ : m₂ ∈ s • N),
      motive m₁ mem₁ → motive m₂ mem₂ → motive (m₁ + m₂) (Submodule.add_mem _ mem₁ mem₂))
    (zero : motive 0 (Submodule.zero_mem _)) :
    motive x hx :=
  let ⟨_, h⟩ := set_smul_le s N
    { carrier := { m | ∃ (mem : m ∈ s • N), motive m mem },
      zero_mem' := ⟨Submodule.zero_mem _, zero⟩
      add_mem' := fun ⟨mem, h⟩ ⟨mem', h'⟩ ↦ ⟨_, add mem mem' h h'⟩
      smul_mem' := fun r _ ⟨mem, h⟩ ↦ ⟨_, smul₁ r mem h⟩ }
    (fun _ _ mem mem' ↦ ⟨mem_set_smul_of_mem_mem mem mem', smul₀ mem mem'⟩) hx
  h

-- Implementation note: if `N` is both an `R`-submodule and `S`-submodule and `SMulCommClass R S M`,
-- this lemma is also true for any `s : Set S`.

lemma set_smul_eq_map [SMulCommClass R R N] :
    sR • N =
    Submodule.map
      (N.subtype.comp (Finsupp.lsum R <| DistribMulAction.toLinearMap _ _))
      (Finsupp.supported N R sR) := by
  classical
  apply set_smul_eq_of_le
  · intro r n hr hn
    exact ⟨Finsupp.single r ⟨n, hn⟩, Finsupp.single_mem_supported _ _ hr, by simp⟩
  · intro x hx
    obtain ⟨c, hc, rfl⟩ := hx
    simp only [LinearMap.coe_comp, coe_subtype, Finsupp.coe_lsum, Finsupp.sum, Function.comp_apply]
    rw [AddSubmonoid.coe_finset_sum]
    refine Submodule.sum_mem (p := sR • N) (t := c.support) ?_ _ ⟨sR • N, ?_⟩
    · rintro r hr
      rw [mem_set_smul_def, Submodule.mem_sInf]
      rintro p hp
      exact hp (hc hr) (c r).2
    · ext x : 1
      simp only [Set.mem_iInter, SetLike.mem_coe]
      fconstructor
      · refine fun h ↦ h fun r n hr hn ↦ ?_
        rw [mem_set_smul_def, mem_sInf]
        exact fun p hp ↦ hp hr hn
      · aesop


lemma mem_set_smul (x : M) [SMulCommClass R R N] :
    x ∈ sR • N ↔ ∃ (c : R →₀ N), (c.support : Set R) ⊆ sR ∧ x = c.sum fun r m ↦ r • m := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    sR : Set R
    N : Submodule R M
    x : M
    inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
    ⊢ Iff (Membership.mem (HSMul.hSMul sR N) x) (Exists fun c => And (HasSubset.Su …
  -/
  fconstructor
    /-
      case mp
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      x : M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      ⊢ Membership.mem (HSMul.hSMul sR N) x → Exists fun c => And (HasSubset.Subset  …
    -/
  · intros h
    /-
      case mp
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      x : M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      h : Membership.mem (HSMul.hSMul sR N) x
      ⊢ Exists fun c => And (HasSubset.Subset (↑c.support) sR) (Eq x ↑(c.sum fun r m …
    -/
    rw [set_smul_eq_map] at h
    /-
      case mp
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      x : M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      h : Membership.mem (Submodule.map (N.subtype.comp ((Finsupp.lsum R) (DistribMu …
      ⊢ Exists fun c => And (HasSubset.Subset (↑c.support) sR) (Eq x ↑(c.sum fun r m …
    -/
    obtain ⟨c, hc, rfl⟩ := h
    /-
      case mp.intro.intro
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      c : Finsupp R (Subtype fun x => Membership.mem N x)
      hc : Membership.mem (↑(Finsupp.supported (Subtype fun x => Membership.mem N x) …
      ⊢ Exists fun c_1 => And (HasSubset.Subset (↑c_1.support) sR) (Eq ((N.subtype.c …
    -/
    exact ⟨c, hc, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      x : M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      ⊢ (Exists fun c => And (HasSubset.Subset (↑c.support) sR) (Eq x ↑(c.sum fun r  …
    -/
  · rw [mem_set_smul_def, Submodule.mem_sInf]
    /-
      case mpr
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      x : M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      ⊢ (Exists fun c => And (HasSubset.Subset (↑c.support) sR) (Eq x ↑(c.sum fun r  …
    -/
    rintro ⟨c, hc1, rfl⟩ p hp
    /-
      case mpr.intro.intro
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      c : Finsupp R (Subtype fun x => Membership.mem N x)
      hc1 : HasSubset.Subset (↑c.support) sR
      p : Submodule R M
      hp : Membership.mem (setOf fun p => ∀ ⦃r : R⦄ {n : M}, Membership.mem sR r → M …
      ⊢ Membership.mem p ↑(c.sum fun r m => HSMul.hSMul r m)
    -/
    rw [Finsupp.sum, AddSubmonoid.coe_finset_sum]
    /-
      case mpr.intro.intro
      R : Type u_2
      M : Type u_3
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      sR : Set R
      N : Submodule R M
      inst✝ : SMulCommClass R R (Subtype fun x => Membership.mem N x)
      c : Finsupp R (Subtype fun x => Membership.mem N x)
      hc1 : HasSubset.Subset (↑c.support) sR
      p : Submodule R M
      hp : Membership.mem (setOf fun p => ∀ ⦃r : R⦄ {n : M}, Membership.mem sR r → M …
      ⊢ Membership.mem p (c.support.sum fun i => ↑(HSMul.hSMul i (c i)))
    -/
    exact Submodule.sum_mem _ fun r hr ↦ hp (hc1 hr) (c _).2
    /-
      🎉 no goals
    -/


@[simp] lemma empty_set_smul : (∅ : Set S) • N = ⊥ := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_4
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S M
    N : Submodule R M
    ⊢ Eq (HSMul.hSMul EmptyCollection.emptyCollection N) Bot.bot
  -/
  ext
  /-
    case h
    R : Type u_2
    M : Type u_3
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_4
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S M
    N : Submodule R M
    x✝ : M
    ⊢ Iff (Membership.mem (HSMul.hSMul EmptyCollection.emptyCollection N) x✝) (Mem …
  -/
  fconstructor
    /-
      case h.mp
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      N : Submodule R M
      x✝ : M
      ⊢ Membership.mem (HSMul.hSMul EmptyCollection.emptyCollection N) x✝ → Membersh …
    -/
  · intro hx
    /-
      case h.mp
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      N : Submodule R M
      x✝ : M
      hx : Membership.mem (HSMul.hSMul EmptyCollection.emptyCollection N) x✝
      ⊢ Membership.mem Bot.bot x✝
    -/
    rw [mem_set_smul_def, Submodule.mem_sInf] at hx
    /-
      case h.mp
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      N : Submodule R M
      x✝ : M
      hx : ∀ (p : Submodule R M), Membership.mem (setOf fun p => ∀ ⦃r : S⦄ {n : M},  …
      ⊢ Membership.mem Bot.bot x✝
    -/
    exact hx ⊥ (fun r _ hr ↦ hr.elim)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      S : Type u_4
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S M
      N : Submodule R M
      x✝ : M
      ⊢ Membership.mem Bot.bot x✝ → Membership.mem (HSMul.hSMul EmptyCollection.empt …
    -/
  · rintro rfl; exact Submodule.zero_mem _
                /-
                  🎉 no goals
                -/


@[simp] lemma set_smul_bot : s • (⊥ : Submodule R M) = ⊥ :=
                               /-
                                 R : Type u_2
                                 M : Type u_3
                                 inst✝⁴ : Semiring R
                                 inst✝³ : AddCommMonoid M
                                 inst✝² : Module R M
                                 S : Type u_4
                                 inst✝¹ : Monoid S
                                 inst✝ : DistribMulAction S M
                                 s : Set S
                                 x : M
                                 hx : Membership.mem (HSMul.hSMul s Bot.bot) x
                                 ⊢ Membership.mem Bot.bot x
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
  eq_bot_iff.mpr fun x hx ↦ by induction x, hx using set_smul_inductionOn <;> aesop
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


lemma singleton_set_smul [SMulCommClass S R M] (r : S) : ({r} : Set S) • N = r • N := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    N : Submodule R M
    inst✝ : SMulCommClass S R M
    r : S
    ⊢ Eq (HSMul.hSMul (Singleton.singleton r) N) (HSMul.hSMul r N)
  -/
  apply set_smul_eq_of_le
    /-
      case closed_under_smul
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      r : S
      ⊢ ∀ ⦃r_1 : S⦄ ⦃n : M⦄, Membership.mem (Singleton.singleton r) r_1 → Membership …
    -/
  · rintro _ m rfl hm; exact ⟨m, hm, rfl⟩
                       /-
                         🎉 no goals
                       -/
    /-
      case le
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      r : S
      ⊢ LE.le (HSMul.hSMul r N) (HSMul.hSMul (Singleton.singleton r) N)
    -/
  · rintro _ ⟨m, hm, rfl⟩
    /-
      case le.intro.intro
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      r : S
      m : M
      hm : Membership.mem (↑N) m
      ⊢ Membership.mem (HSMul.hSMul (Singleton.singleton r) N) ((DistribMulAction.to …
    -/
    rw [mem_set_smul_def, Submodule.mem_sInf]
    /-
      case le.intro.intro
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      r : S
      m : M
      hm : Membership.mem (↑N) m
      ⊢ ∀ (p : Submodule R M), Membership.mem (setOf fun p => ∀ ⦃r_1 : S⦄ {n : M}, M …
    -/
    intro _ hp; exact hp rfl hm
                /-
                  🎉 no goals
                -/


lemma mem_singleton_set_smul [SMulCommClass R S M] (r : S) (x : M) :
    x ∈ ({r} : Set S) • N ↔ ∃ (m : M), m ∈ N ∧ x = r • m := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    N : Submodule R M
    inst✝ : SMulCommClass R S M
    r : S
    x : M
    ⊢ Iff (Membership.mem (HSMul.hSMul (Singleton.singleton r) N) x) (Exists fun m …
  -/
  fconstructor
    /-
      case mp
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass R S M
      r : S
      x : M
      ⊢ Membership.mem (HSMul.hSMul (Singleton.singleton r) N) x → Exists fun m => A …
    -/
  · intro hx
    induction' x, hx using Submodule.set_smul_inductionOn with
      t n memₜ memₙ t n mem h m₁ m₂ mem₁ mem₂ h₁ h₂
      /-
        case mp.smul₀
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x : M
        t : S
        n : M
        memₜ : Membership.mem (Singleton.singleton r) t
        memₙ : Membership.mem N n
        ⊢ Exists fun m => And (Membership.mem N m) (Eq (HSMul.hSMul t n) (HSMul.hSMul  …
      -/
    · aesop
      /-
        🎉 no goals
      -/
      /-
        case mp.smul₁
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x : M
        t : R
        n : M
        mem : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) n
        h : Exists fun m => And (Membership.mem N m) (Eq n (HSMul.hSMul r m))
        ⊢ Exists fun m => And (Membership.mem N m) (Eq (HSMul.hSMul t n) (HSMul.hSMul  …
      -/
    · rcases h with ⟨n, hn, rfl⟩
      /-
        case mp.smul₁.intro.intro
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x : M
        t : R
        n : M
        hn : Membership.mem N n
        mem : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) (HSMul.hSMul r n)
        ⊢ Exists fun m => And (Membership.mem N m) (Eq (HSMul.hSMul t (HSMul.hSMul r n …
      -/
      exact ⟨t • n, by aesop,  smul_comm _ _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.add
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x m₁ m₂ : M
        mem₁ : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) m₁
        mem₂ : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) m₂
        h₁ : Exists fun m => And (Membership.mem N m) (Eq m₁ (HSMul.hSMul r m))
        h₂ : Exists fun m => And (Membership.mem N m) (Eq m₂ (HSMul.hSMul r m))
        ⊢ Exists fun m => And (Membership.mem N m) (Eq (HAdd.hAdd m₁ m₂) (HSMul.hSMul  …
      -/
    · rcases h₁ with ⟨m₁, h₁, rfl⟩
      /-
        case mp.add.intro.intro
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x m₂ : M
        mem₂ : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) m₂
        h₂ : Exists fun m => And (Membership.mem N m) (Eq m₂ (HSMul.hSMul r m))
        m₁ : M
        h₁ : Membership.mem N m₁
        mem₁ : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) (HSMul.hSMul r m₁)
        ⊢ Exists fun m => And (Membership.mem N m) (Eq (HAdd.hAdd (HSMul.hSMul r m₁) m …
      -/
      rcases h₂ with ⟨m₂, h₂, rfl⟩
      /-
        case mp.add.intro.intro.intro.intro
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x m₁ : M
        h₁ : Membership.mem N m₁
        mem₁ : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) (HSMul.hSMul r m₁)
        m₂ : M
        h₂ : Membership.mem N m₂
        mem₂ : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) (HSMul.hSMul r m₂)
        ⊢ Exists fun m => And (Membership.mem N m) (Eq (HAdd.hAdd (HSMul.hSMul r m₁) ( …
      -/
      exact ⟨m₁ + m₂, Submodule.add_mem _ h₁ h₂, by aesop⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.zero
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        N : Submodule R M
        inst✝ : SMulCommClass R S M
        r : S
        x : M
        ⊢ Exists fun m => And (Membership.mem N m) (Eq 0 (HSMul.hSMul r m))
      -/
    · exact ⟨0, Submodule.zero_mem _, by aesop⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass R S M
      r : S
      x : M
      ⊢ (Exists fun m => And (Membership.mem N m) (Eq x (HSMul.hSMul r m))) → Member …
    -/
  · aesop
    /-
      🎉 no goals
    -/


lemma smul_inductionOn_pointwise [SMulCommClass S R M] {a : S} {p : (x : M) → x ∈ a • N → Prop}
    (smul₀ : ∀ (s : M) (hs : s ∈ N), p (a • s) (Submodule.smul_mem_pointwise_smul _ _ _ hs))
    (smul₁ : ∀ (r : R) (m : M) (mem : m ∈ a • N), p m mem → p (r • m) (Submodule.smul_mem _ _ mem))
    (add : ∀ (x y : M) (hx : x ∈ a • N) (hy : y ∈ a • N),
      p x hx → p y hy → p (x + y) (Submodule.add_mem _ hx hy))
    (zero : p 0 (Submodule.zero_mem _)) {x : M} (hx : x ∈ a • N) :
    p x hx := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    S : Type u_4
    inst✝² : Monoid S
    inst✝¹ : DistribMulAction S M
    N : Submodule R M
    inst✝ : SMulCommClass S R M
    a : S
    p : (x : M) → Membership.mem (HSMul.hSMul a N) x → Prop
    smul₀ : ∀ (s : M) (hs : Membership.mem N s), p (HSMul.hSMul a s) ⋯
    smul₁ : ∀ (r : R) (m : M) (mem : Membership.mem (HSMul.hSMul a N) m), p m mem  …
    add : ∀ (x y : M) (hx : Membership.mem (HSMul.hSMul a N) x) (hy : Membership.m …
    zero : p 0 ⋯
    x : M
    hx : Membership.mem (HSMul.hSMul a N) x
    ⊢ p x hx
  -/
  simp_all only [← Submodule.singleton_set_smul]
  let p' (x : M) (hx : x ∈ ({a} : Set S) • N) : Prop :=
    p x (by rwa [← Submodule.singleton_set_smul])
  refine Submodule.set_smul_inductionOn (motive := p') _ (N.singleton_set_smul a ▸ hx)
      (fun r n hr hn ↦ ?_) smul₁ add zero
    /-
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      a : S
      p : (x : M) → Membership.mem (HSMul.hSMul a N) x → Prop
      smul₀ : ∀ (s : M) (hs : Membership.mem N s), p (HSMul.hSMul a s) ⋯
      x : M
      hx : Membership.mem (HSMul.hSMul a N) x
      smul₁ : ∀ (r : R) (m : M) (mem : Membership.mem (HSMul.hSMul (Singleton.single …
      add : ∀ (x y : M) (hx : Membership.mem (HSMul.hSMul (Singleton.singleton a) N) …
      zero : p 0 ⋯
      p' : (x : M) → Membership.mem (HSMul.hSMul (Singleton.singleton a) N) x → Prop …
      r : S
      n : M
      hr : Membership.mem (Singleton.singleton a) r
      hn : Membership.mem N n
      ⊢ p' (HSMul.hSMul r n) ⋯
    -/
  · simp only [Set.mem_singleton_iff] at hr
    /-
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      a : S
      p : (x : M) → Membership.mem (HSMul.hSMul a N) x → Prop
      smul₀ : ∀ (s : M) (hs : Membership.mem N s), p (HSMul.hSMul a s) ⋯
      x : M
      hx : Membership.mem (HSMul.hSMul a N) x
      smul₁ : ∀ (r : R) (m : M) (mem : Membership.mem (HSMul.hSMul (Singleton.single …
      add : ∀ (x y : M) (hx : Membership.mem (HSMul.hSMul (Singleton.singleton a) N) …
      zero : p 0 ⋯
      p' : (x : M) → Membership.mem (HSMul.hSMul (Singleton.singleton a) N) x → Prop …
      r : S
      n : M
      hr✝ : Membership.mem (Singleton.singleton a) r
      hn : Membership.mem N n
      hr : Eq r a
      ⊢ p' (HSMul.hSMul r n) ⋯
    -/
    subst hr
    /-
      R : Type u_2
      M : Type u_3
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      S : Type u_4
      inst✝² : Monoid S
      inst✝¹ : DistribMulAction S M
      N : Submodule R M
      inst✝ : SMulCommClass S R M
      x : M
      r : S
      n : M
      hn : Membership.mem N n
      p : (x : M) → Membership.mem (HSMul.hSMul r N) x → Prop
      smul₀ : ∀ (s : M) (hs : Membership.mem N s), p (HSMul.hSMul r s) ⋯
      hx : Membership.mem (HSMul.hSMul r N) x
      smul₁ : ∀ (r_1 : R) (m : M) (mem : Membership.mem (HSMul.hSMul (Singleton.sing …
      add : ∀ (x y : M) (hx : Membership.mem (HSMul.hSMul (Singleton.singleton r) N) …
      zero : p 0 ⋯
      p' : (x : M) → Membership.mem (HSMul.hSMul (Singleton.singleton r) N) x → Prop …
      hr : Membership.mem (Singleton.singleton r) r
      ⊢ p' (HSMul.hSMul r n) ⋯
    -/
    exact smul₀ n hn
    /-
      🎉 no goals
    -/

-- Note that this can't be generalized to `Set S`, because even though `SMulCommClass R R M` implies
-- `SMulComm R R N` for all `R`-submodules `N`, `SMulCommClass R S N` for all `R`-submodules `N`
-- does not make sense. If we just focus on `R`-submodules that are also `S`-submodule, then this
-- should be true.

/-- A subset of a ring `R` has a multiplicative action on submodules of a module over `R`. -/
protected def pointwiseSetMulAction [SMulCommClass R R M] :
    MulAction (Set R) (Submodule R M) where
  one_smul x := show {(1 : R)} • x = x from SetLike.ext fun m =>
                                             /-
                                               α : Type u_1
                                               R : Type u_2
                                               M : Type u_3
                                               inst✝⁵ : Semiring R
                                               inst✝⁴ : AddCommMonoid M
                                               inst✝³ : Module R M
                                               S : Type u_4
                                               inst✝² : Monoid S
                                               inst✝¹ : DistribMulAction S M
                                               sR : Set R
                                               s : Set S
                                               N : Submodule R M
                                               inst✝ : SMulCommClass R R M
                                               x : Submodule R M
                                               m : M
                                               ⊢ (Exists fun m_1 => And (Membership.mem x m_1) (Eq m (HSMul.hSMul 1 m_1))) →  …
                                             -/
    (mem_singleton_set_smul _ _ _).trans ⟨by rintro ⟨_, h, rfl⟩; rwa [one_smul],
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      fun h => ⟨m, h, (one_smul _ _).symm⟩⟩
  mul_smul s t x := le_antisymm
                             /-
                               α : Type u_1
                               R : Type u_2
                               M : Type u_3
                               inst✝⁵ : Semiring R
                               inst✝⁴ : AddCommMonoid M
                               inst✝³ : Module R M
                               S : Type u_4
                               inst✝² : Monoid S
                               inst✝¹ : DistribMulAction S M
                               sR : Set R
                               s✝ : Set S
                               N : Submodule R M
                               inst✝ : SMulCommClass R R M
                               s t : Set R
                               x : Submodule R M
                               ⊢ ∀ ⦃r : R⦄ ⦃n : M⦄, Membership.mem (HMul.hMul s t) r → Membership.mem x n → M …
                             -/
    (set_smul_le _ _ _ <| by rintro _ _ ⟨_, _, _, _, rfl⟩ _; rw [mul_smul]; aesop)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    (set_smul_le _ _ _ fun r m hr hm ↦ by
      /-
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s t : Set R
        x : Submodule R M
        r : R
        m : M
        hr : Membership.mem s r
        hm : Membership.mem (HSMul.hSMul t x) m
        ⊢ Membership.mem (HSMul.hSMul (HMul.hMul s t) x) (HSMul.hSMul r m)
      -/
      have : SMulCommClass R R x := ⟨fun r s m => Subtype.ext <| smul_comm _ _ _⟩
      /-
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s t : Set R
        x : Submodule R M
        r : R
        m : M
        hr : Membership.mem s r
        hm : Membership.mem (HSMul.hSMul t x) m
        this : SMulCommClass R R (Subtype fun x_1 => Membership.mem x x_1)
        ⊢ Membership.mem (HSMul.hSMul (HMul.hMul s t) x) (HSMul.hSMul r m)
      -/
      obtain ⟨c, hc1, rfl⟩ := mem_set_smul _ _ _ |>.mp hm
      /-
        case intro.intro
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s t : Set R
        x : Submodule R M
        r : R
        hr : Membership.mem s r
        this : SMulCommClass R R (Subtype fun x_1 => Membership.mem x x_1)
        c : Finsupp R (Subtype fun x_1 => Membership.mem x x_1)
        hc1 : HasSubset.Subset (↑c.support) t
        hm : Membership.mem (HSMul.hSMul t x) ↑(c.sum fun r m => HSMul.hSMul r m)
        ⊢ Membership.mem (HSMul.hSMul (HMul.hMul s t) x) (HSMul.hSMul r ↑(c.sum fun r  …
      -/
      rw [Finsupp.sum, AddSubmonoid.coe_finset_sum]
      /-
        case intro.intro
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s t : Set R
        x : Submodule R M
        r : R
        hr : Membership.mem s r
        this : SMulCommClass R R (Subtype fun x_1 => Membership.mem x x_1)
        c : Finsupp R (Subtype fun x_1 => Membership.mem x x_1)
        hc1 : HasSubset.Subset (↑c.support) t
        hm : Membership.mem (HSMul.hSMul t x) ↑(c.sum fun r m => HSMul.hSMul r m)
        ⊢ Membership.mem (HSMul.hSMul (HMul.hMul s t) x) (HSMul.hSMul r (c.support.sum …
      -/
      simp only [SetLike.val_smul, Finset.smul_sum, smul_smul]
      exact Submodule.sum_mem _ fun r' hr' ↦
        mem_set_smul_of_mem_mem (Set.mul_mem_mul hr (hc1 hr')) (c _).2)


/-- In a ring, sets acts on submodules. -/
protected def pointwiseSetDistribMulAction [SMulCommClass R R M] :
    DistribMulAction (Set R) (Submodule R M) where
  smul_zero s := set_smul_bot s
  smul_add s x y := le_antisymm
    (set_smul_le _ _ _ <| by
      /-
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s : Set R
        x y : Submodule R M
        ⊢ ∀ ⦃r : R⦄ ⦃n : M⦄, Membership.mem s r → Membership.mem (HAdd.hAdd x y) n → M …
      -/
      rintro r m hr hm
      /-
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s : Set R
        x y : Submodule R M
        r : R
        m : M
        hr : Membership.mem s r
        hm : Membership.mem (HAdd.hAdd x y) m
        ⊢ Membership.mem (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul s y)) (HSMul.hSMul  …
      -/
      rw [add_eq_sup, Submodule.mem_sup] at hm
      /-
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s : Set R
        x y : Submodule R M
        r : R
        m : M
        hr : Membership.mem s r
        hm : Exists fun y_1 => And (Membership.mem x y_1) (Exists fun z => And (Member …
        ⊢ Membership.mem (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul s y)) (HSMul.hSMul  …
      -/
      obtain ⟨a, ha, b, hb, rfl⟩ := hm
      /-
        case intro.intro.intro.intro
        α : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        S : Type u_4
        inst✝² : Monoid S
        inst✝¹ : DistribMulAction S M
        sR : Set R
        s✝ : Set S
        N : Submodule R M
        inst✝ : SMulCommClass R R M
        s : Set R
        x y : Submodule R M
        r : R
        hr : Membership.mem s r
        a : M
        ha : Membership.mem x a
        b : M
        hb : Membership.mem y b
        ⊢ Membership.mem (HAdd.hAdd (HSMul.hSMul s x) (HSMul.hSMul s y)) (HSMul.hSMul  …
      -/
      rw [smul_add, add_eq_sup, Submodule.mem_sup]
      exact ⟨r • a, mem_set_smul_of_mem_mem (mem1 := hr) (mem2 := ha),
        r • b, mem_set_smul_of_mem_mem (mem1 := hr) (mem2 := hb), rfl⟩)
    (sup_le_iff.mpr ⟨smul_mono_right _ le_sup_left, smul_mono_right _ le_sup_right⟩)


lemma sup_set_smul (s t : Set S) :
    (s ⊔ t) • N = s • N ⊔ t • N :=
  set_smul_eq_of_le _ _ _
        /-
          R : Type u_2
          M : Type u_3
          inst✝⁴ : Semiring R
          inst✝³ : AddCommMonoid M
          inst✝² : Module R M
          S : Type u_4
          inst✝¹ : Monoid S
          inst✝ : DistribMulAction S M
          N : Submodule R M
          s t : Set S
          ⊢ ∀ ⦃r : S⦄ ⦃n : M⦄, Membership.mem (Max.max s t) r → Membership.mem N n → Mem …
        -/
    (by rintro _ _ (hr|hr) hn
          /-
            case inl
            R : Type u_2
            M : Type u_3
            inst✝⁴ : Semiring R
            inst✝³ : AddCommMonoid M
            inst✝² : Module R M
            S : Type u_4
            inst✝¹ : Monoid S
            inst✝ : DistribMulAction S M
            N : Submodule R M
            s t : Set S
            r✝ : S
            n✝ : M
            hr : Membership.mem s r✝
            hn : Membership.mem N n✝
            ⊢ Membership.mem (Max.max (HSMul.hSMul s N) (HSMul.hSMul t N)) (HSMul.hSMul r✝ …
          -/
        · exact Submodule.mem_sup_left (mem_set_smul_of_mem_mem hr hn)
          /-
            🎉 no goals
          -/
          /-
            case inr
            R : Type u_2
            M : Type u_3
            inst✝⁴ : Semiring R
            inst✝³ : AddCommMonoid M
            inst✝² : Module R M
            S : Type u_4
            inst✝¹ : Monoid S
            inst✝ : DistribMulAction S M
            N : Submodule R M
            s t : Set S
            r✝ : S
            n✝ : M
            hr : Membership.mem t r✝
            hn : Membership.mem N n✝
            ⊢ Membership.mem (Max.max (HSMul.hSMul s N) (HSMul.hSMul t N)) (HSMul.hSMul r✝ …
          -/
        · exact Submodule.mem_sup_right (mem_set_smul_of_mem_mem hr hn))
          /-
            🎉 no goals
          -/
    (sup_le (set_smul_mono_left _ le_sup_left) (set_smul_mono_left _ le_sup_right))


