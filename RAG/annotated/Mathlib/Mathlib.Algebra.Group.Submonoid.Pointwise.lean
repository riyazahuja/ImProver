@[to_additive (attr := simp, norm_cast)]
lemma coe_mul_coe [SetLike S M] [SubmonoidClass S M] (H : S) : H * H = (H : Set M) := by
  /-
    M : Type u_3
    S : Type u_6
    inst✝² : Monoid M
    inst✝¹ : SetLike S M
    inst✝ : SubmonoidClass S M
    H : S
    ⊢ Eq (HMul.hMul ↑H ↑H) ↑H
  -/
  aesop (add simp mem_mul)
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
@[to_additive (attr := simp)]
lemma coe_set_pow [SetLike S M] [SubmonoidClass S M] :
    ∀ {n} (hn : n ≠ 0) (H : S), (H ^ n : Set M) = H
                  /-
                    M : Type u_3
                    S : Type u_6
                    inst✝² : Monoid M
                    inst✝¹ : SetLike S M
                    inst✝ : SubmonoidClass S M
                    x✝ : Ne 1 0
                    H : S
                    ⊢ Eq (HPow.hPow (↑H) 1) ↑H
                  -/
  | 1, _, H => by simp
                  /-
                    🎉 no goals
                  -/
                      /-
                        M : Type u_3
                        S : Type u_6
                        inst✝² : Monoid M
                        inst✝¹ : SetLike S M
                        inst✝ : SubmonoidClass S M
                        n : Nat
                        x✝ : Ne (HAdd.hAdd n 2) 0
                        H : S
                        ⊢ Eq (HPow.hPow (↑H) (HAdd.hAdd n 2)) ↑H
                      -/
  | n + 2, _, H => by rw [pow_succ, coe_set_pow n.succ_ne_zero, coe_mul_coe]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem mul_subset {S : Submonoid M} (hs : s ⊆ S) (ht : t ⊆ S) : s * t ⊆ S :=
  mul_subset_iff.2 fun _x hx _y hy ↦ mul_mem (hs hx) (ht hy)


@[to_additive]
theorem mul_subset_closure (hs : s ⊆ u) (ht : t ⊆ u) : s * t ⊆ Submonoid.closure u :=
  mul_subset (Subset.trans hs Submonoid.subset_closure) (Subset.trans ht Submonoid.subset_closure)


@[to_additive]
theorem coe_mul_self_eq (s : Submonoid M) : (s : Set M) * s = s := by
  /-
    M : Type u_3
    inst✝ : Monoid M
    s : Submonoid M
    ⊢ Eq (HMul.hMul ↑s ↑s) ↑s
  -/
  ext x
  /-
    case h
    M : Type u_3
    inst✝ : Monoid M
    s : Submonoid M
    x : M
    ⊢ Iff (Membership.mem (HMul.hMul ↑s ↑s) x) (Membership.mem (↑s) x)
  -/
  refine ⟨?_, fun h => ⟨x, h, 1, s.one_mem, mul_one x⟩⟩
  /-
    case h
    M : Type u_3
    inst✝ : Monoid M
    s : Submonoid M
    x : M
    ⊢ Membership.mem (HMul.hMul ↑s ↑s) x → Membership.mem (↑s) x
  -/
  rintro ⟨a, ha, b, hb, rfl⟩
  /-
    case h.intro.intro.intro.intro
    M : Type u_3
    inst✝ : Monoid M
    s : Submonoid M
    a : M
    ha : Membership.mem (↑s) a
    b : M
    hb : Membership.mem (↑s) b
    ⊢ Membership.mem (↑s) ((fun x1 x2 => HMul.hMul x1 x2) a b)
  -/
  exact s.mul_mem ha hb
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_mul_le (S T : Set M) : closure (S * T) ≤ closure S ⊔ closure T :=
  sInf_le fun _x ⟨_s, hs, _t, ht, hx⟩ => hx ▸
    (closure S ⊔ closure T).mul_mem (SetLike.le_def.mp le_sup_left <| subset_closure hs)
      (SetLike.le_def.mp le_sup_right <| subset_closure ht)


@[to_additive]
lemma closure_pow_le : ∀ {n}, n ≠ 0 → closure (s ^ n) ≤ closure s
               /-
                 M : Type u_3
                 inst✝ : Monoid M
                 s : Set M
                 x✝ : Ne 1 0
                 ⊢ LE.le (Submonoid.closure (HPow.hPow s 1)) (Submonoid.closure s)
               -/
  | 1, _ => by simp
               /-
                 🎉 no goals
               -/
  | n + 2, _ =>
    calc
      closure (s ^ (n + 2))
                                          /-
                                            M : Type u_3
                                            inst✝ : Monoid M
                                            s : Set M
                                            n : Nat
                                            x✝ : Ne (HAdd.hAdd n 2) 0
                                            ⊢ Eq (Submonoid.closure (HPow.hPow s (HAdd.hAdd n 2))) (Submonoid.closure (HMu …
                                          -/
      _ = closure (s ^ (n + 1) * s) := by rw [pow_succ]
                                          /-
                                            🎉 no goals
                                          -/
      _ ≤ closure (s ^ (n + 1)) ⊔ closure s := closure_mul_le ..
                                      /-
                                        M : Type u_3
                                        inst✝ : Monoid M
                                        s : Set M
                                        n : Nat
                                        x✝ : Ne (HAdd.hAdd n 2) 0
                                        ⊢ LE.le (Max.max (Submonoid.closure (HPow.hPow s (HAdd.hAdd n 1))) (Submonoid. …
                                      -/
      _ ≤ closure s ⊔ closure s := by gcongr ?_ ⊔ _; exact closure_pow_le n.succ_ne_zero
                                                     /-
                                                       🎉 no goals
                                                     -/
      _ = closure s := sup_idem _


@[to_additive]
lemma closure_pow {n : ℕ} (hs : 1 ∈ s) (hn : n ≠ 0) : closure (s ^ n) = closure s :=
                                     /-
                                       M : Type u_3
                                       inst✝ : Monoid M
                                       s : Set M
                                       n : Nat
                                       hs : Membership.mem s 1
                                       hn : Ne n 0
                                       ⊢ LE.le (Submonoid.closure s) (Submonoid.closure (HPow.hPow s n))
                                     -/
  (closure_pow_le hn).antisymm <| by gcongr; exact subset_pow hs hn
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive]
theorem sup_eq_closure_mul (H K : Submonoid M) : H ⊔ K = closure ((H : Set M) * (K : Set M)) :=
  le_antisymm
    (sup_le (fun h hh => subset_closure ⟨h, hh, 1, K.one_mem, mul_one h⟩) fun k hk =>
      subset_closure ⟨1, H.one_mem, k, hk, one_mul k⟩)
                                      /-
                                        M : Type u_3
                                        inst✝ : Monoid M
                                        H K : Submonoid M
                                        ⊢ LE.le (Max.max (Submonoid.closure ↑H) (Submonoid.closure ↑K)) (Max.max H K)
                                      -/
    ((closure_mul_le _ _).trans <| by rw [closure_eq, closure_eq])
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
theorem pow_smul_mem_closure_smul {N : Type*} [CommMonoid N] [MulAction M N] [IsScalarTower M N N]
    (r : M) (s : Set N) {x : N} (hx : x ∈ closure s) : ∃ n : ℕ, r ^ n • x ∈ closure (r • s) := by
  induction hx using closure_induction with
  | mem x hx => exact ⟨1, subset_closure ⟨_, hx, by rw [pow_one]⟩⟩
  | one => exact ⟨0, by simpa using one_mem _⟩
  | mul x y _ _ hx hy =>
    obtain ⟨⟨nx, hx⟩, ⟨ny, hy⟩⟩ := And.intro hx hy
    use ny + nx
    rw [pow_add, mul_smul, ← smul_mul_assoc, mul_comm, ← smul_mul_assoc]
    exact mul_mem hy hx


/-- The submonoid with every element inverted. -/
@[to_additive " The additive submonoid with every element negated. "]
protected def inv : Inv (Submonoid G) where
  inv S :=
    { carrier := (S : Set G)⁻¹
                                  /-
                                    α : Type u_1
                                    G : Type u_2
                                    M : Type u_3
                                    R : Type u_4
                                    A : Type u_5
                                    S✝ : Type u_6
                                    inst✝² : Monoid M
                                    inst✝¹ : AddMonoid A
                                    s t u : Set M
                                    inst✝ : Group G
                                    S : Submonoid G
                                    a✝ b✝ : G
                                    ha : Membership.mem (Inv.inv ↑S) a✝
                                    hb : Membership.mem (Inv.inv ↑S) b✝
                                    ⊢ Membership.mem (Inv.inv ↑S) (HMul.hMul a✝ b✝)
                                  -/
      mul_mem' := fun ha hb => by rw [mem_inv, mul_inv_rev]; exact mul_mem hb ha
                                                             /-
                                                               🎉 no goals
                                                             -/
                                  /-
                                    α : Type u_1
                                    G : Type u_2
                                    M : Type u_3
                                    R : Type u_4
                                    A : Type u_5
                                    S✝ : Type u_6
                                    inst✝² : Monoid M
                                    inst✝¹ : AddMonoid A
                                    s t u : Set M
                                    inst✝ : Group G
                                    S : Submonoid G
                                    ⊢ Membership.mem (↑S) (Inv.inv 1)
                                  -/
      one_mem' := mem_inv.2 <| by rw [inv_one]; exact S.one_mem' }
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := simp)]
theorem coe_inv (S : Submonoid G) : ↑S⁻¹ = (S : Set G)⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem mem_inv {g : G} {S : Submonoid G} : g ∈ S⁻¹ ↔ g⁻¹ ∈ S :=
  Iff.rfl


/-- Inversion is involutive on submonoids. -/
@[to_additive "Inversion is involutive on additive submonoids."]
def involutiveInv : InvolutiveInv (Submonoid G) :=
  SetLike.coe_injective.involutiveInv _ fun _ => rfl


@[to_additive (attr := simp)]
theorem inv_le_inv (S T : Submonoid G) : S⁻¹ ≤ T⁻¹ ↔ S ≤ T :=
  SetLike.coe_subset_coe.symm.trans Set.inv_subset_inv


@[to_additive]
theorem inv_le (S T : Submonoid G) : S⁻¹ ≤ T ↔ S ≤ T⁻¹ :=
  SetLike.coe_subset_coe.symm.trans Set.inv_subset


/-- Pointwise inversion of submonoids as an order isomorphism. -/
@[to_additive (attr := simps!) "Pointwise negation of additive submonoids as an order isomorphism"]
def invOrderIso : Submonoid G ≃o Submonoid G where
  toEquiv := Equiv.inv _
  map_rel_iff' := inv_le_inv _ _


@[to_additive]
theorem closure_inv (s : Set G) : closure s⁻¹ = (closure s)⁻¹ := by
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    ⊢ Eq (Submonoid.closure (Inv.inv s)) (Inv.inv (Submonoid.closure s))
  -/
  apply le_antisymm
    /-
      case a
      G : Type u_2
      inst✝ : Group G
      s : Set G
      ⊢ LE.le (Submonoid.closure (Inv.inv s)) (Inv.inv (Submonoid.closure s))
    -/
  · rw [closure_le, coe_inv, ← Set.inv_subset, inv_inv]
    /-
      case a
      G : Type u_2
      inst✝ : Group G
      s : Set G
      ⊢ HasSubset.Subset s ↑(Submonoid.closure s)
    -/
    exact subset_closure
    /-
      🎉 no goals
    -/
    /-
      case a
      G : Type u_2
      inst✝ : Group G
      s : Set G
      ⊢ LE.le (Inv.inv (Submonoid.closure s)) (Submonoid.closure (Inv.inv s))
    -/
  · rw [inv_le, closure_le, coe_inv, ← Set.inv_subset]
    /-
      case a
      G : Type u_2
      inst✝ : Group G
      s : Set G
      ⊢ HasSubset.Subset (Inv.inv s) ↑(Submonoid.closure (Inv.inv s))
    -/
    exact subset_closure
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem inv_inf (S T : Submonoid G) : (S ⊓ T)⁻¹ = S⁻¹ ⊓ T⁻¹ :=
  SetLike.coe_injective Set.inter_inv


@[to_additive (attr := simp)]
theorem inv_sup (S T : Submonoid G) : (S ⊔ T)⁻¹ = S⁻¹ ⊔ T⁻¹ :=
  (invOrderIso : Submonoid G ≃o Submonoid G).map_sup S T


@[to_additive (attr := simp)]
theorem inv_bot : (⊥ : Submonoid G)⁻¹ = ⊥ :=
  SetLike.coe_injective <| (Set.inv_singleton 1).trans <| congr_arg _ inv_one


@[to_additive (attr := simp)]
theorem inv_top : (⊤ : Submonoid G)⁻¹ = ⊤ :=
  SetLike.coe_injective <| Set.inv_univ


@[to_additive (attr := simp)]
theorem inv_iInf {ι : Sort*} (S : ι → Submonoid G) : (⨅ i, S i)⁻¹ = ⨅ i, (S i)⁻¹ :=
  (invOrderIso : Submonoid G ≃o Submonoid G).map_iInf _


@[to_additive (attr := simp)]
theorem inv_iSup {ι : Sort*} (S : ι → Submonoid G) : (⨆ i, S i)⁻¹ = ⨆ i, (S i)⁻¹ :=
  (invOrderIso : Submonoid G ≃o Submonoid G).map_iSup _


/-- The action on a submonoid corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseMulAction : MulAction α (Submonoid M) where
  smul a S := S.map (MulDistribMulAction.toMonoidEnd _ M a)
  one_smul S := by
    /-
      α : Type u_1
      G : Type u_2
      M : Type u_3
      R : Type u_4
      A : Type u_5
      S✝ : Type u_6
      inst✝³ : Monoid M
      inst✝² : AddMonoid A
      inst✝¹ : Monoid α
      inst✝ : MulDistribMulAction α M
      S : Submonoid M
      ⊢ Eq (HSMul.hSMul 1 S) S
    -/
    change S.map _ = S
    /-
      α : Type u_1
      G : Type u_2
      M : Type u_3
      R : Type u_4
      A : Type u_5
      S✝ : Type u_6
      inst✝³ : Monoid M
      inst✝² : AddMonoid A
      inst✝¹ : Monoid α
      inst✝ : MulDistribMulAction α M
      S : Submonoid M
      ⊢ Eq (Submonoid.map ((MulDistribMulAction.toMonoidEnd α M) 1) S) S
    -/
    simpa only [map_one] using S.map_id
    /-
      🎉 no goals
    -/
  mul_smul _ _ S :=
    (congr_arg (fun f : Monoid.End M => S.map f) (MonoidHom.map_mul _ _ _)).trans
      (S.map_map _ _).symm


@[simp]
theorem coe_pointwise_smul (a : α) (S : Submonoid M) : ↑(a • S) = a • (S : Set M) :=
  rfl


theorem smul_mem_pointwise_smul (m : M) (a : α) (S : Submonoid M) : m ∈ S → a • m ∈ a • S :=
  (Set.smul_mem_smul_set : _ → _ ∈ a • (S : Set M))


instance : CovariantClass α (Submonoid M) HSMul.hSMul LE.le :=
  ⟨fun _ _ => image_subset _⟩


theorem mem_smul_pointwise_iff_exists (m : M) (a : α) (S : Submonoid M) :
    m ∈ a • S ↔ ∃ s : M, s ∈ S ∧ a • s = m :=
  (Set.mem_smul_set : m ∈ a • (S : Set M) ↔ _)


@[simp]
theorem smul_bot (a : α) : a • (⊥ : Submonoid M) = ⊥ :=
  map_bot _


theorem smul_sup (a : α) (S T : Submonoid M) : a • (S ⊔ T) = a • S ⊔ a • T :=
  map_sup _ _ _


theorem smul_closure (a : α) (s : Set M) : a • closure s = closure (a • s) :=
  MonoidHom.map_mclosure _ _


lemma pointwise_isCentralScalar [MulDistribMulAction αᵐᵒᵖ M] [IsCentralScalar α M] :
    IsCentralScalar α (Submonoid M) :=
  ⟨fun _ S => (congr_arg fun f : Monoid.End M => S.map f) <| MonoidHom.ext <| op_smul_eq_smul _⟩


@[simp]
theorem smul_mem_pointwise_smul_iff {a : α} {S : Submonoid M} {x : M} : a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff


theorem mem_pointwise_smul_iff_inv_smul_mem {a : α} {S : Submonoid M} {x : M} :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem


theorem mem_inv_pointwise_smul_iff {a : α} {S : Submonoid M} {x : M} : x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff {a : α} {S T : Submonoid M} : a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff


theorem pointwise_smul_subset_iff {a : α} {S T : Submonoid M} : a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff


theorem subset_pointwise_smul_iff {a : α} {S T : Submonoid M} : S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff


@[simp]
theorem smul_mem_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : Submonoid M) (x : M) :
    a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff₀ ha (S : Set M) x


theorem mem_pointwise_smul_iff_inv_smul_mem₀ {a : α} (ha : a ≠ 0) (S : Submonoid M) (x : M) :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem₀ ha (S : Set M) x


theorem mem_inv_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : Submonoid M) (x : M) :
    x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff₀ ha (S : Set M) x


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : Submonoid M} :
    a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff₀ ha


theorem pointwise_smul_le_iff₀ {a : α} (ha : a ≠ 0) {S T : Submonoid M} : a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff₀ ha


theorem le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : Submonoid M} : S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff₀ ha


@[to_additive]
theorem mem_closure_inv {G : Type*} [Group G] (S : Set G) (x : G) :
                                                                /-
                                                                  G : Type u_7
                                                                  inst✝ : Group G
                                                                  S : Set G
                                                                  x : G
                                                                  ⊢ Iff (Membership.mem (Submonoid.closure (Inv.inv S)) x) (Membership.mem (Subm …
                                                                -/
    x ∈ Submonoid.closure S⁻¹ ↔ x⁻¹ ∈ Submonoid.closure S := by rw [closure_inv, mem_inv]
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The action on an additive submonoid corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseMulAction : MulAction α (AddSubmonoid A) where
  smul a S := S.map (DistribMulAction.toAddMonoidEnd _ A a)
  one_smul S :=
    (congr_arg (fun f : AddMonoid.End A => S.map f) (MonoidHom.map_one _)).trans S.map_id
  mul_smul _ _ S :=
    (congr_arg (fun f : AddMonoid.End A => S.map f) (MonoidHom.map_mul _ _ _)).trans
      (S.map_map _ _).symm


@[simp]
theorem coe_pointwise_smul (a : α) (S : AddSubmonoid A) : ↑(a • S) = a • (S : Set A) :=
  rfl


theorem smul_mem_pointwise_smul (m : A) (a : α) (S : AddSubmonoid A) : m ∈ S → a • m ∈ a • S :=
  (Set.smul_mem_smul_set : _ → _ ∈ a • (S : Set A))


theorem mem_smul_pointwise_iff_exists (m : A) (a : α) (S : AddSubmonoid A) :
    m ∈ a • S ↔ ∃ s : A, s ∈ S ∧ a • s = m :=
  (Set.mem_smul_set : m ∈ a • (S : Set A) ↔ _)


@[simp]
theorem smul_bot (a : α) : a • (⊥ : AddSubmonoid A) = ⊥ :=
  map_bot _


theorem smul_sup (a : α) (S T : AddSubmonoid A) : a • (S ⊔ T) = a • S ⊔ a • T :=
  map_sup _ _ _


@[simp]
theorem smul_closure (a : α) (s : Set A) : a • closure s = closure (a • s) :=
  AddMonoidHom.map_mclosure _ _


lemma pointwise_isCentralScalar [DistribMulAction αᵐᵒᵖ A] [IsCentralScalar α A] :
    IsCentralScalar α (AddSubmonoid A) :=
  ⟨fun _ S =>
    (congr_arg fun f : AddMonoid.End A => S.map f) <| AddMonoidHom.ext <| op_smul_eq_smul _⟩


@[simp]
theorem smul_mem_pointwise_smul_iff {a : α} {S : AddSubmonoid A} {x : A} : a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff


theorem mem_pointwise_smul_iff_inv_smul_mem {a : α} {S : AddSubmonoid A} {x : A} :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem


theorem mem_inv_pointwise_smul_iff {a : α} {S : AddSubmonoid A} {x : A} : x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff {a : α} {S T : AddSubmonoid A} :
    a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff


theorem pointwise_smul_le_iff {a : α} {S T : AddSubmonoid A} : a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff


theorem le_pointwise_smul_iff {a : α} {S T : AddSubmonoid A} : S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff


@[simp]
theorem smul_mem_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : AddSubmonoid A) (x : A) :
    a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff₀ ha (S : Set A) x


theorem mem_pointwise_smul_iff_inv_smul_mem₀ {a : α} (ha : a ≠ 0) (S : AddSubmonoid A) (x : A) :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem₀ ha (S : Set A) x


theorem mem_inv_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : AddSubmonoid A) (x : A) :
    x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff₀ ha (S : Set A) x


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : AddSubmonoid A} :
    a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff₀ ha


theorem pointwise_smul_le_iff₀ {a : α} (ha : a ≠ 0) {S T : AddSubmonoid A} :
    a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff₀ ha


theorem le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : AddSubmonoid A} :
    S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff₀ ha


/-- If `R` is an additive monoid with one (e.g., a semiring), then `1 : AddSubmonoid R` is the range
of `Nat.cast : ℕ → R`. -/
protected def one : One (AddSubmonoid R) :=
  ⟨AddMonoidHom.mrange (Nat.castAddMonoidHom R)⟩

theorem one_eq_mrange : (1 : AddSubmonoid R) = AddMonoidHom.mrange (Nat.castAddMonoidHom R) :=
  rfl


theorem natCast_mem_one (n : ℕ) : (n : R) ∈ (1 : AddSubmonoid R) :=
  ⟨_, rfl⟩


@[simp]
theorem mem_one {x : R} : x ∈ (1 : AddSubmonoid R) ↔ ∃ n : ℕ, ↑n = x :=
  Iff.rfl


theorem one_eq_closure : (1 : AddSubmonoid R) = closure {1} := by
  /-
    R : Type u_4
    inst✝ : AddMonoidWithOne R
    ⊢ Eq 1 (AddSubmonoid.closure (Singleton.singleton 1))
  -/
  rw [closure_singleton_eq, one_eq_mrange]
  /-
    R : Type u_4
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (AddMonoidHom.mrange (Nat.castAddMonoidHom R)) (AddMonoidHom.mrange ((mul …
  -/
  congr 1
  /-
    case e_f
    R : Type u_4
    inst✝ : AddMonoidWithOne R
    ⊢ Eq (Nat.castAddMonoidHom R) ((multiplesHom R) 1)
  -/
  ext
  /-
    case e_f.a
    R : Type u_4
    inst✝ : AddMonoidWithOne R
    ⊢ Eq ((Nat.castAddMonoidHom R) 1) (((multiplesHom R) 1) 1)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem one_eq_closure_one_set : (1 : AddSubmonoid R) = closure 1 :=
  one_eq_closure


/-- For `M : Submonoid R` and `N : AddSubmonoid A`, `M • N` is the additive submonoid
generated by all `m • n` where `m ∈ M` and `n ∈ N`. -/
protected def smul : SMul (AddSubmonoid R) (AddSubmonoid A) where
  smul M N := ⨆ s : M, N.map (DistribSMul.toAddMonoidHom A s.1)


theorem smul_mem_smul (hm : m ∈ M) (hn : n ∈ N) : m • n ∈ M • N :=
                                             /-
                                               R : Type u_4
                                               A : Type u_5
                                               inst✝² : AddMonoid A
                                               inst✝¹ : AddMonoid R
                                               inst✝ : DistribSMul R A
                                               M : AddSubmonoid R
                                               N : AddSubmonoid A
                                               m : R
                                               n : A
                                               hm : Membership.mem M m
                                               hn : Membership.mem N n
                                               ⊢ Eq ((DistribSMul.toAddMonoidHom A ↑⟨m, hm⟩) n) (HSMul.hSMul m n)
                                             -/
  (le_iSup _ ⟨m, hm⟩ : _ ≤ M • N) ⟨n, hn, by rfl⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem smul_le : M • N ≤ P ↔ ∀ m ∈ M, ∀ n ∈ N, m • n ∈ P :=
  ⟨fun H _m hm _n hn => H <| smul_mem_smul hm hn, fun H =>
    iSup_le fun ⟨m, hm⟩ => map_le_iff_le_comap.2 fun n hn => H m hm n hn⟩


@[elab_as_elim]
protected theorem smul_induction_on {C : A → Prop} {a : A} (ha : a ∈ M • N)
    (hm : ∀ m ∈ M, ∀ n ∈ N, C (m • n)) (hadd : ∀ x y, C x → C y → C (x + y)) : C a :=
  (@smul_le _ _ _ _ _ _ _ ⟨⟨setOf C, hadd _ _⟩, by
    /-
      R : Type u_4
      A : Type u_5
      inst✝² : AddMonoid A
      inst✝¹ : AddMonoid R
      inst✝ : DistribSMul R A
      M : AddSubmonoid R
      N : AddSubmonoid A
      C : A → Prop
      a : A
      ha : Membership.mem (HSMul.hSMul M N) a
      hm : ∀ (m : R), Membership.mem M m → ∀ (n : A), Membership.mem N n → C (HSMul. …
      hadd : ∀ (x y : A), C x → C y → C (HAdd.hAdd x y)
      ⊢ Membership.mem { carrier := setOf C, add_mem' := ⋯ }.carrier 0
    -/
    simpa only [smul_zero] using hm _ (zero_mem _) _ (zero_mem _)⟩).2 hm ha
    /-
      🎉 no goals
    -/


@[simp]
theorem addSubmonoid_smul_bot (S : AddSubmonoid R) : S • (⊥ : AddSubmonoid A) = ⊥ :=
  eq_bot_iff.2 <| smul_le.2 fun m _ n hn => by
    /-
      R : Type u_4
      A : Type u_5
      inst✝² : AddMonoid A
      inst✝¹ : AddMonoid R
      inst✝ : DistribSMul R A
      S : AddSubmonoid R
      m : R
      x✝ : Membership.mem S m
      n : A
      hn : Membership.mem Bot.bot n
      ⊢ Membership.mem Bot.bot (HSMul.hSMul m n)
    -/
    rw [AddSubmonoid.mem_bot] at hn ⊢; rw [hn, smul_zero]
                                       /-
                                         🎉 no goals
                                       -/


theorem smul_le_smul (h : M ≤ M') (hnp : N ≤ P) : M • N ≤ M' • P :=
  smul_le.2 fun _m hm _n hn => smul_mem_smul (h hm) (hnp hn)


theorem smul_le_smul_left (h : M ≤ M') : M • P ≤ M' • P :=
  smul_le_smul h le_rfl


theorem smul_le_smul_right (h : N ≤ P) : M • N ≤ M • P :=
  smul_le_smul le_rfl h


theorem smul_subset_smul : (↑M : Set R) • (↑N : Set A) ⊆ (↑(M • N) : Set A) :=
  smul_subset_iff.2 fun _i hi _j hj ↦ smul_mem_smul hi hj


theorem addSubmonoid_smul_sup : M • (N ⊔ P) = M • N ⊔ M • P :=
  le_antisymm (smul_le.mpr fun m hm np hnp ↦ by
    /-
      R : Type u_4
      A : Type u_5
      inst✝² : AddMonoid A
      inst✝¹ : AddMonoid R
      inst✝ : DistribSMul R A
      M : AddSubmonoid R
      N P : AddSubmonoid A
      m : R
      hm : Membership.mem M m
      np : A
      hnp : Membership.mem (Max.max N P) np
      ⊢ Membership.mem (Max.max (HSMul.hSMul M N) (HSMul.hSMul M P)) (HSMul.hSMul m  …
    -/
    refine closure_induction (p := (fun _ ↦ _ • · ∈ _)) ?_ ?_ ?_ (sup_eq_closure N P ▸ hnp)
      /-
        case refine_1
        R : Type u_4
        A : Type u_5
        inst✝² : AddMonoid A
        inst✝¹ : AddMonoid R
        inst✝ : DistribSMul R A
        M : AddSubmonoid R
        N P : AddSubmonoid A
        m : R
        hm : Membership.mem M m
        np : A
        hnp : Membership.mem (Max.max N P) np
        ⊢ ∀ (x : A) (h : Membership.mem (Union.union ↑N ↑P) x), (fun x x_1 => Membersh …
      -/
    · rintro x (hx | hx)
      exacts [le_sup_left (a := M • N) (smul_mem_smul hm hx),
        le_sup_right (a := M • N) (smul_mem_smul hm hx)]
      /-
        case refine_2
        R : Type u_4
        A : Type u_5
        inst✝² : AddMonoid A
        inst✝¹ : AddMonoid R
        inst✝ : DistribSMul R A
        M : AddSubmonoid R
        N P : AddSubmonoid A
        m : R
        hm : Membership.mem M m
        np : A
        hnp : Membership.mem (Max.max N P) np
        ⊢ (fun x x_1 => Membership.mem (Max.max (HSMul.hSMul M N) (HSMul.hSMul M P)) ( …
      -/
    · apply (smul_zero (A := A) m).symm ▸ (M • N ⊔ M • P).zero_mem
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        R : Type u_4
        A : Type u_5
        inst✝² : AddMonoid A
        inst✝¹ : AddMonoid R
        inst✝ : DistribSMul R A
        M : AddSubmonoid R
        N P : AddSubmonoid A
        m : R
        hm : Membership.mem M m
        np : A
        hnp : Membership.mem (Max.max N P) np
        ⊢ ∀ (x y : A) (hx : Membership.mem (AddSubmonoid.closure (Union.union ↑N ↑P))  …
      -/
    · intros _ _ _ _ h1 h2; rw [smul_add]; exact add_mem h1 h2)
                                           /-
                                             🎉 no goals
                                           -/
  (sup_le (smul_le_smul_right le_sup_left) <| smul_le_smul_right le_sup_right)


theorem smul_iSup (T : AddSubmonoid R) (S : ι → AddSubmonoid A) : (T • ⨆ i, S i) = ⨆ i, T • S i :=
  le_antisymm (smul_le.mpr fun t ht s hs ↦ iSup_induction _ (C := (t • · ∈ _)) hs
    (fun i s hs ↦ mem_iSup_of_mem i <| smul_mem_smul ht hs)
        /-
          R : Type u_4
          A : Type u_5
          inst✝² : AddMonoid A
          inst✝¹ : AddMonoid R
          inst✝ : DistribSMul R A
          ι : Sort u_7
          T : AddSubmonoid R
          S : ι → AddSubmonoid A
          t : R
          ht : Membership.mem T t
          s : A
          hs : Membership.mem (iSup fun i => S i) s
          ⊢ (fun x => Membership.mem (iSup fun i => HSMul.hSMul T (S i)) (HSMul.hSMul t  …
        -/
                             /-
                               🎉 no goals
                             -/
    (by simp_rw [smul_zero]; apply zero_mem) fun x y ↦ by simp_rw [smul_add]; apply add_mem)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  (iSup_le fun i ↦ smul_le_smul_right <| le_iSup _ i)


/-- Multiplication of additive submonoids of a semiring R. The additive submonoid `S * T` is the
smallest R-submodule of `R` containing the elements `s * t` for `s ∈ S` and `t ∈ T`. -/
protected def mul : Mul (AddSubmonoid R) :=
  ⟨fun M N => ⨆ s : M, N.map (AddMonoidHom.mul s.1)⟩

theorem mul_mem_mul {M N : AddSubmonoid R} {m n : R} (hm : m ∈ M) (hn : n ∈ N) : m * n ∈ M * N :=
  smul_mem_smul hm hn


theorem mul_le {M N P : AddSubmonoid R} : M * N ≤ P ↔ ∀ m ∈ M, ∀ n ∈ N, m * n ∈ P :=
  smul_le


@[elab_as_elim]
protected theorem mul_induction_on {M N : AddSubmonoid R} {C : R → Prop} {r : R} (hr : r ∈ M * N)
    (hm : ∀ m ∈ M, ∀ n ∈ N, C (m * n)) (ha : ∀ x y, C x → C y → C (x + y)) : C r :=
  AddSubmonoid.smul_induction_on hr hm ha

-- this proof is copied directly from `Submodule.span_mul_span`
-- Porting note: proof rewritten
-- need `add_smul` to generalize to `SMul`

theorem closure_mul_closure (S T : Set R) : closure S * closure T = closure (S * T) := by
  /-
    R : Type u_4
    inst✝ : NonUnitalNonAssocSemiring R
    S T : Set R
    ⊢ Eq (HMul.hMul (AddSubmonoid.closure S) (AddSubmonoid.closure T)) (AddSubmono …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      ⊢ LE.le (HMul.hMul (AddSubmonoid.closure S) (AddSubmonoid.closure T)) (AddSubm …
    -/
  · refine mul_le.2 fun a ha b hb => ?_
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem (AddSubmonoid.closure S) a
      b : R
      hb : Membership.mem (AddSubmonoid.closure T) b
      ⊢ Membership.mem (AddSubmonoid.closure (HMul.hMul S T)) (HMul.hMul a b)
    -/
    rw [← AddMonoidHom.mulRight_apply, ← AddSubmonoid.mem_comap]
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem (AddSubmonoid.closure S) a
      b : R
      hb : Membership.mem (AddSubmonoid.closure T) b
      ⊢ Membership.mem (AddSubmonoid.comap (AddMonoidHom.mulRight b) (AddSubmonoid.c …
    -/
    refine (closure_le.2 fun a' ha' => ?_) ha
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem (AddSubmonoid.closure S) a
      b : R
      hb : Membership.mem (AddSubmonoid.closure T) b
      a' : R
      ha' : Membership.mem S a'
      ⊢ Membership.mem (↑(AddSubmonoid.comap (AddMonoidHom.mulRight b) (AddSubmonoid …
    -/
    change b ∈ (closure (S * T)).comap (AddMonoidHom.mulLeft a')
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem (AddSubmonoid.closure S) a
      b : R
      hb : Membership.mem (AddSubmonoid.closure T) b
      a' : R
      ha' : Membership.mem S a'
      ⊢ Membership.mem (AddSubmonoid.comap (AddMonoidHom.mulLeft a') (AddSubmonoid.c …
    -/
    refine (closure_le.2 fun b' hb' => ?_) hb
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem (AddSubmonoid.closure S) a
      b : R
      hb : Membership.mem (AddSubmonoid.closure T) b
      a' : R
      ha' : Membership.mem S a'
      b' : R
      hb' : Membership.mem T b'
      ⊢ Membership.mem (↑(AddSubmonoid.comap (AddMonoidHom.mulLeft a') (AddSubmonoid …
    -/
    change a' * b' ∈ closure (S * T)
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem (AddSubmonoid.closure S) a
      b : R
      hb : Membership.mem (AddSubmonoid.closure T) b
      a' : R
      ha' : Membership.mem S a'
      b' : R
      hb' : Membership.mem T b'
      ⊢ Membership.mem (AddSubmonoid.closure (HMul.hMul S T)) (HMul.hMul a' b')
    -/
    exact subset_closure (Set.mul_mem_mul ha' hb')
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      ⊢ LE.le (AddSubmonoid.closure (HMul.hMul S T)) (HMul.hMul (AddSubmonoid.closur …
    -/
  · rw [closure_le]
    /-
      case a
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      ⊢ HasSubset.Subset (HMul.hMul S T) ↑(HMul.hMul (AddSubmonoid.closure S) (AddSu …
    -/
    rintro _ ⟨a, ha, b, hb, rfl⟩
    /-
      case a.intro.intro.intro.intro
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S T : Set R
      a : R
      ha : Membership.mem S a
      b : R
      hb : Membership.mem T b
      ⊢ Membership.mem (↑(HMul.hMul (AddSubmonoid.closure S) (AddSubmonoid.closure T …
    -/
    exact mul_mem_mul (subset_closure ha) (subset_closure hb)
    /-
      🎉 no goals
    -/


theorem mul_eq_closure_mul_set (M N : AddSubmonoid R) :
    M * N = closure ((M : Set R) * (N : Set R)) := by
  /-
    R : Type u_4
    inst✝ : NonUnitalNonAssocSemiring R
    M N : AddSubmonoid R
    ⊢ Eq (HMul.hMul M N) (AddSubmonoid.closure (HMul.hMul ↑M ↑N))
  -/
  rw [← closure_mul_closure, closure_eq, closure_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_bot (S : AddSubmonoid R) : S * ⊥ = ⊥ :=
  addSubmonoid_smul_bot S

-- need `zero_smul` to generalize to `SMul`

@[simp]
theorem bot_mul (S : AddSubmonoid R) : ⊥ * S = ⊥ :=
  eq_bot_iff.2 <| mul_le.2 fun m hm n _ => by
    /-
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      S : AddSubmonoid R
      m : R
      hm : Membership.mem Bot.bot m
      n : R
      x✝ : Membership.mem S n
      ⊢ Membership.mem Bot.bot (HMul.hMul m n)
    -/
    rw [AddSubmonoid.mem_bot] at hm ⊢; rw [hm, zero_mul]
                                       /-
                                         🎉 no goals
                                       -/


@[mono, gcongr] lemma mul_le_mul (hmp : M ≤ P) (hnq : N ≤ Q) : M * N ≤ P * Q := smul_le_smul hmp hnq


@[gcongr] lemma mul_le_mul_left (h : M ≤ N) : M * P ≤ N * P := smul_le_smul_left h

@[gcongr] lemma mul_le_mul_right (h : N ≤ P) : M * N ≤ M * P := smul_le_smul_right h


theorem mul_subset_mul : (↑M : Set R) * (↑N : Set R) ⊆ (↑(M * N) : Set R) :=
  smul_subset_smul


theorem mul_sup : M * (N ⊔ P) = M * N ⊔ M * P :=
  addSubmonoid_smul_sup

-- need `zero_smul` and `add_smul` to generalize to `SMul`

theorem sup_mul : (M ⊔ N) * P = M * P ⊔ N * P :=
  le_antisymm (mul_le.mpr fun mn hmn p hp ↦ by
    /-
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      M N P : AddSubmonoid R
      mn : R
      hmn : Membership.mem (Max.max M N) mn
      p : R
      hp : Membership.mem P p
      ⊢ Membership.mem (Max.max (HMul.hMul M P) (HMul.hMul N P)) (HMul.hMul mn p)
    -/
    obtain ⟨m, hm, n, hn, rfl⟩ := mem_sup.mp hmn
    /-
      case intro.intro.intro.intro
      R : Type u_4
      inst✝ : NonUnitalNonAssocSemiring R
      M N P : AddSubmonoid R
      p : R
      hp : Membership.mem P p
      m : R
      hm : Membership.mem M m
      n : R
      hn : Membership.mem N n
      hmn : Membership.mem (Max.max M N) (HAdd.hAdd m n)
      ⊢ Membership.mem (Max.max (HMul.hMul M P) (HMul.hMul N P)) (HMul.hMul (HAdd.hA …
    -/
    rw [right_distrib]; exact add_mem_sup (mul_mem_mul hm hp) <| mul_mem_mul hn hp)
                        /-
                          🎉 no goals
                        -/
    (sup_le (mul_le_mul_left le_sup_left) <| mul_le_mul_left le_sup_right)


theorem iSup_mul (S : ι → AddSubmonoid R) (T : AddSubmonoid R) : (⨆ i, S i) * T = ⨆ i, S i * T :=
  le_antisymm (mul_le.mpr fun s hs t ht ↦ iSup_induction _ (C := (· * t ∈ _)) hs
                                                                /-
                                                                  R : Type u_4
                                                                  inst✝ : NonUnitalNonAssocSemiring R
                                                                  ι : Sort u_7
                                                                  S : ι → AddSubmonoid R
                                                                  T : AddSubmonoid R
                                                                  s : R
                                                                  hs : Membership.mem (iSup fun i => S i) s
                                                                  t : R
                                                                  ht : Membership.mem T t
                                                                  ⊢ (fun x => Membership.mem (iSup fun i => HMul.hMul (S i) T) (HMul.hMul x t)) 0
                                                                -/
      (fun i s hs ↦ mem_iSup_of_mem i <| mul_mem_mul hs ht) (by simp_rw [zero_mul]; apply zero_mem)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                   /-
                     R : Type u_4
                     inst✝ : NonUnitalNonAssocSemiring R
                     ι : Sort u_7
                     S : ι → AddSubmonoid R
                     T : AddSubmonoid R
                     s : R
                     hs : Membership.mem (iSup fun i => S i) s
                     t : R
                     ht : Membership.mem T t
                     x✝¹ x✝ : R
                     ⊢ (fun x => Membership.mem (iSup fun i => HMul.hMul (S i) T) (HMul.hMul x t))  …
                   -/
      fun _ _ ↦ by simp_rw [right_distrib]; apply add_mem) <|
                                            /-
                                              🎉 no goals
                                            -/
    iSup_le fun i ↦ mul_le_mul_left (le_iSup _ i)


theorem mul_iSup (T : AddSubmonoid R) (S : ι → AddSubmonoid R) : (T * ⨆ i, S i) = ⨆ i, T * S i :=
  smul_iSup T S


theorem mul_comm_of_commute (h : ∀ m ∈ M, ∀ n ∈ N, Commute m n) : M * N = N * M :=
  le_antisymm (mul_le.mpr fun m hm n hn ↦ h m hm n hn ▸ mul_mem_mul hn hm)
    (mul_le.mpr fun n hn m hm ↦ h m hm n hn ▸ mul_mem_mul hm hn)


/-- `AddSubmonoid.neg` distributes over multiplication.

This is available as an instance in the `Pointwise` locale. -/
protected def hasDistribNeg : HasDistribNeg (AddSubmonoid R) :=
  { AddSubmonoid.involutiveNeg with
    neg_mul := fun x y => by
      refine
          le_antisymm (mul_le.2 fun m hm n hn => ?_)
            ((AddSubmonoid.neg_le _ _).2 <| mul_le.2 fun m hm n hn => ?_) <;>
        /-
          case refine_1
          α : Type u_1
          G : Type u_2
          M : Type u_3
          R : Type u_4
          A : Type u_5
          S : Type u_6
          inst✝² : Monoid M
          inst✝¹ : AddMonoid A
          inst✝ : NonUnitalNonAssocRing R
          x y : AddSubmonoid R
          m : R
          hm : Membership.mem (Neg.neg x) m
          n : R
          hn : Membership.mem y n
          ⊢ Membership.mem (Neg.neg (HMul.hMul x y)) (HMul.hMul m n)
        -/
        simp only [AddSubmonoid.mem_neg, ← neg_mul] at *
        /-
          case refine_1
          α : Type u_1
          G : Type u_2
          M : Type u_3
          R : Type u_4
          A : Type u_5
          S : Type u_6
          inst✝² : Monoid M
          inst✝¹ : AddMonoid A
          inst✝ : NonUnitalNonAssocRing R
          x y : AddSubmonoid R
          m n : R
          hn : Membership.mem y n
          hm : Membership.mem x (Neg.neg m)
          ⊢ Membership.mem (HMul.hMul x y) (HMul.hMul (Neg.neg m) n)
        -/
      · exact mul_mem_mul hm hn
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          G : Type u_2
          M : Type u_3
          R : Type u_4
          A : Type u_5
          S : Type u_6
          inst✝² : Monoid M
          inst✝¹ : AddMonoid A
          inst✝ : NonUnitalNonAssocRing R
          x y : AddSubmonoid R
          m : R
          hm : Membership.mem x m
          n : R
          hn : Membership.mem y n
          ⊢ Membership.mem (HMul.hMul (Neg.neg x) y) (HMul.hMul (Neg.neg m) n)
        -/
      · exact mul_mem_mul (neg_mem_neg.2 hm) hn
        /-
          🎉 no goals
        -/
    mul_neg := fun x y => by
      refine
          le_antisymm (mul_le.2 fun m hm n hn => ?_)
            ((AddSubmonoid.neg_le _ _).2 <| mul_le.2 fun m hm n hn => ?_) <;>
        /-
          case refine_1
          α : Type u_1
          G : Type u_2
          M : Type u_3
          R : Type u_4
          A : Type u_5
          S : Type u_6
          inst✝² : Monoid M
          inst✝¹ : AddMonoid A
          inst✝ : NonUnitalNonAssocRing R
          x y : AddSubmonoid R
          m : R
          hm : Membership.mem x m
          n : R
          hn : Membership.mem (Neg.neg y) n
          ⊢ Membership.mem (Neg.neg (HMul.hMul x y)) (HMul.hMul m n)
        -/
        simp only [AddSubmonoid.mem_neg, ← mul_neg] at *
        /-
          case refine_1
          α : Type u_1
          G : Type u_2
          M : Type u_3
          R : Type u_4
          A : Type u_5
          S : Type u_6
          inst✝² : Monoid M
          inst✝¹ : AddMonoid A
          inst✝ : NonUnitalNonAssocRing R
          x y : AddSubmonoid R
          m : R
          hm : Membership.mem x m
          n : R
          hn : Membership.mem y (Neg.neg n)
          ⊢ Membership.mem (HMul.hMul x y) (HMul.hMul m (Neg.neg n))
        -/
      · exact mul_mem_mul hm hn
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          G : Type u_2
          M : Type u_3
          R : Type u_4
          A : Type u_5
          S : Type u_6
          inst✝² : Monoid M
          inst✝¹ : AddMonoid A
          inst✝ : NonUnitalNonAssocRing R
          x y : AddSubmonoid R
          m : R
          hm : Membership.mem x m
          n : R
          hn : Membership.mem y n
          ⊢ Membership.mem (HMul.hMul x (Neg.neg y)) (HMul.hMul m (Neg.neg n))
        -/
      · exact mul_mem_mul hm (neg_mem_neg.2 hn) }
        /-
          🎉 no goals
        -/


/-- A `MulOneClass` structure on additive submonoids of a (possibly, non-associative) semiring. -/
protected def mulOneClass : MulOneClass (AddSubmonoid R) where
  one := 1
  mul := (· * ·)
                  /-
                    α : Type u_1
                    G : Type u_2
                    M✝ : Type u_3
                    R : Type u_4
                    A : Type u_5
                    S : Type u_6
                    inst✝² : Monoid M✝
                    inst✝¹ : AddMonoid A
                    inst✝ : NonAssocSemiring R
                    M : AddSubmonoid R
                    ⊢ Eq (HMul.hMul 1 M) M
                  -/
  one_mul M := by rw [one_eq_closure_one_set, ← closure_eq M, closure_mul_closure, one_mul]
                  /-
                    🎉 no goals
                  -/
                  /-
                    α : Type u_1
                    G : Type u_2
                    M✝ : Type u_3
                    R : Type u_4
                    A : Type u_5
                    S : Type u_6
                    inst✝² : Monoid M✝
                    inst✝¹ : AddMonoid A
                    inst✝ : NonAssocSemiring R
                    M : AddSubmonoid R
                    ⊢ Eq (HMul.hMul M 1) M
                  -/
  mul_one M := by rw [one_eq_closure_one_set, ← closure_eq M, closure_mul_closure, mul_one]
                  /-
                    🎉 no goals
                  -/

/-- Semigroup structure on additive submonoids of a (possibly, non-unital) semiring. -/
protected def semigroup : Semigroup (AddSubmonoid R) where
  mul := (· * ·)
  mul_assoc _M _N _P :=
    le_antisymm
      (mul_le.2 fun _mn hmn p hp => AddSubmonoid.mul_induction_on hmn
        (fun m hm n hn ↦ mul_assoc m n p ▸ mul_mem_mul hm <| mul_mem_mul hn hp)
        fun x y ↦ (add_mul x y p).symm ▸ add_mem)
      (mul_le.2 fun m hm _np hnp => AddSubmonoid.mul_induction_on hnp
        (fun n hn p hp ↦ mul_assoc m n p ▸ mul_mem_mul (mul_mem_mul hm hn) hp)
        fun x y ↦ (mul_add m x y) ▸ add_mem)

/-- Monoid structure on additive submonoids of a semiring. -/
protected def monoid : Monoid (AddSubmonoid R) :=
  { AddSubmonoid.semigroup, AddSubmonoid.mulOneClass with }

theorem closure_pow (s : Set R) : ∀ n : ℕ, closure s ^ n = closure (s ^ n)
            /-
              R : Type u_4
              inst✝ : Semiring R
              s : Set R
              ⊢ Eq (HPow.hPow (AddSubmonoid.closure s) 0) (AddSubmonoid.closure (HPow.hPow s …
            -/
  | 0 => by rw [pow_zero, pow_zero, one_eq_closure_one_set]
            /-
              🎉 no goals
            -/
                /-
                  R : Type u_4
                  inst✝ : Semiring R
                  s : Set R
                  n : Nat
                  ⊢ Eq (HPow.hPow (AddSubmonoid.closure s) (HAdd.hAdd n 1)) (AddSubmonoid.closur …
                -/
  | n + 1 => by rw [pow_succ, pow_succ, closure_pow s n, closure_mul_closure]
                /-
                  🎉 no goals
                -/


theorem pow_eq_closure_pow_set (s : AddSubmonoid R) (n : ℕ) :
    s ^ n = closure ((s : Set R) ^ n) := by
  /-
    R : Type u_4
    inst✝ : Semiring R
    s : AddSubmonoid R
    n : Nat
    ⊢ Eq (HPow.hPow s n) (AddSubmonoid.closure (HPow.hPow (↑s) n))
  -/
  rw [← closure_pow, closure_eq]
  /-
    🎉 no goals
  -/


theorem pow_subset_pow {s : AddSubmonoid R} {n : ℕ} : (↑s : Set R) ^ n ⊆ ↑(s ^ n) :=
  (pow_eq_closure_pow_set s n).symm ▸ subset_closure


@[to_additive]
theorem submonoid_closure (hpos : ∀ x : α, x ∈ s → 1 ≤ x) (h : s.IsPWO) :
    IsPWO (Submonoid.closure s : Set α) := by
  /-
    α : Type u_1
    inst✝ : OrderedCancelCommMonoid α
    s : Set α
    hpos : ∀ (x : α), Membership.mem s x → LE.le 1 x
    h : s.IsPWO
    ⊢ (↑(Submonoid.closure s)).IsPWO
  -/
  rw [Submonoid.closure_eq_image_prod]
  /-
    α : Type u_1
    inst✝ : OrderedCancelCommMonoid α
    s : Set α
    hpos : ∀ (x : α), Membership.mem s x → LE.le 1 x
    h : s.IsPWO
    ⊢ (Set.image List.prod (setOf fun l => ∀ (x : α), Membership.mem l x → Members …
  -/
  refine (h.partiallyWellOrderedOn_sublistForall₂ (· ≤ ·)).image_of_monotone_on ?_
  /-
    α : Type u_1
    inst✝ : OrderedCancelCommMonoid α
    s : Set α
    hpos : ∀ (x : α), Membership.mem s x → LE.le 1 x
    h : s.IsPWO
    ⊢ ∀ (a₁ : List α), Membership.mem (setOf fun l => ∀ (x : α), Membership.mem l  …
  -/
  exact fun l1 _ l2 hl2 h12 => h12.prod_le_prod' fun x hx => hpos x <| hl2 x hx
  /-
    🎉 no goals
  -/


