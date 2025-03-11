/-- A typeclass for when a `MulAction β α` descends to the quotient `α ⧸ H`. -/
class QuotientAction : Prop where
  /-- The action fulfils a normality condition on products that lie in `H`.
    This ensures that the action descends to an action on the quotient `α ⧸ H`. -/
  inv_mul_mem : ∀ (b : β) {a a' : α}, a⁻¹ * a' ∈ H → (b • a)⁻¹ * b • a' ∈ H


/-- A typeclass for when an `AddAction β α` descends to the quotient `α ⧸ H`. -/
class _root_.AddAction.QuotientAction {α : Type u} (β : Type v) [AddGroup α] [AddMonoid β]
  [AddAction β α] (H : AddSubgroup α) : Prop where
  /-- The action fulfils a normality condition on summands that lie in `H`.
    This ensures that the action descends to an action on the quotient `α ⧸ H`. -/
  inv_mul_mem : ∀ (b : β) {a a' : α}, -a + a' ∈ H → -(b +ᵥ a) + (b +ᵥ a') ∈ H


@[to_additive]
instance left_quotientAction : QuotientAction α H :=
                     /-
                       α : Type u
                       β : Type v
                       γ : Type w
                       inst✝² : Group α
                       inst✝¹ : Monoid β
                       inst✝ : MulAction β α
                       H : Subgroup α
                       x✝³ x✝² x✝¹ : α
                       x✝ : Membership.mem H (HMul.hMul (Inv.inv x✝²) x✝¹)
                       ⊢ Membership.mem H (HMul.hMul (Inv.inv (HSMul.hSMul x✝³ x✝²)) (HSMul.hSMul x✝³ …
                     -/
  ⟨fun _ _ _ _ => by rwa [smul_eq_mul, smul_eq_mul, mul_inv_rev, mul_assoc, inv_mul_cancel_left]⟩
                     /-
                       🎉 no goals
                     -/


@[to_additive]
instance right_quotientAction : QuotientAction (normalizer H).op H :=
  ⟨fun b c _ _ => by
    rwa [smul_def, smul_def, smul_eq_mul_unop, smul_eq_mul_unop, mul_inv_rev, ← mul_assoc,
      mem_normalizer_iff'.mp b.prop, mul_assoc, mul_inv_cancel_left]⟩


@[to_additive]
instance right_quotientAction' [hH : H.Normal] : QuotientAction αᵐᵒᵖ H :=
  ⟨fun _ _ _ _ => by
    rwa [smul_eq_mul_unop, smul_eq_mul_unop, mul_inv_rev, mul_assoc, hH.mem_comm_iff, mul_assoc,
      mul_inv_cancel_right]⟩


@[to_additive]
instance quotient [QuotientAction β H] : MulAction β (α ⧸ H) where
  smul b :=
    Quotient.map' (b • ·) fun _ _ h =>
      leftRel_apply.mpr <| QuotientAction.inv_mul_mem b <| leftRel_apply.mp h
  one_smul q := Quotient.inductionOn' q fun a => congr_arg Quotient.mk'' (one_smul β a)
  mul_smul b b' q := Quotient.inductionOn' q fun a => congr_arg Quotient.mk'' (mul_smul b b' a)


@[to_additive (attr := simp)]
theorem Quotient.smul_mk [QuotientAction β H] (b : β) (a : α) :
    (b • QuotientGroup.mk a : α ⧸ H) = QuotientGroup.mk (b • a) :=
  rfl


@[to_additive (attr := simp)]
theorem Quotient.smul_coe [QuotientAction β H] (b : β) (a : α) :
    b • (a : α ⧸ H) = (↑(b • a) : α ⧸ H) :=
  rfl


@[to_additive (attr := simp)]
theorem Quotient.mk_smul_out [QuotientAction β H] (b : β) (q : α ⧸ H) :
                                               /-
                                                 α : Type u
                                                 β : Type v
                                                 inst✝³ : Group α
                                                 inst✝² : Monoid β
                                                 inst✝¹ : MulAction β α
                                                 H : Subgroup α
                                                 inst✝ : MulAction.QuotientAction β H
                                                 b : β
                                                 q : HasQuotient.Quotient α H
                                                 ⊢ Eq (↑(HSMul.hSMul b (Quotient.out q))) (HSMul.hSMul b q)
                                               -/
    QuotientGroup.mk (b • q.out) = b • q := by rw [← Quotient.smul_mk, QuotientGroup.out_eq']
                                               /-
                                                 🎉 no goals
                                               -/

-- Porting note: removed simp attribute, simp can prove this

@[to_additive]
theorem Quotient.coe_smul_out [QuotientAction β H] (b : β) (q : α ⧸ H) : ↑(b • q.out) = b • q :=
  Quotient.mk_smul_out H b q


theorem _root_.QuotientGroup.out_conj_pow_minimalPeriod_mem (a : α) (q : α ⧸ H) :
    q.out⁻¹ * a ^ Function.minimalPeriod (a • ·) q * q.out ∈ H := by
  rw [mul_assoc, ← QuotientGroup.eq, QuotientGroup.out_eq', ← smul_eq_mul, Quotient.mk_smul_out,
    eq_comm, pow_smul_eq_iff_minimalPeriod_dvd]


@[to_additive]
alias Quotient.mk_smul_out' := Quotient.mk_smul_out

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive]
alias Quotient.coe_smul_out' := Quotient.coe_smul_out

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[deprecated (since := "2024-10-19")]
alias _root_.QuotientGroup.out'_conj_pow_minimalPeriod_mem :=
  QuotientGroup.out_conj_pow_minimalPeriod_mem


/-- The canonical map to the left cosets. -/
def _root_.MulActionHom.toQuotient (H : Subgroup α) : α →[α] α ⧸ H where
  toFun := (↑); map_smul' := Quotient.smul_coe H


@[simp]
theorem _root_.MulActionHom.toQuotient_apply (H : Subgroup α) (g : α) :
    MulActionHom.toQuotient H g = g :=
  rfl


@[to_additive]
instance mulLeftCosetsCompSubtypeVal (H I : Subgroup α) : MulAction I (α ⧸ H) :=
  MulAction.compHom (α ⧸ H) (Subgroup.subtype I)


/-- The canonical map from the quotient of the stabilizer to the set. -/
@[to_additive "The canonical map from the quotient of the stabilizer to the set. "]
def ofQuotientStabilizer (g : α ⧸ MulAction.stabilizer α x) : β :=
  Quotient.liftOn' g (· • x) fun g1 g2 H =>
    calc
      g1 • x = g1 • (g1⁻¹ * g2) • x := congr_arg _ (leftRel_apply.mp H).symm
                       /-
                         α : Type u
                         β : Type v
                         γ : Type w
                         inst✝¹ : Group α
                         inst✝ : MulAction α β
                         x : β
                         g : HasQuotient.Quotient α (MulAction.stabilizer α x)
                         g1 g2 : α
                         H : (QuotientGroup.leftRel (MulAction.stabilizer α x)) g1 g2
                         ⊢ Eq (HSMul.hSMul g1 (HSMul.hSMul (HMul.hMul (Inv.inv g1) g2) x)) (HSMul.hSMul …
                       -/
      _ = g2 • x := by rw [smul_smul, mul_inv_cancel_left]
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp)]
theorem ofQuotientStabilizer_mk (g : α) : ofQuotientStabilizer α x (QuotientGroup.mk g) = g • x :=
  rfl


@[to_additive]
theorem ofQuotientStabilizer_mem_orbit (g) : ofQuotientStabilizer α x g ∈ orbit α x :=
  Quotient.inductionOn' g fun g => ⟨g, rfl⟩


@[to_additive]
theorem ofQuotientStabilizer_smul (g : α) (g' : α ⧸ MulAction.stabilizer α x) :
    ofQuotientStabilizer α x (g • g') = g • ofQuotientStabilizer α x g' :=
  Quotient.inductionOn' g' fun _ => mul_smul _ _ _


@[to_additive]
theorem injective_ofQuotientStabilizer : Function.Injective (ofQuotientStabilizer α x) :=
  fun y₁ y₂ =>
  Quotient.inductionOn₂' y₁ y₂ fun g₁ g₂ (H : g₁ • x = g₂ • x) =>
    Quotient.sound' <| by
      /-
        α : Type u
        β : Type v
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        y₁ y₂ : HasQuotient.Quotient α (MulAction.stabilizer α x)
        g₁ g₂ : α
        H : Eq (HSMul.hSMul g₁ x) (HSMul.hSMul g₂ x)
        ⊢ (QuotientGroup.leftRel (MulAction.stabilizer α x)) g₁ g₂
      -/
      rw [leftRel_apply]
      /-
        α : Type u
        β : Type v
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        y₁ y₂ : HasQuotient.Quotient α (MulAction.stabilizer α x)
        g₁ g₂ : α
        H : Eq (HSMul.hSMul g₁ x) (HSMul.hSMul g₂ x)
        ⊢ Membership.mem (MulAction.stabilizer α x) (HMul.hMul (Inv.inv g₁) g₂)
      -/
      show (g₁⁻¹ * g₂) • x = x
      /-
        α : Type u
        β : Type v
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        y₁ y₂ : HasQuotient.Quotient α (MulAction.stabilizer α x)
        g₁ g₂ : α
        H : Eq (HSMul.hSMul g₁ x) (HSMul.hSMul g₂ x)
        ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv g₁) g₂) x) x
      -/
      rw [mul_smul, ← H, inv_smul_smul]
      /-
        🎉 no goals
      -/


/-- **Orbit-stabilizer theorem**. -/
@[to_additive "Orbit-stabilizer theorem."]
noncomputable def orbitEquivQuotientStabilizer (b : β) : orbit α b ≃ α ⧸ stabilizer α b :=
  Equiv.symm <|
    Equiv.ofBijective (fun g => ⟨ofQuotientStabilizer α b g, ofQuotientStabilizer_mem_orbit α b g⟩)
                                                             /-
                                                               α : Type u
                                                               β : Type v
                                                               γ : Type w
                                                               inst✝¹ : Group α
                                                               inst✝ : MulAction α β
                                                               x✝ b : β
                                                               x y : HasQuotient.Quotient α (MulAction.stabilizer α b)
                                                               hxy : Eq ((fun g => ⟨MulAction.ofQuotientStabilizer α b g, ⋯⟩) x) ((fun g => ⟨ …
                                                               ⊢ Eq (MulAction.ofQuotientStabilizer α b x) (MulAction.ofQuotientStabilizer α  …
                                                             -/
      ⟨fun x y hxy => injective_ofQuotientStabilizer α b (by convert congr_arg Subtype.val hxy),
                                                             /-
                                                               🎉 no goals
                                                             -/
        fun ⟨_, ⟨g, hgb⟩⟩ => ⟨g, Subtype.eq hgb⟩⟩


/-- Orbit-stabilizer theorem. -/
@[to_additive AddAction.orbitProdStabilizerEquivAddGroup "Orbit-stabilizer theorem."]
noncomputable def orbitProdStabilizerEquivGroup (b : β) : orbit α b × stabilizer α b ≃ α :=
  (Equiv.prodCongr (orbitEquivQuotientStabilizer α _) (Equiv.refl _)).trans
    Subgroup.groupEquivQuotientProdSubgroup.symm


/-- Orbit-stabilizer theorem. -/
@[to_additive AddAction.card_orbit_mul_card_stabilizer_eq_card_addGroup "Orbit-stabilizer theorem."]
theorem card_orbit_mul_card_stabilizer_eq_card_group (b : β) [Fintype α] [Fintype <| orbit α b]
    [Fintype <| stabilizer α b] :
    Fintype.card (orbit α b) * Fintype.card (stabilizer α b) = Fintype.card α := by
  /-
    α : Type u
    β : Type v
    inst✝⁴ : Group α
    inst✝³ : MulAction α β
    b : β
    inst✝² : Fintype α
    inst✝¹ : Fintype ↑(MulAction.orbit α b)
    inst✝ : Fintype (Subtype fun x => Membership.mem (MulAction.stabilizer α b) x)
    ⊢ Eq (HMul.hMul (Fintype.card ↑(MulAction.orbit α b)) (Fintype.card (Subtype f …
  -/
  rw [← Fintype.card_prod, Fintype.card_congr (orbitProdStabilizerEquivGroup α b)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem orbitEquivQuotientStabilizer_symm_apply (b : β) (a : α) :
    ((orbitEquivQuotientStabilizer α b).symm a : β) = a • b :=
  rfl


@[to_additive (attr := simp)]
theorem stabilizer_quotient {G} [Group G] (H : Subgroup G) :
    MulAction.stabilizer G ((1 : G) : G ⧸ H) = H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq (MulAction.stabilizer G ↑1) H
  -/
  ext
  /-
    case h
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    x✝ : G
    ⊢ Iff (Membership.mem (MulAction.stabilizer G ↑1) x✝) (Membership.mem H x✝)
  -/
  simp [QuotientGroup.eq]
  /-
    🎉 no goals
  -/


local notation "Ω" => Quotient <| orbitRel α β


/-- **Class formula** : given `G` a group acting on `X` and `φ` a function mapping each orbit of `X`
under this action (that is, each element of the quotient of `X` by the relation `orbitRel G X`) to
an element in this orbit, this gives a (noncomputable) bijection between `X` and the disjoint union
of `G/Stab(φ(ω))` over all orbits `ω`. In most cases you'll want `φ` to be `Quotient.out`, so we
provide `MulAction.selfEquivSigmaOrbitsQuotientStabilizer'` as a special case. -/
@[to_additive
      "**Class formula** : given `G` an additive group acting on `X` and `φ` a function
      mapping each orbit of `X` under this action (that is, each element of the quotient of `X` by
      the relation `orbit_rel G X`) to an element in this orbit, this gives a (noncomputable)
      bijection between `X` and the disjoint union of `G/Stab(φ(ω))` over all orbits `ω`. In most
      cases you'll want `φ` to be `Quotient.out`, so we provide
      `AddAction.selfEquivSigmaOrbitsQuotientStabilizer'` as a special case. "]
noncomputable def selfEquivSigmaOrbitsQuotientStabilizer' {φ : Ω → β}
    (hφ : LeftInverse Quotient.mk'' φ) : β ≃ Σω : Ω, α ⧸ stabilizer α (φ ω) :=
  calc
    β ≃ Σω : Ω, orbitRel.Quotient.orbit ω := selfEquivSigmaOrbits' α β
    _ ≃ Σω : Ω, α ⧸ stabilizer α (φ ω) :=
      Equiv.sigmaCongrRight fun ω =>
        (Equiv.Set.ofEq <| orbitRel.Quotient.orbit_eq_orbit_out _ hφ).trans <|
          orbitEquivQuotientStabilizer α (φ ω)


/-- **Class formula** for a finite group acting on a finite type. See
`MulAction.card_eq_sum_card_group_div_card_stabilizer` for a specialized version using
`Quotient.out`. -/
@[to_additive
      "**Class formula** for a finite group acting on a finite type. See
      `AddAction.card_eq_sum_card_addGroup_div_card_stabilizer` for a specialized version using
      `Quotient.out`."]
theorem card_eq_sum_card_group_div_card_stabilizer' [Fintype α] [Fintype β] [Fintype Ω]
    [∀ b : β, Fintype <| stabilizer α b] {φ : Ω → β} (hφ : LeftInverse Quotient.mk'' φ) :
    Fintype.card β = ∑ ω : Ω, Fintype.card α / Fintype.card (stabilizer α (φ ω)) := by
  classical
    have : ∀ ω : Ω, Fintype.card α / Fintype.card (stabilizer α (φ ω)) =
        Fintype.card (α ⧸ stabilizer α (φ ω)) := by
      intro ω
      rw [Fintype.card_congr (@Subgroup.groupEquivQuotientProdSubgroup α _ (stabilizer α <| φ ω)),
        Fintype.card_prod, Nat.mul_div_cancel]
      exact Fintype.card_pos_iff.mpr (by infer_instance)
    simp_rw [this, ← Fintype.card_sigma,
      Fintype.card_congr (selfEquivSigmaOrbitsQuotientStabilizer' α β hφ)]


/-- **Class formula**. This is a special case of
`MulAction.self_equiv_sigma_orbits_quotient_stabilizer'` with `φ = Quotient.out`. -/
@[to_additive
      "**Class formula**. This is a special case of
      `AddAction.self_equiv_sigma_orbits_quotient_stabilizer'` with `φ = Quotient.out`. "]
noncomputable def selfEquivSigmaOrbitsQuotientStabilizer : β ≃ Σω : Ω, α ⧸ stabilizer α ω.out :=
  selfEquivSigmaOrbitsQuotientStabilizer' α β Quotient.out_eq'


/-- **Class formula** for a finite group acting on a finite type. -/
@[to_additive "**Class formula** for a finite group acting on a finite type."]
theorem card_eq_sum_card_group_div_card_stabilizer [Fintype α] [Fintype β] [Fintype Ω]
    [∀ b : β, Fintype <| stabilizer α b] :
    Fintype.card β = ∑ ω : Ω, Fintype.card α / Fintype.card (stabilizer α ω.out) :=
  card_eq_sum_card_group_div_card_stabilizer' α β Quotient.out_eq'


/-- **Burnside's lemma** : a (noncomputable) bijection between the disjoint union of all
`{x ∈ X | g • x = x}` for `g ∈ G` and the product `G × X/G`, where `G` is a group acting on `X` and
`X/G` denotes the quotient of `X` by the relation `orbitRel G X`. -/
@[to_additive AddAction.sigmaFixedByEquivOrbitsProdAddGroup
      "**Burnside's lemma** : a (noncomputable) bijection between the disjoint union of all
      `{x ∈ X | g • x = x}` for `g ∈ G` and the product `G × X/G`, where `G` is an additive group
      acting on `X` and `X/G`denotes the quotient of `X` by the relation `orbitRel G X`. "]
noncomputable def sigmaFixedByEquivOrbitsProdGroup : (Σa : α, fixedBy β a) ≃ Ω × α :=
  calc
    (Σa : α, fixedBy β a) ≃ { ab : α × β // ab.1 • ab.2 = ab.2 } :=
      (Equiv.subtypeProdEquivSigmaSubtype _).symm
    _ ≃ { ba : β × α // ba.2 • ba.1 = ba.1 } := (Equiv.prodComm α β).subtypeEquiv fun _ => Iff.rfl
    _ ≃ Σb : β, stabilizer α b :=
      Equiv.subtypeProdEquivSigmaSubtype fun (b : β) a => a ∈ stabilizer α b
    _ ≃ Σωb : Σω : Ω, orbit α ω.out, stabilizer α (ωb.2 : β) :=
      (selfEquivSigmaOrbits α β).sigmaCongrLeft'
    _ ≃ Σω : Ω, Σb : orbit α ω.out, stabilizer α (b : β) :=
      Equiv.sigmaAssoc fun (ω : Ω) (b : orbit α ω.out) => stabilizer α (b : β)
    _ ≃ Σω : Ω, Σ _ : orbit α ω.out, stabilizer α ω.out :=
      Equiv.sigmaCongrRight fun _ =>
        Equiv.sigmaCongrRight fun ⟨_, hb⟩ => (stabilizerEquivStabilizerOfOrbitRel hb).toEquiv
    _ ≃ Σω : Ω, orbit α ω.out × stabilizer α ω.out :=
      Equiv.sigmaCongrRight fun _ => Equiv.sigmaEquivProd _ _
    _ ≃ Σ _ : Ω, α := Equiv.sigmaCongrRight fun ω => orbitProdStabilizerEquivGroup α ω.out
    _ ≃ Ω × α := Equiv.sigmaEquivProd Ω α


/-- **Burnside's lemma** : given a finite group `G` acting on a set `X`, the average number of
elements fixed by each `g ∈ G` is the number of orbits. -/
@[to_additive AddAction.sum_card_fixedBy_eq_card_orbits_mul_card_addGroup
      "**Burnside's lemma** : given a finite additive group `G` acting on a set `X`,
      the average number of elements fixed by each `g ∈ G` is the number of orbits. "]
theorem sum_card_fixedBy_eq_card_orbits_mul_card_group [Fintype α] [∀ a : α, Fintype <| fixedBy β a]
    [Fintype Ω] : (∑ a : α, Fintype.card (fixedBy β a)) = Fintype.card Ω * Fintype.card α := by
  rw [← Fintype.card_prod, ← Fintype.card_sigma,
    Fintype.card_congr (sigmaFixedByEquivOrbitsProdGroup α β)]


@[to_additive]
instance isPretransitive_quotient (G) [Group G] (H : Subgroup G) : IsPretransitive G (G ⧸ H) where
  exists_smul_eq := by
    { rintro ⟨x⟩ ⟨y⟩
      refine ⟨y * x⁻¹, QuotientGroup.eq.mpr ?_⟩
      simp only [smul_eq_mul, H.one_mem, inv_mul_cancel, inv_mul_cancel_right]}


@[to_additive]
instance finite_quotient_of_pretransitive_of_finite_quotient [IsPretransitive α β] {H : Subgroup α}
    [Finite (α ⧸ H)] : Finite <| orbitRel.Quotient H β := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝³ : Group α
    inst✝² : MulAction α β
    x : β
    inst✝¹ : MulAction.IsPretransitive α β
    H : Subgroup α
    inst✝ : Finite (HasQuotient.Quotient α H)
    ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β)
  -/
  rcases isEmpty_or_nonempty β with he | ⟨⟨b⟩⟩
    /-
      case inl
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : Group α
      inst✝² : MulAction α β
      x : β
      inst✝¹ : MulAction.IsPretransitive α β
      H : Subgroup α
      inst✝ : Finite (HasQuotient.Quotient α H)
      he : IsEmpty β
      ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β)
    -/
  · exact Quotient.finite _
    /-
      🎉 no goals
    -/
  · have h' : Finite (Quotient (rightRel H)) :=
      Finite.of_equiv _ (quotientRightRelEquivQuotientLeftRel _).symm
    let f : Quotient (rightRel H) → orbitRel.Quotient H β :=
      fun a ↦ Quotient.liftOn' a (fun g ↦ ⟦g • b⟧) fun g₁ g₂ r ↦ by
        replace r := Setoid.symm' _ r
        change (rightRel H).r _ _ at r
        rw [rightRel_eq] at r
        simp only [Quotient.eq]
        change g₁ • b ∈ orbit H (g₂ • b)
        rw [mem_orbit_iff]
        exact ⟨⟨g₁ * g₂⁻¹, r⟩, by simp [mul_smul]⟩
    exact Finite.of_surjective f ((Quotient.surjective_liftOn' _).2
      (Quotient.mk''_surjective.comp (MulAction.surjective_smul _ _)))


/-- A bijection between the quotient of the action of a subgroup `H` on an orbit, and a
corresponding quotient expressed in terms of `Setoid.comap Subtype.val`. -/
@[to_additive "A bijection between the quotient of the action of an additive subgroup `H` on an
orbit, and a corresponding quotient expressed in terms of `Setoid.comap Subtype.val`."]
noncomputable def equivSubgroupOrbitsSetoidComap (H : Subgroup α) (ω : Ω) :
    orbitRel.Quotient H (orbitRel.Quotient.orbit ω) ≃
      Quotient ((orbitRel H β).comap (Subtype.val : Quotient.mk (orbitRel α β) ⁻¹' {ω} → β)) where
  toFun := fun q ↦ q.liftOn' (fun x ↦ ⟦⟨↑x, by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x✝ : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(MulAct …
      x : ↑(MulAction.orbitRel.Quotient.orbit ω)
      ⊢ Membership.mem (Set.preimage (Quotient.mk (MulAction.orbitRel α β)) (Singlet …
    -/
    simp only [Set.mem_preimage, Set.mem_singleton_iff]
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x✝ : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(MulAct …
      x : ↑(MulAction.orbitRel.Quotient.orbit ω)
      ⊢ Eq (Quotient.mk (MulAction.orbitRel α β) ↑x) ω
    -/
    have hx := x.property
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x✝ : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(MulAct …
      x : ↑(MulAction.orbitRel.Quotient.orbit ω)
      hx : Membership.mem (MulAction.orbitRel.Quotient.orbit ω) ↑x
      ⊢ Eq (Quotient.mk (MulAction.orbitRel α β) ↑x) ω
    -/
    rwa [orbitRel.Quotient.mem_orbit] at hx⟩⟧) fun a b h ↦ by
    /-
      🎉 no goals
    -/
      simp only [← Quotient.eq,
                 orbitRel.Quotient.subgroup_quotient_eq_iff] at h
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        H : Subgroup α
        ω : Quotient (MulAction.orbitRel α β)
        q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(MulAct …
        a b : ↑(MulAction.orbitRel.Quotient.orbit ω)
        h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
        ⊢ Eq ((fun x => Quotient.mk (Setoid.comap Subtype.val (MulAction.orbitRel (Sub …
      -/
      simp only [Quotient.eq] at h ⊢
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        H : Subgroup α
        ω : Quotient (MulAction.orbitRel α β)
        q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(MulAct …
        a b : ↑(MulAction.orbitRel.Quotient.orbit ω)
        h : (MulAction.orbitRel (Subtype fun x => Membership.mem H x) β) ↑a ↑b
        ⊢ (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => Membership.m …
      -/
      exact h
      /-
        🎉 no goals
      -/
  invFun := fun q ↦ q.liftOn' (fun x ↦ ⟦⟨↑x, by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x✝ : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
      x : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.orbi …
      ⊢ Membership.mem (MulAction.orbitRel.Quotient.orbit ω) ↑x
    -/
    have hx := x.property
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x✝ : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
      x : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.orbi …
      hx : Membership.mem (Set.preimage (Quotient.mk (MulAction.orbitRel α β)) (Sing …
      ⊢ Membership.mem (MulAction.orbitRel.Quotient.orbit ω) ↑x
    -/
    simp only [Set.mem_preimage, Set.mem_singleton_iff] at hx
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x✝ : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
      x : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.orbi …
      hx : Eq (Quotient.mk (MulAction.orbitRel α β) ↑x) ω
      ⊢ Membership.mem (MulAction.orbitRel.Quotient.orbit ω) ↑x
    -/
    rwa [orbitRel.Quotient.mem_orbit, @Quotient.mk''_eq_mk]⟩⟧) fun a b h ↦ by
    /-
      🎉 no goals
    -/
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        H : Subgroup α
        ω : Quotient (MulAction.orbitRel α β)
        q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
        a b : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.or …
        h : (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => Membership …
        ⊢ Eq ((fun x => Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.m …
      -/
      rw [Setoid.comap_rel, ← Quotient.eq'', @Quotient.mk''_eq_mk] at h
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        H : Subgroup α
        ω : Quotient (MulAction.orbitRel α β)
        q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
        a b : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.or …
        h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
        ⊢ Eq ((fun x => Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.m …
      -/
      simp only [orbitRel.Quotient.subgroup_quotient_eq_iff]
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : Group α
        inst✝ : MulAction α β
        x : β
        H : Subgroup α
        ω : Quotient (MulAction.orbitRel α β)
        q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
        a b : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.or …
        h : Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x)  …
        ⊢ Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x) β) …
      -/
      exact h
      /-
        🎉 no goals
      -/
  left_inv := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      ⊢ Function.LeftInverse (fun q => q.liftOn' (fun x => Quotient.mk (MulAction.or …
    -/
    simp only [LeftInverse]
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      ⊢ ∀ (x : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(M …
    -/
    intro q
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) ↑(MulAct …
      ⊢ Eq ((Quotient.liftOn' q (fun x => Quotient.mk (Setoid.comap Subtype.val (Mul …
    -/
    induction q using Quotient.inductionOn'
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      a✝ : ↑(MulAction.orbitRel.Quotient.orbit ω)
      ⊢ Eq (((Quotient.mk'' a✝).liftOn' (fun x => Quotient.mk (Setoid.comap Subtype. …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      ⊢ Function.RightInverse (fun q => q.liftOn' (fun x => Quotient.mk (MulAction.o …
    -/
    simp only [Function.RightInverse, LeftInverse]
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      ⊢ ∀ (x : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x …
    -/
    intro q
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      q : Quotient (Setoid.comap Subtype.val (MulAction.orbitRel (Subtype fun x => M …
      ⊢ Eq (Quotient.liftOn' (q.liftOn' (fun x => Quotient.mk (MulAction.orbitRel (S …
    -/
    induction q using Quotient.inductionOn'
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝¹ : Group α
      inst✝ : MulAction α β
      x : β
      H : Subgroup α
      ω : Quotient (MulAction.orbitRel α β)
      a✝ : Subtype fun x => Membership.mem (Set.preimage (Quotient.mk (MulAction.orb …
      ⊢ Eq (Quotient.liftOn' ((Quotient.mk'' a✝).liftOn' (fun x => Quotient.mk (MulA …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- A bijection between the orbits under the action of a subgroup `H` on `β`, and the orbits
under the action of `H` on each orbit under the action of `G`. -/
@[to_additive "A bijection between the orbits under the action of an additive subgroup `H` on `β`,
and the orbits under the action of `H` on each orbit under the action of `G`."]
noncomputable def equivSubgroupOrbits (H : Subgroup α) :
    orbitRel.Quotient H β ≃ Σω : Ω, orbitRel.Quotient H (orbitRel.Quotient.orbit ω) :=
  (Setoid.sigmaQuotientEquivOfLe (orbitRel_subgroup_le H)).symm.trans
    (Equiv.sigmaCongrRight fun ω ↦ (equivSubgroupOrbitsSetoidComap H ω).symm)


@[to_additive]
instance finite_quotient_of_finite_quotient_of_finite_quotient {H : Subgroup α}
    [Finite (orbitRel.Quotient α β)] [Finite (α ⧸ H)] :
    Finite <| orbitRel.Quotient H β := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝³ : Group α
    inst✝² : MulAction α β
    x : β
    H : Subgroup α
    inst✝¹ : Finite (MulAction.orbitRel.Quotient α β)
    inst✝ : Finite (HasQuotient.Quotient α H)
    ⊢ Finite (MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β)
  -/
  rw [(equivSubgroupOrbits β H).finite_iff]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝³ : Group α
    inst✝² : MulAction α β
    x : β
    H : Subgroup α
    inst✝¹ : Finite (MulAction.orbitRel.Quotient α β)
    inst✝ : Finite (HasQuotient.Quotient α H)
    ⊢ Finite (Sigma fun ω => MulAction.orbitRel.Quotient (Subtype fun x => Members …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Given a group acting freely and transitively, an equivalence between the orbits under the
action of a subgroup and the quotient group. -/
@[to_additive "Given an additive group acting freely and transitively, an equivalence between the
orbits under the action of an additive subgroup and the quotient group."]
noncomputable def equivSubgroupOrbitsQuotientGroup [IsPretransitive α β]
    (free : ∀ y : β, MulAction.stabilizer α y = ⊥) (H : Subgroup α) :
    orbitRel.Quotient H β ≃ α ⧸ H where
  toFun := fun q ↦ q.liftOn' (fun y ↦ (exists_smul_eq α y x).choose) (by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      ⊢ ∀ (a b : β), (MulAction.orbitRel (Subtype fun x => Membership.mem H x) β) a  …
    -/
    intro y₁ y₂ h
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      y₁ y₂ : β
      h : (MulAction.orbitRel (Subtype fun x => Membership.mem H x) β) y₁ y₂
      ⊢ Eq ((fun y => ↑⋯.choose) y₁) ((fun y => ↑⋯.choose) y₂)
    -/
    rw [orbitRel_apply] at h
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      y₁ y₂ : β
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) y₂) y₁
      ⊢ Eq ((fun y => ↑⋯.choose) y₁) ((fun y => ↑⋯.choose) y₂)
    -/
    rw [Quotient.eq'', leftRel_eq]
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      y₁ y₂ : β
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) y₂) y₁
      ⊢ (fun x y => Membership.mem H (HMul.hMul (Inv.inv x) y)) ⋯.choose ⋯.choose
    -/
    dsimp only
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      y₁ y₂ : β
      h : Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) y₂) y₁
      ⊢ Membership.mem H (HMul.hMul (Inv.inv ⋯.choose) ⋯.choose)
    -/
    rcases h with ⟨g, rfl⟩
    /-
      case intro
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      y₂ : β
      g : Subtype fun x => Membership.mem H x
      ⊢ Membership.mem H (HMul.hMul (Inv.inv ⋯.choose) ⋯.choose)
    -/
    dsimp only
    suffices (exists_smul_eq α (g • y₂) x).choose = (exists_smul_eq α y₂ x).choose * g⁻¹ by
      simp [this]
    /-
      case intro
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      y₂ : β
      g : Subtype fun x => Membership.mem H x
      ⊢ Eq ⋯.choose (HMul.hMul ⋯.choose ↑(Inv.inv g))
    -/
    rw [← inv_mul_eq_one, ← Subgroup.mem_bot, ← free ((g : α) • y₂)]
    simp only [mem_stabilizer_iff, smul_smul, mul_assoc, InvMemClass.coe_inv, inv_mul_cancel,
               mul_one]
    rw [← smul_smul, (exists_smul_eq α y₂ x).choose_spec, inv_smul_eq_iff,
        (exists_smul_eq α ((g : α) • y₂) x).choose_spec])
  invFun := fun q ↦ q.liftOn' (fun g ↦ ⟦g⁻¹ • x⟧) (by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : HasQuotient.Quotient α H
      ⊢ ∀ (a b : α), (QuotientGroup.leftRel H) a b → Eq ((fun g => Quotient.mk (MulA …
    -/
    intro g₁ g₂ h
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : HasQuotient.Quotient α H
      g₁ g₂ : α
      h : (QuotientGroup.leftRel H) g₁ g₂
      ⊢ Eq ((fun g => Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.m …
    -/
    rw [leftRel_eq] at h
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : HasQuotient.Quotient α H
      g₁ g₂ : α
      h : Membership.mem H (HMul.hMul (Inv.inv g₁) g₂)
      ⊢ Eq ((fun g => Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.m …
    -/
    simp only
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : HasQuotient.Quotient α H
      g₁ g₂ : α
      h : Membership.mem H (HMul.hMul (Inv.inv g₁) g₂)
      ⊢ Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x) β) …
    -/
    rw [← @Quotient.mk''_eq_mk, Quotient.eq'', orbitRel_apply]
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      q : HasQuotient.Quotient α H
      g₁ g₂ : α
      h : Membership.mem H (HMul.hMul (Inv.inv g₁) g₂)
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) (HSMul …
    -/
    exact ⟨⟨_, h⟩, by simp [mul_smul]⟩)
    /-
      🎉 no goals
    -/
  left_inv := fun y ↦ by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      y : MulAction.orbitRel.Quotient (Subtype fun x => Membership.mem H x) β
      ⊢ Eq ((fun q => Quotient.liftOn' q (fun g => Quotient.mk (MulAction.orbitRel ( …
    -/
    induction' y using Quotient.inductionOn' with y
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      y : β
      ⊢ Eq ((fun q => Quotient.liftOn' q (fun g => Quotient.mk (MulAction.orbitRel ( …
    -/
    simp only [Quotient.liftOn'_mk'']
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      y : β
      ⊢ Eq (Quotient.mk (MulAction.orbitRel (Subtype fun x => Membership.mem H x) β) …
    -/
    rw [← @Quotient.mk''_eq_mk, Quotient.eq'', orbitRel_apply]
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      y : β
      ⊢ Membership.mem (MulAction.orbit (Subtype fun x => Membership.mem H x) y) (HS …
    -/
    convert mem_orbit_self _
    /-
      case h.e'_5
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      y : β
      ⊢ Eq (HSMul.hSMul (Inv.inv ⋯.choose) x) y
    -/
    rw [inv_smul_eq_iff, (exists_smul_eq α y x).choose_spec]
    /-
      🎉 no goals
    -/
  right_inv := fun g ↦ by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      g : HasQuotient.Quotient α H
      ⊢ Eq ((fun q => Quotient.liftOn' q (fun y => ↑⋯.choose) ⋯) ((fun q => Quotient …
    -/
    induction' g using Quotient.inductionOn' with g
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      g : α
      ⊢ Eq ((fun q => Quotient.liftOn' q (fun y => ↑⋯.choose) ⋯) ((fun q => Quotient …
    -/
    simp only [Quotient.liftOn'_mk'', Quotient.liftOn'_mk, QuotientGroup.mk]
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      g : α
      ⊢ Eq (Quotient.mk'' ⋯.choose) (Quotient.mk'' g)
    -/
    rw [Quotient.eq'', leftRel_eq]
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      g : α
      ⊢ (fun x y => Membership.mem H (HMul.hMul (Inv.inv x) y)) ⋯.choose g
    -/
    simp only
    /-
      case h
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : Group α
      inst✝¹ : MulAction α β
      x : β
      inst✝ : MulAction.IsPretransitive α β
      free : ∀ (y : β), Eq (MulAction.stabilizer α y) Bot.bot
      H : Subgroup α
      g : α
      ⊢ Membership.mem H (HMul.hMul (Inv.inv ⋯.choose) g)
    -/
    convert one_mem H
    · rw [inv_mul_eq_one, eq_comm, ← inv_mul_eq_one, ← Subgroup.mem_bot, ← free (g⁻¹ • x),
        mem_stabilizer_iff, mul_smul, (exists_smul_eq α (g⁻¹ • x) x).choose_spec]


/-- If `α` acts on `β` with trivial stabilizers, `β` is equivalent
to the product of the quotient of `β` by `α` and `α`.
See `MulAction.selfEquivOrbitsQuotientProd` with `φ = Quotient.out`. -/
@[to_additive "If `α` acts freely on `β`, `β` is equivalent
to the product of the quotient of `β` by `α` and `α`.
See `AddAction.selfEquivOrbitsQuotientProd` with `φ = Quotient.out`."]
noncomputable def selfEquivOrbitsQuotientProd'
    {φ : Quotient (MulAction.orbitRel α β) → β} (hφ : Function.LeftInverse Quotient.mk'' φ)
    (h : ∀ b : β, MulAction.stabilizer α b = ⊥) :
    β ≃ Quotient (MulAction.orbitRel α β) × α :=
  (MulAction.selfEquivSigmaOrbitsQuotientStabilizer' α β hφ).trans <|
    (Equiv.sigmaCongrRight <| fun _ ↦
      (Subgroup.quotientEquivOfEq (h _)).trans (QuotientGroup.quotientEquivSelf α)).trans <|
    Equiv.sigmaEquivProd _ _


/-- If `α` acts freely on `β`, `β` is equivalent to the product of the quotient of `β` by `α` and
`α`. -/
@[to_additive "If `α` acts freely on `β`, `β` is equivalent to the product of the quotient of `β` by
`α` and `α`."]
noncomputable def selfEquivOrbitsQuotientProd (h : ∀ b : β, MulAction.stabilizer α b = ⊥) :
    β ≃ Quotient (MulAction.orbitRel α β) × α :=
  MulAction.selfEquivOrbitsQuotientProd' Quotient.out_eq' h


theorem ConjClasses.card_carrier {G : Type*} [Group G] [Fintype G] (g : G)
    [Fintype (ConjClasses.mk g).carrier] [Fintype <| MulAction.stabilizer (ConjAct G) g] :
    Fintype.card (ConjClasses.mk g).carrier =
      Fintype.card G / Fintype.card (MulAction.stabilizer (ConjAct G) g) := by
  classical
  rw [Fintype.card_congr <| ConjAct.toConjAct (G := G) |>.toEquiv]
  rw [← MulAction.card_orbit_mul_card_stabilizer_eq_card_group (ConjAct G) g, Nat.mul_div_cancel]
  · simp_rw [ConjAct.orbit_eq_carrier_conjClasses]
  · exact Fintype.card_pos_iff.mpr inferInstance


theorem normalCore_eq_ker : H.normalCore = (MulAction.toPermHom G (G ⧸ H)).ker := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq H.normalCore (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker
  -/
  apply le_antisymm
    /-
      case a
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      ⊢ LE.le H.normalCore (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker
    -/
  · intro g hg
    /-
      case a
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem H.normalCore g
      ⊢ Membership.mem (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker g
    -/
    apply Equiv.Perm.ext
    /-
      case a.H
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem H.normalCore g
      ⊢ ∀ (x : HasQuotient.Quotient G H), Eq (((MulAction.toPermHom G (HasQuotient.Q …
    -/
    refine fun q ↦ QuotientGroup.induction_on q ?_
    /-
      case a.H
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem H.normalCore g
      q : HasQuotient.Quotient G H
      ⊢ ∀ (z : G), Eq (((MulAction.toPermHom G (HasQuotient.Quotient G H)) g) ↑z) (1 …
    -/
    refine fun g' => (MulAction.Quotient.smul_mk H g g').trans (QuotientGroup.eq.mpr ?_)
    /-
      case a.H
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem H.normalCore g
      q : HasQuotient.Quotient G H
      g' : G
      ⊢ Membership.mem H (HMul.hMul (Inv.inv (HSMul.hSMul g g')) g')
    -/
    rw [smul_eq_mul, mul_inv_rev, ← inv_inv g', inv_inv]
    /-
      case a.H
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem H.normalCore g
      q : HasQuotient.Quotient G H
      g' : G
      ⊢ Membership.mem H (HMul.hMul (HMul.hMul (Inv.inv g') (Inv.inv g)) (Inv.inv (I …
    -/
    exact H.normalCore.inv_mem hg g'⁻¹
    /-
      🎉 no goals
    -/
    /-
      case a
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      ⊢ LE.le (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker H.normalCore
    -/
  · refine (Subgroup.normal_le_normalCore.mpr fun g hg => ?_)
    /-
      case a
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker g
      ⊢ Membership.mem H g
    -/
    rw [← H.inv_mem_iff, ← mul_one g⁻¹, ← QuotientGroup.eq, ← mul_one g]
    /-
      case a
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      g : G
      hg : Membership.mem (MulAction.toPermHom G (HasQuotient.Quotient G H)).ker g
      ⊢ Eq ↑(HMul.hMul g 1) ↑1
    -/
    exact (MulAction.Quotient.smul_mk H g 1).symm.trans (Equiv.Perm.ext_iff.mp hg (1 : G))
    /-
      🎉 no goals
    -/


/-- Cosets of the centralizer of an element embed into the set of commutators. -/
noncomputable def quotientCentralizerEmbedding (g : G) :
    G ⧸ centralizer (zpowers (g : G)) ↪ commutatorSet G :=
  ((MulAction.orbitEquivQuotientStabilizer (ConjAct G) g).trans
            (quotientEquivOfEq (ConjAct.stabilizer_eq_centralizer g))).symm.toEmbedding.trans
    ⟨fun x =>
      ⟨x * g⁻¹,
        let ⟨_, x, rfl⟩ := x
        ⟨x, g, rfl⟩⟩,
      fun _ _ => Subtype.ext ∘ mul_right_cancel ∘ Subtype.ext_iff.mp⟩


theorem quotientCentralizerEmbedding_apply (g : G) (x : G) :
    quotientCentralizerEmbedding g x = ⟨⁅x, g⁆, x, g, rfl⟩ :=
  rfl


/-- If `G` is generated by `S`, then the quotient by the center embeds into `S`-indexed sequences
of commutators. -/
noncomputable def quotientCenterEmbedding {S : Set G} (hS : closure S = ⊤) :
    G ⧸ center G ↪ S → commutatorSet G :=
  (quotientEquivOfEq (center_eq_infi' S hS)).toEmbedding.trans
    ((quotientiInfEmbedding _).trans
      (Function.Embedding.piCongrRight fun g => quotientCentralizerEmbedding (g : G)))


theorem quotientCenterEmbedding_apply {S : Set G} (hS : closure S = ⊤) (g : G) (s : S) :
    quotientCenterEmbedding hS g s = ⟨⁅g, s⁆, g, s, rfl⟩ :=
  rfl


