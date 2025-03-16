/-- Two subgroups `H K` of `G` are commensurable if `H ⊓ K` has finite index in both `H` and `K` -/
def Commensurable (H K : Subgroup G) : Prop :=
  H.relindex K ≠ 0 ∧ K.relindex H ≠ 0


@[refl]
                                                                  /-
                                                                    G : Type u_1
                                                                    inst✝ : Group G
                                                                    H : Subgroup G
                                                                    ⊢ Commensurable H H
                                                                  -/
protected theorem refl (H : Subgroup G) : Commensurable H H := by simp [Commensurable]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem comm {H K : Subgroup G} : Commensurable H K ↔ Commensurable K H := and_comm


@[symm]
theorem symm {H K : Subgroup G} : Commensurable H K → Commensurable K H := And.symm


@[trans]
theorem trans {H K L : Subgroup G} (hhk : Commensurable H K) (hkl : Commensurable K L) :
    Commensurable H L :=
  ⟨Subgroup.relindex_ne_zero_trans hhk.1 hkl.1, Subgroup.relindex_ne_zero_trans hkl.2 hhk.2⟩


theorem equivalence : Equivalence (@Commensurable G _) :=
  ⟨Commensurable.refl, fun h => Commensurable.symm h, fun h₁ h₂ => Commensurable.trans h₁ h₂⟩


/-- Equivalence of `K/H ⊓ K` with `gKg⁻¹/gHg⁻¹ ⊓ gKg⁻¹`-/
def quotConjEquiv (H K : Subgroup G) (g : ConjAct G) :
    K ⧸ H.subgroupOf K ≃ (g • K).1 ⧸ (g • H).subgroupOf (g • K) :=
  Quotient.congr (K.equivSMul g).toEquiv fun a b => by
    /-
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      g : ConjAct G
      a b : Subtype fun x => Membership.mem K x
      ⊢ Iff ((QuotientGroup.leftRel (H.subgroupOf K)) a b) ((QuotientGroup.leftRel ( …
    -/
    dsimp
    rw [← Quotient.eq'', ← Quotient.eq'', QuotientGroup.eq, QuotientGroup.eq,
      Subgroup.mem_subgroupOf, Subgroup.mem_subgroupOf, ← map_inv, ← map_mul,
      Subgroup.equivSMul_apply_coe]
    /-
      G : Type u_1
      inst✝ : Group G
      H K : Subgroup G
      g : ConjAct G
      a b : Subtype fun x => Membership.mem K x
      ⊢ Iff (Membership.mem H ↑(HMul.hMul (Inv.inv a) b)) (Membership.mem (HSMul.hSM …
    -/
    exact Subgroup.smul_mem_pointwise_smul_iff.symm
    /-
      🎉 no goals
    -/


theorem commensurable_conj {H K : Subgroup G} (g : ConjAct G) :
    Commensurable H K ↔ Commensurable (g • H) (g • K) :=
  and_congr (not_iff_not.mpr (Eq.congr_left (Cardinal.toNat_congr (quotConjEquiv H K g))))
    (not_iff_not.mpr (Eq.congr_left (Cardinal.toNat_congr (quotConjEquiv K H g))))


theorem commensurable_inv (H : Subgroup G) (g : ConjAct G) :
                                                              /-
                                                                G : Type u_1
                                                                inst✝ : Group G
                                                                H : Subgroup G
                                                                g : ConjAct G
                                                                ⊢ Iff (Commensurable (HSMul.hSMul g H) H) (Commensurable H (HSMul.hSMul (Inv.i …
                                                              -/
    Commensurable (g • H) H ↔ Commensurable H (g⁻¹ • H) := by rw [commensurable_conj, inv_smul_smul]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- For `H` a subgroup of `G`, this is the subgroup of all elements `g : conjAut G`
such that `Commensurable (g • H) H` -/
def commensurator' (H : Subgroup G) : Subgroup (ConjAct G) where
  carrier := { g : ConjAct G | Commensurable (g • H) H }
                 /-
                   G : Type u_1
                   inst✝ : Group G
                   H : Subgroup G
                   ⊢ Membership.mem { carrier := setOf fun g => Commensurable (HSMul.hSMul g H) H …
                 -/
  one_mem' := by rw [Set.mem_setOf_eq, one_smul]
    /-
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a✝ b✝ : ConjAct G
      ha : Membership.mem (setOf fun g => Commensurable (HSMul.hSMul g H) H) a✝
      hb : Membership.mem (setOf fun g => Commensurable (HSMul.hSMul g H) H) b✝
      ⊢ Membership.mem (setOf fun g => Commensurable (HSMul.hSMul g H) H) (HMul.hMul …
    -/
                 /-
                   🎉 no goals
                 -/
    /-
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      a✝ b✝ : ConjAct G
      ha : Membership.mem (setOf fun g => Commensurable (HSMul.hSMul g H) H) a✝
      hb : Membership.mem (setOf fun g => Commensurable (HSMul.hSMul g H) H) b✝
      ⊢ Commensurable (HSMul.hSMul a✝ (HSMul.hSMul b✝ H)) H
    -/
  mul_mem' ha hb := by
    /-
      🎉 no goals
    -/
    rw [Set.mem_setOf_eq, mul_smul]
    exact trans ((commensurable_conj _).mp hb) ha
                   /-
                     G : Type u_1
                     inst✝ : Group G
                     H : Subgroup G
                     x✝¹ : ConjAct G
                     x✝ : Membership.mem { carrier := setOf fun g => Commensurable (HSMul.hSMul g H …
                     ⊢ Membership.mem { carrier := setOf fun g => Commensurable (HSMul.hSMul g H) H …
                   -/
  inv_mem' _ := by rwa [Set.mem_setOf_eq, comm, ← commensurable_inv]
                   /-
                     🎉 no goals
                   -/


/-- For `H` a subgroup of `G`, this is the subgroup of all elements `g : G`
such that `Commensurable (g H g⁻¹) H` -/
def commensurator (H : Subgroup G) : Subgroup G :=
  (commensurator' H).comap ConjAct.toConjAct.toMonoidHom


@[simp]
theorem commensurator'_mem_iff (H : Subgroup G) (g : ConjAct G) :
    g ∈ commensurator' H ↔ Commensurable (g • H) H := Iff.rfl


@[simp]
theorem commensurator_mem_iff (H : Subgroup G) (g : G) :
    g ∈ commensurator H ↔ Commensurable (ConjAct.toConjAct g • H) H := Iff.rfl


theorem eq {H K : Subgroup G} (hk : Commensurable H K) : commensurator H = commensurator K :=
  Subgroup.ext fun x =>
    let hx := (commensurable_conj x).1 hk
    ⟨fun h => hx.symm.trans (h.trans hk), fun h => hx.trans (h.trans hk.symm)⟩


