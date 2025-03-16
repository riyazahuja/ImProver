/-- The quotient of the transversals of an abelian normal `N` by the `diff` relation. -/
def QuotientDiff :=
  Quotient
    (Setoid.mk (fun α β => diff (MonoidHom.id H) α β = 1)
                                                          /-
                                                            G : Type u_1
                                                            inst✝² : Group G
                                                            H : Subgroup G
                                                            inst✝¹ : H.IsCommutative
                                                            inst✝ : H.FiniteIndex
                                                            α β x✝ y✝ : H.LeftTransversal
                                                            h : Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Members …
                                                            ⊢ Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Membershi …
                                                          -/
      ⟨fun α => diff_self (MonoidHom.id H) α, fun h => by rw [← diff_inv, h, inv_one],
                                                          /-
                                                            🎉 no goals
                                                          -/
                       /-
                         G : Type u_1
                         inst✝² : Group G
                         H : Subgroup G
                         inst✝¹ : H.IsCommutative
                         inst✝ : H.FiniteIndex
                         α β x✝ y✝ z✝ : H.LeftTransversal
                         h : Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Members …
                         h' : Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Member …
                         ⊢ Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Membershi …
                       -/
        fun h h' => by rw [← diff_mul_diff, h, h', one_mul]⟩)
                       /-
                         🎉 no goals
                       -/


instance : Inhabited H.QuotientDiff := by
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    inst✝¹ : H.IsCommutative
    inst✝ : H.FiniteIndex
    α β : H.LeftTransversal
    ⊢ Inhabited H.QuotientDiff
  -/
  dsimp [QuotientDiff] -- Porting note: Added `dsimp`
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    inst✝¹ : H.IsCommutative
    inst✝ : H.FiniteIndex
    α β : H.LeftTransversal
    ⊢ Inhabited (Quotient { r := fun α β => Eq (Subgroup.leftTransversals.diff (Mo …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem smul_diff_smul' [hH : Normal H] (g : Gᵐᵒᵖ) :
    diff (MonoidHom.id H) (g • α) (g • β) =
      ⟨g.unop⁻¹ * (diff (MonoidHom.id H) α β : H) * g.unop,
        hH.mem_comm ((congr_arg (· ∈ H) (mul_inv_cancel_left _ _)).mpr (SetLike.coe_mem _))⟩ := by
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    inst✝¹ : H.IsCommutative
    inst✝ : H.FiniteIndex
    α β : H.LeftTransversal
    hH : H.Normal
    g : MulOpposite G
    ⊢ Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Membershi …
  -/
  letI := H.fintypeQuotientOfFiniteIndex
  let ϕ : H →* H :=
    { toFun := fun h =>
        ⟨g.unop⁻¹ * h * g.unop,
          hH.mem_comm ((congr_arg (· ∈ H) (mul_inv_cancel_left _ _)).mpr (SetLike.coe_mem _))⟩
      map_one' := by rw [Subtype.ext_iff, coe_mk, coe_one, mul_one, inv_mul_cancel]
      map_mul' := fun h₁ h₂ => by
        simp only [Subtype.ext_iff, coe_mk, coe_mul, mul_assoc, mul_inv_cancel_left] }
  /-
    G : Type u_1
    inst✝² : Group G
    H : Subgroup G
    inst✝¹ : H.IsCommutative
    inst✝ : H.FiniteIndex
    α β : H.LeftTransversal
    hH : H.Normal
    g : MulOpposite G
    this : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    ϕ : MonoidHom (Subtype fun x => Membership.mem H x) (Subtype fun x => Membersh …
    ⊢ Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Membershi …
  -/
  refine (Fintype.prod_equiv (MulAction.toPerm g).symm _ _ fun x ↦ ?_).trans (map_prod ϕ _ _).symm
  simp only [ϕ, smul_apply_eq_smul_apply_inv_smul, smul_eq_mul_unop, mul_inv_rev, mul_assoc,
    MonoidHom.id_apply, toPerm_symm_apply, MonoidHom.coe_mk, OneHom.coe_mk]


noncomputable instance : MulAction G H.QuotientDiff where
  smul g :=
    Quotient.map' (fun α => op g⁻¹ • α) fun α β h =>
      Subtype.ext
        (by
          rwa [smul_diff_smul', coe_mk, coe_one, mul_eq_one_iff_eq_inv, mul_right_eq_self, ←
            coe_one, ← Subtype.ext_iff])
  mul_smul g₁ g₂ q :=
    Quotient.inductionOn' q fun T =>
                                  /-
                                    G : Type u_1
                                    inst✝³ : Group G
                                    H : Subgroup G
                                    inst✝² : H.IsCommutative
                                    inst✝¹ : H.FiniteIndex
                                    α β : H.LeftTransversal
                                    inst✝ : H.Normal
                                    g₁ g₂ : G
                                    q : H.QuotientDiff
                                    T : H.LeftTransversal
                                    ⊢ Eq ((fun α => HSMul.hSMul (MulOpposite.op (Inv.inv (HMul.hMul g₁ g₂))) α) T) …
                                  -/
      congr_arg Quotient.mk'' (by rw [mul_inv_rev]; exact mul_smul (op g₁⁻¹) (op g₂⁻¹) T)
                                                    /-
                                                      🎉 no goals
                                                    -/
                                  /-
                                    G : Type u_1
                                    inst✝³ : Group G
                                    H : Subgroup G
                                    inst✝² : H.IsCommutative
                                    inst✝¹ : H.FiniteIndex
                                    α β : H.LeftTransversal
                                    inst✝ : H.Normal
                                    q : H.QuotientDiff
                                    T : H.LeftTransversal
                                    ⊢ Eq ((fun α => HSMul.hSMul (MulOpposite.op (Inv.inv 1)) α) T) T
                                  -/
  one_smul q :=
                                                /-
                                                  🎉 no goals
                                                -/
    Quotient.inductionOn' q fun T =>
      congr_arg Quotient.mk'' (by rw [inv_one]; apply one_smul Gᵐᵒᵖ T)


theorem smul_diff' (h : H) :
    diff (MonoidHom.id H) α (op (h : G) • β) = diff (MonoidHom.id H) α β * h ^ H.index := by
  /-
    G : Type u_1
    inst✝³ : Group G
    H : Subgroup G
    inst✝² : H.IsCommutative
    inst✝¹ : H.FiniteIndex
    α β : H.LeftTransversal
    inst✝ : H.Normal
    h : Subtype fun x => Membership.mem H x
    ⊢ Eq (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x => Membershi …
  -/
  letI := H.fintypeQuotientOfFiniteIndex
  rw [diff, diff, index_eq_card, Nat.card_eq_fintype_card,
      ← Finset.card_univ, ← Finset.prod_const, ← Finset.prod_mul_distrib]
  /-
    G : Type u_1
    inst✝³ : Group G
    H : Subgroup G
    inst✝² : H.IsCommutative
    inst✝¹ : H.FiniteIndex
    α β : H.LeftTransversal
    inst✝ : H.Normal
    h : Subtype fun x => Membership.mem H x
    this : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    ⊢ Eq (Finset.univ.prod fun q => (MonoidHom.id (Subtype fun x => Membership.mem …
  -/
  refine Finset.prod_congr rfl fun q _ => ?_
  /-
    G : Type u_1
    inst✝³ : Group G
    H : Subgroup G
    inst✝² : H.IsCommutative
    inst✝¹ : H.FiniteIndex
    α β : H.LeftTransversal
    inst✝ : H.Normal
    h : Subtype fun x => Membership.mem H x
    this : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    q : HasQuotient.Quotient G H
    x✝ : Membership.mem Finset.univ q
    ⊢ Eq ((MonoidHom.id (Subtype fun x => Membership.mem H x)) ⟨HMul.hMul (Inv.inv …
  -/
  simp_rw [Subtype.ext_iff, MonoidHom.id_apply, coe_mul, mul_assoc, mul_right_inj]
  rw [smul_apply_eq_smul_apply_inv_smul, smul_eq_mul_unop, MulOpposite.unop_op, mul_left_inj,
    ← Subtype.ext_iff, Equiv.apply_eq_iff_eq, inv_smul_eq_iff]
  /-
    G : Type u_1
    inst✝³ : Group G
    H : Subgroup G
    inst✝² : H.IsCommutative
    inst✝¹ : H.FiniteIndex
    α β : H.LeftTransversal
    inst✝ : H.Normal
    h : Subtype fun x => Membership.mem H x
    this : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    q : HasQuotient.Quotient G H
    x✝ : Membership.mem Finset.univ q
    ⊢ Eq q (HSMul.hSMul (MulOpposite.op ↑h) q)
  -/
  exact self_eq_mul_right.mpr ((QuotientGroup.eq_one_iff _).mpr h.2)
  /-
    🎉 no goals
  -/


theorem eq_one_of_smul_eq_one (hH : Nat.Coprime (Nat.card H) H.index) (α : H.QuotientDiff)
    (h : H) : h • α = α → h = 1 :=
  Quotient.inductionOn' α fun α hα =>
    (powCoprime hH).injective <|
      calc
        h ^ H.index = diff (MonoidHom.id H) (op ((h⁻¹ : H) : G) • α) α := by
          /-
            G : Type u_1
            inst✝³ : Group G
            H : Subgroup G
            inst✝² : H.IsCommutative
            inst✝¹ : H.FiniteIndex
            inst✝ : H.Normal
            hH : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
            α✝ : H.QuotientDiff
            h : Subtype fun x => Membership.mem H x
            α : H.LeftTransversal
            hα : Eq (HSMul.hSMul h (Quotient.mk'' α)) (Quotient.mk'' α)
            ⊢ Eq (HPow.hPow h H.index) (Subgroup.leftTransversals.diff (MonoidHom.id (Subt …
          -/
          rw [← diff_inv, smul_diff', diff_self, one_mul, inv_pow, inv_inv]
          /-
            🎉 no goals
          -/
        _ = 1 ^ H.index := (Quotient.exact' hα).trans (one_pow H.index).symm


theorem exists_smul_eq (hH : Nat.Coprime (Nat.card H) H.index) (α β : H.QuotientDiff) :
    ∃ h : H, h • α = β :=
  Quotient.inductionOn' α
    (Quotient.inductionOn' β fun β α =>
      Exists.imp (fun _ => Quotient.sound')
        ⟨(powCoprime hH).symm (diff (MonoidHom.id H) β α),
          (diff_inv _ _ _).symm.trans
            (inv_eq_one.mpr
              ((smul_diff' β α ((powCoprime hH).symm (diff (MonoidHom.id H) β α))⁻¹).trans
                    /-
                      G : Type u_1
                      inst✝³ : Group G
                      H : Subgroup G
                      inst✝² : H.IsCommutative
                      inst✝¹ : H.FiniteIndex
                      inst✝ : H.Normal
                      hH : (Nat.card (Subtype fun x => Membership.mem H x)).Coprime H.index
                      α✝ β✝ : H.QuotientDiff
                      β α : H.LeftTransversal
                      ⊢ Eq (HMul.hMul (Subgroup.leftTransversals.diff (MonoidHom.id (Subtype fun x = …
                    -/
                (by rw [inv_pow, ← powCoprime_apply hH, Equiv.apply_symm_apply, mul_inv_cancel])))⟩)
                    /-
                      🎉 no goals
                    -/


theorem isComplement'_stabilizer_of_coprime {α : H.QuotientDiff}
    (hH : Nat.Coprime (Nat.card H) H.index) : IsComplement' H (stabilizer G α) :=
  isComplement'_stabilizer α (eq_one_of_smul_eq_one hH α) fun g => exists_smul_eq hH (g • α) α


/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem exists_right_complement'_of_coprime_aux (hH : Nat.Coprime (Nat.card H) H.index) :
    ∃ K : Subgroup G, IsComplement' H K :=
  have ne : Nonempty (QuotientDiff H) := inferInstance
  ne.elim fun α => ⟨stabilizer G α, isComplement'_stabilizer_of_coprime hH⟩


/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step0 : N ≠ ⊥ := by
  /-
    G : Type u
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    ⊢ Ne N Bot.bot
  -/
  rintro rfl
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Bot.bot.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem Bot.bot x)).Coprime Bot.bot.in …
    h3 : ∀ (H : Subgroup G), Not (Bot.bot.IsComplement' H)
    ⊢ False
  -/
  exact h3 ⊤ isComplement'_bot_top
  /-
    🎉 no goals
  -/


include h2 in
/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step1 (K : Subgroup G) (hK : K ⊔ N = ⊤) : K = ⊤ := by
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    K : Subgroup G
    hK : Eq (Max.max K N) Top.top
    ⊢ Eq K Top.top
  -/
  contrapose! h3
  have h4 : (N.comap K.subtype).index = N.index := by
    rw [← N.relindex_top_right, ← hK]
    exact (relindex_sup_right K N).symm
  have h5 : Nat.card K < Nat.card G := by
    rw [← K.index_mul_card]
    exact lt_mul_of_one_lt_left Nat.card_pos (one_lt_index_of_ne_top h3)
  have h6 : Nat.Coprime (Nat.card (N.comap K.subtype)) (N.comap K.subtype).index := by
    rw [h4]
    exact h1.coprime_dvd_left (card_comap_dvd_of_injective N K.subtype Subtype.coe_injective)
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    inst✝ : Finite G
    K : Subgroup G
    hK : Eq (Max.max K N) Top.top
    h3 : Ne K Top.top
    h4 : Eq (Subgroup.comap K.subtype N).index N.index
    h5 : LT.lt (Nat.card (Subtype fun x => Membership.mem K x)) (Nat.card G)
    h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.comap K.subtype N) x …
    ⊢ Exists fun H => N.IsComplement' H
  -/
  obtain ⟨H, hH⟩ := h2 K h5 h6
  replace hH : Nat.card (H.map K.subtype) = N.index := by
    rw [← relindex_bot_left, ← relindex_comap, MonoidHom.comap_bot, Subgroup.ker_subtype,
      relindex_bot_left, ← IsComplement'.index_eq_card (IsComplement'.symm hH), index_comap,
      range_subtype, ← relindex_sup_right, hK, relindex_top_right]
  have h7 : Nat.card N * Nat.card (H.map K.subtype) = Nat.card G := by
    rw [hH, ← N.index_mul_card, mul_comm]
  have h8 : (Nat.card N).Coprime (Nat.card (H.map K.subtype)) := by
    rwa [hH]
  /-
    case intro
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    inst✝ : Finite G
    K : Subgroup G
    hK : Eq (Max.max K N) Top.top
    h3 : Ne K Top.top
    h4 : Eq (Subgroup.comap K.subtype N).index N.index
    h5 : LT.lt (Nat.card (Subtype fun x => Membership.mem K x)) (Nat.card G)
    h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.comap K.subtype N) x …
    H : Subgroup (Subtype fun x => Membership.mem K x)
    hH : Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.map K.subtype H)  …
    h7 : Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem N x)) (Nat.card  …
    h8 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime (Nat.card (Subty …
    ⊢ Exists fun H => N.IsComplement' H
  -/
  exact ⟨H.map K.subtype, isComplement'_of_coprime h7 h8⟩
  /-
    🎉 no goals
  -/


include h2 in
/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step2 (K : Subgroup G) [K.Normal] (hK : K ≤ N) : K = ⊥ ∨ K = N := by
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup G
    inst✝ : K.Normal
    hK : LE.le K N
    ⊢ Or (Eq K Bot.bot) (Eq K N)
  -/
  have : Function.Surjective (QuotientGroup.mk' K) := Quotient.mk''_surjective
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup G
    inst✝ : K.Normal
    hK : LE.le K N
    this : Function.Surjective ⇑(QuotientGroup.mk' K)
    ⊢ Or (Eq K Bot.bot) (Eq K N)
  -/
  have h4 := step1 h1 h2 h3
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup G
    inst✝ : K.Normal
    hK : LE.le K N
    this : Function.Surjective ⇑(QuotientGroup.mk' K)
    h4 : ∀ (K : Subgroup G), Eq (Max.max K N) Top.top → Eq K Top.top
    ⊢ Or (Eq K Bot.bot) (Eq K N)
  -/
  contrapose! h4
  have h5 : Nat.card (G ⧸ K) < Nat.card G := by
    rw [← index_eq_card, ← K.index_mul_card]
    refine
      lt_mul_of_one_lt_right (Nat.pos_of_ne_zero index_ne_zero_of_finite)
        (K.one_lt_card_iff_ne_bot.mpr h4.1)
  have h6 :
    (Nat.card (N.map (QuotientGroup.mk' K))).Coprime (N.map (QuotientGroup.mk' K)).index := by
    have index_map := N.index_map_eq this (by rwa [QuotientGroup.ker_mk'])
    have index_pos : 0 < N.index := Nat.pos_of_ne_zero index_ne_zero_of_finite
    rw [index_map]
    refine h1.coprime_dvd_left ?_
    rw [← Nat.mul_dvd_mul_iff_left index_pos, index_mul_card, ← index_map, index_mul_card]
    exact K.card_quotient_dvd_card
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup G
    inst✝ : K.Normal
    hK : LE.le K N
    this : Function.Surjective ⇑(QuotientGroup.mk' K)
    h4 : And (Ne K Bot.bot) (Ne K N)
    h5 : LT.lt (Nat.card (HasQuotient.Quotient G K)) (Nat.card G)
    h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.map (QuotientGroup.m …
    ⊢ Exists fun K => And (Eq (Max.max K N) Top.top) (Ne K Top.top)
  -/
  obtain ⟨H, hH⟩ := h2 (G ⧸ K) h5 h6
  /-
    case intro
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup G
    inst✝ : K.Normal
    hK : LE.le K N
    this : Function.Surjective ⇑(QuotientGroup.mk' K)
    h4 : And (Ne K Bot.bot) (Ne K N)
    h5 : LT.lt (Nat.card (HasQuotient.Quotient G K)) (Nat.card G)
    h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.map (QuotientGroup.m …
    H : Subgroup (HasQuotient.Quotient G K)
    hH : (Subgroup.map (QuotientGroup.mk' K) N).IsComplement' H
    ⊢ Exists fun K => And (Eq (Max.max K N) Top.top) (Ne K Top.top)
  -/
  refine ⟨H.comap (QuotientGroup.mk' K), ?_, ?_⟩
  · have key : (N.map (QuotientGroup.mk' K)).comap (QuotientGroup.mk' K) = N := by
      refine comap_map_eq_self ?_
      rwa [QuotientGroup.ker_mk']
    /-
      case intro.refine_1
      G : Type u
      inst✝³ : Group G
      N : Subgroup G
      inst✝² : N.Normal
      h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
      h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
      h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
      inst✝¹ : Finite G
      K : Subgroup G
      inst✝ : K.Normal
      hK : LE.le K N
      this : Function.Surjective ⇑(QuotientGroup.mk' K)
      h4 : And (Ne K Bot.bot) (Ne K N)
      h5 : LT.lt (Nat.card (HasQuotient.Quotient G K)) (Nat.card G)
      h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.map (QuotientGroup.m …
      H : Subgroup (HasQuotient.Quotient G K)
      hH : (Subgroup.map (QuotientGroup.mk' K) N).IsComplement' H
      key : Eq (Subgroup.comap (QuotientGroup.mk' K) (Subgroup.map (QuotientGroup.mk …
      ⊢ Eq (Max.max (Subgroup.comap (QuotientGroup.mk' K) H) N) Top.top
    -/
    rwa [← key, comap_sup_eq, hH.symm.sup_eq_top, comap_top]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      G : Type u
      inst✝³ : Group G
      N : Subgroup G
      inst✝² : N.Normal
      h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
      h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
      h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
      inst✝¹ : Finite G
      K : Subgroup G
      inst✝ : K.Normal
      hK : LE.le K N
      this : Function.Surjective ⇑(QuotientGroup.mk' K)
      h4 : And (Ne K Bot.bot) (Ne K N)
      h5 : LT.lt (Nat.card (HasQuotient.Quotient G K)) (Nat.card G)
      h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.map (QuotientGroup.m …
      H : Subgroup (HasQuotient.Quotient G K)
      hH : (Subgroup.map (QuotientGroup.mk' K) N).IsComplement' H
      ⊢ Ne (Subgroup.comap (QuotientGroup.mk' K) H) Top.top
    -/
  · rw [← comap_top (QuotientGroup.mk' K)]
    /-
      case intro.refine_2
      G : Type u
      inst✝³ : Group G
      N : Subgroup G
      inst✝² : N.Normal
      h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
      h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
      h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
      inst✝¹ : Finite G
      K : Subgroup G
      inst✝ : K.Normal
      hK : LE.le K N
      this : Function.Surjective ⇑(QuotientGroup.mk' K)
      h4 : And (Ne K Bot.bot) (Ne K N)
      h5 : LT.lt (Nat.card (HasQuotient.Quotient G K)) (Nat.card G)
      h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.map (QuotientGroup.m …
      H : Subgroup (HasQuotient.Quotient G K)
      hH : (Subgroup.map (QuotientGroup.mk' K) N).IsComplement' H
      ⊢ Ne (Subgroup.comap (QuotientGroup.mk' K) H) (Subgroup.comap (QuotientGroup.m …
    -/
    intro hH'
    rw [comap_injective this hH', isComplement'_top_right, map_eq_bot_iff,
      QuotientGroup.ker_mk'] at hH
    /-
      case intro.refine_2
      G : Type u
      inst✝³ : Group G
      N : Subgroup G
      inst✝² : N.Normal
      h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
      h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
      h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
      inst✝¹ : Finite G
      K : Subgroup G
      inst✝ : K.Normal
      hK : LE.le K N
      this : Function.Surjective ⇑(QuotientGroup.mk' K)
      h4 : And (Ne K Bot.bot) (Ne K N)
      h5 : LT.lt (Nat.card (HasQuotient.Quotient G K)) (Nat.card G)
      h6 : (Nat.card (Subtype fun x => Membership.mem (Subgroup.map (QuotientGroup.m …
      H : Subgroup (HasQuotient.Quotient G K)
      hH : LE.le N K
      hH' : Eq (Subgroup.comap (QuotientGroup.mk' K) H) (Subgroup.comap (QuotientGro …
      ⊢ False
    -/
    exact h4.2 (le_antisymm hK hH)
    /-
      🎉 no goals
    -/


include h2 in
/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step3 (K : Subgroup N) [(K.map N.subtype).Normal] : K = ⊥ ∨ K = ⊤ := by
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup (Subtype fun x => Membership.mem N x)
    inst✝ : (Subgroup.map N.subtype K).Normal
    ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
  -/
  have key := step2 h1 h2 h3 (K.map N.subtype) (map_subtype_le K)
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup (Subtype fun x => Membership.mem N x)
    inst✝ : (Subgroup.map N.subtype K).Normal
    key : Or (Eq (Subgroup.map N.subtype K) Bot.bot) (Eq (Subgroup.map N.subtype K …
    ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
  -/
  rw [← map_bot N.subtype] at key
  conv at key =>
    rhs
    rhs
    rw [← N.range_subtype, N.subtype.range_eq_map]
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup (Subtype fun x => Membership.mem N x)
    inst✝ : (Subgroup.map N.subtype K).Normal
    key : Or (Eq (Subgroup.map N.subtype K) (Subgroup.map N.subtype Bot.bot)) (Eq  …
    ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
  -/
  have inj := map_injective N.subtype_injective
  /-
    G : Type u
    inst✝³ : Group G
    N : Subgroup G
    inst✝² : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝¹ : Finite G
    K : Subgroup (Subtype fun x => Membership.mem N x)
    inst✝ : (Subgroup.map N.subtype K).Normal
    key : Or (Eq (Subgroup.map N.subtype K) (Subgroup.map N.subtype Bot.bot)) (Eq  …
    inj : Function.Injective (Subgroup.map N.subtype)
    ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
  -/
  rwa [inj.eq_iff, inj.eq_iff] at key
  /-
    🎉 no goals
  -/


/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step4 : (Nat.card N).minFac.Prime :=
  Nat.minFac_prime (N.one_lt_card_iff_ne_bot.mpr (step0 h1 h3)).ne'


/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step5 {P : Sylow (Nat.card N).minFac N} : P.1 ≠ ⊥ := by
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    P : Sylow (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fun …
    ⊢ Ne (↑P) Bot.bot
  -/
  haveI : Fact (Nat.card N).minFac.Prime := ⟨step4 h1 h3⟩
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    P : Sylow (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fun …
    this : Fact (Nat.Prime (Nat.card (Subtype fun x => Membership.mem N x)).minFac)
    ⊢ Ne (↑P) Bot.bot
  -/
  apply P.ne_bot_of_dvd_card
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    P : Sylow (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fun …
    this : Fact (Nat.Prime (Nat.card (Subtype fun x => Membership.mem N x)).minFac)
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Nat.card (S …
  -/
  exact (Nat.card N).minFac_dvd
  /-
    🎉 no goals
  -/


include h2 in
/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem step6 : IsPGroup (Nat.card N).minFac N := by
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    ⊢ IsPGroup (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fu …
  -/
  haveI : Fact (Nat.card N).minFac.Prime := ⟨step4 h1 h3⟩
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    this : Fact (Nat.Prime (Nat.card (Subtype fun x => Membership.mem N x)).minFac)
    ⊢ IsPGroup (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fu …
  -/
  refine Sylow.nonempty.elim fun P => P.2.of_surjective P.1.subtype ?_
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    this : Fact (Nat.Prime (Nat.card (Subtype fun x => Membership.mem N x)).minFac)
    P : Sylow (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fun …
    ⊢ Function.Surjective ⇑(↑P).subtype
  -/
  rw [← MonoidHom.range_eq_top, range_subtype]
  haveI : (P.1.map N.subtype).Normal :=
    normalizer_eq_top_iff.mp (step1 h1 h2 h3 (P.map N.subtype).normalizer P.normalizer_sup_eq_top)
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    this✝ : Fact (Nat.Prime (Nat.card (Subtype fun x => Membership.mem N x)).minFac)
    P : Sylow (Nat.card (Subtype fun x => Membership.mem N x)).minFac (Subtype fun …
    this : (Subgroup.map N.subtype ↑P).Normal
    ⊢ Eq (↑P) Top.top
  -/
  exact (step3 h1 h2 h3 P.1).resolve_left (step5 h1 h3)
  /-
    🎉 no goals
  -/


include h2 in
/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
theorem step7 : IsCommutative N := by
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    ⊢ N.IsCommutative
  -/
  haveI := N.bot_or_nontrivial.resolve_left (step0 h1 h3)
  /-
    G : Type u
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : N.Normal
    h1 : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h2 : ∀ (G' : Type u) [inst : Group G'] [inst_1 : Finite G'], LT.lt (Nat.card G …
    h3 : ∀ (H : Subgroup G), Not (N.IsComplement' H)
    inst✝ : Finite G
    this : Nontrivial (Subtype fun x => Membership.mem N x)
    ⊢ N.IsCommutative
  -/
  haveI : Fact (Nat.card N).minFac.Prime := ⟨step4 h1 h3⟩
  exact
    ⟨⟨fun g h => ((eq_top_iff.mp ((step3 h1 h2 h3 (center N)).resolve_left
      (step6 h1 h2 h3).bot_lt_center.ne') (mem_top h)).comm g).symm⟩⟩


/-- Do not use this lemma: It is made obsolete by `exists_right_complement'_of_coprime` -/
private theorem exists_right_complement'_of_coprime_aux' [Finite G] (hG : Nat.card G = n)
    {N : Subgroup G} [N.Normal] (hN : Nat.Coprime (Nat.card N) N.index) :
    ∃ H : Subgroup G, IsComplement' N H := by
  /-
    n : Nat
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    hG : Eq (Nat.card G) n
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    ⊢ Exists fun H => N.IsComplement' H
  -/
  revert G
  /-
    n : Nat
    ⊢ ∀ {G : Type u} [inst : Group G] [inst_1 : Finite G], Eq (Nat.card G) n → ∀ { …
  -/
  induction n using Nat.strongRecOn with | ind n ih => ?_
  /-
    case ind
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → ∀ {G : Type u} [inst : Group G] [inst_1 : Finite …
    ⊢ ∀ {G : Type u} [inst : Group G] [inst_1 : Finite G], Eq (Nat.card G) n → ∀ { …
  -/
  rintro G _ _ rfl N _ hN
  /-
    case ind
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    ih : ∀ (m : Nat), LT.lt m (Nat.card G) → ∀ {G : Type u} [inst : Group G] [inst …
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    ⊢ Exists fun H => N.IsComplement' H
  -/
  refine not_forall_not.mp fun h3 => ?_
  /-
    case ind
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    ih : ∀ (m : Nat), LT.lt m (Nat.card G) → ∀ {G : Type u} [inst : Group G] [inst …
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h3 : ∀ (x : Subgroup G), Not (N.IsComplement' x)
    ⊢ False
  -/
  haveI := SchurZassenhausInduction.step7 hN (fun G' _ _ hG' => by apply ih _ hG'; rfl) h3
  /-
    case ind
    G : Type u
    inst✝² : Group G
    inst✝¹ : Finite G
    ih : ∀ (m : Nat), LT.lt m (Nat.card G) → ∀ {G : Type u} [inst : Group G] [inst …
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    h3 : ∀ (x : Subgroup G), Not (N.IsComplement' x)
    this : N.IsCommutative
    ⊢ False
  -/
  exact not_exists_of_forall_not h3 (exists_right_complement'_of_coprime_aux hN)
  /-
    🎉 no goals
  -/


/-- **Schur-Zassenhaus** for normal subgroups:
  If `H : Subgroup G` is normal, and has order coprime to its index, then there exists a
  subgroup `K` which is a (right) complement of `H`. -/
theorem exists_right_complement'_of_coprime {N : Subgroup G} [N.Normal]
    (hN : Nat.Coprime (Nat.card N) N.index) : ∃ H : Subgroup G, IsComplement' N H := by
  /-
    G : Type u
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    ⊢ Exists fun H => N.IsComplement' H
  -/
  by_cases hN1 : Nat.card N = 0
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
      hN1 : Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0
      ⊢ Exists fun H => N.IsComplement' H
    -/
  · rw [hN1, Nat.coprime_zero_left, index_eq_one] at hN
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : Eq N Top.top
      hN1 : Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0
      ⊢ Exists fun H => N.IsComplement' H
    -/
    rw [hN]
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : Eq N Top.top
      hN1 : Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0
      ⊢ Exists fun H => Top.top.IsComplement' H
    -/
    exact ⟨⊥, isComplement'_top_bot⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    hN1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0)
    ⊢ Exists fun H => N.IsComplement' H
  -/
  by_cases hN2 : N.index = 0
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
      hN1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0)
      hN2 : Eq N.index 0
      ⊢ Exists fun H => N.IsComplement' H
    -/
  · rw [hN2, Nat.coprime_zero_right] at hN
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : Eq (Nat.card (Subtype fun x => Membership.mem N x)) 1
      hN1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0)
      hN2 : Eq N.index 0
      ⊢ Exists fun H => N.IsComplement' H
    -/
    haveI := (Cardinal.toNat_eq_one_iff_unique.mp hN).1
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : Eq (Nat.card (Subtype fun x => Membership.mem N x)) 1
      hN1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0)
      hN2 : Eq N.index 0
      this : Subsingleton (Subtype fun x => Membership.mem N x)
      ⊢ Exists fun H => N.IsComplement' H
    -/
    rw [N.eq_bot_of_subsingleton]
    /-
      case pos
      G : Type u
      inst✝¹ : Group G
      N : Subgroup G
      inst✝ : N.Normal
      hN : Eq (Nat.card (Subtype fun x => Membership.mem N x)) 1
      hN1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0)
      hN2 : Eq N.index 0
      this : Subsingleton (Subtype fun x => Membership.mem N x)
      ⊢ Exists fun H => Bot.bot.IsComplement' H
    -/
    exact ⟨⊤, isComplement'_bot_top⟩
    /-
      🎉 no goals
    -/
  have hN3 : Nat.card G ≠ 0 := by
    rw [← N.card_mul_index]
    exact mul_ne_zero hN1 hN2
  haveI := (Cardinal.lt_aleph0_iff_fintype.mp
    (lt_of_not_ge (mt Cardinal.toNat_apply_of_aleph0_le hN3))).some
  /-
    case neg
    G : Type u
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : N.Normal
    hN : (Nat.card (Subtype fun x => Membership.mem N x)).Coprime N.index
    hN1 : Not (Eq (Nat.card (Subtype fun x => Membership.mem N x)) 0)
    hN2 : Not (Eq N.index 0)
    hN3 : Ne (Nat.card G) 0
    this : Fintype G
    ⊢ Exists fun H => N.IsComplement' H
  -/
  exact exists_right_complement'_of_coprime_aux' rfl hN
  /-
    🎉 no goals
  -/


/-- **Schur-Zassenhaus** for normal subgroups:
  If `H : Subgroup G` is normal, and has order coprime to its index, then there exists a
  subgroup `K` which is a (left) complement of `H`. -/
theorem exists_left_complement'_of_coprime {N : Subgroup G} [N.Normal]
    (hN : Nat.Coprime (Nat.card N) N.index) : ∃ H : Subgroup G, IsComplement' H N :=
  Exists.imp (fun _ => IsComplement'.symm) (exists_right_complement'_of_coprime hN)


