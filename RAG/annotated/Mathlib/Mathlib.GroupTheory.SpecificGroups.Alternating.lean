/-- The alternating group on a finite type, realized as a subgroup of `Equiv.Perm`.
  For $A_n$, use `alternatingGroup (Fin n)`. -/
def alternatingGroup : Subgroup (Perm α) :=
  sign.ker

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): manually added instance

instance alternatingGroup.instFintype : Fintype (alternatingGroup α) :=
  @Subtype.fintype _ _ sign.decidableMemKer _


instance [Subsingleton α] : Unique (alternatingGroup α) :=
  ⟨⟨1⟩, fun ⟨p, _⟩ => Subtype.eq (Subsingleton.elim p _)⟩


theorem alternatingGroup_eq_sign_ker : alternatingGroup α = sign.ker :=
  rfl


@[simp]
theorem mem_alternatingGroup {f : Perm α} : f ∈ alternatingGroup α ↔ sign f = 1 :=
  sign.mem_ker


theorem prod_list_swap_mem_alternatingGroup_iff_even_length {l : List (Perm α)}
    (hl : ∀ g ∈ l, IsSwap g) : l.prod ∈ alternatingGroup α ↔ Even l.length := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List (Equiv.Perm α)
    hl : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap
    ⊢ Iff (Membership.mem (alternatingGroup α) l.prod) (Even l.length)
  -/
  rw [mem_alternatingGroup, sign_prod_list_swap hl, neg_one_pow_eq_one_iff_even]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    l : List (Equiv.Perm α)
    hl : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap
    ⊢ Ne (-1) 1
  -/
  decide
  /-
    🎉 no goals
  -/


theorem IsThreeCycle.mem_alternatingGroup {f : Perm α} (h : IsThreeCycle f) :
    f ∈ alternatingGroup α :=
  Perm.mem_alternatingGroup.mpr h.sign


theorem finRotate_bit1_mem_alternatingGroup {n : ℕ} :
    finRotate (2 * n + 1) ∈ alternatingGroup (Fin (2 * n + 1)) := by
  /-
    n : Nat
    ⊢ Membership.mem (alternatingGroup (Fin (HAdd.hAdd (HMul.hMul 2 n) 1))) (finRo …
  -/
  rw [mem_alternatingGroup, sign_finRotate, pow_mul, pow_two, Int.units_mul_self, one_pow]
  /-
    🎉 no goals
  -/


theorem two_mul_card_alternatingGroup [Nontrivial α] :
    2 * card (alternatingGroup α) = card (Perm α) := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nontrivial α
    ⊢ Eq (HMul.hMul 2 (Fintype.card (Subtype fun x => Membership.mem (alternatingG …
  -/
  let this := (QuotientGroup.quotientKerEquivOfSurjective _ (sign_surjective α)).toEquiv
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nontrivial α
    this : Equiv (HasQuotient.Quotient (Equiv.Perm α) Equiv.Perm.sign.ker) (Units  …
    ⊢ Eq (HMul.hMul 2 (Fintype.card (Subtype fun x => Membership.mem (alternatingG …
  -/
  rw [← Fintype.card_units_int, ← Fintype.card_congr this]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nontrivial α
    this : Equiv (HasQuotient.Quotient (Equiv.Perm α) Equiv.Perm.sign.ker) (Units  …
    ⊢ Eq (HMul.hMul (Fintype.card (HasQuotient.Quotient (Equiv.Perm α) Equiv.Perm. …
  -/
  simp only [← Nat.card_eq_fintype_card]
  /-
    α : Type u_1
    inst✝² : Fintype α
    inst✝¹ : DecidableEq α
    inst✝ : Nontrivial α
    this : Equiv (HasQuotient.Quotient (Equiv.Perm α) Equiv.Perm.sign.ker) (Units  …
    ⊢ Eq (HMul.hMul (Nat.card (HasQuotient.Quotient (Equiv.Perm α) Equiv.Perm.sign …
  -/
  apply (Subgroup.card_eq_card_quotient_mul_card_subgroup _).symm
  /-
    🎉 no goals
  -/


instance normal : (alternatingGroup α).Normal :=
  sign.normal_ker


theorem isConj_of {σ τ : alternatingGroup α} (hc : IsConj (σ : Perm α) (τ : Perm α))
    (hσ : (σ : Perm α).support.card + 2 ≤ Fintype.card α) : IsConj σ τ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ τ : Subtype fun x => Membership.mem (alternatingGroup α) x
    hc : IsConj ↑σ ↑τ
    hσ : LE.le (HAdd.hAdd (↑σ).support.card 2) (Fintype.card α)
    ⊢ IsConj σ τ
  -/
  obtain ⟨σ, hσ⟩ := σ
  /-
    case mk
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    τ : Subtype fun x => Membership.mem (alternatingGroup α) x
    σ : Equiv.Perm α
    hσ✝ : Membership.mem (alternatingGroup α) σ
    hc : IsConj ↑⟨σ, hσ✝⟩ ↑τ
    hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
    ⊢ IsConj ⟨σ, hσ✝⟩ τ
  -/
  obtain ⟨τ, hτ⟩ := τ
  /-
    case mk.mk
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    hσ✝ : Membership.mem (alternatingGroup α) σ
    hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
    τ : Equiv.Perm α
    hτ : Membership.mem (alternatingGroup α) τ
    hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
    ⊢ IsConj ⟨σ, hσ✝⟩ ⟨τ, hτ⟩
  -/
  obtain ⟨π, hπ⟩ := isConj_iff.1 hc
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    hσ✝ : Membership.mem (alternatingGroup α) σ
    hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
    τ : Equiv.Perm α
    hτ : Membership.mem (alternatingGroup α) τ
    hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
    π : Equiv.Perm α
    hπ : Eq (HMul.hMul (HMul.hMul π ↑⟨σ, hσ✝⟩) (Inv.inv π)) ↑⟨τ, hτ⟩
    ⊢ IsConj ⟨σ, hσ✝⟩ ⟨τ, hτ⟩
  -/
  rw [Subtype.coe_mk, Subtype.coe_mk] at hπ
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    σ : Equiv.Perm α
    hσ✝ : Membership.mem (alternatingGroup α) σ
    hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
    τ : Equiv.Perm α
    hτ : Membership.mem (alternatingGroup α) τ
    hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
    π : Equiv.Perm α
    hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
    ⊢ IsConj ⟨σ, hσ✝⟩ ⟨τ, hτ⟩
  -/
  cases' Int.units_eq_one_or (Perm.sign π) with h h
    /-
      case mk.mk.intro.inl
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ✝ : Membership.mem (alternatingGroup α) σ
      hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
      τ : Equiv.Perm α
      hτ : Membership.mem (alternatingGroup α) τ
      hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
      π : Equiv.Perm α
      hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
      h : Eq (Equiv.Perm.sign π) 1
      ⊢ IsConj ⟨σ, hσ✝⟩ ⟨τ, hτ⟩
    -/
  · rw [isConj_iff]
    /-
      case mk.mk.intro.inl
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ✝ : Membership.mem (alternatingGroup α) σ
      hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
      τ : Equiv.Perm α
      hτ : Membership.mem (alternatingGroup α) τ
      hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
      π : Equiv.Perm α
      hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
      h : Eq (Equiv.Perm.sign π) 1
      ⊢ Exists fun c => Eq (HMul.hMul (HMul.hMul c ⟨σ, hσ✝⟩) (Inv.inv c)) ⟨τ, hτ⟩
    -/
    refine ⟨⟨π, mem_alternatingGroup.mp h⟩, Subtype.val_injective ?_⟩
    /-
      case mk.mk.intro.inl
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ✝ : Membership.mem (alternatingGroup α) σ
      hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
      τ : Equiv.Perm α
      hτ : Membership.mem (alternatingGroup α) τ
      hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
      π : Equiv.Perm α
      hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
      h : Eq (Equiv.Perm.sign π) 1
      ⊢ Eq ↑(HMul.hMul (HMul.hMul ⟨π, ⋯⟩ ⟨σ, hσ✝⟩) (Inv.inv ⟨π, ⋯⟩)) ↑⟨τ, hτ⟩
    -/
    simpa only [Subtype.val, Subgroup.coe_mul, coe_inv, coe_mk] using hπ
    /-
      🎉 no goals
    -/
  · have h2 : 2 ≤ σ.supportᶜ.card := by
      rw [Finset.card_compl, le_tsub_iff_left σ.support.card_le_univ]
      exact hσ
    /-
      case mk.mk.intro.inr
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ✝ : Membership.mem (alternatingGroup α) σ
      hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
      τ : Equiv.Perm α
      hτ : Membership.mem (alternatingGroup α) τ
      hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
      π : Equiv.Perm α
      hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
      h : Eq (Equiv.Perm.sign π) (-1)
      h2 : LE.le 2 (HasCompl.compl σ.support).card
      ⊢ IsConj ⟨σ, hσ✝⟩ ⟨τ, hτ⟩
    -/
    obtain ⟨a, ha, b, hb, ab⟩ := Finset.one_lt_card.1 h2
    /-
      case mk.mk.intro.inr.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ✝ : Membership.mem (alternatingGroup α) σ
      hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
      τ : Equiv.Perm α
      hτ : Membership.mem (alternatingGroup α) τ
      hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
      π : Equiv.Perm α
      hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
      h : Eq (Equiv.Perm.sign π) (-1)
      h2 : LE.le 2 (HasCompl.compl σ.support).card
      a : α
      ha : Membership.mem (HasCompl.compl σ.support) a
      b : α
      hb : Membership.mem (HasCompl.compl σ.support) b
      ab : Ne a b
      ⊢ IsConj ⟨σ, hσ✝⟩ ⟨τ, hτ⟩
    -/
    refine isConj_iff.2 ⟨⟨π * swap a b, ?_⟩, Subtype.val_injective ?_⟩
      /-
        case mk.mk.intro.inr.intro.intro.intro.intro.refine_1
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        hσ✝ : Membership.mem (alternatingGroup α) σ
        hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
        τ : Equiv.Perm α
        hτ : Membership.mem (alternatingGroup α) τ
        hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
        π : Equiv.Perm α
        hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
        h : Eq (Equiv.Perm.sign π) (-1)
        h2 : LE.le 2 (HasCompl.compl σ.support).card
        a : α
        ha : Membership.mem (HasCompl.compl σ.support) a
        b : α
        hb : Membership.mem (HasCompl.compl σ.support) b
        ab : Ne a b
        ⊢ Membership.mem (alternatingGroup α) (HMul.hMul π (Equiv.swap a b))
      -/
    · rw [mem_alternatingGroup, MonoidHom.map_mul, h, sign_swap ab, Int.units_mul_self]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.intro.inr.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        hσ✝ : Membership.mem (alternatingGroup α) σ
        hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
        τ : Equiv.Perm α
        hτ : Membership.mem (alternatingGroup α) τ
        hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
        π : Equiv.Perm α
        hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
        h : Eq (Equiv.Perm.sign π) (-1)
        h2 : LE.le 2 (HasCompl.compl σ.support).card
        a : α
        ha : Membership.mem (HasCompl.compl σ.support) a
        b : α
        hb : Membership.mem (HasCompl.compl σ.support) b
        ab : Ne a b
        ⊢ Eq ↑(HMul.hMul (HMul.hMul ⟨HMul.hMul π (Equiv.swap a b), ⋯⟩ ⟨σ, hσ✝⟩) (Inv.i …
      -/
    · simp only [← hπ, coe_mk, Subgroup.coe_mul, Subtype.val]
      have hd : Disjoint (swap a b) σ := by
        rw [disjoint_iff_disjoint_support, support_swap ab, Finset.disjoint_insert_left,
          Finset.disjoint_singleton_left]
        exact ⟨Finset.mem_compl.1 ha, Finset.mem_compl.1 hb⟩
      /-
        case mk.mk.intro.inr.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        hσ✝ : Membership.mem (alternatingGroup α) σ
        hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
        τ : Equiv.Perm α
        hτ : Membership.mem (alternatingGroup α) τ
        hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
        π : Equiv.Perm α
        hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
        h : Eq (Equiv.Perm.sign π) (-1)
        h2 : LE.le 2 (HasCompl.compl σ.support).card
        a : α
        ha : Membership.mem (HasCompl.compl σ.support) a
        b : α
        hb : Membership.mem (HasCompl.compl σ.support) b
        ab : Ne a b
        hd : (Equiv.swap a b).Disjoint σ
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul π (Equiv.swap a b)) σ) ↑(Inv.inv ⟨HMul.h …
      -/
      rw [mul_assoc π _ σ, hd.commute.eq, coe_inv, coe_mk]
      /-
        case mk.mk.intro.inr.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        hσ✝ : Membership.mem (alternatingGroup α) σ
        hσ : LE.le (HAdd.hAdd (↑⟨σ, hσ✝⟩).support.card 2) (Fintype.card α)
        τ : Equiv.Perm α
        hτ : Membership.mem (alternatingGroup α) τ
        hc : IsConj ↑⟨σ, hσ✝⟩ ↑⟨τ, hτ⟩
        π : Equiv.Perm α
        hπ : Eq (HMul.hMul (HMul.hMul π σ) (Inv.inv π)) τ
        h : Eq (Equiv.Perm.sign π) (-1)
        h2 : LE.le 2 (HasCompl.compl σ.support).card
        a : α
        ha : Membership.mem (HasCompl.compl σ.support) a
        b : α
        hb : Membership.mem (HasCompl.compl σ.support) b
        ab : Ne a b
        hd : (Equiv.swap a b).Disjoint σ
        ⊢ Eq (HMul.hMul (HMul.hMul π (HMul.hMul σ (Equiv.swap a b))) (Inv.inv (HMul.hM …
      -/
      simp [mul_assoc]
      /-
        🎉 no goals
      -/


theorem isThreeCycle_isConj (h5 : 5 ≤ Fintype.card α) {σ τ : alternatingGroup α}
    (hσ : IsThreeCycle (σ : Perm α)) (hτ : IsThreeCycle (τ : Perm α)) : IsConj σ τ :=
  alternatingGroup.isConj_of (isConj_iff_cycleType_eq.2 (hσ.trans hτ.symm))
        /-
          α : Type u_1
          inst✝¹ : Fintype α
          inst✝ : DecidableEq α
          h5 : LE.le 5 (Fintype.card α)
          σ τ : Subtype fun x => Membership.mem (alternatingGroup α) x
          hσ : (↑σ).IsThreeCycle
          hτ : (↑τ).IsThreeCycle
          ⊢ LE.le (HAdd.hAdd (↑σ).support.card 2) (Fintype.card α)
        -/
    (by rwa [hσ.card_support])
        /-
          🎉 no goals
        -/


@[simp]
theorem closure_three_cycles_eq_alternating :
    closure { σ : Perm α | IsThreeCycle σ } = alternatingGroup α :=
  closure_eq_of_le _ (fun _ hσ => mem_alternatingGroup.2 hσ.sign) fun σ hσ => by
    suffices hind :
      ∀ (n : ℕ) (l : List (Perm α)) (_ : ∀ g, g ∈ l → IsSwap g) (_ : l.length = 2 * n),
        l.prod ∈ closure { σ : Perm α | IsThreeCycle σ } by
      obtain ⟨l, rfl, hl⟩ := truncSwapFactors σ
      obtain ⟨n, hn⟩ := (prod_list_swap_mem_alternatingGroup_iff_even_length hl).1 hσ
      rw [← two_mul] at hn
      exact hind n l hl hn
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      ⊢ ∀ (n : Nat) (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem …
    -/
    intro n
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ⊢ ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g → g.I …
    -/
    induction' n with n ih <;> intro l hl hn
      /-
        case zero
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        σ : Equiv.Perm α
        hσ : Membership.mem (alternatingGroup α) σ
        l : List (Equiv.Perm α)
        hl : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap
        hn : Eq l.length (HMul.hMul 2 0)
        ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) l.prod
      -/
    · simp [List.length_eq_zero.1 hn, one_mem]
      /-
        🎉 no goals
      -/
    /-
      case succ
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ih : ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g →  …
      l : List (Equiv.Perm α)
      hl : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap
      hn : Eq l.length (HMul.hMul 2 (HAdd.hAdd n 1))
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) l.prod
    -/
    rw [Nat.mul_succ] at hn
    /-
      case succ
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ih : ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g →  …
      l : List (Equiv.Perm α)
      hl : ∀ (g : Equiv.Perm α), Membership.mem l g → g.IsSwap
      hn : Eq l.length (HAdd.hAdd (HMul.hMul 2 n) 2)
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) l.prod
    -/
    obtain ⟨a, l, rfl⟩ := l.exists_of_length_succ hn
    /-
      case succ.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ih : ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g →  …
      a : Equiv.Perm α
      l : List (Equiv.Perm α)
      hl : ∀ (g : Equiv.Perm α), Membership.mem (List.cons a l) g → g.IsSwap
      hn : Eq (List.cons a l).length (HAdd.hAdd (HMul.hMul 2 n) 2)
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (List.cons …
    -/
    rw [List.length_cons, Nat.succ_inj'] at hn
    /-
      case succ.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ih : ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g →  …
      a : Equiv.Perm α
      l : List (Equiv.Perm α)
      hl : ∀ (g : Equiv.Perm α), Membership.mem (List.cons a l) g → g.IsSwap
      hn : Eq l.length (HAdd.hAdd (HMul.hMul 2 n) 1)
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (List.cons …
    -/
    obtain ⟨b, l, rfl⟩ := l.exists_of_length_succ hn
    /-
      case succ.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ih : ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g →  …
      a b : Equiv.Perm α
      l : List (Equiv.Perm α)
      hl : ∀ (g : Equiv.Perm α), Membership.mem (List.cons a (List.cons b l)) g → g. …
      hn : Eq (List.cons b l).length (HAdd.hAdd (HMul.hMul 2 n) 1)
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (List.cons …
    -/
    rw [List.prod_cons, List.prod_cons, ← mul_assoc]
    /-
      case succ.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      σ : Equiv.Perm α
      hσ : Membership.mem (alternatingGroup α) σ
      n : Nat
      ih : ∀ (l : List (Equiv.Perm α)), (∀ (g : Equiv.Perm α), Membership.mem l g →  …
      a b : Equiv.Perm α
      l : List (Equiv.Perm α)
      hl : ∀ (g : Equiv.Perm α), Membership.mem (List.cons a (List.cons b l)) g → g. …
      hn : Eq (List.cons b l).length (HAdd.hAdd (HMul.hMul 2 n) 1)
      ⊢ Membership.mem (Subgroup.closure (setOf fun σ => σ.IsThreeCycle)) (HMul.hMul …
    -/
    rw [List.length_cons, Nat.succ_inj'] at hn
    exact
      mul_mem
        (IsSwap.mul_mem_closure_three_cycles (hl a (List.mem_cons_self a _))
          (hl b (List.mem_cons_of_mem a (l.mem_cons_self b))))
        (ih _ (fun g hg => hl g (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hg))) hn)


/-- A key lemma to prove $A_5$ is simple. Shows that any normal subgroup of an alternating group on
  at least 5 elements is the entire alternating group if it contains a 3-cycle. -/
theorem IsThreeCycle.alternating_normalClosure (h5 : 5 ≤ Fintype.card α) {f : Perm α}
    (hf : IsThreeCycle f) :
    normalClosure ({⟨f, hf.mem_alternatingGroup⟩} : Set (alternatingGroup α)) = ⊤ :=
  eq_top_iff.2
    (by
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        ⊢ LE.le Top.top (Subgroup.normalClosure (Singleton.singleton ⟨f, ⋯⟩))
      -/
      have hi : Function.Injective (alternatingGroup α).subtype := Subtype.coe_injective
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        ⊢ LE.le Top.top (Subgroup.normalClosure (Singleton.singleton ⟨f, ⋯⟩))
      -/
      refine eq_top_iff.1 (map_injective hi (le_antisymm (map_mono le_top) ?_))
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        ⊢ LE.le (Subgroup.map (alternatingGroup α).subtype Top.top) (Subgroup.map (alt …
      -/
      rw [← MonoidHom.range_eq_map, range_subtype, normalClosure, MonoidHom.map_closure]
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        ⊢ LE.le (alternatingGroup α) (Subgroup.closure (_root_.Set.image (⇑(alternatin …
      -/
      refine (le_of_eq closure_three_cycles_eq_alternating.symm).trans (closure_mono ?_)
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        ⊢ HasSubset.Subset (setOf fun σ => σ.IsThreeCycle) (_root_.Set.image (⇑(altern …
      -/
      intro g h
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        g : Equiv.Perm α
        h : Membership.mem (setOf fun σ => σ.IsThreeCycle) g
        ⊢ Membership.mem (_root_.Set.image (⇑(alternatingGroup α).subtype) (Group.conj …
      -/
      obtain ⟨c, rfl⟩ := isConj_iff.1 (isConj_iff_cycleType_eq.2 (hf.trans h.symm))
      /-
        case intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        c : Equiv.Perm α
        h : Membership.mem (setOf fun σ => σ.IsThreeCycle) (HMul.hMul (HMul.hMul c f)  …
        ⊢ Membership.mem (_root_.Set.image (⇑(alternatingGroup α).subtype) (Group.conj …
      -/
      refine ⟨⟨c * f * c⁻¹, h.mem_alternatingGroup⟩, ?_, rfl⟩
      /-
        case intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        c : Equiv.Perm α
        h : Membership.mem (setOf fun σ => σ.IsThreeCycle) (HMul.hMul (HMul.hMul c f)  …
        ⊢ Membership.mem (Group.conjugatesOfSet (Singleton.singleton ⟨f, ⋯⟩)) ⟨HMul.hM …
      -/
      rw [Group.mem_conjugatesOfSet_iff]
      /-
        case intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        h5 : LE.le 5 (Fintype.card α)
        f : Equiv.Perm α
        hf : f.IsThreeCycle
        hi : Function.Injective ⇑(alternatingGroup α).subtype
        c : Equiv.Perm α
        h : Membership.mem (setOf fun σ => σ.IsThreeCycle) (HMul.hMul (HMul.hMul c f)  …
        ⊢ Exists fun a => And (Membership.mem (Singleton.singleton ⟨f, ⋯⟩) a) (IsConj  …
      -/
      exact ⟨⟨f, hf.mem_alternatingGroup⟩, Set.mem_singleton _, isThreeCycle_isConj h5 hf h⟩)
      /-
        🎉 no goals
      -/


/-- Part of proving $A_5$ is simple. Shows that the square of any element of $A_5$ with a 3-cycle in
  its cycle decomposition is a 3-cycle, so the normal closure of the original element must be
  $A_5$. -/
theorem isThreeCycle_sq_of_three_mem_cycleType_five {g : Perm (Fin 5)} (h : 3 ∈ cycleType g) :
    IsThreeCycle (g * g) := by
  /-
    g : Equiv.Perm (Fin 5)
    h : Membership.mem g.cycleType 3
    ⊢ (HMul.hMul g g).IsThreeCycle
  -/
  obtain ⟨c, g', rfl, hd, _, h3⟩ := mem_cycleType_iff.1 h
  /-
    case intro.intro.intro.intro.intro
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    ⊢ (HMul.hMul (HMul.hMul c g') (HMul.hMul c g')).IsThreeCycle
  -/
  simp only [mul_assoc]
  /-
    case intro.intro.intro.intro.intro
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    ⊢ (HMul.hMul c (HMul.hMul g' (HMul.hMul c g'))).IsThreeCycle
  -/
  rw [hd.commute.eq, ← mul_assoc g']
  suffices hg' : orderOf g' ∣ 2 by
    rw [← pow_two, orderOf_dvd_iff_pow_eq_one.1 hg', one_mul]
    exact (card_support_eq_three_iff.1 h3).isThreeCycle_sq
  /-
    case intro.intro.intro.intro.intro
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    ⊢ Dvd.dvd (orderOf g') 2
  -/
  rw [← lcm_cycleType, Multiset.lcm_dvd]
  /-
    case intro.intro.intro.intro.intro
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    ⊢ ∀ (b : Nat), Membership.mem g'.cycleType b → Dvd.dvd b 2
  -/
  intro n hn
  /-
    case intro.intro.intro.intro.intro
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    n : Nat
    hn : Membership.mem g'.cycleType n
    ⊢ Dvd.dvd n 2
  -/
  rw [le_antisymm (two_le_of_mem_cycleType hn) (le_trans (le_card_support_of_mem_cycleType hn) _)]
  /-
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    n : Nat
    hn : Membership.mem g'.cycleType n
    ⊢ LE.le g'.support.card 2
  -/
  apply le_of_add_le_add_left
  /-
    case bc
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    n : Nat
    hn : Membership.mem g'.cycleType n
    ⊢ LE.le (HAdd.hAdd ?a g'.support.card) (HAdd.hAdd ?a 2)
  -/
  rw [← hd.card_support_mul, h3]
  /-
    case bc
    c g' : Equiv.Perm (Fin 5)
    h : Membership.mem (HMul.hMul c g').cycleType 3
    hd : c.Disjoint g'
    left✝ : c.IsCycle
    h3 : Eq c.support.card 3
    n : Nat
    hn : Membership.mem g'.cycleType n
    ⊢ LE.le (HMul.hMul c g').support.card (HAdd.hAdd 3 2)
  -/
  exact (c * g').support.card_le_univ
  /-
    🎉 no goals
  -/


theorem nontrivial_of_three_le_card (h3 : 3 ≤ card α) : Nontrivial (alternatingGroup α) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    h3 : LE.le 3 (Fintype.card α)
    ⊢ Nontrivial (Subtype fun x => Membership.mem (alternatingGroup α) x)
  -/
  haveI := Fintype.one_lt_card_iff_nontrivial.1 (lt_trans (by decide) h3)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    h3 : LE.le 3 (Fintype.card α)
    this : Nontrivial α
    ⊢ Nontrivial (Subtype fun x => Membership.mem (alternatingGroup α) x)
  -/
  rw [← Fintype.one_lt_card_iff_nontrivial]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    h3 : LE.le 3 (Fintype.card α)
    this : Nontrivial α
    ⊢ LT.lt 1 (Fintype.card (Subtype fun x => Membership.mem (alternatingGroup α)  …
  -/
  refine lt_of_mul_lt_mul_left ?_ (le_of_lt Nat.prime_two.pos)
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    h3 : LE.le 3 (Fintype.card α)
    this : Nontrivial α
    ⊢ LT.lt (HMul.hMul 2 1) (HMul.hMul 2 (Fintype.card (Subtype fun x => Membershi …
  -/
  rw [two_mul_card_alternatingGroup, card_perm, ← Nat.succ_le_iff]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    h3 : LE.le 3 (Fintype.card α)
    this : Nontrivial α
    ⊢ LE.le (HMul.hMul 2 1).succ (Fintype.card α).factorial
  -/
  exact le_trans h3 (card α).self_le_factorial
  /-
    🎉 no goals
  -/


instance {n : ℕ} : Nontrivial (alternatingGroup (Fin (n + 3))) :=
  nontrivial_of_three_le_card
    (by
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        n : Nat
        ⊢ LE.le 3 (Fintype.card (Fin (HAdd.hAdd n 3)))
      -/
      rw [card_fin]
      /-
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        n : Nat
        ⊢ LE.le 3 (HAdd.hAdd n 3)
      -/
      exact le_add_left (le_refl 3))
      /-
        🎉 no goals
      -/


/-- The normal closure of the 5-cycle `finRotate 5` within $A_5$ is the whole group. This will be
  used to show that the normal closure of any 5-cycle within $A_5$ is the whole group. -/
theorem normalClosure_finRotate_five : normalClosure ({⟨finRotate 5,
    finRotate_bit1_mem_alternatingGroup (n := 2)⟩} : Set (alternatingGroup (Fin 5))) = ⊤ :=
  eq_top_iff.2
    (by
      have h3 :
        IsThreeCycle (Fin.cycleRange 2 * finRotate 5 * (Fin.cycleRange 2)⁻¹ * (finRotate 5)⁻¹) :=
        card_support_eq_three_iff.1 (by decide)
      /-
        h3 : (HMul.hMul (HMul.hMul (HMul.hMul (Fin.cycleRange 2) (finRotate 5)) (Inv.i …
        ⊢ LE.le Top.top (Subgroup.normalClosure (Singleton.singleton ⟨finRotate 5, ⋯⟩))
      -/
      rw [← h3.alternating_normalClosure (by rw [card_fin])]
      /-
        h3 : (HMul.hMul (HMul.hMul (HMul.hMul (Fin.cycleRange 2) (finRotate 5)) (Inv.i …
        ⊢ LE.le (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (HMul.hMul (HM …
      -/
      refine normalClosure_le_normal ?_
      /-
        h3 : (HMul.hMul (HMul.hMul (HMul.hMul (Fin.cycleRange 2) (finRotate 5)) (Inv.i …
        ⊢ HasSubset.Subset (Singleton.singleton ⟨HMul.hMul (HMul.hMul (HMul.hMul (Fin. …
      -/
      rw [Set.singleton_subset_iff, SetLike.mem_coe]
      have h :
        (⟨finRotate 5, finRotate_bit1_mem_alternatingGroup (n := 2)⟩ : alternatingGroup (Fin 5)) ∈
          normalClosure _ :=
        SetLike.mem_coe.1 (subset_normalClosure (Set.mem_singleton _))
      exact (mul_mem (Subgroup.normalClosure_normal.conj_mem _ h
        -- Porting note: added `: _`
        ⟨Fin.cycleRange 2, Fin.isThreeCycle_cycleRange_two.mem_alternatingGroup⟩) (inv_mem h) : _))


/-- The normal closure of $(04)(13)$ within $A_5$ is the whole group. This will be
  used to show that the normal closure of any permutation of cycle type $(2,2)$ is the whole group.
  -/
theorem normalClosure_swap_mul_swap_five :
    normalClosure
                                                           /-
                                                             α : Type u_1
                                                             inst✝¹ : Fintype α
                                                             inst✝ : DecidableEq α
                                                             ⊢ Eq (Equiv.Perm.sign (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3))) 1
                                                           -/
        ({⟨swap 0 4 * swap 1 3, mem_alternatingGroup.2 (by decide)⟩} :
                                                           /-
                                                             🎉 no goals
                                                           -/
          Set (alternatingGroup (Fin 5))) =
      ⊤ := by
  /-
    ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (Equiv.swap 0 4)  …
  -/
  let g1 := (⟨swap 0 2 * swap 0 1, mem_alternatingGroup.2 (by decide)⟩ : alternatingGroup (Fin 5))
  /-
    g1 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (Equiv.swap 0 4)  …
  -/
  let g2 := (⟨swap 0 4 * swap 1 3, mem_alternatingGroup.2 (by decide)⟩ : alternatingGroup (Fin 5))
  have h5 : g1 * g2 * g1⁻¹ * g2⁻¹ =
      ⟨finRotate 5, finRotate_bit1_mem_alternatingGroup (n := 2)⟩ := by
    rw [Subtype.ext_iff]
    simp only [Fin.val_mk, Subgroup.coe_mul, Subgroup.coe_inv, Fin.val_mk]
    decide
  /-
    g1 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    g2 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    h5 : Eq (HMul.hMul (HMul.hMul (HMul.hMul g1 g2) (Inv.inv g1)) (Inv.inv g2)) ⟨f …
    ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (Equiv.swap 0 4)  …
  -/
  rw [eq_top_iff, ← normalClosure_finRotate_five]
  /-
    g1 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    g2 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    h5 : Eq (HMul.hMul (HMul.hMul (HMul.hMul g1 g2) (Inv.inv g1)) (Inv.inv g2)) ⟨f …
    ⊢ LE.le (Subgroup.normalClosure (Singleton.singleton ⟨finRotate 5, ⋯⟩)) (Subgr …
  -/
  refine normalClosure_le_normal ?_
  /-
    g1 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    g2 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    h5 : Eq (HMul.hMul (HMul.hMul (HMul.hMul g1 g2) (Inv.inv g1)) (Inv.inv g2)) ⟨f …
    ⊢ HasSubset.Subset (Singleton.singleton ⟨finRotate 5, ⋯⟩) ↑(Subgroup.normalClo …
  -/
  rw [Set.singleton_subset_iff, SetLike.mem_coe, ← h5]
  have h : g2 ∈ normalClosure {g2} :=
    SetLike.mem_coe.1 (subset_normalClosure (Set.mem_singleton _))
  /-
    g1 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    g2 : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x := ⟨HMul.hMu …
    h5 : Eq (HMul.hMul (HMul.hMul (HMul.hMul g1 g2) (Inv.inv g1)) (Inv.inv g2)) ⟨f …
    h : Membership.mem (Subgroup.normalClosure (Singleton.singleton g2)) g2
    ⊢ Membership.mem (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul (Equi …
  -/
  exact mul_mem (Subgroup.normalClosure_normal.conj_mem _ h g1) (inv_mem h)
  /-
    🎉 no goals
  -/


/-- Shows that any non-identity element of $A_5$ whose cycle decomposition consists only of swaps
  is conjugate to $(04)(13)$. This is used to show that the normal closure of such a permutation
  in $A_5$ is $A_5$. -/
theorem isConj_swap_mul_swap_of_cycleType_two {g : Perm (Fin 5)} (ha : g ∈ alternatingGroup (Fin 5))
    (h1 : g ≠ 1) (h2 : ∀ n, n ∈ cycleType (g : Perm (Fin 5)) → n = 2) :
    IsConj (swap 0 4 * swap 1 3) g := by
  /-
    g : Equiv.Perm (Fin 5)
    ha : Membership.mem (alternatingGroup (Fin 5)) g
    h1 : Ne g 1
    h2 : ∀ (n : Nat), Membership.mem g.cycleType n → Eq n 2
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  have h := g.support.card_le_univ
  /-
    g : Equiv.Perm (Fin 5)
    ha : Membership.mem (alternatingGroup (Fin 5)) g
    h1 : Ne g 1
    h2 : ∀ (n : Nat), Membership.mem g.cycleType n → Eq n 2
    h : LE.le g.support.card (Fintype.card (Fin 5))
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  rw [← Multiset.eq_replicate_card] at h2
  /-
    g : Equiv.Perm (Fin 5)
    ha : Membership.mem (alternatingGroup (Fin 5)) g
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h : LE.le g.support.card (Fintype.card (Fin 5))
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  rw [← sum_cycleType, h2, Multiset.sum_replicate, smul_eq_mul] at h
  have h : Multiset.card g.cycleType ≤ 3 :=
    le_of_mul_le_mul_right (le_trans h (by norm_num only [card_fin])) (by simp)
  /-
    g : Equiv.Perm (Fin 5)
    ha : Membership.mem (alternatingGroup (Fin 5)) g
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h✝ : LE.le (HMul.hMul g.cycleType.card 2) (Fintype.card (Fin 5))
    h : LE.le g.cycleType.card 3
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  rw [mem_alternatingGroup, sign_of_cycleType, h2] at ha
  /-
    g : Equiv.Perm (Fin 5)
    ha : Eq (HPow.hPow (-1) (HAdd.hAdd (Multiset.replicate g.cycleType.card 2).sum …
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h✝ : LE.le (HMul.hMul g.cycleType.card 2) (Fintype.card (Fin 5))
    h : LE.le g.cycleType.card 3
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  norm_num at ha
  /-
    g : Equiv.Perm (Fin 5)
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h✝ : LE.le (HMul.hMul g.cycleType.card 2) (Fintype.card (Fin 5))
    h : LE.le g.cycleType.card 3
    ha : Eq (HPow.hPow (-1) (HAdd.hAdd (HMul.hMul g.cycleType.card 2) g.cycleType. …
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  rw [pow_add, pow_mul, Int.units_pow_two, one_mul, neg_one_pow_eq_one_iff_even] at ha
  /-
    g : Equiv.Perm (Fin 5)
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h✝ : LE.le (HMul.hMul g.cycleType.card 2) (Fintype.card (Fin 5))
    h : LE.le g.cycleType.card 3
    ha : Even g.cycleType.card
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  swap; · decide
          /-
            🎉 no goals
          -/
  /-
    g : Equiv.Perm (Fin 5)
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h✝ : LE.le (HMul.hMul g.cycleType.card 2) (Fintype.card (Fin 5))
    h : LE.le g.cycleType.card 3
    ha : Even g.cycleType.card
    ⊢ IsConj (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)) g
  -/
  rw [isConj_iff_cycleType_eq, h2]
  /-
    g : Equiv.Perm (Fin 5)
    h1 : Ne g 1
    h2 : Eq g.cycleType (Multiset.replicate g.cycleType.card 2)
    h✝ : LE.le (HMul.hMul g.cycleType.card 2) (Fintype.card (Fin 5))
    h : LE.le g.cycleType.card 3
    ha : Even g.cycleType.card
    ⊢ Eq (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)).cycleType (Multiset.replica …
  -/
  interval_cases h_1 : Multiset.card g.cycleType
    /-
      case «0»
      g : Equiv.Perm (Fin 5)
      h1 : Ne g 1
      h_1 : Eq g.cycleType.card 0
      h2 : Eq g.cycleType (Multiset.replicate 0 2)
      h✝ : LE.le (HMul.hMul 0 2) (Fintype.card (Fin 5))
      h : LE.le 0 3
      ha : Even 0
      ⊢ Eq (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)).cycleType (Multiset.replica …
    -/
  · exact (h1 (card_cycleType_eq_zero.1 h_1)).elim
    /-
      🎉 no goals
    -/
    /-
      case «1»
      g : Equiv.Perm (Fin 5)
      h1 : Ne g 1
      h_1 : Eq g.cycleType.card 1
      h2 : Eq g.cycleType (Multiset.replicate 1 2)
      h✝ : LE.le (HMul.hMul 1 2) (Fintype.card (Fin 5))
      h : LE.le 1 3
      ha : Even 1
      ⊢ Eq (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)).cycleType (Multiset.replica …
    -/
  · simp at ha
    /-
      🎉 no goals
    -/
    /-
      case «2»
      g : Equiv.Perm (Fin 5)
      h1 : Ne g 1
      h_1 : Eq g.cycleType.card 2
      h2 : Eq g.cycleType (Multiset.replicate 2 2)
      h✝ : LE.le (HMul.hMul 2 2) (Fintype.card (Fin 5))
      h : LE.le 2 3
      ha : Even 2
      ⊢ Eq (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)).cycleType (Multiset.replica …
    -/
  · have h04 : (0 : Fin 5) ≠ 4 := by decide
    /-
      case «2»
      g : Equiv.Perm (Fin 5)
      h1 : Ne g 1
      h_1 : Eq g.cycleType.card 2
      h2 : Eq g.cycleType (Multiset.replicate 2 2)
      h✝ : LE.le (HMul.hMul 2 2) (Fintype.card (Fin 5))
      h : LE.le 2 3
      ha : Even 2
      h04 : Ne 0 4
      ⊢ Eq (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)).cycleType (Multiset.replica …
    -/
    have h13 : (1 : Fin 5) ≠ 3 := by decide
    rw [Disjoint.cycleType, (isCycle_swap h04).cycleType, (isCycle_swap h13).cycleType,
      card_support_swap h04, card_support_swap h13]
      /-
        case «2»
        g : Equiv.Perm (Fin 5)
        h1 : Ne g 1
        h_1 : Eq g.cycleType.card 2
        h2 : Eq g.cycleType (Multiset.replicate 2 2)
        h✝ : LE.le (HMul.hMul 2 2) (Fintype.card (Fin 5))
        h : LE.le 2 3
        ha : Even 2
        h04 : Ne 0 4
        h13 : Ne 1 3
        ⊢ Eq (HAdd.hAdd ↑(List.cons 2 List.nil) ↑(List.cons 2 List.nil)) (Multiset.rep …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case «2»
        g : Equiv.Perm (Fin 5)
        h1 : Ne g 1
        h_1 : Eq g.cycleType.card 2
        h2 : Eq g.cycleType (Multiset.replicate 2 2)
        h✝ : LE.le (HMul.hMul 2 2) (Fintype.card (Fin 5))
        h : LE.le 2 3
        ha : Even 2
        h04 : Ne 0 4
        h13 : Ne 1 3
        ⊢ (Equiv.swap 0 4).Disjoint (Equiv.swap 1 3)
      -/
    · rw [disjoint_iff_disjoint_support, support_swap h04, support_swap h13]
      /-
        case «2»
        g : Equiv.Perm (Fin 5)
        h1 : Ne g 1
        h_1 : Eq g.cycleType.card 2
        h2 : Eq g.cycleType (Multiset.replicate 2 2)
        h✝ : LE.le (HMul.hMul 2 2) (Fintype.card (Fin 5))
        h : LE.le 2 3
        ha : Even 2
        h04 : Ne 0 4
        h13 : Ne 1 3
        ⊢ Disjoint (Insert.insert 0 (Singleton.singleton 4)) (Insert.insert 1 (Singlet …
      -/
      decide
      /-
        🎉 no goals
      -/
    /-
      case «3»
      g : Equiv.Perm (Fin 5)
      h1 : Ne g 1
      h_1 : Eq g.cycleType.card 3
      h2 : Eq g.cycleType (Multiset.replicate 3 2)
      h✝ : LE.le (HMul.hMul 3 2) (Fintype.card (Fin 5))
      h : LE.le 3 3
      ha : Even 3
      ⊢ Eq (HMul.hMul (Equiv.swap 0 4) (Equiv.swap 1 3)).cycleType (Multiset.replica …
    -/
  · contradiction
    /-
      🎉 no goals
    -/


/-- Shows that $A_5$ is simple by taking an arbitrary non-identity element and showing by casework
  on its cycle type that its normal closure is all of $A_5$. -/
instance isSimpleGroup_five : IsSimpleGroup (alternatingGroup (Fin 5)) :=
  ⟨fun H => by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      ⊢ H.Normal → Or (Eq H Bot.bot) (Eq H Top.top)
    -/
    intro Hn
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      ⊢ Or (Eq H Bot.bot) (Eq H Top.top)
    -/
    refine or_not.imp id fun Hb => ?_
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      Hb : Not (Eq H Bot.bot)
      ⊢ Eq H Top.top
    -/
    rw [eq_bot_iff_forall] at Hb
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      Hb : Not (∀ (x : Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x) …
      ⊢ Eq H Top.top
    -/
    push_neg at Hb
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      Hb : Exists fun x => And (Membership.mem H x) (Ne x 1)
      ⊢ Eq H Top.top
    -/
    obtain ⟨⟨g, gA⟩, gH, g1⟩ : ∃ x : ↥(alternatingGroup (Fin 5)), x ∈ H ∧ x ≠ 1 := Hb
    -- `g` is a non-identity alternating permutation in a normal subgroup `H` of $A_5$.
    /-
      case intro.mk.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : Membership.mem H ⟨g, gA⟩
      g1 : Ne ⟨g, gA⟩ 1
      ⊢ Eq H Top.top
    -/
    rw [← SetLike.mem_coe, ← Set.singleton_subset_iff] at gH
    /-
      case intro.mk.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      ⊢ Eq H Top.top
    -/
    refine eq_top_iff.2 (le_trans (ge_of_eq ?_) (normalClosure_le_normal gH))
    -- It suffices to show that the normal closure of `g` in $A_5$ is $A_5$.
    /-
      case intro.mk.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    by_cases h2 : ∀ n ∈ g.cycleType, n = 2
    -- If the cycle decomposition of `g` consists entirely of swaps, then the cycle type is $(2,2)$.
    -- This means that it is conjugate to $(04)(13)$, whose normal closure is $A_5$.
      /-
        case pos
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        h2 : ∀ (n : Nat), Membership.mem g.cycleType n → Eq n 2
        ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
      -/
    · rw [Ne, Subtype.ext_iff] at g1
      exact
        (isConj_swap_mul_swap_of_cycleType_two gA g1 h2).normalClosure_eq_top_of
          normalClosure_swap_mul_swap_five
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      h2 : Not (∀ (n : Nat), Membership.mem g.cycleType n → Eq n 2)
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    push_neg at h2
    /-
      case neg
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      h2 : Exists fun n => And (Membership.mem g.cycleType n) (Ne n 2)
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    obtain ⟨n, ng, n2⟩ : ∃ n : ℕ, n ∈ g.cycleType ∧ n ≠ 2 := h2
    -- `n` is the size of a non-swap cycle in the decomposition of `g`.
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      n : Nat
      ng : Membership.mem g.cycleType n
      n2 : Ne n 2
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    have n2' : 2 < n := lt_of_le_of_ne (two_le_of_mem_cycleType ng) n2.symm
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      n : Nat
      ng : Membership.mem g.cycleType n
      n2 : Ne n 2
      n2' : LT.lt 2 n
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    have n5 : n ≤ 5 := le_trans ?_ g.support.card_le_univ
    -- We check that `2 < n ≤ 5`, so that `interval_cases` has a precise range to check.
    /-
      case neg.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      n : Nat
      ng : Membership.mem g.cycleType n
      n2 : Ne n 2
      n2' : LT.lt 2 n
      n5 : LE.le n 5
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    swap
      /-
        case neg.intro.intro.refine_1
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType n
        n2 : Ne n 2
        n2' : LT.lt 2 n
        ⊢ LE.le n g.support.card
      -/
    · obtain ⟨m, hm⟩ := Multiset.exists_cons_of_mem ng
      /-
        case neg.intro.intro.refine_1.intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType n
        n2 : Ne n 2
        n2' : LT.lt 2 n
        m : Multiset Nat
        hm : Eq g.cycleType (Multiset.cons n m)
        ⊢ LE.le n g.support.card
      -/
      rw [← sum_cycleType, hm, Multiset.sum_cons]
      /-
        case neg.intro.intro.refine_1.intro
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType n
        n2 : Ne n 2
        n2' : LT.lt 2 n
        m : Multiset Nat
        hm : Eq g.cycleType (Multiset.cons n m)
        ⊢ LE.le n (HAdd.hAdd n m.sum)
      -/
      exact le_add_right le_rfl
      /-
        🎉 no goals
      -/
    /-
      case neg.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : DecidableEq α
      H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
      Hn : H.Normal
      g : Equiv.Perm (Fin 5)
      gA : Membership.mem (alternatingGroup (Fin 5)) g
      gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
      g1 : Ne ⟨g, gA⟩ 1
      n : Nat
      ng : Membership.mem g.cycleType n
      n2 : Ne n 2
      n2' : LT.lt 2 n
      n5 : LE.le n 5
      ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
    -/
    interval_cases n
    -- This breaks into cases `n = 3`, `n = 4`, `n = 5`.
    -- If `n = 3`, then `g` has a 3-cycle in its decomposition, so `g^2` is a 3-cycle.
    -- `g^2` is in the normal closure of `g`, so that normal closure must be $A_5$.
    · rw [eq_top_iff, ← (isThreeCycle_sq_of_three_mem_cycleType_five ng).alternating_normalClosure
        (by rw [card_fin])]
      /-
        case neg.intro.intro.refine_2.«3»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 3
        n2 : Ne 3 2
        n2' : LT.lt 2 3
        n5 : LE.le 3 5
        ⊢ LE.le (Subgroup.normalClosure (Singleton.singleton ⟨HMul.hMul g g, ⋯⟩)) (Sub …
      -/
      refine normalClosure_le_normal ?_
      /-
        case neg.intro.intro.refine_2.«3»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 3
        n2 : Ne 3 2
        n2' : LT.lt 2 3
        n5 : LE.le 3 5
        ⊢ HasSubset.Subset (Singleton.singleton ⟨HMul.hMul g g, ⋯⟩) ↑(Subgroup.normalC …
      -/
      rw [Set.singleton_subset_iff, SetLike.mem_coe]
      have h := SetLike.mem_coe.1 (subset_normalClosure
        (G := alternatingGroup (Fin 5)) (Set.mem_singleton ⟨g, gA⟩))
      /-
        case neg.intro.intro.refine_2.«3»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 3
        n2 : Ne 3 2
        n2' : LT.lt 2 3
        n5 : LE.le 3 5
        h : Membership.mem (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) ⟨g,  …
        ⊢ Membership.mem (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) ⟨HMul. …
      -/
      exact mul_mem h h
      /-
        🎉 no goals
      -/
    · -- The case `n = 4` leads to contradiction, as no element of $A_5$ includes a 4-cycle.
      /-
        case neg.intro.intro.refine_2.«4»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 4
        n2 : Ne 4 2
        n2' : LT.lt 2 4
        n5 : LE.le 4 5
        ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
      -/
      have con := mem_alternatingGroup.1 gA
      /-
        case neg.intro.intro.refine_2.«4»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 4
        n2 : Ne 4 2
        n2' : LT.lt 2 4
        n5 : LE.le 4 5
        con : Eq (Equiv.Perm.sign g) 1
        ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
      -/
      rw [sign_of_cycleType, cycleType_of_card_le_mem_cycleType_add_two (by decide) ng] at con
      /-
        case neg.intro.intro.refine_2.«4»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 4
        n2 : Ne 4 2
        n2' : LT.lt 2 4
        n5 : LE.le 4 5
        con : Eq (HPow.hPow (-1) (HAdd.hAdd (Singleton.singleton 4).sum (Singleton.sin …
        ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
      -/
      have : Odd 5 := by decide
      /-
        case neg.intro.intro.refine_2.«4»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 4
        n2 : Ne 4 2
        n2' : LT.lt 2 4
        n5 : LE.le 4 5
        con : Eq (HPow.hPow (-1) (HAdd.hAdd (Singleton.singleton 4).sum (Singleton.sin …
        this : Odd 5
        ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
      -/
      simp [this] at con
      /-
        🎉 no goals
      -/
    · -- If `n = 5`, then `g` is itself a 5-cycle, conjugate to `finRotate 5`.
      /-
        case neg.intro.intro.refine_2.«5»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 5
        n2 : Ne 5 2
        n2' : LT.lt 2 5
        n5 : LE.le 5 5
        ⊢ Eq (Subgroup.normalClosure (Singleton.singleton ⟨g, gA⟩)) Top.top
      -/
      refine (isConj_iff_cycleType_eq.2 ?_).normalClosure_eq_top_of normalClosure_finRotate_five
      /-
        case neg.intro.intro.refine_2.«5»
        α : Type u_1
        inst✝¹ : Fintype α
        inst✝ : DecidableEq α
        H : Subgroup (Subtype fun x => Membership.mem (alternatingGroup (Fin 5)) x)
        Hn : H.Normal
        g : Equiv.Perm (Fin 5)
        gA : Membership.mem (alternatingGroup (Fin 5)) g
        gH : HasSubset.Subset (Singleton.singleton ⟨g, gA⟩) ↑H
        g1 : Ne ⟨g, gA⟩ 1
        n : Nat
        ng : Membership.mem g.cycleType 5
        n2 : Ne 5 2
        n2' : LT.lt 2 5
        n5 : LE.le 5 5
        ⊢ Eq (finRotate 5).cycleType g.cycleType
      -/
      rw [cycleType_of_card_le_mem_cycleType_add_two (by decide) ng, cycleType_finRotate]⟩
      /-
        🎉 no goals
      -/


