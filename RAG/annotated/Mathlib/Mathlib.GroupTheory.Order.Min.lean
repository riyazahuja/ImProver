/-- The minimum order of a non-identity element. Also the minimum size of a nontrivial subgroup, see
`Monoid.le_minOrder_iff_forall_subgroup`. Returns `∞` if the monoid is torsion-free. -/
@[to_additive "The minimum order of a non-identity element. Also the minimum size of a nontrivial
subgroup, see `AddMonoid.le_minOrder_iff_forall_addSubgroup`. Returns `∞` if the monoid is
torsion-free."]
noncomputable def minOrder : ℕ∞ := ⨅ (a : α) (_ha : a ≠ 1) (_ha' : IsOfFinOrder a), orderOf a


@[to_additive (attr := simp)]
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : Monoid α
                                                                 ⊢ Iff (Eq (Monoid.minOrder α) Top.top) (Monoid.IsTorsionFree α)
                                                               -/
lemma minOrder_eq_top : minOrder α = ⊤ ↔ IsTorsionFree α := by simp [minOrder, IsTorsionFree]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive (attr := simp)] protected alias ⟨_, IsTorsionFree.minOrder⟩ := minOrder_eq_top


@[to_additive (attr := simp)]
lemma le_minOrder {n : ℕ∞} :
                                                                             /-
                                                                               α : Type u_1
                                                                               inst✝ : Monoid α
                                                                               n : ENat
                                                                               ⊢ Iff (LE.le n (Monoid.minOrder α)) (∀ ⦃a : α⦄, Ne a 1 → IsOfFinOrder a → LE.l …
                                                                             -/
    n ≤ minOrder α ↔ ∀ ⦃a : α⦄, a ≠ 1 → IsOfFinOrder a → n ≤ orderOf a := by simp [minOrder]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[to_additive]
lemma minOrder_le_orderOf (ha : a ≠ 1) (ha' : IsOfFinOrder a) : minOrder α ≤ orderOf a :=
  le_minOrder.1 le_rfl ha ha'


@[to_additive]
lemma le_minOrder_iff_forall_subgroup {n : ℕ∞} :
    n ≤ minOrder α ↔ ∀ ⦃s : Subgroup α⦄, s ≠ ⊥ → (s : Set α).Finite → n ≤ Nat.card s := by
  /-
    α : Type u_1
    inst✝ : Group α
    n : ENat
    ⊢ Iff (LE.le n (Monoid.minOrder α)) (∀ ⦃s : Subgroup α⦄, Ne s Bot.bot → (↑s).F …
  -/
  rw [le_minOrder]
  /-
    α : Type u_1
    inst✝ : Group α
    n : ENat
    ⊢ Iff (∀ ⦃a : α⦄, Ne a 1 → IsOfFinOrder a → LE.le n ↑(orderOf a)) (∀ ⦃s : Subg …
  -/
  refine ⟨fun h s hs hs' ↦ ?_, fun h a ha ha' ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : Group α
      n : ENat
      h : ∀ ⦃a : α⦄, Ne a 1 → IsOfFinOrder a → LE.le n ↑(orderOf a)
      s : Subgroup α
      hs : Ne s Bot.bot
      hs' : (↑s).Finite
      ⊢ LE.le n ↑(Nat.card (Subtype fun x => Membership.mem s x))
    -/
  · obtain ⟨a, has, ha⟩ := s.bot_or_exists_ne_one.resolve_left hs
    exact
      (h ha <| finite_zpowers.1 <| hs'.subset <| zpowers_le.2 has).trans
        (WithTop.coe_le_coe.2 <| s.orderOf_le_card hs' has)
    /-
      case refine_2
      α : Type u_1
      inst✝ : Group α
      n : ENat
      h : ∀ ⦃s : Subgroup α⦄, Ne s Bot.bot → (↑s).Finite → LE.le n ↑(Nat.card (Subty …
      a : α
      ha : Ne a 1
      ha' : IsOfFinOrder a
      ⊢ LE.le n ↑(orderOf a)
    -/
  · simpa using h (zpowers_ne_bot.2 ha) ha'.finite_zpowers
    /-
      🎉 no goals
    -/


@[to_additive]
lemma minOrder_le_natCard (hs : s ≠ ⊥) (hs' : (s : Set α).Finite) : minOrder α ≤ Nat.card s :=
  le_minOrder_iff_forall_subgroup.1 le_rfl hs hs'


@[simp]
protected lemma minOrder {n : ℕ} (hn : n ≠ 0) (hn₁ : n ≠ 1) : minOrder (ZMod n) = n.minFac := by
  /-
    n : Nat
    hn : Ne n 0
    hn₁ : Ne n 1
    ⊢ Eq (AddMonoid.minOrder (ZMod n)) ↑n.minFac
  -/
  have : Fact (1 < n) := ⟨one_lt_iff_ne_zero_and_ne_one.mpr ⟨hn, hn₁⟩⟩
  classical
  have : (↑(n / n.minFac) : ZMod n) ≠ 0 := by
    rw [Ne, ringChar.spec, ringChar.eq (ZMod n) n]
    exact
      not_dvd_of_pos_of_lt (Nat.div_pos (minFac_le hn.bot_lt) n.minFac_pos)
        (div_lt_self hn.bot_lt (minFac_prime hn₁).one_lt)
  refine ((minOrder_le_natCard (zmultiples_eq_bot.not.2 this) <| toFinite _).trans ?_).antisymm <|
    le_minOrder_iff_forall_addSubgroup.2 fun s hs _ ↦ ?_
  · rw [Nat.card_zmultiples, ZMod.addOrderOf_coe _ hn,
      gcd_eq_right (div_dvd_of_dvd n.minFac_dvd), Nat.div_div_self n.minFac_dvd hn]
  · haveI : Nontrivial s := s.bot_or_nontrivial.resolve_left hs
    exact WithTop.coe_le_coe.2 <| minFac_le_of_dvd Finite.one_lt_card <|
      (card_addSubgroup_dvd_card _).trans n.card_zmod.dvd


@[simp]
lemma minOrder_of_prime {p : ℕ} (hp : p.Prime) : minOrder (ZMod p) = p := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq (AddMonoid.minOrder (ZMod p)) ↑p
  -/
  rw [ZMod.minOrder hp.ne_zero hp.ne_one, hp.minFac_eq]
  /-
    🎉 no goals
  -/


